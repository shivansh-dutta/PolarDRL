"""
DQN training core for LinkRecEnv (edge-selection MDP over I(G)).

Reference: PolarDRL vault Concepts/PACIFIER Framework.md -- PACIFIER-RL is
full n-step bootstrapped DQN over a GNN-encoded state; this is the same
value-based lineage (S2V-DQN / FINDER / PACIFIER), ported to the edge action
space defined in env.py, with a plain 1-step Bellman target (episodes here
are short -- k<=8 -- so full n-step bootstrapping isn't needed for the MVP).

Standard DQN: online QNetwork picks epsilon-greedy actions among the
currently *available* candidate edges (env.State.available), a target
QNetwork (periodically synced) provides the bootstrap target, and a replay
buffer decorrelates consecutive transitions.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .env import State
from .gnn import QNetwork, build_adjacency_tensor, build_node_features


@dataclass
class Transition:
    state: State
    edge: tuple[int, int]
    reward: float
    next_state: State
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int, rng: random.Random) -> list[Transition]:
        return rng.sample(self._buffer, min(batch_size, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)


class DQNAgent:
    """
    Epsilon-greedy DQN over LinkRecEnv's edge action space.

    Parameters mirror the plan's MVP defaults: gamma<1 so the agent still
    values the full k-step return despite short episodes, periodic target
    sync, Adam. `k` is needed to normalize the steps_left feature the same
    way build_node_features does at eval time.
    """

    def __init__(
        self,
        k: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        gamma: float = 0.95,
        lr: float = 1e-3,
        target_sync_every: int = 25,
        seed: int = 0,
    ):
        self.k = k
        self.gamma = gamma
        self.target_sync_every = target_sync_every
        self._updates = 0

        self.online = QNetwork(hidden_dim, embed_dim)
        self.target = QNetwork(hidden_dim, embed_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self._py_rng = random.Random(seed)

        self.buffer = ReplayBuffer()
        self.replay_rng = random.Random(seed + 1)

    def _q_values(self, net: QNetwork, state: State) -> torch.Tensor:
        node_features = build_node_features(state, self.k)
        adjacency = build_adjacency_tensor(state)
        return net(node_features, adjacency, state.available)

    def act(self, state: State, epsilon: float) -> tuple[int, int]:
        """Epsilon-greedy action selection among state.available."""
        if self._py_rng.random() < epsilon:
            return self._py_rng.choice(state.available)
        with torch.no_grad():
            q_values = self._q_values(self.online, state)
        best_idx = int(torch.argmax(q_values).item())
        return state.available[best_idx]

    def greedy_action(self, state: State) -> tuple[int, int]:
        """Pure argmax policy (epsilon=0), used at evaluation time."""
        return self.act(state, epsilon=0.0)

    def update(self, batch: list[Transition]) -> float:
        """One gradient step on a batch of transitions. Returns the loss value."""
        if not batch:
            return 0.0

        losses = []
        for transition in batch:
            q_values = self._q_values(self.online, transition.state)
            edge_idx = transition.state.available.index(transition.edge)
            q_pred = q_values[edge_idx]

            with torch.no_grad():
                if transition.done or not transition.next_state.available:
                    target = torch.tensor(transition.reward, dtype=torch.float32)
                else:
                    next_q = self._q_values(self.target, transition.next_state)
                    target = transition.reward + self.gamma * next_q.max()

            losses.append(F.mse_loss(q_pred, target))

        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._updates += 1
        if self._updates % self.target_sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())
