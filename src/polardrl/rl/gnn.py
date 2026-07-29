"""
Hand-rolled GraphSAGE-style GNN + edge Q-head (PyTorch, no torch-geometric).

Reference: PolarDRL vault Concepts/PACIFIER Framework.md (Liao 2026) -- PTGE
is PACIFIER's GraphSAGE-style message-passing encoder over node-moderation
state; this ports the same message-passing idea to LinkRecEnv's edge-addition
state (see env.py). Graphs here are tiny (Karate n=34, Dolphins n=62) so a
dense adjacency matmul is simpler and fast enough -- no sparse kernels needed.

Node input features (see env.py State): [s_i, z_i, normalized degree_i,
normalized steps_left]. Topology already changes every step (an edge is
literally added), unlike PACIFIER's node-opinion moderation where the graph
never changes -- so the state-aliasing problem (Concepts/State-Aliasing
Problem.md) that forces PACIFIER's explicit history mask is expected to be
much milder here; z_i and degree_i already move when an edge lands near node
i. Not proven, just the working assumption for this MVP.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import State

N_NODE_FEATURES = 4  # s_i, z_i, degree_i (norm), steps_left (norm)


class MessagePassingLayer(nn.Module):
    """h_i' = ReLU(W_self h_i + W_neigh * mean_{j in N(i)} h_j)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        h: (n, in_dim) node features.
        adjacency: (n, n) dense 0/1 adjacency (no self-loops).
        """
        degree = adjacency.sum(dim=1, keepdim=True).clamp(min=1.0)
        neighbor_mean = (adjacency @ h) / degree
        return F.relu(self.w_self(h) + self.w_neigh(neighbor_mean))


class GraphEncoder(nn.Module):
    """2-layer message-passing encoder producing a per-node embedding."""

    def __init__(self, in_dim: int = N_NODE_FEATURES, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.layer1 = MessagePassingLayer(in_dim, hidden_dim)
        self.layer2 = MessagePassingLayer(hidden_dim, embed_dim)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.layer1(node_features, adjacency)
        h = self.layer2(h, adjacency)
        return h  # (n, embed_dim)


class EdgeQHead(nn.Module):
    """Q(s, e=(u,v)) from a symmetric combination of the two endpoint embeddings."""

    def __init__(self, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, edges: list[tuple[int, int]]) -> torch.Tensor:
        """Returns a (len(edges),) tensor of Q-values, one per candidate edge."""
        u_idx = torch.tensor([e[0] for e in edges], dtype=torch.long)
        v_idx = torch.tensor([e[1] for e in edges], dtype=torch.long)
        h_u, h_v = node_embeddings[u_idx], node_embeddings[v_idx]
        # Symmetric in (u, v): sum and elementwise product don't depend on order.
        features = torch.cat([h_u + h_v, h_u * h_v], dim=-1)
        return self.mlp(features).squeeze(-1)


class QNetwork(nn.Module):
    """Full state -> per-candidate-edge Q-values pipeline."""

    def __init__(self, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.encoder = GraphEncoder(N_NODE_FEATURES, hidden_dim, embed_dim)
        self.q_head = EdgeQHead(embed_dim, hidden_dim)

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor, edges: list[tuple[int, int]]
    ) -> torch.Tensor:
        embeddings = self.encoder(node_features, adjacency)
        return self.q_head(embeddings, edges)


def build_node_features(state: State, k: int) -> torch.Tensor:
    """
    State (env.State) -> (n, N_NODE_FEATURES) float32 tensor:
    [s_i, z_i, degree_i / (n-1), steps_left / k].
    """
    n = state.s.shape[0]
    degree_norm = state.degrees / max(n - 1, 1)
    steps_norm = np.full(n, state.steps_left / k, dtype=float)
    features = np.stack([state.s, state.z, degree_norm, steps_norm], axis=1)
    return torch.tensor(features, dtype=torch.float32)


def build_adjacency_tensor(state: State) -> torch.Tensor:
    """State.adjacency (scipy csr) -> dense (n, n) float32 torch tensor."""
    return torch.tensor(state.adjacency.toarray(), dtype=torch.float32)
