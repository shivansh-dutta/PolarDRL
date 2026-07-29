"""
LinkRecEnv: sequential link-recommendation MDP over the P-D index I(G).

Reference: PolarDRL vault Concepts/PACIFIER Framework.md (Liao 2026) for the
GNN+DQN template this environment feeds; the reward/objective itself is
exactly Zhu et al. 2021's edge-addition objective f(e) (Concepts/Edge-Addition
Objective Function.md, Lemma 4.1), reused unchanged from spgreedy.py.

MDP:
    - Instance: fixed graph G, fixed candidate non-edge pool E_C, budget k.
    - Episode: internal opinions s are redrawn each reset() so the agent must
      condition on s rather than memorize one fixed edge sequence.
    - State: current adjacency, s, current equilibrium z = Omega @ s, the
      still-available candidates, and steps remaining.
    - Action: one edge from the currently-available candidates.
    - Reward: f(e) = I(G_t) - I(G_t + e), i.e. spgreedy.marginal_gain -- the
      same quantity SPGREEDY picks greedily at each round. Summed over an
      episode this telescopes exactly to I(G) - I(G+T) (see test_rl_env.py).
    - Transition: Omega is updated incrementally via spgreedy.update_omega
      (Sherman-Morrison, O(n^2)) rather than a fresh O(n^3) inversion.

This module adds no new objective math -- it is a thin, independently
testable wrapper over the already-validated fj_model/spgreedy/pd_index code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp

from .. import fj_model, opinions, spgreedy

OpinionGenerator = Callable[[int, np.random.Generator], np.ndarray]


@dataclass
class State:
    """Everything the GNN encoder / eval scripts need to read at a given step."""

    adjacency: sp.csr_matrix
    s: np.ndarray
    z: np.ndarray  # equilibrium expressed opinions, z = Omega @ s
    degrees: np.ndarray
    available: list[tuple[int, int]]
    steps_left: int


@dataclass
class LinkRecEnv:
    """
    Fixed graph + fixed candidate pool; opinions redrawn each episode.

    Parameters
    ----------
    adjacency: base graph G (never mutated -- reset() copies it).
    candidate_edges: E_C, the fixed action pool for every episode.
    k: edge-addition budget per episode.
    opinion_generator: draws s ~ distribution(n, rng); defaults to Zhu et
        al.'s uniform distribution (opinions.uniform).
    """

    adjacency: sp.csr_matrix
    candidate_edges: list[tuple[int, int]]
    k: int = 8
    opinion_generator: OpinionGenerator = field(default=opinions.uniform)

    def __post_init__(self) -> None:
        self.n = self.adjacency.shape[0]
        self._omega: np.ndarray | None = None
        self._adjacency_dense: np.ndarray | None = None
        self._s: np.ndarray | None = None
        self._available: list[tuple[int, int]] | None = None
        self._steps_left: int = 0

    def reset(self, rng: np.random.Generator) -> State:
        """Start a new episode: redraw s, reset the graph to G, refill E_C."""
        self._s = self.opinion_generator(self.n, rng)
        self._adjacency_dense = self.adjacency.toarray().astype(float)
        self._omega = fj_model.forest_matrix(self.adjacency)
        self._available = list(self.candidate_edges)
        self._steps_left = self.k
        return self._state()

    def step(self, edge: tuple[int, int]) -> tuple[State, float, bool]:
        """
        Add `edge` (must be in the currently-available set). Returns
        (next_state, reward, done). reward = f(edge) = I(G_t) - I(G_t+edge).
        """
        if self._available is None:
            raise RuntimeError("step() called before reset()")
        if edge not in self._available:
            raise ValueError(f"edge {edge} is not in the available candidate set")

        reward = spgreedy.marginal_gain(self._omega, self._s, edge)
        self._omega = spgreedy.update_omega(self._omega, edge)

        u, v = edge
        self._adjacency_dense[u, v] = 1.0
        self._adjacency_dense[v, u] = 1.0
        self._available.remove(edge)
        self._steps_left -= 1

        done = self._steps_left == 0 or not self._available
        return self._state(), float(reward), done

    def _state(self) -> State:
        adjacency_csr = sp.csr_matrix(self._adjacency_dense)
        degrees = self._adjacency_dense.sum(axis=1)
        z = self._omega @ self._s
        return State(
            adjacency=adjacency_csr,
            s=self._s.copy(),
            z=z,
            degrees=degrees,
            available=list(self._available),
            steps_left=self._steps_left,
        )
