"""
SPGREEDY: exact greedy edge-recommendation algorithm.

Reference: PolarDRL vault Concepts/SPGREEDY.md (Zhu et al. 2021, Algorithm 1).

Iteratively selects the candidate edge with the largest exact marginal gain
f(e) = (s^T Omega b_e)(b_e^T Omega s) / (1 + b_e^T Omega b_e), adds it, and
updates Omega via a rank-1 Sherman-Morrison update rather than a fresh
O(n^3) matrix inversion each round.

Time complexity (paper): O(n^3 + k|E_C|n^2).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp

from . import fj_model


def _omega_be(omega: np.ndarray, edge: tuple[int, int]) -> np.ndarray:
    """Omega @ b_e where b_e = e_u - e_v, computed as a column difference (O(n), no matvec)."""
    u, v = edge
    return omega[:, u] - omega[:, v]


def marginal_gain(omega: np.ndarray, s: np.ndarray, edge: tuple[int, int]) -> float:
    """f(e) for a single candidate edge e=(u,v). See Concepts/Edge-Addition Objective Function.md eq. (3)."""
    u, v = edge
    omega_be = _omega_be(omega, edge)
    numerator = (s @ omega_be) * (omega_be @ s)
    denominator = 1.0 + (omega[u, u] - 2.0 * omega[u, v] + omega[v, v])
    return float(numerator / denominator)


def update_omega(omega: np.ndarray, edge: tuple[int, int]) -> np.ndarray:
    """Sherman-Morrison rank-1 update of Omega after adding edge e (Concepts/SPGREEDY.md)."""
    omega_be = _omega_be(omega, edge)
    denominator = 1.0 + (omega_be[edge[0]] - omega_be[edge[1]])
    return omega - np.outer(omega_be, omega_be) / denominator


def spgreedy(
    adjacency: sp.spmatrix,
    s: np.ndarray,
    candidate_edges: Iterable[tuple[int, int]],
    k: int,
) -> list[tuple[int, int]]:
    """
    Select k edges from candidate_edges greedily maximizing marginal_gain at each step.
    Returns the chosen edge set T, |T| = k.
    """
    remaining = list(candidate_edges)
    omega = fj_model.forest_matrix(adjacency)
    chosen: list[tuple[int, int]] = []

    for _ in range(k):
        gains = [marginal_gain(omega, s, e) for e in remaining]
        best_idx = int(np.argmax(gains))
        best_edge = remaining.pop(best_idx)
        chosen.append(best_edge)
        omega = update_omega(omega, best_edge)

    return chosen
