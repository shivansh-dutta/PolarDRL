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


def marginal_gain(omega: np.ndarray, s: np.ndarray, edge: tuple[int, int]) -> float:
    """f(e) for a single candidate edge e=(u,v). See Concepts/Edge-Addition Objective Function.md eq. (3)."""
    raise NotImplementedError(
        "TODO: b_e = e_u - e_v (standard basis difference); "
        "f(e) = (s^T Omega b_e)(b_e^T Omega s) / (1 + b_e^T Omega b_e)"
    )


def update_omega(omega: np.ndarray, edge: tuple[int, int]) -> np.ndarray:
    """Sherman-Morrison rank-1 update of Omega after adding edge e (Concepts/SPGREEDY.md)."""
    raise NotImplementedError(
        "TODO: Omega <- Omega - (Omega b_e b_e^T Omega) / (1 + b_e^T Omega b_e)"
    )


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
    raise NotImplementedError(
        "TODO: Algorithm 1 -- initialize T = [], compute Omega once, "
        "then for i in range(k): pick argmax marginal_gain over remaining "
        "candidates, append to T, update_omega, repeat."
    )
