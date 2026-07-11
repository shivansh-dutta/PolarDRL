"""
Friedkin-Johnsen (FJ) opinion-dynamics model.

Reference: PolarDRL vault Concepts/Friedkin-Johnsen Model.md (Zhu et al. 2021, eq. 1).

Each node i has a static internal opinion s_i and a time-varying expressed
opinion z_i(t) that converges to an equilibrium z = Omega @ s, where
Omega = (I + L)^-1 is the forest matrix (see pd_index.py).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def laplacian(adjacency: sp.spmatrix) -> sp.spmatrix:
    """Graph Laplacian L = D - A for an undirected graph given by its adjacency matrix."""
    degrees = np.asarray(adjacency.sum(axis=1)).flatten()
    return sp.diags(degrees) - adjacency


def forest_matrix(adjacency: sp.spmatrix) -> np.ndarray:
    """Omega = (I + L)^-1 -- the doubly stochastic forest matrix (Concepts/Forest Matrix.md)."""
    n = adjacency.shape[0]
    l = laplacian(adjacency)
    return np.linalg.inv((sp.eye(n) + l).toarray())


def equilibrium_opinions(adjacency: sp.spmatrix, s: np.ndarray) -> np.ndarray:
    """z = Omega @ s -- the equilibrium expressed-opinion vector (Concepts/Friedkin-Johnsen Model.md)."""
    return forest_matrix(adjacency) @ s
