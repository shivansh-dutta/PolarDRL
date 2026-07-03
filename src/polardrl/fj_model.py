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
    raise NotImplementedError(
        "TODO: L = D - A, with D = diag(row sums of adjacency). "
        "See Concepts/Forest Matrix.md for how L feeds into Omega."
    )


def forest_matrix(adjacency: sp.spmatrix) -> np.ndarray:
    """Omega = (I + L)^-1 -- the doubly stochastic forest matrix (Concepts/Forest Matrix.md)."""
    raise NotImplementedError(
        "TODO: compute L via laplacian(), then invert (I + L). "
        "For large n this O(n^3) inversion is exactly what FASTGREEDY's COMP "
        "subroutine (see fastgreedy.py) is built to avoid -- keep this exact "
        "version for SPGREEDY and small-network validation only."
    )


def equilibrium_opinions(adjacency: sp.spmatrix, s: np.ndarray) -> np.ndarray:
    """z = Omega @ s -- the equilibrium expressed-opinion vector (Concepts/Friedkin-Johnsen Model.md)."""
    raise NotImplementedError("TODO: z = forest_matrix(adjacency) @ s")
