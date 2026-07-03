"""
Polarization-Disagreement (P-D) index.

Reference: PolarDRL vault Concepts/Polarization-Disagreement Index.md
(Musco et al. 2018; adopted unchanged by Zhu et al. 2021).

    D(G) = z^T L z = s^T Omega L Omega s        (disagreement)
    P(G) = z_bar^T z_bar = s_bar^T Omega^2 s_bar   (polarization, mean-centered)
    I(G) = P(G) + D(G) = s_bar^T Omega s_bar       (P-D index, the objective to minimize)

This is the objective every baseline (SPGREEDY, FASTGREEDY) and, eventually,
the DRL agent, is trying to minimize via edge additions.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def disagreement(adjacency: sp.spmatrix, s: np.ndarray) -> float:
    """D(G) = s^T Omega L Omega s. See fj_model.py for Omega and L."""
    raise NotImplementedError("TODO: implement via fj_model.forest_matrix + fj_model.laplacian")


def polarization(adjacency: sp.spmatrix, s: np.ndarray) -> float:
    """P(G) = s_bar^T Omega^2 s_bar, with s_bar the mean-centered internal opinion vector."""
    raise NotImplementedError("TODO: mean-center s, then s_bar^T Omega^2 s_bar")


def pd_index(adjacency: sp.spmatrix, s: np.ndarray) -> float:
    """I(G) = P(G) + D(G). The scalar objective minimized by adding edges."""
    raise NotImplementedError("TODO: polarization(adjacency, s) + disagreement(adjacency, s)")
