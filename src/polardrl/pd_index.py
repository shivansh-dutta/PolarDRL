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

from . import fj_model


def disagreement(adjacency: sp.spmatrix, s: np.ndarray) -> float:
    """D(G) = s^T Omega L Omega s. See fj_model.py for Omega and L."""
    z = fj_model.equilibrium_opinions(adjacency, s)
    l = fj_model.laplacian(adjacency)
    return float(z @ (l @ z))


def polarization(adjacency: sp.spmatrix, s: np.ndarray) -> float:
    """P(G) = s_bar^T Omega^2 s_bar, with s_bar the mean-centered internal opinion vector."""
    s_bar = s - s.mean()
    omega = fj_model.forest_matrix(adjacency)
    z_bar = omega @ s_bar
    return float(z_bar @ z_bar)


def pd_index(adjacency: sp.spmatrix, s: np.ndarray) -> float:
    """I(G) = P(G) + D(G). The scalar objective minimized by adding edges."""
    return polarization(adjacency, s) + disagreement(adjacency, s)
