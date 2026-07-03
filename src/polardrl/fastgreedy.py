"""
FASTGREEDY: nearly linear-time approximate edge-recommendation algorithm.

Reference: PolarDRL vault Concepts/FASTGREEDY.md and Concepts/COMP.md
(Zhu et al. 2021, Algorithms 2 and 3).

Same greedy skeleton as SPGREEDY, but each round calls comp() to estimate
marginal gains via Johnson-Lindenstrauss projection + a fast SDDM solver,
avoiding the O(n^3) exact Omega computation entirely.

Time complexity (paper): O~(m*k*eps^-2).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp


def comp(
    adjacency: sp.spmatrix,
    s: np.ndarray,
    candidate_edges: Iterable[tuple[int, int]],
    epsilon: float,
) -> dict[tuple[int, int], float]:
    """
    Approximate f(e) for every candidate edge via JL projection + fast SDDM solve.
    See Concepts/COMP.md for the exact estimator f_hat(e).

    NOTE: a first working version may use a dense/exact SDDM solve (e.g.
    scipy.sparse.linalg.spsolve on (I+L)) to validate correctness, then swap
    in a true fast solver (e.g. PyAMG / a Laplacian-specific preconditioned
    CG) once the exact-vs-approximate pipeline is verified on small graphs.
    """
    raise NotImplementedError(
        "TODO: project B*Omega and Omega via random +-1/sqrt(p) matrices P, Q "
        "(p = ceil(24*log(n)/(eps/12)**2)), solve (I+L)x=b approximately for "
        "the projected systems, then f_hat(e) per Concepts/COMP.md."
    )


def fastgreedy(
    adjacency: sp.spmatrix,
    s: np.ndarray,
    candidate_edges: Iterable[tuple[int, int]],
    k: int,
    epsilon: float = 0.3,
) -> list[tuple[int, int]]:
    """
    Select k edges greedily using comp()'s approximate marginal gains at each round.
    epsilon=0.3 matches the paper's experimental default (Concepts/FASTGREEDY.md).
    """
    raise NotImplementedError(
        "TODO: Algorithm 3 -- for i in range(k): call comp() over remaining "
        "candidates, pick argmax f_hat(e), add to T, update the graph, repeat."
    )
