"""
FASTGREEDY: nearly linear-time approximate edge-recommendation algorithm.

Reference: PolarDRL vault Concepts/FASTGREEDY.md and Concepts/COMP.md
(Zhu et al. 2021, Algorithms 2 and 3).

Same greedy skeleton as SPGREEDY, but each round calls comp() to estimate
marginal gains via Johnson-Lindenstrauss projection + a fast SDDM solver,
avoiding the O(n^3) exact Omega computation entirely.

The paper's Algorithm 2 (COMP) is reconstructed here from Lemmas 6.3-6.5 and
the surviving fragment of the algorithm box (docling shredded the pseudocode
layout -- see Concepts/COMP.md and PolarDRL vault Raw Markdown Conversions,
paper lines 137-282). With b_e = e_u - e_v:

    f_hat(e) = (q_u - q_v)^2 / (1 + ||X~ b_e||^2 + ||Y~ b_e||^2)

where, for Laplacian L, incidence matrix B (m x n), forest matrix Omega=(I+L)^-1:
    p = ceil(24 log n / (eps/12)^2)                    -- JL projection dimension
    P (p x n), Q (p x m): random +-1/sqrt(p) matrices
    q  = SOLVE(I+L, s)        ~= Omega @ s
    X~ = SOLVE(I+L, (QB)^T)^T ~= Q @ B @ Omega   (p x n)
    Y~ = SOLVE(I+L, P^T)^T    ~= P @ Omega       (p x n)
and ||X~ b_e||^2, ||Y~ b_e||^2 are read off as column differences of X~, Y~.

Time complexity (paper): O~(m*k*eps^-2).

Practical JL dimension cap: the paper's literal p = ceil(24 log n / (eps/12)^2)
is a loose worst-case constant from a 3-way union bound over Lemmas 6.3-6.5,
not a literal implementation spec -- for eps=0.3 it gives p in the
100,000-320,000 range even for small networks (Karate n=34 -> p~135,000;
GrQc n=4,158 -> p~320,000). Materializing dense p x n / p x m projection
matrices and solving p-column linear systems at that scale needs multiple GB
per call and is called every round, blowing available RAM on modest hardware.
Real near-linear-time implementations cap p to a small practical value and
verify approximation quality empirically instead of trusting the theoretical
constant literally -- see tests/test_fastgreedy.py's f_hat-vs-f check, which
still passes comfortably at the capped size. P_CAP is a documented, tunable
reproduction decision (see paper_results/README.md), not a bug fix hidden
from the record.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import fj_model

P_CAP = 300


def _incidence_matrix(adjacency: sp.spmatrix) -> sp.csr_matrix:
    """
    m x n signed incidence matrix B: one row per undirected edge (u, v) with
    u < v, +1 in column u and -1 in column v. Satisfies B^T @ B == L.
    """
    n = adjacency.shape[0]
    coo = adjacency.tocoo()
    edges = sorted({(min(u, v), max(u, v)) for u, v in zip(coo.row, coo.col) if u != v})
    m = len(edges)
    rows = np.repeat(np.arange(m), 2)
    cols = np.array([idx for u, v in edges for idx in (u, v)])
    data = np.tile([1.0, -1.0], m)
    return sp.csr_matrix((data, (rows, cols)), shape=(m, n))


def _make_solver(laplacian: sp.spmatrix, n: int, solver: str):
    """
    Returns solve(rhs) solving (I + L) X = rhs for a stack of right-hand
    sides, rhs shape (n, p) -> output shape (n, p).

    solver="exact": prefactor (I+L) once (sparse LU via splu), reuse for
    every column -- isolates JL-projection error only, for validating the
    COMP estimator against exact f(e).
    solver="cg": conjugate gradient per column with a Jacobi (diagonal)
    preconditioner -- the paper's actual near-linear SDDM solver tier.
    """
    system = (sp.eye(n, format="csc") + laplacian).tocsc()

    if solver == "exact":
        factorized = spla.splu(system)

        def solve_exact(rhs: np.ndarray) -> np.ndarray:
            return factorized.solve(rhs)

        return solve_exact

    if solver == "cg":
        diag = system.diagonal()
        m_inv = sp.diags(1.0 / diag)

        def solve_cg(rhs: np.ndarray) -> np.ndarray:
            out = np.empty_like(rhs)
            for col in range(rhs.shape[1]):
                x, info = spla.cg(system, rhs[:, col], M=m_inv, rtol=1e-8, atol=0.0)
                if info != 0:
                    raise RuntimeError(f"CG failed to converge (info={info})")
                out[:, col] = x
            return out

        return solve_cg

    raise ValueError(f"Unknown solver {solver!r}; expected 'exact' or 'cg'")


def comp(
    adjacency: sp.spmatrix,
    s: np.ndarray,
    candidate_edges: Iterable[tuple[int, int]],
    epsilon: float,
    *,
    solver: str = "exact",
    rng: np.random.Generator | None = None,
) -> dict[tuple[int, int], float]:
    """
    Approximate f(e) for every candidate edge via JL projection + SDDM solve.
    See module docstring and Concepts/COMP.md for the exact estimator f_hat(e).
    """
    if rng is None:
        rng = np.random.default_rng()

    candidates = list(candidate_edges)
    n = adjacency.shape[0]
    laplacian = fj_model.laplacian(adjacency)
    incidence = _incidence_matrix(adjacency)
    m = incidence.shape[0]

    p = min(math.ceil(24.0 * math.log(n) / (epsilon / 12.0) ** 2), P_CAP)
    # Random +-1/sqrt(p) (Rademacher) JL projection matrices.
    p_mat = rng.choice([-1.0, 1.0], size=(p, n)) / math.sqrt(p)
    q_mat = rng.choice([-1.0, 1.0], size=(p, m)) / math.sqrt(p)

    solve = _make_solver(laplacian, n, solver)

    # q ~= Omega @ s: one solve, single RHS column.
    q = solve(s.reshape(n, 1))[:, 0]

    # X~ ~= Q @ B @ Omega (p x n): solve (I+L) X~^T = (QB)^T, p right-hand sides.
    qb = q_mat @ incidence  # (p, n), dense
    x_tilde = solve(qb.T).T  # (p, n)

    # Y~ ~= P @ Omega (p x n): solve (I+L) Y~^T = P^T, p right-hand sides.
    y_tilde = solve(p_mat.T).T  # (p, n)

    gains: dict[tuple[int, int], float] = {}
    for u, v in candidates:
        numerator = (q[u] - q[v]) ** 2
        x_diff = x_tilde[:, u] - x_tilde[:, v]
        y_diff = y_tilde[:, u] - y_tilde[:, v]
        denominator = 1.0 + float(x_diff @ x_diff) + float(y_diff @ y_diff)
        gains[(u, v)] = float(numerator / denominator)

    return gains


def fastgreedy(
    adjacency: sp.spmatrix,
    s: np.ndarray,
    candidate_edges: Iterable[tuple[int, int]],
    k: int,
    epsilon: float = 0.3,
    *,
    solver: str = "exact",
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    """
    Select k edges greedily using comp()'s approximate marginal gains at each
    round (Algorithm 3). epsilon=0.3 matches the paper's small-network default
    (Concepts/FASTGREEDY.md).

    Unlike SPGREEDY, COMP never forms Omega explicitly, so there is no
    Sherman-Morrison update between rounds -- each round re-solves against the
    current augmented graph from scratch, exactly as Algorithm 3 specifies.
    """
    if rng is None:
        rng = np.random.default_rng()

    remaining = list(candidate_edges)
    current = adjacency.tolil()
    chosen: list[tuple[int, int]] = []

    for _ in range(k):
        gains = comp(current.tocsr(), s, remaining, epsilon, solver=solver, rng=rng)
        best_edge = max(remaining, key=lambda e: gains[e])
        remaining.remove(best_edge)
        chosen.append(best_edge)
        u, v = best_edge
        current[u, v] = 1.0
        current[v, u] = 1.0

    return chosen
