"""
Reference edge-selection strategies for the Figure-1 reproduction (Zhu et al.
2021, Sec 7 "Methods"): brute-force Optimum and Random.

Optimum needs two layers of care:

1. Naively re-inverting (I+L) for each of the C(|E_C|, k) candidate subsets
   is O(C(|E_C|,k) * n^3) -- infeasible even at |E_C|=30, k=8 on Diseasome
   (n=516; ~1e14 flops). We precompute Omega once and express I(G+T) for
   every subset T of the (small) candidate pool via a rank-k Woodbury
   identity, reducing each subset evaluation to an O(k^3) k-by-k solve.
2. Even with that identity, a plain Python `for subset in combinations(...)`
   loop calling np.linalg.solve once per subset is dominated by per-call
   Python/BLAS dispatch overhead, not FLOPs -- C(30,8) ~= 5.85M subsets took
   ~237s that way. Batching subsets into chunks and calling a single
   broadcasted np.linalg.solve per chunk amortizes that overhead and is
   roughly two orders of magnitude faster in practice.
"""
from __future__ import annotations

import itertools
import random
from math import comb
from typing import Iterable

import numpy as np
import scipy.sparse as sp

from . import fj_model
from . import pd_index as pd_index_module

_DEFAULT_CHUNK_SIZE = 200_000


def _b_matrix(n: int, candidate_edges: list[tuple[int, int]]) -> np.ndarray:
    """n x |E_C| matrix whose columns are b_e = e_u - e_v for each candidate edge."""
    b = np.zeros((n, len(candidate_edges)))
    for col, (u, v) in enumerate(candidate_edges):
        b[u, col] = 1.0
        b[v, col] = -1.0
    return b


def _precompute(
    adjacency: sp.spmatrix, s: np.ndarray, candidates: list[tuple[int, int]]
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    One-time O(n^3 + n^2|E_C|) setup shared across every k: I(G), the
    |E_C| x |E_C| Gram matrix G_C = B_C^T Omega B_C, and v_C = s_bar^T Omega B_C.
    """
    n = adjacency.shape[0]
    omega = fj_model.forest_matrix(adjacency)
    s_bar = s - s.mean()
    i_g = pd_index_module.pd_index(adjacency, s)

    b_c = _b_matrix(n, candidates)
    omega_b_c = omega @ b_c
    gram = b_c.T @ omega_b_c
    v_c = s_bar @ omega_b_c
    return i_g, gram, v_c


def _search_optimal_subset(
    i_g: float,
    gram: np.ndarray,
    v_c: np.ndarray,
    m: int,
    k: int,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> tuple[int, ...]:
    """
    Exhaustively search all C(m, k) subsets of range(m) for the one minimizing
        I(G+T) = i_g - v_T @ inv(I_k + G_TT) @ v_T
    in chunks, using a single broadcasted np.linalg.solve per chunk.
    """
    identity_k = np.eye(k)
    best_value = np.inf
    best_idx: tuple[int, ...] | None = None

    combo_iter = itertools.combinations(range(m), k)
    while True:
        batch = list(itertools.islice(combo_iter, chunk_size))
        if not batch:
            break
        idx_arr = np.array(batch)  # (B, k)
        g_tt = gram[idx_arr[:, :, None], idx_arr[:, None, :]]  # (B, k, k)
        v_t = v_c[idx_arr]  # (B, k)
        a = identity_k[None, :, :] + g_tt  # (B, k, k)
        # b must be (..., k, 1) -- a bare (B, k) is read as one non-batched
        # k-vector of a (B,B) system, not a batch of B length-k vectors.
        x = np.linalg.solve(a, v_t[:, :, None])[:, :, 0]  # (B, k)
        drop = np.einsum("bi,bi->b", v_t, x)
        values = i_g - drop

        local_best = int(np.argmin(values))
        if values[local_best] < best_value:
            best_value = values[local_best]
            best_idx = tuple(int(i) for i in idx_arr[local_best])

    assert best_idx is not None  # m >= k guaranteed by caller
    return best_idx


def brute_force_optimum(
    adjacency: sp.spmatrix,
    s: np.ndarray,
    candidate_edges: Iterable[tuple[int, int]],
    k: int,
) -> list[tuple[int, int]]:
    """
    Exhaustively select the size-k subset T of candidate_edges minimizing
    I(G+T). See module docstring for the Woodbury identity and batching.
    """
    candidates = list(candidate_edges)
    i_g, gram, v_c = _precompute(adjacency, s, candidates)
    best_idx = _search_optimal_subset(i_g, gram, v_c, len(candidates), k)
    return [candidates[i] for i in best_idx]


def brute_force_optimum_multi_k(
    adjacency: sp.spmatrix,
    s: np.ndarray,
    candidate_edges: Iterable[tuple[int, int]],
    ks: Iterable[int],
) -> dict[int, list[tuple[int, int]]]:
    """
    Like brute_force_optimum, but reuses one Omega/Gram precomputation across
    every k in ks -- the O(n^3 + n^2|E_C|) setup cost is paid once instead of
    once per k, which matters when scanning k=1..8 for the same (network,
    opinion distribution) pair as scripts/reproduce_fig1.py does.
    """
    candidates = list(candidate_edges)
    i_g, gram, v_c = _precompute(adjacency, s, candidates)
    result: dict[int, list[tuple[int, int]]] = {}
    for k in ks:
        best_idx = _search_optimal_subset(i_g, gram, v_c, len(candidates), k)
        result[k] = [candidates[i] for i in best_idx]
    return result


def random_selection(
    candidate_edges: Iterable[tuple[int, int]], k: int, seed: int | None = None
) -> list[tuple[int, int]]:
    """The paper's Random baseline: k edges chosen uniformly at random from E_C."""
    candidates = list(candidate_edges)
    rng = random.Random(seed)
    return rng.sample(candidates, k)
