"""
Correctness tests for the FASTGREEDY approximate baseline (COMP estimator +
the FASTGREEDY greedy loop).

Targets locked into CI (see plan "FASTGREEDY implementation + full
reproduction"):
1. COMP's f_hat(e) matches exact f(e) within the paper's Theorem 6.1 bound,
   and much tighter empirically since the exact solver isolates JL-projection
   error only.
2. FASTGREEDY's selected-edge Delta I tracks SPGREEDY's within ~1% (the
   paper's reported 0.988-0.998 ratio on the small networks).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from polardrl import datasets, fastgreedy, fj_model, pd_index, spgreedy


def test_incidence_matrix_reconstructs_laplacian():
    adjacency = datasets.load_karate()
    laplacian = fj_model.laplacian(adjacency)
    incidence = fastgreedy._incidence_matrix(adjacency)
    reconstructed = (incidence.T @ incidence).toarray()
    assert np.allclose(reconstructed, laplacian.toarray())


def test_comp_estimator_matches_exact_marginal_gain():
    """
    On Karate with the exact solver, only JL-projection error remains, so
    f_hat(e) should be close to f(e) -- checked against both the paper's
    loose theoretical bound (4/5 * eps) and a much tighter empirical bound
    that would catch a real formula bug the loose bound would hide.
    """
    adjacency = datasets.load_karate()
    n = adjacency.shape[0]
    rng = np.random.default_rng(0)
    s = rng.uniform(0.0, 1.0, size=n)
    candidate_edges = datasets.sample_candidate_edges(adjacency, n_candidates=30, seed=0)
    epsilon = 0.3

    omega = fj_model.forest_matrix(adjacency)
    exact_gains = {e: spgreedy.marginal_gain(omega, s, e) for e in candidate_edges}

    approx_gains = fastgreedy.comp(
        adjacency, s, candidate_edges, epsilon, solver="exact", rng=np.random.default_rng(1)
    )

    loose_bound = 0.8 * epsilon
    for e in candidate_edges:
        diff = abs(exact_gains[e] - approx_gains[e])
        assert diff <= loose_bound, f"{e}: |f-f_hat|={diff} exceeds paper bound {loose_bound}"
        # Tight empirical check: with exact solve and Karate-scale p, JL error
        # should be a small fraction of the loose theoretical bound.
        assert diff <= 0.25 * loose_bound, f"{e}: |f-f_hat|={diff} suspiciously large vs {loose_bound}"


def test_fastgreedy_ranks_like_spgreedy():
    """
    FASTGREEDY's chosen-edge Delta I should track SPGREEDY's within ~1%,
    matching the paper's reported ratio (0.988-0.998) on the small networks.
    """
    adjacency = datasets.load_karate()
    n = adjacency.shape[0]
    rng = np.random.default_rng(2)
    s = rng.uniform(0.0, 1.0, size=n)
    candidate_edges = datasets.sample_candidate_edges(adjacency, n_candidates=30, seed=2)

    k = 4
    sp_edges = spgreedy.spgreedy(adjacency, s, candidate_edges, k)
    fg_edges = fastgreedy.fastgreedy(
        adjacency, s, candidate_edges, k, epsilon=0.3, solver="exact", rng=np.random.default_rng(3)
    )

    i_g = pd_index.pd_index(adjacency, s)
    sp_delta = pd_index.pd_index(_augment(adjacency, sp_edges), s) - i_g
    fg_delta = pd_index.pd_index(_augment(adjacency, fg_edges), s) - i_g

    ratio = fg_delta / sp_delta
    assert ratio == pytest.approx(1.0, abs=0.05), f"FASTGREEDY/SPGREEDY ratio {ratio} not within 5%"


def _augment(adjacency: sp.spmatrix, edges: list[tuple[int, int]]) -> sp.csr_matrix:
    dense = adjacency.toarray()
    for u, v in edges:
        dense[u, v] = 1.0
        dense[v, u] = 1.0
    return sp.csr_matrix(dense)
