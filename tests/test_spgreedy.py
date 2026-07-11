"""
Correctness tests for the SPGREEDY exact greedy baseline.

Fixture reused from test_pd_index.py's hand-checkable 4-node path graph,
plus a small Karate-based check that greedy selection matches the
brute-force Optimum (locking in the paper's Figure 1 "curves overlap"
claim at a scale small enough for CI).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from polardrl import baselines, datasets, fj_model, pd_index, spgreedy


@pytest.fixture
def path_graph_4():
    adjacency = sp.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=float,
        )
    )
    s = np.array([0.0, 0.0, 1.0, 1.0])
    return adjacency, s


def _add_edge(adjacency: sp.spmatrix, edge: tuple[int, int]) -> sp.csr_matrix:
    u, v = edge
    dense = adjacency.toarray()
    dense[u, v] = 1.0
    dense[v, u] = 1.0
    return sp.csr_matrix(dense)


def test_marginal_gain_matches_direct_index_drop(path_graph_4):
    """f(e) should equal I(G) - I(G+e) computed directly via pd_index."""
    adjacency, s = path_graph_4
    omega = fj_model.forest_matrix(adjacency)

    # (0, 3) is the only non-existing edge on the 4-node path.
    edge = (0, 3)
    gain = spgreedy.marginal_gain(omega, s, edge)

    i_before = pd_index.pd_index(adjacency, s)
    i_after = pd_index.pd_index(_add_edge(adjacency, edge), s)
    assert gain == pytest.approx(i_before - i_after)


def test_update_omega_matches_fresh_forest_matrix(path_graph_4):
    """Sherman-Morrison update should match recomputing Omega from scratch."""
    adjacency, s = path_graph_4
    omega = fj_model.forest_matrix(adjacency)
    edge = (0, 3)

    updated = spgreedy.update_omega(omega, edge)
    fresh = fj_model.forest_matrix(_add_edge(adjacency, edge))

    assert np.allclose(updated, fresh)


def test_spgreedy_matches_brute_force_optimum_on_karate():
    """
    On a small candidate pool, SPGREEDY's chosen-edge Delta I should equal
    the brute-force Optimum's Delta I -- this is the CI-scale version of
    the paper's Figure 1 claim that the two curves overlap.
    """
    adjacency = datasets.load_karate()
    rng = np.random.default_rng(0)
    s = rng.uniform(0.0, 1.0, size=adjacency.shape[0])
    candidate_edges = datasets.sample_candidate_edges(adjacency, n_candidates=12, seed=0)

    for k in (1, 2, 3):
        sp_edges = spgreedy.spgreedy(adjacency, s, candidate_edges, k)
        opt_edges = baselines.brute_force_optimum(adjacency, s, candidate_edges, k)

        i_g = pd_index.pd_index(adjacency, s)
        sp_delta = pd_index.pd_index(_augment(adjacency, sp_edges), s) - i_g
        opt_delta = pd_index.pd_index(_augment(adjacency, opt_edges), s) - i_g

        assert sp_delta == pytest.approx(opt_delta, abs=1e-9)


def _augment(adjacency: sp.spmatrix, edges: list[tuple[int, int]]) -> sp.csr_matrix:
    dense = adjacency.toarray()
    for u, v in edges:
        dense[u, v] = 1.0
        dense[v, u] = 1.0
    return sp.csr_matrix(dense)
