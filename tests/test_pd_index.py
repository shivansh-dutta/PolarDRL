"""
Hand-checkable smoke test for the P-D index, once pd_index.py is implemented.

Fixture: a 4-node path graph 0-1-2-3 with internal opinions s = [0, 0, 1, 1].
This is small enough to compute D(G), P(G), I(G) by hand from Concepts/
Polarization-Disagreement Index.md's definitions and cross-check the
implementation against that value.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from polardrl import pd_index


@pytest.fixture
def path_graph_4():
    # 0-1-2-3 path, unweighted, undirected
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


@pytest.mark.skip(reason="pd_index.py is a stub -- unskip once implemented")
def test_pd_index_is_nonnegative(path_graph_4):
    adjacency, s = path_graph_4
    value = pd_index.pd_index(adjacency, s)
    assert value >= 0
