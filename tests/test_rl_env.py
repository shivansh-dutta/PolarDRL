"""
Correctness tests for LinkRecEnv (src/polardrl/rl/env.py).

These check the environment is a faithful, bug-free wrapper over the
already-validated fj_model/spgreedy/pd_index code -- no new objective math
is introduced here, so every check reduces to an identity that must hold
exactly (up to floating-point slack).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from polardrl import datasets, fj_model, pd_index, spgreedy
from polardrl.rl.env import LinkRecEnv


@pytest.fixture
def karate_env():
    adjacency = datasets.load_karate()
    candidate_edges = datasets.sample_candidate_edges(adjacency, n_candidates=12, seed=0)
    return LinkRecEnv(adjacency, candidate_edges, k=4)


def test_reset_returns_consistent_state(karate_env):
    state = karate_env.reset(np.random.default_rng(0))
    assert state.adjacency.shape == (34, 34)
    assert state.s.shape == (34,)
    assert state.z.shape == (34,)
    assert state.steps_left == 4
    assert len(state.available) == 12
    # z = Omega @ s should already reflect the un-augmented graph's equilibrium.
    assert np.allclose(state.z, fj_model.equilibrium_opinions(karate_env.adjacency, state.s))


def test_reward_matches_spgreedy_marginal_gain(karate_env):
    """The env's per-step reward must equal spgreedy.marginal_gain on the same Omega."""
    rng = np.random.default_rng(1)
    state = karate_env.reset(rng)
    omega = fj_model.forest_matrix(karate_env.adjacency)
    edge = state.available[0]
    expected_reward = spgreedy.marginal_gain(omega, state.s, edge)

    _next_state, reward, _done = karate_env.step(edge)
    assert reward == pytest.approx(expected_reward)


def test_episode_reward_telescopes_to_direct_index_drop(karate_env):
    """Sum of step rewards over a full episode == I(G) - I(G+T) computed directly."""
    rng = np.random.default_rng(2)
    state = karate_env.reset(rng)
    s = state.s.copy()
    i_before = pd_index.pd_index(karate_env.adjacency, s)

    chosen: list[tuple[int, int]] = []
    total_reward = 0.0
    done = False
    while not done:
        edge = state.available[0]
        state, reward, done = karate_env.step(edge)
        chosen.append(edge)
        total_reward += reward

    dense = karate_env.adjacency.toarray().copy()
    for u, v in chosen:
        dense[u, v] = 1.0
        dense[v, u] = 1.0
    i_after = pd_index.pd_index(sp.csr_matrix(dense), s)

    assert total_reward == pytest.approx(i_before - i_after, abs=1e-8)


def test_adjacency_stays_symmetric_and_available_shrinks(karate_env):
    state = karate_env.reset(np.random.default_rng(3))
    edge = state.available[0]
    n_before = len(state.available)

    state, _reward, _done = karate_env.step(edge)

    dense = state.adjacency.toarray()
    assert np.allclose(dense, dense.T)
    assert edge not in state.available
    assert len(state.available) == n_before - 1


def test_episode_ends_after_k_steps(karate_env):
    state = karate_env.reset(np.random.default_rng(4))
    done = False
    steps_taken = 0
    while not done:
        edge = state.available[0]
        state, _reward, done = karate_env.step(edge)
        steps_taken += 1

    assert steps_taken == karate_env.k
    assert state.steps_left == 0


def test_step_before_reset_raises():
    adjacency = datasets.load_karate()
    candidates = datasets.sample_candidate_edges(adjacency, n_candidates=5, seed=0)
    env = LinkRecEnv(adjacency, candidates, k=2)
    with pytest.raises(RuntimeError):
        env.step(candidates[0])


def test_step_on_unavailable_edge_raises(karate_env):
    state = karate_env.reset(np.random.default_rng(5))
    edge = state.available[0]
    karate_env.step(edge)
    with pytest.raises(ValueError):
        karate_env.step(edge)  # already consumed
