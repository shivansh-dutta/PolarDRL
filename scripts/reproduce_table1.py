"""
Reproduce Zhu et al. 2021, Table 1, GrQc row: SPGREEDY's exact Delta I(G) at
k=50 on GrQc (n=4,158, m=13,422 -- the smallest network in Table 1, and the
only one small enough to run in this codebase without FASTGREEDY/COMP).

Unlike Figure 1 (curves for the 4 small networks, no numeric table), Table 1
publishes a literal number to match: SPGREEDY Delta I(G) = -4.9966. This is
a stronger check than Figure 1's greedy-vs-optimum overlap, since it
compares directly against a value the paper's own Julia implementation
produced -- not just an internal self-consistency check.

Two protocol details the paper does not state explicitly for Table 1 and
that we document as assumptions:
  - Candidate edge set size: the paper states |E_C| = 10,000 for the
    Figure 2 large-network experiments (Yeast/GridWorm/Erdos992/Reality),
    immediately preceding the Table 1 paragraph -- we reuse |E_C| = 10,000
    for GrQc under the same assumption, with a documented fixed seed for
    which 10,000 non-existing edges are sampled.
  - Opinion distribution: Table 1 (unlike Figures 1/2) is not broken out by
    distribution. We use uniform [0, 1], matching the "(a)" panel default
    convention used throughout the paper's figures.

An exact match is not expected -- the paper's specific opinion draw and
candidate-edge sample are unpublished, and SPGREEDY's own output is
opinion/candidate-set-dependent -- but the reproduced Delta I(G) should be
the same order of magnitude and comparably negative.

Usage:
    python scripts/reproduce_table1.py
"""
from __future__ import annotations

import time

import numpy as np

from polardrl import datasets, fastgreedy, opinions, pd_index, spgreedy

K = 50
N_CANDIDATES = 10_000
SEED = 20260709
FASTGREEDY_EPSILON = 0.3
PAPER_DELTA_I = -4.9966  # Zhu et al. 2021, Table 1, GrQc / SpGreedy column
PAPER_FASTGREEDY_DELTA_I = -4.9774  # Zhu et al. 2021, Table 1, GrQc / FastGreedy column
PAPER_RATIO = 0.9962  # Zhu et al. 2021, Table 1, GrQc / Ratio column (FastGreedy / SpGreedy)


def delta_i(adjacency, s, edges: list[tuple[int, int]]) -> float:
    i_before = pd_index.pd_index(adjacency, s)
    augmented = adjacency.tolil()
    for u, v in edges:
        augmented[u, v] = 1.0
        augmented[v, u] = 1.0
    i_after = pd_index.pd_index(augmented.tocsr(), s)
    return i_after - i_before


def run() -> None:
    adjacency = datasets.load_grqc()
    n = adjacency.shape[0]
    m = int(adjacency.nnz // 2)
    print(f"GrQc: n={n}, m={m} (paper: n=4158, m=13422)")
    assert n == 4158 and m == 13422, "GrQc largest-component size does not match paper's Table 1"

    rng = np.random.default_rng(SEED)
    s = opinions.uniform(n, rng)
    candidate_edges = datasets.sample_candidate_edges(adjacency, n_candidates=N_CANDIDATES, seed=SEED)

    t0 = time.time()
    sp_edges = spgreedy.spgreedy(adjacency, s, candidate_edges, K)
    sp_elapsed = time.time() - t0

    sp_delta = delta_i(adjacency, s, sp_edges)
    sp_ratio_to_paper = sp_delta / PAPER_DELTA_I

    print(f"SPGREEDY k={K}, |E_C|={N_CANDIDATES}: Delta I(G) = {sp_delta:.4f} in {sp_elapsed:.2f}s")
    print(f"Paper (Table 1, GrQc, SPGREEDY): Delta I(G) = {PAPER_DELTA_I:.4f}")
    print(f"Ratio (ours / paper): {sp_ratio_to_paper:.4f}")

    # Tier-B CG solver: at n=4158, k=50, the exact-solve tier would need a
    # fresh O(n^3) factorization every round (50 rounds -- infeasible); the
    # whole point of FASTGREEDY/COMP is to avoid that via the SDDM CG solver.
    rng = np.random.default_rng(SEED + 1)
    t0 = time.time()
    fg_edges = fastgreedy.fastgreedy(
        adjacency, s, candidate_edges, K, epsilon=FASTGREEDY_EPSILON, solver="cg", rng=rng
    )
    fg_elapsed = time.time() - t0

    fg_delta = delta_i(adjacency, s, fg_edges)
    fg_ratio_to_paper = fg_delta / PAPER_FASTGREEDY_DELTA_I
    our_fg_sp_ratio = fg_delta / sp_delta

    print(
        f"FASTGREEDY k={K}, |E_C|={N_CANDIDATES}, eps={FASTGREEDY_EPSILON}: "
        f"Delta I(G) = {fg_delta:.4f} in {fg_elapsed:.2f}s"
    )
    print(f"Paper (Table 1, GrQc, FASTGREEDY): Delta I(G) = {PAPER_FASTGREEDY_DELTA_I:.4f}")
    print(f"Ratio (ours / paper): {fg_ratio_to_paper:.4f}")
    print(f"Our FASTGREEDY/SPGREEDY ratio: {our_fg_sp_ratio:.4f} (paper: {PAPER_RATIO:.4f})")
    print(
        "Note: exact match not expected -- opinion draw and candidate-edge "
        "sample are unpublished. Same order of magnitude and sign, and a "
        "FASTGREEDY/SPGREEDY ratio close to the paper's, are the meaningful "
        "checks here."
    )


if __name__ == "__main__":
    run()
