"""
Evaluate the trained DQN checkpoint (scripts/train_dqn.py) against the
already-validated baselines (Optimum, SPGREEDY, Random -- see
scripts/reproduce_fig1.py) on held-out opinion draws, on Karate (the
training graph) and Dolphins (unseen at training time -- the GNN encoder is
inductive over node count, so this also probes zero-shot transfer, though
that isn't the MVP's goal).

For a fair per-seed comparison, every method sees the *same* internal
opinions s and the *same* candidate pool E_C: s is drawn from
np.random.default_rng(seed) via opinions.uniform as the very first random
draw for that seed, both directly (for the baselines) and inside
LinkRecEnv.reset (for the agent) -- so the two draws are identical.

Both SPGREEDY and the agent's greedy rollout are forward/prefix-consistent
(each round only adds one more edge to the previous round's choice), so each
is run once at k=K_MAX and sliced into prefixes for k=1..K_MAX, matching the
existing reproduce_fig1.py/reproduce_table1.py pattern.

Usage:
    python scripts/eval_dqn.py
Outputs:
    results/dqn_eval.csv
    results/dqn_eval.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from polardrl import baselines, datasets, opinions, pd_index, spgreedy
from polardrl.rl.dqn import DQNAgent
from polardrl.rl.env import LinkRecEnv

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHECKPOINT_PATH = RESULTS_DIR / "dqn_karate.pt"
N_CANDIDATES = 30
K_MAX = 8
EVAL_SEEDS = [101, 102, 103, 104, 105]  # disjoint from train_dqn.py's SEED=20260719 draw stream
NETWORKS = {"Karate": datasets.load_karate, "Dolphins": datasets.load_dolphins}


def delta_i(adjacency, s, edges: list[tuple[int, int]]) -> float:
    """Delta I(G) = I(G + edges) - I(G), matching reproduce_fig1.py's sign convention."""
    i_before = pd_index.pd_index(adjacency, s)
    augmented = adjacency.tolil()
    for u, v in edges:
        augmented[u, v] = 1.0
        augmented[v, u] = 1.0
    i_after = pd_index.pd_index(augmented.tocsr(), s)
    return i_after - i_before


def agent_edges_by_k(agent: DQNAgent, env: LinkRecEnv, seed: int) -> dict[int, list[tuple[int, int]]]:
    """Greedy (epsilon=0) rollout for K_MAX steps, sliced into k=1..K_MAX prefixes."""
    state = env.reset(np.random.default_rng(seed))
    chosen: list[tuple[int, int]] = []
    result: dict[int, list[tuple[int, int]]] = {}
    for k in range(1, K_MAX + 1):
        edge = agent.greedy_action(state)
        state, _reward, _done = env.step(edge)
        chosen.append(edge)
        result[k] = list(chosen)
    return result


def evaluate_network(agent: DQNAgent, net_name: str, adjacency) -> list[dict]:
    rows: list[dict] = []
    candidate_edges = datasets.sample_candidate_edges(adjacency, n_candidates=N_CANDIDATES, seed=0)
    env = LinkRecEnv(adjacency, candidate_edges, k=K_MAX)

    for seed in EVAL_SEEDS:
        s = opinions.uniform(adjacency.shape[0], np.random.default_rng(seed))

        agent_edges = agent_edges_by_k(agent, env, seed)
        sp_edges_full = spgreedy.spgreedy(adjacency, s, candidate_edges, K_MAX)
        opt_edges_by_k = baselines.brute_force_optimum_multi_k(
            adjacency, s, candidate_edges, range(1, K_MAX + 1)
        )

        for k in range(1, K_MAX + 1):
            rand_edges = baselines.random_selection(candidate_edges, k, seed=seed + k)
            rows.append(
                dict(
                    network=net_name,
                    seed=seed,
                    k=k,
                    agent=delta_i(adjacency, s, agent_edges[k]),
                    spgreedy=delta_i(adjacency, s, sp_edges_full[:k]),
                    optimum=delta_i(adjacency, s, opt_edges_by_k[k]),
                    random=delta_i(adjacency, s, rand_edges),
                )
            )
        print(f"  {net_name} seed={seed}: done")

    return rows


def write_csv(rows: list[dict]) -> None:
    path = RESULTS_DIR / "dqn_eval.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def write_plot(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(NETWORKS), figsize=(11, 5))
    for ax, net_name in zip(axes, NETWORKS):
        panel_rows = [r for r in rows if r["network"] == net_name]
        ks = sorted(set(r["k"] for r in panel_rows))

        for method, marker in [
            ("optimum", "o"),
            ("spgreedy", "x"),
            ("agent", "d"),
            ("random", "^"),
        ]:
            means = [np.mean([r[method] for r in panel_rows if r["k"] == k]) for k in ks]
            ax.plot(ks, means, marker=marker, label=method, markersize=5)

        ax.set_title(f"{net_name} (mean over {len(EVAL_SEEDS)} opinion draws)")
        ax.set_xlabel("k")
        ax.set_ylabel("Delta I(G)")
        ax.legend(fontsize=8)

    fig.suptitle("DQN agent vs Optimum / SPGREEDY / Random")
    fig.tight_layout()
    path = RESULTS_DIR / "dqn_eval.png"
    fig.savefig(path, dpi=150)
    print(f"Wrote {path}")


def run() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"{CHECKPOINT_PATH} not found -- run scripts/train_dqn.py first.")

    agent = DQNAgent(k=K_MAX, seed=0)
    agent.online.load_state_dict(torch.load(CHECKPOINT_PATH))
    agent.online.eval()

    all_rows: list[dict] = []
    for net_name, loader in NETWORKS.items():
        print(f"{net_name}:")
        all_rows.extend(evaluate_network(agent, net_name, loader()))

    write_csv(all_rows)
    write_plot(all_rows)

    for net_name in NETWORKS:
        panel = [r for r in all_rows if r["network"] == net_name]
        agent_mean = np.mean([r["agent"] for r in panel])
        sp_mean = np.mean([r["spgreedy"] for r in panel])
        rand_mean = np.mean([r["random"] for r in panel])
        print(
            f"{net_name}: mean Delta I(G) over all k -- "
            f"agent={agent_mean:.4f}, spgreedy={sp_mean:.4f}, random={rand_mean:.4f} "
            f"(agent/spgreedy ratio={agent_mean / sp_mean:.3f})"
        )


if __name__ == "__main__":
    run()
