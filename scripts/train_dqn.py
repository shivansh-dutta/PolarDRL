"""
Train the MVP GNN-DQN agent (src/polardrl/rl) on Karate.

Zhu et al. 2021's own SPGREEDY/FASTGREEDY baselines are the benchmark this
agent is trying to approach (see scripts/reproduce_fig1.py) -- this script
only trains the checkpoint; scripts/eval_dqn.py does the comparison.

The env's per-step reward is exactly spgreedy.marginal_gain (see
src/polardrl/rl/env.py docstring), so a well-trained greedy-argmax policy
should land close to SPGREEDY's own greedy choices.

Usage:
    python scripts/train_dqn.py
Outputs:
    results/dqn_karate.pt        -- trained online-network checkpoint
    results/dqn_training_curve.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from polardrl import datasets
from polardrl.rl.dqn import DQNAgent, Transition
from polardrl.rl.env import LinkRecEnv

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
N_CANDIDATES = 30
K = 8
N_EPISODES = 600
# Batch=16 (not a larger, statistically nicer batch) is a deliberate
# CPU-time tradeoff: dqn.py's update() forward-passes the GNN once per
# transition in a Python loop (no cross-graph batching), so wall-clock cost
# scales ~linearly with batch size -- 16 keeps a full run to a few minutes
# on this machine (profiled: ~0.65s/episode at batch=16 vs ~1.4s at 32).
BATCH_SIZE = 16
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 400  # linear decay from start to end over this many episodes
SEED = 20260719


def epsilon_at(episode: int) -> float:
    frac = min(episode / EPSILON_DECAY_EPISODES, 1.0)
    return EPSILON_START + frac * (EPSILON_END - EPSILON_START)


def run() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    adjacency = datasets.load_karate()
    candidate_edges = datasets.sample_candidate_edges(adjacency, n_candidates=N_CANDIDATES, seed=SEED)
    env = LinkRecEnv(adjacency, candidate_edges, k=K)

    agent = DQNAgent(k=K, seed=SEED)
    np_rng = np.random.default_rng(SEED)

    episode_returns: list[float] = []

    for episode in range(N_EPISODES):
        epsilon = epsilon_at(episode)
        state = env.reset(np_rng)
        done = False
        total_reward = 0.0

        while not done:
            edge = agent.act(state, epsilon)
            next_state, reward, done = env.step(edge)
            agent.buffer.push(Transition(state, edge, reward, next_state, done))
            total_reward += reward
            state = next_state

            batch = agent.buffer.sample(BATCH_SIZE, agent.replay_rng)
            agent.update(batch)

        episode_returns.append(total_reward)

        if (episode + 1) % 100 == 0:
            recent = episode_returns[-100:]
            print(
                f"episode {episode + 1}/{N_EPISODES}  "
                f"epsilon={epsilon:.3f}  "
                f"mean return (last 100)={np.mean(recent):.4f}"
            )

    checkpoint_path = RESULTS_DIR / "dqn_karate.pt"
    torch.save(agent.online.state_dict(), checkpoint_path)
    print(f"Wrote {checkpoint_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episode_returns, alpha=0.3, label="episode return")
    window = 50
    if len(episode_returns) >= window:
        smoothed = np.convolve(episode_returns, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(episode_returns)), smoothed, label=f"{window}-episode moving avg")
    ax.set_xlabel("episode")
    ax.set_ylabel("total reward (= I(G) - I(G+T) over the episode)")
    ax.set_title("DQN training curve: Karate, |E_C|=30, k=8")
    ax.legend()
    fig.tight_layout()
    plot_path = RESULTS_DIR / "dqn_training_curve.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    run()
