# PolarDRL — DRL Agent Results (Phase 4)

Everything here covers the work done *after* the reproduction phase (Task 3, where we
reproduced Zhu et al. 2021's SPGREEDY/FASTGREEDY baselines and validated them against
brute-force optimum — see the sibling `paper_results/` folder). This phase builds and
evaluates the actual DRL agent.

## What problem the agent solves

Given a social network with internal opinions on each node, we want to add `k` new edges
(social connections) that most reduce the network's **Polarization-Disagreement index**
`I(G) = P(G) + D(G)`. Zhu et al.'s SPGREEDY does this by repeatedly picking the single
edge that gives the biggest one-step drop in `I(G)`. Our DRL agent learns to do the same
edge-picking task, but via a trained policy instead of a hand-coded greedy rule.

## How the agent works

- **Environment** (`src/polardrl/rl/env.py`): each episode starts from a fixed graph and a
  freshly-sampled opinion vector `s`. The state is the current graph, `s`, the equilibrium
  opinion vector `z = Ω·s`, the set of still-available candidate edges, and steps remaining.
  The action is "add this candidate edge." The reward for each step is the *exact* marginal
  drop in `I(G)` from adding that edge — the same quantity SPGREEDY's greedy rule uses
  (`spgreedy.marginal_gain`), so a well-trained agent's choices are directly comparable to
  SPGREEDY's. `Ω` is updated incrementally after each edge (no full O(n³) recompute).

- **Model** (`src/polardrl/rl/gnn.py`): a small hand-rolled 2-layer graph neural network
  (GraphSAGE-style message passing, no external graph library) reads each node's opinion,
  equilibrium value, degree, and steps-remaining, and produces an embedding per node. An
  edge-scoring head turns any two node embeddings into a Q-value for adding that edge.

- **Training** (`src/polardrl/rl/dqn.py`, `scripts/train_dqn.py`): standard DQN — online +
  target network, ε-greedy exploration over available edges, replay buffer, Adam optimizer.
  Trained for 600 episodes on the Karate Club graph, CPU only (no GPU used or required).

- **Evaluation** (`scripts/eval_dqn.py`): the trained agent's greedy (ε=0) rollout is
  compared against SPGREEDY, brute-force Optimum, and Random, all given the *same* opinion
  draws and candidate-edge pool, for budgets k=1..8, on 5 held-out seeds per graph.

## Results

Evaluated on two graphs:
- **Karate Club (34 nodes)** — the graph the agent was trained on.
- **Dolphins (62 nodes)** — never seen during training. Because the GNN is inductive
  (it doesn't depend on a fixed node count), this checks whether the trained policy
  transfers zero-shot to an unseen graph.

Mean ΔI(G) achieved across k=1..8 and 5 opinion draws, as a fraction of SPGREEDY's:

| Graph | Agent's ΔI(G) | SPGREEDY's ΔI(G) | Random's ΔI(G) | Agent / SPGREEDY |
|---|---|---|---|---|
| Karate (trained) | -0.0924 | -0.1010 | -0.0415 | **91.5%** |
| Dolphins (unseen) | -0.1721 | -0.1828 | -0.0589 | **94.2%** |

(ΔI(G) is negative because adding edges *reduces* polarization-disagreement; larger
magnitude = better. Numbers recomputed directly from `dqn_eval.csv` in this folder.)

**Takeaway:** the trained agent reaches ~92-94% of SPGREEDY's edge-selection quality and
clearly beats random edge selection, on both the graph it trained on and an unseen graph —
the full learn-then-generalize loop works end-to-end on this (CPU-only) machine.

## Files in this folder

- `dqn_training_curve.png` — per-episode return during training (600 episodes, Karate).
- `dqn_eval.png` — ΔI(G) vs. k for Agent / SPGREEDY / Optimum / Random, Karate + Dolphins.
- `dqn_eval.csv` — raw per-seed, per-k numbers behind the plot and the table above.
- `dqn_karate.pt` — trained model checkpoint (PyTorch state dict).

## Next steps (not yet done)

- Testing transfer to larger networks (deferred earlier due to this machine's limits).
- Training with a non-myopic / trajectory-level reward, since `I(G)` is proven
  non-submodular — a planning agent could in principle beat step-by-step greedy.
- Checking for state-aliasing as more edges get added within an episode.
