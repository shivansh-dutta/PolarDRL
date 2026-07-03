# PolarDRL — Reproducibility Codebase

Deep reinforcement learning for polarization-disagreement minimization via sequential link recommendation. Foundation paper: Zhu et al. 2021, *Minimizing Polarization and Disagreement in Social Networks via Link Recommendation* (NeurIPS 2021).

This repo covers Task 3 of the project (reproducibility & baselines) — see the sibling Obsidian vault at `../PolarDRL/PaperGuidelines.md` for the full project scope, and `../PolarDRL/Concepts/` for reference notes on every concept these modules implement (FJ model, P-D index, SPGREEDY, FASTGREEDY, COMP).

## Status

Scaffold only — module signatures and docstrings are in place (`src/polardrl/`), pointing at the exact concept note and equation each one implements, but the numerics themselves are `NotImplementedError` stubs. Next steps: implement `fj_model.py` → `pd_index.py` → `spgreedy.py` (exact baseline) → validate on the 4 small networks (Karate/Dolphins/Netscience/Diseasome, where brute-force optimum is checkable) → `fastgreedy.py` (approximate baseline) → scale up to the paper's full dataset list.

## Environment

This machine (Fedora) has **Podman**, not Docker Engine — the `Containerfile` uses standard Docker syntax and works with either:

```bash
podman build -t polardrl .
podman run --rm polardrl
```

Local (non-container) development uses [`uv`](https://docs.astral.sh/uv/), already installed:

```bash
uv venv --python 3.11        # system Python is 3.14, too new for some DRL libs (PyTorch/JAX) — pin 3.11
source .venv/bin/activate
uv pip install -e .
pytest
```

## Layout

```
src/polardrl/
  fj_model.py      Friedkin-Johnsen opinion-equilibrium computation
  pd_index.py      Polarization-Disagreement index (the objective)
  spgreedy.py      Exact greedy baseline (Algorithm 1)
  fastgreedy.py    Fast approximate baseline (Algorithms 2-3)
datasets/          Network data + fetch instructions (raw data gitignored)
tests/             pytest suite, starting with a hand-checkable P-D index fixture
```

## Reproducing the paper's results

1. Implement `fj_model.py` and `pd_index.py`, validate against `tests/test_pd_index.py`'s hand-computed fixture.
2. Implement `spgreedy.py`, run on the 4 small networks (Karate, Dolphins, Netscience, Diseasome) with k=1..8, compare Δ I(G) against brute-force optimum (feasible at this scale) and against the paper's reported numbers/figures.
3. Implement `fastgreedy.py`, verify it tracks SPGREEDY within ~1% on medium networks (paper reports ratio 0.988–0.998), then scale to larger datasets.
4. Once baselines are verified, this becomes the benchmark for the DRL agent (Phase 4, not yet started).
