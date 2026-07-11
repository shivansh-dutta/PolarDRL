"""
Internal-opinion generators for the three distributions used in Zhu et al.
2021 (Sec 7, "Opinions and evaluation metrics"): uniform, exponential, and
power-law, each normalized to [0, 1] with at least one node forced to 1
(a direct consequence of min-max normalization against the sample maximum).
"""
from __future__ import annotations

import numpy as np


def _normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; the sample max is mapped to exactly 1."""
    low, high = values.min(), values.max()
    if high == low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    """s_i ~ U[0, 1] i.i.d. -- already in [0, 1], no renormalization needed."""
    return rng.uniform(0.0, 1.0, size=n)


def exponential(n: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """s_i drawn from an exponential distribution, then normalized to [0, 1]."""
    raw = rng.exponential(scale=scale, size=n)
    return _normalize(raw)


def powerlaw(n: int, rng: np.random.Generator, alpha: float = 2.5) -> np.ndarray:
    """
    s_i drawn from a power-law (Pareto) distribution with shape alpha,
    then normalized to [0, 1]. alpha=2.5 matches a typical heavy-tailed
    slope in the spirit of the paper's randht.py-generated opinions.
    """
    raw = rng.pareto(alpha, size=n)
    return _normalize(raw)


# Name -> generator registry for scripts iterating over all three distributions.
GENERATORS = {
    "uniform": uniform,
    "exponential": exponential,
    "powerlaw": powerlaw,
}
