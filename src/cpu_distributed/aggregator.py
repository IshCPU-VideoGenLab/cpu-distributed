"""Fitness aggregation strategies for ES training."""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def rank_normalize(fitnesses: List[float]) -> np.ndarray:
    """Rank-based fitness normalization for robust ES updates.

    Converts raw fitness values to centered, normalized ranks.
    This makes ES robust to fitness scale and outliers.

    Args:
        fitnesses: Raw fitness values.

    Returns:
        Normalized rank weights, centered around 0.
    """
    n = len(fitnesses)
    if n == 0:
        return np.array([])

    fit = np.array(fitnesses, dtype=np.float32)
    ranks = np.argsort(np.argsort(fit)).astype(np.float32)
    centered = (ranks / (n - 1)) - 0.5
    std = centered.std()
    if std > 0:
        centered /= std
    return centered


def fitness_shaping(fitnesses: List[float], method: str = "rank") -> np.ndarray:
    """Apply fitness shaping to raw fitness values.

    Args:
        fitnesses: Raw fitness values.
        method: Shaping method ("rank", "centered", "raw").

    Returns:
        Shaped fitness weights.
    """
    if method == "rank":
        return rank_normalize(fitnesses)
    elif method == "centered":
        fit = np.array(fitnesses, dtype=np.float32)
        return (fit - fit.mean()) / (fit.std() + 1e-8)
    elif method == "raw":
        return np.array(fitnesses, dtype=np.float32)
    else:
        raise ValueError(f"Unknown fitness shaping method: {method}")
