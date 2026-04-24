"""
Score normalization utilities.

Min-max normalization maps any score array to [0, 1].
Used to produce the *_normalized columns in the features CSV.
"""

from __future__ import annotations

import numpy as np


def minmax_normalize(scores: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    """
    Min-max normalization to [0, 1].

    If all values are equal (range ≈ 0), returns an array of zeros.

    Args:
        scores: Input array, shape (N,).
        eps:    Threshold for detecting constant arrays.

    Returns:
        Normalized array in [0, 1], shape (N,).
    """
    scores = np.asarray(scores, dtype=float)
    s_min, s_max = scores.min(), scores.max()
    denom = s_max - s_min
    if np.isclose(denom, 0.0, atol=eps):
        return np.zeros_like(scores)
    return (scores - s_min) / denom


def zscore_normalize(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalization (mean=0, std=1).

    Args:
        scores: Input array, shape (N,).
        eps:    Stability constant for near-zero std.

    Returns:
        Standardized array, shape (N,).
    """
    scores = np.asarray(scores, dtype=float)
    std = scores.std()
    return (scores - scores.mean()) / (std if std > eps else 1.0)
