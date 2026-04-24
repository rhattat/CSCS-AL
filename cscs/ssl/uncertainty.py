"""
Uncertainty metrics for SSL models.

The primary metric used in CSCS is multi-view inpainting variance (see extract_embeddings.py).
This module provides additional uncertainty definitions for experimentation.
"""

from __future__ import annotations

import numpy as np


def reconstruction_variance(reconstructions: np.ndarray) -> float:
    """
    Compute uncertainty as variance across multiple reconstructions.

    Args:
        reconstructions: Array of shape (n_views, *spatial), e.g. (9, 96, 96, 96).

    Returns:
        Mean variance across views and spatial dimensions.
    """
    return float(np.var(reconstructions, axis=0).mean())


def entropy_from_probabilities(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute per-voxel entropy from a probability map.

    Args:
        probs: Probability array, shape (N, C, *spatial) or (C, *spatial).
        eps:   Numerical stability constant.

    Returns:
        Entropy array, shape (N,) or scalar.
    """
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)
