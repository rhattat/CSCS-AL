"""
CSCS composite score computation.

Score formula (from paper):
    S(x) = T_rank(x)^(1 - gamma) * U_rank(x)^gamma

where T_rank and U_rank are rank-normalized versions of typicality T and
uncertainty U within the local cluster (or globally if no clustering is used).

Rank normalization:
    rank_norm(x_i) = rank(x_i) / (n - 1), clipped to [eps, 1.0]
    This makes the score scale-invariant and bounded in (0, 1].
"""

from __future__ import annotations

import numpy as np


def rank_normalize(arr: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """
    Rank-normalize a 1-D array to [eps, 1.0].

    The smallest value gets rank 0/(n-1)=0 → clipped to eps.
    The largest value gets rank (n-1)/(n-1)=1.0.

    Args:
        arr: Input array, shape (N,)
        eps: Lower clip bound (avoids zero scores, default 0.01)

    Returns:
        Rank-normalized array in [eps, 1.0], shape (N,)
    """
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 1:
        return np.array([1.0])
    ranks = np.argsort(np.argsort(arr)).astype(float) / (n - 1)
    return np.clip(ranks, eps, 1.0)


def compute_score(
    T: np.ndarray,
    U: np.ndarray,
    gamma: float,
    eps: float = 0.01,
) -> np.ndarray:
    """
    Compute CSCS composite score: S(x) = T_rank^(1-gamma) * U_rank^gamma.

    T and U are rank-normalized within the provided array (intended to be
    called per-cluster, so ranks are local).

    Args:
        T:     Typicality scores, shape (N,)
        U:     Uncertainty scores, shape (N,)
        gamma: Balance parameter in [0, 1].
               gamma=0 → pure typicality (T only)
               gamma=1 → pure uncertainty (U only)
               gamma=0.5 → equal weight
        eps:   Lower clip bound for rank normalization

    Returns:
        S: Composite scores, shape (N,), in (0, 1]
    """
    T = np.asarray(T, dtype=float)
    U = np.asarray(U, dtype=float)
    T_rank = rank_normalize(T, eps=eps)
    U_rank = rank_normalize(U, eps=eps)
    return (T_rank ** (1.0 - gamma)) * (U_rank ** gamma)


def global_score_ranking(
    T: np.ndarray,
    U: np.ndarray,
    gamma: float,
    exclude_indices: set[int] | None = None,
) -> np.ndarray:
    """
    Compute global scores and return sorted indices (highest score first).

    Used to fill missing selections when clustering leaves gaps.

    Args:
        T:               Global typicality scores, shape (N,)
        U:               Global uncertainty scores, shape (N,)
        gamma:           Balance parameter
        exclude_indices: Indices to exclude from the ranking (already selected)

    Returns:
        Sorted indices by descending score, excluding already-selected ones.
    """
    scores = compute_score(T, U, gamma)
    order = np.argsort(scores)[::-1]
    if exclude_indices:
        order = np.array([i for i in order if i not in exclude_indices])
    return order
