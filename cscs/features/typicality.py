"""
Typicality computation: density-based representativeness score.

    T(x) = 1 / (mean k-NN distance + eps)

High typicality → sample lies in a dense region → representative of the distribution.
Low typicality → sample is an outlier or atypical point.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_typicality(
    embeddings: np.ndarray,
    k: int = 20,
    eps: float = 1e-8,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute typicality for each sample in the pool.

        T(x_i) = 1 / (mean_{j in kNN(i)} dist(x_i, x_j) + eps)

    Args:
        embeddings: Feature matrix, shape (N, d).
        k:          Number of nearest neighbors (default 20).
        eps:        Numerical stability constant to avoid division by zero.
        metric:     Distance metric for k-NN (default "euclidean").

    Returns:
        typicality: Array of shape (N,), higher = more representative.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    N = embeddings.shape[0]
    k_eff = min(k, N - 1)
    if k_eff < 1:
        return np.ones(N)

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    nbrs.fit(embeddings)
    dists, _ = nbrs.kneighbors(embeddings)

    # dists[:, 0] is always 0 (self), so we skip it
    mean_knn_dist = dists[:, 1:].mean(axis=1)
    return 1.0 / (mean_knn_dist + eps)


def compute_typicality_faiss(
    embeddings: np.ndarray,
    k: int = 20,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Faster typicality using FAISS (squared L2 distances, as in TypiClust).

    Falls back to sklearn if FAISS is not installed.

    Args:
        embeddings: Feature matrix, shape (N, d), will be cast to float32.
        k:          Number of nearest neighbors.
        eps:        Stability constant.

    Returns:
        typicality: Array of shape (N,).
    """
    try:
        import faiss
        embeddings = np.asarray(embeddings, dtype=np.float32)
        N, d = embeddings.shape
        k_eff = min(k, N - 1)
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        # k_eff + 1 because the first result is the sample itself (distance 0)
        dists_sq, _ = index.search(embeddings, k_eff + 1)
        mean_sq_dist = dists_sq[:, 1:].mean(axis=1)
        return 1.0 / (mean_sq_dist + eps)
    except ImportError:
        return compute_typicality(embeddings, k=k, eps=eps)
