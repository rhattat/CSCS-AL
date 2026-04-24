"""
Clustering utilities for CSCS.

CSCS uses K-means++ to partition the unlabeled pool into B clusters
(one cluster per sample to select), ensuring spatial diversity.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_embeddings(
    embeddings: np.ndarray,
    K: int,
    seed: int = 42,
    scale: bool = True,
    n_init: int = 10,
) -> np.ndarray:
    """
    K-means++ clustering of embeddings into K clusters.

    Args:
        embeddings: Feature matrix, shape (N, d)
        K:          Number of clusters (= selection budget B)
        seed:       Random seed for reproducibility
        scale:      If True, standardize embeddings before clustering
        n_init:     Number of K-means restarts (higher = more stable)

    Returns:
        labels: Cluster assignments, shape (N,), values in [0, K-1]

    Raises:
        ValueError: If K > N (more clusters than samples)
    """
    N = len(embeddings)
    if K > N:
        raise ValueError(
            f"Cannot create {K} clusters from {N} samples. "
            f"Reduce budget or increase pool size."
        )

    X = StandardScaler().fit_transform(embeddings) if scale else np.asarray(embeddings, dtype=float)

    km = KMeans(
        n_clusters=K,
        init="k-means++",
        random_state=seed,
        n_init=n_init,
        max_iter=300,
    )
    labels = km.fit_predict(X)
    return labels


def get_cluster_sizes(labels: np.ndarray, K: int) -> np.ndarray:
    """Return size of each cluster, shape (K,)."""
    return np.bincount(labels, minlength=K)


def handle_small_clusters(
    labels: np.ndarray,
    K: int,
    s_min: int = 3,
) -> tuple[np.ndarray, list[int]]:
    """
    Identify clusters smaller than s_min and flag them as "thin".

    CSCS strategy for thin clusters:
    - They are included in selection but their samples go into a fallback pool.
    - If fewer than B samples are ultimately selected (one per valid cluster),
      the remainder is filled using global score ranking.

    Args:
        labels: Cluster labels, shape (N,)
        K:      Total number of clusters
        s_min:  Minimum cluster size threshold (default 3)

    Returns:
        labels:       Unchanged original labels
        thin_clusters: List of cluster IDs with fewer than s_min members
    """
    sizes = get_cluster_sizes(labels, K)
    thin_clusters = [cid for cid in range(K) if sizes[cid] < s_min]
    if thin_clusters:
        warnings.warn(
            f"{len(thin_clusters)} cluster(s) have fewer than {s_min} samples "
            f"(cluster IDs: {thin_clusters}). "
            f"These will be skipped and gaps filled by global score ranking.",
            UserWarning,
            stacklevel=2,
        )
    return labels, thin_clusters


def fallback_embedding(n_samples: int, U: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Build a 2-D fallback embedding from (U, T) when no embedding directory is available.

    Returns a standardized 2-column matrix usable for clustering.
    """
    X = np.column_stack([U, T])
    return StandardScaler().fit_transform(X)
