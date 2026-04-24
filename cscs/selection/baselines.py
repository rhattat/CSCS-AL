"""
Baseline active learning selectors for comparison with CSCS.

All selectors share a common output format:
    (selection_df, metadata)

where selection_df has columns:
    volume_id, uncertainty, typicality, cluster_id, score, selected, rank

Available methods:
    - random:            Pure random sampling
    - fps:               Farthest Point Sampling (greedy k-center)
    - typiclust:         K-means + highest typicality per cluster
    - probcover:         Greedy max-coverage via epsilon-ball graph
    - csal3d:            Multi-kernel k-means + typical + uncertain selection

Notes
-----
All baselines are re-implemented from scratch for the cold-start 3D medical
imaging setting (no GPU, no dataset-specific preprocessing, unified interface).
We cite the original papers and document algorithmic simplifications in each
function's docstring. Results may differ from original implementations.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances


# ---------------------------------------------------------------------------
# Shared output builder
# ---------------------------------------------------------------------------

def _build_result(
    volume_ids: list[str],
    U: np.ndarray,
    T: np.ndarray,
    selected_indices: list[int],
    cluster_labels: Optional[np.ndarray] = None,
    method: str = "unknown",
    metadata_extra: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """Build a standard (selection_df, metadata) from selected indices."""
    N = len(volume_ids)
    selected_set = set(selected_indices)

    rows = []
    for i, vid in enumerate(volume_ids):
        rows.append({
            "volume_id":  vid,
            "uncertainty": U[i],
            "typicality":  T[i],
            "cluster_id":  int(cluster_labels[i]) if cluster_labels is not None else -1,
            "score":       float(np.nan),  # baselines do not use composite score
            "selected":    i in selected_set,
        })

    df = pd.DataFrame(rows)
    df["rank"] = df["selected"].apply(lambda s: selected_indices.index(
        df.index[df["selected"]].tolist()[0]  # placeholder
    ) if s else N + 1)
    # Simpler rank: selected first, then rest
    df["rank"] = 0
    for rank, idx in enumerate(selected_indices, start=1):
        df.at[idx, "rank"] = rank
    unselected = df[~df["selected"]].index.tolist()
    for rank, idx in enumerate(unselected, start=len(selected_indices) + 1):
        df.at[idx, "rank"] = rank

    meta = {"method": method, "n_selected": len(selected_indices), "n_pool": N}
    if metadata_extra:
        meta.update(metadata_extra)
    return df, meta


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

def select_random(
    volume_ids: list[str],
    U: np.ndarray,
    T: np.ndarray,
    budget: int,
    seed: int = 42,
    **_,
) -> tuple[pd.DataFrame, dict]:
    """
    Random selection: uniform sampling without replacement.

    Standard baseline used in virtually all active learning benchmarks.
    No reference for the method itself; used as a lower-bound baseline.

    Simplifications vs typical AL setups: none — this is exact random sampling.

    Args:
        volume_ids: Pool volume IDs.
        U, T:       Uncertainty / typicality (kept for unified API, not used).
        budget:     Number to select.
        seed:       RNG seed.
    """
    N = len(volume_ids)
    rng = np.random.RandomState(seed)
    selected_indices = sorted(rng.choice(N, size=min(budget, N), replace=False).tolist())
    return _build_result(volume_ids, U, T, selected_indices, method="random",
                         metadata_extra={"seed": seed})


# ---------------------------------------------------------------------------
# FPS — Farthest Point Sampling
# ---------------------------------------------------------------------------

def select_fps(
    volume_ids: list[str],
    U: np.ndarray,
    T: np.ndarray,
    budget: int,
    embeddings: np.ndarray,
    seed: int = 42,
    **_,
) -> tuple[pd.DataFrame, dict]:
    """
    Farthest Point Sampling (FPS) — greedy k-center coreset construction.

    Maximises the minimum pairwise distance between selected points, providing
    a 2-approximation to the k-center objective (Gonzalez 1985).

    Original algorithmic reference:
        Gonzalez, T. (1985). Clustering to minimize the maximum intercluster
        distance. Theoretical Computer Science, 38, 293–306.

    Active learning application:
        Sener, O. & Savarese, S. (2018). Active Learning for Convolutional
        Neural Networks: A Core-Set Approach. ICLR 2018.

    Simplifications vs original AL setting:
        - Applied here in the cold-start setting (no previously labeled set).
        - First point drawn at random (seeded); not fixed to any labeled anchor.
        - Embeddings standardized with StandardScaler before distance computation.
        - No GPU: runs on CPU with numpy norms (feasible for N < 10k).

    Args:
        embeddings: SSL feature matrix (N, d). Standardized internally.
        seed:       Seed for random first-point selection.
    """
    N = len(volume_ids)
    K = min(budget, N)
    X = StandardScaler().fit_transform(embeddings)

    rng = np.random.RandomState(seed)
    selected = [int(rng.randint(N))]
    min_dists = np.full(N, np.inf)

    for _ in range(K - 1):
        d = np.linalg.norm(X - X[selected[-1]], axis=1)
        min_dists = np.minimum(min_dists, d)
        min_dists[selected] = -1.0
        selected.append(int(np.argmax(min_dists)))

    return _build_result(volume_ids, U, T, selected, method="fps",
                         metadata_extra={"seed": seed})


# ---------------------------------------------------------------------------
# TypiClust
# ---------------------------------------------------------------------------

def select_typiclust(
    volume_ids: list[str],
    U: np.ndarray,
    T: np.ndarray,
    budget: int,
    embeddings: np.ndarray,
    seed: int = 42,
    k_nn: int = 20,
    **_,
) -> tuple[pd.DataFrame, dict]:
    """
    TypiClust: K-means clustering, then select the most typical sample per cluster.

    Original paper:
        Hacohen, G., Dekel, A. & Weinshall, D. (2022). Active Learning on a
        Budget: Opposite Strategies Suit High and Low Budgets.
        ICML 2022. https://arxiv.org/abs/2202.02794

    Typicality is defined as the inverse mean k-NN distance in feature space.
    High typicality ≈ dense neighbourhood ≈ representative of the distribution.

    Simplifications vs original TypiClust:
        - Original uses k-NN over the full pool; here k-NN is computed within
          each cluster (faster, still meaningful for representativeness).
        - Original implementation uses FAISS for large-scale k-NN; here we use
          sklearn NearestNeighbors (exact, no GPU required).
        - Original applies a round-robin cluster-selection strategy across
          iterations; here we use K=budget clusters and pick one per cluster
          (cold-start single-round setting).

    Args:
        embeddings: SSL feature matrix (N, d). Standardized internally.
        k_nn:       Number of neighbours for typicality estimation (default 20).
        seed:       K-means random seed.
    """
    from sklearn.neighbors import NearestNeighbors

    N = len(volume_ids)
    K = min(budget, N)
    X = StandardScaler().fit_transform(embeddings)

    km = KMeans(n_clusters=K, init="k-means++", random_state=seed, n_init=10)
    labels = km.fit_predict(X)

    selected = []
    for cid in range(K):
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            continue
        if len(members) == 1:
            selected.append(int(members[0]))
            continue
        X_local = X[members]
        k_use = max(1, min(k_nn, len(members) - 1))
        nbrs = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean")
        nbrs.fit(X_local)
        dists, _ = nbrs.kneighbors(X_local)
        typ = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-8)
        selected.append(int(members[np.argmax(typ)]))

    return _build_result(volume_ids, U, T, selected, cluster_labels=labels,
                         method="typiclust", metadata_extra={"seed": seed})


# ---------------------------------------------------------------------------
# ProbCover
# ---------------------------------------------------------------------------

def select_probcover(
    volume_ids: list[str],
    U: np.ndarray,
    T: np.ndarray,
    budget: int,
    embeddings: np.ndarray,
    seed: int = 42,
    **_,
) -> tuple[pd.DataFrame, dict]:
    """
    ProbCover: greedy maximum-coverage selection via an epsilon-ball graph.

    Original paper:
        Yehuda, O., Dekel, A., Hacohen, G. & Weinshall, D. (2022).
        Active Learning Through a Covering Lens.
        NeurIPS 2022. https://arxiv.org/abs/2205.11320

    At each step, the algorithm greedily selects the point that covers the
    largest number of still-uncovered samples within a ball of radius delta,
    maximising coverage of the feature-space distribution.

    Simplifications vs original ProbCover:
        - Original estimates delta from a held-out validation set assumed to
          follow the true data distribution. Here delta is estimated via binary
          search on nearest-neighbour distances (no validation set required),
          targeting an average coverage of ~1.5 × N / K per selected point.
        - Original code uses a sparse adjacency graph (networkx/scipy); here we
          use dense pairwise Euclidean distances (feasible for N < ~5k).
        - Falls back to greedy FPS to fill any remaining quota if the epsilon-ball
          graph is exhausted before budget is reached.

    Args:
        embeddings: SSL feature matrix (N, d). Standardized internally.
        seed:       Unused (kept for unified API; ProbCover is deterministic).
    """
    N = len(volume_ids)
    K = min(budget, N)
    X = StandardScaler().fit_transform(embeddings)

    # Pairwise distances
    dists = euclidean_distances(X)
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)
    np.fill_diagonal(dists, 0.0)

    # Auto delta via binary search
    lo = float(np.percentile(nn_dists, 10))
    hi = float(np.percentile(nn_dists, 90))
    for _ in range(20):
        mid = (lo + hi) / 2
        avg_cov = (dists <= mid).sum(axis=1).mean()
        if avg_cov * K > N * 1.5:
            hi = mid
        else:
            lo = mid
    delta = (lo + hi) / 2

    covered = np.zeros(N, dtype=bool)
    selected: list[int] = []

    for _ in range(K):
        coverage = np.array(
            [np.sum((dists[i] <= delta) & ~covered) if i not in selected else -1
             for i in range(N)]
        )
        best = int(np.argmax(coverage))
        selected.append(best)
        covered[dists[best] <= delta] = True
        if covered.all():
            break

    # Fill remainder with FPS if needed
    if len(selected) < K:
        min_d = np.full(N, np.inf)
        for s in selected:
            min_d = np.minimum(min_d, np.linalg.norm(X - X[s], axis=1))
        for s in selected:
            min_d[s] = -1.0
        while len(selected) < K:
            best = int(np.argmax(min_d))
            selected.append(best)
            min_d[best] = -1.0
            min_d = np.minimum(min_d, np.linalg.norm(X - X[best], axis=1))

    return _build_result(volume_ids, U, T, selected[:K], method="probcover",
                         metadata_extra={"delta": delta})


# ---------------------------------------------------------------------------
# CSAL-3D
# ---------------------------------------------------------------------------

def select_csal3d(
    volume_ids: list[str],
    U: np.ndarray,
    T: np.ndarray,
    budget: int,
    embeddings: np.ndarray,
    seed: int = 42,
    n_cand: int = 3,
    **_,
) -> tuple[pd.DataFrame, dict]:
    """
    CSAL-3D: uncertainty-weighted clustering with typicality + uncertainty selection.

    Original paper:
        Zhu, H. et al. (2025). Cold-Start Active Learning for 3D Medical Image
        Segmentation. MICCAI 2025. https://arxiv.org/abs/2505.XXXXX
        Code: https://github.com/med-air/CSAL-3D

    CSAL-3D uses multi-kernel k-means (RBF kernels over uncertainty-weighted
    embeddings) to form clusters. Within each cluster, the top-n_cand most
    typical samples are identified, and the most uncertain among them is selected.
    This balances representativeness and informativeness.

    Simplifications vs original CSAL-3D:
        - Original uses multi-kernel k-means (sum of RBF kernels with multiple
          bandwidth values); here we use standard Euclidean K-means on
          uncertainty-weighted features (X * (1 + U_norm)), which approximates
          the uncertainty-aware clustering at much lower computational cost.
        - Original implementation relies on the CSAL-3D codebase and requires
          GPU for embedding extraction; this re-implementation is CPU-only.
        - Typicality is computed with sklearn NearestNeighbors (k=20) instead of
          FAISS.

    Args:
        embeddings: SSL feature matrix (N, d). Standardized internally.
        n_cand:     Candidate pool size per cluster for uncertainty tie-breaking.
        seed:       K-means random seed.
    """
    from sklearn.neighbors import NearestNeighbors

    N = len(volume_ids)
    K = min(budget, N)
    X = StandardScaler().fit_transform(embeddings)

    # Uncertainty-weighted features (scale X by normalized uncertainty)
    U_norm = (U - U.min()) / (U.max() - U.min() + 1e-10)
    X_weighted = X * (1.0 + U_norm[:, None])

    km = KMeans(n_clusters=K, init="k-means++", random_state=seed, n_init=10)
    labels = km.fit_predict(X_weighted)

    selected: list[int] = []
    rng = np.random.RandomState(seed)

    for cid in range(K):
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            continue
        if len(members) == 1:
            selected.append(int(members[0]))
            continue

        # Typicality within cluster
        X_local = X[members]
        k_use = max(1, min(20, len(members) - 1))
        nbrs = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean")
        nbrs.fit(X_local)
        dists, _ = nbrs.kneighbors(X_local)
        typ = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-8)

        # Top n_cand most typical → pick most uncertain
        n_actual = min(n_cand, len(members))
        top_typ = np.argsort(typ)[-n_actual:]
        best_local = int(top_typ[np.argmax(U[members][top_typ])])
        selected.append(int(members[best_local]))

    # Fill gaps
    if len(selected) < K:
        pool = [i for i in range(N) if i not in set(selected)]
        fill = rng.choice(pool, size=K - len(selected), replace=False).tolist()
        selected.extend(fill)

    return _build_result(volume_ids, U, T, selected[:K], cluster_labels=labels,
                         method="csal3d", metadata_extra={"seed": seed, "n_cand": n_cand})
