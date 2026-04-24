"""
CSCS: Cold-Start Curriculum Selection.

Main entry-points:
    select_cscs(features_csv, budget, output_dir, ...)
    select_cscs_from_df(df, embeddings, budget, ...)

Algorithm (as defined in the paper):
    1. Load typicality T and uncertainty U for the unlabeled pool.
    2. Compute DCR = Spearman(U, T).
    3. Compute alpha_eff = B / sqrt(N), gamma = f(DCR, alpha_eff).
    4. Cluster embeddings into B clusters via K-means++.
    5. Within each cluster: select argmax S(x) = T^(1-gamma) * U^gamma
       (rank-normalized within the cluster).
    6. Skip clusters with < s_min samples; fill gaps via global score ranking.
    7. Return selection table and metadata.

Input CSV format:
    volume_id, uncertainty, typicality[, split, uncertainty_normalized, typicality_normalized, ...]

Output DataFrame columns:
    volume_id, uncertainty, typicality, cluster_id, score, selected, rank
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .gamma_scheduler import compute_dcr, compute_gamma, gamma_summary
from .scoring import compute_score, global_score_ranking
from .clustering import cluster_embeddings, handle_small_clusters, fallback_embedding


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_cscs(
    features_csv: str | Path,
    budget: int,
    output_dir: str | Path,
    embeddings_dir: Optional[str | Path] = None,
    k_neighbors: int = 20,
    s_min: int = 3,
    seed: int = 42,
    gamma_lo: float = 0.3,
    gamma_hi: float = 0.7,
    volume_id_col: str = "volume_id",
    uncertainty_col: str = "uncertainty",
    typicality_col: str = "typicality",
    split_col: Optional[str] = "split",
    split_value: str = "train",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Run CSCS selection from a features CSV file.

    Args:
        features_csv:    Path to CSV with volume_id, uncertainty, typicality columns.
        budget:          Number of samples to select (B).
        output_dir:      Directory to write selection outputs.
        embeddings_dir:  Directory containing one .npy file per volume (for clustering).
                         If None, clustering falls back to (U, T) 2-D space.
        k_neighbors:     Not used here (typicality is precomputed); kept for API compat.
        s_min:           Minimum cluster size. Thin clusters are skipped and gaps
                         filled by global score ranking.
        seed:            Random seed for K-means++ reproducibility.
        gamma_lo:        Lower clip bound for gamma (default 0.3).
        gamma_hi:        Upper clip bound for gamma (default 0.7).
        volume_id_col:   Column name for volume IDs in features_csv.
        uncertainty_col: Column name for uncertainty scores.
        typicality_col:  Column name for typicality scores.
        split_col:       Column name for split labels (e.g., "train"/"val").
                         If None, all rows are used.
        split_value:     Which split to select from (default "train").
        verbose:         Print progress messages.

    Returns:
        selection_df:  DataFrame with one row per pool volume, containing:
                       volume_id, uncertainty, typicality, cluster_id,
                       score, selected (bool), rank (1 = best)
        metadata:      Dict with dcr, alpha_eff, gamma, budget, n_pool, seed, ...
    """
    features_csv = Path(features_csv)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load features ---
    df = pd.read_csv(features_csv)
    df.columns = [c.strip() for c in df.columns]

    # Filter to train split if split column exists
    if split_col and split_col in df.columns:
        n_before = len(df)
        df = df[df[split_col] == split_value].copy()
        if verbose and len(df) < n_before:
            print(f"[CSCS] Filtered to split='{split_value}': "
                  f"{len(df)}/{n_before} volumes")
    elif split_col:
        warnings.warn(
            f"split_col='{split_col}' not found in CSV. Using all rows.",
            UserWarning, stacklevel=2
        )

    if volume_id_col not in df.columns:
        raise ValueError(f"Column '{volume_id_col}' not found. Available: {list(df.columns)}")
    if uncertainty_col not in df.columns:
        raise ValueError(f"Column '{uncertainty_col}' not found. Available: {list(df.columns)}")
    if typicality_col not in df.columns:
        raise ValueError(f"Column '{typicality_col}' not found. Available: {list(df.columns)}")

    df = df.reset_index(drop=True)
    volume_ids = df[volume_id_col].astype(str).tolist()
    U = df[uncertainty_col].values.astype(float)
    T = df[typicality_col].values.astype(float)

    # --- Load embeddings ---
    embeddings = _load_embeddings(embeddings_dir, volume_ids, verbose)
    if embeddings is None:
        warnings.warn(
            "No embeddings found. Falling back to (U, T) 2-D space for clustering. "
            "Diversity-based selection may be suboptimal.",
            UserWarning, stacklevel=2
        )
        embeddings = fallback_embedding(len(volume_ids), U, T)

    # --- Core selection ---
    selection_df, meta = select_cscs_from_df(
        volume_ids=volume_ids,
        U=U,
        T=T,
        embeddings=embeddings,
        budget=budget,
        s_min=s_min,
        seed=seed,
        gamma_lo=gamma_lo,
        gamma_hi=gamma_hi,
        verbose=verbose,
    )

    # --- Persist outputs ---
    _save_outputs(selection_df, meta, output_dir, verbose)

    return selection_df, meta


def select_cscs_from_df(
    volume_ids: list[str],
    U: np.ndarray,
    T: np.ndarray,
    embeddings: np.ndarray,
    budget: int,
    s_min: int = 3,
    seed: int = 42,
    gamma_lo: float = 0.3,
    gamma_hi: float = 0.7,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Core CSCS selection from in-memory arrays.

    Args:
        volume_ids: List of N volume IDs.
        U:          Uncertainty scores, shape (N,).
        T:          Typicality scores, shape (N,).
        embeddings: Feature matrix for clustering, shape (N, d).
        budget:     Number of samples to select (B).
        s_min:      Minimum cluster size threshold.
        seed:       Random seed.
        gamma_lo:   Lower gamma clip bound.
        gamma_hi:   Upper gamma clip bound.
        verbose:    Print progress.

    Returns:
        (selection_df, metadata)
    """
    N = len(volume_ids)
    if budget > N:
        raise ValueError(f"Budget {budget} exceeds pool size {N}.")

    # Step 1: DCR
    dcr, dcr_pval = compute_dcr(U, T)

    # Step 2: gamma
    gamma, alpha_eff = compute_gamma(budget, N, dcr, gamma_lo, gamma_hi)

    if verbose:
        sig = ("***" if dcr_pval < 0.001 else "**" if dcr_pval < 0.01
               else "*" if dcr_pval < 0.05 else "n.s.")
        print(f"[CSCS] N={N}, B={budget}, DCR={dcr:+.4f} ({sig}), "
              f"alpha_eff={alpha_eff:.3f}, gamma={gamma:.3f}")

    # Step 3: K-means++ clustering
    labels = cluster_embeddings(embeddings, K=budget, seed=seed)

    # Step 4: Handle small clusters
    labels, thin_clusters = handle_small_clusters(labels, K=budget, s_min=s_min)
    thin_set = set(thin_clusters)

    # Step 5: Per-cluster selection
    selected_indices: list[int] = []
    cluster_of_selected: list[int] = []

    for cid in range(budget):
        if cid in thin_set:
            continue
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            continue
        if len(members) == 1:
            best = int(members[0])
        else:
            S_local = compute_score(T[members], U[members], gamma)
            best = int(members[np.argmax(S_local)])
        selected_indices.append(best)
        cluster_of_selected.append(cid)

    # Step 6: Fill gaps via global score ranking
    n_selected = len(selected_indices)
    if n_selected < budget:
        n_fill = budget - n_selected
        if verbose:
            print(f"[CSCS] {n_fill} gap(s) due to thin clusters — filling via global ranking.")
        exclude = set(selected_indices)
        order = global_score_ranking(T, U, gamma, exclude_indices=exclude)
        fill_indices = order[:n_fill].tolist()
        selected_indices.extend(fill_indices)
        cluster_of_selected.extend([-1] * n_fill)

    # Step 7: Build selection DataFrame
    selected_set = set(selected_indices)

    # Global scores for full ranking table
    global_scores = compute_score(T, U, gamma)

    rows = []
    for i, vid in enumerate(volume_ids):
        rows.append({
            "volume_id":   vid,
            "uncertainty": U[i],
            "typicality":  T[i],
            "cluster_id":  int(labels[i]),
            "score":       float(global_scores[i]),
            "selected":    i in selected_set,
        })

    result_df = pd.DataFrame(rows)
    # Rank by score descending (rank 1 = best overall score)
    result_df["rank"] = result_df["score"].rank(ascending=False, method="first").astype(int)
    result_df = result_df.sort_values("rank").reset_index(drop=True)

    metadata = {
        "method":    "cscs",
        "budget":    budget,
        "n_pool":    N,
        "dcr":       dcr,
        "dcr_pval":  dcr_pval,
        "alpha_eff": alpha_eff,
        "gamma":     gamma,
        "gamma_lo":  gamma_lo,
        "gamma_hi":  gamma_hi,
        "s_min":     s_min,
        "seed":      seed,
        "n_thin_clusters": len(thin_clusters),
        "thin_cluster_ids": thin_clusters,
        "n_selected": len(selected_indices),
    }

    return result_df, metadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_embeddings(
    embeddings_dir: Optional[str | Path],
    volume_ids: list[str],
    verbose: bool,
) -> Optional[np.ndarray]:
    """Load .npy embeddings from directory; return None if unavailable."""
    if embeddings_dir is None:
        return None
    emb_dir = Path(embeddings_dir)
    if not emb_dir.is_dir():
        warnings.warn(f"Embeddings directory not found: {emb_dir}", UserWarning, stacklevel=3)
        return None

    embs = []
    missing = []
    for vid in volume_ids:
        fp = emb_dir / f"{vid}.npy"
        if fp.exists():
            arr = np.load(fp)
            embs.append(arr.flatten())
        else:
            missing.append(vid)
            embs.append(None)

    if missing:
        warnings.warn(
            f"{len(missing)} embedding file(s) missing in {emb_dir}. "
            f"These volumes will use zero vectors as fallback.",
            UserWarning, stacklevel=3
        )
        dim = next(e.shape[0] for e in embs if e is not None)
        embs = [e if e is not None else np.zeros(dim) for e in embs]

    if all(e is None for e in embs):
        return None

    matrix = np.stack(embs)
    if verbose:
        print(f"[CSCS] Loaded embeddings: {matrix.shape} from {emb_dir}")
    return matrix


def _save_outputs(
    df: pd.DataFrame,
    meta: dict,
    output_dir: Path,
    verbose: bool,
) -> None:
    """Write selection table, selected IDs, and metadata to output_dir."""
    import json

    # Full ranking table
    df.to_csv(output_dir / "selection_table.csv", index=False)

    # Selected IDs only
    selected = df[df["selected"]]["volume_id"].tolist()
    pd.DataFrame({"volume_id": selected}).to_csv(
        output_dir / "selected_ids.csv", index=False
    )

    # Metadata JSON
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"[CSCS] Saved {len(selected)} selected IDs → {output_dir}/")
