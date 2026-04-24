"""
Feature I/O utilities.

Handles loading and validating the features CSV that serves as input to all selectors.

Expected CSV format (minimum):
    volume_id, uncertainty, typicality

Optional columns (auto-detected):
    split, uncertainty_normalized, typicality_normalized

Embeddings are stored as separate .npy files (one per volume):
    embeddings_dir/{volume_id}.npy
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["volume_id", "uncertainty", "typicality"]
OPTIONAL_COLUMNS = ["split", "uncertainty_normalized", "typicality_normalized"]


def load_features(
    csv_path: str | Path,
    split: Optional[str] = "train",
    split_col: str = "split",
) -> pd.DataFrame:
    """
    Load and validate a features CSV.

    Args:
        csv_path:  Path to the CSV file.
        split:     If not None, filter rows where split_col == split.
        split_col: Column name for train/val split labels.

    Returns:
        DataFrame with at least: volume_id, uncertainty, typicality.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) in {csv_path.name}: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df["volume_id"] = df["volume_id"].astype(str).str.strip()
    df["uncertainty"] = pd.to_numeric(df["uncertainty"], errors="coerce")
    df["typicality"] = pd.to_numeric(df["typicality"], errors="coerce")

    n_nan = df[["uncertainty", "typicality"]].isna().any(axis=1).sum()
    if n_nan > 0:
        warnings.warn(
            f"{n_nan} row(s) with NaN uncertainty/typicality dropped from {csv_path.name}.",
            UserWarning, stacklevel=2,
        )
        df = df.dropna(subset=["uncertainty", "typicality"])

    if split is not None:
        if split_col in df.columns:
            n_before = len(df)
            df = df[df[split_col] == split].copy()
            if len(df) == 0:
                raise ValueError(
                    f"No rows with {split_col}='{split}' in {csv_path.name}. "
                    f"Available values: {df[split_col].unique().tolist()}"
                )
        else:
            warnings.warn(
                f"split_col='{split_col}' not found in {csv_path.name}. Using all rows.",
                UserWarning, stacklevel=2,
            )

    return df.reset_index(drop=True)


def load_embeddings_dir(
    embeddings_dir: str | Path,
    volume_ids: list[str],
    flatten: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Load embeddings from a directory of .npy files.

    Args:
        embeddings_dir: Directory containing {volume_id}.npy files.
        volume_ids:     List of volume IDs to load (order is preserved).
        flatten:        If True, flatten multi-dimensional embeddings to 1-D.

    Returns:
        embeddings:    Array of shape (N_found, d).
        found_ids:     Volume IDs for which embeddings were found (same order).
    """
    emb_dir = Path(embeddings_dir)
    if not emb_dir.is_dir():
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")

    found_ids, arrays = [], []
    missing = []
    for vid in volume_ids:
        fp = emb_dir / f"{vid}.npy"
        if fp.exists():
            arr = np.load(fp)
            if flatten and arr.ndim > 1:
                arr = arr.flatten()
            arrays.append(arr)
            found_ids.append(vid)
        else:
            missing.append(vid)

    if missing:
        warnings.warn(
            f"{len(missing)} .npy file(s) not found in {emb_dir}: {missing[:5]}{'...' if len(missing) > 5 else ''}",
            UserWarning, stacklevel=2,
        )
    if not arrays:
        raise FileNotFoundError(f"No embeddings found for any of the {len(volume_ids)} volume IDs in {emb_dir}")

    return np.stack(arrays), found_ids


def save_features(
    df: pd.DataFrame,
    output_path: str | Path,
    index: bool = False,
) -> None:
    """Save a features DataFrame to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)


def normalize_volume_id(vid: str, suffixes: tuple[str, ...] = ("_0000", ".nii.gz", ".nii")) -> str:
    """Strip common file suffixes from a volume ID string."""
    vid = str(vid).strip()
    for s in suffixes:
        vid = vid.replace(s, "")
    return vid
