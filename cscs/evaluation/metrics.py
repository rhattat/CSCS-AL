"""
Segmentation evaluation metrics.

These functions aggregate per-volume evaluation CSVs produced by nnU-Net
into a summary table suitable for comparison across methods and budgets.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Canonical column names expected in nnU-Net eval CSVs
DICE_COLUMNS = ["dice_mean_fg", "dice_mean_composite"]
HD95_COLUMNS = ["hd95_mean_fg", "hd95_mean_composite"]
METRIC_SCALE = {"dice": 100.0, "hd95": 1.0}  # dice: [0,1]→%, hd95: mm as-is


def load_eval_csv(csv_path: str | Path) -> Optional[pd.DataFrame]:
    """Load one nnU-Net eval summary CSV."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"  [ERR] Cannot read {csv_path}: {e}")
        return None


def normalize_method_name(raw: str, method_map: dict[str, str]) -> Optional[str]:
    """Map raw method string to canonical display name."""
    s = raw.lower().strip()
    for key, label in method_map.items():
        if s.startswith(key):
            return label
    return None


def discover_result_folders(
    root: str | Path,
    pattern: str = r"^(\w+)_results_cscsv5_k(\d+)_seed(\d+)$",
    datasets_filter: Optional[list[str]] = None,
    budgets_filter: Optional[list[int]] = None,
) -> list[tuple[str, int, int, Path]]:
    """
    Scan a root directory for result folders matching the naming pattern.

    Default pattern: {dataset}_results_cscsv5_k{K}_seed{S}

    Returns:
        List of (dataset, K, seed, folder_path) tuples.
    """
    root = Path(root)
    compiled = re.compile(pattern)
    configs = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        m = compiled.match(folder.name)
        if not m:
            continue
        ds, K, seed = m.group(1), int(m.group(2)), int(m.group(3))
        if datasets_filter and ds not in datasets_filter:
            continue
        if budgets_filter and K not in budgets_filter:
            continue
        configs.append((ds, K, seed, folder))
    return configs


def find_eval_csv(folder: Path, ds: str, K: int) -> Optional[Path]:
    """Locate the eval_summary CSV inside a result folder."""
    for pat in [
        f"eval_summary_k{K}_{ds}_composite.csv",
        f"eval_summary_k{K}_{ds}.csv",
        "eval_summary*.csv",
    ]:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def parse_result_row(
    row: pd.Series,
    ds: str,
    K: int,
    seed: int,
    metric_cols: dict[str, str],
    method_map: dict[str, str],
) -> Optional[dict]:
    """
    Parse one row of an eval CSV.

    Args:
        row:         One row from eval_summary CSV.
        ds:          Dataset name.
        K:           Budget.
        seed:        Seed index.
        metric_cols: Mapping of output label → CSV column name.
        method_map:  Mapping of raw method prefix → display label.

    Returns:
        Dict with method, dataset, K, seed, and metric values.
        None if the row should be skipped (failed train/predict, unknown method).
    """
    label = normalize_method_name(str(row.get("method", "")), method_map)
    if label is None:
        return None
    if str(row.get("train_success", "True")).strip() != "True":
        return None
    if str(row.get("predict_success", "True")).strip() != "True":
        return None

    result = {"method": label, "dataset": ds, "K": K, "seed": seed}
    for out_col, csv_col in metric_cols.items():
        val = float(row[csv_col]) if csv_col in row.index and not pd.isna(row.get(csv_col)) else np.nan
        if out_col.lower().startswith("dice"):
            val *= METRIC_SCALE["dice"]
        result[out_col] = round(val, 6)
    return result
