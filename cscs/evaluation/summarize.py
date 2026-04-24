"""
Results aggregation across seeds: mean ± std per (method, dataset, K).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_seeds(
    raw_rows: list[dict],
    metric_cols: list[str],
    n_expected_seeds: int = 3,
) -> pd.DataFrame:
    """
    Aggregate raw per-seed result rows into mean ± std.

    Args:
        raw_rows:         List of dicts with keys: method, dataset, K, seed, *metrics.
        metric_cols:      List of metric column names to aggregate.
        n_expected_seeds: Expected number of seeds (for completeness warnings).

    Returns:
        DataFrame with columns: method, dataset, K, n_seeds, {metric}_mean, {metric}_std.
    """
    if not raw_rows:
        return pd.DataFrame()

    df = pd.DataFrame(raw_rows)
    agg_rows = []

    for (method, ds, K), grp in df.groupby(["method", "dataset", "K"]):
        row = {"method": method, "dataset": ds, "K": K, "n_seeds": len(grp)}
        for col in metric_cols:
            vals = grp[col].dropna().values if col in grp.columns else []
            row[f"{col}_mean"] = round(float(np.mean(vals)), 4) if len(vals) > 0 else np.nan
            row[f"{col}_std"]  = round(float(np.std(vals)),  4) if len(vals) > 1 else 0.0
            row[f"{col}_n"]    = len(vals)
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows).sort_values(["dataset", "K", "method"]).reset_index(drop=True)
    return agg_df


def format_cell(mean: float, std: float, is_dice: bool = True, n: int = 3) -> str:
    """Format a mean±std cell for console/LaTeX tables."""
    if np.isnan(mean):
        return "—"
    decimals = 1 if is_dice else 2
    marker = "†" if n < 3 else ""
    if std == 0 or np.isnan(std):
        return f"{mean:.{decimals}f}{marker}"
    return f"{mean:.{decimals}f}±{std:.{decimals}f}{marker}"
