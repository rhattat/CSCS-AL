#!/usr/bin/env python3
"""
Aggregate nnU-Net evaluation results across seeds and budgets.

Expected folder structure (auto-detected):
    results_root/
        {dataset}_results_cscsv5_k{K}_seed{S}/
            eval_summary_k{K}_{dataset}.csv

Usage:
    python scripts/summarize_results.py \\
        --root outputs/Article/ \\
        --outdir outputs/tables/ \\
        --datasets brats spleen feta diane \\
        --latex

Output:
    {outdir}/{dataset}_main_table.csv   — mean±std per method × budget
    {outdir}/all_datasets_summary.csv  — all datasets combined
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cscs.evaluation.metrics import (
    discover_result_folders, find_eval_csv, parse_result_row,
)
from cscs.evaluation.summarize import aggregate_seeds, format_cell

import numpy as np
import pandas as pd


DATASET_METRICS = {
    "brats":  {"Dice_fg": "dice_mean_fg", "Dice_comp": "dice_mean_composite",
               "HD95_fg": "hd95_mean_fg", "HD95_comp": "hd95_mean_composite"},
    "spleen": {"Dice_fg": "dice_mean_fg", "HD95_fg": "hd95_mean_fg"},
    "feta":   {"Dice_fg": "dice_mean_fg", "HD95_fg": "hd95_mean_fg"},
    "diane":  {"Dice_fg": "dice_mean_fg", "HD95_fg": "hd95_mean_fg"},
}

METHOD_MAP = {
    "random":    "Random",
    "typiclust": "TypiClust",
    "fps":       "FPS",
    "probcover": "ProbCover",
    "csal3d":    "CSAL-3D",
    "cscs_v5":   "CSCS",
    "cscs":      "CSCS",
}

METHOD_ORDER = ["Random", "TypiClust", "FPS", "ProbCover", "CSAL-3D", "CSCS"]


def print_table(agg_df: pd.DataFrame, ds: str, metrics: dict, latex: bool = False) -> None:
    ds_df = agg_df[agg_df["dataset"] == ds]
    if ds_df.empty:
        return
    budgets = sorted(ds_df["K"].unique())
    met_cols = list(metrics.keys())

    print(f"\n{'='*80}")
    print(f"  {ds.upper()}  — mean±std (Dice in %, HD95 in mm)")
    print(f"{'='*80}")
    header = f"  {'Method':<14}" + "".join(
        f"  K={K:>4} " + " ".join(f"{m:>12}" for m in met_cols)
        for K in budgets
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for method in METHOD_ORDER:
        parts = []
        for K in budgets:
            sub = ds_df[(ds_df["method"] == method) & (ds_df["K"] == K)]
            part = f"  K={K:>4} "
            for m in met_cols:
                if sub.empty or f"{m}_mean" not in sub.columns:
                    part += f"  {'—':>12}"
                else:
                    mean = sub[f"{m}_mean"].values[0]
                    std  = sub[f"{m}_std"].values[0] if f"{m}_std" in sub.columns else 0.0
                    n    = int(sub["n_seeds"].values[0]) if "n_seeds" in sub.columns else 3
                    cell = format_cell(mean, std, is_dice=m.startswith("Dice"), n=n)
                    part += f"  {cell:>12}"
            parts.append(part)
        print(f"  {method:<14}" + "".join(parts))

    if latex:
        print(f"\n  --- LaTeX rows ({ds}, Dice_fg only) ---")
        for method in METHOD_ORDER:
            cells = []
            for K in budgets:
                sub = ds_df[(ds_df["method"] == method) & (ds_df["K"] == K)]
                if sub.empty:
                    cells.append("—")
                    continue
                mean = sub["Dice_fg_mean"].values[0]
                std  = sub["Dice_fg_std"].values[0]
                n    = int(sub["n_seeds"].values[0]) if "n_seeds" in sub.columns else 3
                cells.append(format_cell(mean, std, is_dice=True, n=n))
            print(f"  {method} & " + " & ".join(cells) + r" \\")


def main():
    p = argparse.ArgumentParser(description="Aggregate CSCS evaluation results.")
    p.add_argument("--root", type=str, required=True, help="Results root directory")
    p.add_argument("--outdir", type=str, required=True, help="Output directory for tables")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--budgets", nargs="+", type=int, default=None)
    p.add_argument("--latex", action="store_true", help="Print LaTeX rows")
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = discover_result_folders(root, datasets_filter=args.datasets,
                                      budgets_filter=args.budgets)
    if not configs:
        print(f"No result folders found in {root}")
        sys.exit(1)

    all_raw = []
    for ds, K, seed, folder in configs:
        metrics = DATASET_METRICS.get(ds)
        if not metrics:
            continue
        csv_path = find_eval_csv(folder, ds, K)
        if not csv_path:
            continue
        try:
            df_csv = pd.read_csv(csv_path)
        except Exception:
            continue
        for _, row in df_csv.iterrows():
            r = parse_result_row(row, ds, K, seed, metrics, METHOD_MAP)
            if r:
                all_raw.append(r)

    if not all_raw:
        print("No data parsed.")
        sys.exit(1)

    datasets_found = sorted(set(r["dataset"] for r in all_raw))
    all_agg = []
    for ds in datasets_found:
        metrics = DATASET_METRICS.get(ds, {})
        ds_rows = [r for r in all_raw if r["dataset"] == ds]
        agg = aggregate_seeds(ds_rows, list(metrics.keys()))
        all_agg.append(agg)
        print_table(agg, ds, metrics, latex=args.latex)
        agg.to_csv(out_dir / f"{ds}_main_table.csv", index=False)

    grand = pd.concat(all_agg, ignore_index=True)
    grand.to_csv(out_dir / "all_datasets_summary.csv", index=False)
    print(f"\nSaved tables to {out_dir}/")


if __name__ == "__main__":
    main()
