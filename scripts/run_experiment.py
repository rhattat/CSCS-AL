#!/usr/bin/env python3
"""
Run a full CSCS experiment: selection for all methods × budgets × seeds.

Usage:
    python scripts/run_experiment.py --config configs/experiments/example_cscs.yaml
    python scripts/run_experiment.py --config configs/experiments/example_cscs.yaml \\
        --methods cscs random --seeds 42 123 --dry_run

Output structure:
    {output_root}/{dataset}/selections/{method}_k{K}_seed{seed}/
        selected_ids.csv
        selection_table.csv
        metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from cscs.utils.config import load_experiment_config
from cscs.utils.seed import set_seed
from cscs.features.io import load_features, load_embeddings_dir
from cscs.selection.registry import get_selector, list_methods


def parse_args():
    p = argparse.ArgumentParser(description="Run full CSCS comparison experiment.")
    p.add_argument("--config", required=True, help="Experiment YAML config path")
    p.add_argument("--methods", nargs="+", default=None,
                   help=f"Methods to run (default: all). Available: {list_methods()}")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Random seeds (default: from config)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print plan without writing files")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)

    ds_cfg = cfg.get("dataset", {})
    exp_cfg = cfg.get("experiment", cfg)

    features_csv   = ds_cfg.get("features_csv")
    embeddings_dir = ds_cfg.get("embeddings_dir")
    dataset_name   = ds_cfg.get("name", "dataset")
    budgets        = exp_cfg.get("budgets", [cfg.get("budget")])
    seeds          = args.seeds or exp_cfg.get("seeds", [42])
    methods        = args.methods or exp_cfg.get("methods", list_methods())
    output_root    = Path(exp_cfg.get("output_dir", "outputs"))
    split          = ds_cfg.get("split", "train")
    split_col      = ds_cfg.get("split_col", "split")

    print("=" * 70)
    print(f"CSCS Experiment: {dataset_name}")
    print(f"  Methods : {methods}")
    print(f"  Budgets : {budgets}")
    print(f"  Seeds   : {seeds}")
    print(f"  Dry run : {args.dry_run}")
    print("=" * 70)

    # Load data once
    df = load_features(features_csv, split=split, split_col=split_col)
    volume_ids = df["volume_id"].tolist()
    U = df["uncertainty"].values
    T = df["typicality"].values

    embeddings = None
    if embeddings_dir:
        try:
            embeddings, found = load_embeddings_dir(embeddings_dir, volume_ids)
            if len(found) < len(volume_ids):
                idx = {v: i for i, v in enumerate(found)}
                volume_ids = [v for v in volume_ids if v in idx]
                U = np.array([U[volume_ids.index(v)] for v in volume_ids])
                T = np.array([T[volume_ids.index(v)] for v in volume_ids])
                embeddings = embeddings[[idx[v] for v in volume_ids]]
        except FileNotFoundError as e:
            print(f"  [WARN] {e} — proceeding without embeddings.")

    N = len(volume_ids)
    print(f"\n  Pool: {N} volumes loaded from {features_csv}")

    grand_results = []

    for budget in budgets:
        for seed in seeds:
            for method in methods:
                out_dir = output_root / dataset_name / "selections" / f"{method}_k{budget}_seed{seed}"
                print(f"\n  [{method.upper():12s}] K={budget}, seed={seed} → {out_dir}")

                if args.dry_run:
                    continue

                set_seed(seed)
                selector = get_selector(method)
                kwargs = {}
                if method == "cscs":
                    kwargs["s_min"] = exp_cfg.get("s_min", 3)

                selection_df, meta = selector(
                    volume_ids=volume_ids, U=U, T=T,
                    budget=budget, embeddings=embeddings, seed=seed, **kwargs
                )

                # Check leakage (if val_ids available)
                val_ids_file = ds_cfg.get("val_file")
                if val_ids_file and Path(val_ids_file).exists():
                    from cscs.features.io import normalize_volume_id
                    val_ids = set()
                    with open(val_ids_file) as f:
                        for line in f:
                            val_ids.add(normalize_volume_id(line.strip()))
                    selected = selection_df[selection_df["selected"]]["volume_id"].tolist()
                    leaked = set(selected) & val_ids
                    if leaked:
                        print(f"  [ERROR] Data leakage detected: {leaked}")
                        continue

                out_dir.mkdir(parents=True, exist_ok=True)
                selection_df.to_csv(out_dir / "selection_table.csv", index=False)
                selected = selection_df[selection_df["selected"]]["volume_id"].tolist()
                pd.DataFrame({"volume_id": selected}).to_csv(out_dir / "selected_ids.csv", index=False)
                with open(out_dir / "metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)

                row = {"method": method, "budget": budget, "seed": seed, "n_selected": len(selected)}
                row.update({k: v for k, v in meta.items() if isinstance(v, (int, float, str))})
                grand_results.append(row)
                print(f"    → {len(selected)} selected")

    if not args.dry_run and grand_results:
        summary_path = output_root / dataset_name / "grand_summary.csv"
        pd.DataFrame(grand_results).to_csv(summary_path, index=False)
        print(f"\nGrand summary → {summary_path}")

    print(f"\n{'='*70}")
    print("DONE" + (" (dry run — no files written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
