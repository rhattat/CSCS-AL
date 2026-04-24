#!/usr/bin/env python3
"""
Generate an active learning selection (CSCS or any baseline).

Usage:
    python scripts/generate_selection.py --config configs/experiments/example_cscs.yaml
    python scripts/generate_selection.py \\
        --features_csv data/features/spleen_ssl_features_train.csv \\
        --budget 10 \\
        --method cscs \\
        --embeddings_dir data/features/embeddings/ \\
        --output_dir outputs/spleen/selections/cscs_k10_seed42/ \\
        --seed 42

Input CSV format (minimum required columns):
    volume_id, uncertainty, typicality

Output (in --output_dir):
    selected_ids.csv        — selected volume IDs, one per line
    selection_table.csv     — full ranking with cluster_id, score, selected columns
    metadata.json           — gamma, DCR, alpha_eff, seed, etc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from cscs.utils.config import load_yaml, load_experiment_config
from cscs.utils.seed import set_seed
from cscs.features.io import load_features, load_embeddings_dir
from cscs.selection.registry import get_selector, list_methods


def parse_args():
    p = argparse.ArgumentParser(
        description="Run CSCS or baseline selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Config-based usage
    p.add_argument("--config", type=str, default=None,
                   help="Path to experiment YAML config (overrides other args)")

    # Direct usage
    p.add_argument("--features_csv", type=str, default=None,
                   help="Path to features CSV (volume_id, uncertainty, typicality[, split])")
    p.add_argument("--budget", type=int, default=None,
                   help="Number of samples to select")
    p.add_argument("--method", type=str, default="cscs",
                   choices=list_methods(),
                   help=f"Selection method (default: cscs). Available: {list_methods()}")
    p.add_argument("--embeddings_dir", type=str, default=None,
                   help="Directory with .npy embedding files (optional, improves diversity)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory to write outputs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--s_min", type=int, default=3, help="Minimum cluster size (CSCS only)")
    p.add_argument("--split", type=str, default="train",
                   help="Split value to filter (default: train)")
    p.add_argument("--split_col", type=str, default="split",
                   help="Column name for split labels (default: split)")
    p.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Load config ---
    if args.config:
        cfg = load_experiment_config(args.config)
        features_csv  = cfg.get("features_csv") or cfg.get("dataset", {}).get("features_csv")
        budget        = cfg.get("budget") or cfg.get("method", {}).get("budget")
        method        = cfg.get("method_name", "cscs")
        embeddings_dir = cfg.get("embeddings_dir") or cfg.get("dataset", {}).get("embeddings_dir")
        output_dir    = cfg.get("output_dir")
        seed          = cfg.get("seed", 42)
        s_min         = cfg.get("s_min", 3)
        split         = cfg.get("split", "train")
        split_col     = cfg.get("split_col", "split")
    else:
        features_csv   = args.features_csv
        budget         = args.budget
        method         = args.method
        embeddings_dir = args.embeddings_dir
        output_dir     = args.output_dir
        seed           = args.seed
        s_min          = args.s_min
        split          = args.split
        split_col      = args.split_col

    if not features_csv or not budget or not output_dir:
        print("ERROR: --features_csv, --budget, and --output_dir are required "
              "(or provide --config).", file=sys.stderr)
        sys.exit(1)

    verbose = not args.quiet
    set_seed(seed)

    # --- Load data ---
    df = load_features(features_csv, split=split, split_col=split_col)
    volume_ids = df["volume_id"].tolist()
    U = df["uncertainty"].values
    T = df["typicality"].values

    if verbose:
        print(f"[generate_selection] Method={method}, budget={budget}, N={len(volume_ids)}, seed={seed}")

    # --- Load embeddings ---
    embeddings = None
    if embeddings_dir:
        try:
            emb_matrix, found_ids = load_embeddings_dir(embeddings_dir, volume_ids)
            if len(found_ids) < len(volume_ids):
                # Align df with found embeddings
                found_set = set(found_ids)
                mask = [v in found_set for v in volume_ids]
                volume_ids = [v for v, m in zip(volume_ids, mask) if m]
                U = U[[i for i, m in enumerate(mask) if m]]
                T = T[[i for i, m in enumerate(mask) if m]]
                emb_matrix = emb_matrix[[found_ids.index(v) for v in volume_ids]]
            embeddings = emb_matrix
        except FileNotFoundError as e:
            print(f"  [WARN] {e} — proceeding without embeddings.")

    # --- Fallback embeddings ---
    # Methods that need a feature matrix (all except random) fall back to
    # 2-D (U, T) space when no embeddings_dir is provided. This matches the
    # CSCS selector's own behaviour and avoids a cryptic sklearn crash.
    NEEDS_EMBEDDINGS = {"fps", "typiclust", "probcover", "csal3d"}
    if embeddings is None and method in NEEDS_EMBEDDINGS:
        import warnings
        warnings.warn(
            f"No embeddings provided for method '{method}'. "
            "Falling back to (U, T) 2-D space for clustering. "
            "Provide --embeddings_dir for better diversity.",
            UserWarning,
            stacklevel=2,
        )
        embeddings = np.column_stack([U, T])

    # --- Select ---
    selector = get_selector(method)

    kwargs = {"s_min": s_min} if method == "cscs" else {}
    selection_df, metadata = selector(
        volume_ids=volume_ids,
        U=U,
        T=T,
        budget=budget,
        embeddings=embeddings,
        seed=seed,
        **kwargs,
    )

    # --- Save ---
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    selection_df.to_csv(out / "selection_table.csv", index=False)
    selected = selection_df[selection_df["selected"]]["volume_id"].tolist()
    pd.DataFrame({"volume_id": selected}).to_csv(out / "selected_ids.csv", index=False)

    import json
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\n[generate_selection] Selected {len(selected)}/{budget} volumes")
        if "gamma" in metadata:
            print(f"  DCR={metadata['dcr']:+.4f}, alpha_eff={metadata['alpha_eff']:.3f}, "
                  f"gamma={metadata['gamma']:.3f}")
        print(f"  Outputs → {out}/")
        print(f"    selected_ids.csv ({len(selected)} volumes)")
        print(f"    selection_table.csv")
        print(f"    metadata.json")


if __name__ == "__main__":
    main()
