#!/usr/bin/env python3
"""
Minimal CSCS selection example using precomputed features.

Runs entirely from a small synthetic CSV — no GPU, no medical images required.
Demonstrates the core CSCS API.

Usage:
    python examples/minimal_selection_example.py
    python examples/minimal_selection_example.py --csv examples/toy_features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from cscs.selection import select_cscs


def generate_toy_csv(path: Path, N: int = 40, seed: int = 42) -> Path:
    """Generate a small synthetic features CSV for demonstration."""
    rng = np.random.RandomState(seed)
    volume_ids = [f"volume_{i:04d}" for i in range(N)]
    uncertainty = rng.rand(N)
    typicality = 1.0 - uncertainty + rng.randn(N) * 0.2  # mild anti-correlation
    typicality = np.clip(typicality, 0, 1)

    df = pd.DataFrame({
        "volume_id":              volume_ids,
        "split":                  ["train"] * N,
        "uncertainty":            uncertainty,
        "typicality":             typicality,
        "uncertainty_normalized": (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()),
        "typicality_normalized":  (typicality  - typicality.min())  / (typicality.max()  - typicality.min()),
    })
    df.to_csv(path, index=False)
    print(f"Toy CSV saved → {path}  ({N} volumes)")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to features CSV (generated if not provided)")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Prepare CSV ---
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = Path(__file__).parent / "toy_features.csv"
        generate_toy_csv(csv_path)

    # --- Run CSCS ---
    print(f"\nRunning CSCS (budget={args.budget}, seed={args.seed})...")
    import tempfile
    with tempfile.TemporaryDirectory() as outdir:
        selection_df, metadata = select_cscs(
            features_csv=csv_path,
            budget=args.budget,
            output_dir=outdir,
            embeddings_dir=None,   # no embeddings → 2-D fallback
            seed=args.seed,
            verbose=True,
        )

    # --- Print results ---
    selected = selection_df[selection_df["selected"]]
    print(f"\n{'='*50}")
    print(f"CSCS Selection Results")
    print(f"{'='*50}")
    print(f"N pool     : {len(selection_df)}")
    print(f"Budget     : {args.budget}")
    print(f"DCR        : {metadata['dcr']:+.4f}")
    print(f"alpha_eff  : {metadata['alpha_eff']:.3f}")
    print(f"gamma      : {metadata['gamma']:.3f}")
    print(f"\nSelected volumes (ranked by score):")
    print(selected[["volume_id", "uncertainty", "typicality", "score", "cluster_id"]].to_string(index=False))
    print(f"\n  Avg uncertainty of selected: {selected['uncertainty'].mean():.3f}")
    print(f"  Avg typicality  of selected: {selected['typicality'].mean():.3f}")


if __name__ == "__main__":
    main()
