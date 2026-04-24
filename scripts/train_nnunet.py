#!/usr/bin/env python3
"""
Train nnU-Net on a given active learning selection.

Requires nnU-Net v2 installed and environment variables set:
    export nnUNet_raw=/path/to/nnUNet_raw
    export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
    export nnUNet_results=/path/to/nnUNet_results

Usage:
    python scripts/train_nnunet.py \\
        --selected_ids outputs/diane/selections/cscs_k7_seed42/selected_ids.csv \\
        --images_dir /path/to/imagesTr/ \\
        --labels_dir /path/to/labelsTr/ \\
        --dataset_id 42 \\
        --dataset_name Dataset042_DIANE \\
        --channel_names '{"0": "T2"}' \\
        --labels '{"background": 0, "foreground": 1}' \\
        --fold 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cscs.training.dataset_builder import build_nnunet_dataset
from cscs.training.nnunet_runner import run_preprocessing, run_training

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Build nnU-Net dataset and launch training.")
    p.add_argument("--selected_ids", required=True, help="CSV with 'volume_id' column")
    p.add_argument("--images_dir", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--dataset_id", type=int, required=True)
    p.add_argument("--dataset_name", required=True, help="e.g. Dataset042_DIANE")
    p.add_argument("--channel_names", type=json.loads, default={"0": "T2"},
                   help='JSON dict e.g. \'{"0": "T2"}\'')
    p.add_argument("--labels", type=json.loads, default={"background": 0, "foreground": 1})
    p.add_argument("--output_dataset_dir", type=str, default=None,
                   help="Override output dir (default: $nnUNet_raw/{dataset_name})")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--trainer", type=str, default="nnUNetTrainer")
    p.add_argument("--config", type=str, default="3d_fullres")
    p.add_argument("--skip_preprocessing", action="store_true")
    p.add_argument("--skip_training", action="store_true")
    args = p.parse_args()

    import os
    nnunet_raw = os.environ.get("nnUNet_raw")
    if not nnunet_raw:
        print("ERROR: Set nnUNet_raw environment variable.", file=sys.stderr)
        sys.exit(1)

    # Load selected IDs
    df = pd.read_csv(args.selected_ids)
    selected_ids = df["volume_id"].astype(str).tolist()
    print(f"Building dataset with {len(selected_ids)} volumes...")

    # Output dataset directory
    out_dir = Path(args.output_dataset_dir) if args.output_dataset_dir \
        else Path(nnunet_raw) / args.dataset_name

    build_nnunet_dataset(
        selected_ids=selected_ids,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=out_dir,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        channel_names=args.channel_names,
        labels=args.labels,
    )

    if not args.skip_preprocessing:
        print("\nRunning nnU-Net preprocessing...")
        rc = run_preprocessing(args.dataset_id, config=args.config)
        if rc != 0:
            print(f"ERROR: Preprocessing failed (exit code {rc})", file=sys.stderr)
            sys.exit(rc)

    if not args.skip_training:
        print("\nLaunching nnU-Net training...")
        rc = run_training(args.dataset_id, fold=args.fold,
                          config=args.config, trainer=args.trainer)
        if rc != 0:
            print(f"ERROR: Training failed (exit code {rc})", file=sys.stderr)
            sys.exit(rc)

    print("\nDone.")


if __name__ == "__main__":
    main()
