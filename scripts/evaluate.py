#!/usr/bin/env python3
"""
Run nnU-Net prediction and compute Dice/HD95 on the validation set.

Usage:
    python scripts/evaluate.py \\
        --dataset_id 42 \\
        --images_val /path/to/imagesTs/ \\
        --labels_val /path/to/labelsTs/ \\
        --output_dir outputs/diane/predictions/cscs_k7_seed42/ \\
        --fold 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cscs.training.nnunet_runner import run_prediction


def main():
    p = argparse.ArgumentParser(description="Run nnU-Net prediction.")
    p.add_argument("--dataset_id", type=int, required=True)
    p.add_argument("--images_val", required=True, help="Directory with validation images")
    p.add_argument("--output_dir", required=True, help="Prediction output directory")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--trainer", type=str, default="nnUNetTrainer")
    p.add_argument("--config", type=str, default="3d_fullres")
    args = p.parse_args()

    rc = run_prediction(
        dataset_id=args.dataset_id,
        input_dir=args.images_val,
        output_dir=args.output_dir,
        fold=args.fold,
        config=args.config,
        trainer=args.trainer,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
