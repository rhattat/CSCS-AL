"""
nnU-Net v2 training launcher.

Wraps the nnU-Net CLI commands (nnUNetv2_train, nnUNetv2_predict) as Python functions.
All paths are passed through environment variables (nnUNet_raw, nnUNet_preprocessed,
nnUNet_results) — never hardcoded.

Usage:
    from cscs.training.nnunet_runner import run_training, run_prediction
    run_training(dataset_id=42, fold=0, trainer="nnUNetTrainer")
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def _check_env() -> None:
    required = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            f"Missing nnU-Net environment variable(s): {missing}. "
            f"Set them before running training:\n"
            f"  export nnUNet_raw=/path/to/nnUNet_raw\n"
            f"  export nnUNet_preprocessed=/path/to/preprocessed\n"
            f"  export nnUNet_results=/path/to/results"
        )


def run_preprocessing(
    dataset_id: int,
    config: str = "3d_fullres",
    plans: str = "nnUNetPlans",
    num_processes: int = 4,
) -> int:
    """Run nnUNetv2_plan_and_preprocess."""
    _check_env()
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "-c", config,
        "-pl", plans,
        "--verify_dataset_integrity",
        "-np", str(num_processes),
    ]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def run_training(
    dataset_id: int,
    fold: int = 0,
    config: str = "3d_fullres",
    trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetPlans",
    extra_args: Optional[list[str]] = None,
) -> int:
    """Run nnUNetv2_train."""
    _check_env()
    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        config,
        str(fold),
        "-tr", trainer,
        "-p", plans,
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def run_prediction(
    dataset_id: int,
    input_dir: str | Path,
    output_dir: str | Path,
    fold: int = 0,
    config: str = "3d_fullres",
    trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetPlans",
) -> int:
    """Run nnUNetv2_predict."""
    _check_env()
    results_base = Path(os.environ["nnUNet_results"])
    # Infer checkpoint path
    ckpt_dir = results_base / f"Dataset{dataset_id:03d}_*" / f"{trainer}__{plans}__{config}" / f"fold_{fold}"
    cmd = [
        "nnUNetv2_predict",
        "-d", str(dataset_id),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-f", str(fold),
        "-tr", trainer,
        "-c", config,
        "-p", plans,
    ]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode
