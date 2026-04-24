"""
Build nnU-Net-compatible dataset.json from a list of selected volume IDs.

This helper generates the minimal JSON required by nnU-Net v2 to recognize
a custom labeled training set.

Usage:
    python scripts/train_nnunet.py --config configs/experiments/example_cscs.yaml
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional


def build_nnunet_dataset(
    selected_ids: list[str],
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    dataset_id: int,
    dataset_name: str,
    channel_names: dict[str, str],
    labels: dict[str, int],
    file_suffix: str = ".nii.gz",
    copy_files: bool = False,
) -> Path:
    """
    Build an nnU-Net v2 dataset folder from selected volume IDs.

    Creates:
        {output_dir}/
            dataset.json
            imagesTr/   (symlinks or copies)
            labelsTr/   (symlinks or copies)

    Args:
        selected_ids:   List of selected volume ID strings.
        images_dir:     Directory with all raw image files.
        labels_dir:     Directory with all label files.
        output_dir:     Output dataset directory (nnUNet_raw/{dataset_name}).
        dataset_id:     nnU-Net dataset ID integer.
        dataset_name:   Dataset name (e.g. "Dataset042_DIANE").
        channel_names:  Dict like {"0": "T2"} for dataset.json.
        labels:         Dict like {"background": 0, "foreground": 1}.
        file_suffix:    Image file extension (default ".nii.gz").
        copy_files:     If True, copy files; otherwise create symlinks.

    Returns:
        Path to the created dataset directory.
    """
    output_dir = Path(output_dir)
    images_out = output_dir / "imagesTr"
    labels_out = output_dir / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    training_cases = []
    missing = []

    for vid in selected_ids:
        img_src = Path(images_dir) / f"{vid}{file_suffix}"
        lbl_src = Path(labels_dir) / f"{vid}{file_suffix}"

        if not img_src.exists():
            missing.append((vid, "image"))
            continue
        if not lbl_src.exists():
            missing.append((vid, "label"))
            continue

        img_dst = images_out / f"{vid}{file_suffix}"
        lbl_dst = labels_out / f"{vid}{file_suffix}"

        if copy_files:
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)
        else:
            if not img_dst.exists():
                img_dst.symlink_to(img_src.resolve())
            if not lbl_dst.exists():
                lbl_dst.symlink_to(lbl_src.resolve())

        training_cases.append({"image": f"./imagesTr/{vid}{file_suffix}",
                                "label": f"./labelsTr/{vid}{file_suffix}"})

    if missing:
        print(f"  [WARN] {len(missing)} file(s) not found:")
        for vid, kind in missing[:10]:
            print(f"         {kind}: {vid}")

    # Write dataset.json
    dataset_json = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": len(training_cases),
        "file_ending": file_suffix,
        "name": dataset_name,
        "dataset_id": dataset_id,
        "training": training_cases,
    }
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"  nnU-Net dataset: {len(training_cases)} volumes → {output_dir}")
    return output_dir
