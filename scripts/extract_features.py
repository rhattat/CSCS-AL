#!/usr/bin/env python3
"""
Extract SSL embeddings and compute typicality/uncertainty for a dataset.

Requires:
    - A pretrained SSL checkpoint (SwinViT, CSAL-3D format)
    - CSAL-3D repository on PYTHONPATH (for view_ops, view_transforms, NetworkLoader)
    - PyTorch with CUDA (recommended)
    - nibabel

Usage:
    python scripts/extract_features.py \\
        --config configs/datasets/brats.yaml \\
        --ssl_checkpoint /path/to/ssl_brats/best_model.pth \\
        --images_dir /path/to/imagesTr/ \\
        --output_dir outputs/brats/features/ \\
        --device cuda:0

Output:
    {output_dir}/embeddings/{volume_id}.npy
    {output_dir}/{dataset}_ssl_features_all.csv
    {output_dir}/{dataset}_ssl_features_train.csv
    {output_dir}/embedding_space.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cscs.utils.config import load_yaml
from cscs.utils.seed import set_seed
from cscs.features.typicality import compute_typicality
from cscs.features.normalization import minmax_normalize
from cscs.features.io import normalize_volume_id


def load_split_file(path: str | Path, suffixes=("_0000", ".nii.gz", ".nii")) -> list[str]:
    """Load a train.txt or val.txt file."""
    p = Path(path)
    if not p.exists():
        return []
    ids = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(normalize_volume_id(line, suffixes))
    return ids


def main():
    parser = argparse.ArgumentParser(description="Extract SSL features for CSCS.")
    parser.add_argument("--config", type=str, required=False,
                        help="Path to dataset YAML config")
    parser.add_argument("--dataset_name", type=str, default="dataset")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory with .nii.gz image files")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to train.txt (volume IDs, one per line)")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Path to val.txt (optional)")
    parser.add_argument("--ssl_checkpoint", type=str, required=True,
                        help="Path to pretrained SSL .pth checkpoint")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--channel", type=int, default=None,
                        help="Channel index for 4-D volumes (e.g. 2 for BraTS T2)")
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--mask_ratio", type=float, default=0.45)
    parser.add_argument("--knn", type=int, default=20,
                        help="Number of nearest neighbors for typicality")
    parser.add_argument("--csal3d_path", type=str, default=None,
                        help="Path to CSAL-3D repo (added to PYTHONPATH)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # --- Add CSAL-3D to path ---
    if args.csal3d_path:
        sys.path.insert(0, args.csal3d_path)

    try:
        import torch
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        from scipy.stats import spearmanr
    except ImportError as e:
        print(f"ERROR: {e}. Install required packages: torch nibabel tqdm scipy", file=sys.stderr)
        sys.exit(1)

    from cscs.ssl.extract_embeddings import load_volume, extract_embedding, compute_uncertainty

    # --- Load model ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        from models.NetworkLoader import NetworkLoader
    except ImportError:
        print("ERROR: CSAL-3D NetworkLoader not found. Set --csal3d_path.", file=sys.stderr)
        sys.exit(1)

    ckpt_parent = Path(args.ssl_checkpoint).parent
    opt = argparse.Namespace(
        nbase=8, num_pool=5, input_nc=1, num_classes=1,
        dropout=0.0, dropout_path_rate=0.2,
        depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
        embedding_dim=48, swin_patch_size=(4, 4, 4), swin_window_size=(4, 4, 8, 4),
        crop_size=[96, 96, 96], do_ds=False, init="kaiming",
        multi_gpu=False, local_rank=0,
        checkpoints_dir=str(ckpt_parent.parent), name=ckpt_parent.name,
        expr_dir=str(ckpt_parent),
        epoch="best_model.pth", plan="", ft_id=-1,
        num_labeled=1e9, resolution="1mm", nid=14, load_ckpt=True,
    )
    model = NetworkLoader(opt).load()
    ckpt = torch.load(args.ssl_checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("net", ckpt.get("state_dict", ckpt))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    print("SSL model loaded.")

    # --- Load splits ---
    suffixes = ("_0000", ".nii.gz", ".nii")
    train_ids = load_split_file(args.train_file, suffixes)
    val_ids = load_split_file(args.val_file, suffixes) if args.val_file else []
    all_ids = train_ids + val_ids
    split_map = {v: "train" for v in train_ids}
    split_map.update({v: "val" for v in val_ids})
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Total: {len(all_ids)}")

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    results, all_embeddings, missing = [], [], []

    print("Extracting features...")
    for vid in tqdm(all_ids):
        nii_path = images_dir / f"{vid}.nii.gz"
        if not nii_path.exists():
            missing.append(vid)
            continue
        try:
            t_full, t_crop = load_volume(nii_path, channel=args.channel, crop_size=args.crop_size)
            emb = extract_embedding(model, t_crop, device)
            np.save(emb_dir / f"{vid}.npy", emb)
            all_embeddings.append(emb)
            unc = compute_uncertainty(model, t_crop, device, mask_ratio=args.mask_ratio)
            results.append({"volume_id": vid, "uncertainty": unc,
                             "split": split_map.get(vid, "unknown")})
        except Exception as e:
            print(f"\nError on {vid}: {e}")
            missing.append(vid)

    if missing:
        print(f"\n{len(missing)} volumes skipped: {missing[:5]}")

    all_embeddings = np.array(all_embeddings)
    print(f"Embeddings shape: {all_embeddings.shape}")

    # --- Typicality + DCR ---
    typicality = compute_typicality(all_embeddings, k=args.knn)
    df = pd.DataFrame(results)
    df["typicality"] = typicality
    df["uncertainty_normalized"] = minmax_normalize(df["uncertainty"].values)
    df["typicality_normalized"] = minmax_normalize(df["typicality"].values)

    from scipy.stats import spearmanr
    dcr, pval = spearmanr(df["uncertainty"].values, df["typicality"].values)
    print(f"\nDCR = {dcr:+.4f}  (p={pval:.4f})")

    # --- Save ---
    ds_name = args.dataset_name
    df_all = df.sort_values("volume_id")
    df_train = df_all[df_all["split"] == "train"]
    df_all.to_csv(output_dir / f"{ds_name}_ssl_features_all.csv", index=False)
    df_train.to_csv(output_dir / f"{ds_name}_ssl_features_train.csv", index=False)

    # --- UMAP ---
    try:
        from cscs.visualization.plot_embeddings import plot_embedding_space
        plot_embedding_space(
            all_embeddings,
            [r["volume_id"] for r in results],
            df["uncertainty_normalized"].values,
            df["typicality_normalized"].values,
            dcr, output_dir,
            dataset_name=ds_name,
        )
    except Exception as e:
        print(f"Embedding plot skipped: {e}")

    print(f"\nDone. Features saved to {output_dir}/")


if __name__ == "__main__":
    main()
