"""
SSL embedding extraction from a SwinViT encoder.

This module wraps the CSAL-3D SSL encoder to extract:
    - Per-volume embeddings z_i (Global Average Pool on the 4th SwinViT stage → 768-D)
    - Per-volume uncertainty U(x) via multi-view inpainting variance

IMPORTANT: This module requires:
    - PyTorch (torch)
    - nibabel
    - The CSAL-3D repository on PYTHONPATH (for models.NetworkLoader and utils.view_ops)
    - A pretrained SSL checkpoint (.pth)

It is intentionally separated from the core selection module so that CSCS can run
with precomputed features (features CSV + optional .npy embeddings) without any
deep learning dependency.

Usage (standalone):
    python scripts/extract_features.py --config configs/experiments/example_cscs.yaml

Output:
    {output_dir}/embeddings/{volume_id}.npy   (one file per volume)
    {output_dir}/{dataset}_ssl_features_train.csv
    {output_dir}/{dataset}_ssl_features_all.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Volume loading
# ---------------------------------------------------------------------------

def load_volume(
    nii_path: str | Path,
    channel: Optional[int] = None,
    crop_size: int = 96,
):
    """
    Load a .nii.gz volume and return:
        t_full : (1, 1, H', W', D') padded to multiples of 16  [full-volume embedding]
        t_crop : (1, 1, 96, 96, 96) center-cropped               [uncertainty + crop embedding]

    Preprocessing:
        - Channel extraction (if 4-D, e.g. BraTS multi-modal)
        - Z-score normalization (clip 1–99 percentile)
        - Padding to multiples of 16

    Args:
        nii_path: Path to the .nii.gz file.
        channel:  Index of the channel to extract for 4-D volumes (None = use as-is).
        crop_size: Size of the cubic center crop (default 96).

    Returns:
        (t_full, t_crop) as torch.Tensor or None if loading fails.
    """
    try:
        import nibabel as nib
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise ImportError(
            "nibabel and torch are required for SSL feature extraction. "
            "Install them or use precomputed features."
        ) from e

    nii_path = Path(nii_path)
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)

    if data.ndim == 4:
        if channel is None:
            raise ValueError(
                f"4-D volume detected but channel=None. "
                f"Specify which channel to extract (e.g. channel=2 for BraTS T2)."
            )
        vol = data[..., channel]
    elif data.ndim == 3:
        vol = data
    else:
        raise ValueError(f"Unexpected volume shape: {data.shape}")

    # Z-score with percentile clipping
    p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99)
    vol = np.clip(vol, p1, p99)
    std = vol.std()
    vol = (vol - vol.mean()) / (std if std > 1e-8 else 1.0)

    t = torch.tensor(vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)

    # Pad to multiples of 16
    s = t.shape[-3:]
    pv = [(16 - x % 16) % 16 for x in s]
    t_full = F.pad(t, (0, pv[2], 0, pv[1], 0, pv[0]))

    # Center crop
    pad2 = [max(0, crop_size - x) for x in s]
    tp = F.pad(t, (0, pad2[2], 0, pad2[1], 0, pad2[0])) if any(p > 0 for p in pad2) else t
    sh = tp.shape[-3:]
    starts = [(sh[i] - crop_size) // 2 for i in range(3)]
    t_crop = tp[
        :, :,
        starts[0]: starts[0] + crop_size,
        starts[1]: starts[1] + crop_size,
        starts[2]: starts[2] + crop_size,
    ]
    return t_full, t_crop


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embedding(model, img_tensor, device) -> np.ndarray:
    """
    Global Average Pool on the 4th SwinViT hidden layer → 768-D embedding.

    Args:
        model:      The CSAL-3D SSL model (SwinViT encoder).
        img_tensor: (1, 1, *) torch.Tensor.
        device:     torch.device.

    Returns:
        embedding: 1-D numpy array of shape (768,).
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        x = img_tensor.to(device)
        h = model.swinViT(x.contiguous(), normalize=False)
        feat = h[4]  # 4th stage output
        feat = F.adaptive_avg_pool3d(feat, 1).squeeze().cpu().numpy()
    return feat


# ---------------------------------------------------------------------------
# Uncertainty computation
# ---------------------------------------------------------------------------

def compute_uncertainty(
    model,
    img_tensor,
    device,
    mask_ratio: float = 0.45,
    window_size: int = 16,
) -> float:
    """
    Multi-view inpainting uncertainty.

    Strategy: apply 3 orthogonal rotations × 3 random masked views → 9 reconstructions.
    Uncertainty = variance across reconstructions, averaged over all voxels.

    This reproduces the CSAL-3D uncertainty computation exactly.
    Requires the CSAL-3D utils (view_ops, view_transforms, ops) on PYTHONPATH.

    Args:
        model:       CSAL-3D SSL model.
        img_tensor:  (1, 1, 96, 96, 96) center-cropped tensor.
        device:      torch.device.
        mask_ratio:  Fraction of patches masked (default 0.45).
        window_size: Patch window size in voxels (default 16).

    Returns:
        uncertainty: Scalar float.
    """
    import random
    import torch
    from torch.cuda.amp import autocast

    try:
        from utils import view_ops, view_transforms
        from utils.ops import mask_rand_patch
    except ImportError:
        raise ImportError(
            "CSAL-3D utils not found. Add the CSAL-3D repository to PYTHONPATH "
            "before running SSL feature extraction."
        )

    img = img_tensor.to(device)
    isz = list(img.shape[-3:])
    wsz = tuple(window_size for _ in range(3))

    x0, rot0 = view_ops.rot_rand_0(img)
    x1, rot1 = view_ops.rot_rand_1(img)
    x2, rot2 = view_ops.rot_rand_2(img)

    x0v0, _ = mask_rand_patch(wsz, isz, mask_ratio, x0)
    x1v0, _ = mask_rand_patch(wsz, isz, mask_ratio, x1)
    x2v0, _ = mask_rand_patch(wsz, isz, mask_ratio, x2)

    pc = set(view_transforms.permutation_transforms.keys()) - {0}
    perm = [random.choice(list(pc)) for _ in range(6)]
    x0v1, x0v2, x1v1, x1v2, x2v1, x2v2 = [
        view_transforms.permutation_inverse_transforms[vn](val)
        for vn, val in zip(perm, [x0v0, x0v0, x1v0, x1v0, x2v0, x2v0])
    ]

    def e5(t):
        return t.unsqueeze(0) if t.dim() == 4 else t

    def ar(x, s, d):
        x = e5(x)
        return view_transforms.rotation_transforms[d](
            view_transforms.rotation_inverse_transforms[s](x)
        ).contiguous()

    model.eval()
    with torch.no_grad():
        with autocast(enabled=False):
            _, _, r00 = model(x0v0)
            _, _, r01 = model(x0v1)
            _, _, r02 = model(x0v2)
            _, _, r10 = model(x1v0)
            _, _, r11 = model(x1v1)
            _, _, r12 = model(x1v2)
            _, _, r20 = model(x2v0)
            _, _, r21 = model(x2v1)
            _, _, r22 = model(x2v2)

        for rl, rs in [([r10, r11, r12], rot1), ([r20, r21, r22], rot2)]:
            for i in range(3):
                rl[i] = torch.stack(
                    [ar(v, s.item(), d.item()) for v, s, d in zip(rl[i], rs, rot0)]
                ).squeeze(0)

        tensors = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=0)
        return torch.var(tensors, dim=0).mean().cpu().item()


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------

def extract_ssl_features(
    image_paths: list[Path],
    volume_ids: list[str],
    model,
    device,
    output_dir: Path,
    channel: Optional[int] = None,
    crop_size: int = 96,
    mask_ratio: float = 0.45,
    k_neighbors: int = 20,
    split_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Extract SSL embeddings and uncertainty for a list of volumes.

    Args:
        image_paths:  List of .nii.gz paths (aligned with volume_ids).
        volume_ids:   List of volume IDs (strings).
        model:        Loaded CSAL-3D SSL model.
        device:       torch.device.
        output_dir:   Directory to save embeddings and CSV.
        channel:      Channel index for multi-modal volumes (None = single channel).
        crop_size:    Center crop size (default 96).
        mask_ratio:   Inpainting mask ratio (default 0.45).
        k_neighbors:  K for typicality k-NN (default 20).
        split_map:    Dict mapping volume_id → "train"/"val". If None, split col is omitted.

    Returns:
        DataFrame with columns: volume_id, uncertainty, typicality,
        uncertainty_normalized, typicality_normalized[, split]
    """
    from tqdm import tqdm
    from ..features.typicality import compute_typicality
    from ..features.normalization import minmax_normalize

    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    results, all_embeddings = [], []
    missing = []

    for vid, nii_path in tqdm(zip(volume_ids, image_paths), total=len(volume_ids)):
        if not nii_path.exists():
            warnings.warn(f"File not found: {nii_path}")
            missing.append(vid)
            continue
        try:
            t_full, t_crop = load_volume(nii_path, channel=channel, crop_size=crop_size)
            emb = extract_embedding(model, t_crop, device)
            np.save(emb_dir / f"{vid}.npy", emb)
            all_embeddings.append(emb)

            unc = compute_uncertainty(model, t_crop, device, mask_ratio=mask_ratio)
            row = {"volume_id": vid, "uncertainty": unc}
            if split_map:
                row["split"] = split_map.get(vid, "unknown")
            results.append(row)
        except Exception as e:
            warnings.warn(f"Error on {vid}: {e}")
            missing.append(vid)

    if missing:
        warnings.warn(f"{len(missing)} volumes skipped: {missing[:5]}")

    all_embeddings = np.array(all_embeddings)
    typicality = compute_typicality(all_embeddings, k=k_neighbors)

    df = pd.DataFrame(results)
    df["typicality"] = typicality
    df["uncertainty_normalized"] = minmax_normalize(df["uncertainty"].values)
    df["typicality_normalized"] = minmax_normalize(df["typicality"].values)

    return df
