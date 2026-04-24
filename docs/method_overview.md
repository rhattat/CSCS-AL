# CSCS Method Overview

## Problem Setting

**Cold-start active learning** for 3D medical image segmentation:
given a fully unlabeled pool of N volumes, select B samples to annotate first,
with no prior labeled examples.

## Pipeline

```
Unlabeled pool (N volumes)
        │
        ▼
[0] SSL Feature Extraction    (requires GPU + pretrained encoder)
    ├── z_i: 768-D SwinViT embedding
    ├── U(x): uncertainty via multi-view inpainting
    └── T(x): typicality = 1/mean_k-NN_dist
        │
        ▼
[1] Dataset Characterisation
    ├── DCR = Spearman(U, T)
    │       > 0: hard ≈ atypical → active learning beneficial
    │       < 0: hard ≈ typical  → emphasize representativeness
    ├── alpha_eff = B / sqrt(N)
    └── gamma = clip(0.5 + DCR/4 · alpha_eff/(1+alpha_eff), 0.3, 0.7)
        │
        ▼
[2] K-means++ Clustering   (K = B clusters)
    Ensures spatial diversity — one sample selected per cluster
        │
        ▼
[3] Composite Score  S(x) = T_rank^(1-gamma) · U_rank^gamma
    (rank-normalized within each cluster)
        │
        ▼
[4] Selection
    Per cluster: x* = argmax S(x)
    Thin clusters (< s_min=3): skipped → gaps filled by global ranking
        │
        ▼
Output: B selected volume IDs + full ranking table + gamma, DCR, alpha_eff
```

## Key Formula

```
gamma = clip(0.5 + (DCR / 4) * (alpha_eff / (1 + alpha_eff)), 0.3, 0.7)
```

- `DCR = 0`: equal weight on typicality and uncertainty (gamma = 0.5)
- `DCR > 0`: harder samples are atypical → increase uncertainty weight (gamma > 0.5)
- `DCR < 0`: harder samples are typical → increase typicality weight (gamma < 0.5)
- Larger `alpha_eff` (higher budget): moves gamma closer to the DCR-driven extreme

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| gamma_lo  | 0.3     | Lower clip for gamma |
| gamma_hi  | 0.7     | Upper clip for gamma |
| s_min     | 3       | Minimum cluster size |
| seed      | 42      | Random seed |

## Baselines

| Method | Strategy |
|--------|----------|
| Random | Uniform random sampling |
| FPS | Farthest Point Sampling (diversity only) |
| TypiClust | K-means + most typical per cluster |
| ProbCover | Greedy max-coverage via ε-ball graph |
| CSAL-3D | Multi-kernel k-means + typical+uncertain per cluster |

## Output Format

Every selector returns a DataFrame with columns:

| Column | Type | Description |
|--------|------|-------------|
| volume_id | str | Volume identifier |
| uncertainty | float | Raw uncertainty score |
| typicality | float | Raw typicality score |
| cluster_id | int | K-means cluster assignment |
| score | float | Composite score S(x) |
| selected | bool | True if selected |
| rank | int | Global score rank (1 = best) |

Plus a metadata dict with: `gamma`, `dcr`, `alpha_eff`, `budget`, `n_pool`, `seed`.
