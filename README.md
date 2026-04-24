# CSCS-AL: Cold-Start Curriculum Selection for Active Learning

**CSCS** is a dataset-aware method for *cold-start active learning* in 3D medical image
segmentation. Given a fully unlabeled pool, CSCS selects an initial labeled set L₀ by
combining SSL embeddings, typicality, uncertainty, and an adaptive composite score.

> **Paper**: "CSCS: A Dataset-Aware Curriculum for Cold-Start Active Learning in 3D Medical
> Image Segmentation"  
> Rémi HATTAT, [et al.] — Université de Lorraine  
> *[Citation placeholder — link to be added upon publication]*

---

## Method Summary

```
For a pool of N unlabeled volumes and budget B:

1. Extract SSL embeddings z_i, uncertainty U(x), typicality T(x)
2. Compute DCR = Spearman(U, T)               ← dataset characterization
3. Compute alpha_eff = B / sqrt(N)
4. Compute gamma = clip(0.5 + DCR/4 · alpha_eff/(1+alpha_eff), 0.3, 0.7)
5. K-means++ clustering into B clusters        ← diversity
6. Per cluster: x* = argmax S(x)              ← informativeness + representativeness
   where S(x) = T_rank^(1-gamma) · U_rank^gamma
```

Gamma adapts to the dataset structure (DCR) and budget (alpha_eff):
- High DCR (uncertain ≈ atypical) → γ closer to 0.7 → more uncertainty emphasis
- Low/negative DCR → γ closer to 0.3 → more typicality emphasis

---

## Installation

```bash
git clone https://github.com/your-username/cscs-al.git
cd cscs-al

# Core (selection only — no GPU required)
pip install -e .

# Full (SSL extraction + visualization)
pip install -e ".[ssl,viz]"
```

**Requirements**: Python ≥ 3.10, numpy, pandas, scipy, scikit-learn, pyyaml, matplotlib.

---

## Quick Start

### From a features CSV (no GPU needed)

```python
from cscs.selection import select_cscs

selection_df, metadata = select_cscs(
    features_csv="data/spleen_ssl_features_train.csv",
    budget=10,
    output_dir="outputs/spleen/cscs_k10/",
    embeddings_dir="data/embeddings/",   # optional but recommended
    seed=42,
)

selected_ids = selection_df[selection_df["selected"]]["volume_id"].tolist()
print(f"Selected: {selected_ids}")
print(f"gamma={metadata['gamma']:.3f}, DCR={metadata['dcr']:+.4f}")
```

### Via CLI

```bash
python scripts/generate_selection.py \
    --features_csv data/spleen_ssl_features_train.csv \
    --embeddings_dir data/embeddings/ \
    --budget 10 \
    --method cscs \
    --output_dir outputs/spleen/cscs_k10_seed42/ \
    --seed 42
```

### Run full experiment (all methods × budgets × seeds)

```bash
python scripts/run_experiment.py \
    --config configs/experiments/example_cscs.yaml
```

---

## Input CSV Format

Minimum required columns:

```
volume_id, uncertainty, typicality
```

Recommended:
```
volume_id, split, uncertainty, typicality, uncertainty_normalized, typicality_normalized
```

See [docs/data_format.md](docs/data_format.md) for full specification.

---

## Output Format

Each selection run produces:

| File | Description |
|------|-------------|
| `selected_ids.csv` | Selected volume IDs (one per line) |
| `selection_table.csv` | Full pool ranked by score, with cluster_id and selected column |
| `metadata.json` | gamma, DCR, alpha_eff, seed, budget, n_pool |

---

## Available Methods

| Method | Description |
|--------|-------------|
| `cscs` | **CSCS** (this paper) — adaptive gamma composite score |
| `random` | Uniform random sampling |
| `fps` | Farthest Point Sampling (diversity) |
| `typiclust` | K-means + most typical per cluster |
| `probcover` | Greedy max-coverage (ε-ball graph) |
| `csal3d` | CSAL-3D uncertainty-weighted clustering |

---

## Datasets

> **Datasets, SSL checkpoints, and trained models are NOT included.**
> See [docs/reproduce_experiments.md](docs/reproduce_experiments.md) for setup instructions.

| Dataset | Task | N_train | Budgets |
|---------|------|---------|---------|
| BraTS 2021 | Brain tumor segmentation | 387 | 77, 116, 155 |
| Spleen (MSD) | Spleen segmentation | 34 | 7, 10, 14 |
| FeTA | Fetal brain MRI | 39 | 8, 12, 16 |
| DIANE | Fetal DWI (private) | 55 | 5, 7, 9 |

---

## Repository Structure

```
CSCS-AL/
├── cscs/                    ← installable Python package
│   ├── selection/           ← core method (CSCS + baselines)
│   │   ├── cscs.py          ← select_cscs()
│   │   ├── gamma_scheduler.py
│   │   ├── scoring.py
│   │   ├── clustering.py
│   │   ├── baselines.py
│   │   └── registry.py
│   ├── features/            ← typicality, normalization, I/O
│   ├── ssl/                 ← embedding + uncertainty extraction (optional)
│   ├── evaluation/          ← results aggregation
│   ├── visualization/       ← figures
│   └── utils/               ← config, logging, seed, paths
├── scripts/                 ← CLI entry points
├── configs/                 ← YAML configs (datasets, methods, experiments)
├── tests/                   ← unit and integration tests
├── examples/                ← minimal runnable example
└── docs/                    ← method overview, data format, reproduce guide
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Citation

```bibtex
@article{hattat2026cscs,
  title   = {CSCS: A Dataset-Aware Curriculum for Cold-Start Active Learning
             in 3D Medical Image Segmentation},
  author  = {Hattat, Rémi and others},
  journal = {[To be announced]},
  year    = {2026},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*This repository contains the method implementation only. It does not include
medical imaging datasets, SSL checkpoints, or nnU-Net training results.*
