# Reproducing Experiments

> **Note**: Datasets, SSL checkpoints, and trained models are NOT included in this repository.
> See the paper for details on data sources.

## Prerequisites

1. Clone the CSAL-3D repository (for the SSL encoder):
   ```bash
   git clone https://github.com/...  # TODO: add CSAL-3D repo URL
   export CSAL3D_PATH=/path/to/CSAL-3D
   ```

2. Obtain pretrained SSL checkpoints (one per dataset).

3. Install CSCS-AL:
   ```bash
   pip install -e ".[ssl,viz]"
   ```

4. Set nnU-Net environment variables:
   ```bash
   export nnUNet_raw=/path/to/nnUNet_raw
   export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
   export nnUNet_results=/path/to/nnUNet_results
   ```

## Step 1: Extract SSL Features

```bash
python scripts/extract_features.py \
    --config configs/datasets/spleen.yaml \
    --images_dir /path/to/imagesTr/ \
    --train_file data/splits/spleen/train.txt \
    --val_file   data/splits/spleen/val.txt \
    --ssl_checkpoint /path/to/ssl_spleen/best_model.pth \
    --output_dir outputs/spleen/features/ \
    --csal3d_path $CSAL3D_PATH \
    --device cuda:0
```

Outputs:
- `outputs/spleen/features/embeddings/*.npy`
- `outputs/spleen/features/spleen_ssl_features_train.csv`

## Step 2: Generate Selections

Run all methods × budgets × seeds:
```bash
python scripts/run_experiment.py \
    --config configs/experiments/example_cscs.yaml
```

Or run a single selection:
```bash
python scripts/generate_selection.py \
    --features_csv outputs/spleen/features/spleen_ssl_features_train.csv \
    --embeddings_dir outputs/spleen/features/embeddings/ \
    --budget 10 \
    --method cscs \
    --output_dir outputs/spleen/selections/cscs_k10_seed42/ \
    --seed 42
```

## Step 3: Train nnU-Net

```bash
python scripts/train_nnunet.py \
    --selected_ids outputs/spleen/selections/cscs_k10_seed42/selected_ids.csv \
    --images_dir /path/to/imagesTr/ \
    --labels_dir /path/to/labelsTr/ \
    --dataset_id 109 \
    --dataset_name Dataset109_Spleen_cscs_k10_seed42 \
    --channel_names '{"0": "CT"}' \
    --labels '{"background": 0, "spleen": 1}' \
    --fold 0
```

## Step 4: Aggregate Results

```bash
python scripts/summarize_results.py \
    --root outputs/Article/ \
    --outdir outputs/tables/ \
    --latex
```

## Expected Results

See the paper (Table 1, Figure 3) for expected Dice/HD95 values per dataset.

## Datasets

| Dataset | Source | N_train |
|---------|--------|---------|
| BraTS 2021 | https://www.kaggle.com/... | 387 |
| Spleen (MSD) | http://medicaldecathlon.com/ | 34 |
| FeTA | https://feta.grand-challenge.org/ | 39 |
| DIANE | Private (contact authors) | 55 |
