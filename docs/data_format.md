# Data Format

## Features CSV

The main input to all selectors. Minimum required columns:

```
volume_id, uncertainty, typicality
```

Recommended additional columns:

```
volume_id, split, uncertainty, typicality, uncertainty_normalized, typicality_normalized
```

Example rows:
```
volume_id,split,uncertainty,typicality,uncertainty_normalized,typicality_normalized
BraTS2021_00001,train,0.000342,1245.3,0.21,0.87
BraTS2021_00002,train,0.001102,892.1,0.68,0.63
...
```

**Column descriptions:**

| Column | Required | Description |
|--------|----------|-------------|
| volume_id | Yes | Unique identifier (must match embedding .npy filename) |
| uncertainty | Yes | Raw uncertainty score (higher = more uncertain) |
| typicality | Yes | Raw typicality score (higher = more representative) |
| split | Recommended | "train" or "val" (used to filter pool) |
| uncertainty_normalized | Optional | Min-max normalized U ∈ [0,1] |
| typicality_normalized | Optional | Min-max normalized T ∈ [0,1] |

## Embeddings Directory

One `.npy` file per volume, named `{volume_id}.npy`:

```
embeddings/
    BraTS2021_00001.npy    # shape (768,) or (D,)
    BraTS2021_00002.npy
    ...
```

Multi-dimensional arrays are automatically flattened to 1-D.

## Split Files

Plain text files with one volume ID per line (no header):

```
# train.txt
BraTS2021_00001
BraTS2021_00002
...
```

IDs are normalized: `_0000`, `.nii.gz`, `.nii` suffixes are stripped automatically.

## Output Format

### selected_ids.csv
```
volume_id
BraTS2021_00042
BraTS2021_00117
...
```

### selection_table.csv
Full pool with selection metadata:
```
volume_id,uncertainty,typicality,cluster_id,score,selected,rank
BraTS2021_00001,0.000342,1245.3,3,0.71,False,12
BraTS2021_00042,0.001102,1543.2,3,0.94,True,1
...
```

### metadata.json
```json
{
  "method": "cscs",
  "budget": 10,
  "n_pool": 387,
  "dcr": 0.182,
  "dcr_pval": 0.003,
  "alpha_eff": 3.914,
  "gamma": 0.538,
  "seed": 42,
  "n_selected": 10
}
```
