# Baseline Reproducibility

We were unable to reproduce the CLIP-EBC authors' reported NWPU val performance using their released pretrained checkpoint.

| Source | MAE | RMSE |
|---|---|---|
| Paper (Table III) | 36.6 | 81.7 |
| Our evaluation | 45.11 | 157.46 |

## What we verified matches

- Model config: ViT-B/16, reduction=8, truncation=4, fine granularity, average anchor points
- Bins and anchors: `[{0},{1},{2},{3},[4,inf)]`, anchors `[0, 1, 2, 3, 4.21931]` (from `reduction_8.json`)
- Sliding window: window=224, stride=224 (no overlap), matching `run.sh` and `test_nwpu.py`
- Normalization: ImageNet mean/std
- Checkpoint: authors' released `best_mae.pth`, loaded with `strict=True`

## Possible explanations

- Unreported training details or best-of-N run selection
- Data loading differences (authors may use pre-saved `.npy` files vs our `.jpg` loading)
- Subtle preprocessing differences we could not identify
