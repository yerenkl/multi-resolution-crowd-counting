# Model Initialization

## Decision

RQ1 starts from **raw OpenAI CLIP pretrained weights** (randomly initialized trainable layers). RQ2-RQ4 start from the **CLIP-EBC authors' pretrained checkpoint** (`best_mae.pth`).

## Why

**RQ1 trains from scratch** because it asks what effect training data composition has (HR-only / LR-only / mixed). Starting from a blank slate means any difference in MAE/RMSE is attributable to the data alone.

**RQ2-RQ4 start from pretrained** because they ask whether a specific technique can improve an already-good model:

| RQ | Strategy | Question |
|----|----------|----------|
| 2 | Downscaling algorithm study | Which degradation hurts most? Does the model overfit to a specific resize kernel? |
| 3 | Consistency loss | Does penalizing HR/LR disagreement improve LR robustness? |
| 4 | DANN | Does adversarial training force resolution-invariant features? |

The pretrained checkpoint is the practical starting point — it already counts well on HR, and each technique's job is to add LR robustness on top. The baseline comparison for RQ2-RQ4 is the pretrained model's own performance before vs. after fine-tuning.

## What "from scratch" means in practice

**Frozen** (loaded from OpenAI CLIP, never updated):
- ViT-B/16 image encoder: 151 tensors, 85,799,424 params
- Text encoder: 149 tensors, 63,428,096 params
- Total frozen: 149,227,520 params

**Randomly initialized** (trained during our experiments):
- VPT tokens: 32 learnable vectors x 12 layers = 294,912 params
- Image decoder: two 3x3 conv + batchnorm = 10,620,480 params
- Projection: Conv2d 768->512 + bias = 393,728 params
- Logit scale: 1 param
- Total trainable: 11,308,545 params (21 tensors)

**DANN domain classifier** (added on top for RQ4):
- GAP + 3-layer MLP: 6 tensors, 262,913 params

## In code

- `build_model(device)` — constructs CLIP-EBC with raw CLIP weights. Used by RQ1 training.
- `load_model(device)` — loads the CLIP-EBC authors' pretrained checkpoint. Used by RQ2-RQ4 training and evaluation scripts.

Both live in `src/models/clip_ebc.py`.
