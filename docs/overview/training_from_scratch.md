# Training from Scratch (Raw CLIP Weights)

## Decision

All training experiments (RQ1-RQ4) start from **raw OpenAI CLIP pretrained weights**, not the CLIP-EBC authors' pretrained checkpoint. This means the VPT tokens, image decoder, projection layer, and logit scale are randomly initialized at the start of every training run.

## Why

The CLIP-EBC authors provide a checkpoint (`best_mae.pth`) that was already trained on NWPU-Crowd. If some experiments started from that checkpoint while others (Yusuf's baselines via `trainer.py`) start from raw CLIP, the comparison would be unfair — improvements might come from better initialization rather than the training strategy.

By starting all experiments from the same point, any difference in MAE/RMSE is attributable to the training strategy alone:

| RQ | Strategy | What differs |
|----|----------|-------------|
| 1 | HR-only / LR-only / Mixed | Which data the model sees |
| 2 | Downscaling algorithm study | How synthetic LR is generated |
| 3 | Consistency loss | Extra loss term penalizing HR/LR disagreement |
| 4 | DANN | Adversarial gradient forcing resolution-invariant features |

## What "from scratch" means in practice

**Frozen** (loaded from OpenAI CLIP, never updated):
- ViT-B/16 image encoder (~86M params)
- Text encoder (~37M params)

**Randomly initialized** (trained during our experiments):
- VPT tokens: 32 learnable vectors x 12 layers (~295K params)
- Image decoder: ResNet block (~3.5M params)
- Projection: Conv2d 768->512 (~393K params)
- Logit scale: 1 scalar param

## In code

- `build_model(device)` — constructs CLIP-EBC with raw CLIP weights. Used by all training entrypoints.
- `load_model(device)` — loads the CLIP-EBC authors' pretrained checkpoint. Used only by evaluation scripts that need the published baseline.

Both live in `src/models/clip_ebc.py`.
