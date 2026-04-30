# DANN v1 Post-mortem

## Results

DANN v1 (dann_weight=1.0, down_scales=[2,4,8], 50 epochs, 3 runs) was a negative result:

| Condition | Baseline | DANN (best) | Δ |
|---|---|---|---|
| Native | 45.11 | 51.32 | +6.21 worse |
| 2x down | 44.35 | 51.06 | +6.71 worse |
| 4x down | 96.03 | 100.19 | +4.16 worse |
| Zoom pairs mean |HR-LR| | 166.0 | 166.2 | no change |

Training dynamics were healthy: domain loss rose as alpha reached 1.0 (discriminator losing → features becoming resolution-invariant), task loss trended down, no instability. The model mechanically did what DANN is supposed to do. But resolution invariance did not translate to resolution robustness.

## Suspected cause: train/eval domain mismatch

DANN v1 generates LR images **on-the-fly from 224x224 tensor crops** using `F.interpolate(bilinear)` down to 28–112px then back up to 224. The discriminator learns to detect bilinear interpolation artifacts on tiny normalized tensors.

Evaluation uses **pre-saved downscaled full-resolution images** (PIL bilinear on raw pixels, saved as JPEG) run through sliding-window inference. The zoom pairs are real optical captures.

These are fundamentally different degradation distributions:

| | DANN v1 training LR | Eval downscaled |
|---|---|---|
| Source | 224x224 normalized tensor crop | Full-res image (e.g. 5400x3600) |
| Downscale method | `F.interpolate` on float tensors | PIL resize on uint8 pixels, JPEG-saved |
| Intermediate size | 28–112 px | 1350–2700 px |
| Upscaled back? | Yes, back to 224x224 | No, stays at reduced size |
| Inference | Single forward pass | Sliding window over downscaled image |

The discriminator learned to distinguish tensor interpolation artifacts, not actual resolution characteristics. The feature extractor suppressed those artifact signals — which are irrelevant at eval time.

## Fix: DANN v2

Train on the **pre-saved downscaled NWPU images** (`NWPU_downscaled/2x/` and `4x/`) as the LR domain. Both HR and LR go through the same crop → resize → augment → sliding-window-shaped pipeline. The discriminator sees the same kind of resolution difference the model faces at evaluation.

Three runs planned:
- LR domain = 2x only
- LR domain = 4x only
- LR domain = mixed 2x + 4x
