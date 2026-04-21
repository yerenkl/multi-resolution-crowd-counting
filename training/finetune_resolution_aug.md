# finetune_resolution_aug.py

Fine-tunes the pretrained CLIP-EBC (ViT-B/16) checkpoint on NWPU-Crowd with **resolution augmentation** — each training crop is randomly degraded to a lower resolution before being fed to the model, forcing it to generalise across image quality levels.

---

## Goal

The pretrained model was trained on native-resolution images. The hypothesis is that by exposing it to randomly downscaled versions during fine-tuning, it will become more robust to low-resolution inputs (e.g. zoomed-out or low-quality camera feeds) while retaining accuracy at native resolution.

---

## Data pipeline (`NWPUTrain`)

Each training sample goes through four sequential transforms:

### 1. Random oversized crop
A crop larger than 224×224 is taken from the original image:
```
scale   ~ Uniform(1.0, 2.0)
crop_px = min(scale × 224, image_width, image_height)
```
Annotated head points are filtered to keep only those inside the crop and their coordinates are remapped to crop-local space.

### 2. Resolution augmentation (the key step)
The crop is resized *down* by a random factor before being brought back to 224×224:
```
down_factor ~ Uniform(min_scale, max_scale)   # default: [1.0, 4.0]
lr_size     = max(224, crop_size / down_factor)
img         → resize to lr_size  (bilinear)
img         → resize to 224×224  (bilinear)
```
At `down_factor=1` the image passes through unchanged. At `down_factor=4` the crop is shrunk to ~56px then blown back up, introducing significant blur/pixelation. This simulates low-optical-zoom conditions at training time.

Point coordinates are rescaled proportionally to match the final 224×224 canvas.

### 3. Random horizontal flip
Applied with 50% probability; point x-coordinates are mirrored accordingly.

### 4. Normalise
Standard ImageNet mean/std normalisation.

A **density map** is generated from the final point set using CLIP-EBC's `generate_density_map` (adaptive sigma).

---

## Model

Starts from the released NWPU checkpoint (`best_mae.pth`). Architecture is identical to the pretrained model — no layers are added or frozen explicitly. All parameters with `requires_grad=True` are updated.

| Setting | Value |
|---|---|
| Backbone | CLIP ViT-B/16 |
| Input size | 224 × 224 |
| Reduction | 8 (density map is 28×28) |
| Prompt type | word |
| VPT tokens | 32, deep |

---

## Training setup

| Hyperparameter | Default |
|---|---|
| Epochs | 50 |
| Batch size | 8 |
| Learning rate | 1e-5 |
| Optimiser | Adam (weight decay 1e-4) |
| Loss | DACELoss (dmcount variant) |
| Mixed precision | AMP (`GradScaler`) |

The low learning rate (1e-5 vs typical 1e-4) is intentional — this is fine-tuning from a strong pretrained checkpoint, not training from scratch.

---

## Loss: `DACELoss`

Reused from CLIP-EBC. Combines:
- A **count loss** (DM-Count style) on the summed density map
- A **classification loss** on the predicted count bin

Both are computed against the generated density map and the point annotations.

---

## Validation

After each epoch, a quick evaluation runs on the **first 100 NWPU val images** at native resolution using sliding-window inference (window=224, stride=224). This is intentionally lightweight — full val (~3,109 images) would be too slow per epoch.

The model with the lowest MAE on this subset is saved as `best_mae.pth`.

---

## Outputs

```
results/finetune_resolution_aug/
  best_mae.pth    ← state dict of the epoch with lowest val MAE
  latest.pth      ← full checkpoint (weights + optimizer + metrics) from last epoch
```

---

## Limitations / known issues

- The random crop uses `scale ~ Uniform(1.0, 2.0)` (hardcoded), while the resolution downscale uses `--min_scale / --max_scale`. These are independent, which means the effective pixel-level degradation also depends on how large the crop was before downscaling.
- Validation uses only 100 images and native resolution — it does not directly measure improvement on downscaled inputs.
- No learning rate schedule is applied.
