# finetune_consistency_reg.py

Fine-tunes the pretrained CLIP-EBC (ViT-B/16) checkpoint on NWPU-Crowd using **consistency regularization**: the model is simultaneously shown a high-resolution and a low-resolution view of the *same* crop at each step, and penalized whenever it predicts different counts for the two views.

---

## Motivation: what the baseline evaluation showed

Before designing this training scheme, the pretrained CLIP-EBC model was evaluated on NWPU val at native resolution and at 2× and 4× bilinear downscale, comparing predictions against the original ground-truth annotations. The results were:

| Resolution | MAE   | RMSE   | Mean bias |
|---|---|---|---|
| native (1×) | 45.11 | 157.46 | +15.4 |
| 2× down    | 44.35 | 118.30 | −24.0 |
| 4× down    | 96.03 | 377.30 | −88.7 |

Three findings shaped this approach:

**1. The problem is 4× downscale, not 2×.**
2× barely changes MAE (44 vs 45). The model already handles moderate downscaling. 4× is where it breaks — MAE more than doubles. The LR degradation range `[min_down, max_down]` defaults to `[2.0, 4.0]` to cover the full problem range.

**2. Degradation causes systematic undercounting.**
At 4×, 77% of images are undercounted with a mean bias of −89. The model stops detecting people at low resolution rather than miscounting them. This directs the consistency loss: we need to push LR predictions *up* towards the HR anchor, not find a midpoint between them. This is why the loss is **one-way** — LR is pulled towards HR, never the reverse.

**3. The problem scales with crowd density.**
The HR–LR count gap correlates strongly with crowd size (r = 0.866 at 4×). Breaking down by bucket:

| Bucket | MAE native | MAE 2× | MAE 4× |
|---|---|---|---|
| sparse (0–50)   | 9.2  | 3.5  | 3.5  |
| medium (50–500) | 26.4 | 23.8 | 39.2 |
| dense (500+)    | 151.5| 163.2| 398.8|

Dense scenes are where the model fails hardest at low resolution. Sparse scenes are unaffected — at 4× they actually improve slightly (less background noise). Consistency pressure is most needed on dense crops.

---

## Core idea

For every training sample, produce two views of the same crop:

| View | How it's made |
|---|---|
| **HR** (high-res) | 224×224 crop from the original image |
| **LR** (low-res) | same crop, bilinearly shrunk by `down_factor ∈ [min_down, max_down]`, then bilinearly upscaled back to 224×224 |

Both views are fed through the model in **a single batched forward pass** (batch doubled in size). The loss is:

```
total_loss = supervised_loss(HR) + λ · consistency_loss(HR, LR)
```

where:
- `supervised_loss` = DACELoss on the HR branch against the ground-truth density map and point annotations
- `consistency_loss` = L1 distance between the predicted *count* (sum of density map) for HR vs LR
- `λ` = `--lambda_cons` (default 0.5)

### One-way consistency (LR → HR only)

`hr_count` is detached before computing the consistency loss:

```python
cons_loss = F.l1_loss(lr_count, hr_count.detach())
```

This makes the HR prediction a **fixed target**. Gradients from the consistency term flow only through `lr_count` — the LR branch is pulled towards HR, never the other way around. The HR branch is shaped exclusively by the supervised loss against ground truth. This design is motivated by the bias finding above: we know LR undercounts, and we want to correct that in one direction.

---

## Data pipeline (`NWPUConsistency`)

Each call to `__getitem__` returns `(hr_tensor, lr_tensor, points, density)`.

### Step 1 — Random oversized crop
```
scale   ~ Uniform(1.0, 2.0)
crop_px = min(scale × 224, image_width, image_height)
```
A crop region is sampled from the original image. Head-point annotations are filtered to this region and remapped to crop-local coordinates.

### Step 2 — HR view
The crop is resized to 224×224 with bilinear interpolation. Point coordinates are rescaled proportionally.

### Step 3 — LR view
Starting from the same 224×224 HR image:
```
down_factor ~ Uniform(min_down, max_down)   # default: [2.0, 4.0]
lr_size     = max(1, 224 / down_factor)     # e.g. ~56px at 4×
lr_img      → resize down to lr_size × lr_size   (bilinear)
lr_img      → resize back up to 224 × 224        (bilinear)
```
The result is a blurry/pixelated version of the HR crop at the same spatial canvas size. No point remapping is needed — the spatial layout is identical to HR.

### Step 4 — Shared flip
The same random horizontal flip is applied to **both** views and point coordinates are mirrored accordingly, keeping the pair geometrically consistent.

### Step 5 — Normalise
Standard ImageNet normalisation applied to both tensors.

### Step 6 — Density map
Generated from the (flipped, rescaled) HR point set using CLIP-EBC's `generate_density_map` with `sigma=None` (no Gaussian smoothing — the map is a sum of unit point masses, so `density.sum()` equals exactly the number of people in the crop). Used only to supervise the HR branch.

---

## Forward pass

```python
combined = cat([hr_batch, lr_batch], dim=0)      # shape: (2B, 3, 224, 224)
pred_class, pred_density = model(combined)

hr_class   = pred_class[:B]
hr_density = pred_density[:B]
lr_density = pred_density[B:]
```

Stacking into a single call is more efficient than two separate forward passes and avoids any per-batch normalisation inconsistencies.

---

## Loss breakdown

```python
sup_loss, loss_info = DACELoss(hr_class, hr_density, gt_density, points)

hr_count = hr_density.sum(dim=(1, 2, 3))          # scalar per image in batch
lr_count = lr_density.sum(dim=(1, 2, 3))

cons_loss = L1(lr_count, hr_count.detach())        # one-way: LR → HR only

total_loss = sup_loss + λ · cons_loss
```

---

## Training setup

| Hyperparameter | Default | Notes |
|---|---|---|
| `--epochs` | 50 | |
| `--batch_size` | 8 | 16 images per forward pass (8 HR + 8 LR) |
| `--lr` | 1e-5 | low rate — fine-tuning from a strong checkpoint |
| `--lambda_cons` | 0.5 | consistency loss weight |
| `--min_down` | 2.0 | minimum LR downscale factor |
| `--max_down` | 4.0 | maximum LR downscale factor |
| Optimiser | Adam, weight_decay=1e-4 | |
| Mixed precision | AMP (`GradScaler`) | |

**Choosing `lambda_cons`**: too large overwhelms the supervised signal and HR accuracy drops; too small means the LR branch receives little pressure. A sweep of {0.1, 0.5, 1.0} is recommended. Niote that the consistency loss operates on raw count values (not normalised), so its scale grows with crowd density — in dense scenes a single step may produce a cons_loss of several hundred.

---

## Comparison to `finetune_resolution_aug.py`

| | `finetune_resolution_aug` | `finetune_consistency_reg` |
|---|---|---|
| Resolution exposure | Each crop is *either* HR *or* LR | Each crop is *both* HR and LR simultaneously |
| Supervision | GT on all crops (whatever resolution) | GT only on HR; consistency signal on LR |
| How invariance is learned | Implicit — model must handle any resolution to minimise GT loss | Explicit — penalised directly for count disagreement between HR and LR |
| Memory per step | 1× batch | 2× batch (paired views) |
| Key assumption | Degraded crops still have meaningful GT-comparable density maps | GT is cleanest at native resolution; LR only needs to match HR |

---

## Validation

After each epoch, the first 100 NWPU val images at native resolution are evaluated with sliding-window inference (window=224, stride=224). The checkpoint with the lowest MAE on this subset is saved as `best_mae.pth`.

To measure the actual target metric (resolution consistency), run `evaluation/eval_nwpu_downscaled.py` on the saved checkpoint and compare against the baseline numbers above.

---

## Outputs

```
results/finetune_consistency_reg/
  best_mae.pth    ← state dict of the epoch with lowest val MAE (first 100 images)
  latest.pth      ← full checkpoint: weights, optimizer, epoch, MAE, RMSE,
                     supervised loss, consistency loss
```
