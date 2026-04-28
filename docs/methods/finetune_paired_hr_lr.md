# Method: train_finetune_paired_hr_lr.py

Fine-tunes pretrained CLIP-EBC on NWPU with hard-patch mining and paired HR/LR supervision.

## What it does
- Mine multiple crop candidates and prefer dense/far-field patches
- Keep a configurable random-patch fraction to avoid overfitting
- Create paired HR and synthetic LR views of the same crop
- Random horizontal flip
- ImageNet normalization
- Train with DACELoss on both views
- Add a small one-way HR->LR count consistency term
- Upweight hard patches with a capped smooth weighting rule

## Why use it
- Improves robustness to dense blurry regions while retaining the pretrained CLIP-EBC counting prior
- Encourages the LR branch to stay close to the better-informed HR prediction without treating HR as ground truth

## Default training setup
- Epochs: 50
- Batch size: 8
- Learning rate: 1e-5
- Random patches kept: 25%
- HR->LR consistency weight: 0.05
- Hard-patch weight cap: 4x
- Optimizer: Adam (weight decay 1e-4)
- AMP enabled

## Outputs
- results/finetune_paired_hr_lr/best_mae.pth
- results/finetune_paired_hr_lr/latest.pth

## Known limitations
- Validation is a fast subset (first 100 val images)
- No LR-specific validation inside the training loop
- No LR scheduler by default
- Hardness is inferred from local point density and crop position, not from explicit blur labels

For the previous version of this method summary, see git history for this file before the rename from `finetune_resolution_aug.md`.
