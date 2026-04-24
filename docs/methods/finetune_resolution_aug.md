# Method: finetune_resolution_aug.py

Fine-tunes pretrained CLIP-EBC on NWPU with resolution augmentation.

## What it does
- Random crop
- Random downscale-upscale degradation
- Random horizontal flip
- ImageNet normalization
- Train with DACELoss on generated density maps

## Why use it
- Improves robustness to lower-resolution inputs while retaining native-resolution performance

## Default training setup
- Epochs: 50
- Batch size: 8
- Learning rate: 1e-5
- Optimizer: Adam (weight decay 1e-4)
- AMP enabled

## Outputs
- results/finetune_resolution_aug/best_mae.pth
- results/finetune_resolution_aug/latest.pth

## Known limitations
- Validation is a fast subset (first 100 val images)
- No LR-specific validation inside the training loop
- No LR scheduler by default

For the full deep-dive version, see git history of the previous docs/finetune_resolution_aug.md.
