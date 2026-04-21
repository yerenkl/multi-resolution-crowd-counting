# Augmentation Notes

Risk:
- A model may overfit to one synthetic resize pipeline and fail on real LR images.

Mitigations:
- Randomize downscale factor
- Randomize interpolation method
- Add blur/noise/compression style artifacts

Validation check:
- Use real zoom-pair HR/LR images to confirm robustness is not tied to one resize kernel.

Crop-match interpretation:
- Mostly a motivation for label reliability and consistency learning
- Not a separate, complex data construction pipeline
