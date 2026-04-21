# Downscaling and Label Reliability

Low-resolution (LR) training inputs are generated on-the-fly from high-resolution (HR) NWPU images.

Core idea:
- Use original image + point annotations as source of truth
- Apply synthetic degradation to image only
- Remap points by scale so labels remain geometrically consistent

Example:
- lr_image = resize(hr_image, scale)
- lr_points = hr_points * scale

Why this is useful:
- No relabeling needed for blurry images
- Keeps supervision tied to reliable HR annotation space
- Allows controlled ablations over scale and degradation strength
