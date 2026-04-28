# Visualization Notes

## What to use

**Predicted density map overlay** — the model outputs a spatial density map (28×28 for 224px input). Upsample, apply jet colormap, alpha-blend onto the original image. This is the standard in crowd counting papers and is inherently faithful — it IS the model's output, not a post-hoc approximation.

**HR vs LR comparison strips** — show the same scene at native and degraded resolution with overlaid density heatmaps. Directly illustrates the resolution robustness story.

## What NOT to use (and why)

**GradCAM** — fundamentally broken on ViTs. The classification head operates only on [CLS], so gradients to patch tokens are zero at the last layer. A reshape hack exists (pytorch-grad-cam) but quality is poor (Chefer et al. 2021: ~41% mIoU vs ~62% for transformer-native methods). Also awkward for regression tasks — you need an ad hoc scalar target.

**Attention rollout** — class-agnostic (shows where the model looked, not what drove the prediction), averages heads blindly, and ignores non-linearities between layers. Useful as a quick sanity check, not as a faithful explanation. See Jain & Wallace 2019 ("Attention is not Explanation").

## If deeper interpretability is needed

Chefer et al. relevancy maps (CVPR 2021) are the current gold standard for ViTs — class-specific, theoretically grounded. GitHub: `hila-chefer/Transformer-Explainability`.
