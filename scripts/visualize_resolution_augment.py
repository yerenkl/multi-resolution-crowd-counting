"""Visualize ResolutionAugment at each downscale factor for a single image."""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import random
import matplotlib.pyplot as plt
from PIL import Image

from datasets.transforms import ResolutionAugment
from settings import settings

IMAGES = [settings.nwpu_dir / "images" / f"{i:04d}.jpg" for i in range(1, 5)]
OUTPUT_PATH = Path("results/resolution_augment_preview.png")
SCALES = (1, 1.5, 2, 4, 8)

random.seed(settings.RANDOM_SEED)

fig, axes = plt.subplots(len(IMAGES), len(SCALES), figsize=(4 * len(SCALES), 4 * len(IMAGES)))

for row, image_path in enumerate(IMAGES):
    img = Image.open(image_path).convert("RGB")
    for col, scale in enumerate(SCALES):
        aug = ResolutionAugment(down_scales=(scale,), output_size=224, pre_blur=True)
        out, _ = aug(img, None)
        ax = axes[row][col]
        ax.imshow(out)
        if row == 0:
            ax.set_title(f"{scale}x downscale" if scale > 1 else "native (1x)")
        ax.axis("off")

fig.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150)
print(f"Saved to {OUTPUT_PATH}")
