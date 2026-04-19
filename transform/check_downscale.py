"""
Visual comparison of native vs 2x vs 4x downscaled NWPU images.

Picks 4 random val images and saves a side-by-side comparison.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python transform/check_downscale.py
"""

import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

NWPU_ROOT = Path("/dtu/blackhole/02/137570/MultiRes/NWPU_crowd")
DOWNSCALED_ROOT = Path("/dtu/blackhole/0a/224426/NWPU_downscaled")
OUT_PATH = Path("transform/downscale_check.png")

# Pick 4 random val images
with open(NWPU_ROOT / "val.txt") as f:
    image_ids = [line.strip().split()[0] for line in f if line.strip()]

random.seed(42)
sample_ids = random.sample(image_ids, 4)

fig, axes = plt.subplots(4, 3, figsize=(15, 20), dpi=100)
fig.suptitle("Native vs 2x vs 4x Downscale", fontsize=16, y=0.98)

for row, image_id in enumerate(sample_ids):
    native = Image.open(NWPU_ROOT / "images" / f"{image_id}.jpg").convert("RGB")
    down2x = Image.open(DOWNSCALED_ROOT / "2x" / "images" / f"{image_id}.jpg").convert("RGB")
    down4x = Image.open(DOWNSCALED_ROOT / "4x" / "images" / f"{image_id}.jpg").convert("RGB")

    for col, (img, label) in enumerate([
        (native, f"Native ({native.size[0]}x{native.size[1]})"),
        (down2x, f"2x ({down2x.size[0]}x{down2x.size[1]})"),
        (down4x, f"4x ({down4x.size[0]}x{down4x.size[1]})"),
    ]):
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"ID {image_id} — {label}", fontsize=10)
        axes[row, col].axis("off")

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, bbox_inches="tight")
plt.close()
print(f"Saved to {OUT_PATH}")
