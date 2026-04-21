"""
Downscale NWPU-Crowd train and val images at 2x and 4x.

Uses src/image_ops/downsample.py for the resize logic (bilinear, no blur, no noise).

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python entrypoints/downscale_nwpu.py

Output structure:
    settings.NWPU_DOWNSCALED_DIR/
      2x/images/{image_id}.jpg
      4x/images/{image_id}.jpg
"""

import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.settings import settings
from src.image_ops.downsample import resize

SCALES = [2, 4]
SPLITS = ["train", "val"]


def main():
    nwpu_root = settings.nwpu_dir
    out_root = settings.NWPU_DOWNSCALED_DIR

    for scale in SCALES:
        out_dir = out_root / f"{scale}x" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        image_ids = []
        for split in SPLITS:
            with open(nwpu_root / f"{split}.txt") as f:
                image_ids.extend(line.strip().split()[0] for line in f if line.strip())

        print(f"\n{'='*50}")
        print(f"  Downscaling {len(image_ids)} images at {scale}x")
        print(f"  Output: {out_dir}")
        print(f"{'='*50}")

        for image_id in tqdm(image_ids, desc=f"{scale}x"):
            src_path = nwpu_root / "images" / f"{image_id}.jpg"
            dst_path = out_dir / f"{image_id}.jpg"
            if dst_path.exists():
                continue
            img = Image.open(src_path).convert("RGB")
            img_down = resize(img, scale=1 / scale, method="bilinear")
            img_down.save(dst_path, quality=95)

    print("\nDone.")


if __name__ == "__main__":
    main()
