"""
Downscale NWPU-Crowd train and val images at 2x and 4x.

Uses Jan's downsample.py for the resize logic (bilinear, no blur, no noise).

Usage:
    cd ~/project/multi-resolution-crowd-counting
    python downscale_nwpu.py

Output structure:
    /dtu/blackhole/0a/224426/NWPU_downscaled/
      2x/
        images/
          0001.jpg
          ...
      4x/
        images/
          0001.jpg
          ...
"""

import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Add project root so we can import Jan's module ────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from transform.downsample import resize

# ── Paths ─────────────────────────────────────────────────────────────
NWPU_ROOT = Path("/dtu/blackhole/02/137570/MultiRes/NWPU_crowd")
OUT_ROOT = Path("/dtu/blackhole/0a/224426/NWPU_downscaled")

SCALES = [2, 4]
SPLITS = ["train", "val"]


def main():
    for scale in SCALES:
        out_dir = OUT_ROOT / f"{scale}x" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect image IDs from both splits
        image_ids = []
        for split in SPLITS:
            with open(NWPU_ROOT / f"{split}.txt") as f:
                ids = [line.strip().split()[0] for line in f if line.strip()]
                image_ids.extend(ids)

        print(f"\n{'='*50}")
        print(f"  Downscaling {len(image_ids)} images at {scale}x")
        print(f"  Output: {out_dir}")
        print(f"{'='*50}")

        for image_id in tqdm(image_ids, desc=f"{scale}x"):
            src_path = NWPU_ROOT / "images" / f"{image_id}.jpg"
            dst_path = out_dir / f"{image_id}.jpg"

            if dst_path.exists():
                continue

            img = Image.open(src_path).convert("RGB")
            img_down = resize(img, scale=1 / scale, method="bilinear")
            img_down.save(dst_path, quality=95)

    print("\nDone.")


if __name__ == "__main__":
    main()
