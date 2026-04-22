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

from image_ops.downsample import transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.settings import settings
from src.image_ops.downsample import resize, MethodWeights

SCALES = [4]
SPLITS = ["train", "val"]
methods_weights = [
    MethodWeights.bicubic_only(),
]

nwpu_root = settings.nwpu_dir
out_root = settings.NWPU_DOWNSCALED_DIR


def main():
    for scale in SCALES:
        for method in methods_weights:
            method_tag = "bilinear" if method.bilinear == 1.0 else "bicubic" if method.bicubic == 1.0 else "nearest" if method.nearest == 1.0 else "lanczos"
            out_dir = out_root / f"{method_tag}" / f"{scale}x" / "images"
            out_dir.mkdir(parents=True, exist_ok=True)

            image_ids = []
            for split in SPLITS:
                with open(nwpu_root / f"{split}.txt") as f:
                    image_ids.extend(line.strip().split()[0] for line in f if line.strip())

            print(f"\n{'=' * 50}")
            print(f"  Downscaling {len(image_ids)} images at {scale}x")
            print(f"  Output: {out_dir}")
            print(f"{'=' * 50}")

            for image_id in tqdm(image_ids, desc=f"{scale}x"):
                src_path = nwpu_root / "images" / f"{image_id}.jpg"
                dst_path = out_dir / f"{image_id}.jpg"
                if dst_path.exists():
                    continue
                img = Image.open(src_path).convert("RGB")
                img_down = transform(img, pre_downsampling_blur=True, downsample_factor=scale, method_weights=method)
                img_down.save(dst_path)

        print("\nDone.")


import os
import json


def downscale_annotations(input_folder, output_folder, scale=4.0):
    """
    Downscale all JSON annotation files in a folder.

    Args:
        input_folder (str): Path to folder with input JSON files
        output_folder (str): Path to save processed JSON files
        scale (float): Downscaling factor (default: 4.0)
    """
    os.makedirs(output_folder, exist_ok=True)

    def downscale_point(point):
        return [point[0] / scale, point[1] / scale]

    def downscale_box(box):
        # box format: [x_min, y_min, x_max, y_max]
        return [
            box[0] / scale,
            box[1] / scale,
            box[2] / scale,
            box[3] / scale
        ]

    for filename in os.listdir(input_folder):
        if not filename.endswith(".json"):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, "r") as f:
            data = json.load(f)

        if "points" in data:
            data["points"] = [downscale_point(p) for p in data["points"]]

        if "boxes" in data:
            data["boxes"] = [downscale_box(b) for b in data["boxes"]]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

    print(f"Processed all JSON files from '{input_folder}' → '{output_folder}'")
if __name__ == "__main__":
    downscale_annotations(nwpu_root / "jsons", out_root / "jsons" / "4x")
