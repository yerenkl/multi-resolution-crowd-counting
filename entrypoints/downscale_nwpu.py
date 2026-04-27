import argparse
import os
import random

import numpy as np
import torchvision.transforms as tf
from PIL import ImageFilter, Image


class MethodWeights:

    def __init__(self, bilinear: float, bicubic: float, nearest: float, lanczos: float):
        total = bilinear + bicubic + nearest + lanczos
        self.bilinear = bilinear
        self.bicubic = bicubic
        self.nearest = nearest
        self.lanczos = lanczos

    @staticmethod
    def bilinear_only():
        return MethodWeights(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def bicubic_only():
        return MethodWeights(0.0, 1.0, 0.0, 0.0)

    @staticmethod
    def nearest_only():
        return MethodWeights(0.0, 0.0, 1.0, 0.0)

    @staticmethod
    def lanczos_only():
        return (MethodWeights(0.0, 0.0, 0.0, 1.0))

    @staticmethod
    def mix():
        return MethodWeights(0.25, 0.25, 0.25, 0.25)

    def as_list(self):
        return [self.bilinear, self.bicubic, self.nearest, self.lanczos]


def gaussian_blur(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize(img, scale, method='bilinear'):
    methods = {
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC,
        'nearest': Image.Resampling.NEAREST,
        'lanczos': Image.Resampling.LANCZOS
    }

    new_size = (int(img.width * scale), int(img.height * scale))

    return img.resize(new_size, methods[method])


def transform(img, pre_downsampling_blur: bool = True, downsample_factor: int = 4,
              method_weights: MethodWeights = MethodWeights(0.25, 0.25, 0.25, 0.25), upsample: bool = False,
              add_noise: bool = False):
    if pre_downsampling_blur:
        img = gaussian_blur(img, sigma=random.uniform(0.3, 1.3))

    methods = ['bilinear', 'bicubic', 'nearest', 'lanczos']
    method = random.choices(methods, weights=method_weights.as_list(), k=1)[0]
    img = resize(img, scale=1 / downsample_factor, method=method)

    if upsample:
        img = resize(img, downsample_factor, method="bilinear")

    if add_noise:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 1.5, arr.shape)
        img = np.clip(arr + noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(img)

    # img = tf.ToTensor()(img)

    return img

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

import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.settings import settings

SCALES = [4]
SPLITS = ["train", "val"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    type=str,
    default="mix",
    choices=["bilinear", "bicubic", "nearest", "lanczos", "mix"],
    help="Downscaling method"
)
parser.add_argument(
    "--noise",
    type=bool,
    default=False,
)

args = parser.parse_args()

match args.method:
    case "bilinear":
        method = MethodWeights.bilinear_only()
    case "bicubic":
        method = MethodWeights.bicubic_only()
    case "nearest":
        method = MethodWeights.nearest_only()
    case "lanczos":
        method = MethodWeights.lanczos_only()
    case "mix":
        method = MethodWeights.mix()

nwpu_root = settings.nwpu_dir
out_root = settings.NWPU_DOWNSCALED_DIR


def main():
    for scale in SCALES:

        noise_tag = "noise" if args.noise else "no_noise"

        method_tag = (
            "bilinear" if method.bilinear == 1.0 else
            "bicubic" if method.bicubic == 1.0 else
            "nearest" if method.nearest == 1.0 else
            "lanczos" if method.lanczos == 1.0 else
            "mix"
        )

        out_dir = out_root / f"{method_tag}" / f"{scale}x"/ noise_tag / "images"
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
            src_json = nwpu_root / "jsons" / f"{image_id}.json"

            if dst_path.exists():
                continue

            img = Image.open(src_path).convert("RGB")
            w, h = img.size

            # Resulting size with requested scale
            new_w = w / scale
            new_h = h / scale

            if new_w >= 224 and new_h >= 224:
                final_scale = scale
            else:
                # fallback: largest possible scale that keeps >=224
                final_scale = min(w / 224, h / 224)

            final_scale = max(1, final_scale)
            img_down = transform(
                img,
                pre_downsampling_blur=True,
                downsample_factor=int(final_scale),
                method_weights=method,
                add_noise= args.noise,
            )
            img_down.save(dst_path)

            dst_json = out_root / f"{method_tag}" / f"{scale}x" / noise_tag / "jsons" / f"{image_id}.json"

            if src_json.exists():
                with open(src_json) as f:
                    data = json.load(f)

                # scale all points
                if "points" in data:
                    data["points"] = [
                        [x / final_scale, y / final_scale]
                        for x, y in data["points"]
                    ]

                dst_json.parent.mkdir(parents=True, exist_ok=True)

                with open(dst_json, "w") as f:
                    json.dump(data, f, indent=2)

        print("\nDone.")

if __name__ == "__main__":
    main()