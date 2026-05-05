"""
Evaluate CLIP-EBC (ViT-B/16) on NWPU-Crowd val at 2x and 4x downscale.

Uses pre-saved downscaled images from settings.NWPU_DOWNSCALED_DIR.

Also saves a few predicted density heatmaps for quick inspection.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python entrypoints/eval_nwpu_downscaled.py [--device cuda:0]
"""

import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/dtu/blackhole/0a/224426/CLIP-EBC-main")
from utils.eval_utils import sliding_window_predict  # CLIP-EBC utility for super-res inference

import matplotlib.pyplot as plt
from PIL import Image

from src.models.clip_ebc import load_model  # also puts CLIP_EBC_DIR in sys.path
from src.models.clip_ebc import NORMALIZE
from src.settings import settings


DEFAULT_IMAGE_DIR = Path("/dtu/blackhole/0a/224426/NWPU_downscaled/mix/4x/no_noise/images")


def image_to_tensor(image: Image.Image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--num-viz", type=int, default=4, help="Number of images to visualize as heatmaps.")
    parser.add_argument("--viz-dir", type=Path, default=settings.RESULTS_DIR / "superres" / "heatmaps")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    viz_dir = args.viz_dir.expanduser().resolve()
    viz_dir.mkdir(parents=True, exist_ok=True)


    image_dir = args.image_dir.expanduser().resolve()
    with open(settings.nwpu_dir / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()][: max(1, args.num_viz)]

    model.eval()
    for image_id in image_ids:
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            preview = img.resize((224, 224), Image.Resampling.BILINEAR)
            img_tensor = NORMALIZE(image_to_tensor(preview))
            window = 224
            stride = 224
            _, h, w = img_tensor.shape
            if h < window or w < window:
                scale = window / min(h, w)
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=(int(h * scale), int(w * scale)),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            density = sliding_window_predict(model, img_tensor.unsqueeze(0).to(device), window, stride)

            density_map = density.squeeze().detach().cpu().numpy()

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(img)
            ax.imshow(
                density_map,
                cmap="magma",
                alpha=0.55,
                extent=(0, img.width, img.height, 0),
                interpolation="bilinear",
            )
            ax.set_title(f"{image_id} heatmap")
            ax.axis("off")
            fig.tight_layout(pad=0)

            output_path = viz_dir / f"{image_id}_heatmap.png"
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()
