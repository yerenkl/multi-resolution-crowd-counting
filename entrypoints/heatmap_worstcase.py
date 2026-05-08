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
from scipy.ndimage import gaussian_filter
from PIL import Image

from src.models.clip_ebc import load_model  # also puts CLIP_EBC_DIR in sys.path
from src.models.clip_ebc import NORMALIZE
from src.models.clip_ebc import make_density_map
from src.settings import settings


DEFAULT_IMAGE_DIR = settings.nwpu_dir / "images"


def image_to_tensor(image: Image.Image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def build_gt_density(points: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if points.size == 0:
        pts = torch.zeros((0, 2), dtype=torch.float32)
    else:
        arr = points.astype(np.float32).copy()
        arr[:, 0] = np.clip(arr[:, 0], 0, out_w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, out_h - 1)
        pts = torch.from_numpy(arr)

    gt = make_density_map(pts, out_h, out_w)
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    return gaussian_filter(np.squeeze(gt), sigma=4.0)

def pad_to_window(tensor, window):
    """Pad tensor so H and W are divisible by window size."""
    _, h, w = tensor.shape
    pad_h = (window - h % window) % window
    pad_w = (window - w % window) % window
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    return tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--num-viz", type=int, default=4, help="Number of images to visualize as heatmaps.")
    parser.add_argument("--viz-dir", type=Path, default=settings.RESULTS_DIR / "superres" / "heatmaps")
    parser.add_argument("--tta", action="store_true", default=False, help="Whether to apply test-time augmentation (TTA) via multi-scale prediction and averaging")
    args = parser.parse_args()

    tta = args.tta

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    viz_dir = args.viz_dir.expanduser().resolve()
    viz_dir.mkdir(parents=True, exist_ok=True)


    image_dir = args.image_dir.expanduser().resolve()
    with open(settings.nwpu_dir / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()]

    model.eval()
    total = 0
    for image_id in image_ids:
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
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
            if tta:
                densities = []
                scales = [1.0, 0.75, 0.5]
                for s in scales:
                    resized = F.interpolate(
                        img_tensor.unsqueeze(0),
                        scale_factor=s,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    
                    resized = pad_to_window(resized, window)  # ← add this
                    densities.append(sliding_window_predict(model, resized.unsqueeze(0).to(device), window, stride))
                
                # Interpolate all density maps to match the first one's spatial size
                target_shape = densities[0].shape[-2:]
                densities_aligned = [densities[0]]
                for d in densities[1:]:
                    d_aligned = F.interpolate(
                        d,
                        size=target_shape,
                        mode="bilinear",
                        align_corners=False,
                    )
                    densities_aligned.append(d_aligned)
                
                # Average the aligned density maps
                density = sum(densities_aligned) / len(densities_aligned)
            else:
                density = sliding_window_predict(model, img_tensor.unsqueeze(0).to(device), window, stride)
            pred_count = density.sum().item()
            ann_path = settings.nwpu_dir / "jsons" / f"{image_id}.json"
            with ann_path.open() as f:
                ann = json.load(f)
            gt_count = ann["human_num"]
            print(f"{image_id}: Predicted count = {pred_count:.1f}, GT count = {gt_count}")

            # Keep only samples with a large absolute count error.
            if abs(gt_count - pred_count) < 100:
                continue

            pred_map = density.detach().cpu()
            
            # --- SCALE TO ORIGINAL IMAGE SIZE ---
            original_sum = pred_map.sum().item()
            if pred_map.dim() == 4:
                pred_map = F.interpolate(
                    pred_map,
                    size=(orig_h, orig_w), # Changed from (224, 224)
                    mode="bilinear",
                    align_corners=False,
                )
            
            # Preserve count mass after interpolation
            new_sum = pred_map.sum().item()
            if new_sum > 0:
                pred_map = pred_map * (original_sum / new_sum)
                
            pred_map = np.squeeze(pred_map.numpy())

# --- BUILD GT ON ORIGINAL RESOLUTION ---
            points = np.array(ann.get("points", []), dtype=np.float32)
            gt_map = build_gt_density(points, orig_h, orig_w)
            
            # --- THE FIX: REGIONAL DIFFERENCE ---
            # Calculate a dynamic blur radius based on image size (e.g., 3% of the width)
            # This makes the "blobs" large enough to overlap if they are in the same general area.
            compare_sigma = max(orig_w, orig_h) * 0.03 
            
            # Smooth both maps so slight spatial shifts don't register as "misses"
            smoothed_gt = gaussian_filter(gt_map, sigma=compare_sigma)
            smoothed_pred = gaussian_filter(pred_map, sigma=compare_sigma)

            # Now subtract the smoothed maps
            missed_map = np.clip(smoothed_gt - smoothed_pred, 0, None)

            # Mask near-zero values for transparent overlay
            # We lower the threshold slightly since blurring reduces the peak values
            threshold = missed_map.max() * 0.15 if missed_map.max() > 0 else 1e-6
            masked_missed = np.ma.masked_where(missed_map < threshold, missed_map)

            # --- PLOTTING ---
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            tta_label = " (TTA)" if tta else ""

            axes[0].imshow(img) 
            axes[0].set_title(f"Original Image\n{image_id}")
            axes[0].axis("off")

            vmax = max(float(gt_map.max()), float(pred_map.max()), 1e-6)
            gt_threshold = gt_map.max() * 0.05 if gt_map.max() > 0 else 1e-6
            masked_gt = np.ma.masked_where(gt_map < gt_threshold, gt_map)
            axes[1].imshow(img)
            axes[1].imshow(
                masked_gt,
                cmap="magma",
                alpha=0.7,
                vmin=0.0,
                vmax=vmax,
                interpolation="bilinear",
            )
            axes[1].set_title(f"GT overlay\ncount={gt_count}")
            axes[1].axis("off")

            axes[2].imshow(img) 
            axes[2].imshow(
                masked_missed,
                cmap="Reds",
                alpha=0.75,
                # We can use bilinear here because the map is heavily smoothed
                interpolation="bilinear", 
            )
            axes[2].set_title(f"Improved Prediction{tta_label}\n(Regional Analysis)")
            axes[2].axis("off")

            fig.tight_layout(pad=0.4)

            output_path = viz_dir / f"{image_id}_heatmap.png"
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"Saved heatmap to {output_path}")
            total += 1
            if total >= args.num_viz:
                break


if __name__ == "__main__":
    main()
