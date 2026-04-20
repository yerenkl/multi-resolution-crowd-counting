"""
Baseline evaluation of CLIP-EBC (ViT-B/16) on NWPU-Crowd val at 2x and 4x downscale.

Uses pre-saved downscaled images from /dtu/blackhole/0a/224426/NWPU_downscaled/.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python evaluation/eval_nwpu_downscaled.py [--device cuda:0]
"""

import sys
import json
import argparse
import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIP_EBC_ROOT = Path("/dtu/blackhole/0a/224426/CLIP-EBC-main")
sys.path.insert(0, str(CLIP_EBC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from models import get_model
from utils.eval_utils import calculate_errors, sliding_window_predict

WEIGHTS_PATH = Path("/dtu/blackhole/0a/224426/best_mae.pth")
NWPU_ROOT = Path("/dtu/blackhole/02/137570/MultiRes/NWPU_crowd")
DOWNSCALED_ROOT = Path("/dtu/blackhole/0a/224426/NWPU_downscaled")

# ── Model config (matches NWPU_CLIP_ViT_B_16_Word release) ────────────
MODEL_CFG = dict(
    backbone="clip_vit_b_16",
    input_size=224,
    reduction=8,
    bins=[[0, 0], [1, 1], [2, 2], [3, 3], [4, float("inf")]],
    anchor_points=[0, 1, 2, 3, 4.21931],
    prompt_type="word",
    num_vpt=32,
    vpt_drop=0.0,
    deep_vpt=True,
)

WINDOW_SIZE = 224
STRIDE = 224

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_model(device):
    model = get_model(**MODEL_CFG)
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print(f"Loaded weights from {WEIGHTS_PATH}")
    return model


@torch.no_grad()
def predict_count(model, img_tensor, device):
    # Ensure minimum size of 224x224 for ViT window
    _, h, w = img_tensor.shape
    if h < WINDOW_SIZE or w < WINDOW_SIZE:
        scale = WINDOW_SIZE / min(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
        ).squeeze(0)
    img = img_tensor.unsqueeze(0).to(device)
    density = sliding_window_predict(model, img, WINDOW_SIZE, STRIDE)
    return density.sum().item()


def eval_scale(model, device, scale):
    images_dir = DOWNSCALED_ROOT / f"{scale}x" / "images"
    assert images_dir.exists(), f"Downscaled images not found at {images_dir}. Run downscale_nwpu.py first."

    with open(NWPU_ROOT / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()]

    print(f"\nEvaluating NWPU val — {scale}x downscale ({len(image_ids)} images)")

    pred_counts = []
    gt_counts = []

    for image_id in tqdm(image_ids, desc=f"{scale}x"):
        img_path = images_dir / f"{image_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        img_tensor = NORMALIZE(T.ToTensor()(img))

        with open(NWPU_ROOT / "jsons" / f"{image_id}.json") as f:
            ann = json.load(f)
        gt_count = ann["human_num"]

        pred = predict_count(model, img_tensor, device)
        pred_counts.append(pred)
        gt_counts.append(gt_count)

    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)
    errors = calculate_errors(pred_counts, gt_counts)

    print(f"\n  Results ({scale}x downscale):")
    print(f"    MAE:  {errors['mae']:.2f}")
    print(f"    RMSE: {errors['rmse']:.2f}")

    return errors, pred_counts, gt_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)

    results_dir = PROJECT_ROOT / "results" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for scale in [2, 4]:
        errors, preds, gts = eval_scale(model, device, scale)
        tag = f"{scale}x_down"
        summary[tag] = {"mae": float(errors["mae"]), "rmse": float(errors["rmse"])}
        np.savez(results_dir / f"nwpu_val_{tag}.npz", pred_counts=preds, gt_counts=gts)

    print(f"\n{'='*40}")
    print(f"  NWPU Val Downscaled Summary")
    print(f"{'='*40}")
    print(f"  {'Scale':<12} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*28}")
    for tag, err in summary.items():
        print(f"  {tag:<12} {err['mae']:>8.2f} {err['rmse']:>8.2f}")

    with open(results_dir / "nwpu_val_downscaled.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
