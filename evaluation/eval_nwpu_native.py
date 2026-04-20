"""
Baseline evaluation of CLIP-EBC (ViT-B/16) on NWPU-Crowd val at native resolution.

Should reproduce roughly MAE ~61 as reported in the paper.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python evaluation/eval_nwpu_native.py [--device cuda:0]
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

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIP_EBC_ROOT = Path("/dtu/blackhole/0a/224426/CLIP-EBC-main")
sys.path.insert(0, str(CLIP_EBC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from models import get_model
from utils.eval_utils import calculate_errors, sliding_window_predict

WEIGHTS_PATH = Path("/dtu/blackhole/0a/224426/best_mae.pth")
NWPU_ROOT = Path("/dtu/blackhole/02/137570/MultiRes/NWPU_crowd")

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
    img = img_tensor.unsqueeze(0).to(device)
    density = sliding_window_predict(model, img, WINDOW_SIZE, STRIDE)
    return density.sum().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)

    with open(NWPU_ROOT / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()]

    print(f"\nEvaluating NWPU val — native resolution ({len(image_ids)} images)")

    pred_counts = []
    gt_counts = []

    for image_id in tqdm(image_ids, desc="NWPU val native"):
        img = Image.open(NWPU_ROOT / "images" / f"{image_id}.jpg").convert("RGB")
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

    print(f"\n  Results (native):")
    print(f"    MAE:  {errors['mae']:.2f}")
    print(f"    RMSE: {errors['rmse']:.2f}")

    results_dir = PROJECT_ROOT / "results" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(results_dir / "nwpu_val_native.npz", pred_counts=pred_counts, gt_counts=gt_counts)
    with open(results_dir / "nwpu_val_native.json", "w") as f:
        json.dump({"mae": float(errors["mae"]), "rmse": float(errors["rmse"])}, f, indent=2)

    print(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
