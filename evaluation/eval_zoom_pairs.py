"""
Baseline evaluation of CLIP-EBC (ViT-B/16) on supervisor's Zoom Pairs.

Measures HR vs LR count consistency — no ground truth, just comparing
predictions on the same scene at two different optical resolutions.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python evaluation/eval_zoom_pairs.py [--device cuda:0]
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
from utils.eval_utils import sliding_window_predict

WEIGHTS_PATH = Path("/dtu/blackhole/0a/224426/best_mae.pth")
ZOOM_ROOT = Path("/dtu/blackhole/02/137570/MultiRes/test")

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

    pair_dirs = sorted(
        [p for p in ZOOM_ROOT.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )

    print(f"\nEvaluating Zoom Pairs — HR vs LR consistency ({len(pair_dirs)} pairs)")

    results = []

    for pair_dir in tqdm(pair_dirs, desc="Zoom pairs"):
        pair_idx = pair_dir.name

        hr_img = Image.open(pair_dir / f"{pair_idx}_hr.jpg").convert("RGB")
        lr_img = Image.open(pair_dir / f"{pair_idx}_lr.jpg").convert("RGB")

        hr_tensor = NORMALIZE(T.ToTensor()(hr_img))
        lr_tensor = NORMALIZE(T.ToTensor()(lr_img))

        hr_count = predict_count(model, hr_tensor, device)
        lr_count = predict_count(model, lr_tensor, device)

        diff = abs(hr_count - lr_count)
        ratio = hr_count / lr_count if lr_count > 0 else float("inf")

        results.append(dict(
            pair=pair_idx,
            hr_count=hr_count,
            lr_count=lr_count,
            abs_diff=diff,
            ratio=ratio,
            hr_size=list(hr_img.size),
            lr_size=list(lr_img.size),
        ))

    # Summary statistics
    abs_diffs = [r["abs_diff"] for r in results]
    ratios = [r["ratio"] for r in results if r["ratio"] != float("inf")]
    hr_counts = [r["hr_count"] for r in results]
    lr_counts = [r["lr_count"] for r in results]

    print(f"\n{'='*50}")
    print(f"  Zoom Pairs Results")
    print(f"{'='*50}")
    print(f"  Mean HR count:        {np.mean(hr_counts):.1f}")
    print(f"  Mean LR count:        {np.mean(lr_counts):.1f}")
    print(f"  Mean |HR - LR|:       {np.mean(abs_diffs):.1f}")
    print(f"  Median |HR - LR|:     {np.median(abs_diffs):.1f}")
    print(f"  Mean HR/LR ratio:     {np.mean(ratios):.3f}")
    print(f"  Median HR/LR ratio:   {np.median(ratios):.3f}")

    worst = sorted(results, key=lambda r: r["abs_diff"], reverse=True)[:5]
    print(f"\n  Worst 5 pairs by |HR - LR|:")
    for r in worst:
        print(f"    Pair {r['pair']:>3s}: HR={r['hr_count']:.0f}, LR={r['lr_count']:.0f}, "
              f"diff={r['abs_diff']:.0f}, ratio={r['ratio']:.2f}")

    best = sorted(results, key=lambda r: r["abs_diff"])[:5]
    print(f"\n  Best 5 pairs by |HR - LR|:")
    for r in best:
        print(f"    Pair {r['pair']:>3s}: HR={r['hr_count']:.0f}, LR={r['lr_count']:.0f}, "
              f"diff={r['abs_diff']:.0f}, ratio={r['ratio']:.2f}")

    results_dir = PROJECT_ROOT / "results" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "zoom_pairs.json", "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "mean_abs_diff": float(np.mean(abs_diffs)),
        "median_abs_diff": float(np.median(abs_diffs)),
        "mean_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "mean_hr_count": float(np.mean(hr_counts)),
        "mean_lr_count": float(np.mean(lr_counts)),
    }
    with open(results_dir / "zoom_pairs_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
