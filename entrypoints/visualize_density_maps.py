"""
Visualize density maps side-by-side: baseline vs DANN (mixed) model.

Picks a sample of images from each density bucket (sparse / medium / dense)
and saves a multi-panel figure per image showing:
  - Original image
  - Baseline density map overlay + predicted count
  - DANN density map overlay + predicted count
  - GT count in title

Usage:
    uv run python entrypoints/visualize_density_maps.py \
        --dann_weights /work3/s225224/.../best_mae.pth \
        --n_per_bucket 4 \
        --output_dir results/dann_v2/mixed/viz
"""

import sys
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.clip_ebc import load_model, NORMALIZE
from src.settings import settings
from utils.eval_utils import sliding_window_predict

WINDOW = 224
STRIDE = 224

BUCKET_THRESHOLDS = {"sparse": (0, 50), "medium": (51, 499), "dense": (500, 10**9)}


def load_weights(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=True)
    return model


@torch.no_grad()
def predict_density_map(model, img_tensor, device):
    _, h, w = img_tensor.shape
    if h < WINDOW or w < WINDOW:
        scale = WINDOW / min(h, w)
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(int(h * scale), int(w * scale)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    density = sliding_window_predict(model, img_tensor.unsqueeze(0).to(device), WINDOW, STRIDE)
    return density.squeeze().cpu().numpy()


def bucket(gt_count):
    if gt_count <= 50:
        return "sparse"
    if gt_count < 500:
        return "medium"
    return "dense"


def pick_samples(nwpu_root, n_per_bucket, n_hardest=0, seed=42):
    with open(nwpu_root / "val.txt") as f:
        rows = [l.strip().split() for l in f if l.strip()]

    all_items = []
    by_bucket = {"sparse": [], "medium": [], "dense": []}
    for row in rows:
        image_id = row[0]
        with open(nwpu_root / "jsons" / f"{image_id}.json") as f:
            gt = json.load(f)["human_num"]
        by_bucket[bucket(gt)].append((image_id, gt))
        all_items.append((image_id, gt))

    rng = random.Random(seed)
    selected = []
    seen = set()

    for name, items in by_bucket.items():
        chosen = rng.sample(items, min(n_per_bucket, len(items)))
        chosen.sort(key=lambda x: x[1])
        for image_id, gt in chosen:
            selected.append((name, image_id, gt))
            seen.add(image_id)

    if n_hardest > 0:
        hardest = sorted(all_items, key=lambda x: x[1], reverse=True)[:n_hardest]
        for image_id, gt in hardest:
            if image_id not in seen:
                selected.append(("hardest", image_id, gt))
                seen.add(image_id)

    order = {"sparse": 0, "medium": 1, "dense": 2, "hardest": 3}
    selected.sort(key=lambda x: (order[x[0]], x[2]))
    return selected


def make_figure(image_id, gt_count, bucket_name, img_pil, baseline_map, dann_map):
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.05, hspace=0.12)

    # shared scale across both maps so they're comparable
    vmax = max(np.percentile(baseline_map, 99.5), np.percentile(dann_map, 99.5), 1e-8)

    top_titles = [
        f"Original  |  GT: {gt_count}",
        f"Baseline  |  pred: {baseline_map.sum():.0f}  |  err: {baseline_map.sum() - gt_count:+.0f}",
        f"DANN mixed  |  pred: {dann_map.sum():.0f}  |  err: {dann_map.sum() - gt_count:+.0f}",
    ]

    # top row: image + overlays
    for col, (title, overlay) in enumerate(zip(top_titles, [None, baseline_map, dann_map])):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img_pil)
        if overlay is not None:
            ax.imshow(
                overlay,
                cmap="magma",
                alpha=0.6,
                vmin=0,
                vmax=vmax,
                extent=(0, img_pil.width, img_pil.height, 0),
                interpolation="bilinear",
            )
        ax.set_title(title, fontsize=10, pad=4)
        ax.axis("off")

    # bottom row: standalone density maps + difference
    diff_map = dann_map - baseline_map
    diff_abs = max(np.abs(diff_map).max(), 1e-8)

    bottom_data = [
        ("Baseline density map", baseline_map, "magma", 0, vmax),
        ("DANN density map", dann_map, "magma", 0, vmax),
        (f"Diff (DANN − baseline)  |  range: [{diff_map.min():.3f}, {diff_map.max():.3f}]",
         diff_map, "RdBu_r", -diff_abs, diff_abs),
    ]
    for col, (title, data, cmap, vmin, vmax_) in enumerate(bottom_data):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax_, interpolation="bilinear", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.set_title(title, fontsize=9, pad=4)
        ax.axis("off")

    fig.suptitle(f"{image_id}  [{bucket_name}]", fontsize=12, y=1.01)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dann_weights", required=True)
    parser.add_argument("--n_per_bucket", type=int, default=4)
    parser.add_argument("--n_hardest", type=int, default=0, help="Also include the N images with highest GT count")
    parser.add_argument("--output_dir", type=str, default="results/dann_v2/mixed/viz")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nwpu_root = settings.nwpu_dir

    print("Loading baseline model...")
    baseline = load_model(device)
    baseline.eval()

    print("Loading DANN (mixed) model...")
    dann = load_model(device)
    load_weights(dann, args.dann_weights)
    dann.eval()

    samples = pick_samples(nwpu_root, args.n_per_bucket, n_hardest=args.n_hardest, seed=args.seed)
    print(f"Generating {len(samples)} figures ({args.n_per_bucket} per bucket)...")

    to_tensor = T.ToTensor()
    for bucket_name, image_id, gt_count in tqdm(samples):
        img_path = nwpu_root / "images" / f"{image_id}.jpg"
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = NORMALIZE(to_tensor(img_pil))

        baseline_map = predict_density_map(baseline, img_tensor, device)
        dann_map = predict_density_map(dann, img_tensor, device)

        fig = make_figure(image_id, gt_count, bucket_name, img_pil, baseline_map, dann_map)
        out_path = output_dir / f"{bucket_name}_{image_id}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.name}  (GT={gt_count}, baseline={baseline_map.sum():.0f}, dann={dann_map.sum():.0f})")

    print(f"\nDone. All figures in {output_dir}/")


if __name__ == "__main__":
    main()
