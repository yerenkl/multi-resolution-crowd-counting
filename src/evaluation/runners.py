import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

from src.settings import settings
from src.models.clip_ebc import NORMALIZE
from src.evaluation.inference import predict_count
from utils.eval_utils import calculate_errors  # CLIP-EBC utility


def _eval_single_nwpu_root(model, device, root_dir, image_ids):
    pred_counts, gt_counts = [], []

    for image_id in tqdm(image_ids, desc=f"Evaluating {root_dir.name}"):
        img = Image.open(root_dir / "images" / f"{image_id}.jpg").convert("RGB")
        img_tensor = NORMALIZE(T.ToTensor()(img))

        with open(root_dir / "jsons" / f"{image_id}.json") as f:
            gt_count = json.load(f)["human_num"]

        pred_counts.append(predict_count(model, img_tensor, device))
        gt_counts.append(gt_count)

    pred_arr = np.array(pred_counts)
    gt_arr = np.array(gt_counts)

    errors = calculate_errors(pred_arr, gt_arr)
    avg_diff = np.mean(pred_arr - gt_arr)

    return {
        **errors,
        "avg_diff": avg_diff
    }


def eval_nwpu(model, device, nwpu_downscaled_directory, limit: int = None) -> dict:
    """Evaluate on NWPU val for both original and downscaled datasets."""
    model.eval()

    nwpu_root = settings.nwpu_dir

    # Load shared val.txt
    with open(nwpu_root / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()]

    if limit is not None:
        image_ids = image_ids[:limit]

    # Evaluate both datasets
    results_original = _eval_single_nwpu_root(model, device, nwpu_root, image_ids)
    results_downscaled = _eval_single_nwpu_root(model, device, nwpu_downscaled_directory, image_ids)

    return {
        "original": results_original,
        "downscaled": results_downscaled
    }


def eval_nwpu_downscaled(model, device, scale: int) -> dict:
    """Evaluate on pre-saved downscaled NWPU val images. Returns dict with mae and rmse."""
    model.eval()
    nwpu_root = settings.nwpu_dir
    images_dir = settings.NWPU_DOWNSCALED_DIR / f"{scale}x" / "images"
    assert images_dir.exists(), (
        f"Downscaled images not found at {images_dir}. Run entrypoints/downscale_nwpu.py first."
    )

    with open(nwpu_root / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()]

    pred_counts, gt_counts = [], []
    for image_id in tqdm(image_ids, desc=f"NWPU val {scale}x"):
        img = Image.open(images_dir / f"{image_id}.jpg").convert("RGB")
        img_tensor = NORMALIZE(T.ToTensor()(img))
        with open(nwpu_root / "jsons" / f"{image_id}.json") as f:
            gt_count = json.load(f)["human_num"]
        pred_counts.append(predict_count(model, img_tensor, device))
        gt_counts.append(gt_count)

    return calculate_errors(np.array(pred_counts), np.array(gt_counts))


def eval_zoom_pairs(model, device) -> list:
    """Evaluate HR vs LR count consistency on Zoom Pairs. Returns list of per-pair dicts."""
    model.eval()
    zoom_root = settings.zoom_pairs_dir
    pair_dirs = sorted(
        [p for p in zoom_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )

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

    return results
