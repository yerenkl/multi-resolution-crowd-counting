import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

from src.settings import settings
from src.models.clip_ebc import NORMALIZE
from src.evaluation.inference import predict_count
from utils.eval_utils import calculate_errors, sliding_window_predict  # CLIP-EBC utility


def eval_nwpu(model, device, limit: int = None) -> dict:
    """Evaluate on NWPU val at native resolution. Returns dict with mae and rmse."""
    model.eval()
    nwpu_root = settings.nwpu_dir
    with open(nwpu_root / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()]

    if limit is not None:
        image_ids = image_ids[:limit]

    pred_counts, gt_counts = [], []
    desc = f"NWPU val{f' (first {limit})' if limit else ''}"
    for image_id in tqdm(image_ids, desc=desc):
        img = Image.open(nwpu_root / "images" / f"{image_id}.jpg").convert("RGB")
        img_tensor = NORMALIZE(T.ToTensor()(img))
        with open(nwpu_root / "jsons" / f"{image_id}.json") as f:
            gt_count = json.load(f)["human_num"]
        pred_counts.append(predict_count(model, img_tensor, device))
        gt_counts.append(gt_count)

    return calculate_errors(np.array(pred_counts), np.array(gt_counts))

def eval_nwpu_tta(model, device, limit: int = None) -> dict:
    """Evaluate on NWPU val at native resolution. Returns dict with mae and rmse."""
    model.eval()
    nwpu_root = settings.nwpu_dir
    with open(nwpu_root / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()]

    if limit is not None:
        image_ids = image_ids[:limit]

    pred_counts, gt_counts = [], []
    
    desc = f"NWPU val{f' (first {limit})' if limit else ''}"
    scales = [1.0, 0.75, 0.5]
    for image_id in tqdm(image_ids, desc=desc):
        counts = []
        img = Image.open(nwpu_root / "images" / f"{image_id}.jpg").convert("RGB")
        img_tensor = T.ToTensor()(img)  # convert once outside the loop

        for s in scales:
            resized = F.interpolate(
                img_tensor.unsqueeze(0),  # needs batch dim
                scale_factor=s,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            counts.append(predict_count(model, NORMALIZE(resized), device))
        pred_counts.append(sum(counts) / len(counts))
        with open(nwpu_root / "jsons" / f"{image_id}.json") as f:
            gt_counts.append(json.load(f)["human_num"])

    return calculate_errors(np.array(pred_counts), np.array(gt_counts))

def eval_nwpu_downscaled(model, device, scale: int) -> dict:
    """Evaluate on pre-saved downscaled NWPU val images. Returns dict with mae and rmse."""
    model.eval()
    nwpu_root = settings.nwpu_dir
    images_dir = Path("/dtu/blackhole/0a/224426/NWPU_downscaled/4x/images")
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


def eval_nwpu_superres(model, device) -> dict:
    """Evaluate on pre-saved downscaled NWPU val images. Returns dict with mae and rmse."""
    model.eval()
    nwpu_root = settings.nwpu_dir
    images_dir = Path("/work3/s252653/hr/hr/images")
    with open(nwpu_root / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()]

    pred_counts, gt_counts = [], []
    for image_id in tqdm(image_ids, desc=f"NWPU val superres"):
        img = Image.open(images_dir / f"{image_id}.jpg").convert("RGB")
        img_tensor = NORMALIZE(T.ToTensor()(img))
        with open(nwpu_root / "jsons" / f"{image_id}.json") as f:
            gt_count = json.load(f)["human_num"]
        pred_counts.append(predict_count(model, img_tensor, device))
        gt_counts.append(gt_count)

    return calculate_errors(np.array(pred_counts), np.array(gt_counts))

def _density_bucket(gt_count: int) -> str:
    if gt_count <= 50:
        return "sparse"
    if gt_count < 500:
        return "medium"
    return "dense"


def eval_nwpu_by_density(model, device) -> dict:
    """Evaluate on full NWPU val at native resolution, split by crowd density.

    Buckets: sparse (0–50), medium (51–499), dense (500+).
    Returns a dict with keys 'overall', 'sparse', 'medium', 'dense',
    each containing {'mae', 'rmse', 'n'}.
    """
    model.eval()
    nwpu_root = settings.nwpu_dir

    with open(nwpu_root / "val.txt") as f:
        rows = [l.strip().split() for l in f if l.strip()]

    buckets = {"sparse": ([], []), "medium": ([], []), "dense": ([], [])}
    all_preds, all_gts = [], []

    for parts in tqdm(rows, desc="NWPU val (by density)"):
        img_id = parts[0]
        img = Image.open(nwpu_root / "images" / f"{img_id}.jpg").convert("RGB")
        img_tensor = NORMALIZE(T.ToTensor()(img))
        with open(nwpu_root / "jsons" / f"{img_id}.json") as f:
            gt = json.load(f)["human_num"]
        pred = predict_count(model, img_tensor, device)
        bucket = _density_bucket(gt)
        buckets[bucket][0].append(pred)
        buckets[bucket][1].append(gt)
        all_preds.append(pred)
        all_gts.append(gt)

    results = {"overall": {**calculate_errors(np.array(all_preds), np.array(all_gts)), "n": len(all_gts)}}
    for name, (preds, gts) in buckets.items():
        results[name] = {**calculate_errors(np.array(preds), np.array(gts)), "n": len(gts)}
    return results


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
