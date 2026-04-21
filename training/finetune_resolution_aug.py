"""
Resolution-augmented fine-tuning of CLIP-EBC (ViT-B/16) on NWPU-Crowd.

During training each crop is randomly downscaled by 1x-4x before being fed
to the model, forcing it to learn resolution-invariant crowd counting.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python training/finetune_resolution_aug.py [--device cuda:0]
"""

import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List
from torch import Tensor

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIP_EBC_ROOT = Path("/dtu/blackhole/0a/224426/CLIP-EBC-main")
sys.path.insert(0, str(CLIP_EBC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from models import get_model
from losses import DACELoss
from datasets.utils import generate_density_map
from src.settings import settings

# ── Config ────────────────────────────────────────────────────────────
WEIGHTS_PATH = Path("/dtu/blackhole/0a/224426/best_mae.pth")

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

LOSS_CFG = dict(
    bins=[[0, 0], [1, 1], [2, 2], [3, 3], [4, float("inf")]],
    reduction=8,
    weight_count_loss=1.0,
    count_loss="dmcount",
    input_size=224,
)

CROP_SIZE = 224
NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ── Dataset ───────────────────────────────────────────────────────────
class NWPUTrain(torch.utils.data.Dataset):
    """
    NWPU train split with random crop + resolution augmentation.
    Returns (image, points, density_map) matching DACELoss expectations.
    """

    def __init__(self, min_scale: float = 1.0, max_scale: float = 4.0):
        self.root = settings.nwpu_dir
        self.min_scale = min_scale
        self.max_scale = max_scale

        with open(self.root / "train.txt") as f:
            self.image_ids = [line.strip().split()[0] for line in f if line.strip()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        img = Image.open(self.root / "images" / f"{image_id}.jpg").convert("RGB")
        with open(self.root / "jsons" / f"{image_id}.json") as f:
            ann = json.load(f)

        points = ann["points"]
        points = (torch.tensor(points, dtype=torch.float32)
                  if points else torch.zeros((0, 2), dtype=torch.float32))

        orig_w, orig_h = img.size

        # ── Random crop (scale * CROP_SIZE) then resize to CROP_SIZE ──
        scale = random.uniform(1.0, 2.0)
        crop_size = int(CROP_SIZE * scale)
        crop_size = min(crop_size, orig_w, orig_h)

        x0 = random.randint(0, max(0, orig_w - crop_size))
        y0 = random.randint(0, max(0, orig_h - crop_size))
        x1, y1 = x0 + crop_size, y0 + crop_size

        img = img.crop((x0, y0, x1, y1))

        # Filter and remap points to crop
        if points.numel() > 0:
            px, py = points[:, 0], points[:, 1]
            mask = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
            points = points[mask]
            if points.numel() > 0:
                points[:, 0] -= x0
                points[:, 1] -= y0

        # ── Resolution augmentation: randomly downscale ───────────────
        down_factor = random.uniform(self.min_scale, self.max_scale)
        lr_w = max(CROP_SIZE, int(crop_size / down_factor))
        lr_h = max(CROP_SIZE, int(crop_size / down_factor))
        img = img.resize((lr_w, lr_h), Image.BILINEAR)

        # Resize to CROP_SIZE
        img = img.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)

        # Rescale points to final image size
        if points.numel() > 0:
            points[:, 0] *= CROP_SIZE / crop_size
            points[:, 1] *= CROP_SIZE / crop_size

        # ── Random horizontal flip ─────────────────────────────────────
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if points.numel() > 0:
                points[:, 0] = CROP_SIZE - points[:, 0]

        # ── To tensor + normalise ──────────────────────────────────────
        img_tensor = NORMALIZE(T.ToTensor()(img))

        # ── Generate density map ───────────────────────────────────────
        density = generate_density_map(points, CROP_SIZE, CROP_SIZE, sigma=None)

        return img_tensor, points, density


def collate_fn(batch):
    images, points_list, densities = zip(*batch)
    images = torch.stack(images, 0)
    densities = torch.stack(densities, 0)
    return images, list(points_list), densities


# ── Training ──────────────────────────────────────────────────────────
def train_epoch(model, loader, loss_fn, optimizer, scaler, device):
    model.train()
    total_loss = 0.0

    for images, points, densities in tqdm(loader, desc="Training"):
        images = images.to(device)
        densities = densities.to(device)
        points = [p.to(device) for p in points]

        optimizer.zero_grad()

        with autocast():
            pred_class, pred_density = model(images)
            loss, loss_info = loss_fn(pred_class, pred_density, densities, points)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_info["loss"].item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, device, nwpu_root: Path):
    """Quick eval on NWPU val — sliding window, native resolution."""
    from utils.eval_utils import calculate_errors, sliding_window_predict

    model.eval()
    with open(nwpu_root / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()]

    pred_counts, gt_counts = [], []
    for image_id in tqdm(image_ids[:100], desc="Val (first 100)"):  # quick eval
        img = Image.open(nwpu_root / "images" / f"{image_id}.jpg").convert("RGB")
        img_tensor = NORMALIZE(T.ToTensor()(img)).unsqueeze(0).to(device)
        density = sliding_window_predict(model, img_tensor, 224, 224)
        pred_counts.append(density.sum().item())
        with open(nwpu_root / "jsons" / f"{image_id}.json") as f:
            pred_counts_gt = json.load(f)["human_num"]
        gt_counts.append(pred_counts_gt)

    return calculate_errors(np.array(pred_counts), np.array(gt_counts))


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min_scale", type=float, default=1.0,
                        help="Min downscale factor for resolution augmentation")
    parser.add_argument("--max_scale", type=float, default=4.0,
                        help="Max downscale factor for resolution augmentation")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Model ──────────────────────────────────────────────────────────
    model = get_model(**MODEL_CFG)
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    print(f"Loaded pretrained weights from {WEIGHTS_PATH}")

    # ── Loss, optimiser, scaler ────────────────────────────────────────
    loss_fn = DACELoss(**LOSS_CFG).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scaler = GradScaler()

    # ── Data ───────────────────────────────────────────────────────────
    dataset = NWPUTrain(min_scale=args.min_scale, max_scale=args.max_scale)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    print(f"Training on {len(dataset)} images, {len(loader)} batches/epoch")

    # ── Output dir ─────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "results" / "finetune_resolution_aug"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loader, loss_fn, optimizer, scaler, device)

        errors = eval_epoch(model, device, settings.nwpu_dir)
        mae, rmse = errors["mae"], errors["rmse"]

        print(f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | MAE={mae:.2f} | RMSE={rmse:.2f}")

        # Save best checkpoint
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), out_dir / "best_mae.pth")
            print(f"  → Saved best model (MAE={mae:.2f})")

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mae": mae,
            "rmse": rmse,
        }, out_dir / "latest.pth")

    print(f"\nTraining done. Best MAE: {best_mae:.2f}")
    print(f"Weights saved to {out_dir}/")


if __name__ == "__main__":
    main()
