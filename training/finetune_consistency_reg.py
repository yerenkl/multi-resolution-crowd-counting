"""
Consistency-regularized fine-tuning of CLIP-EBC (ViT-B/16) on NWPU-Crowd.

At each training step, the same 224×224 crop is fed to the model twice:
  - once at native resolution (HR)
  - once after bilinear downscale + upscale back to 224 (LR, blurry)

Loss is personalised to the observed bias profile per crowd-density bucket:

  Sparse  (0–50):   Native overcounts (+4.8), LR is nearly unbiased (−0.1).
                    → Pull HR down toward LR only. Leave LR alone.

  Medium (50–500):  Native overcounts (+6.5), LR undershoots badly (−34).
                    → Penalise HR above GT and LR below GT separately.
                    → LR correction weighted 2× (it is the bigger error).

  Dense  (500+):    Native massively overcounts (+57.9), LR catastrophically
                    undershoots (−379). HR is also unreliable as a reference.
                    → Both branches anchored to GT directly.
                    → LR undershoot penalised hardest (default 5×).

Total loss = sup_hr + alpha_lr * sup_lr
           + lambda_cons * personalised_consistency(HR, LR, GT)

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python training/finetune_consistency_reg.py [--device cuda:0]
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIP_EBC_ROOT = Path("/dtu/blackhole/0a/224426/CLIP-EBC-main")
sys.path.insert(0, str(CLIP_EBC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from models import get_model
from losses import DACELoss
from datasets.utils import generate_density_map
from src.settings import settings

# ── Config ────────────────────────────────────────────────────────────────────
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


# ── Dataset ───────────────────────────────────────────────────────────────────
class NWPUConsistency(torch.utils.data.Dataset):
    """
    NWPU train split returning (hr_tensor, lr_tensor, points, density).

    hr_tensor : 224×224 crop, standard augmentation.
    lr_tensor : same crop downscaled by down_factor then upscaled back to 224×224.
    GT supervises both branches; the consistency loss is bucket-personalised.
    """

    def __init__(self, min_down: float = 2.0, max_down: float = 4.0):
        self.root     = settings.nwpu_dir
        self.min_down = min_down
        self.max_down = max_down

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

        # ── Random oversized crop ──────────────────────────────────────────────
        scale     = random.uniform(1.0, 2.0)
        crop_size = min(int(CROP_SIZE * scale), orig_w, orig_h)

        x0 = random.randint(0, max(0, orig_w - crop_size))
        y0 = random.randint(0, max(0, orig_h - crop_size))
        x1, y1 = x0 + crop_size, y0 + crop_size

        crop = img.crop((x0, y0, x1, y1))

        # Filter and remap points to crop-local coordinates
        if points.numel() > 0:
            px, py = points[:, 0], points[:, 1]
            mask   = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
            points = points[mask]
            if points.numel() > 0:
                points[:, 0] -= x0
                points[:, 1] -= y0

        # ── HR: resize crop to 224×224 ─────────────────────────────────────────
        hr_img = crop.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)

        if points.numel() > 0:
            points[:, 0] *= CROP_SIZE / crop_size
            points[:, 1] *= CROP_SIZE / crop_size

        # ── LR: downsample then upsample back to 224×224 ───────────────────────
        down_factor = random.uniform(self.min_down, self.max_down)
        lr_size     = max(1, int(CROP_SIZE / down_factor))
        lr_img      = hr_img.resize((lr_size, lr_size), Image.BILINEAR)
        lr_img      = lr_img.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)

        # ── Random horizontal flip (applied to both views) ─────────────────────
        if random.random() > 0.5:
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
            if points.numel() > 0:
                points[:, 0] = CROP_SIZE - points[:, 0]

        # ── To tensor + normalise ──────────────────────────────────────────────
        hr_tensor = NORMALIZE(T.ToTensor()(hr_img))
        lr_tensor = NORMALIZE(T.ToTensor()(lr_img))

        # ── Density map from HR points ─────────────────────────────────────────
        density = generate_density_map(points, CROP_SIZE, CROP_SIZE, sigma=None)

        return hr_tensor, lr_tensor, points, density


def collate_fn(batch):
    hr_imgs, lr_imgs, points_list, densities = zip(*batch)
    return (
        torch.stack(hr_imgs, 0),
        torch.stack(lr_imgs, 0),
        list(points_list),
        torch.stack(densities, 0),
    )


# ── Personalised consistency loss ─────────────────────────────────────────────
def consistency_loss_personalised(
    hr_count,
    lr_count,
    gt_count,
    asym_dense_over:  float = 2.0,
    asym_dense_under: float = 5.0,
):
    """
    Per-sample consistency loss personalised to the observed bias profile.

    Sparse  (gt < 50)
      - LR is nearly unbiased (bias −0.1); HR overcounts (+4.8).
      - Only penalise HR when it is above LR. Leave LR alone.
      - One-directional, weight 1.0.

    Medium  (50 ≤ gt < 500)
      - HR overcounts (+6.5); LR undershoots badly (−10 to −34).
      - Both branches anchored to GT directly.
      - LR-below-GT weighted 2× because it is the larger error.

    Dense   (gt ≥ 500)
      - HR massively overcounts (+57.9); LR catastrophically undershoots (−379).
      - HR is too unreliable to use as a reference for LR.
      - Both branches anchored to GT with configurable asymmetric weights.
      - Default: LR undershoot penalised 5× (the dominant failure mode).
    """

    sparse_mask = gt_count < 50
    medium_mask = (gt_count >= 50) & (gt_count < 500)
    dense_mask  =  gt_count >= 500

    # ── Sparse: pull HR toward LR only ────────────────────────────────────────
    # LR is accurate here — use it as the reference, not GT.
    # Only fires when HR > LR (the observed failure direction).
    sparse_penalty = torch.clamp(hr_count - lr_count, min=0.0)

    # ── Medium: anchor both branches to GT ────────────────────────────────────
    hr_above_gt_med = torch.clamp(hr_count - gt_count, min=0.0)  # HR overshoot
    lr_below_gt_med = torch.clamp(gt_count - lr_count, min=0.0)  # LR undershoot
    medium_penalty  = 1.0 * hr_above_gt_med + 2.0 * lr_below_gt_med

    # ── Dense: anchor both branches to GT, heavy LR correction ────────────────
    hr_above_gt_den = torch.clamp(hr_count - gt_count, min=0.0)
    lr_below_gt_den = torch.clamp(gt_count - lr_count, min=0.0)
    dense_penalty   = asym_dense_over * hr_above_gt_den + asym_dense_under * lr_below_gt_den

    # ── Select per sample ──────────────────────────────────────────────────────
    per_sample = torch.where(sparse_mask, sparse_penalty,
                 torch.where(medium_mask, medium_penalty,
                                          dense_penalty))

    return per_sample.mean()


# ── Training epoch ────────────────────────────────────────────────────────────
def train_epoch(model, loader, loss_fn, optimizer, scaler, device,
                lambda_cons, alpha_lr, asym_dense_over, asym_dense_under):
    model.train()
    total_loss = total_sup_hr = total_sup_lr = total_cons = 0.0

    for hr_imgs, lr_imgs, points, densities in tqdm(loader, desc="Training"):
        hr_imgs   = hr_imgs.to(device)
        lr_imgs   = lr_imgs.to(device)
        densities = densities.to(device)
        points    = [p.to(device) for p in points]

        optimizer.zero_grad()

        with autocast():
            B        = hr_imgs.size(0)
            combined = torch.cat([hr_imgs, lr_imgs], dim=0)   # (2B, C, H, W)
            pred_class, pred_density = model(combined)

            hr_class   = pred_class[:B]
            lr_class   = pred_class[B:]
            hr_density = pred_density[:B]
            lr_density = pred_density[B:]

            # ── 1. Supervised loss on HR (primary) ─────────────────────────────
            sup_hr, info_hr = loss_fn(hr_class, hr_density, densities, points)

            # ── 2. Supervised loss on LR (secondary) ───────────────────────────
            # Directly teaches the LR branch not to undershoot rather than
            # relying solely on the consistency term to pull it up.
            sup_lr, info_lr = loss_fn(lr_class, lr_density, densities, points)

            # ── 3. Personalised consistency loss ──────────────────────────────
            hr_count = hr_density.sum(dim=(1, 2, 3))
            lr_count = lr_density.sum(dim=(1, 2, 3))
            gt_count = densities.sum(dim=(1, 2, 3))

            cons_loss = consistency_loss_personalised(
                hr_count, lr_count, gt_count,
                asym_dense_over=asym_dense_over,
                asym_dense_under=asym_dense_under,
            )

            loss = sup_hr + alpha_lr * sup_lr + lambda_cons * cons_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss   += loss.item()
        total_sup_hr += info_hr["loss"].item()
        total_sup_lr += info_lr["loss"].item()
        total_cons   += cons_loss.item()

    n = len(loader)
    return total_loss / n, total_sup_hr / n, total_sup_lr / n, total_cons / n


# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, device, nwpu_root: Path):
    from utils.eval_utils import calculate_errors, sliding_window_predict

    model.eval()
    with open(nwpu_root / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()]

    pred_counts, gt_counts = [], []
    for image_id in tqdm(image_ids[:100], desc="Val (first 100)"):
        img        = Image.open(nwpu_root / "images" / f"{image_id}.jpg").convert("RGB")
        img_tensor = NORMALIZE(T.ToTensor()(img)).unsqueeze(0).to(device)
        density    = sliding_window_predict(model, img_tensor, 224, 224)
        pred_counts.append(density.sum().item())
        with open(nwpu_root / "jsons" / f"{image_id}.json") as f:
            gt_counts.append(json.load(f)["human_num"])

    return calculate_errors(np.array(pred_counts), np.array(gt_counts))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Consistency-regularized fine-tuning with per-density-bucket "
            "personalised loss targeting the observed HR/LR bias profile."
        )
    )
    parser.add_argument("--device",           type=str,   default="cuda:0")
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=8)
    parser.add_argument("--lr",               type=float, default=1e-5)
    parser.add_argument("--lambda_cons",      type=float, default=0.5,
                        help="Overall weight of the consistency loss term")
    parser.add_argument("--alpha_lr",         type=float, default=0.3,
                        help="Weight of LR supervised loss (0=off, 1=same as HR)")
    # Medium bucket weights are fixed (1.0 HR / 2.0 LR) based on bias data.
    # Dense bucket weights are tunable:
    parser.add_argument("--asym_dense_over",  type=float, default=2.0,
                        help="Dense bucket: penalty weight for HR overcounting vs GT")
    parser.add_argument("--asym_dense_under", type=float, default=5.0,
                        help="Dense bucket: penalty weight for LR undercounting vs GT")
    parser.add_argument("--min_down",         type=float, default=2.0,
                        help="Min downscale factor for the LR view")
    parser.add_argument("--max_down",         type=float, default=4.0,
                        help="Max downscale factor for the LR view")
    parser.add_argument("--num_workers",      type=int,   default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Personalised consistency-regularized fine-tuning")
    print("=" * 60)
    print(f"Device:              {device}")
    print(f"LR downscale range:  [{args.min_down}, {args.max_down}]")
    print(f"lambda_cons:         {args.lambda_cons}")
    print(f"alpha_lr:            {args.alpha_lr}")
    print()
    print("Consistency loss per density bucket:")
    print(f"  Sparse  (0–50):   HR pulled toward LR only (1.0×, one-directional)")
    print(f"  Medium (50–500):  HR-above-GT (1.0×) + LR-below-GT (2.0×)")
    print(f"  Dense  (500+):    HR-above-GT ({args.asym_dense_over}×) + LR-below-GT ({args.asym_dense_under}×)")
    print("=" * 60)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = get_model(**MODEL_CFG)
    ckpt  = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    print(f"Loaded pretrained weights from {WEIGHTS_PATH}\n")

    # ── Loss, optimiser, scaler ────────────────────────────────────────────────
    loss_fn   = DACELoss(**LOSS_CFG).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scaler = GradScaler()

    # ── Data ───────────────────────────────────────────────────────────────────
    dataset = NWPUConsistency(min_down=args.min_down, max_down=args.max_down)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    print(f"Training on {len(dataset)} images, {len(loader)} batches/epoch\n")

    # ── Output dir ─────────────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "results" / "finetune_consistency_reg"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        total_loss, sup_hr, sup_lr, cons_loss = train_epoch(
            model, loader, loss_fn, optimizer, scaler, device,
            args.lambda_cons, args.alpha_lr,
            args.asym_dense_over, args.asym_dense_under,
        )

        errors    = eval_epoch(model, device, settings.nwpu_dir)
        mae, rmse = errors["mae"], errors["rmse"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={total_loss:.4f} "
            f"(sup_hr={sup_hr:.4f}, sup_lr={sup_lr:.4f}, cons={cons_loss:.4f}) | "
            f"MAE={mae:.2f} | RMSE={rmse:.2f}"
        )

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), out_dir / "best_mae.pth")
            print(f"  → Saved best model (MAE={mae:.2f})")

        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mae":                  mae,
            "rmse":                 rmse,
            "sup_hr_loss":          sup_hr,
            "sup_lr_loss":          sup_lr,
            "cons_loss":            cons_loss,
            "args":                 vars(args),
        }, out_dir / "latest.pth")

    print(f"\nTraining done. Best MAE: {best_mae:.2f}")
    print(f"Weights saved to {out_dir}/")


if __name__ == "__main__":
    main()