"""
Hard-patch, paired HR/LR fine-tuning of CLIP-EBC (ViT-B/16) on NWPU-Crowd.

Each training sample mines a dense crop candidate, keeps some random crops to
avoid overfitting, creates HR/LR views of the same patch, and optimizes CLIP-EBC
with supervised density loss plus a small one-way HR->LR consistency term.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python entrypoints/train_finetune_paired_hr_lr.py [--device cuda:0]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from src.models.clip_ebc import load_model  # also puts CLIP_EBC_DIR in sys.path
from src.settings import settings
from src.datasets import NWPU
from src.datasets.transforms_v2 import HardPatchPairTransform
from src.training.loops import train_paired_epoch, eval_epoch
from src.training.collate import nwpu_paired_train_collate_fn

from losses import DACELoss  # CLIP-EBC's losses package (in sys.path after clip_ebc import)

LOSS_CFG = dict(
    bins=[[0, 0], [1, 1], [2, 2], [3, 3], [4, float("inf")]],
    reduction=8,
    weight_count_loss=1.0,
    count_loss="dmcount",
    input_size=224,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min_scale", type=float, default=2.0,
                        help="Min downscale factor for the LR partner view")
    parser.add_argument("--max_scale", type=float, default=4.0,
                        help="Max downscale factor for the LR partner view")
    parser.add_argument("--crop_scale_min", type=float, default=1.0,
                        help="Minimum crop side length as a multiple of input size")
    parser.add_argument("--crop_scale_max", type=float, default=2.0,
                        help="Maximum crop side length as a multiple of input size")
    parser.add_argument("--random_patch_prob", type=float, default=0.25,
                        help="Fraction of random patches kept alongside hard-mined patches")
    parser.add_argument("--hard_candidates", type=int, default=6,
                        help="How many random crop candidates to score per image")
    parser.add_argument("--hard_weight_max", type=float, default=4.0,
                        help="Upper cap for hard-patch weighting")
    parser.add_argument("--consistency_weight", type=float, default=0.05,
                        help="Weight on one-way HR->LR count consistency")
    parser.add_argument("--hard_count_weight", type=float, default=0.25,
                        help="Weight on the extra hard-patch count loss")
    parser.add_argument("--density_temperature", type=float, default=20.0,
                        help="Controls how quickly density weighting saturates")
    parser.add_argument("--far_field_bonus", type=float, default=0.25,
                        help="How much upper-image crops are favored during hard mining")
    parser.add_argument("--val_limit", type=int, default=100,
                        help="How many NWPU val images to use for fast model selection")
    parser.add_argument("--output_subdir", type=str, default="finetune_paired_hr_lr")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)

    loss_fn = DACELoss(**LOSS_CFG).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    transform = HardPatchPairTransform(
        output_size=LOSS_CFG["input_size"],
        crop_scale=(args.crop_scale_min, args.crop_scale_max),
        random_patch_prob=args.random_patch_prob,
        num_candidates=args.hard_candidates,
        hard_weight_max=args.hard_weight_max,
        density_temperature=args.density_temperature,
        far_field_bonus=args.far_field_bonus,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        pre_blur=True,
        add_noise=False,
        hflip_p=0.5,
    )
    dataset = NWPU(split="train", transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=nwpu_paired_train_collate_fn,
    )
    print(f"Training on {len(dataset)} images, {len(loader)} batches/epoch")

    out_dir = settings.RESULTS_DIR / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = train_paired_epoch(
            model,
            loader,
            loss_fn,
            optimizer,
            scaler,
            device,
            consistency_weight=args.consistency_weight,
            hard_count_weight=args.hard_count_weight,
        )
        errors = eval_epoch(model, device, limit=args.val_limit)
        mae, rmse = errors["mae"], errors["rmse"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_stats['loss']:.4f} | "
            f"hr={train_stats['loss_hr']:.4f} | "
            f"lr={train_stats['loss_lr']:.4f} | "
            f"hard={train_stats['hard_count_loss']:.4f} | "
            f"cons={train_stats['consistency_loss']:.4f} | "
            f"hard_frac={train_stats['hard_fraction']:.2f} | "
            f"patch_w={train_stats['mean_patch_weight']:.2f} | "
            f"MAE={mae:.2f} | RMSE={rmse:.2f}"
        )

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), out_dir / "best_mae.pth")
            print(f"  → Saved best model (MAE={mae:.2f})")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mae": mae,
            "rmse": rmse,
            "train_stats": train_stats,
            "config": vars(args),
        }, out_dir / "latest.pth")

    print(f"\nTraining done. Best MAE: {best_mae:.2f}")
    print(f"Weights saved to {out_dir}/")


if __name__ == "__main__":
    main()
