"""
Resolution-augmented fine-tuning of CLIP-EBC (ViT-B/16) on NWPU-Crowd.

During training each crop is randomly downscaled by 1x-4x before being fed
to the model, forcing it to learn resolution-invariant crowd counting.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python entrypoints/train_finetune_res_aug.py [--device cuda:0]
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
from src.datasets.transforms import Compose, RandomCrop, ResolutionAugment, RandomHorizontalFlip, ToTensor, Normalize
from src.training.loops import train_epoch, eval_epoch
from src.training.collate import nwpu_train_collate_fn

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
    parser.add_argument("--min_scale", type=float, default=1.0,
                        help="Min downscale factor for resolution augmentation")
    parser.add_argument("--max_scale", type=float, default=4.0,
                        help="Max downscale factor for resolution augmentation")
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
    scaler = GradScaler()

    transform = Compose([
        RandomCrop(size=224, scale=(1.0, 1.0)),
        # ResolutionAugment(min_scale=args.min_scale, max_scale=args.max_scale),
        # RandomHorizontalFlip(),
        ToTensor(),
        Normalize(),
    ])
    dataset = NWPU(split="train", transform=transform,
                   image_path="/dtu/blackhole/0a/224426/NWPU_downscaled/4x/images",
                   jsons_path="/dtu/blackhole/0a/224426/NWPU_downscaled/jsons/4x",
                   txt_path="/dtu/blackhole/02/137570/MultiRes/NWPU_crowd"
                   )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=nwpu_train_collate_fn,
    )
    print(f"Training on {len(dataset)} images, {len(loader)} batches/epoch")

    out_dir = settings.RESULTS_DIR / "finetune_resolution_aug"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loader, loss_fn, optimizer, scaler, device)
        errors = eval_epoch(model, device)
        mae, rmse = errors["mae"], errors["rmse"]

        print(f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | MAE={mae:.2f} | RMSE={rmse:.2f}")

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
        }, out_dir / "latest.pth")

    print(f"\nTraining done. Best MAE: {best_mae:.2f}")
    print(f"Weights saved to {out_dir}/")


if __name__ == "__main__":
    main()
