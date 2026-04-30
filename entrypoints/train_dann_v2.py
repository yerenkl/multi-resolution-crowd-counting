"""
DANN v2: resolution-adversarial training using pre-saved downscaled NWPU images.

Unlike v1 (which degraded 224px tensor crops on-the-fly), v2 loads real
downscaled images from disk so the LR domain matches the eval distribution.

Usage:
    uv run python entrypoints/train_dann_v2.py --lr_scales 2        # 2x only
    uv run python entrypoints/train_dann_v2.py --lr_scales 4        # 4x only
    uv run python entrypoints/train_dann_v2.py --lr_scales 2 4      # mixed
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from src.models.clip_ebc import load_model
from src.datasets import NWPUPairedHRLR
from src.datasets.transforms import PairedHRLRTransform
from src.training.collate import nwpu_paired_hr_lr_collate_fn
from src.training.loops import eval_epoch
from src.training.metrics import MetricsLogger
from src.dann import DANNModel, ganin_alpha_schedule, dann_train_epoch_v2
from src.settings import settings
from src.logger import get_logger

from losses import DACELoss

logger = get_logger(__name__)

LOSS_CFG = dict(
    bins=[[0, 0], [1, 1], [2, 2], [3, 3], [4, float("inf")]],
    reduction=8,
    weight_count_loss=1.0,
    count_loss="dmcount",
    input_size=224,
)


def main():
    parser = argparse.ArgumentParser(description="DANN v2 — pre-saved downscaled LR")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_disc", type=float, default=1e-4)
    parser.add_argument("--dann_weight", type=float, default=1.0)
    parser.add_argument(
        "--lr_scales", type=int, nargs="+", default=[2, 4],
        help="Which downscale factors to use as LR domain (2, 4, or both)",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_limit", type=int, default=500)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    crowd_model = load_model(device)

    model = DANNModel(
        crowd_model=crowd_model,
        feature_dim=768,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    loss_fn = DACELoss(**LOSS_CFG).to(device)

    crowd_params = [p for p in crowd_model.parameters() if p.requires_grad]
    disc_params = list(model.domain_classifier.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": crowd_params, "lr": args.lr, "weight_decay": 1e-4},
            {"params": disc_params, "lr": args.lr_disc, "weight_decay": 1e-4},
        ]
    )
    scaler = GradScaler()

    transform = PairedHRLRTransform(output_size=224, crop_scale=(1.0, 2.0))
    dataset = NWPUPairedHRLR(
        split="train",
        lr_scales=tuple(args.lr_scales),
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=nwpu_paired_hr_lr_collate_fn,
    )

    scale_tag = "_".join(f"{s}x" for s in sorted(args.lr_scales))
    logger.info(f"Training on {len(dataset)} paired images, {len(loader)} batches/epoch")
    logger.info(f"LR scales: {args.lr_scales} (tag: {scale_tag})")
    logger.info(f"DANN config: dann_weight={args.dann_weight}")
    logger.info(f"Discriminator: hidden_dim={args.hidden_dim}, dropout={args.dropout}, lr={args.lr_disc}")

    metrics = MetricsLogger(
        experiment=f"dann_v2_{scale_tag}",
        args=args,
        fieldnames=["epoch", "loss", "task_loss", "domain_loss", "alpha", "mae", "rmse"],
        base_dir=settings.DANN_CHECKPOINTS_DIR,
    )
    logger.info(f"Run directory: {metrics.run_dir}")

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        alpha = ganin_alpha_schedule(epoch, args.epochs)

        losses = dann_train_epoch_v2(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            alpha=alpha,
            dann_weight=args.dann_weight,
        )

        errors = eval_epoch(model.crowd_model, device, limit=args.eval_limit)
        mae, rmse = errors["mae"], errors["rmse"]

        metrics.log({"epoch": epoch, **losses, "mae": mae, "rmse": rmse})

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={losses['loss']:.4f} task={losses['task_loss']:.4f} "
            f"domain={losses['domain_loss']:.4f} alpha={alpha:.3f} | "
            f"MAE={mae:.2f} RMSE={rmse:.2f}"
        )

        if mae < best_mae:
            best_mae = mae
            torch.save(model.crowd_model.state_dict(), metrics.run_dir / "best_mae.pth")
            logger.success(f"Saved best model", MAE=f"{mae:.2f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "crowd_model_state_dict": model.crowd_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "alpha": alpha,
                "mae": mae,
                "rmse": rmse,
                "losses": losses,
            },
            metrics.run_dir / "latest.pth",
        )

    logger.success(f"Training done. Best MAE: {best_mae:.2f}")
    logger.info(f"Weights saved to {metrics.run_dir}/")


if __name__ == "__main__":
    main()
