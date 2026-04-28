"""
Evaluate CLIP-EBC (ViT-B/16) on NWPU-Crowd val at 2x and 4x downscale.

Uses pre-saved downscaled images from settings.NWPU_DOWNSCALED_DIR.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python entrypoints/eval_nwpu_downscaled.py [--device cuda:0]
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.clip_ebc import load_model  # also puts CLIP_EBC_DIR in sys.path
from src.settings import settings
from src.evaluation.runners import eval_nwpu_downscaled
from src.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = load_model(device)

    results_dir = settings.RESULTS_DIR / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for scale in [2, 4]:
        errors = eval_nwpu_downscaled(model, device, scale)
        tag = f"{scale}x_down"
        summary[tag] = {"mae": float(errors["mae"]), "rmse": float(errors["rmse"])}
        logger.info(f"Results ({scale}x downscale): MAE={errors['mae']:.2f}  RMSE={errors['rmse']:.2f}")

    logger.info("NWPU Val Downscaled Summary:")
    for tag, err in summary.items():
        logger.info(f"  {tag:<12} MAE={err['mae']:>8.2f}  RMSE={err['rmse']:>8.2f}")

    with open(results_dir / "nwpu_val_downscaled.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
