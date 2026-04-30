"""
Evaluate CLIP-EBC (ViT-B/16) on NWPU-Crowd val at 2x and 4x downscale.

Uses pre-saved downscaled images from settings.NWPU_DOWNSCALED_DIR.

Usage:
    # base pretrained weights
    uv run python entrypoints/eval_nwpu_downscaled.py

    # finetuned checkpoint
    uv run python entrypoints/eval_nwpu_downscaled.py --weights path/to/best_mae.pth
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.clip_ebc import load_model  # also puts CLIP_EBC_DIR in sys.path
from src.settings import settings
from src.evaluation.runners import eval_nwpu_downscaled
from src.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to finetuned checkpoint. Omit to evaluate base pretrained weights.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to write results. Defaults to checkpoint dir or results/baseline.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = load_model(device)

    if args.weights is not None:
        weights_path = Path(args.weights)
        if not weights_path.is_absolute():
            weights_path = Path(__file__).resolve().parent.parent / weights_path
        try:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logger.error(f"Failed to load weights from {weights_path}: {e}")
            raise
        logger.info(f"Loaded finetuned weights from {weights_path}")
        results_dir = Path(args.output_dir) if args.output_dir else weights_path.parent
    else:
        logger.info("Evaluating base pretrained weights")
        results_dir = Path(args.output_dir) if args.output_dir else settings.RESULTS_DIR / "baseline"

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
