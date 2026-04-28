"""
Evaluate CLIP-EBC (ViT-B/16) on NWPU-Crowd val at native resolution.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python entrypoints/eval_nwpu_native.py [--device cuda:0]
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.clip_ebc import load_model  # also puts CLIP_EBC_DIR in sys.path
from src.settings import settings
from src.evaluation.runners import eval_nwpu
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
    errors = eval_nwpu(model, device)

    logger.success(f"Results (native): MAE={errors['mae']:.2f}  RMSE={errors['rmse']:.2f}")

    results_dir = settings.RESULTS_DIR / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "nwpu_val_native.json", "w") as f:
        json.dump({"mae": float(errors["mae"]), "rmse": float(errors["rmse"])}, f, indent=2)
    logger.info(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
