"""
Evaluate CLIP-EBC (ViT-B/16) on NWPU-Crowd val at native resolution.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python entrypoints/eval_nwpu_native.py [--device cuda:0]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.clip_ebc import load_model  # also puts CLIP_EBC_DIR in sys.path
from src.evaluation.runners import eval_nwpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    out_dir = Path(args.path)

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device, out_dir / "results" / "best_mae.pth")
    errors = eval_nwpu(model, device)

    print(f"\n  Results (native):")
    print(f"    MAE:       {errors['mae']:.2f}")
    print(f"    RMSE:      {errors['rmse']:.2f}")
    print(f"    Avg diff:  {errors['avg_diff']:.2f}")

    results_dir = out_dir / "results" / "best_mae"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "nwpu_val_native_2.json", "w") as f:
        json.dump(
            {
                "mae": float(errors["mae"]),
                "rmse": float(errors["rmse"]),
                "avg_diff": float(errors["avg_diff"]),
            },
            f,
            indent=2,
        )

    print(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
