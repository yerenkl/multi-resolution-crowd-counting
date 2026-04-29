"""
Evaluate a CLIP-EBC checkpoint on NWPU val at native resolution,
reporting overall MAE/RMSE and a breakdown by crowd density:
  sparse (0–50), medium (51–499), dense (500+).

Usage:
    # base pretrained weights (no --weights → uses CLIP_EBC_WEIGHTS from settings)
    uv run python entrypoints/eval_checkpoint.py --output_dir results/baseline

    # finetuned checkpoint
    uv run python entrypoints/eval_checkpoint.py --weights results/finetune_paired_hr_lr/best_mae.pth
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.clip_ebc import load_model
from src.settings import settings
from src.evaluation.runners import eval_nwpu_by_density


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to finetuned checkpoint. Omit to evaluate base pretrained weights.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to write eval_by_level.json. Defaults to the checkpoint directory.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)

    if args.weights is not None:
        weights_path = Path(args.weights)
        if not weights_path.is_absolute():
            weights_path = Path(__file__).resolve().parent.parent / weights_path
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded finetuned weights from {weights_path}")
        out_dir = Path(args.output_dir) if args.output_dir else weights_path.parent
    else:
        print("Evaluating base pretrained weights")
        out_dir = Path(args.output_dir) if args.output_dir else settings.RESULTS_DIR / "baseline"

    results = eval_nwpu_by_density(model, device)

    print("\n  NWPU val — native resolution")
    print(f"  {'Subset':<12}  {'N':>5}  {'MAE':>7}  {'RMSE':>7}")
    print("  " + "-" * 38)
    for key in ["overall", "sparse", "medium", "dense"]:
        if key not in results:
            continue
        r = results[key]
        print(f"  {key:<12}  {r['n']:>5}  {r['mae']:>7.2f}  {r['rmse']:>7.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_by_level.json"
    with open(out_path, "w") as f:
        json.dump({k: {m: float(v) for m, v in r.items()} for k, r in results.items()}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
