"""
Full evaluation of DANN checkpoint with TTA.

Runs all eval modes on both baseline and DANN, saving results side by side:
  1. Native (no TTA)
  2. Native with TTA (multi-scale: 1.0, 0.75, 0.5)
  3. Downscaled 2x and 4x
  4. Zoom pairs consistency

Usage (on HPC):
    uv run python entrypoints/eval_dann_tta.py --dann_weights /work3/s225224/.../best_mae.pth
    uv run python entrypoints/eval_dann_tta.py --dann_weights /work3/s225224/.../best_mae.pth --output_dir results/dann_tta
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.clip_ebc import load_model
from src.settings import settings
from src.evaluation.runners import (
    eval_nwpu,
    eval_nwpu_tta,
    eval_nwpu_downscaled,
    eval_nwpu_by_density,
    eval_zoom_pairs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_weights(model, weights_path):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    return model


def summarize_zoom_pairs(pairs):
    diffs = [p["abs_diff"] for p in pairs]
    ratios = [p["ratio"] for p in pairs]
    import numpy as np
    return {
        "mean_abs_diff": float(np.mean(diffs)),
        "median_abs_diff": float(np.median(diffs)),
        "mean_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
    }


def run_all_evals(model, device, label):
    log.info(f"\n{'='*60}")
    log.info(f"Evaluating: {label}")
    log.info(f"{'='*60}")
    results = {}

    log.info("  [1/5] Native (no TTA)...")
    native = eval_nwpu(model, device)
    results["native"] = {"mae": float(native["mae"]), "rmse": float(native["rmse"])}
    log.info(f"         MAE={native['mae']:.2f}  RMSE={native['rmse']:.2f}")

    log.info("  [2/5] Native with TTA (scales: 1.0, 0.75, 0.5)...")
    tta = eval_nwpu_tta(model, device)
    results["native_tta"] = {"mae": float(tta["mae"]), "rmse": float(tta["rmse"])}
    log.info(f"         MAE={tta['mae']:.2f}  RMSE={tta['rmse']:.2f}")

    log.info("  [3/5] By density bucket...")
    by_density = eval_nwpu_by_density(model, device)
    results["by_density"] = {
        k: {m: float(v) for m, v in r.items()} for k, r in by_density.items()
    }
    for bucket in ["sparse", "medium", "dense"]:
        r = by_density[bucket]
        log.info(f"         {bucket:<8} MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}  n={r['n']}")

    log.info("  [4/5] Downscaled 2x and 4x...")
    results["downscaled"] = {}
    for scale in [2, 4]:
        ds = eval_nwpu_downscaled(model, device, scale)
        results["downscaled"][f"{scale}x"] = {"mae": float(ds["mae"]), "rmse": float(ds["rmse"])}
        log.info(f"         {scale}x  MAE={ds['mae']:.2f}  RMSE={ds['rmse']:.2f}")

    log.info("  [5/5] Zoom pairs...")
    pairs = eval_zoom_pairs(model, device)
    results["zoom_pairs"] = summarize_zoom_pairs(pairs)
    zp = results["zoom_pairs"]
    log.info(f"         mean |HR-LR|={zp['mean_abs_diff']:.1f}  median={zp['median_abs_diff']:.1f}")

    return results


def print_comparison(baseline, dann):
    log.info(f"\n{'='*60}")
    log.info("COMPARISON: Baseline vs DANN")
    log.info(f"{'='*60}")
    log.info(f"  {'Condition':<20} {'Baseline':>10} {'DANN':>10} {'Delta':>10}")
    log.info(f"  {'-'*50}")

    rows = [
        ("Native", baseline["native"]["mae"], dann["native"]["mae"]),
        ("Native (TTA)", baseline["native_tta"]["mae"], dann["native_tta"]["mae"]),
        ("2x down", baseline["downscaled"]["2x"]["mae"], dann["downscaled"]["2x"]["mae"]),
        ("4x down", baseline["downscaled"]["4x"]["mae"], dann["downscaled"]["4x"]["mae"]),
        ("Zoom |HR-LR|", baseline["zoom_pairs"]["mean_abs_diff"], dann["zoom_pairs"]["mean_abs_diff"]),
    ]
    for label, b, d in rows:
        delta = d - b
        sign = "+" if delta > 0 else ""
        log.info(f"  {label:<20} {b:>10.2f} {d:>10.2f} {sign}{delta:>9.2f}")


def main():
    parser = argparse.ArgumentParser(description="Full DANN evaluation with TTA")
    parser.add_argument("--dann_weights", type=str, required=True,
                        help="Path to DANN best_mae.pth (crowd model state dict)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Defaults to same dir as dann_weights.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    dann_path = Path(args.dann_weights)
    output_dir = Path(args.output_dir) if args.output_dir else dann_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading baseline model (pretrained CLIP-EBC)...")
    baseline_model = load_model(device)
    baseline_results = run_all_evals(baseline_model, device, "Baseline (pretrained)")

    log.info("Loading DANN model...")
    dann_model = load_model(device)
    load_weights(dann_model, dann_path)
    dann_results = run_all_evals(dann_model, device, f"DANN ({dann_path.name})")

    print_comparison(baseline_results, dann_results)

    combined = {"baseline": baseline_results, "dann": dann_results}
    out_path = output_dir / "eval_dann_tta.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    log.info(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
