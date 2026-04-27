"""
Evaluate CLIP-EBC (ViT-B/16) on supervisor's Zoom Pairs.

Measures HR vs LR count consistency — no ground truth, comparing predictions
on the same scene at two different optical resolutions.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python entrypoints/eval_zoom_pairs.py [--device cuda:0]
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.clip_ebc import load_model  # also puts CLIP_EBC_DIR in sys.path
from src.settings import settings
from src.evaluation.runners import eval_zoom_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    out_dir = Path(args.path)


    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device, out_dir / "results/best_mae.pth")
    results = eval_zoom_pairs(model, device)

    abs_diffs = [r["abs_diff"] for r in results]
    ratios = [r["ratio"] for r in results if r["ratio"] != float("inf")]
    hr_counts = [r["hr_count"] for r in results]
    lr_counts = [r["lr_count"] for r in results]

    print(f"\n{'='*50}")
    print(f"  Zoom Pairs Results")
    print(f"{'='*50}")
    print(f"  Mean HR count:        {np.mean(hr_counts):.1f}")
    print(f"  Mean LR count:        {np.mean(lr_counts):.1f}")
    print(f"  Mean |HR - LR|:       {np.mean(abs_diffs):.1f}")
    print(f"  Median |HR - LR|:     {np.median(abs_diffs):.1f}")
    print(f"  Mean HR/LR ratio:     {np.mean(ratios):.3f}")
    print(f"  Median HR/LR ratio:   {np.median(ratios):.3f}")

    worst = sorted(results, key=lambda r: r["abs_diff"], reverse=True)[:5]
    print(f"\n  Worst 5 pairs by |HR - LR|:")
    for r in worst:
        print(f"    Pair {r['pair']:>3s}: HR={r['hr_count']:.0f}, LR={r['lr_count']:.0f}, "
              f"diff={r['abs_diff']:.0f}, ratio={r['ratio']:.2f}")

    best = sorted(results, key=lambda r: r["abs_diff"])[:5]
    print(f"\n  Best 5 pairs by |HR - LR|:")
    for r in best:
        print(f"    Pair {r['pair']:>3s}: HR={r['hr_count']:.0f}, LR={r['lr_count']:.0f}, "
              f"diff={r['abs_diff']:.0f}, ratio={r['ratio']:.2f}")

    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "zoom_pairs.json", "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "mean_abs_diff": float(np.mean(abs_diffs)),
        "median_abs_diff": float(np.median(abs_diffs)),
        "mean_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "mean_hr_count": float(np.mean(hr_counts)),
        "mean_lr_count": float(np.mean(lr_counts)),
    }
    with open(results_dir / "zoom_pairs_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
