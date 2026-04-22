"""
Compare CLIP-EBC baseline performance across native and downscaled resolutions.

Loads pre-computed .npz result files from results/baseline/ and prints a
summary table with MAE, RMSE, bias, and per-density-bucket breakdowns.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python evaluation/compare_resolutions.py
"""

import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "baseline"


def load_results():
    native = np.load(RESULTS_DIR / "nwpu_val_native.npz")
    d2     = np.load(RESULTS_DIR / "nwpu_val_2x_down.npz")
    d4     = np.load(RESULTS_DIR / "nwpu_val_4x_down.npz")
    return native, d2, d4


def mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))

def bias(pred, gt):
    return np.mean(pred - gt)


def print_summary(native, d2, d4):
    gt = native["gt_counts"]

    rows = [
        ("native (1x)", native["pred_counts"]),
        ("2x down",     d2["pred_counts"]),
        ("4x down",     d4["pred_counts"]),
    ]

    print(f"\n{'='*52}")
    print(f"  NWPU Val: native vs downscaled (baseline CLIP-EBC)")
    print(f"{'='*52}")
    print(f"  {'Resolution':<14} {'MAE':>8} {'RMSE':>8} {'Bias':>9}")
    print(f"  {'-'*42}")
    for label, pred in rows:
        print(f"  {label:<14} {mae(pred,gt):>8.2f} {rmse(pred,gt):>8.2f} {bias(pred,gt):>+9.2f}")


def print_direction(native, d2, d4):
    gt = native["gt_counts"]

    print(f"\n  {'Resolution':<14} {'% under':>8} {'% over':>8}  {'median AE':>10}  {'max AE':>8}")
    print(f"  {'-'*52}")
    for label, data in [("native (1x)", native), ("2x down", d2), ("4x down", d4)]:
        pred = data["pred_counts"]
        ae   = np.abs(pred - gt)
        pct_under = (pred < gt).mean() * 100
        pct_over  = (pred > gt).mean() * 100
        print(f"  {label:<14} {pct_under:>7.1f}% {pct_over:>7.1f}%  {np.median(ae):>10.1f}  {np.max(ae):>8.0f}")


def print_by_density(native, d2, d4):
    gt = native["gt_counts"]

    buckets = [
        ("sparse  (0–50)",    gt < 50),
        ("medium (50–500)",  (gt >= 50) & (gt < 500)),
        ("dense  (500+)",     gt >= 500),
    ]

    print(f"\n  {'Bucket':<20} {'N':>5}  {'MAE 1x':>8} {'MAE 2x':>8} {'MAE 4x':>8}")
    print(f"  {'-'*54}")
    for label, mask in buckets:
        n    = mask.sum()
        m1   = mae(native["pred_counts"][mask], gt[mask])
        m2   = mae(d2["pred_counts"][mask],     gt[mask])
        m4   = mae(d4["pred_counts"][mask],     gt[mask])
        print(f"  {label:<20} {n:>5}  {m1:>8.2f} {m2:>8.2f} {m4:>8.2f}")


def print_correlation(native, d2, d4):
    gt     = native["gt_counts"]
    gap_2x = np.abs(native["pred_counts"] - d2["pred_counts"])
    gap_4x = np.abs(native["pred_counts"] - d4["pred_counts"])
    r2     = np.corrcoef(gt, gap_2x)[0, 1]
    r4     = np.corrcoef(gt, gap_4x)[0, 1]

    print(f"\n  Correlation: crowd size vs |HR − LR| count gap")
    print(f"    native vs 2x:  r = {r2:.3f}")
    print(f"    native vs 4x:  r = {r4:.3f}")
    print(f"  (higher = gap grows larger in denser scenes)")


def main():
    native, d2, d4 = load_results()

    print_summary(native, d2, d4)

    print(f"\n  Error direction and spread")
    print(f"  {'-'*52}")
    print_direction(native, d2, d4)

    print(f"\n  MAE by crowd density")
    print(f"  {'-'*54}")
    print_by_density(native, d2, d4)

    print_correlation(native, d2, d4)
    print()


if __name__ == "__main__":
    main()
