"""
Regenerate bias plots from an existing bias_per_image.csv without re-running inference.

Usage:
    uv run python entrypoints/replot_bias.py --csv results/dann_v2/mixed/bias/bias_per_image.csv
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _scatter_panel(ax, gts, preds, label, color, xlim=None):
    max_val = xlim if xlim else max(gts.max(), preds.max()) * 1.05
    ax.scatter(gts, preds, alpha=0.4, s=10, color=color)
    ax.plot([0, max_val], [0, max_val], "k--", lw=0.8, label="Perfect")
    ax.set_xlabel("GT count")
    ax.set_ylabel("Predicted count")
    ax.set_title(label)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv).sort_values("gt").reset_index(drop=True)
    output_dir = Path(args.csv).parent

    gts = df["gt"].values
    b_preds = df["baseline_pred"].values
    d_preds = df["dann_pred"].values
    b_errs = df["baseline_error"].values
    d_errs = df["dann_error"].values
    zoom = int(np.percentile(gts, 90))

    # --- scatter ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    _scatter_panel(axes[0, 0], gts, b_preds, "Baseline — full range", "steelblue")
    _scatter_panel(axes[0, 1], gts, d_preds, "DANN mixed — full range", "tomato")
    _scatter_panel(axes[1, 0], gts, b_preds, f"Baseline — zoomed (GT ≤ {zoom:,})", "steelblue", xlim=zoom)
    _scatter_panel(axes[1, 1], gts, d_preds, f"DANN mixed — zoomed (GT ≤ {zoom:,})", "tomato", xlim=zoom)
    fig.suptitle("Predicted vs GT count — NWPU val", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "bias_scatter.png", dpi=130)
    plt.close(fig)
    print("Saved bias_scatter.png")

    # --- error vs gt ---
    zoom_mask = gts <= zoom
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, mask, title_suffix in [
        (axes[0], np.ones(len(gts), dtype=bool), "full range"),
        (axes[1], zoom_mask, f"zoomed (GT ≤ {zoom:,})"),
    ]:
        ax.scatter(gts[mask], b_errs[mask], alpha=0.35, s=10, color="steelblue", label="Baseline")
        ax.scatter(gts[mask], d_errs[mask], alpha=0.35, s=10, color="tomato", label="DANN mixed")
        window = min(30, mask.sum() // 3)
        for errs, color in [(b_errs[mask], "steelblue"), (d_errs[mask], "tomato")]:
            rolling = np.convolve(errs, np.ones(window) / window, mode="valid")
            x_roll = gts[mask][window // 2: window // 2 + len(rolling)]
            ax.plot(x_roll, rolling, color=color, lw=2, alpha=0.9)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axvline(50, color="gray", lw=0.6, ls=":", label="sparse/medium")
        ax.axvline(500, color="gray", lw=0.6, ls="-.", label="medium/dense")
        ax.set_xlabel("GT count")
        ax.set_ylabel("Signed error (pred − gt)")
        ax.set_title(f"Signed error vs GT — {title_suffix}")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    fig.savefig(output_dir / "bias_error_vs_gt.png", dpi=130)
    plt.close(fig)
    print("Saved bias_error_vs_gt.png")

    # --- outliers ---
    print(f"\nTop 10 outliers by |dann_error|:")
    print(f"  {'image_id':<12} {'gt':>8} {'baseline_err':>14} {'dann_err':>12}")
    for _, row in df.reindex(df["dann_error"].abs().nlargest(10).index).iterrows():
        print(f"  {row['image_id']:<12} {int(row['gt']):>8} {row['baseline_error']:>+14.0f} {row['dann_error']:>+12.0f}")


if __name__ == "__main__":
    main()
