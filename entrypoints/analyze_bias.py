"""
Per-image bias analysis: baseline vs DANN.

For each val image records (image_id, gt, baseline_pred, dann_pred) and checks
whether DANN systematically under/overcounts relative to baseline, especially
at high crowd densities.

Outputs:
  <output_dir>/bias_per_image.csv   — raw per-image numbers
  <output_dir>/bias_summary.json    — mean signed error by density bucket
  <output_dir>/bias_scatter.png     — pred vs GT scatter for both models
  <output_dir>/bias_error_vs_gt.png — signed error vs GT count

Usage:
    uv run python entrypoints/analyze_bias.py \
        --dann_weights /work3/s225224/.../best_mae.pth \
        --output_dir results/dann_v2/mixed/bias
"""

import sys
import json
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.clip_ebc import load_model, NORMALIZE
from src.settings import settings
from src.evaluation.inference import predict_count

BUCKETS = {"sparse": (0, 50), "medium": (51, 499), "dense": (500, 10**9)}


def bucket(gt):
    if gt <= 50:
        return "sparse"
    if gt < 500:
        return "medium"
    return "dense"


def load_weights(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=True)
    return model


def run_inference(model, device, nwpu_root):
    to_tensor = T.ToTensor()
    with open(nwpu_root / "val.txt") as f:
        image_ids = [l.strip().split()[0] for l in f if l.strip()]

    records = []
    for image_id in tqdm(image_ids, desc="Inferring"):
        img = Image.open(nwpu_root / "images" / f"{image_id}.jpg").convert("RGB")
        img_tensor = NORMALIZE(to_tensor(img))
        with open(nwpu_root / "jsons" / f"{image_id}.json") as f:
            gt = json.load(f)["human_num"]
        pred = predict_count(model, img_tensor, device)
        records.append((image_id, gt, pred))
    return records


def save_csv(path, baseline_records, dann_records):
    baseline_map = {r[0]: r[2] for r in baseline_records}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "gt", "baseline_pred", "dann_pred",
                    "baseline_error", "dann_error", "bucket"])
        for image_id, gt, dann_pred in dann_records:
            b_pred = baseline_map[image_id]
            w.writerow([
                image_id, gt,
                round(b_pred, 2), round(dann_pred, 2),
                round(b_pred - gt, 2), round(dann_pred - gt, 2),
                bucket(gt),
            ])


def compute_summary(baseline_records, dann_records):
    baseline_map = {r[0]: r[2] for r in baseline_records}
    by_bucket = {b: {"baseline_errors": [], "dann_errors": [], "gts": []} for b in BUCKETS}
    by_bucket["overall"] = {"baseline_errors": [], "dann_errors": [], "gts": []}

    for image_id, gt, dann_pred in dann_records:
        b_pred = baseline_map[image_id]
        b_err = b_pred - gt
        d_err = dann_pred - gt
        b = bucket(gt)
        for key in (b, "overall"):
            by_bucket[key]["baseline_errors"].append(b_err)
            by_bucket[key]["dann_errors"].append(d_err)
            by_bucket[key]["gts"].append(gt)

    summary = {}
    for name, data in by_bucket.items():
        be = np.array(data["baseline_errors"])
        de = np.array(data["dann_errors"])
        summary[name] = {
            "n": len(be),
            "baseline": {
                "mean_signed_error": round(float(be.mean()), 2),
                "mae": round(float(np.abs(be).mean()), 2),
            },
            "dann": {
                "mean_signed_error": round(float(de.mean()), 2),
                "mae": round(float(np.abs(de).mean()), 2),
            },
        }
    return summary


def print_summary(summary):
    print(f"\n{'='*65}")
    print(f"  {'Bucket':<10} {'N':>5}  {'Baseline MSE':>14}  {'DANN MSE':>10}  {'Delta':>8}")
    print(f"  {'-'*63}")
    for name in ("sparse", "medium", "dense", "overall"):
        s = summary[name]
        b_mse = s["baseline"]["mean_signed_error"]
        d_mse = s["dann"]["mean_signed_error"]
        delta = d_mse - b_mse
        sign = "+" if delta > 0 else ""
        print(f"  {name:<10} {s['n']:>5}  {b_mse:>+14.1f}  {d_mse:>+10.1f}  {sign}{delta:>7.1f}")
    print(f"{'='*65}")
    print("  MSE = mean signed error (pred − gt). Negative = undercounting.\n")


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


def plot_scatter(output_dir, baseline_records, dann_records):
    baseline_map = {r[0]: r[2] for r in baseline_records}
    gts, b_preds, d_preds = [], [], []
    for image_id, gt, d_pred in dann_records:
        gts.append(gt)
        b_preds.append(baseline_map[image_id])
        d_preds.append(d_pred)

    gts = np.array(gts)
    b_preds = np.array(b_preds)
    d_preds = np.array(d_preds)
    zoom = int(np.percentile(gts, 90))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    _scatter_panel(axes[0, 0], gts, b_preds, "Baseline — full range", "steelblue")
    _scatter_panel(axes[0, 1], gts, d_preds, "DANN mixed — full range", "tomato")
    _scatter_panel(axes[1, 0], gts, b_preds, f"Baseline — zoomed (GT ≤ {zoom:,})", "steelblue", xlim=zoom)
    _scatter_panel(axes[1, 1], gts, d_preds, f"DANN mixed — zoomed (GT ≤ {zoom:,})", "tomato", xlim=zoom)

    fig.suptitle("Predicted vs GT count — NWPU val", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "bias_scatter.png", dpi=130)
    plt.close(fig)
    print(f"Saved bias_scatter.png")


def plot_error_vs_gt(output_dir, baseline_records, dann_records):
    baseline_map = {r[0]: r[2] for r in baseline_records}
    image_ids_ordered, gts, b_errs, d_errs = [], [], [], []
    for image_id, gt, d_pred in dann_records:
        image_ids_ordered.append(image_id)
        gts.append(gt)
        b_errs.append(baseline_map[image_id] - gt)
        d_errs.append(d_pred - gt)

    gts = np.array(gts)
    b_errs = np.array(b_errs)
    d_errs = np.array(d_errs)
    sort_idx = np.argsort(gts)
    gts_s = gts[sort_idx]
    b_errs_s = b_errs[sort_idx]
    d_errs_s = d_errs[sort_idx]
    ids_s = [image_ids_ordered[i] for i in sort_idx]

    zoom_gt = int(np.percentile(gts, 90))
    zoom_mask = gts_s <= zoom_gt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, mask, title_suffix in [
        (axes[0], np.ones(len(gts_s), dtype=bool), "full range"),
        (axes[1], zoom_mask, f"zoomed (GT ≤ {zoom_gt:,})"),
    ]:
        ax.scatter(gts_s[mask], b_errs_s[mask], alpha=0.35, s=10, color="steelblue", label="Baseline")
        ax.scatter(gts_s[mask], d_errs_s[mask], alpha=0.35, s=10, color="tomato", label="DANN mixed")

        window = min(30, mask.sum() // 3)
        for errs, color in [(b_errs_s[mask], "steelblue"), (d_errs_s[mask], "tomato")]:
            rolling = np.convolve(errs, np.ones(window) / window, mode="valid")
            x_rolling = gts_s[mask][window // 2: window // 2 + len(rolling)]
            ax.plot(x_rolling, rolling, color=color, lw=2, alpha=0.9)

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
    print(f"Saved bias_error_vs_gt.png")

    # print top outliers
    print("\nTop 10 outliers by |dann_error|:")
    print(f"  {'image_id':<12} {'gt':>8} {'baseline_err':>14} {'dann_err':>12}")
    abs_d = np.abs(d_errs)
    for i in np.argsort(abs_d)[::-1][:10]:
        print(f"  {image_ids_ordered[i]:<12} {gts[i]:>8} {b_errs[i]:>+14.0f} {d_errs[i]:>+12.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dann_weights", required=True)
    parser.add_argument("--output_dir", type=str, default="results/dann_v2/mixed/bias")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nwpu_root = settings.nwpu_dir

    print("Loading baseline model...")
    baseline = load_model(device)
    baseline.eval()
    print("Running baseline inference...")
    baseline_records = run_inference(baseline, device, nwpu_root)

    print("Loading DANN model...")
    dann = load_model(device)
    load_weights(dann, args.dann_weights)
    dann.eval()
    print("Running DANN inference...")
    dann_records = run_inference(dann, device, nwpu_root)

    save_csv(output_dir / "bias_per_image.csv", baseline_records, dann_records)
    print(f"Saved bias_per_image.csv")

    summary = compute_summary(baseline_records, dann_records)
    with open(output_dir / "bias_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)

    plot_scatter(output_dir, baseline_records, dann_records)
    plot_error_vs_gt(output_dir, baseline_records, dann_records)
    print(f"\nAll outputs in {output_dir}/")


if __name__ == "__main__":
    main()
