"""
eval_finetuned_models.py

For each fine-tuned model (one per downscaling method), run prediction on
three versions of every zoom pair:

  HR          — the full-resolution capture (upper bound / reference)
  synthetic LR — HR center-cropped to LR aspect, then downscaled to exact LR
                 pixel dimensions using the model's own training method
  real LR     — the true low-resolution camera capture

Reports how far apart those three predictions are, per model and across pairs.

Usage:
    uv run python entrypoints/eval_finetuned_models.py \\
        --bilinear results/finetune_bilinear/<timestamp>/best_mae.pth \\
        --bicubic  results/finetune_bicubic/<timestamp>/best_mae.pth \\
        --lanczos  results/finetune_lanczos/<timestamp>/best_mae.pth \\
        --nearest  results/finetune_nearest/<timestamp>/best_mae.pth \\
        [--device cuda:0]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.clip_ebc import load_model, NORMALIZE
from src.settings import settings
from src.evaluation.inference import predict_count
from src.logger import get_logger

logger = get_logger(__name__)

METHODS = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic":  Image.Resampling.BICUBIC,
    "lanczos":  Image.Resampling.LANCZOS,
    "nearest":  Image.Resampling.NEAREST,
}


def _to_tensor(img: Image.Image) -> torch.Tensor:
    return NORMALIZE(T.ToTensor()(img))


def center_crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = img.size
    target_ar = target_w / target_h
    if abs(w / h - target_ar) < 1e-3:
        return img
    if w / h > target_ar:
        new_w = int(round(h * target_ar))
        x0 = (w - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, h))
    new_h = int(round(w / target_ar))
    y0 = (h - new_h) // 2
    return img.crop((0, y0, w, y0 + new_h))


def eval_model(model, device, resampler, pair_dirs) -> list[dict]:
    """Run one model on all pairs. Returns per-pair dicts with hr/synth/real counts."""
    results = []
    for pair_dir in tqdm(pair_dirs, leave=False):
        idx = pair_dir.name
        try:
            hr_img = Image.open(pair_dir / f"{idx}_hr.jpg").convert("RGB")
            lr_img = Image.open(pair_dir / f"{idx}_lr.jpg").convert("RGB")
        except Exception as exc:
            logger.warning(f"Pair {idx}: {exc} — skipping")
            continue

        lr_w, lr_h = lr_img.size
        hr_crop = center_crop_to_aspect(hr_img, lr_w, lr_h)
        synth_lr = hr_crop.resize((lr_w, lr_h), resampler)

        results.append({
            "pair":     idx,
            "hr":       predict_count(model, _to_tensor(hr_img),   device),
            "synth_lr": predict_count(model, _to_tensor(synth_lr), device),
            "real_lr":  predict_count(model, _to_tensor(lr_img),   device),
        })
    return results


def summarise(results: list[dict]) -> dict:
    hr       = np.array([r["hr"]       for r in results])
    synth_lr = np.array([r["synth_lr"] for r in results])
    real_lr  = np.array([r["real_lr"]  for r in results])
    return {
        "mean_count_hr":       float(np.mean(hr)),
        "mean_count_synth_lr": float(np.mean(synth_lr)),
        "mean_count_real_lr":  float(np.mean(real_lr)),
        # key gaps — how different are the three views?
        "mae_synth_vs_real":   float(np.mean(np.abs(synth_lr - real_lr))),
        "mae_hr_vs_real":      float(np.mean(np.abs(hr       - real_lr))),
        "mae_hr_vs_synth":     float(np.mean(np.abs(hr       - synth_lr))),
        # bias
        "bias_synth_vs_real":  float(np.mean(synth_lr - real_lr)),
        "bias_hr_vs_real":     float(np.mean(hr       - real_lr)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bilinear", required=True, help="Checkpoint for bilinear-finetuned model")
    parser.add_argument("--bicubic",  required=True, help="Checkpoint for bicubic-finetuned model")
    parser.add_argument("--lanczos",  required=True, help="Checkpoint for lanczos-finetuned model")
    parser.add_argument("--nearest",  required=True, help="Checkpoint for nearest-finetuned model")
    parser.add_argument("--data",   default=str(settings.zoom_pairs_dir))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_dir  = Path(args.data)
    pair_dirs = sorted(
        [p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    logger.info(f"Found {len(pair_dirs)} pairs")

    checkpoints = {
        "bilinear": Path(args.bilinear),
        "bicubic":  Path(args.bicubic),
        "lanczos":  Path(args.lanczos),
        "nearest":  Path(args.nearest),
    }

    all_results  = {}
    all_summaries = {}

    for method, ckpt_path in checkpoints.items():
        logger.info(f"\nLoading {method} model from {ckpt_path}")
        model = load_model(device, ckpt_path=ckpt_path)
        model.eval()

        per_pair = eval_model(model, device, METHODS[method], pair_dirs)
        summary  = summarise(per_pair)
        all_results[method]   = per_pair
        all_summaries[method] = summary

        logger.info(
            f"  {method:<10}  "
            f"synth_vs_real MAE={summary['mae_synth_vs_real']:.1f}  "
            f"hr_vs_real MAE={summary['mae_hr_vs_real']:.1f}  "
            f"hr_vs_synth MAE={summary['mae_hr_vs_synth']:.1f}"
        )

        del model
        torch.cuda.empty_cache()

    # ── Ranking: which model's synthetic LR is closest to real LR ────────────
    logger.info("\nRanking by synth_vs_real MAE (lower = synthetic LR closer to real LR):")
    logger.info(f"  {'Model':<10}  {'synth↔real':>10}  {'hr↔real':>8}  {'hr↔synth':>9}  {'bias synth↔real':>16}")
    logger.info(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*16}")
    for method, s in sorted(all_summaries.items(), key=lambda kv: kv[1]["mae_synth_vs_real"]):
        logger.info(
            f"  {method:<10}  {s['mae_synth_vs_real']:>10.1f}  "
            f"{s['mae_hr_vs_real']:>8.1f}  "
            f"{s['mae_hr_vs_synth']:>9.1f}  "
            f"{s['bias_synth_vs_real']:>+16.1f}"
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = settings.RESULTS_DIR / "finetuned_model_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_pair.json",  "w") as f:
        json.dump(all_results,   f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
