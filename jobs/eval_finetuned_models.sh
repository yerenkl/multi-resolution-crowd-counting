#!/bin/bash
#BSUB -q gpua100
#BSUB -J eval_finetuned
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o jobs/logs/eval_finetuned_%J.out
#BSUB -e jobs/logs/eval_finetuned_%J.err
set -euo pipefail

# ── Set these to your fine-tuned checkpoint paths ────────────────────────────
BILINEAR="results/finetune_bilinear/<timestamp>/best_mae.pth"
BICUBIC="results/finetune_bicubic/<timestamp>/best_mae.pth"
LANCZOS="results/finetune_lanczos/<timestamp>/best_mae.pth"
NEAREST="results/finetune_nearest/<timestamp>/best_mae.pth"
# ─────────────────────────────────────────────────────────────────────────────

uv sync

uv run python entrypoints/eval_finetuned_models.py \
    --bilinear "$BILINEAR" \
    --bicubic  "$BICUBIC"  \
    --lanczos  "$LANCZOS"  \
    --nearest  "$NEAREST"  \
    --device cuda:0
