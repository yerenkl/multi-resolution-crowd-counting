#!/bin/bash
#BSUB -q gpua100
#BSUB -W 2:00
#BSUB -J eval_baseline
#BSUB -o jobs/logs/eval_baseline_%J.out
#BSUB -e jobs/logs/eval_baseline_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail

uv sync
mkdir -p jobs/logs

echo "============================================================"
echo "Evaluating base pretrained CLIP-EBC weights on NWPU val"
echo "============================================================"

uv run python entrypoints/eval_checkpoint.py \
    --device     cuda:0 \
    --weights    results/finetune_paired_hr_lr/best_mae.pth \
    --output_dir results/baseline

echo "Done."
