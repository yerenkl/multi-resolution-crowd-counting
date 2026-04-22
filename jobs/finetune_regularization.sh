#!/bin/bash
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -J finetune_cons_reg
#BSUB -o jobs/logs/finetune_cons_reg_%J.out
#BSUB -e jobs/logs/finetune_cons_reg_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail

DEVICE="cuda:0"

# Sync dependencies using uv
uv sync

# Create necessary directories
mkdir -p results/finetune_consistency_reg
mkdir -p jobs/logs

echo "============================================================"
echo "Consistency-regularized fine-tuning of CLIP-EBC (ViT-B/16)"
echo "============================================================"

uv run python training/finetune_consistency_reg.py \
    --device "${DEVICE}" \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-5 \
    --lambda_cons 0.5 \
    --alpha_lr 0.3 \
    --asym_dense_over 2.0 \
    --asym_dense_under 5.0 \
    --min_down 2.0 \
    --max_down 4.0 \
    --num_workers 4

echo "Fine-tuning completed."