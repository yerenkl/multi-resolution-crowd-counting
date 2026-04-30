#!/bin/bash
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -J dann_train
#BSUB -o jobs/logs/dann_%J.out
#BSUB -e jobs/logs/dann_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail

DEVICE="cuda:0"

uv sync

mkdir -p results/dann
mkdir -p jobs/logs

echo "============================================================"
echo "DANN resolution-adversarial training of CLIP-EBC (ViT-B/16)"
echo "============================================================"

uv run python entrypoints/train_dann.py \
    --device "${DEVICE}" \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-5 \
    --lr_disc 1e-4 \
    --dann_weight 1.0 \
    --down_scales 2 4 8 \
    --hidden_dim 256 \
    --dropout 0.5 \
    --num_workers 4 \
    --eval_limit 500

echo "DANN training completed."
