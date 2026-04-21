#!/bin/bash
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -J finetune_res_aug
#BSUB -o jobs/logs/finetune_res_aug_%J.out
#BSUB -e jobs/logs/finetune_res_aug_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail

DEVICE="cuda:0"

uv sync

mkdir -p results/finetune_resolution_aug
mkdir -p jobs/logs

echo "============================================================"
echo "Resolution-augmented fine-tuning of CLIP-EBC (ViT-B/16)"
echo "============================================================"

uv run python training/finetune_resolution_aug.py \
    --device "${DEVICE}" \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-5 \
    --min_scale 1.0 \
    --max_scale 4.0 \
    --num_workers 4

echo "Fine-tuning completed."
