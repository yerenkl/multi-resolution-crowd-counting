#!/bin/bash
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -J dann_v2_mix
#BSUB -o jobs/logs/dann_v2_mixed_%J.out
#BSUB -e jobs/logs/dann_v2_mixed_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
uv sync
mkdir -p jobs/logs

echo "============================================================"
echo "DANN v2 — LR domain: mixed 2x + 4x"
echo "============================================================"

uv run python entrypoints/train_dann_v2.py \
    --device cuda:0 \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-5 \
    --lr_disc 1e-4 \
    --dann_weight 1.0 \
    --lr_scales 2 4 \
    --hidden_dim 256 \
    --dropout 0.5 \
    --num_workers 4 \
    --eval_limit 500

echo "Done."
