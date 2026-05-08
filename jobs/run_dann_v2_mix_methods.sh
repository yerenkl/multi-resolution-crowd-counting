#!/bin/bash
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -J dann_v2_mix_methods
#BSUB -o jobs/logs/dann_v2_mix_methods_%J.out
#BSUB -e jobs/logs/dann_v2_mix_methods_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
uv sync
mkdir -p jobs/logs

echo "============================================================"
echo "DANN v2 — LR domain: mix/4x/no_noise (bilinear+bicubic+nearest+lanczos)"
echo "============================================================"

uv run python entrypoints/train_dann_v2.py \
    --device cuda:0 \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-5 \
    --lr_disc 1e-4 \
    --dann_weight 1.0 \
    --lr_image_dir /dtu/blackhole/0a/224426/NWPU_downscaled/mix/4x/no_noise/images \
    --lr_scale 4 \
    --hidden_dim 256 \
    --dropout 0.5 \
    --num_workers 4 \
    --eval_limit 500 \
    --checkpoints_dir /work3/"$USER"/multi-resolution-crowd-counting/checkpoints/dann

echo "Done."
