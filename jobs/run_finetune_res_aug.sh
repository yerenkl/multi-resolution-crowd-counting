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


uv run python entrypoints/eval_zoom_pairs.py \
    --device "${DEVICE}" \
    --path "/dtu/blackhole/0a/224426/NWPU_downscaled/random"

echo "Fine-tuning completed."
