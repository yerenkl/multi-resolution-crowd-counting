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


uv sync



METHODS=("mix")
DEVICE="cuda:0"

NOISE_OPTIONS=("false")

for method in "${METHODS[@]}"; do
    for noise in "${NOISE_OPTIONS[@]}"; do


        METHOD_PATH="/dtu/blackhole/0a/224426/NWPU_downscaled/mix/4x/no_noise"

        uv run python entrypoints/train_finetune_res_aug.py \
            --device "$DEVICE" \
            --epochs 50 \
            --batch_size 8 \
            --lr 1e-5 \
            --num_workers 4 \
            --path "$METHOD_PATH"

        uv run python entrypoints/eval_zoom_pairs.py \
            --device "$DEVICE" \
            --path "$METHOD_PATH"

#        uv run python entrypoints/eval_nwpu_native.py \
#            --device "$DEVICE" \
#            --path "$METHOD_PATH"

        echo "Finished method: $method | noise: $noise"
    done
done

echo "All methods completed."