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



DEVICE="cuda:0"

METHOD_PATHS=(
  "/dtu/blackhole/0a/224426/NWPU_downscaled/mix/4x/no_noise"
  "/dtu/blackhole/0a/224426/NWPU_downscaled/bilinear/4x/"
  "/dtu/blackhole/0a/224426/NWPU_downscaled/bicubic/4x/"
  "/dtu/blackhole/0a/224426/NWPU_downscaled/lanczos/4x/"
  "/dtu/blackhole/0a/224426/NWPU_downscaled/nearest/4x/"
)

for METHOD_PATH in "${METHOD_PATHS[@]}"; do
  echo "Running for $METHOD_PATH"
  uv run python entrypoints/eval_nwpu_native.py \
    --device "$DEVICE" \
    --path "$METHOD_PATH"
done

echo "All methods completed."