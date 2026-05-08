#!/bin/bash
#BSUB -q gpua100
#BSUB -W 12:00
#BSUB -J random
#BSUB -o jobs/logs/random_%J.out
#BSUB -e jobs/logs/random_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

#For bash submission:
#gpua100 - A100 nodes (4x 40GB + 6x 80GB models) - Best for your work
#gpuv100 - V100 nodes (various configs: 16GB, 32GB, some with NVlink)
#gpul40s - L40s nodes (6 nodes, 48GB each) - Newer option
#gpua40 - A40 node (1 node, 48GB with NVlink)
#gpua10 - A10 node (1 node, 24GB)
#gpuamd - AMD Radeon nodes (if you need AMD)

set -euo pipefail

#LOG_DIR="logs"
#mkdir -p "$LOG_DIR"
#
#OUT_LOG="$LOG_DIR/output.log"
#ERR_LOG="$LOG_DIR/error.log"
#
#exec > >(tee -a "$OUT_LOG") \
#     2> >(tee -a "$ERR_LOG" >&2)

uv sync



DEVICE="cuda:0"

METHODS=(
    "/dtu/blackhole/0a/224426/NWPU_downscaled/bilinear/4x/results/best_mae.pth"
    "/dtu/blackhole/0a/224426/NWPU_downscaled/bicubic/4x/results/best_mae.pth"
    "/dtu/blackhole/0a/224426/NWPU_downscaled/lanczos/4x/results/best_mae.pth"
    "/dtu/blackhole/0a/224426/NWPU_downscaled/nearest/4x/results/best_mae.pth"
    "/dtu/blackhole/0a/224426/NWPU_downscaled/mix/4x/no_noise/results/best_mae.pth"
    "/dtu/blackhole/0a/224426/NWPU_downscaled/random/results_new_pipeline/best_mae_original.pth"
)

for METHOD in "${METHODS[@]}"; do
  echo "Running method: $METHOD"

  uv run python entrypoints/eval_nwpu_native.py \
      --device "$DEVICE" \
      --path "$METHOD"

done

echo "All methods completed."