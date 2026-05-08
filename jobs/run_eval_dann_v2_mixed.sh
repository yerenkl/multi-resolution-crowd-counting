#!/bin/bash
#BSUB -q gpua100
#BSUB -W 4:00
#BSUB -J eval_dann_v2_mixed
#BSUB -o jobs/logs/eval_dann_v2_mixed_%J.out
#BSUB -e jobs/logs/eval_dann_v2_mixed_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
uv sync
mkdir -p jobs/logs

WEIGHTS="/work3/s225224/multi-resolution-crowd-counting/checkpoints/dann/2026-04-30_15-46-40/best_mae.pth"
OUTPUT_DIR="results/dann_v2/mixed"

echo "============================================================"
echo "DANN v2 mixed (2x+4x) full evaluation with TTA"
echo "Weights: ${WEIGHTS}"
echo "============================================================"

uv run python entrypoints/eval_dann_tta.py \
    --dann_weights "${WEIGHTS}" \
    --device cuda:0 \
    --output_dir "${OUTPUT_DIR}"

echo "Done."
