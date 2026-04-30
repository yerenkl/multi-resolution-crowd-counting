#!/bin/bash
#BSUB -q gpua100
#BSUB -W 4:00
#BSUB -J eval_dann
#BSUB -o jobs/logs/eval_dann_%J.out
#BSUB -e jobs/logs/eval_dann_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
uv sync
mkdir -p jobs/logs

DEVICE="cuda:0"
WEIGHTS="/work3/s225224/multi-resolution-crowd-counting/checkpoints/dann/2026-04-28_18-39-57/best_mae.pth"
OUTPUT_DIR="/work3/s225224/multi-resolution-crowd-counting/checkpoints/dann/2026-04-28_18-39-57"

echo "============================================================"
echo "Evaluating DANN checkpoint: ${WEIGHTS}"
echo "============================================================"

echo ""
echo "1/4  NWPU val — native resolution (by density bucket)"
echo "------------------------------------------------------------"
uv run python entrypoints/eval_checkpoint.py \
    --device "${DEVICE}" \
    --weights "${WEIGHTS}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "2/4  NWPU val — 2x and 4x downscale"
echo "------------------------------------------------------------"
uv run python entrypoints/eval_nwpu_downscaled.py \
    --device "${DEVICE}" \
    --weights "${WEIGHTS}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "3/4  Zoom Pairs — HR vs LR consistency"
echo "------------------------------------------------------------"
uv run python entrypoints/eval_zoom_pairs.py \
    --device "${DEVICE}" \
    --weights "${WEIGHTS}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "All DANN evaluations completed. Results in ${OUTPUT_DIR}"
echo "============================================================"
