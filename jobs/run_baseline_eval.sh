#!/bin/bash
#BSUB -q gpua40
#BSUB -W 4:00
#BSUB -J baseline_eval
#BSUB -o results/baseline/baseline_eval_%J.out
#BSUB -e results/baseline/baseline_eval_%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail

DEVICE="cuda:0"

uv sync

echo "============================================================"
echo "1/3  NWPU val — native resolution"
echo "============================================================"
uv run python evaluation/eval_nwpu_native.py --device "${DEVICE}"

echo ""
echo "============================================================"
echo "2/3  NWPU val — 2x and 4x downscale"
echo "============================================================"
uv run python evaluation/eval_nwpu_downscaled.py --device "${DEVICE}"

echo ""
echo "============================================================"
echo "3/3  Zoom Pairs — HR vs LR consistency"
echo "============================================================"
uv run python evaluation/eval_zoom_pairs.py --device "${DEVICE}"

echo ""
echo "All baseline evaluations completed successfully."
