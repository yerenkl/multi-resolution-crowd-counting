#!/bin/bash
#BSUB -q gpua100
#BSUB -J ebc_eval
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o jobs/logs/ebc_eval_%J.out
#BSUB -e jobs/logs/ebc_eval_%J.err
set -euo pipefail
echo "Running CLIP-EBC evaluation..."


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
