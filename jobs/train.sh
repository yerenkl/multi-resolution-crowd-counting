#!/bin/bash
#BSUB -q gpuv100
#BSUB -J ebc_train
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o jobs/logs/ebc_train_%J.out
#BSUB -e jobs/logs/ebc_train_%J.err
echo "Running CLIP-EBC training..."
uv sync
uv run src/CLIP-EBC/trainer.py   --model clip_resnet50   --input_size 448   --reduction 8   --truncation 4   --anchor_points average   --prompt_type word   --dataset nwpu   --count_loss dmcount