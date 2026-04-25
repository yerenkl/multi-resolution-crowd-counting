#!/bin/bash
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -J high_res
#BSUB -o jobs/logs/high_res_%J.out
#BSUB -e jobs/logs/high_res_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

uv run src/CLIP-EBC/trainer.py     --model clip_vit_b_16 --input_size 224 --reduction 8 --truncation 4     --dataset nwpu --batch_size 16 --amp     --num_crops 2 --sliding_window --window_size 224 --stride 224 --warmup_lr 1e-3     --count_loss dmcount