#!/bin/bash
#BSUB -q gpuv100
#BSUB -W 24:00
#BSUB -J mixed_res
#BSUB -o jobs/logs/mixed_res_%J.out
#BSUB -e jobs/logs/mixed_res_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

uv run src/CLIP-EBC/trainer.py --model clip_vit_b_16 --input_size 224 --reduction 8 --truncation 4     --dataset nwpu_mixed --batch_size 16 --amp     --num_crops 2 --sliding_window --window_size 224 --stride 224 --warmup_lr 1e-3     --count_loss dmcount --ckpt_dir mixed_res_train