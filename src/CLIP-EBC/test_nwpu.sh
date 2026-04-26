#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

uv run ./src/CLIP-EBC/test_nwpu.py \
    --model clip_vit_b_16 --input_size 224 --reduction 8 --truncation 4 --anchor_points average --prompt_type word \
    --num_vpt 32 --vpt_drop 0.0 --sliding_window --stride 224 \
    --weight_path ./src/CLIP-EBC/checkpoints/nwpu_low/low_res_train/best_mae_0.pth --dataset nwpu_low
    