echo $(pwd)

uv run python entrypoints/anotate_predict.py \
    --weights-1 "/dtu/blackhole/0a/224426/NWPU_downscaled/base/results/best_mae.pth" \
    --weights-2 "/dtu/blackhole/0a/224426/NWPU_downscaled/mix/4x/no_noise/results/best_mae.pth" \
    --label-1 "base" \
    --label-2 "mix" \
    --num-viz 5 \
    --device cuda:0