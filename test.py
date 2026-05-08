import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Hardcoded model -> file path mapping
MODEL_FILES = {
    "mix": "/dtu/blackhole/0a/224426/NWPU_downscaled/mix/4x/no_noise/results/best_mae/zoom_pairs.json",
    "base": "/dtu/blackhole/0a/224426/NWPU_downscaled/base/results/best_mae/zoom_pairs.json",
    # "bicubic": "/dtu/blackhole/0a/224426/NWPU_downscaled/bicubic/4x/results/best_mae/zoom_pairs.json",
    "nearest": "/dtu/blackhole/0a/224426/NWPU_downscaled/nearest/4x/results/best_mae/zoom_pairs.json",
}


def bucket_hr(hr):
    if hr < 100:
        return "<100"
    elif 100 <= hr <= 500:
        return "100-500"
    else:
        return ">500"


# Step 1: collect raw data (include pair!)
all_data = []

for model_name, file_path in MODEL_FILES.items():
    with open(file_path, "r") as f:
        data = json.load(f)

        for entry in data:
            all_data.append({
                "model": model_name,
                "pair": entry["pair"],
                "hr_count": entry["hr_count"],
                "lr_count": entry["lr_count"],
            })

df = pd.DataFrame(all_data)

# Step 2: average per (model, pair)
pair_avg = df.groupby(["model", "pair"]).agg({
    "hr_count": "mean",
    "lr_count": "mean"
}).reset_index()

# Optional (better ratio computation if you need it later)
# pair_avg["ratio"] = pair_avg["hr_count"] / pair_avg["lr_count"]

# Step 3: bucket AFTER averaging
pair_avg["bucket"] = pair_avg["hr_count"].apply(bucket_hr)

# Step 4: aggregate per bucket
grouped = pair_avg.groupby(["model", "bucket"]).agg({
    "hr_count": "mean",
    "lr_count": "mean"
}).reset_index()

# Ensure bucket order
bucket_order = ["<100", "100-500", ">500"]
grouped["bucket"] = pd.Categorical(grouped["bucket"], categories=bucket_order, ordered=True)
grouped = grouped.sort_values(["model", "bucket"])

pivot_hr = grouped.pivot(index="bucket", columns="model", values="hr_count")
pivot_lr = grouped.pivot(index="bucket", columns="model", values="lr_count")

x = np.arange(len(pivot_hr.index))
width = 0.15  # smaller width since we have more bars

plt.figure()

models = pivot_hr.columns

for i, model in enumerate(models):
    # HR bars
    plt.bar(
        x + i * 2 * width,
        pivot_hr[model],
        width,
        label=f"{model} HR"
    )

    # LR bars (shifted slightly right)
    plt.bar(
        x + i * 2 * width + width,
        pivot_lr[model],
        width,
        label=f"{model} LR",
        alpha=0.7
    )

# Center ticks
total_width = len(models) * 2 * width
plt.xticks(x + total_width / 2 - width, pivot_hr.index)

plt.title("HR vs LR Count per Bucket")
plt.xlabel("HR Buckets")
plt.ylabel("Average Count")
plt.legend(ncol=2)  # cleaner legend
plt.grid(axis='y')

plt.show()