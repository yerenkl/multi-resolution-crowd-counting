import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

conditions = ["Native", "2x down", "4x down"]
baseline_mae = [45.11, 44.35, 96.03]
dann_v1_mae = [51.32, 51.06, 100.19]
deltas = [d - b for b, d in zip(baseline_mae, dann_v1_mae)]

x = np.arange(len(conditions))
width = 0.32

fig, ax = plt.subplots(figsize=(6, 4))

bars_b = ax.bar(x - width / 2, baseline_mae, width, label="Baseline", color="#74c0fc", edgecolor="white", linewidth=0.8)
bars_d = ax.bar(x + width / 2, dann_v1_mae, width, label="DANN v1", color="#ff6b6b", edgecolor="white", linewidth=0.8)

for bar, val in zip(bars_b, baseline_mae):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{val:.1f}",
            ha="center", va="bottom", fontsize=9, color="#333")

for bar, val, delta in zip(bars_d, dann_v1_mae, deltas):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{val:.1f}\n(+{delta:.1f})",
            ha="center", va="bottom", fontsize=9, color="#cc3333", fontweight="bold")

ax.set_ylabel("MAE ↓", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=10)
ax.set_ylim(0, max(dann_v1_mae) * 1.25)
ax.legend(fontsize=10, loc="upper left")
ax.set_title("DANN v1: MAE by evaluation condition", fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, linestyle="--")

fig.tight_layout()
fig.savefig("scratch/dann_v1_mae_comparison.svg", format="svg", bbox_inches="tight")
fig.savefig("scratch/dann_v1_mae_comparison.png", format="png", dpi=200, bbox_inches="tight")
