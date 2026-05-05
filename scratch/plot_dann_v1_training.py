import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path("jobs/logs")
LOG_PATH = LOG_DIR / "dann_28314573.out"

PATTERN = re.compile(
    r"Epoch\s+(\d+)/50 \| "
    r"loss=([\d.]+) task=([\d.]+) domain=([\d.]+) alpha=([\d.]+) \| "
    r"MAE=([\d.]+) RMSE=([\d.]+)"
)

BLUE = "#74c0fc"
RED = "#ff6b6b"
GRAY = "#888888"
DARK_RED = "#cc3333"


def parse_log(path):
    epochs, loss, task, domain, alpha, mae, rmse = [], [], [], [], [], [], []
    for line in path.read_text().splitlines():
        m = PATTERN.search(line)
        if m:
            epochs.append(int(m.group(1)))
            loss.append(float(m.group(2)))
            task.append(float(m.group(3)))
            domain.append(float(m.group(4)))
            alpha.append(float(m.group(5)))
            mae.append(float(m.group(6)))
            rmse.append(float(m.group(7)))
    return dict(epoch=epochs, loss=loss, task=task, domain=domain,
                alpha=alpha, mae=mae, rmse=rmse)


d = parse_log(LOG_PATH)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# --- Left panel: Domain loss + alpha schedule ---
ax1_twin = ax1.twinx()

ax1.plot(d["epoch"], d["domain"], color=RED, alpha=0.9, linewidth=2.0, label="Domain loss")
ax1_twin.plot(d["epoch"], d["alpha"], color=GRAY, linestyle="--", linewidth=1.5, alpha=0.7, label="α (Ganin schedule)")
ax1_twin.fill_between(d["epoch"], 0, d["alpha"], color=GRAY, alpha=0.06)

ax1.set_title("Domain loss rises → features becoming invariant", fontsize=10, fontweight="bold")
ax1.set_xlabel("Epoch", fontsize=9)
ax1.set_ylabel("Domain loss (BCE)", fontsize=9, color=DARK_RED)
ax1_twin.set_ylabel("α (GRL strength)", fontsize=9, color=GRAY)
ax1_twin.set_ylim(0, 1.15)
ax1.tick_params(axis="y", labelcolor=DARK_RED)
ax1_twin.tick_params(axis="y", labelcolor=GRAY)
ax1.grid(axis="y", alpha=0.2, linestyle="--")
ax1.spines["top"].set_visible(False)
ax1_twin.spines["top"].set_visible(False)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

# --- Right panel: Val MAE vs baseline ---
baseline_mae = 45.11
ax2.axhline(y=baseline_mae, color=BLUE, linestyle="-", linewidth=2.0, alpha=0.8, label=f"Baseline MAE ({baseline_mae})")
ax2.axhspan(0, baseline_mae, color=BLUE, alpha=0.04)

ax2.plot(d["epoch"], d["mae"], color=RED, alpha=0.9, linewidth=2.0, label="DANN v1")

best_idx = int(np.argmin(d["mae"]))
best_epoch = d["epoch"][best_idx]
best_mae = d["mae"][best_idx]
ax2.scatter([best_epoch], [best_mae], color=RED, s=60, zorder=5, edgecolors="white", linewidths=1.5)
ax2.annotate(f"Best: {best_mae:.1f}", (best_epoch, best_mae),
             textcoords="offset points", xytext=(10, -15),
             fontsize=9, color=DARK_RED, fontweight="bold",
             arrowprops=dict(arrowstyle="-", color=DARK_RED, lw=0.8))

ax2.set_title("Val MAE stays above baseline → negative transfer", fontsize=10, fontweight="bold")
ax2.set_xlabel("Epoch", fontsize=9)
ax2.set_ylabel("MAE ↓", fontsize=9)
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(axis="y", alpha=0.2, linestyle="--")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("scratch/dann_v1_training.svg", format="svg", bbox_inches="tight")
fig.savefig("scratch/dann_v1_training.png", format="png", dpi=200, bbox_inches="tight")
