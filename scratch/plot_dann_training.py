import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

LOG_DIR = Path("jobs/logs")
LOGS = {
    "A100 (18-39-57)": LOG_DIR / "dann_28314573.out",
    "A40 (19-41-33)": LOG_DIR / "dann_28314574.out",
    "L40S (17-22-44)": LOG_DIR / "dann_28314575.out",
}

PATTERN = re.compile(
    r"Epoch\s+(\d+)/50 \| "
    r"loss=([\d.]+) task=([\d.]+) domain=([\d.]+) alpha=([\d.]+) \| "
    r"MAE=([\d.]+) RMSE=([\d.]+)"
)


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


runs = {name: parse_log(path) for name, path in LOGS.items()}

colors = {"A100 (18-39-57)": "#2196F3", "A40 (19-41-33)": "#FF9800", "L40S (17-22-44)": "#4CAF50"}
best_run = "A100 (18-39-57)"

fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
fig.suptitle("DANN Training — 3 runs, same config, different GPUs", fontsize=14, fontweight="bold")

# --- 1. Task loss ---
ax = axes[0, 0]
for name, d in runs.items():
    ax.plot(d["epoch"], d["task"], color=colors[name], alpha=0.8, linewidth=1.5, label=name)
ax.set_title("Task Loss (DACELoss on HR + LR)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Task Loss")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- 2. Domain loss + alpha ---
ax = axes[0, 1]
ax2 = ax.twinx()
for name, d in runs.items():
    ax.plot(d["epoch"], d["domain"], color=colors[name], alpha=0.8, linewidth=1.5, label=name)
d0 = runs[best_run]
ax2.plot(d0["epoch"], d0["alpha"], color="red", linestyle="--", linewidth=1.5, alpha=0.6, label="alpha (Ganin)")
ax.set_title("Domain Loss (BCE) + Alpha Schedule")
ax.set_xlabel("Epoch")
ax.set_ylabel("Domain Loss")
ax2.set_ylabel("Alpha (GRL strength)", color="red")
ax2.tick_params(axis="y", labelcolor="red")
ax2.set_ylim(0, 1.1)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
ax.grid(True, alpha=0.3)

# --- 3. Val MAE ---
ax = axes[1, 0]
baseline_mae = 45.11
ax.axhline(y=baseline_mae, color="gray", linestyle=":", linewidth=1.5, label=f"Pretrained baseline ({baseline_mae})")
for name, d in runs.items():
    ax.plot(d["epoch"], d["mae"], color=colors[name], alpha=0.8, linewidth=1.5, label=name)
    best_epoch = d["epoch"][np.argmin(d["mae"])]
    best_mae = min(d["mae"])
    ax.scatter([best_epoch], [best_mae], color=colors[name], s=60, zorder=5, edgecolors="black", linewidths=0.8)
    ax.annotate(f"{best_mae:.1f}", (best_epoch, best_mae), textcoords="offset points",
                xytext=(8, -8), fontsize=8, color=colors[name], fontweight="bold")
ax.set_title("Val MAE (NWPU, 500 images, native res)")
ax.set_xlabel("Epoch")
ax.set_ylabel("MAE")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- 4. Domain loss vs alpha (phase diagram) ---
ax = axes[1, 1]
for name, d in runs.items():
    ax.scatter(d["alpha"], d["domain"], color=colors[name], alpha=0.5, s=20, label=name)
    for i in range(len(d["alpha"]) - 1):
        ax.annotate("", xy=(d["alpha"][i+1], d["domain"][i+1]),
                    xytext=(d["alpha"][i], d["domain"][i]),
                    arrowprops=dict(arrowstyle="-", color=colors[name], alpha=0.2, lw=0.8))
ax.set_title("Domain Loss vs Alpha (adversarial dynamics)")
ax.set_xlabel("Alpha (GRL strength)")
ax.set_ylabel("Domain Loss")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

out_path = Path("scratch/dann_training.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
