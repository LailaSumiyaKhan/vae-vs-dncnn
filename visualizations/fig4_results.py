"""Figure 4: Quantitative results -- PSNR, SSIM, LPIPS bar charts and noise-level lines."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = Path("results/metrics/results.csv")
OUT_PATH = Path("report/figures/fig4_results.pdf")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
NOISE_TYPES = sorted(df["noise_type"].unique())
SIGMAS      = sorted(df["sigma"].unique())
MODELS      = ["dncnn", "vae"]
COLORS      = {"dncnn": "#1565C0", "vae": "#E53935"}
LABELS      = {"dncnn": "DnCNN", "vae": "VAE"}
METRICS     = [("psnr", "PSNR (dB)", "higher"), ("ssim", "SSIM", "higher"), ("lpips", "LPIPS", "lower")]

# Compute means
mean = df.groupby(["model", "noise_type", "sigma"])[["psnr", "ssim", "lpips"]].mean()

fig, axes = plt.subplots(len(METRICS), len(NOISE_TYPES),
                         figsize=(5 * len(NOISE_TYPES), 4.2 * len(METRICS)))
fig.suptitle("Denoising Metrics: DnCNN vs VAE", fontsize=14, y=1.01)

x = np.arange(len(SIGMAS))
width = 0.32

for r, (metric, mlabel, direction) in enumerate(METRICS):
    for c, nt in enumerate(NOISE_TYPES):
        ax = axes[r][c]
        for i, model in enumerate(MODELS):
            vals = []
            for s in SIGMAS:
                try:
                    v = mean.loc[(model, nt, s), metric]
                except KeyError:
                    v = float("nan")
                vals.append(v)
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=LABELS[model], color=COLORS[model], alpha=0.88)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.012,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        arrow = "$\\uparrow$" if direction == "higher" else "$\\downarrow$"
        ax.set_title(f"{nt.capitalize()} Noise -- {mlabel} {arrow}", fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$\\sigma$={s}" for s in SIGMAS])
        ax.set_ylabel(mlabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
print(f"Saved {OUT_PATH}")


# --- Figure 4b: line plots (metric vs sigma) ---
OUT_B = Path("report/figures/fig4b_metrics_vs_sigma.pdf")
fig2, axes2 = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
fig2.suptitle("Metric Degradation with Increasing Noise Level", fontsize=12)

for ax, (metric, mlabel, direction) in zip(axes2, METRICS):
    for nt in NOISE_TYPES:
        for model in MODELS:
            vals = []
            for s in SIGMAS:
                try:
                    v = mean.loc[(model, nt, s), metric]
                except KeyError:
                    v = float("nan")
                vals.append(v)
            ls = "-" if model == "dncnn" else "--"
            marker = "o" if model == "dncnn" else "s"
            label = f"{LABELS[model]} / {nt}"
            ax.plot(SIGMAS, vals, ls + marker, label=label, linewidth=1.8, markersize=6)

    arrow = "$\\uparrow$" if direction == "higher" else "$\\downarrow$"
    ax.set_title(f"{mlabel} {arrow}", fontsize=10)
    ax.set_xlabel("Noise level ($\\sigma$)")
    ax.set_ylabel(mlabel)
    ax.set_xticks(SIGMAS)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

fig2.tight_layout()
fig2.savefig(OUT_B, bbox_inches="tight", dpi=200)
print(f"Saved {OUT_B}")
