"""Figure 6: Model efficiency comparison (parameters, inference speed, training loss)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.models.dncnn import DnCNN
from src.models.vae import DenoisingVAE

OUT_PATH = Path("report/figures/fig6_efficiency.pdf")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

cfg = Config()

# Parameter counts
dncnn_params = DnCNN().count_parameters()
vae_params   = DenoisingVAE(latent_dim=cfg.train.latent_dim, patch_size=cfg.train.patch_size).count_parameters()

# Inference speed from results.csv
df        = pd.read_csv("results/metrics/results.csv")
speed_dn  = df[df["model"] == "dncnn"]["inference_ms"].dropna().mean()
speed_vae = df[df["model"] == "vae"]["inference_ms"].dropna().mean()

# Training loss curves from checkpoints
CKPT_DIR = Path("results/checkpoints")
noise_type, sigma = "gaussian", 15

def _load_history(model_name):
    ckpt = CKPT_DIR / f"{model_name}_{noise_type}_{sigma}.pth"
    if not ckpt.exists():
        return None
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    return state.get("history", {})

dn_history  = _load_history("dncnn")
vae_history = _load_history("vae")

# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Computational Efficiency Comparison: DnCNN vs VAE", fontsize=13)

COLORS = ["#1565C0", "#E53935"]
LABELS = ["DnCNN", "VAE"]

# Panel 1: Parameter count
ax1 = axes[0]
params_vals = [dncnn_params / 1e3, vae_params / 1e3]
bars = ax1.bar(LABELS, params_vals, color=COLORS, alpha=0.88, width=0.45)
for bar, v, raw in zip(bars, params_vals, [dncnn_params, vae_params]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
             f"{raw:,}", ha="center", fontsize=9)
ax1.set_ylabel("Parameters (thousands)")
ax1.set_title("Model Size (Trainable Parameters)")
ax1.grid(axis="y", alpha=0.3)
ax1.spines[["top", "right"]].set_visible(False)

# Panel 2: Inference speed
ax2 = axes[1]
speed_vals = [speed_dn, speed_vae]
bars2 = ax2.bar(LABELS, speed_vals, color=COLORS, alpha=0.88, width=0.45)
for bar, v in zip(bars2, speed_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
             f"{v:.1f} ms", ha="center", fontsize=9)
ax2.set_ylabel("Inference Time (ms/image)")
ax2.set_title("Inference Speed (BSD68 test images)")
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top", "right"]].set_visible(False)

# Panel 3: Training loss curves
ax3 = axes[2]
if dn_history and "train_loss" in dn_history:
    epochs = range(1, len(dn_history["train_loss"]) + 1)
    ax3.plot(epochs, dn_history["train_loss"], color=COLORS[0], linewidth=2, label="DnCNN (MSE)")
if vae_history and "recon_loss" in vae_history:
    epochs = range(1, len(vae_history["recon_loss"]) + 1)
    ax3.plot(epochs, vae_history["recon_loss"], color=COLORS[1], linewidth=2,
             label="VAE (Recon Loss)")
    ax3.plot(epochs, vae_history.get("kl_loss", []), color=COLORS[1], linewidth=1.5,
             linestyle="--", label="VAE (KL Loss)", alpha=0.7)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Loss")
ax3.set_title(f"Training Loss -- Gaussian $\\sigma$={sigma}")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)
ax3.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
print(f"Saved {OUT_PATH}")
