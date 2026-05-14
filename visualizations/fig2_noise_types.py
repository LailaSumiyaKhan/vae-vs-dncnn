"""Figure 2: Same test image corrupted with each noise type at sigma=15 and sigma=25."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.noise import add_noise

torch.manual_seed(42)

TEST_IMG  = Path("data/test/test001.png")
OUT_PATH  = Path("report/figures/fig2_noise_types.pdf")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

arr = np.array(Image.open(TEST_IMG).convert("L"), dtype=np.float32) / 255.0
clean_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

NOISE_TYPES = ["gaussian", "poisson", "speckle"]
SIGMAS      = [15, 25]
LABELS      = {"gaussian": "Gaussian", "poisson": "Poisson", "speckle": "Speckle"}

n_cols = 1 + len(NOISE_TYPES) * len(SIGMAS)  # clean + 6 noisy
fig, axes = plt.subplots(1, n_cols, figsize=(2.4 * n_cols, 3.2))
fig.suptitle("Noise Type Comparison on a BSD68 Test Image", fontsize=11, y=1.02)

# Column 0: clean
axes[0].imshow(arr, cmap="gray", vmin=0, vmax=1)
axes[0].axis("off")
axes[0].set_title("Clean", fontsize=9, fontweight="bold")

col = 1
for noise_type in NOISE_TYPES:
    for sigma in SIGMAS:
        noisy = add_noise(clean_t, noise_type, float(sigma)).squeeze().numpy()
        psnr_val = 10 * np.log10(1.0 / (np.mean((arr - noisy) ** 2) + 1e-10))
        ax = axes[col]
        ax.imshow(noisy, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{LABELS[noise_type]}\n$\\sigma$={sigma}", fontsize=8)
        ax.text(0.5, -0.04, f"PSNR={psnr_val:.1f} dB",
                transform=ax.transAxes, ha="center", fontsize=7.5, color="#333")
        col += 1

fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
print(f"Saved {OUT_PATH}")
