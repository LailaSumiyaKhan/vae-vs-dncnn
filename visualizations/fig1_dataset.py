"""Figure 1: Dataset samples from BSD300 (training) and BSD68 (test)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TRAIN_DIR = Path("data/train")
TEST_DIR  = Path("data/test")
OUT_PATH  = Path("report/figures/fig1_dataset.pdf")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TRAIN_NAMES = ["100075.jpg", "101085.jpg", "103041.jpg", "105019.jpg"]
TEST_NAMES  = ["test001.png", "test007.png", "test015.png", "test022.png"]

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle("Dataset Overview: BSD300 Training Samples (top) and BSD68 Test Samples (bottom)",
             fontsize=11, y=1.01)

for col, name in enumerate(TRAIN_NAMES):
    img = Image.open(TRAIN_DIR / name).convert("L")
    arr = np.array(img)
    ax = axes[0][col]
    ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    ax.set_title(f"Train: {name[:6]}", fontsize=9)
    # Annotate size
    ax.text(0.5, -0.04, f"{arr.shape[1]}x{arr.shape[0]}",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")

for col, name in enumerate(TEST_NAMES):
    img = Image.open(TEST_DIR / name).convert("L")
    arr = np.array(img)
    ax = axes[1][col]
    ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    ax.set_title(f"Test: {name}", fontsize=9)
    ax.text(0.5, -0.04, f"{arr.shape[1]}x{arr.shape[0]}",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")

axes[0][0].set_ylabel("Training\n(BSD300)", fontsize=9, rotation=90, labelpad=8)
axes[1][0].set_ylabel("Test\n(BSD68)", fontsize=9, rotation=90, labelpad=8)

fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
print(f"Saved {OUT_PATH}")
