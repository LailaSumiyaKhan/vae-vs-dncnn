"""Figure 3: Data preprocessing pipeline and pixel statistics."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import random

random.seed(42)
np.random.seed(42)

TRAIN_DIR = Path("data/train")
OUT_PATH  = Path("report/figures/fig3_preprocessing.pdf")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load a sample training image
sample_path = sorted(TRAIN_DIR.iterdir())[0]
color_img   = np.array(Image.open(sample_path))               # H x W x 3
gray_img    = np.array(Image.open(sample_path).convert("L"))  # H x W
norm_img    = gray_img.astype(np.float32) / 255.0             # [0,1]

# Extract a random 64x64 patch
h, w = gray_img.shape
ps = 64
top  = random.randint(0, h - ps)
left = random.randint(0, w - ps)
patch = norm_img[top:top+ps, left:left+ps]

# -----------------------------------------------------------------------
# Row 1: pipeline (original -> grayscale -> normalized -> patch)
# Row 2: histogram comparisons
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(14, 8))
gs  = fig.add_gridspec(2, 5, hspace=0.5, wspace=0.4)

# --- Pipeline images ---
img_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
titles   = ["(a) Original RGB", "(b) Grayscale", "(c) Normalized [0,1]", "(d) 64x64 Patch"]
imgs     = [color_img, gray_img, norm_img, patch]
cmaps    = [None, "gray", "gray", "gray"]
ranges   = [(0, 255), (0, 255), (0.0, 1.0), (0.0, 1.0)]

for ax, img, title, cmap, (vmin, vmax) in zip(img_axes, imgs, titles, cmaps, ranges):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title(title, fontsize=9)

# Annotate sizes
for ax, img in zip(img_axes, imgs):
    if img.ndim == 3:
        lbl = f"{img.shape[1]}x{img.shape[0]} px"
    else:
        lbl = f"{img.shape[1]}x{img.shape[0]} px"
    ax.text(0.5, -0.06, lbl, transform=ax.transAxes, ha="center", fontsize=8, color="gray")

# Patch extraction: draw rectangle on the normalized image panel
rect = mpatches.FancyArrowPatch(
    posA=(0.75, 0.5), posB=(1.05, 0.5),
    arrowstyle="-|>", mutation_scale=18,
    transform=img_axes[2].transAxes,
    color="#E53935", zorder=5, clip_on=False,
)
img_axes[2].add_patch(rect)
img_axes[2].add_patch(
    mpatches.Rectangle(
        (left / w, top / h),
        ps / w, ps / h,
        linewidth=2, edgecolor="#E53935", facecolor="none",
        transform=img_axes[2].transAxes,
    )
)

# Augmentation schematic (label only)
ax_aug = fig.add_subplot(gs[0, 4])
ax_aug.axis("off")
aug_text = (
    "Data Augmentation\n\n"
    "  Horizontal flip\n"
    "  Vertical flip\n"
    "  90/180/270 rotation\n\n"
    "200 patches/image\n"
    "300 images\n"
    "= 60,000 patches"
)
ax_aug.text(0.1, 0.9, aug_text, transform=ax_aug.transAxes,
            fontsize=8.5, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#BDBDBD"))
ax_aug.set_title("(e) Augmentation", fontsize=9)

# --- Histograms ---
# Pixel distribution before and after normalization, and patch distribution
ax_h1 = fig.add_subplot(gs[1, 0:2])
ax_h2 = fig.add_subplot(gs[1, 2:4])
ax_h3 = fig.add_subplot(gs[1, 4])

# Load all training images for dataset-wide stats
all_gray_pixels = []
all_norm_pixels = []
paths = sorted(TRAIN_DIR.iterdir())[:50]  # sample 50 for speed
for p in paths:
    g = np.array(Image.open(p).convert("L")).flatten()
    all_gray_pixels.append(g)
    all_norm_pixels.append(g / 255.0)

all_gray_pixels = np.concatenate(all_gray_pixels)
all_norm_pixels = np.concatenate(all_norm_pixels)

ax_h1.hist(all_gray_pixels, bins=64, color="#1565C0", alpha=0.8, density=True)
ax_h1.set_title("(f) Pixel Distribution (8-bit)", fontsize=9)
ax_h1.set_xlabel("Pixel Value [0, 255]")
ax_h1.set_ylabel("Density")
ax_h1.grid(alpha=0.3)

ax_h2.hist(all_norm_pixels, bins=64, color="#2E7D32", alpha=0.8, density=True)
ax_h2.set_title("(g) Pixel Distribution (Normalized)", fontsize=9)
ax_h2.set_xlabel("Pixel Value [0, 1]")
ax_h2.set_ylabel("Density")
ax_h2.grid(alpha=0.3)

ax_h3.hist(patch.flatten(), bins=32, color="#6A1B9A", alpha=0.8, density=True)
ax_h3.set_title("(h) Patch Distribution", fontsize=9)
ax_h3.set_xlabel("Value [0, 1]")
ax_h3.set_ylabel("Density")
ax_h3.grid(alpha=0.3)

fig.suptitle("Data Preprocessing Pipeline and Pixel Statistics", fontsize=12, y=1.01)
fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
print(f"Saved {OUT_PATH}")
