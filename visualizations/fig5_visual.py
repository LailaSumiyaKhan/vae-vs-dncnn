"""Figure 5: Side-by-side visual denoising comparison (clean/noisy/DnCNN/VAE)."""
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

from src.config import Config
from src.models.dncnn import DnCNN
from src.models.vae import DenoisingVAE
from src.noise import add_noise
from src.evaluator import _vae_denoise_full
from src.metrics import compute_psnr, compute_ssim

torch.manual_seed(42)

TEST_IMAGES  = ["test001.png", "test007.png", "test015.png"]
NOISE_CFG    = ("gaussian", 25)
CKPT_DIR     = Path("results/checkpoints")
OUT_PATH     = Path("report/figures/fig5_visual.pdf")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

cfg = Config()
noise_type, sigma = NOISE_CFG

def _load_dncnn():
    ckpt = CKPT_DIR / f"dncnn_{noise_type}_{sigma}.pth"
    if not ckpt.exists():
        return None
    m = DnCNN().to(cfg.device)
    m.load_state_dict(torch.load(ckpt, map_location=cfg.device, weights_only=True)["model"])
    m.eval()
    return m

def _load_vae():
    ckpt = CKPT_DIR / f"vae_{noise_type}_{sigma}.pth"
    if not ckpt.exists():
        return None
    m = DenoisingVAE(latent_dim=cfg.train.latent_dim, patch_size=cfg.train.patch_size).to(cfg.device)
    m.load_state_dict(torch.load(ckpt, map_location=cfg.device, weights_only=True)["model"])
    m.eval()
    return m

dncnn = _load_dncnn()
vae   = _load_vae()

if dncnn is None and vae is None:
    print("No checkpoints found. Run training first.")
    sys.exit(0)

COL_LABELS = ["Clean", f"Noisy ($\\sigma$={sigma})", "DnCNN", "VAE"]
n_imgs = len(TEST_IMAGES)
fig, axes = plt.subplots(n_imgs, 4, figsize=(14, 3.8 * n_imgs))
fig.suptitle(f"Visual Denoising Comparison -- {noise_type.capitalize()} Noise ($\\sigma$={sigma})",
             fontsize=12, y=1.01)

for col_i, label in enumerate(COL_LABELS):
    axes[0][col_i].set_title(label, fontsize=10, fontweight="bold")

for row_i, fname in enumerate(TEST_IMAGES):
    arr = np.array(Image.open(Path("data/test") / fname).convert("L"),
                   dtype=np.float32) / 255.0
    clean_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(cfg.device)
    noisy_t = add_noise(clean_t, noise_type, float(sigma))

    with torch.no_grad():
        dn_out  = dncnn(noisy_t).squeeze().cpu().numpy() if dncnn else noisy_t.squeeze().cpu().numpy()
        vae_out = _vae_denoise_full(vae, noisy_t, cfg.train.patch_size).squeeze().cpu().numpy() \
                  if vae else noisy_t.squeeze().cpu().numpy()

    noisy_np = noisy_t.squeeze().cpu().numpy()
    cols = [arr, noisy_np, dn_out, vae_out]

    for col_i, img in enumerate(cols):
        ax = axes[row_i][col_i]
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if col_i > 0:
            p = compute_psnr(arr, img)
            s = compute_ssim(arr, img)
            ax.set_xlabel(f"PSNR={p:.2f} dB  SSIM={s:.3f}", fontsize=8)

    axes[row_i][0].set_ylabel(fname, fontsize=8, rotation=0, labelpad=65)

fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
print(f"Saved {OUT_PATH}")
