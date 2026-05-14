from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import Config
from .dataset import DenoisingTestDataset
from .metrics import compute_all
from .models.dncnn import DnCNN
from .models.vae import DenoisingVAE
from .noise import add_noise


def _vae_denoise_full(
    model: DenoisingVAE, noisy: torch.Tensor, patch_size: int
) -> torch.Tensor:
    """Patch-based inference for VAE on arbitrary-size images (no overlap)."""
    _, _, H, W = noisy.shape
    pad_h = (-H) % patch_size
    pad_w = (-W) % patch_size
    x = F.pad(noisy, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, Hp, Wp = x.shape

    out = torch.zeros_like(x)
    for i in range(0, Hp, patch_size):
        for j in range(0, Wp, patch_size):
            patch = x[:, :, i : i + patch_size, j : j + patch_size]
            out[:, :, i : i + patch_size, j : j + patch_size] = model.denoise(patch)

    return out[:, :, :H, :W]


def _load_dncnn(config: Config, noise_type: str, sigma: int) -> DnCNN | None:
    ckpt = config.results_dir / "checkpoints" / f"dncnn_{noise_type}_{sigma}.pth"
    if not ckpt.exists():
        print(f"[eval] DnCNN {noise_type} σ={sigma}: checkpoint not found, skipping.")
        return None
    m = DnCNN().to(config.device)
    state = torch.load(ckpt, map_location=config.device, weights_only=True)
    m.load_state_dict(state["model"])
    m.eval()
    return m


def _load_vae(config: Config, noise_type: str, sigma: int) -> DenoisingVAE | None:
    ckpt = config.results_dir / "checkpoints" / f"vae_{noise_type}_{sigma}.pth"
    if not ckpt.exists():
        print(f"[eval] VAE {noise_type} σ={sigma}: checkpoint not found, skipping.")
        return None
    m = DenoisingVAE(
        latent_dim=config.train.latent_dim,
        patch_size=config.train.patch_size,
        beta=config.train.beta,
    ).to(config.device)
    state = torch.load(ckpt, map_location=config.device, weights_only=True)
    m.load_state_dict(state["model"])
    m.eval()
    return m


def _time_inference(fn, warmup: int = 3, runs: int = 10) -> float:
    """Return mean inference time in milliseconds over `runs` measured calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def evaluate_single_config(
    config: Config,
    noise_type: str,
    sigma: int,
    measure_speed: bool = True,
) -> list[dict]:
    """Evaluate DnCNN and VAE for one noise_type/sigma on the full test set."""
    device = config.device
    patch_size = config.train.patch_size
    dataset = DenoisingTestDataset(config.test_dir)

    dncnn = _load_dncnn(config, noise_type, sigma)
    vae = _load_vae(config, noise_type, sigma)

    if dncnn is None and vae is None:
        return []

    rows: list[dict] = []

    for clean_t, fname in dataset:
        clean_t = clean_t.unsqueeze(0).to(device)          # (1,1,H,W)
        noisy_t = add_noise(clean_t, noise_type, float(sigma))

        clean_np = clean_t.squeeze().cpu().numpy()
        noisy_np = noisy_t.squeeze().cpu().numpy()

        # Noisy baseline metrics (same for both models per image)
        noisy_m = compute_all(clean_np, noisy_np, device="cpu")

        for model_name, model in [("dncnn", dncnn), ("vae", vae)]:
            if model is None:
                continue

            with torch.no_grad():
                if model_name == "dncnn":
                    denoised_t = model(noisy_t)
                else:
                    denoised_t = _vae_denoise_full(model, noisy_t, patch_size)

            denoised_np = denoised_t.squeeze().cpu().numpy()
            m = compute_all(clean_np, denoised_np, device="cpu")

            # Measure inference speed on first image only (representative)
            inf_ms: float | None = None
            if measure_speed and fname == dataset._paths[0].name:
                if model_name == "dncnn":
                    inf_ms = _time_inference(lambda: model(noisy_t))
                else:
                    inf_ms = _time_inference(
                        lambda: _vae_denoise_full(model, noisy_t, patch_size)
                    )

            rows.append(
                {
                    "model": model_name,
                    "noise_type": noise_type,
                    "sigma": sigma,
                    "image": fname,
                    "psnr": m["psnr"],
                    "ssim": m["ssim"],
                    "lpips": m["lpips"],
                    "psnr_noisy": noisy_m["psnr"],
                    "ssim_noisy": noisy_m["ssim"],
                    "lpips_noisy": noisy_m["lpips"],
                    "inference_ms": inf_ms,
                }
            )

    return rows


def run_full_evaluation(config: Config) -> pd.DataFrame:
    all_rows: list[dict] = []
    for noise_type in config.noise_types:
        for sigma in config.sigma_levels:
            print(f"\n--- Evaluating: {noise_type} σ={sigma} ---")
            rows = evaluate_single_config(config, noise_type, sigma)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("Warning: no evaluation results produced (no checkpoints found?).")
        return df

    out = config.results_dir / "metrics" / "results.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved → {out}")

    # Print summary table
    summary = (
        df.groupby(["model", "noise_type", "sigma"])[["psnr", "ssim", "lpips"]]
        .mean()
        .round(4)
    )
    print("\n=== Summary (mean over test set) ===")
    print(summary.to_string())
    return df
