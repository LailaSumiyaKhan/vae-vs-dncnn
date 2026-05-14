from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from .config import Config, NOISE_TYPES, SIGMA_LEVELS
from .dataset import DenoisingTestDataset
from .models.dncnn import DnCNN
from .models.vae import DenoisingVAE
from .noise import add_noise
from .evaluator import _vae_denoise_full, _load_dncnn, _load_vae

_PALETTE = {"dncnn": "#2196F3", "vae": "#FF5722"}
_FIG_DIR_NAME = "figures"


def _fig_path(config: Config, name: str) -> Path:
    return config.results_dir / _FIG_DIR_NAME / name


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


# ---------------------------------------------------------------------------
# 1. Training loss curves
# ---------------------------------------------------------------------------

def plot_training_curves(config: Config) -> None:
    """One subplot per (noise_type, sigma) showing DnCNN vs VAE loss."""
    n_rows, n_cols = len(NOISE_TYPES), len(SIGMA_LEVELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle("Training Loss Curves — DnCNN vs VAE", fontsize=16, y=1.01)

    for r, noise_type in enumerate(NOISE_TYPES):
        for c, sigma in enumerate(SIGMA_LEVELS):
            ax = axes[r][c]
            for model_name, color in _PALETTE.items():
                ckpt = config.results_dir / "checkpoints" / f"{model_name}_{noise_type}_{sigma}.pth"
                if not ckpt.exists():
                    continue
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                history = state.get("history", {})
                losses = history.get("train_loss", [])
                if losses:
                    ax.plot(
                        range(1, len(losses) + 1),
                        losses,
                        color=color,
                        label=model_name.upper(),
                        linewidth=1.8,
                    )
            ax.set_title(f"{noise_type.capitalize()} σ={sigma}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, _fig_path(config, "training_curves.png"))


# ---------------------------------------------------------------------------
# 2. Metrics bar charts (per noise type, grouped by sigma)
# ---------------------------------------------------------------------------

def plot_metrics_bar(df: pd.DataFrame, config: Config) -> None:
    """3×3 grid: rows=metric (PSNR/SSIM/LPIPS), cols=noise_type; bars=sigma groups."""
    metrics = ["psnr", "ssim", "lpips"]
    metric_labels = ["PSNR (dB) ↑", "SSIM ↑", "LPIPS ↓"]
    n_rows, n_cols = len(metrics), len(NOISE_TYPES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle("Denoising Metrics: DnCNN vs VAE", fontsize=16, y=1.01)

    x = np.arange(len(SIGMA_LEVELS))
    width = 0.35

    for r, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        for c, noise_type in enumerate(NOISE_TYPES):
            ax = axes[r][c]
            sub = df[df["noise_type"] == noise_type]
            for i, (model_name, color) in enumerate(_PALETTE.items()):
                vals = [
                    sub[(sub["model"] == model_name) & (sub["sigma"] == s)][metric].mean()
                    for s in SIGMA_LEVELS
                ]
                offset = (i - 0.5) * width
                bars = ax.bar(x + offset, vals, width, label=model_name.upper(), color=color, alpha=0.85)
                for bar, v in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.002 if metric != "lpips" else 0.0005),
                        f"{v:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels([f"σ={s}" for s in SIGMA_LEVELS])
            ax.set_title(f"{noise_type.capitalize()} — {mlabel}", fontsize=10)
            ax.set_ylabel(mlabel)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, _fig_path(config, "metrics_bar.png"))


# ---------------------------------------------------------------------------
# 3. Performance vs sigma line plots
# ---------------------------------------------------------------------------

def plot_metrics_vs_sigma(df: pd.DataFrame, config: Config) -> None:
    """Line plots: metric vs sigma for each noise type."""
    metrics = ["psnr", "ssim", "lpips"]
    metric_labels = ["PSNR (dB) ↑", "SSIM ↑", "LPIPS ↓"]

    fig, axes = plt.subplots(len(metrics), len(NOISE_TYPES), figsize=(5 * len(NOISE_TYPES), 4 * len(metrics)))
    fig.suptitle("Metric vs Noise Level (σ)", fontsize=16, y=1.01)

    for r, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        for c, noise_type in enumerate(NOISE_TYPES):
            ax = axes[r][c]
            sub = df[df["noise_type"] == noise_type]
            for model_name, color in _PALETTE.items():
                m_sub = sub[sub["model"] == model_name]
                vals = [m_sub[m_sub["sigma"] == s][metric].mean() for s in SIGMA_LEVELS]
                ax.plot(SIGMA_LEVELS, vals, "o-", color=color, label=model_name.upper(), linewidth=2, markersize=7)

            ax.set_xticks(SIGMA_LEVELS)
            ax.set_xlabel("Sigma (σ)")
            ax.set_ylabel(mlabel)
            ax.set_title(f"{noise_type.capitalize()} — {mlabel}", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, _fig_path(config, "metrics_vs_sigma.png"))


# ---------------------------------------------------------------------------
# 4. Heatmaps (PSNR, SSIM, LPIPS) for each model
# ---------------------------------------------------------------------------

def plot_heatmaps(df: pd.DataFrame, config: Config) -> None:
    """Heatmap: noise_type × sigma for each metric and model."""
    metrics = ["psnr", "ssim", "lpips"]
    models = ["dncnn", "vae"]

    for metric in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"{metric.upper()} Heatmap (noise type × sigma)", fontsize=13)

        for ax, model_name in zip(axes, models):
            pivot = (
                df[df["model"] == model_name]
                .groupby(["noise_type", "sigma"])[metric]
                .mean()
                .unstack()
            )
            annot = pivot.round(3).astype(str)
            sns.heatmap(
                pivot,
                ax=ax,
                annot=annot,
                fmt="",
                cmap="RdYlGn" if metric != "lpips" else "RdYlGn_r",
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(model_name.upper())
            ax.set_xlabel("Sigma")
            ax.set_ylabel("Noise Type")

        fig.tight_layout()
        _save(fig, _fig_path(config, f"heatmap_{metric}.png"))


# ---------------------------------------------------------------------------
# 5. Qualitative side-by-side comparisons
# ---------------------------------------------------------------------------

def plot_denoising_examples(config: Config, n_images: int = 4) -> None:
    """For each (noise_type, sigma): show clean / noisy / DnCNN / VAE side-by-side."""
    dataset = DenoisingTestDataset(config.test_dir)
    device = config.device

    # Pick evenly-spaced image indices
    indices = np.linspace(0, len(dataset) - 1, n_images, dtype=int).tolist()

    for noise_type in NOISE_TYPES:
        for sigma in SIGMA_LEVELS:
            dncnn = _load_dncnn(config, noise_type, sigma)
            vae = _load_vae(config, noise_type, sigma)
            if dncnn is None and vae is None:
                continue

            fig, axes = plt.subplots(
                n_images, 4, figsize=(16, 4 * n_images)
            )
            if n_images == 1:
                axes = axes[np.newaxis, :]

            fig.suptitle(
                f"Denoising: {noise_type.capitalize()} σ={sigma}",
                fontsize=15,
                y=1.01,
            )
            col_titles = ["Clean", "Noisy", "DnCNN", "VAE"]
            for ax, t in zip(axes[0], col_titles):
                ax.set_title(t, fontsize=12, fontweight="bold")

            for row_idx, img_idx in enumerate(indices):
                clean_t, fname = dataset[img_idx]
                clean_t = clean_t.unsqueeze(0).to(device)
                noisy_t = add_noise(clean_t, noise_type, float(sigma))

                with torch.no_grad():
                    if dncnn is not None:
                        dn_out = dncnn(noisy_t).squeeze().cpu().numpy()
                    else:
                        dn_out = noisy_t.squeeze().cpu().numpy()

                    if vae is not None:
                        vae_out = _vae_denoise_full(
                            vae, noisy_t, config.train.patch_size
                        ).squeeze().cpu().numpy()
                    else:
                        vae_out = noisy_t.squeeze().cpu().numpy()

                clean_np = clean_t.squeeze().cpu().numpy()
                noisy_np = noisy_t.squeeze().cpu().numpy()

                images = [clean_np, noisy_np, dn_out, vae_out]
                for col, (img, ax) in enumerate(zip(images, axes[row_idx])):
                    ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
                    ax.axis("off")
                    if col > 0:
                        from .metrics import compute_psnr, compute_ssim
                        p = compute_psnr(clean_np, img)
                        s = compute_ssim(clean_np, img)
                        ax.set_xlabel(f"PSNR={p:.2f}  SSIM={s:.3f}", fontsize=8)

                axes[row_idx][0].set_ylabel(fname, fontsize=8, rotation=0, labelpad=60)

            fig.tight_layout()
            _save(
                fig,
                _fig_path(config, f"examples_{noise_type}_sigma{sigma}.png"),
            )


# ---------------------------------------------------------------------------
# 6. Computational efficiency comparison
# ---------------------------------------------------------------------------

def plot_efficiency(df: pd.DataFrame, config: Config) -> None:
    """Bar chart comparing parameter count and average inference time."""
    device = config.device

    dncnn_tmp = DnCNN().to(device)
    vae_tmp = DenoisingVAE(
        latent_dim=config.train.latent_dim,
        patch_size=config.train.patch_size,
        beta=config.train.beta,
    ).to(device)
    params = {
        "DnCNN": dncnn_tmp.count_parameters(),
        "VAE": vae_tmp.count_parameters(),
    }
    del dncnn_tmp, vae_tmp

    # Extract per-image inference speeds from df (first image where measured)
    speed = {}
    for mname in ["dncnn", "vae"]:
        col = df[df["model"] == mname]["inference_ms"].dropna()
        speed[mname.upper()] = col.mean() if not col.empty else float("nan")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Computational Efficiency Comparison", fontsize=14)

    colors = [_PALETTE["dncnn"], _PALETTE["vae"]]

    bars1 = ax1.bar(list(params.keys()), list(params.values()), color=colors, alpha=0.85)
    ax1.set_ylabel("Number of Parameters")
    ax1.set_title("Model Size")
    for bar in bars1:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{bar.get_height()/1e3:.1f}K",
            ha="center",
            fontsize=10,
        )
    ax1.grid(axis="y", alpha=0.3)

    s_keys = list(speed.keys())
    s_vals = [speed[k] for k in s_keys]
    bars2 = ax2.bar(s_keys, s_vals, color=colors, alpha=0.85)
    ax2.set_ylabel("Inference Time (ms)")
    ax2.set_title("Inference Speed per Image")
    for bar, v in zip(bars2, s_vals):
        if not np.isnan(v):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{v:.1f}ms",
                ha="center",
                fontsize=10,
            )
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, _fig_path(config, "efficiency.png"))


# ---------------------------------------------------------------------------
# 7. VAE latent-space analysis (training loss components)
# ---------------------------------------------------------------------------

def plot_vae_loss_components(config: Config) -> None:
    """Plot reconstruction vs KL loss for VAE training."""
    fig, axes = plt.subplots(
        len(NOISE_TYPES), len(SIGMA_LEVELS),
        figsize=(5 * len(SIGMA_LEVELS), 4 * len(NOISE_TYPES)),
    )
    fig.suptitle("VAE Training: Reconstruction vs KL Loss", fontsize=15, y=1.01)

    for r, noise_type in enumerate(NOISE_TYPES):
        for c, sigma in enumerate(SIGMA_LEVELS):
            ax = axes[r][c]
            ckpt = config.results_dir / "checkpoints" / f"vae_{noise_type}_{sigma}.pth"
            if ckpt.exists():
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                h = state.get("history", {})
                epochs = range(1, len(h.get("recon_loss", [])) + 1)
                ax.plot(epochs, h.get("recon_loss", []), label="Recon", color="#E91E63", linewidth=1.8)
                ax.plot(epochs, h.get("kl_loss", []), label="KL", color="#9C27B0", linewidth=1.8, linestyle="--")
            ax.set_title(f"{noise_type.capitalize()} σ={sigma}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, _fig_path(config, "vae_loss_components.png"))


# ---------------------------------------------------------------------------
# 8. Summary radar / spider chart
# ---------------------------------------------------------------------------

def plot_radar_summary(df: pd.DataFrame, config: Config) -> None:
    """Radar chart comparing DnCNN vs VAE across all noise configs."""
    # Normalize metrics to [0,1] for radar (higher always better)
    categories = []
    dncnn_vals = []
    vae_vals = []

    for noise_type in NOISE_TYPES:
        for sigma in SIGMA_LEVELS:
            sub = df[df["noise_type"] == noise_type]
            for metric, higher_better in [("psnr", True), ("ssim", True), ("lpips", False)]:
                dn_m = sub[(sub["model"] == "dncnn") & (sub["sigma"] == sigma)][metric].mean()
                vae_m = sub[(sub["model"] == "vae") & (sub["sigma"] == sigma)][metric].mean()
                categories.append(f"{noise_type[:3]}\nσ={sigma}\n{metric.upper()}")
                if higher_better:
                    rng = max(dn_m, vae_m) - min(dn_m, vae_m) + 1e-9
                    dncnn_vals.append((dn_m - min(dn_m, vae_m)) / rng)
                    vae_vals.append((vae_m - min(dn_m, vae_m)) / rng)
                else:
                    rng = max(dn_m, vae_m) - min(dn_m, vae_m) + 1e-9
                    dncnn_vals.append((max(dn_m, vae_m) - dn_m) / rng)
                    vae_vals.append((max(dn_m, vae_m) - vae_m) / rng)

    N = len(categories)
    if N == 0:
        return

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    dncnn_plot = dncnn_vals + dncnn_vals[:1]
    vae_plot = vae_vals + vae_vals[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    ax.plot(angles, dncnn_plot, "o-", color=_PALETTE["dncnn"], linewidth=2, label="DnCNN")
    ax.fill(angles, dncnn_plot, color=_PALETTE["dncnn"], alpha=0.15)
    ax.plot(angles, vae_plot, "s-", color=_PALETTE["vae"], linewidth=2, label="VAE")
    ax.fill(angles, vae_plot, color=_PALETTE["vae"], alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("DnCNN vs VAE — Normalized Performance Radar", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    fig.tight_layout()
    _save(fig, _fig_path(config, "radar_summary.png"))


# ---------------------------------------------------------------------------
# 9. Improvement over noisy input
# ---------------------------------------------------------------------------

def plot_improvement(df: pd.DataFrame, config: Config) -> None:
    """Bar chart: PSNR gain over noisy baseline for each model and noise config."""
    df = df.copy()
    df["psnr_gain"] = df["psnr"] - df["psnr_noisy"]
    df["ssim_gain"] = df["ssim"] - df["ssim_noisy"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Improvement Over Noisy Input", fontsize=14)

    for ax, metric, label in zip(
        axes,
        ["psnr_gain", "ssim_gain"],
        ["ΔPSNR (dB)", "ΔSSIM"],
    ):
        summary = (
            df.groupby(["model", "noise_type", "sigma"])[metric]
            .mean()
            .reset_index()
        )
        x = np.arange(len(NOISE_TYPES) * len(SIGMA_LEVELS))
        width = 0.35
        labels = [f"{n[:3]} σ={s}" for n in NOISE_TYPES for s in SIGMA_LEVELS]

        for i, (model_name, color) in enumerate(_PALETTE.items()):
            vals = [
                summary[
                    (summary["model"] == model_name)
                    & (summary["noise_type"] == nt)
                    & (summary["sigma"] == s)
                ][metric].values
                for nt in NOISE_TYPES
                for s in SIGMA_LEVELS
            ]
            vals = [v[0] if len(v) > 0 else 0.0 for v in vals]
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width, label=model_name.upper(), color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, _fig_path(config, "improvement_over_noisy.png"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all_visualizations(df: pd.DataFrame, config: Config) -> None:
    figs_dir = config.results_dir / _FIG_DIR_NAME
    figs_dir.mkdir(exist_ok=True)
    print("\n=== Generating visualizations ===")

    print("1/9 Training curves...")
    plot_training_curves(config)

    print("2/9 Metrics bar charts...")
    plot_metrics_bar(df, config)

    print("3/9 Metrics vs sigma line plots...")
    plot_metrics_vs_sigma(df, config)

    print("4/9 Heatmaps...")
    plot_heatmaps(df, config)

    print("5/9 Denoising examples...")
    plot_denoising_examples(config, n_images=4)

    print("6/9 Efficiency comparison...")
    plot_efficiency(df, config)

    print("7/9 VAE loss components...")
    plot_vae_loss_components(config)

    print("8/9 Radar chart...")
    plot_radar_summary(df, config)

    print("9/9 Improvement over noisy input...")
    plot_improvement(df, config)

    print(f"\nAll figures saved in {figs_dir}/")
