from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .dataset import DenoisingPatchDataset
from .models.dncnn import DnCNN
from .models.vae import DenoisingVAE
from .noise import add_noise


def _make_loader(config: Config) -> DataLoader:
    cfg = config.train
    ds = DenoisingPatchDataset(
        config.train_dir,
        patch_size=cfg.patch_size,
        patches_per_image=cfg.patches_per_image,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=config.device != "cpu",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )


def _checkpoint_path(config: Config, model_name: str, noise_type: str, sigma: int) -> Path:
    return config.results_dir / "checkpoints" / f"{model_name}_{noise_type}_{sigma}.pth"


def train_dncnn(config: Config, noise_type: str, sigma: int) -> dict:
    cfg = config.train
    device = config.device
    loader = _make_loader(config)

    model = DnCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.MSELoss()

    history: dict[str, list] = {"train_loss": [], "time_per_epoch": []}
    ckpt = _checkpoint_path(config, "dncnn", noise_type, sigma)
    start_epoch = 0

    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        history = state.get("history", history)
        if start_epoch >= cfg.epochs:
            print(f"[DnCNN {noise_type} σ={sigma}] already trained, skipping.")
            return {"model": model, "history": history, "params": model.count_parameters()}
        print(f"[DnCNN {noise_type} σ={sigma}] resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        running = 0.0

        bar = tqdm(loader, desc=f"DnCNN {noise_type} σ={sigma} [{epoch+1}/{cfg.epochs}]", leave=False)
        for clean in bar:
            clean = clean.to(device, non_blocking=True)
            noisy = add_noise(clean, noise_type, float(sigma))

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(noisy), clean)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            bar.set_postfix(loss=f"{loss.item():.5f}")

        scheduler.step()
        epoch_loss = running / len(loader)
        elapsed = time.perf_counter() - t0
        history["train_loss"].append(epoch_loss)
        history["time_per_epoch"].append(elapsed)

        if (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
            tqdm.write(
                f"[DnCNN {noise_type} σ={sigma}] epoch {epoch+1}/{cfg.epochs} "
                f"loss={epoch_loss:.6f}  {elapsed:.1f}s"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                },
                ckpt,
            )

    return {"model": model, "history": history, "params": model.count_parameters()}


def train_vae(config: Config, noise_type: str, sigma: int) -> dict:
    cfg = config.train
    device = config.device
    loader = _make_loader(config)

    model = DenoisingVAE(
        latent_dim=cfg.latent_dim,
        patch_size=cfg.patch_size,
        beta=cfg.beta,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history: dict[str, list] = {
        "train_loss": [],
        "recon_loss": [],
        "kl_loss": [],
        "time_per_epoch": [],
    }
    ckpt = _checkpoint_path(config, "vae", noise_type, sigma)
    start_epoch = 0

    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        history = state.get("history", history)
        if start_epoch >= cfg.epochs:
            print(f"[VAE {noise_type} σ={sigma}] already trained, skipping.")
            return {"model": model, "history": history, "params": model.count_parameters()}
        print(f"[VAE {noise_type} σ={sigma}] resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        tot = rec = kl = 0.0

        bar = tqdm(loader, desc=f"VAE {noise_type} σ={sigma} [{epoch+1}/{cfg.epochs}]", leave=False)
        for clean in bar:
            clean = clean.to(device, non_blocking=True)
            noisy = add_noise(clean, noise_type, float(sigma))

            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(noisy)
            loss, recon_l, kl_l = model.loss(recon, clean, mu, logvar)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot += loss.item()
            rec += recon_l.item()
            kl += kl_l.item()
            bar.set_postfix(loss=f"{loss.item():.5f}")

        scheduler.step()
        n = len(loader)
        elapsed = time.perf_counter() - t0
        history["train_loss"].append(tot / n)
        history["recon_loss"].append(rec / n)
        history["kl_loss"].append(kl / n)
        history["time_per_epoch"].append(elapsed)

        if (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
            tqdm.write(
                f"[VAE {noise_type} σ={sigma}] epoch {epoch+1}/{cfg.epochs} "
                f"loss={tot/n:.6f}  recon={rec/n:.6f}  kl={kl/n:.6f}  {elapsed:.1f}s"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                },
                ckpt,
            )

    return {"model": model, "history": history, "params": model.count_parameters()}
