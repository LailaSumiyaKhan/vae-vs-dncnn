from __future__ import annotations

import torch


def add_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    noise = torch.randn_like(images) * (sigma / 255.0)
    return torch.clamp(images + noise, 0.0, 1.0)


def add_poisson_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    # Scale to photon counts calibrated to sigma, apply Poisson, scale back
    peak = (255.0 / sigma) ** 2
    photons = (images * peak).clamp(min=1e-6)
    noisy = torch.poisson(photons) / peak
    return torch.clamp(noisy, 0.0, 1.0)


def add_speckle_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    # Multiplicative noise: x_n = x * (1 + N(0, (sigma/255)^2))
    noise = torch.randn_like(images) * (sigma / 255.0)
    return torch.clamp(images + images * noise, 0.0, 1.0)


_NOISE_FNS = {
    "gaussian": add_gaussian_noise,
    "poisson": add_poisson_noise,
    "speckle": add_speckle_noise,
}


def add_noise(images: torch.Tensor, noise_type: str, sigma: float) -> torch.Tensor:
    try:
        fn = _NOISE_FNS[noise_type]
    except KeyError:
        raise ValueError(f"Unknown noise type '{noise_type}'. Choose from {list(_NOISE_FNS)}")
    return fn(images, sigma)
