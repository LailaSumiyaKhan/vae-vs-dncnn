from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim

_lpips_net: object | None = None


def _get_lpips(device: str) -> object:
    global _lpips_net
    if _lpips_net is None:
        import lpips
        _lpips_net = lpips.LPIPS(net="alex", verbose=False).eval()
    return _lpips_net.to(device)  # type: ignore[union-attr]


def compute_psnr(clean: np.ndarray, denoised: np.ndarray) -> float:
    return float(_psnr(clean, np.clip(denoised, 0.0, 1.0), data_range=1.0))


def compute_ssim(clean: np.ndarray, denoised: np.ndarray) -> float:
    return float(_ssim(clean, np.clip(denoised, 0.0, 1.0), data_range=1.0))


def compute_lpips(
    clean: np.ndarray, denoised: np.ndarray, device: str = "cpu"
) -> float:
    fn = _get_lpips(device)

    def _prep(arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.clip(arr, 0.0, 1.0)).float()
        if t.dim() == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:
            t = t.unsqueeze(0)
        if t.shape[1] == 1:
            t = t.expand(-1, 3, -1, -1)
        return (t * 2.0 - 1.0).to(device)

    with torch.no_grad():
        return float(fn(_prep(clean), _prep(denoised)).item())  # type: ignore[operator]


def compute_all(
    clean: np.ndarray,
    denoised: np.ndarray,
    device: str = "cpu",
) -> dict[str, float]:
    return {
        "psnr": compute_psnr(clean, denoised),
        "ssim": compute_ssim(clean, denoised),
        "lpips": compute_lpips(clean, denoised, device=device),
    }
