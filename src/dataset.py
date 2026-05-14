from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _load_image_paths(directory: Path) -> list[Path]:
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMG_EXTS)
    if not paths:
        raise RuntimeError(f"No images found in {directory}")
    return paths


class DenoisingPatchDataset(Dataset):
    """Randomly-cropped patch dataset from training images (grayscale, [0,1])."""

    def __init__(
        self,
        image_dir: Path,
        patch_size: int = 64,
        patches_per_image: int = 200,
        augment: bool = True,
    ) -> None:
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment

        paths = _load_image_paths(Path(image_dir))
        self._images: list[np.ndarray] = []
        for p in paths:
            arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            # Ensure both dims are at least patch_size
            h, w = arr.shape
            if h < patch_size or w < patch_size:
                pad_h = max(0, patch_size - h)
                pad_w = max(0, patch_size - w)
                arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")
            self._images.append(arr)

        self._total = len(self._images) * patches_per_image

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self._images[idx // self.patches_per_image]
        ps = self.patch_size
        h, w = img.shape
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        patch = img[top : top + ps, left : left + ps].copy()

        if self.augment:
            if random.random() > 0.5:
                patch = patch[:, ::-1].copy()
            if random.random() > 0.5:
                patch = patch[::-1].copy()
            k = random.randint(0, 3)
            if k:
                patch = np.rot90(patch, k).copy()

        return torch.from_numpy(patch).unsqueeze(0)  # (1, H, W)


class DenoisingTestDataset(Dataset):
    """Full-resolution grayscale test images in [0,1]."""

    def __init__(self, image_dir: Path) -> None:
        self._paths = _load_image_paths(Path(image_dir))

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        p = self._paths[idx]
        arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0), p.name
