from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


NOISE_TYPES: list[str] = ["gaussian", "poisson", "speckle"]
SIGMA_LEVELS: list[int] = [15, 25, 50]


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    patch_size: int = 64
    patches_per_image: int = 200
    num_workers: int = 4
    seed: int = 42
    # VAE-specific
    latent_dim: int = 256
    beta: float = 1e-4


@dataclass
class Config:
    train_dir: Path = field(default_factory=lambda: Path("data/train"))
    test_dir: Path = field(default_factory=lambda: Path("data/test"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "auto"
    noise_types: list[str] = field(default_factory=lambda: list(NOISE_TYPES))
    sigma_levels: list[int] = field(default_factory=lambda: list(SIGMA_LEVELS))

    def __post_init__(self) -> None:
        import torch

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        for sub in ("checkpoints", "metrics", "figures"):
            (self.results_dir / sub).mkdir(parents=True, exist_ok=True)
