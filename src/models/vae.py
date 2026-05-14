from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Encoder(nn.Module):
    def __init__(self, in_ch: int, latent_dim: int, spatial: int) -> None:
        super().__init__()
        # 64→32→16→8→4  (4 stride-2 convolutions)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1),          # 32×32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),              # 16×16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),             # 8×8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),            # 4×4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        flat = 256 * spatial * spatial
        self.fc = nn.Linear(flat, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_lv = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(1)
        h = F.leaky_relu(self.fc(h), 0.2)
        return self.fc_mu(h), self.fc_lv(h)


class _Decoder(nn.Module):
    def __init__(self, in_ch: int, latent_dim: int, spatial: int) -> None:
        super().__init__()
        self.spatial = spatial
        flat = 256 * spatial * spatial
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, flat),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8×8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16×16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32×32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, in_ch, 4, 2, 1), # 64×64
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, self.spatial, self.spatial)
        return self.deconv(h)


class DenoisingVAE(nn.Module):
    """Convolutional VAE for image denoising.

    Input: noisy patch → reconstructed clean patch.
    Loss: MSE(recon, clean) + beta * KL(q(z|noisy) || N(0,I))
    At inference, the decoder mean (mu) is used directly (no sampling).
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 256,
        patch_size: int = 64,
        beta: float = 1e-4,
    ) -> None:
        super().__init__()
        spatial = patch_size // 16   # 4 for patch_size=64
        self.encoder = _Encoder(in_channels, latent_dim, spatial)
        self.decoder = _Decoder(in_channels, latent_dim, spatial)
        self.beta = beta

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu, _ = self.encoder(x)
            return self.decoder(mu)

    def loss(
        self,
        recon: torch.Tensor,
        clean: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(recon, clean)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl, recon_loss, kl

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
