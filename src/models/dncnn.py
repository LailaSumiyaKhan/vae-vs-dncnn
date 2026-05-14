from __future__ import annotations

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """DnCNN: Beyond a Gaussian Denoiser (Zhang et al., 2017).

    Uses residual learning: clean_estimate = noisy - model(noisy).
    The network is trained to predict the noise component directly.
    """

    def __init__(
        self,
        num_layers: int = 17,
        num_features: int = 64,
        in_channels: int = 1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        # First layer: Conv + ReLU (no BatchNorm per original paper)
        layers += [
            nn.Conv2d(in_channels, num_features, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        ]
        # Middle layers: Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(num_features, num_features, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_features, eps=1e-4, momentum=0.95),
                nn.ReLU(inplace=True),
            ]
        # Final layer: Conv only (predicts noise residual)
        layers.append(nn.Conv2d(num_features, in_channels, 3, padding=1, bias=False))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the denoised image: x - estimated_noise."""
        return x - self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
