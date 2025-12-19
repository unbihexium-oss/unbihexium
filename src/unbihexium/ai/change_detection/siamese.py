"""Siamese network for change detection.

This module implements a Siamese architecture for detecting changes
between two temporal images of the same location.

Architecture:
    Two identical encoder branches share weights and produce feature maps.
    A difference/fusion module combines the features.
    A decoder produces a binary change mask.

Applications:
    - Urban expansion monitoring
    - Deforestation detection
    - Disaster damage assessment
    - Infrastructure monitoring
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


@dataclass
class SiameseConfig:
    """Configuration for Siamese change detector.

    Attributes:
        in_channels: Number of input channels per image.
        base_features: Base feature count for encoder.
        num_classes: Number of output classes (2 for binary change).
    """

    in_channels: int = 3
    base_features: int = 32
    num_classes: int = 2


if TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        """Convolutional block: Conv -> BN -> ReLU."""

        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    class SiameseEncoder(nn.Module):
        """Shared encoder for both temporal images."""

        def __init__(self, in_channels: int, base_features: int) -> None:
            super().__init__()
            self.conv1 = ConvBlock(in_channels, base_features)
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = ConvBlock(base_features, base_features * 2)
            self.pool2 = nn.MaxPool2d(2)
            self.conv3 = ConvBlock(base_features * 2, base_features * 4)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass returning multi-scale features."""
            f1 = self.conv1(x)
            f2 = self.conv2(self.pool1(f1))
            f3 = self.conv3(self.pool2(f2))
            return f1, f2, f3

    class SiameseDecoder(nn.Module):
        """Decoder to produce change mask from difference features."""

        def __init__(self, base_features: int, num_classes: int) -> None:
            super().__init__()
            self.up1 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, stride=2)
            self.conv1 = ConvBlock(base_features * 8, base_features * 4)
            self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, stride=2)
            self.conv2 = ConvBlock(base_features * 4, base_features * 2)
            self.out = nn.Conv2d(base_features * 2, num_classes, 1)

        def forward(
            self,
            f3: torch.Tensor,
            f2: torch.Tensor,
            f1: torch.Tensor,
        ) -> torch.Tensor:
            x = self.up1(f3)
            x = torch.cat([x, f2], dim=1)
            x = self.conv1(x)
            x = self.up2(x)
            x = torch.cat([x, f1], dim=1)
            x = self.conv2(x)
            return self.out(x)

    class SiameseChangeDetector(nn.Module):
        """Siamese network for change detection.

        Takes two co-registered images and outputs a change mask.
        """

        def __init__(self, config: SiameseConfig | None = None) -> None:
            """Initialize Siamese change detector.

            Args:
                config: Model configuration.
            """
            super().__init__()
            cfg = config or SiameseConfig()
            self.config = cfg

            self.encoder = SiameseEncoder(cfg.in_channels, cfg.base_features)
            self.decoder = SiameseDecoder(cfg.base_features, cfg.num_classes)

        def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass.

            Args:
                x1: First temporal image (B, C, H, W).
                x2: Second temporal image (B, C, H, W).

            Returns:
                Change mask logits (B, num_classes, H, W).
            """
            f1_1, f2_1, f3_1 = self.encoder(x1)
            f1_2, f2_2, f3_2 = self.encoder(x2)

            diff_f1 = torch.abs(f1_1 - f1_2)
            diff_f2 = torch.abs(f2_1 - f2_2)
            diff_f3 = torch.cat([f3_1, f3_2], dim=1)

            return self.decoder(diff_f3, diff_f2, diff_f1)

        def count_parameters(self) -> int:
            """Count trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:

    class SiameseChangeDetector:
        """Stub when PyTorch not available."""

        def __init__(self, config: Any = None) -> None:
            raise ImportError("PyTorch required for SiameseChangeDetector")
