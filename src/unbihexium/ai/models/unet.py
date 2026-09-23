# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""UNet architecture for semantic segmentation.

This module implements a configurable UNet architecture for:
- Land cover classification
- Building footprint extraction
- Water body segmentation
- Change detection

Architecture:
    Encoder (contracting path): Conv -> Conv -> MaxPool
    Decoder (expanding path): UpConv -> Concat -> Conv -> Conv

The skip connections between encoder and decoder preserve spatial information.
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
class UNetConfig:
    """Configuration for UNet model.

    Attributes:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        features: List of feature sizes for each encoder level.
        bilinear: Whether to use bilinear upsampling (vs transposed conv).
    """

    in_channels: int = 3
    num_classes: int = 2
    features: tuple[int, ...] = (64, 128, 256, 512)
    bilinear: bool = True


if TORCH_AVAILABLE:

    class DoubleConv(nn.Module):
        """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.double_conv(x)

    class Down(nn.Module):
        """Downscaling block: MaxPool -> DoubleConv."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.maxpool_conv(x)

    class Up(nn.Module):
        """Upscaling block: Upsample -> Concat -> DoubleConv."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bilinear: bool = True,
        ) -> None:
            super().__init__()

            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                self.conv = DoubleConv(in_channels, out_channels)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels)

        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            x1 = self.up(x1)

            # Handle size mismatch
            diff_y = x2.size()[2] - x1.size()[2]
            diff_x = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class UNet(nn.Module):
        """UNet semantic segmentation model.

        A popular encoder-decoder architecture with skip connections.
        """

        def __init__(self, config: UNetConfig | None = None) -> None:
            """Initialize UNet.

            Args:
                config: Model configuration. Uses defaults if None.
            """
            super().__init__()
            cfg = config or UNetConfig()
            self.config = cfg

            features = cfg.features
            self.inc = DoubleConv(cfg.in_channels, features[0])

            # Encoder
            self.down1 = Down(features[0], features[1])
            self.down2 = Down(features[1], features[2])
            self.down3 = Down(features[2], features[3])

            factor = 2 if cfg.bilinear else 1
            self.down4 = Down(features[3], features[3] * 2 // factor)

            # Decoder
            self.up1 = Up(features[3] * 2, features[3] // factor, cfg.bilinear)
            self.up2 = Up(features[3], features[2] // factor, cfg.bilinear)
            self.up3 = Up(features[2], features[1] // factor, cfg.bilinear)
            self.up4 = Up(features[1], features[0], cfg.bilinear)

            # Output
            self.outc = nn.Conv2d(features[0], cfg.num_classes, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor of shape (B, C, H, W).

            Returns:
                Output tensor of shape (B, num_classes, H, W).
            """
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

            return self.outc(x)

        def count_parameters(self) -> int:
            """Count trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:

    class UNet:
        """UNet stub when PyTorch is not available."""

        def __init__(self, config: UNetConfig | None = None) -> None:
            raise ImportError("PyTorch is required for UNet. Install with: pip install torch")
