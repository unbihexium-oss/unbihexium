# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ResNet backbone for feature extraction and classification.

This module implements ResNet architectures for:
- Scene classification
- Feature extraction for detection heads
- Transfer learning backbone

Architectures:
    - ResNet18: 18 layers, ~11M parameters
    - ResNet34: 34 layers, ~21M parameters
    - ResNet50: 50 layers, ~25M parameters (bottleneck)
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
class ResNetConfig:
    """Configuration for ResNet model.

    Attributes:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        variant: ResNet variant (18, 34, or 50).
        pretrained: Whether to use pretrained weights.
    """

    in_channels: int = 3
    num_classes: int = 10
    variant: int = 18
    pretrained: bool = False


if TORCH_AVAILABLE:

    class BasicBlock(nn.Module):
        """Basic residual block for ResNet18/34."""

        expansion = 1

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: nn.Module | None = None,
        ) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = x

            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return F.relu(out)

    class Bottleneck(nn.Module):
        """Bottleneck residual block for ResNet50+."""

        expansion = 4

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: nn.Module | None = None,
        ) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv3 = nn.Conv2d(
                out_channels,
                out_channels * self.expansion,
                kernel_size=1,
                bias=False,
            )
            self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
            self.downsample = downsample

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = x

            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return F.relu(out)

    class ResNet(nn.Module):
        """ResNet classification/feature extraction model."""

        def __init__(self, config: ResNetConfig | None = None) -> None:
            """Initialize ResNet.

            Args:
                config: Model configuration. Uses defaults if None.
            """
            super().__init__()
            cfg = config or ResNetConfig()
            self.config = cfg

            # Select block type and layers based on variant
            if cfg.variant == 18:
                block = BasicBlock
                layers = [2, 2, 2, 2]
            elif cfg.variant == 34:
                block = BasicBlock
                layers = [3, 4, 6, 3]
            elif cfg.variant == 50:
                block = Bottleneck
                layers = [3, 4, 6, 3]
            else:
                block = BasicBlock
                layers = [2, 2, 2, 2]

            self.in_channels = 64

            # Stem
            self.conv1 = nn.Conv2d(
                cfg.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # Residual layers
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

            # Classifier
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, cfg.num_classes)

        def _make_layer(
            self,
            block: type[BasicBlock] | type[Bottleneck],
            out_channels: int,
            blocks: int,
            stride: int = 1,
        ) -> nn.Sequential:
            """Create a residual layer."""
            downsample = None
            if stride != 1 or self.in_channels != out_channels * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.in_channels,
                        out_channels * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )

            layers = [block(self.in_channels, out_channels, stride, downsample)]
            self.in_channels = out_channels * block.expansion

            for _ in range(1, blocks):
                layers.append(block(self.in_channels, out_channels))

            return nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor of shape (B, C, H, W).

            Returns:
                Class logits of shape (B, num_classes).
            """
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

        def extract_features(self, x: torch.Tensor) -> torch.Tensor:
            """Extract features before classifier.

            Args:
                x: Input tensor.

            Returns:
                Feature tensor.
            """
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            return torch.flatten(x, 1)

        def count_parameters(self) -> int:
            """Count trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:

    class ResNet:
        """ResNet stub when PyTorch is not available."""

        def __init__(self, config: ResNetConfig | None = None) -> None:
            raise ImportError("PyTorch is required for ResNet. Install with: pip install torch")
