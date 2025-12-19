"""Feature Pyramid Network (FPN) for multi-scale detection.

FPN builds a feature pyramid with strong semantics at all scales, enabling
detection of objects at multiple sizes.

Architecture:
    Bottom-up: Backbone (ResNet) extracts multi-scale features
    Top-down: Upsample higher-level features
    Lateral: 1x1 convs to match channel dimensions
    Merge: Element-wise addition
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
class FPNConfig:
    """Configuration for FPN.

    Attributes:
        in_channels_list: Channel counts from backbone stages.
        out_channels: Output channels for all pyramid levels.
        num_levels: Number of FPN levels.
    """

    in_channels_list: tuple[int, ...] = (256, 512, 1024, 2048)
    out_channels: int = 256
    num_levels: int = 5


if TORCH_AVAILABLE:

    class FPN(nn.Module):
        """Feature Pyramid Network.

        Combines features from a backbone at multiple scales.
        """

        def __init__(self, config: FPNConfig | None = None) -> None:
            """Initialize FPN.

            Args:
                config: FPN configuration.
            """
            super().__init__()
            cfg = config or FPNConfig()
            self.config = cfg

            self.lateral_convs = nn.ModuleList()
            self.output_convs = nn.ModuleList()

            for in_ch in cfg.in_channels_list:
                self.lateral_convs.append(
                    nn.Conv2d(in_ch, cfg.out_channels, 1)
                )
                self.output_convs.append(
                    nn.Conv2d(cfg.out_channels, cfg.out_channels, 3, padding=1)
                )

            extra_levels = cfg.num_levels - len(cfg.in_channels_list)
            if extra_levels > 0:
                self.extra_convs = nn.ModuleList()
                for i in range(extra_levels):
                    in_ch = cfg.in_channels_list[-1] if i == 0 else cfg.out_channels
                    self.extra_convs.append(
                        nn.Conv2d(in_ch, cfg.out_channels, 3, stride=2, padding=1)
                    )
            else:
                self.extra_convs = None

        def forward(
            self,
            features: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            """Forward pass.

            Args:
                features: List of feature maps from backbone.

            Returns:
                List of FPN feature maps.
            """
            laterals = [
                conv(f) for f, conv in zip(features, self.lateral_convs)
            ]

            for i in range(len(laterals) - 2, -1, -1):
                upsampled = F.interpolate(
                    laterals[i + 1],
                    size=laterals[i].shape[2:],
                    mode="nearest",
                )
                laterals[i] = laterals[i] + upsampled

            outputs = [
                conv(lat) for lat, conv in zip(laterals, self.output_convs)
            ]

            if self.extra_convs is not None:
                last = features[-1]
                for conv in self.extra_convs:
                    last = F.relu(conv(last))
                    outputs.append(last)

            return outputs

        def count_parameters(self) -> int:
            """Count trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:

    class FPN:
        """Stub when PyTorch not available."""

        def __init__(self, config: Any = None) -> None:
            raise ImportError("PyTorch required for FPN")
