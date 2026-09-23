# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Detection and segmentation heads for AI models.

This module provides task-specific heads that can be attached to
backbone networks for different computer vision tasks.

Heads:
    - DetectionHead: Object detection (bounding boxes + classes)
    - SegmentationHead: Semantic segmentation (pixel-wise classification)
    - ClassificationHead: Image classification
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
class DetectionHeadConfig:
    """Configuration for detection head.

    Attributes:
        in_channels: Number of input channels from backbone.
        num_classes: Number of object classes.
        num_anchors: Number of anchor boxes per location.
    """

    in_channels: int = 512
    num_classes: int = 10
    num_anchors: int = 9


@dataclass
class SegmentationHeadConfig:
    """Configuration for segmentation head.

    Attributes:
        in_channels: Number of input channels from backbone.
        num_classes: Number of segmentation classes.
        hidden_channels: Hidden layer channels.
    """

    in_channels: int = 512
    num_classes: int = 2
    hidden_channels: int = 256


if TORCH_AVAILABLE:

    class DetectionHead(nn.Module):
        """Detection head for object detection.

        Produces bounding box regression and class predictions.
        """

        def __init__(self, config: DetectionHeadConfig | None = None) -> None:
            """Initialize detection head.

            Args:
                config: Head configuration.
            """
            super().__init__()
            cfg = config or DetectionHeadConfig()
            self.config = cfg

            # Shared layers
            self.conv = nn.Sequential(
                nn.Conv2d(cfg.in_channels, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

            # Class prediction
            self.cls_head = nn.Conv2d(
                256, cfg.num_anchors * cfg.num_classes, kernel_size=3, padding=1
            )

            # Box regression (4 values: x, y, w, h)
            self.box_head = nn.Conv2d(256, cfg.num_anchors * 4, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                x: Feature map from backbone (B, C, H, W).

            Returns:
                Tuple of (class_logits, box_regression).
            """
            features = self.conv(x)
            cls_logits = self.cls_head(features)
            box_reg = self.box_head(features)
            return cls_logits, box_reg

    class SegmentationHead(nn.Module):
        """Segmentation head for semantic segmentation.

        Produces pixel-wise class predictions.
        """

        def __init__(self, config: SegmentationHeadConfig | None = None) -> None:
            """Initialize segmentation head.

            Args:
                config: Head configuration.
            """
            super().__init__()
            cfg = config or SegmentationHeadConfig()
            self.config = cfg

            self.conv = nn.Sequential(
                nn.Conv2d(cfg.in_channels, cfg.hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(cfg.hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(cfg.hidden_channels, cfg.hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(cfg.hidden_channels),
                nn.ReLU(inplace=True),
            )

            self.classifier = nn.Conv2d(cfg.hidden_channels, cfg.num_classes, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Feature map from backbone (B, C, H, W).

            Returns:
                Class logits (B, num_classes, H, W).
            """
            features = self.conv(x)
            return self.classifier(features)

    class ClassificationHead(nn.Module):
        """Classification head for image classification."""

        def __init__(
            self,
            in_features: int = 512,
            num_classes: int = 10,
            dropout: float = 0.5,
        ) -> None:
            """Initialize classification head.

            Args:
                in_features: Number of input features.
                num_classes: Number of classes.
                dropout: Dropout rate.
            """
            super().__init__()
            self.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Feature vector (B, in_features).

            Returns:
                Class logits (B, num_classes).
            """
            return self.fc(x)

else:

    class DetectionHead:
        """Stub when PyTorch not available."""

        def __init__(self, config: Any = None) -> None:
            raise ImportError("PyTorch required")

    class SegmentationHead:
        """Stub when PyTorch not available."""

        def __init__(self, config: Any = None) -> None:
            raise ImportError("PyTorch required")

    class ClassificationHead:
        """Stub when PyTorch not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch required")
