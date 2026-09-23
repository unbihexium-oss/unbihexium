# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""SRCNN (Super-Resolution Convolutional Neural Network) implementation.

This module implements the SRCNN architecture as described in:
"Image Super-Resolution Using Deep Convolutional Networks" (Dong et al., 2014).

SRCNN is a lightweight 3-layer CNN that learns end-to-end mapping from
low-resolution to high-resolution images.

Architecture:
    1. Patch extraction: 9x9 conv, 64 filters
    2. Non-linear mapping: 1x1 conv, 32 filters
    3. Reconstruction: 5x5 conv, output channels

Quality Metrics:
    PSNR = 10 * log10(MAX^2 / MSE)
    where MAX is the maximum pixel value and MSE is mean squared error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
class SRCNNConfig:
    """Configuration for SRCNN model.

    Attributes:
        input_channels: Number of input channels (1 for grayscale, 3 for RGB).
        f1: Number of filters in first convolution layer.
        f2: Number of filters in second convolution layer.
        scale_factor: Upscaling factor (2, 3, or 4).
    """

    input_channels: int = 3
    f1: int = 64
    f2: int = 32
    scale_factor: int = 2


if TORCH_AVAILABLE:

    class SRCNN(nn.Module):
        """SRCNN PyTorch implementation.

        A 3-layer CNN for single image super-resolution.
        """

        def __init__(self, config: SRCNNConfig | None = None) -> None:
            """Initialize SRCNN model.

            Args:
                config: Model configuration. Uses defaults if None.
            """
            super().__init__()
            cfg = config or SRCNNConfig()

            self.config = cfg
            self.scale_factor = cfg.scale_factor

            # Layer 1: Patch extraction and representation
            # 9x9 kernel, 64 filters
            self.conv1 = nn.Conv2d(
                cfg.input_channels,
                cfg.f1,
                kernel_size=9,
                padding=4,
            )

            # Layer 2: Non-linear mapping
            # 1x1 kernel, 32 filters
            self.conv2 = nn.Conv2d(
                cfg.f1,
                cfg.f2,
                kernel_size=1,
                padding=0,
            )

            # Layer 3: Reconstruction
            # 5x5 kernel, output channels
            self.conv3 = nn.Conv2d(
                cfg.f2,
                cfg.input_channels,
                kernel_size=5,
                padding=2,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor of shape (B, C, H, W).

            Returns:
                Super-resolved output of shape (B, C, H*scale, W*scale).
            """
            # Bicubic upscale first (SRCNN operates on upscaled LR image)
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode="bicubic",
                align_corners=False,
            )

            # 3-layer CNN
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)

            return x

        def count_parameters(self) -> int:
            """Count trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:
    # Stub class when PyTorch not available
    class SRCNN:
        """SRCNN stub when PyTorch is not available."""

        def __init__(self, config: SRCNNConfig | None = None) -> None:
            raise ImportError("PyTorch is required for SRCNN. Install with: pip install torch")


def compute_psnr(
    output: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]],
    max_value: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(MAX^2 / MSE)

    Args:
        output: Predicted image array.
        target: Ground truth image array.
        max_value: Maximum pixel value (1.0 for normalized, 255 for uint8).

    Returns:
        PSNR value in dB.
    """
    mse = np.mean((output - target) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((max_value**2) / mse))


def compute_mse(
    output: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]],
) -> float:
    """Compute Mean Squared Error.

    Args:
        output: Predicted image array.
        target: Ground truth image array.

    Returns:
        MSE value.
    """
    return float(np.mean((output - target) ** 2))


def preprocess_for_srcnn(
    image: NDArray[np.floating[Any]],
    scale_factor: int = 2,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Create LR-HR pair for SRCNN training/evaluation.

    Args:
        image: High-resolution image (C, H, W) or (H, W).
        scale_factor: Downscaling factor.

    Returns:
        Tuple of (low_resolution, high_resolution) images.
    """
    from scipy.ndimage import zoom

    # Ensure 3D
    if image.ndim == 2:
        image = image[np.newaxis, ...]

    # Downsample to create LR
    lr = zoom(image, (1, 1 / scale_factor, 1 / scale_factor), order=1)

    # HR is original
    hr = image

    return lr.astype(np.float32), hr.astype(np.float32)
