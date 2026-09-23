# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/models/blocks.py
# Title       : Neural network building blocks shared by all architectures
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch (unbihexium[torch])
# =============================================================================
#
# Abstract
# --------
# Convolutional building blocks used by the encoder, the U-Net decoder, the
# CenterNet detector and the image-to-image networks of the model zoo:
#
#   ConvNormAct    3x3 (or 1x1) convolution, group normalisation, ReLU
#   ResidualBlock  two ConvNormAct layers with an identity shortcut
#                  (He et al., 2016)
#   DownStage      strided convolution followed by residual blocks
#   UpStage        bilinear up-sampling, skip concatenation and fusion
#                  (Ronneberger et al., 2015)
#   HeadConv       3x3 convolution, ReLU and a 1x1 output convolution
#
# Design choices
# --------------
# Group normalisation (Wu and He, 2018) is used instead of batch
# normalisation because Earth observation models are often fine-tuned with
# batch sizes of one or two large tiles, where batch statistics are unstable.
# Up-sampling uses bilinear interpolation to the exact size of the skip
# connection, so inputs whose size is not a multiple of 2**depth work, and
# the operation exports to ONNX with dynamic spatial sizes.
#
# References
# ----------
#   He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning for
#     image recognition. CVPR, 770-778.
#   Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional
#     networks for biomedical image segmentation. MICCAI, 234-241.
#   Wu, Y., He, K. (2018). Group normalization. ECCV, 3-19.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Tensor type and core functions.
import torch

# Functional interface for interpolation.
import torch.nn.functional as F

# Module base classes and layers.
from torch import nn


# Choose the number of normalisation groups for a channel count.
def group_count(channels: int) -> int:
    # Prefer 8 groups, falling back to the largest power of two that divides.
    for groups in (8, 4, 2):
        # A valid group count divides the number of channels.
        if channels % groups == 0:
            # Return the first valid count.
            return groups
    # One group (layer normalisation over channels) always works.
    return 1


# Build a group normalisation layer for a channel count.
def norm_layer(channels: int) -> nn.GroupNorm:
    # Group normalisation with an automatically chosen number of groups.
    return nn.GroupNorm(group_count(channels), channels)


# Convolution, group normalisation and ReLU activation.
class ConvNormAct(nn.Module):
    # Create the layer.
    def __init__(
        self,  # Module being created.
        in_channels: int,  # Channels of the input.
        out_channels: int,  # Channels of the output.
        kernel_size: int = 3,  # Size of the square kernel.
        stride: int = 1,  # Stride; 2 halves the resolution.
    ) -> None:  # The constructor returns nothing.
        # Initialise the module base class.
        super().__init__()
        # Same padding for odd kernel sizes.
        padding = kernel_size // 2
        # Convolution without bias; the normalisation provides the shift.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # Group normalisation of the convolution output.
        self.norm = norm_layer(out_channels)
        # Rectified linear activation, in place to save memory.
        self.act = nn.ReLU(inplace=True)

    # Apply convolution, normalisation and activation.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chain the three operations.
        return self.act(self.norm(self.conv(x)))


# Residual block: x + F(x) with two ConvNormAct-style layers.
class ResidualBlock(nn.Module):
    # Create the block for a fixed number of channels.
    def __init__(self, channels: int) -> None:
        # Initialise the module base class.
        super().__init__()
        # First convolution with activation.
        self.conv1 = ConvNormAct(channels, channels)
        # Second convolution without activation before the addition.
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        # Normalisation of the second convolution.
        self.norm2 = norm_layer(channels)
        # Activation after the residual addition.
        self.act = nn.ReLU(inplace=True)

    # Apply the residual transformation.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual branch.
        y = self.norm2(self.conv2(self.conv1(x)))
        # Identity shortcut followed by the activation.
        return self.act(x + y)


# Encoder stage: halve the resolution, then refine with residual blocks.
class DownStage(nn.Module):
    # Create the stage.
    def __init__(self, in_channels: int, out_channels: int, blocks: int) -> None:
        # Initialise the module base class.
        super().__init__()
        # Strided convolution halves height and width.
        self.down = ConvNormAct(in_channels, out_channels, stride=2)
        # Residual blocks at the new resolution.
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(blocks)])

    # Apply the stage.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Down-sample, then refine.
        return self.blocks(self.down(x))


# Decoder stage: up-sample, concatenate the skip features and fuse them.
class UpStage(nn.Module):
    # Create the stage.
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        # Initialise the module base class.
        super().__init__()
        # Fuse the concatenated features into the output width.
        self.fuse = ConvNormAct(in_channels + skip_channels, out_channels)
        # Refine the fused features.
        self.block = ResidualBlock(out_channels)

    # Apply the stage to deep features x and skip features.
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Up-sample to the exact spatial size of the skip features.
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        # Concatenate along the channel axis.
        x = torch.cat([x, skip], dim=1)
        # Fuse and refine.
        return self.block(self.fuse(x))


# Prediction head: 3x3 convolution with ReLU and a 1x1 output convolution.
class HeadConv(nn.Module):
    # Create the head. init_bias sets a constant bias of the output layer and
    # init_std the standard deviation of its weights (see models.init).
    def __init__(
        self,  # Module being created.
        in_channels: int,  # Channels of the input features.
        hidden_channels: int,  # Width of the hidden layer.
        out_channels: int,  # Number of outputs per pixel.
        init_bias: float = 0.0,  # Initial bias of the output layer.
        init_std: float | None = None,  # Initial weight scale of the output layer.
    ) -> None:  # The constructor returns nothing.
        # Initialise the module base class.
        super().__init__()
        # Hidden 3x3 convolution with bias, as heads are not normalised.
        self.hidden = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        # Activation of the hidden layer.
        self.act = nn.ReLU(inplace=True)
        # 1x1 output convolution.
        self.out = nn.Conv2d(hidden_channels, out_channels, 1)
        # Remember the requested initial bias for the initialiser.
        self.out.init_bias = init_bias  # type: ignore[assignment]
        # Remember the requested initial weight scale for the initialiser.
        self.out.init_std = init_std  # type: ignore[assignment]

    # Apply the head.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hidden layer, activation and output layer.
        return self.out(self.act(self.hidden(x)))


# Map unbounded network outputs to a closed interval with a sigmoid.
class RangeActivation(nn.Module):
    # Create the activation for the interval [low, high].
    def __init__(self, low: float, high: float) -> None:
        # Initialise the module base class.
        super().__init__()
        # Lower bound of the interval.
        self.low = float(low)
        # Width of the interval.
        self.span = float(high) - float(low)

    # Squash the input into the interval.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # low + span * sigmoid(x) lies strictly inside (low, high).
        return self.low + self.span * torch.sigmoid(x)

    # Human-readable description used by print(model).
    def extra_repr(self) -> str:
        # Show the interval.
        return f"low={self.low}, high={self.low + self.span}"


# =============================================================================
# End of module src/unbihexium/ai/models/blocks.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
