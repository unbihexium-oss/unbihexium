# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/models/networks.py
# Title       : Task architectures of the model zoo
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch (unbihexium[torch])
# =============================================================================
#
# Abstract
# --------
# One network class per task of the model zoo. All networks take a float32
# tensor of shape (N, C, H, W) with the input bands on the channel axis:
#
#   UNet                segmentation, change detection, dense regression and
#                       enhancement; output (N, K, H, W)
#   CenterNet           object detection; output (N, K + 4, H/4, W/4) with K
#                       class heatmap logits, box width and height, and the
#                       sub-pixel centre offset (Zhou et al., 2019)
#   SceneRegressor      scene regression; output (N, K)
#   SuperResolutionNet  super-resolution; output (N, K, s*H, s*W), residual
#                       blocks without normalisation (Lim et al., 2017) and
#                       sub-pixel convolution (Shi et al., 2016)
#
# Classification networks return logits; apply softmax (segmentation, change
# detection) or sigmoid (detection heatmaps) to obtain probabilities.
# Regression networks with a value range end in a scaled sigmoid, so their
# outputs lie inside the range. Spectral index modules live in spectral.py.
#
# References
# ----------
#   Zhou, X., Wang, D., Kraehenbuehl, P. (2019). Objects as points.
#     arXiv:1904.07850.
#   Lim, B., Son, S., Kim, H., Nah, S., Lee, K. M. (2017). Enhanced deep
#     residual networks for single image super-resolution. CVPR Workshops.
#   Shi, W., et al. (2016). Real-time single image and video super-resolution
#     using an efficient sub-pixel convolutional neural network. CVPR.
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

# Shared building blocks.
from unbihexium.ai.models.blocks import HeadConv, RangeActivation

# Residual encoder and U-Net decoder.
from unbihexium.ai.models.encoder import Decoder, Encoder

# Initial bias of CenterNet heatmap logits: sigmoid(-2.19) = 0.1, so that
# training starts from a low background probability (Zhou et al., 2019).
HEATMAP_PRIOR_BIAS = -2.19

# Output stride of the CenterNet detector.
DETECTION_STRIDE = 4


# U-Net for dense per-pixel outputs.
class UNet(nn.Module):
    # Create the network.
    def __init__(
        self,  # Module being created.
        in_channels: int,  # Number of input bands.
        out_channels: int,  # Number of classes, targets or output bands.
        base_channels: int,  # Width of the first encoder level.
        depth: int,  # Number of down-sampling levels.
        blocks_per_stage: int,  # Residual blocks per encoder level.
        value_range: tuple[float, float] | None = None,  # Bounds of regression outputs.
        residual: bool = False,  # Add the input bands to the output.
    ) -> None:  # The constructor returns nothing.
        # Initialise the module base class.
        super().__init__()
        # A residual output needs at least as many input as output channels.
        if residual and out_channels > in_channels:
            # Reject impossible configurations early.
            raise ValueError("residual output needs out_channels <= in_channels")
        # Encoder over the input bands.
        self.encoder = Encoder(in_channels, base_channels, depth, blocks_per_stage)
        # Decoder back to full resolution.
        self.decoder = Decoder(self.encoder.channels, out_level=0)
        # 1x1 projection to the outputs; small initial weights keep residual
        # and regression outputs close to their starting point.
        self.head = nn.Conv2d(self.decoder.out_channels, out_channels, 1)
        # Small initial weights for regression and residual outputs.
        self.head.init_std = 1e-3 if (residual or value_range) else None  # type: ignore[assignment]
        # Optional range activation for bounded regression outputs.
        self.activation = RangeActivation(*value_range) if value_range else nn.Identity()
        # Whether the first out_channels input bands are added to the output.
        self.residual = residual
        # Number of outputs, kept for the residual slice.
        self.out_channels = out_channels

    # Run the network.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode, decode and project.
        y = self.head(self.decoder(self.encoder(x)))
        # Residual networks predict a correction of the input bands.
        if self.residual:
            # Add the first out_channels input bands.
            y = y + x[:, : self.out_channels]
        # Apply the output activation.
        return self.activation(y)


# CenterNet object detector with output stride 4.
class CenterNet(nn.Module):
    # Create the detector.
    def __init__(
        self,  # Module being created.
        in_channels: int,  # Number of input bands.
        num_classes: int,  # Number of object classes.
        base_channels: int,  # Width of the first encoder level.
        depth: int,  # Number of down-sampling levels (at least 3).
        blocks_per_stage: int,  # Residual blocks per encoder level.
        head_channels: int,  # Width of the prediction heads.
    ) -> None:  # The constructor returns nothing.
        # Initialise the module base class.
        super().__init__()
        # Stride 4 features need at least three encoder levels.
        if depth < 3:
            # Reject impossible configurations early.
            raise ValueError("CenterNet needs depth >= 3")
        # Encoder over the input bands.
        self.encoder = Encoder(in_channels, base_channels, depth, blocks_per_stage)
        # Decoder up to a quarter of the input resolution (level 2).
        self.decoder = Decoder(self.encoder.channels, out_level=2)
        # Width of the stride-4 features.
        features = self.decoder.out_channels
        # Class heatmap head with the background prior bias.
        self.heatmap = HeadConv(features, head_channels, num_classes, init_bias=HEATMAP_PRIOR_BIAS)
        # Box width and height head, in output-stride pixels.
        self.size = HeadConv(features, head_channels, 2, init_std=1e-3)
        # Sub-pixel centre offset head.
        self.offset = HeadConv(features, head_channels, 2, init_std=1e-3)
        # Number of classes, used by the decoder of the predictions.
        self.num_classes = num_classes

    # Run the detector; returns heatmap logits, sizes and offsets stacked.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared stride-4 features.
        f = self.decoder(self.encoder(x))
        # Concatenate the three heads on the channel axis.
        return torch.cat([self.heatmap(f), self.size(f), self.offset(f)], dim=1)


# Scene regressor: encoder, global average pooling and a two-layer head.
class SceneRegressor(nn.Module):
    # Create the network.
    def __init__(
        self,  # Module being created.
        in_channels: int,  # Number of input bands.
        out_channels: int,  # Number of regression targets.
        base_channels: int,  # Width of the first encoder level.
        depth: int,  # Number of down-sampling levels.
        blocks_per_stage: int,  # Residual blocks per encoder level.
        head_channels: int,  # Width of the hidden layer of the head.
        value_range: tuple[float, float] | None = None,  # Bounds of the targets.
    ) -> None:  # The constructor returns nothing.
        # Initialise the module base class.
        super().__init__()
        # Encoder over the input bands.
        self.encoder = Encoder(in_channels, base_channels, depth, blocks_per_stage)
        # Global average pooling of the deepest features.
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Hidden fully connected layer.
        self.hidden = nn.Linear(self.encoder.channels[-1], head_channels)
        # Activation of the hidden layer.
        self.act = nn.ReLU(inplace=True)
        # Output layer with small initial weights.
        self.out = nn.Linear(head_channels, out_channels)
        # Small initial weights start the predictions near the bias.
        self.out.init_std = 1e-3  # type: ignore[assignment]
        # Optional range activation for bounded targets.
        self.activation = RangeActivation(*value_range) if value_range else nn.Identity()

    # Run the network.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pool the deepest features to one vector per sample.
        z = torch.flatten(self.pool(self.encoder(x)[-1]), 1)
        # Hidden layer, output layer and activation.
        return self.activation(self.out(self.act(self.hidden(z))))


# Residual block without normalisation, as in EDSR.
class _SRBlock(nn.Module):
    # Create the block.
    def __init__(self, channels: int, residual_scale: float = 0.1) -> None:
        # Initialise the module base class.
        super().__init__()
        # First convolution.
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        # Activation.
        self.act = nn.ReLU(inplace=True)
        # Second convolution.
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        # Residual scaling stabilises training of deep stacks (Lim et al.).
        self.residual_scale = residual_scale

    # Apply the block.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scaled residual branch added to the identity.
        return x + self.residual_scale * self.conv2(self.act(self.conv1(x)))


# Super-resolution network with sub-pixel up-sampling and a global skip.
class SuperResolutionNet(nn.Module):
    # Create the network.
    def __init__(
        self,  # Module being created.
        in_channels: int,  # Number of input bands.
        out_channels: int,  # Number of output bands.
        channels: int,  # Width of the feature maps.
        num_blocks: int,  # Number of residual blocks.
        scale: int,  # Upscaling factor.
    ) -> None:  # The constructor returns nothing.
        # Initialise the module base class.
        super().__init__()
        # The global skip adds the up-sampled first out_channels input bands.
        if out_channels > in_channels:
            # Reject impossible configurations early.
            raise ValueError("super-resolution needs out_channels <= in_channels")
        # Head convolution from bands to features.
        self.head = nn.Conv2d(in_channels, channels, 3, 1, 1)
        # Residual body.
        self.body = nn.Sequential(*[_SRBlock(channels) for _ in range(num_blocks)])
        # Convolution closing the body before the long skip.
        self.body_out = nn.Conv2d(channels, channels, 3, 1, 1)
        # Convolution producing scale**2 channels per output band.
        self.upsample_conv = nn.Conv2d(channels, out_channels * scale * scale, 3, 1, 1)
        # Small initial weights: the untrained network returns the bilinear
        # up-sampling of the input.
        self.upsample_conv.init_std = 1e-3  # type: ignore[assignment]
        # Rearrange channels into space.
        self.shuffle = nn.PixelShuffle(scale)
        # Upscaling factor.
        self.scale = scale
        # Number of output bands.
        self.out_channels = out_channels

    # Run the network.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shallow features.
        f = self.head(x)
        # Deep features with a long skip connection.
        f = f + self.body_out(self.body(f))
        # Learned high-frequency detail at the target resolution.
        detail = self.shuffle(self.upsample_conv(f))
        # Input bands that form the base image.
        bands = x[:, : self.out_channels]
        # Up-sample the bands to the target resolution.
        base = F.interpolate(bands, scale_factor=self.scale, mode="bilinear", align_corners=False)
        # Base image plus learned detail.
        return base + detail


# =============================================================================
# End of module src/unbihexium/ai/models/networks.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
