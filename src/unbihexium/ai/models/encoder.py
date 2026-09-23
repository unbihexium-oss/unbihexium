# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/models/encoder.py
# Title       : Residual encoder and U-Net decoder
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch (unbihexium[torch])
# =============================================================================
#
# Abstract
# --------
# The encoder is a residual convolutional network with a configurable number
# of input channels, so that it accepts any band combination (RGB,
# multispectral, SAR, elevation or stacks of several acquisitions). It returns
# the feature maps of every stage:
#
#   level 0   full resolution,     base_channels
#   level 1   1/2 resolution,      2 * base_channels
#   level k   1/2**k resolution,   min(2**k, 8) * base_channels
#
# The decoder up-samples the deepest features step by step and fuses them
# with the encoder features of the same resolution (U-Net skip connections),
# stopping at a requested output level: level 0 for dense per-pixel tasks and
# level 2 (a quarter of the input resolution) for the CenterNet detector.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Tensor type.
import torch

# Module base classes and containers.
from torch import nn

# Shared building blocks.
from unbihexium.ai.models.blocks import ConvNormAct, DownStage, ResidualBlock, UpStage

# Maximum channel multiplier relative to base_channels.
MAX_MULTIPLIER = 8


# Channel widths of the encoder levels for a base width and depth.
def stage_channels(base_channels: int, depth: int) -> list[int]:
    # Width doubles per level and is capped at MAX_MULTIPLIER * base.
    return [base_channels * min(2**level, MAX_MULTIPLIER) for level in range(depth + 1)]


# Residual encoder returning the features of every level.
class Encoder(nn.Module):
    # Create the encoder.
    def __init__(
        self,  # Module being created.
        in_channels: int,  # Number of input bands.
        base_channels: int,  # Width of the first level.
        depth: int,  # Number of down-sampling levels.
        blocks_per_stage: int,  # Residual blocks per level.
    ) -> None:  # The constructor returns nothing.
        # Initialise the module base class.
        super().__init__()
        # At least one input channel is needed.
        if in_channels < 1:
            # Reject impossible configurations early.
            raise ValueError("in_channels must be at least 1")
        # Width of every level.
        self.channels = stage_channels(base_channels, depth)
        # Stem: full-resolution convolution and residual blocks.
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, base_channels),  # Map the input bands to features.
            *[ResidualBlock(base_channels) for _ in range(blocks_per_stage)],  # Refine.
        )  # End of the stem.
        # One down-sampling stage per level below the stem.
        self.stages = nn.ModuleList(
            DownStage(self.channels[i], self.channels[i + 1], blocks_per_stage)  # Stage i+1.
            for i in range(depth)  # One stage per level.
        )  # End of the stage list.

    # Return the feature maps of all levels, from full to lowest resolution.
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Full-resolution features.
        features = [self.stem(x)]
        # Apply the stages in order.
        for stage in self.stages:
            # Each stage consumes the previous level.
            features.append(stage(features[-1]))
        # Return all levels.
        return features


# U-Net decoder from the deepest level up to a target level.
class Decoder(nn.Module):
    # Create the decoder for the given encoder widths.
    def __init__(self, channels: list[int], out_level: int = 0) -> None:
        # Initialise the module base class.
        super().__init__()
        # The target level must exist and lie above the deepest level.
        if not 0 <= out_level < len(channels) - 1:
            # Reject impossible configurations early.
            raise ValueError("out_level must be between 0 and depth - 1")
        # Level at which decoding stops.
        self.out_level = out_level
        # Up-sampling stages from level depth to out_level. The stage that
        # reaches a level receives the width of the level below it and
        # outputs the width of the level it reaches.
        self.stages = nn.ModuleList(
            UpStage(channels[level + 1], channels[level], channels[level])  # Stage to level.
            for level in range(len(channels) - 2, out_level - 1, -1)  # Deepest first.
        )  # End of the stage list.
        # Width of the decoder output.
        self.out_channels = channels[out_level]

    # Decode the encoder features.
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # Start from the deepest features.
        x = features[-1]
        # Walk up the levels.
        for index, stage in enumerate(self.stages):
            # Level of the skip connection used by this stage.
            level = len(features) - 2 - index
            # Up-sample and fuse with the skip connection.
            x = stage(x, features[level])
        # Features at the output level.
        return x


# =============================================================================
# End of module src/unbihexium/ai/models/encoder.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
