# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/models/factory.py
# Title       : Build model zoo networks from the catalogue
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch (unbihexium[torch])
# =============================================================================
#
# Abstract
# --------
# Turns a catalogue entry and a size variant into a PyTorch network with
# deterministic starter weights. The result is a ZooModel: the network plus
# the BuildConfig that describes its inputs and outputs, so that training,
# inference and export code know how to feed and read the network without
# consulting the catalogue again.
#
# Customisation
# -------------
# Fine-tuning often needs a different input or output layout than the
# catalogue entry, for example a four-band sensor for an RGB model or project
# specific classes. build_model accepts:
#
#   channel_names  names of the input channels (sets the input width)
#   outputs        class, target or band names (sets the output width)
#
# Customised models are initialised from the seed of their model id like the
# catalogue models, but their digests differ from the published ones.
#
# Usage
# -----
#   from unbihexium.ai.models import build_model
#   model = build_model("ship_detector", "base")
#   custom = build_model("lulc_classifier", "tiny", outputs=["water", "land"])
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Immutable configuration records and replacement of fields.
from dataclasses import dataclass, field, replace

# Type of loosely structured values.
from typing import Any

# Tensor type.
import torch

# Module base class.
from torch import nn

# Deterministic initialisation, digest and parameter count.
from unbihexium.ai.models.init import count_parameters, initialize, seed_for, weights_digest

# Task networks.
from unbihexium.ai.models.networks import (
    DETECTION_STRIDE,  # Output stride of the detector.
    CenterNet,  # Object detector.
    SceneRegressor,  # Scene-level regressor.
    SuperResolutionNet,  # Super-resolution network.
    UNet,  # Dense per-pixel network.
)  # End of the network imports.

# Parameter-free spectral index modules.
from unbihexium.ai.models.spectral import FORMULA_CHANNELS, SpectralIndex

# Catalogue types and lookups.
from unbihexium.zoo.catalog import (
    ModelSpec,  # Catalogue entry of a family.
    Task,  # Task enumeration.
    Variant,  # Size variant enumeration.
    VariantSpec,  # Variant hyperparameters.
    get_spec,  # Look up a family.
    get_variant,  # Look up a variant.
    parse_model_id,  # Split a model id.
)  # End of the catalogue imports.

# Tasks served by the U-Net architecture.
UNET_TASKS = frozenset(
    {  # Set literal of the tasks.
        Task.SEGMENTATION,  # Per-pixel classes.
        Task.CHANGE_DETECTION,  # Per-pixel change classes.
        Task.DENSE_REGRESSION,  # Per-pixel continuous values.
        Task.ENHANCEMENT,  # Image-to-image translation.
    }  # End of the set literal.
)  # End of the task set.

# Output names that denote displacement fields rather than image bands.
DISPLACEMENT_OUTPUTS = frozenset({"dx", "dy"})


# Effective configuration of a built model.
@dataclass(frozen=True)
class BuildConfig:
    # Model id: family and variant.
    model_id: str
    # Model family.
    family: str
    # Size variant.
    variant: Variant
    # Task of the model.
    task: Task
    # Names of the input channels, in order.
    channel_names: tuple[str, ...]
    # Class, target or output band names.
    outputs: tuple[str, ...]
    # Units of regression targets.
    units: tuple[str, ...] = ()
    # Bounds of regression targets.
    value_range: tuple[float, float] | None = None
    # Upscaling factor of super-resolution models.
    scale: int = 1
    # Formula of spectral index models.
    formula: str | None = None
    # Recommended square tile size.
    tile_size: int = 256
    # True when channel_names or outputs differ from the catalogue.
    customised: bool = False
    # Additional metadata, for example the training history.
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    # Number of input channels.
    @property
    def in_channels(self) -> int:
        # One channel per name.
        return len(self.channel_names)

    # Number of network outputs (classes, targets or bands).
    @property
    def out_channels(self) -> int:
        # One output per name.
        return len(self.outputs)

    # Serialise to plain Python types for checkpoints and JSON files.
    def to_dict(self) -> dict[str, Any]:
        # Plain dictionary.
        return {
            "model_id": self.model_id,  # Model id.
            "family": self.family,  # Model family.
            "variant": self.variant.value,  # Size variant.
            "task": self.task.value,  # Task.
            "channel_names": list(self.channel_names),  # Input channels.
            "outputs": list(self.outputs),  # Outputs.
            "units": list(self.units),  # Target units.
            "value_range": list(self.value_range) if self.value_range else None,  # Bounds.
            "scale": self.scale,  # Upscaling factor.
            "formula": self.formula,  # Index formula.
            "tile_size": self.tile_size,  # Tile size.
            "customised": self.customised,  # Deviates from the catalogue.
            "extra": dict(self.extra),  # Additional metadata.
        }  # End of the dictionary.

    # Rebuild a configuration from its dictionary form.
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuildConfig:
        # Convert every field back to its Python type.
        return cls(
            model_id=str(data["model_id"]),  # Model id.
            family=str(data["family"]),  # Model family.
            variant=Variant(data["variant"]),  # Size variant.
            task=Task(data["task"]),  # Task.
            channel_names=tuple(data["channel_names"]),  # Input channels.
            outputs=tuple(data["outputs"]),  # Outputs.
            units=tuple(data.get("units") or ()),  # Target units.
            value_range=tuple(data["value_range"]) if data.get("value_range") else None,  # Bounds.
            scale=int(data.get("scale", 1)),  # Upscaling factor.
            formula=data.get("formula"),  # Index formula.
            tile_size=int(data.get("tile_size", 256)),  # Tile size.
            customised=bool(data.get("customised", False)),  # Deviation flag.
            extra=dict(data.get("extra") or {}),  # Additional metadata.
        )  # End of the configuration.

    # Derive the configuration of a catalogue entry and variant.
    @classmethod
    def from_spec(
        cls,  # The class itself.
        spec: ModelSpec,  # Catalogue entry.
        variant: Variant | str,  # Size variant.
        channel_names: tuple[str, ...] | list[str] | None = None,  # Input override.
        outputs: tuple[str, ...] | list[str] | None = None,  # Output override.
    ) -> BuildConfig:  # The derived configuration.
        # Normalise the variant.
        variant = Variant(variant)
        # Use the catalogue channels unless overridden.
        names = tuple(channel_names) if channel_names else spec.channel_names
        # Use the catalogue outputs unless overridden.
        outs = tuple(outputs) if outputs else spec.outputs
        # Units only apply when the outputs are unchanged.
        units = spec.units if outs == spec.outputs else ()
        # Build the configuration.
        return cls(
            model_id=spec.model_id(variant),  # Model id.
            family=spec.family,  # Model family.
            variant=variant,  # Size variant.
            task=spec.task,  # Task.
            channel_names=names,  # Input channels.
            outputs=outs,  # Outputs.
            units=units,  # Target units.
            value_range=spec.value_range,  # Bounds.
            scale=spec.scale,  # Upscaling factor.
            formula=spec.formula,  # Index formula.
            tile_size=get_variant(variant).tile_size,  # Tile size.
            customised=(names != spec.channel_names or outs != spec.outputs),  # Deviation flag.
        )  # End of the configuration.


# A network together with the configuration that describes it.
class ZooModel(nn.Module):
    # Wrap a network.
    def __init__(self, network: nn.Module, config: BuildConfig) -> None:
        # Initialise the module base class.
        super().__init__()
        # The wrapped network; its parameters appear under "network.".
        self.network = network
        # Input and output description.
        self.config = config

    # Run the wrapped network.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Delegate to the network.
        return self.network(x)

    # Model id of the model.
    @property
    def model_id(self) -> str:
        # Read it from the configuration.
        return self.config.model_id

    # Task of the model.
    @property
    def task(self) -> Task:
        # Read it from the configuration.
        return self.config.task

    # Output stride of dense feature maps relative to the input.
    @property
    def output_stride(self) -> int:
        # Detectors predict at a quarter of the input resolution.
        return DETECTION_STRIDE if self.config.task is Task.DETECTION else 1

    # Number of trainable parameters.
    def num_parameters(self) -> int:
        # Count parameters that require gradients.
        return count_parameters(self)

    # SHA-256 digest of the weights (see models.init.weights_digest).
    def digest(self) -> str:
        # Digest of the full state dict.
        return weights_digest(self)

    # One-line human-readable summary.
    def summary(self) -> str:
        # Model id, task, channels and parameter count.
        channels = f"{self.config.in_channels} input channels"
        # Number of outputs and parameters.
        size = f"{self.config.out_channels} outputs, {self.num_parameters():,} parameters"
        # Join the parts.
        return f"{self.model_id}: {self.task.value}, {channels}, {size}"


# Whether an enhancement model predicts a correction of its input bands.
def _is_residual(config: BuildConfig) -> bool:
    # Only enhancement models are residual.
    if config.task is not Task.ENHANCEMENT:
        # Other tasks predict their outputs directly.
        return False
    # Displacement fields are not corrections of image bands.
    if DISPLACEMENT_OUTPUTS & set(config.outputs):
        # Predict the field directly.
        return False
    # A correction needs as many input bands as outputs.
    return config.out_channels <= config.in_channels


# Instantiate the network of a configuration without initialising it.
def _make_network(config: BuildConfig, v: VariantSpec) -> nn.Module:
    # Detection: CenterNet.
    if config.task is Task.DETECTION:
        # Heatmap per class plus size and offset.
        return CenterNet(
            in_channels=config.in_channels,  # Input bands.
            num_classes=config.out_channels,  # One heatmap per class.
            base_channels=v.base_channels,  # Encoder width.
            depth=v.depth,  # Encoder depth.
            blocks_per_stage=v.blocks_per_stage,  # Blocks per level.
            head_channels=v.head_channels,  # Head width.
        )  # End of the detector.
    # Dense tasks: U-Net.
    if config.task in UNET_TASKS:
        # Value ranges only apply to regression.
        value_range = config.value_range if config.task is Task.DENSE_REGRESSION else None
        # U-Net with optional residual output.
        return UNet(
            in_channels=config.in_channels,  # Input bands.
            out_channels=config.out_channels,  # Classes, targets or bands.
            base_channels=v.base_channels,  # Encoder width.
            depth=v.depth,  # Encoder depth.
            blocks_per_stage=v.blocks_per_stage,  # Blocks per level.
            value_range=value_range,  # Bounded regression outputs.
            residual=_is_residual(config),  # Correction of the input bands.
        )  # End of the U-Net.
    # Scene regression: pooled encoder.
    if config.task is Task.SCENE_REGRESSION:
        # One value per target.
        return SceneRegressor(
            in_channels=config.in_channels,  # Input bands.
            out_channels=config.out_channels,  # Regression targets.
            base_channels=v.base_channels,  # Encoder width.
            depth=v.depth,  # Encoder depth.
            blocks_per_stage=v.blocks_per_stage,  # Blocks per level.
            head_channels=v.head_channels,  # Hidden layer width.
            value_range=config.value_range,  # Bounded targets.
        )  # End of the regressor.
    # Super-resolution: residual network with sub-pixel up-sampling.
    if config.task is Task.SUPER_RESOLUTION:
        # Width and depth grow with the variant.
        return SuperResolutionNet(
            in_channels=config.in_channels,  # Input bands.
            out_channels=config.out_channels,  # Output bands.
            channels=2 * v.base_channels,  # Feature width.
            num_blocks=2 * v.depth * v.blocks_per_stage,  # Residual blocks.
            scale=config.scale,  # Upscaling factor.
        )  # End of the network.
    # Spectral index: the formula fixes the channel count.
    if config.formula is None or config.in_channels != FORMULA_CHANNELS.get(config.formula):
        # Report the inconsistent configuration.
        raise ValueError(f"{config.model_id}: formula and channel count do not match")
    # Parameter-free index module.
    return SpectralIndex(config.formula)


# Build a model from a configuration, with deterministic starter weights.
def build_from_config(config: BuildConfig, initialise: bool = True) -> ZooModel:
    # Network for the task and variant.
    network = _make_network(config, get_variant(config.variant))
    # Wrap the network with its configuration.
    model = ZooModel(network, config)
    # Deterministic initialisation from the model id.
    if initialise:
        # Seeded, platform-independent initialisation.
        initialize(model, seed_for(config.model_id))
    # Models start in evaluation mode, ready for inference.
    return model.eval()


# Build a model zoo model by name.
def build_model(
    name: str,  # Model family or model id.
    variant: Variant | str | None = None,  # Size variant; overrides a suffix in name.
    channel_names: list[str] | tuple[str, ...] | None = None,  # Input override.
    outputs: list[str] | tuple[str, ...] | None = None,  # Output override.
    initialise: bool = True,  # Apply the deterministic initialisation.
) -> ZooModel:  # The model with its configuration.
    # Split an optional variant suffix from the name.
    family, parsed_variant = parse_model_id(name)
    # An explicit variant argument takes precedence.
    chosen = Variant(variant) if variant is not None else parsed_variant
    # Catalogue entry of the family.
    spec = get_spec(family)
    # Spectral index models have a fixed channel layout.
    if spec.task is Task.SPECTRAL_INDEX and (channel_names or outputs):
        # Reject overrides that would break the formula.
        raise ValueError("spectral index models cannot change their inputs or outputs")
    # Effective configuration.
    config = BuildConfig.from_spec(spec, chosen, channel_names, outputs)
    # Build and initialise the model.
    return build_from_config(config, initialise=initialise)


# Return a copy of a configuration with some fields replaced.
def with_changes(config: BuildConfig, **changes: Any) -> BuildConfig:
    # dataclasses.replace creates the modified copy.
    return replace(config, **changes)


# =============================================================================
# End of module src/unbihexium/ai/models/factory.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
