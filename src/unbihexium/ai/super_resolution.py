# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/super_resolution.py
# Title       : Super-resolution and image enhancement
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and PyTorch or ONNX
#               Runtime
# =============================================================================
#
# Abstract
# --------
# SuperResolution upscales an image with the EDSR-style network of the model
# zoo and returns a raster whose pixels are `scale_factor` times smaller;
# the georeferencing is adjusted so that the output covers the same area.
# The catalogue model upscales by 4. Other factors build a customised model
# that must be trained for that factor before use.
#
# Enhancer runs the image-to-image models of the catalogue, for example
# pansharpening, orthorectification corrections, mosaicking and SAR to
# optical translation, and returns the output bands as a raster on the input
# grid.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Array type annotations.
from numpy.typing import NDArray

# Task API base and pipeline registration.
from unbihexium.ai.base import ZooTask, register_task_pipeline

# Result records and georeferencing.
from unbihexium.ai.results import EnhancementResult, SuperResolutionResult, scaled_transform

# Raster container.
from unbihexium.core.raster import Raster

# Catalogue lookups.
from unbihexium.zoo.catalog import Task, get_spec, parse_model_id


# Super-resolution of multispectral imagery.
class SuperResolution(ZooTask):
    # Catalogue family.
    default_model = "super_resolution"
    # Super-resolution models only.
    tasks = (Task.SUPER_RESOLUTION,)

    # Configure the model.
    def __init__(
        self,  # The model API.
        model: Any = None,  # Family, model id, checkpoint, ONNX file or ZooModel.
        scale_factor: int | None = None,  # Upscaling factor; default from the model.
        tile_size: int = 256,  # Tile size of the inference.
        **kwargs: Any,  # Options of ZooTask (variant, weights, device, ...).
    ) -> None:  # The constructor returns nothing.
        # Model selection and inference options.
        super().__init__(model, tile_size=tile_size, **kwargs)
        # Catalogue scale of named models, used when no factor is given.
        default = self._catalogue_scale()
        # Requested upscaling factor.
        self.scale_factor = int(scale_factor or default or 4)
        # A different factor needs a customised network.
        if default is not None and self.scale_factor != default:
            # Build the customised model now; it must be trained before use.
            self.source = self._customised(self.scale_factor)

    # Scale of a catalogue name, or None for files and objects.
    def _catalogue_scale(self) -> int | None:
        # Files and objects carry their own configuration.
        if not isinstance(self.source, str) or Path(self.source).suffix:
            # Unknown until loaded.
            return None
        # Catalogue entry of the family.
        return get_spec(parse_model_id(self.source)[0]).scale

    # Build a model with another upscaling factor.
    def _customised(self, scale: int) -> Any:
        # PyTorch-dependent modules are imported lazily.
        from unbihexium.ai.models.factory import build_from_config  # Builds networks.
        from unbihexium.zoo.config import BuildConfig  # Configuration record.

        # Family and variant of the name.
        family, variant = parse_model_id(str(self.source))
        # Catalogue configuration.
        config = BuildConfig.from_spec(get_spec(family), self.variant or variant)
        # Replace the scale and mark the deviation.
        config = BuildConfig.from_dict({**config.to_dict(), "scale": scale, "customised": True})
        # Build the network.
        return build_from_config(config)

    # Upscale an image, raster or raster file.
    def enhance(self, image: Raster | NDArray[Any] | str | Path) -> SuperResolutionResult:
        # Array and georeferencing.
        data, crs, transform, source = self.prepare(image)
        # Upscaled bands.
        upscaled = self.predictor.dense(data)
        # Factor of the loaded model.
        factor = self.predictor.config.scale
        # Raster on the finer grid.
        raster = Raster.from_array(upscaled, crs=crs, transform=scaled_transform(transform, factor))
        # Result record.
        return SuperResolutionResult(raster, factor, source, self.model_id)

    # Alias used by the pipelines and the command line.
    def predict(self, image: Raster | NDArray[Any] | str | Path) -> SuperResolutionResult:
        # Same as enhance.
        return self.enhance(image)


# Image-to-image models on the input grid.
class Enhancer(ZooTask):
    # Pansharpening by default.
    default_model = "pansharpening"
    # Enhancement models only.
    tasks = (Task.ENHANCEMENT,)

    # Enhance an image, raster or raster file.
    def predict(self, image: Raster | NDArray[Any] | str | Path) -> EnhancementResult:
        # Array and georeferencing.
        data, crs, transform, source = self.prepare(image)
        # Output bands.
        bands = self.predictor.dense(data)
        # Raster on the input grid.
        raster = Raster.from_array(bands, crs=crs, transform=transform)
        # Result record.
        return EnhancementResult(raster, self.outputs, source, self.model_id)


# Pipeline: super-resolution of a raster file.
create_super_resolution_pipeline = register_task_pipeline(
    "super_resolution",  # Registry id.
    "Super Resolution Pipeline",  # Name.
    "Enhance satellite imagery resolution",  # Description.
    ["ai", "imaging"],  # Domains.
    SuperResolution,  # Task API.
    method="enhance",  # Upscaling method.
)  # End of the registration.


# =============================================================================
# End of module src/unbihexium/ai/super_resolution.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
