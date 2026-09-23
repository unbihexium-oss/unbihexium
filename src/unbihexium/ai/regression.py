# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/regression.py
# Title       : Per-pixel and scene-level regression
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
# DenseRegressor predicts continuous values for every pixel, for example
# canopy height, land surface temperature, risk or suitability scores and
# elevation; it also runs the exact spectral index models (NDVI, NDWI, EVI,
# SAVI, MSI, NBR, VCI). SceneRegressor predicts one value per target for a
# whole image chip, for example crop yield of a field or the number of
# animals in a paddock. Both return a RegressionResult with output names and
# units; dense results can be written as float32 rasters.
#
# Regression models with a value range in the catalogue end in a scaled
# sigmoid, so their outputs always lie inside the range.
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

# Task API base.
from unbihexium.ai.base import ZooTask

# Result record.
from unbihexium.ai.results import RegressionResult

# Raster container.
from unbihexium.core.raster import Raster

# Task enumeration.
from unbihexium.zoo.catalog import Task


# Per-pixel regression with a model zoo U-Net or a spectral index formula.
class DenseRegressor(ZooTask):
    # Canopy height by default.
    default_model = "tree_height_estimator"
    # Dense regression and spectral index models.
    tasks = (Task.DENSE_REGRESSION, Task.SPECTRAL_INDEX)

    # Predict values for every pixel of an image, raster or raster file.
    def predict(self, image: Raster | NDArray[Any] | str | Path) -> RegressionResult:
        # Array and georeferencing.
        data, crs, transform, source = self.prepare(image)
        # Blended values.
        values = self.predictor.dense(data)
        # Configuration of the model.
        config = self.predictor.config
        # Result with names and units.
        return RegressionResult(
            values=values,  # Values (K, H, W).
            names=list(config.outputs),  # Output names.
            units=list(config.units),  # Units.
            model_id=self.model_id,  # Model.
            source=source,  # Input.
            crs=crs,  # Coordinate system.
            transform=transform,  # Georeferencing.
        )  # End of the result.


# Scene-level regression with a pooled encoder.
class SceneRegressor(ZooTask):
    # Crop yield by default.
    default_model = "yield_predictor"
    # Scene regression models only.
    tasks = (Task.SCENE_REGRESSION,)

    # Predict one value per target for an image, raster or raster file.
    def predict(self, image: Raster | NDArray[Any] | str | Path) -> RegressionResult:
        # Array and georeferencing.
        data, crs, transform, source = self.prepare(image)
        # One vector for the whole image.
        values = self.predictor.scene(data)
        # Configuration of the model.
        config = self.predictor.config
        # Result with names and units.
        return RegressionResult(
            values=values,  # Values (K,).
            names=list(config.outputs),  # Output names.
            units=list(config.units),  # Units.
            model_id=self.model_id,  # Model.
            source=source,  # Input.
            crs=crs,  # Coordinate system.
            transform=transform,  # Georeferencing of the input.
        )  # End of the result.


# Canopy height in metres.
class TreeHeightEstimator(DenseRegressor):
    # Catalogue family.
    default_model = "tree_height_estimator"


# Land surface temperature in kelvin.
class LandSurfaceTemperature(DenseRegressor):
    # Catalogue family.
    default_model = "land_surface_temperature"


# Normalised difference vegetation index, exact formula.
class NDVICalculator(DenseRegressor):
    # Catalogue family.
    default_model = "ndvi_calculator"


# Crop yield per field.
class YieldPredictor(SceneRegressor):
    # Catalogue family.
    default_model = "yield_predictor"


# =============================================================================
# End of module src/unbihexium/ai/regression.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
