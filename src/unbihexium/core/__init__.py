# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/core/__init__.py
# Title       : Core data model of the library
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy; file input and output
#               need rasterio, vectors need GeoPandas, Shapely and pyproj
# =============================================================================
#
# Abstract
# --------
# The building blocks shared by every other package:
#
#   raster      Raster, RasterMetadata: georeferenced arrays
#   vector      Vector, VectorMetadata: georeferenced features
#   tile        Tile, TileGrid, TileIndex: tiling and mosaicking
#   index       SpectralIndex, IndexRegistry, compute_index: spectral indices
#   sensor      SensorModel, SensorType, get_sensor: sensor band tables
#   scene       Scene, SceneMetadata: one acquisition as named bands
#   product     Product, ProductMetadata, ProductType: derived products
#   model       ModelConfig, ModelWrapper: framework-neutral inference
#   pipeline    Pipeline, PipelineConfig, PipelineRun: processing runs
#   evidence    Evidence, ProvenanceRecord: SHA-256 audit trail
#
# Importing this package imports NumPy only; rasterio, GeoPandas, PyTorch
# and ONNX Runtime are imported by the functions that need them.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Evidence and provenance.
from unbihexium.core.evidence import (
    Evidence,  # One artefact and its SHA-256 digest.
    EvidenceType,  # Role of an artefact.
    ProvenanceRecord,  # Provenance of a run.
)  # End of the evidence imports.

# Spectral indices.
from unbihexium.core.index import (
    IndexCategory,  # Thematic group of an index.
    IndexRegistry,  # Registry of indices.
    SpectralIndex,  # Index definition.
    compute_index,  # Compute an index by name.
)  # End of the index imports.

# Models.
from unbihexium.core.model import (
    ModelConfig,  # Model configuration.
    ModelFramework,  # Framework of a model.
    ModelTask,  # Task of a model.
    ModelWrapper,  # Pre- and postprocessing around a model.
)  # End of the model imports.

# Pipelines.
from unbihexium.core.pipeline import (
    Pipeline,  # Ordered processing steps.
    PipelineConfig,  # Pipeline configuration.
    PipelineRun,  # Record of a run.
    PipelineStatus,  # State of a run.
)  # End of the pipeline imports.

# Products.
from unbihexium.core.product import (
    Product,  # Derived product.
    ProductMetadata,  # Product metadata.
    ProductType,  # Kind of product.
)  # End of the product imports.

# Rasters.
from unbihexium.core.raster import (
    Raster,  # Georeferenced array.
    RasterMetadata,  # Raster metadata.
)  # End of the raster imports.

# Scenes.
from unbihexium.core.scene import (
    Scene,  # One acquisition.
    SceneMetadata,  # Acquisition metadata.
)  # End of the scene imports.

# Sensors.
from unbihexium.core.sensor import (
    SensorModel,  # Sensor definition.
    SensorType,  # Family of a sensor.
    SpectralBand,  # One optical band.
    get_sensor,  # Sensor by identifier.
)  # End of the sensor imports.

# Tiles.
from unbihexium.core.tile import (
    Tile,  # One tile of data.
    TileGrid,  # Grid of tiles over a raster.
    TileIndex,  # Position of a tile.
)  # End of the tile imports.

# Vectors.
from unbihexium.core.vector import (
    Vector,  # Georeferenced features.
    VectorMetadata,  # Vector metadata.
)  # End of the vector imports.

# Public names of the package.
__all__ = [
    "Evidence",  # One artefact and its SHA-256 digest.
    "EvidenceType",  # Role of an artefact.
    "IndexCategory",  # Thematic group of an index.
    "IndexRegistry",  # Registry of indices.
    "ModelConfig",  # Model configuration.
    "ModelFramework",  # Framework of a model.
    "ModelTask",  # Task of a model.
    "ModelWrapper",  # Model wrapper.
    "Pipeline",  # Ordered processing steps.
    "PipelineConfig",  # Pipeline configuration.
    "PipelineRun",  # Record of a run.
    "PipelineStatus",  # State of a run.
    "Product",  # Derived product.
    "ProductMetadata",  # Product metadata.
    "ProductType",  # Kind of product.
    "ProvenanceRecord",  # Provenance of a run.
    "Raster",  # Georeferenced array.
    "RasterMetadata",  # Raster metadata.
    "Scene",  # One acquisition.
    "SceneMetadata",  # Acquisition metadata.
    "SensorModel",  # Sensor definition.
    "SensorType",  # Family of a sensor.
    "SpectralBand",  # One optical band.
    "SpectralIndex",  # Index definition.
    "Tile",  # One tile of data.
    "TileGrid",  # Grid of tiles.
    "TileIndex",  # Position of a tile.
    "Vector",  # Georeferenced features.
    "VectorMetadata",  # Vector metadata.
    "compute_index",  # Compute an index by name.
    "get_sensor",  # Sensor by identifier.
]

# =============================================================================
# End of module src/unbihexium/core/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
