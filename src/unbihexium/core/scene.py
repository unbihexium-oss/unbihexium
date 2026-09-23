# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/core/scene.py
# Title       : Multi-band satellite scenes
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy; reading files and
#               harmonising band grids require rasterio
# =============================================================================
#
# Abstract
# --------
# A Scene is one acquisition: a set of named band rasters (for example the
# Sentinel-2 bands B02..B12, which come at 10, 20 and 60 m) and the
# acquisition metadata (SceneMetadata: sensor, time, cloud cover, sun
# angles, processing level):
#
#   from_files, from_raster     build a scene from band files or one stack
#   harmonize                   warp every band onto the grid of a reference
#                               band (for example 20 m bands onto 10 m)
#   stack, to_array             one multi-band raster or array
#   compute_index               a spectral index from the scene bands, with
#                               the product band names of the sensor
#
# Sun geometry: the solar zenith angle is 90 degrees minus the sun
# elevation.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Band dictionaries.
from collections.abc import Mapping, Sequence

# Record containers.
from dataclasses import dataclass, field

# Acquisition times.
from datetime import datetime

# Acquisition modes.
from enum import Enum

# File paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Spectral indices.
from unbihexium.core.index import compute_index

# Band rasters.
from unbihexium.core.raster import Raster, ResamplingName

# Sensor definitions.
from unbihexium.core.sensor import SensorModel, get_sensor


# Viewing geometry of an acquisition.
class AcquisitionMode(str, Enum):
    # Single view.
    MONO = "mono"
    # Two views for stereo elevation models.
    STEREO = "stereo"
    # Three views.
    TRI_STEREO = "tri-stereo"


# Metadata of one acquisition.
@dataclass
class SceneMetadata:
    # Product identifier.
    scene_id: str
    # Sensor identifier, for example "sentinel2_msi" or "landsat8_oli".
    sensor: str
    # Acquisition time.
    acquisition_date: datetime | None = None
    # Viewing geometry.
    acquisition_mode: AcquisitionMode = AcquisitionMode.MONO
    # Cloud cover in percent.
    cloud_cover: float = 0.0
    # Sun azimuth in degrees clockwise from north.
    sun_azimuth: float | None = None
    # Sun elevation above the horizon in degrees.
    sun_elevation: float | None = None
    # Nominal pixel size in metres.
    resolution: float = 1.0
    # Band names.
    bands: list[str] = field(default_factory=list)
    # Coordinate reference system.
    crs: str = "EPSG:4326"
    # Bounds (min x, min y, max x, max y).
    bounds: tuple[float, float, float, float] | None = None
    # Processing level, for example "L1C" or "L2A".
    processing_level: str = "L1"
    # Free-form tags.
    tags: dict[str, str] = field(default_factory=dict)

    # Validate the angles and the cloud cover.
    def __post_init__(self) -> None:
        # Accept the mode as a string.
        self.acquisition_mode = AcquisitionMode(self.acquisition_mode)
        # Cloud cover is a percentage.
        if not 0.0 <= self.cloud_cover <= 100.0:
            # Explain the problem.
            raise ValueError(f"cloud_cover must be in [0, 100], got {self.cloud_cover}")
        # Elevation is an angle above or below the horizon.
        if self.sun_elevation is not None and not -90.0 <= self.sun_elevation <= 90.0:
            # Explain the problem.
            raise ValueError(f"sun_elevation must be in [-90, 90], got {self.sun_elevation}")
        # Azimuth is a compass direction.
        if self.sun_azimuth is not None and not 0.0 <= self.sun_azimuth <= 360.0:
            # Explain the problem.
            raise ValueError(f"sun_azimuth must be in [0, 360], got {self.sun_azimuth}")

    # Solar zenith angle in degrees.
    @property
    def sun_zenith(self) -> float | None:
        # Complement of the elevation.
        return None if self.sun_elevation is None else 90.0 - self.sun_elevation

    # Sensor definition, when the sensor is known.
    @property
    def sensor_model(self) -> SensorModel | None:
        # Look the sensor up.
        return get_sensor(self.sensor)

    # Plain dictionary for JSON output.
    def to_dict(self) -> dict[str, Any]:
        # Acquisition time as ISO 8601.
        when = self.acquisition_date.isoformat() if self.acquisition_date else None
        # One entry per field.
        return {
            "scene_id": self.scene_id,  # Identifier.
            "sensor": self.sensor,  # Sensor.
            "acquisition_date": when,  # Time.
            "acquisition_mode": self.acquisition_mode.value,  # Mode.
            "cloud_cover": self.cloud_cover,  # Cloud cover.
            "sun_azimuth": self.sun_azimuth,  # Sun azimuth.
            "sun_elevation": self.sun_elevation,  # Sun elevation.
            "resolution": self.resolution,  # Pixel size.
            "bands": list(self.bands),  # Bands.
            "crs": self.crs,  # Coordinate system.
            "bounds": list(self.bounds) if self.bounds else None,  # Bounds.
            "processing_level": self.processing_level,  # Level.
            "tags": dict(self.tags),  # Tags.
        }  # End of the dictionary.

    # Metadata from a dictionary written by to_dict.
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SceneMetadata:
        # Acquisition time, when present.
        when = data.get("acquisition_date")
        # Bounds, when present.
        bounds = data.get("bounds")
        # Build the metadata.
        return cls(
            scene_id=data["scene_id"],  # Identifier.
            sensor=data["sensor"],  # Sensor.
            acquisition_date=datetime.fromisoformat(when) if when else None,  # Time.
            acquisition_mode=AcquisitionMode(data.get("acquisition_mode", "mono")),  # Mode.
            cloud_cover=float(data.get("cloud_cover", 0.0)),  # Cloud cover.
            sun_azimuth=data.get("sun_azimuth"),  # Sun azimuth.
            sun_elevation=data.get("sun_elevation"),  # Sun elevation.
            resolution=float(data.get("resolution", 1.0)),  # Pixel size.
            bands=list(data.get("bands", [])),  # Bands.
            crs=data.get("crs", "EPSG:4326"),  # Coordinate system.
            bounds=tuple(bounds) if bounds else None,  # type: ignore[arg-type]
            processing_level=data.get("processing_level", "L1"),  # Level.
            tags=dict(data.get("tags", {})),  # Tags.
        )  # End of the metadata.


# One acquisition as a set of named band rasters.
@dataclass
class Scene:
    # Band rasters by band name.
    rasters: dict[str, Raster] = field(default_factory=dict)
    # Acquisition metadata.
    metadata: SceneMetadata | None = None
    # Directory or file the scene was read from.
    source: str | Path | None = None

    # Band raster by name.
    def __getitem__(self, band: str) -> Raster:
        # Look the band up.
        return self.rasters[band]

    # Set a band raster.
    def __setitem__(self, band: str, raster: Raster) -> None:
        # Store the band.
        self.rasters[band] = raster

    # Whether a band is present.
    def __contains__(self, band: object) -> bool:
        # Key test.
        return band in self.rasters

    # Band names in insertion order.
    @property
    def bands(self) -> list[str]:
        # Keys of the dictionary.
        return list(self.rasters)

    # Shape (bands, height, width) of the first band's grid.
    @property
    def shape(self) -> tuple[int, int, int]:
        # Empty scenes.
        if not self.rasters:
            # No bands.
            return (0, 0, 0)
        # First band.
        first = next(iter(self.rasters.values()))
        # Band count and grid size.
        return (len(self.rasters), first.height, first.width)

    # Add a band raster.
    def add_band(self, name: str, raster: Raster) -> None:
        # Store the band.
        self.rasters[name] = raster

    # Whether every band shares the grid of the first band.
    def is_aligned(self) -> bool:
        # Band rasters.
        rasters = list(self.rasters.values())
        # Compare every band with the first.
        return all(rasters[0].same_grid(r) for r in rasters[1:])

    # Scene with every band warped onto the grid of a reference band.
    def harmonize(
        self,  # This object.
        reference: str | None = None,  # Reference band; the finest band by default.
        method: ResamplingName = "bilinear",  # Resampling method.
    ) -> Scene:  # Scene on one grid.
        # Empty scenes are returned unchanged.
        if not self.rasters:
            # Nothing to do.
            return self
        # Finest band (largest pixel count) as the default reference.
        if reference is None:
            # Band with the most pixels.
            reference = max(self.rasters, key=lambda b: np.prod(self.rasters[b].shape[1:]))
        # Reference grid.
        target = self.rasters[reference]
        # Bands on the reference grid; bands already on it are kept.
        warped = {
            name: r if r.same_grid(target) else r.match(target, method)  # Band on the grid.
            for name, r in self.rasters.items()  # Every band.
        }  # End of the bands.
        # New scene.
        return Scene(rasters=warped, metadata=self.metadata, source=self.source)

    # Bands as one (bands, height, width) array.
    def to_array(self, bands: Sequence[str] | None = None) -> NDArray[Any]:
        # Requested bands, or all.
        names = list(bands) if bands is not None else self.bands
        # Unknown bands are an error.
        missing = [b for b in names if b not in self.rasters]
        # Report them.
        if missing:
            # Explain the problem.
            raise KeyError(f"bands {missing} not in the scene; bands: {self.bands}")
        # Arrays of the bands.
        arrays = [self.rasters[b].require_data() for b in names]
        # Grid sizes of the bands.
        sizes = {a.shape[1:] for a in arrays}
        # The bands must share the grid size.
        if len(sizes) > 1:
            # Explain the problem.
            raise ValueError(f"bands have different sizes {sorted(sizes)}; call harmonize()")
        # Concatenate along the band axis.
        return np.concatenate(arrays, axis=0)

    # Bands as one multi-band raster on the common grid.
    def stack(self, bands: Sequence[str] | None = None) -> Raster:
        # Requested bands, or all.
        names = list(bands) if bands is not None else self.bands
        # Stack the band rasters (checks the grids).
        stacked = Raster.stack([self.rasters[b] for b in names])
        # Record the band names as a tag.
        tags = {**stacked.metadata.tags, "band_names": ",".join(names)} if stacked.metadata else {}
        # Stacked raster with the tag.
        return stacked.with_data(stacked.require_data(), tags=tags)

    # Spectral index from the scene bands as a single-band raster.
    def compute_index(self, name: str, **parameters: float) -> Raster:
        # Sensor of the scene, for its band names.
        sensor = self.metadata.sensor if self.metadata else None
        # First band of every raster, with invalid pixels as NaN.
        arrays = {b: r.masked()[0].astype(float).filled(np.nan) for b, r in self.rasters.items()}
        # Unknown sensors fall back to common band names.
        known = sensor if sensor and get_sensor(sensor) else None
        # Index values.
        values = compute_index(name, arrays, sensor=known, **parameters)
        # Grid of the first band.
        first = next(iter(self.rasters.values()))
        # Raster on that grid with NaN as no-data.
        return Raster.from_array(values.astype(np.float32), first.crs, first.transform, np.nan)

    # Scene from band files.
    @classmethod
    def from_files(
        cls,  # The class.
        paths: Mapping[str, str | Path],  # Band name to file.
        metadata: SceneMetadata | None = None,  # Acquisition metadata.
        lazy: bool = False,  # Defer reading the pixels.
    ) -> Scene:  # The scene.
        # Read every band file.
        rasters = {name: Raster.from_file(path, lazy=lazy) for name, path in paths.items()}
        # Build the scene.
        return cls(rasters=rasters, metadata=metadata)

    # Scene from a multi-band raster, one band per name.
    @classmethod
    def from_raster(
        cls,  # The class.
        raster: Raster,  # Multi-band raster.
        band_names: Sequence[str] | None = None,  # Names; band_1.. by default.
        metadata: SceneMetadata | None = None,  # Acquisition metadata.
    ) -> Scene:  # The scene.
        # Names of the bands.
        names = list(band_names) if band_names else [f"band_{i + 1}" for i in range(raster.count)]
        # One name per band.
        if len(names) != raster.count:
            # Explain the problem.
            raise ValueError(f"{len(names)} names for {raster.count} bands")
        # Single-band rasters with the georeferencing of the stack.
        rasters = {name: raster.select_bands([i + 1]) for i, name in enumerate(names)}
        # Build the scene.
        return cls(rasters=rasters, metadata=metadata, source=raster.source)


# =============================================================================
# End of module src/unbihexium/core/scene.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
