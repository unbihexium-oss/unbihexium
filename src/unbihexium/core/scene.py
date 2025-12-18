"""Scene abstraction for multi-band satellite imagery."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unbihexium.core.raster import Raster


class AcquisitionMode(str, Enum):
    """Satellite acquisition modes."""

    MONO = "mono"
    STEREO = "stereo"
    TRI_STEREO = "tri-stereo"


@dataclass
class SceneMetadata:
    """Metadata for a satellite scene."""

    scene_id: str
    sensor: str
    acquisition_date: datetime | None = None
    acquisition_mode: AcquisitionMode = AcquisitionMode.MONO
    cloud_cover: float = 0.0
    sun_azimuth: float | None = None
    sun_elevation: float | None = None
    resolution: float = 1.0
    bands: list[str] = field(default_factory=list)
    crs: str = "EPSG:4326"
    bounds: tuple[float, float, float, float] | None = None
    processing_level: str = "L1"
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class Scene:
    """Multi-band satellite scene abstraction."""

    rasters: dict[str, Raster] = field(default_factory=dict)
    metadata: SceneMetadata | None = None
    source: str | Path | None = None

    def __getitem__(self, band: str) -> Raster:
        return self.rasters[band]

    def __setitem__(self, band: str, raster: Raster) -> None:
        self.rasters[band] = raster

    @property
    def bands(self) -> list[str]:
        return list(self.rasters.keys())

    @property
    def shape(self) -> tuple[int, int, int]:
        if not self.rasters:
            return (0, 0, 0)
        first = next(iter(self.rasters.values()))
        return (len(self.rasters), first.height, first.width)

    def add_band(self, name: str, raster: Raster) -> None:
        self.rasters[name] = raster

    def to_array(self, bands: list[str] | None = None) -> NDArray[np.floating[Any]]:
        bands = bands or self.bands
        arrays = [self.rasters[b].data for b in bands if b in self.rasters]
        if not arrays:
            return np.array([])
        return np.concatenate([a if a is not None else np.array([]) for a in arrays], axis=0)

    @classmethod
    def from_raster(cls, raster: Raster, band_names: list[str] | None = None) -> Scene:
        if raster.data is None:
            return cls()
        count = raster.count
        band_names = band_names or [f"band_{i + 1}" for i in range(count)]
        rasters = {}
        for i, name in enumerate(band_names[:count]):
            band_data = (
                raster.data[i : i + 1] if raster.data.ndim == 3 else raster.data[np.newaxis, ...]
            )
            rasters[name] = Raster.from_array(
                band_data,
                crs=raster.metadata.crs if raster.metadata else "EPSG:4326",
            )
        return cls(rasters=rasters)
