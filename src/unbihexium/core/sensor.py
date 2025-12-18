"""Sensor model abstraction for satellite sensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SensorType(str, Enum):
    """Types of satellite sensors."""

    OPTICAL = "optical"
    SAR = "sar"
    MULTISPECTRAL = "multispectral"
    HYPERSPECTRAL = "hyperspectral"
    PANCHROMATIC = "panchromatic"
    THERMAL = "thermal"


@dataclass
class SensorModel:
    """Sensor model abstraction for satellite imagery."""

    name: str
    sensor_type: SensorType
    platform: str = ""
    resolution: float = 1.0
    swath_width: float = 0.0
    bands: list[str] = field(default_factory=list)
    band_wavelengths: dict[str, tuple[float, float]] = field(default_factory=dict)
    revisit_time_days: float = 0.0
    altitude_km: float = 0.0
    inclination_deg: float = 0.0
    launch_date: str = ""
    operator: str = ""

    def get_band_wavelength(self, band: str) -> tuple[float, float] | None:
        return self.band_wavelengths.get(band)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sensor_type": self.sensor_type.value,
            "platform": self.platform,
            "resolution": self.resolution,
            "swath_width": self.swath_width,
            "bands": self.bands,
            "band_wavelengths": self.band_wavelengths,
            "revisit_time_days": self.revisit_time_days,
            "altitude_km": self.altitude_km,
        }


# Common sensor definitions
SENSORS: dict[str, SensorModel] = {
    "sentinel2_msi": SensorModel(
        name="Sentinel-2 MSI",
        sensor_type=SensorType.MULTISPECTRAL,
        platform="Sentinel-2",
        resolution=10.0,
        swath_width=290.0,
        bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
        revisit_time_days=5.0,
        altitude_km=786.0,
        operator="ESA",
    ),
    "landsat8_oli": SensorModel(
        name="Landsat-8 OLI",
        sensor_type=SensorType.MULTISPECTRAL,
        platform="Landsat-8",
        resolution=30.0,
        swath_width=185.0,
        bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
        revisit_time_days=16.0,
        altitude_km=705.0,
        operator="USGS/NASA",
    ),
    "sentinel1_sar": SensorModel(
        name="Sentinel-1 SAR",
        sensor_type=SensorType.SAR,
        platform="Sentinel-1",
        resolution=5.0,
        swath_width=250.0,
        bands=["VV", "VH"],
        revisit_time_days=6.0,
        altitude_km=693.0,
        operator="ESA",
    ),
}


def get_sensor(sensor_id: str) -> SensorModel | None:
    """Get a sensor model by ID."""
    return SENSORS.get(sensor_id)
