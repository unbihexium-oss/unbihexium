"""Zonal statistics for raster analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unbihexium.core.raster import Raster
from unbihexium.core.vector import Vector


@dataclass
class ZonalResult:
    """Result of zonal statistics calculation."""

    zone_id: int | str
    count: int
    sum: float
    mean: float
    std: float
    min: float
    max: float
    median: float
    majority: float | None = None
    minority: float | None = None
    variety: int = 0
    percentiles: dict[int, float] = field(default_factory=dict)


def zonal_statistics(
    raster: Raster,
    zones: Vector | Raster,
    stats: list[str] | None = None,
    percentiles: list[int] | None = None,
) -> list[ZonalResult]:
    """Calculate zonal statistics.

    Args:
        raster: Raster with values to analyze.
        zones: Vector or raster defining zones.
        stats: Statistics to calculate (count, sum, mean, std, min, max, median).
        percentiles: Percentile values to calculate (e.g., [25, 50, 75]).

    Returns:
        List of ZonalResult for each zone.
    """
    raster.load()
    if raster.data is None:
        return []

    stats = stats or ["count", "sum", "mean", "std", "min", "max", "median"]
    percentiles = percentiles or []

    results = []

    if isinstance(zones, Raster):
        results = _zonal_from_raster(raster.data, zones, stats, percentiles)
    else:
        results = _zonal_from_vector(raster, zones, stats, percentiles)

    return results


def _zonal_from_raster(
    values: NDArray[np.floating[Any]],
    zones: Raster,
    stats: list[str],
    percentiles: list[int],
) -> list[ZonalResult]:
    """Calculate zonal stats from raster zones."""
    zones.load()
    if zones.data is None:
        return []

    zone_data = zones.data[0] if zones.data.ndim == 3 else zones.data
    value_data = values[0] if values.ndim == 3 else values

    unique_zones = np.unique(zone_data[~np.isnan(zone_data)])
    results = []

    for zone_id in unique_zones:
        mask = zone_data == zone_id
        zone_values = value_data[mask]
        zone_values = zone_values[~np.isnan(zone_values)]

        if len(zone_values) == 0:
            continue

        pcts = {p: float(np.percentile(zone_values, p)) for p in percentiles}

        results.append(
            ZonalResult(
                zone_id=int(zone_id),
                count=len(zone_values),
                sum=float(np.sum(zone_values)),
                mean=float(np.mean(zone_values)),
                std=float(np.std(zone_values)),
                min=float(np.min(zone_values)),
                max=float(np.max(zone_values)),
                median=float(np.median(zone_values)),
                variety=len(np.unique(zone_values)),
                percentiles=pcts,
            )
        )

    return results


def _zonal_from_vector(
    raster: Raster,
    zones: Vector,
    stats: list[str],
    percentiles: list[int],
) -> list[ZonalResult]:
    """Calculate zonal stats from vector zones."""
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine

    if raster.data is None or raster.metadata is None:
        return []

    transform = Affine(*raster.metadata.transform[:6])
    results = []

    for i, feature in enumerate(zones.features()):
        zone_id = feature.get("id", i)
        geom = feature["geometry"]

        mask = geometry_mask(
            [geom],
            out_shape=(raster.metadata.height, raster.metadata.width),
            transform=transform,
            invert=True,
        )

        value_data = raster.data[0] if raster.data.ndim == 3 else raster.data
        zone_values = value_data[mask]
        zone_values = zone_values[~np.isnan(zone_values)]

        if len(zone_values) == 0:
            continue

        pcts = {p: float(np.percentile(zone_values, p)) for p in percentiles}

        results.append(
            ZonalResult(
                zone_id=zone_id,
                count=len(zone_values),
                sum=float(np.sum(zone_values)),
                mean=float(np.mean(zone_values)),
                std=float(np.std(zone_values)),
                min=float(np.min(zone_values)),
                max=float(np.max(zone_values)),
                median=float(np.median(zone_values)),
                variety=len(np.unique(zone_values)),
                percentiles=pcts,
            )
        )

    return results
