# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/analysis/zonal.py
# Title       : Zonal statistics of rasters
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy; rasterize_zones needs
#               rasterio
# =============================================================================
#
# Abstract
# --------
# Statistics of the value raster within each zone of a zone raster of the
# same shape (Tomlin, 1990):
#
#   count, sum, mean, std (population), min, max, range, median,
#   majority (most frequent value; the smallest on ties), minority (least
#   frequent; the smallest on ties), variety (number of distinct values)
#   and arbitrary percentiles (linear interpolation, as numpy.percentile)
#
#   ZonalStatistics.calculate   {statistic: {zone: value}}
#   zonal_statistics            list of ZonalResult records, one per zone
#   rasterize_zones             burns polygons into a zone raster
#
# Values that are NaN or equal to `nodata` are ignored, as are cells whose
# zone is NaN or equal to `zone_nodata`. Every statistic is computed for all
# zones at once with sorting and bincount, so the cost is O(N log N) for N
# cells whatever the number of zones. Rasters are accepted wherever arrays
# are: their `data` attribute is used (first band).
#
# References
# ----------
# Tomlin, C. D. (1990). Geographic Information Systems and Cartographic
#   Modeling. Prentice Hall, Englewood Cliffs NJ.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Result record.
from dataclasses import dataclass, field

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Statistics computed by default.
DEFAULT_STATS = ("count", "sum", "mean", "std", "min", "max", "median")

# Every supported statistic.
STATISTICS = (*DEFAULT_STATS, "range", "majority", "minority", "variety")


# Statistics of one zone.
@dataclass
class ZonalResult:
    # Zone identifier.
    zone_id: int | float | str
    # Number of valid cells.
    count: int
    # Sum of the values.
    sum: float
    # Mean of the values.
    mean: float
    # Population standard deviation.
    std: float
    # Smallest value.
    min: float
    # Largest value.
    max: float
    # Median.
    median: float
    # Most frequent value.
    majority: float | None = None
    # Least frequent value.
    minority: float | None = None
    # Number of distinct values.
    variety: int = 0
    # Requested percentiles by percent.
    percentiles: dict[float, float] = field(default_factory=dict)

    # Range of the values.
    @property
    def range(self) -> float:
        # Largest minus smallest.
        return self.max - self.min


# First band of an array or raster as float64.
def _band(layer: Any) -> NDArray[np.float64]:
    # Rasters expose their pixels as .data.
    a = np.asarray(getattr(layer, "data", layer), dtype=np.float64)
    # First band of (bands, rows, cols).
    return a[0] if a.ndim == 3 else a


# Burn polygons into an integer zone raster (0 = outside every polygon).
def rasterize_zones(
    geometries: list[Any],  # GeoJSON-like mappings or shapely geometries.
    shape: tuple[int, int],  # Output (rows, cols).
    transform: tuple[float, ...],  # Affine coefficients (a, b, c, d, e, f).
    ids: list[int] | None = None,  # Zone id per geometry; 1, 2, ... by default.
    all_touched: bool = False,  # Burn every touched cell, not only cell centres.
) -> NDArray[np.int32]:  # Zone raster; later geometries overwrite earlier ones.
    # rasterio is optional; import it only here.
    from rasterio.features import rasterize

    # Affine transform type of rasterio.
    from rasterio.transform import Affine

    # Default ids.
    zone_ids = ids if ids is not None else list(range(1, len(geometries) + 1))
    # One id per geometry.
    if len(zone_ids) != len(geometries):
        # Report the mismatch.
        raise ValueError("ids must have one entry per geometry")
    # Burn the geometries.
    return rasterize(
        list(zip(geometries, zone_ids)),  # Shapes with their values.
        out_shape=shape,  # Grid size.
        transform=Affine(*transform[:6]),  # Georeferencing.
        fill=0,  # Background.
        all_touched=all_touched,  # Cell selection rule.
        dtype="int32",  # Zone ids.
    )  # End of the rasterisation.


# Statistics of all zones at once.
def zonal_table(
    values: Any,  # Value array or raster.
    zones: Any,  # Zone array or raster of the same shape.
    stats: list[str] | tuple[str, ...] | None = None,  # Names from STATISTICS.
    percentiles: list[float] | None = None,  # Percentiles in [0, 100].
    nodata: float | None = None,  # Value to ignore in the value raster.
    zone_nodata: float | None = None,  # Zone to ignore.
) -> dict[str, dict[Any, float]]:  # {statistic or "p<q>": {zone: value}}.
    # Value band.
    v = _band(values)
    # Zone band.
    z = _band(zones)
    # Shapes must agree.
    if v.shape != z.shape:
        # Report the mismatch.
        raise ValueError(f"values {v.shape} and zones {z.shape} differ in shape")
    # Requested statistics.
    names = list(stats) if stats is not None else list(DEFAULT_STATS)
    # All names must be known.
    unknown = [s for s in names if s not in STATISTICS]
    # Report unknown names.
    if unknown:
        # Name them.
        raise ValueError(f"unknown statistics {unknown}; expected names from {STATISTICS}")
    # Percentiles must lie in [0, 100].
    pcts = [float(p) for p in (percentiles or [])]
    # Check the range.
    if any(not 0.0 <= p <= 100.0 for p in pcts):
        # Report the invalid percentile.
        raise ValueError("percentiles must lie in [0, 100]")
    # Cells that take part.
    ok = np.isfinite(v) & np.isfinite(z)
    # Drop the value nodata.
    if nodata is not None:
        # Exclude nodata values.
        ok &= v != nodata
    # Drop the zone nodata.
    if zone_nodata is not None:
        # Exclude nodata zones.
        ok &= z != zone_nodata
    # Valid values and zones.
    val, zon = v[ok], z[ok]
    # Zone ids and the zone index of every cell.
    ids, inv = np.unique(zon, return_inverse=True)
    # Sort cells by zone, then value.
    order = np.lexsort((val, inv))
    # Sorted values and zone indices.
    sv, si = val[order], inv[order]
    # Number of zones.
    k = ids.size
    # Cells per zone.
    count = np.bincount(inv, minlength=k)
    # First sorted position of each zone.
    start = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int64)
    # Sum per zone.
    total = np.bincount(inv, weights=val, minlength=k)
    # Mean per zone.
    mean = total / np.maximum(count, 1)
    # Population standard deviation, two-pass for accuracy.
    squares = np.bincount(inv, weights=(val - mean[inv]) ** 2, minlength=k)
    # Square root of the mean squared deviation.
    std = np.sqrt(squares / np.maximum(count, 1))
    # Smallest and largest value per zone from the sorted order.
    vmin, vmax = sv[start] if k else np.zeros(0), sv[start + count - 1] if k else np.zeros(0)

    # Percentile q of every zone with linear interpolation.
    def percentile(q: float) -> NDArray[np.float64]:
        # Fractional position within each zone.
        pos = q / 100.0 * (count - 1)
        # Lower and upper neighbours.
        lo, hi = np.floor(pos).astype(np.int64), np.ceil(pos).astype(np.int64)
        # Interpolate between them.
        return sv[start + lo] + (pos - lo) * (sv[start + hi] - sv[start + lo])

    # Runs of equal (zone, value) in the sorted order.
    new_run = np.ones(sv.size, dtype=bool)
    # A run starts where the zone or the value changes.
    new_run[1:] = (si[1:] != si[:-1]) | (sv[1:] != sv[:-1])
    # Start positions of the runs.
    run_start = np.flatnonzero(new_run)
    # Length of every run.
    run_len = np.diff(np.append(run_start, sv.size))
    # Zone and value of every run.
    run_zone, run_value = si[run_start], sv[run_start]
    # Distinct values per zone.
    variety = np.bincount(run_zone, minlength=k)

    # Value of the first run per zone after sorting by (zone, key, value).
    def pick(key: NDArray[np.int64]) -> NDArray[np.float64]:
        # Order of the runs.
        o = np.lexsort((run_value, key, run_zone))
        # First run of every zone in that order.
        first = np.ones(o.size, dtype=bool)
        # A zone starts where the zone index changes.
        first[1:] = run_zone[o][1:] != run_zone[o][:-1]
        # Values of those runs.
        return run_value[o][first]

    # Every computable statistic.
    columns: dict[str, Any] = {
        "count": lambda: count.astype(np.float64),  # Cells.
        "sum": lambda: total,  # Sum.
        "mean": lambda: mean,  # Mean.
        "std": lambda: std,  # Standard deviation.
        "min": lambda: vmin,  # Minimum.
        "max": lambda: vmax,  # Maximum.
        "range": lambda: vmax - vmin,  # Range.
        "median": lambda: percentile(50.0),  # Median.
        "majority": lambda: pick(-run_len),  # Longest run first.
        "minority": lambda: pick(run_len),  # Shortest run first.
        "variety": lambda: variety.astype(np.float64),  # Distinct values.
    }  # End of the statistics.
    # Zone ids as Python numbers; integral ids become int.
    keys = [int(i) if float(i).is_integer() else float(i) for i in ids]
    # Requested statistics by zone.
    table = {name: dict(zip(keys, (float(x) for x in columns[name]()))) for name in names}
    # Requested percentiles.
    for q in pcts:
        # Column name such as p25 or p97.5.
        label = f"p{q:g}"
        # Values by zone.
        table[label] = dict(zip(keys, (float(x) for x in percentile(q))))
    # Return the table.
    return table


# Zonal statistics as a table of dictionaries.
class ZonalStatistics:
    # Configure the nodata values.
    def __init__(self, nodata: float | None = None, zone_nodata: float | None = None) -> None:
        # Value to ignore in the value raster.
        self.nodata = nodata
        # Zone to ignore.
        self.zone_nodata = zone_nodata

    # Compute statistics per zone.
    def calculate(
        self,  # The instance.
        values: Any,  # Value array or raster.
        zones: Any,  # Zone array or raster.
        stats: list[str] | None = None,  # Statistic names.
        percentiles: list[float] | None = None,  # Percentiles in [0, 100].
    ) -> dict[str, dict[Any, float]]:  # {statistic: {zone: value}}.
        # Shared implementation.
        return zonal_table(values, zones, stats, percentiles, self.nodata, self.zone_nodata)


# Zonal statistics as one record per zone.
def zonal_statistics(
    raster: Any,  # Value array or raster.
    zones: Any,  # Zone array or raster, or polygons (needs a georeferenced raster).
    stats: list[str] | None = None,  # Ignored except for validation; records hold all fields.
    percentiles: list[float] | None = None,  # Percentiles in [0, 100].
    nodata: float | None = None,  # Value to ignore.
) -> list[ZonalResult]:  # One record per zone, sorted by zone id.
    # Polygons are burnt into a zone raster first.
    if isinstance(zones, (list, tuple)):
        # The value raster must be georeferenced.
        metadata = getattr(raster, "metadata", None)
        # Report missing georeferencing.
        if metadata is None:
            # Polygons need a transform.
            raise ValueError("polygon zones need a raster with metadata (transform)")
        # Burn the polygons with ids 1, 2, ...
        zones = rasterize_zones(list(zones), _band(raster).shape, tuple(metadata.transform))
        # Cells outside every polygon form no zone.
        zone_nodata: float | None = 0
    # Arrays and rasters are used directly.
    else:
        # No zone nodata.
        zone_nodata = None
    # Validate the requested names.
    if stats is not None:
        # Unknown names raise.
        zonal_table(np.zeros((1, 1)), np.zeros((1, 1)), stats)
    # All record fields.
    fields = [
        "count",
        "sum",
        "mean",
        "std",
        "min",
        "max",
        "median",
        "majority",
        "minority",
        "variety",
    ]
    # Statistics of all zones.
    table = zonal_table(raster, zones, fields, percentiles, nodata, zone_nodata)
    # Percentile columns.
    pcols = {float(q): f"p{float(q):g}" for q in (percentiles or [])}
    # One record per zone.
    records = []
    # Zones in ascending order.
    for zone in table["count"]:
        # Record of the zone.
        record = ZonalResult(
            zone_id=zone,  # Zone.
            count=int(table["count"][zone]),  # Cells.
            sum=table["sum"][zone],  # Sum.
            mean=table["mean"][zone],  # Mean.
            std=table["std"][zone],  # Standard deviation.
            min=table["min"][zone],  # Minimum.
            max=table["max"][zone],  # Maximum.
            median=table["median"][zone],  # Median.
            majority=table["majority"][zone],  # Most frequent value.
            minority=table["minority"][zone],  # Least frequent value.
            variety=int(table["variety"][zone]),  # Distinct values.
            percentiles={q: table[c][zone] for q, c in pcols.items()},  # Percentiles.
        )  # End of the record.
        # Collect it.
        records.append(record)
    # Return the records.
    return records


# =============================================================================
# End of module src/unbihexium/analysis/zonal.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
