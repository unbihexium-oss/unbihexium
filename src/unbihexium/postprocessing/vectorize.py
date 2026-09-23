# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/postprocessing/vectorize.py
# Title       : Raster to polygon vectorisation and polygon simplification
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy, rasterio and Shapely;
#               GeoPandas for polygons_to_geodataframe
# =============================================================================
#
# Abstract
# --------
# Conversion of class maps and masks to vector polygons:
#
#   raster_to_polygons       polygons of connected regions of equal value,
#                            in map coordinates of an affine transform
#   simplify_polygons        Douglas-Peucker simplification that keeps
#                            polygons valid
#   polygons_to_geodataframe GeoDataFrame with value and area columns
#
# The transform is an affine.Affine (as used by rasterio) or its six
# coefficients (a, b, c, d, e, f) of x = a col + b row + c,
# y = d col + e row + f; note that GDAL geotransforms order them
# (c, a, b, f, d, e). Without a transform, pixel coordinates are used.
#
# References
# ----------
#   Douglas, D. H., Peucker, T. K. (1973). Algorithms for the reduction of
#     the number of points required to represent a digitized line or its
#     caricature. Cartographica 10(2), 112-122.
#   GDAL/OGR contributors. GDALPolygonize, GDAL geospatial data abstraction
#     library (used through rasterio.features.shapes).
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values and sequences.
from typing import Any, Sequence

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Polygonisation of rasters.
from rasterio.features import shapes

# Affine transforms used by rasterio.
from rasterio.transform import Affine

# Geometry types and constructors.
from shapely.geometry import shape as to_geometry

# Data types accepted by GDALPolygonize.
_SHAPES_DTYPES = (np.int16, np.int32, np.uint8, np.uint16, np.float32)


# Affine transform from any accepted representation.
def as_affine(transform: Any | None) -> Affine:
    # Identity: pixel coordinates.
    if transform is None:
        # Column is x, row is y.
        return Affine.identity()
    # Already an affine transform.
    if isinstance(transform, Affine):
        # Return it unchanged.
        return transform
    # Six coefficients (a, b, c, d, e, f).
    coef = tuple(float(v) for v in transform)
    # Only six or nine coefficients describe a 2D affine transform.
    if len(coef) not in (6, 9):
        # Explain the requirement.
        raise ValueError("transform must have 6 (or 9) coefficients")
    # Build the transform.
    return Affine(*coef[:6])


# Polygons of connected regions of equal value.
def raster_to_polygons(
    image: NDArray[Any],  # Class map or mask, (H, W).
    transform: Any | None = None,  # Affine transform of the grid.
    mask: NDArray[Any] | None = None,  # True where pixels are vectorised.
    connectivity: int = 4,  # 4 or 8.
    skip_values: Sequence[float] = (0,),  # Values that produce no polygon.
) -> list[tuple[Any, float]]:  # (polygon, value) pairs.
    # Input as array.
    arr = np.asarray(image)
    # Only single-band images are supported.
    if arr.ndim != 2:
        # Explain the requirement.
        raise ValueError(f"expected an (H, W) array, got shape {arr.shape}")
    # Only 4 and 8 connectivity exist.
    if connectivity not in (4, 8):
        # Explain the accepted values.
        raise ValueError("connectivity must be 4 or 8")
    # Booleans become uint8.
    if arr.dtype == bool:
        # 0 and 1.
        arr = arr.astype(np.uint8)
    # Other types are converted to the nearest supported type.
    if arr.dtype not in _SHAPES_DTYPES:
        # Integers fit int32 for class maps; other values use float32.
        arr = arr.astype(np.int32 if np.issubdtype(arr.dtype, np.integer) else np.float32)
    # Pixels whose value is skipped.
    keep = ~np.isin(arr, np.asarray(skip_values, dtype=arr.dtype))
    # Combine with the user mask.
    if mask is not None:
        # Both conditions must hold.
        keep &= np.asarray(mask, dtype=bool)
    # Polygonise.
    found = shapes(arr, mask=keep, connectivity=connectivity, transform=as_affine(transform))
    # Convert GeoJSON-like dictionaries to Shapely polygons.
    return [(to_geometry(geom), float(value)) for geom, value in found]


# Douglas-Peucker simplification of polygons.
def simplify_polygons(
    polygons: Sequence[Any],  # Shapely geometries or (geometry, value) pairs.
    tolerance: float,  # Maximum distance of removed vertices, in map units.
    preserve_topology: bool = True,  # Keep polygons valid.
) -> list[Any]:  # Simplified items of the same kind as the input.
    # The tolerance must not be negative.
    if tolerance < 0:
        # Explain the requirement.
        raise ValueError("tolerance must not be negative")
    # Simplified items.
    out = []
    # Visit every item.
    for item in polygons:
        # Pairs keep their value.
        if isinstance(item, tuple):
            # Simplified geometry of the pair.
            geom = item[0].simplify(tolerance, preserve_topology=preserve_topology)
            # Keep the other members of the pair.
            out.append((geom, *item[1:]))
        # Plain geometries.
        else:
            # Simplify the geometry.
            out.append(item.simplify(tolerance, preserve_topology=preserve_topology))
    # Return the simplified items.
    return out


# GeoDataFrame of the polygons of a class map.
def polygons_to_geodataframe(
    image: NDArray[Any],  # Class map or mask, (H, W).
    transform: Any | None = None,  # Affine transform of the grid.
    crs: Any | None = None,  # Coordinate reference system.
    connectivity: int = 4,  # 4 or 8.
    skip_values: Sequence[float] = (0,),  # Values that produce no polygon.
    simplify_tolerance: float | None = None,  # Optional simplification tolerance.
) -> Any:  # geopandas.GeoDataFrame with value, area and geometry columns.
    # GeoPandas is imported only when needed.
    import geopandas as gpd

    # Polygons of the regions.
    pairs = raster_to_polygons(image, transform, None, connectivity, skip_values)
    # Simplify when requested.
    if simplify_tolerance is not None:
        # Simplified pairs.
        pairs = simplify_polygons(pairs, simplify_tolerance)
    # Geometries.
    geoms = [g for g, _ in pairs]
    # Values.
    values = [v for _, v in pairs]
    # Build the data frame.
    frame = gpd.GeoDataFrame({"value": values}, geometry=geoms, crs=crs)
    # Planar area in squared map units.
    frame["area"] = frame.geometry.area
    # Return the frame.
    return frame


# =============================================================================
# End of module src/unbihexium/postprocessing/vectorize.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
