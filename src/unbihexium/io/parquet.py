# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/io/parquet.py
# Title       : GeoParquet input and output
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires geopandas and pyarrow
# =============================================================================
#
# Abstract
# --------
# GeoParquet stores vector data in Apache Parquet: geometries as WKB (or
# native GeoArrow) columns and a "geo" key in the file metadata that
# records the version, the primary geometry column, its encoding, CRS
# (PROJJSON), geometry types and bounding box. Columnar storage makes it
# much faster and smaller than GeoJSON or Shapefiles for large detection or
# parcel tables.
#
#   read_geoparquet        read a file, optionally selected columns, only
#                          the features that intersect a box and
#                          reprojected
#   write_geoparquet       write a GeoDataFrame, optionally with the
#                          per-row bounding box "covering" column of
#                          GeoParquet 1.1 that enables spatial filtering
#                          without decoding the geometries
#   geoparquet_metadata    the decoded "geo" metadata, without reading rows
#   geoparquet_bounds      the bounding box of the primary geometry column
#   geojson_to_geoparquet  conversions between GeoJSON documents and
#   geoparquet_to_geojson  GeoParquet files
#
# References
# ----------
# Open Geospatial Consortium (2024). GeoParquet specification, version
# 1.1.0. https://geoparquet.org/releases/v1.1.0/
# Apache Software Foundation (2013 to 2026). Apache Parquet file format.
# https://parquet.apache.org/docs/file-format/
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON metadata.
import json

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any


# geopandas, imported lazily with a clear error.
def _geopandas() -> Any:
    # Import on first use.
    try:
        # Vector data frames.
        import geopandas
    # Missing installation.
    except ImportError as exc:
        # Explain the requirement.
        raise ImportError("geopandas and pyarrow are required for GeoParquet support") from exc
    # Return the module.
    return geopandas


# Read a GeoParquet file into a GeoDataFrame.
def read_geoparquet(
    path: str | Path,  # GeoParquet file.
    columns: list[str] | None = None,  # Attribute columns; the geometry is always read.
    bbox: tuple[float, float, float, float] | None = None,  # Keep features intersecting it.
    to_crs: Any = None,  # Reproject the result to this CRS.
) -> Any:  # GeoDataFrame.
    # Vector library.
    gpd = _geopandas()
    # Primary geometry column from the metadata.
    geometry = geoparquet_metadata(path).get("primary_column", "geometry")
    # Requested columns plus the geometry.
    wanted = None if columns is None else [*[c for c in columns if c != geometry], geometry]
    # Read the rows.
    frame = gpd.read_parquet(Path(path), columns=wanted)
    # Spatial filter with exact intersection (in the CRS of the file).
    if bbox is not None:
        # Box geometry.
        from shapely.geometry import box

        # Features that intersect the box.
        frame = frame[frame.intersects(box(*bbox))]
    # Reprojection on request.
    if to_crs is not None:
        # Transform the geometries.
        frame = frame.to_crs(to_crs)
    # Return the frame.
    return frame


# Accept (frame, path) and the (path, frame) order of earlier releases.
def _frame_and_path(first: Any, second: Any) -> tuple[Any, Path]:
    # A path in the first position belongs to the earlier order.
    if isinstance(first, (str, Path)) and not isinstance(second, (str, Path)):
        # Swap the arguments.
        return second, Path(first)
    # The documented order.
    return first, Path(second)


# Write a GeoDataFrame as GeoParquet; returns the path.
def write_geoparquet(
    gdf: Any,  # GeoDataFrame with an active geometry column.
    path: str | Path,  # Output file.
    compression: str = "snappy",  # snappy, zstd, gzip, brotli or none.
    write_bbox: bool = False,  # Add the GeoParquet 1.1 bounding box covering column.
    index: bool | None = None,  # Store the index; None stores non-default indexes.
) -> Path:  # Written file.
    # Support the argument order of earlier releases.
    gdf, path = _frame_and_path(gdf, path)
    # Only GeoDataFrames carry geometry metadata.
    if not hasattr(gdf, "geometry") or not hasattr(gdf, "to_parquet"):
        # Explain the requirement.
        raise TypeError("write_geoparquet needs a geopandas GeoDataFrame")
    # Parent directory.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Codec; pyarrow expects None for no compression.
    codec = None if compression.lower() == "none" else compression
    # Write the file with the geo metadata.
    gdf.to_parquet(path, index=index, compression=codec, write_covering_bbox=write_bbox)
    # Return the path.
    return path


# Decoded "geo" metadata of a GeoParquet file.
def geoparquet_metadata(path: str | Path) -> dict[str, Any]:
    # Parquet schema reader.
    import pyarrow.parquet as pq

    # Key-value metadata of the schema.
    metadata = pq.read_schema(Path(path)).metadata or {}
    # The GeoParquet key.
    raw = metadata.get(b"geo")
    # Plain Parquet files have none.
    if raw is None:
        # Explain the problem.
        raise ValueError(f"{path} has no GeoParquet metadata")
    # Parse the JSON text.
    return json.loads(raw.decode("utf-8"))


# Bounding box (min x, min y, max x, max y) of the primary geometry column.
def geoparquet_bounds(path: str | Path) -> tuple[float, float, float, float]:
    # Metadata of the file.
    meta = geoparquet_metadata(path)
    # Description of the primary column.
    column = meta["columns"][meta["primary_column"]]
    # Recorded box (optional in the specification).
    recorded = column.get("bbox")
    # Two-dimensional boxes have four values.
    if recorded and len(recorded) == 4:
        # Return the recorded box.
        return tuple(float(v) for v in recorded)  # type: ignore[return-value]
    # Otherwise read the geometries.
    bounds = read_geoparquet(path).total_bounds
    # Box of all geometries.
    return (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))


# Write a GeoJSON document as GeoParquet.
def geojson_to_geoparquet(document: dict[str, Any], path: str | Path, **kwargs: Any) -> Path:
    # Conversion to a data frame.
    from unbihexium.io.geojson import to_geodataframe

    # Frame of the document, then the file.
    return write_geoparquet(to_geodataframe(document), path, **kwargs)


# Read a GeoParquet file as a GeoJSON FeatureCollection.
def geoparquet_to_geojson(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    # Conversion from a data frame.
    from unbihexium.io.geojson import from_geodataframe

    # Frame of the file, then the document.
    return from_geodataframe(read_geoparquet(path, **kwargs))


# =============================================================================
# End of module src/unbihexium/io/parquet.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
