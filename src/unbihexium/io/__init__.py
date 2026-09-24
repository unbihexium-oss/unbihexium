# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/io/__init__.py
# Title       : Raster, vector and catalogue input and output
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy; rasterio, zarr,
#               geopandas, pyproj and requests are imported when used
# =============================================================================
#
# Abstract
# --------
# Adapters for the file formats of Earth observation workflows:
#
#   geotiff   GeoTIFF and Cloud Optimized GeoTIFF (rasterio / GDAL)
#   zarr_io   chunked arrays and georeferenced rasters in Zarr v2 and v3
#   geojson   GeoJSON documents (RFC 7946) as dictionaries
#   parquet   GeoParquet vector tables (geopandas / pyarrow)
#   stac      SpatioTemporal Asset Catalog items, catalogues and API search
#
# Writers take the data first and the path second; the path-first order of
# earlier releases is still accepted. Optional dependencies are imported
# inside the functions, so importing this package is cheap.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# GeoJSON.
from unbihexium.io.geojson import (
    features_to_geojson,  # FeatureCollection from features.
    geojson_bounds,  # Bounding box.
    geojson_crs,  # CRS of a document.
    geojson_problems,  # Structural problems of a document.
    geometry_to_feature,  # Feature from a geometry.
    read_geojson,  # Read a file.
    reproject_geojson,  # Change the CRS.
    rewind,  # Right-hand rule.
    validate_geojson,  # Structural validation.
    write_geojson,  # Write a file.
)  # End of the GeoJSON imports.

# GeoTIFF and COG.
from unbihexium.io.geotiff import (
    build_overviews,  # Add overviews.
    geotiff_info,  # Metadata without pixels.
    is_cog,  # COG layout test.
    read_cog,  # Read a remote COG.
    read_geotiff,  # Read a file.
    read_raster,  # Read into a Raster.
    write_cog,  # Write a COG.
    write_geotiff,  # Write a file.
    write_raster,  # Write a Raster.
)  # End of the GeoTIFF imports.

# GeoParquet.
from unbihexium.io.parquet import (
    geoparquet_bounds,  # Bounding box.
    geoparquet_metadata,  # The geo metadata.
    read_geoparquet,  # Read a file.
    write_geoparquet,  # Write a file.
)  # End of the GeoParquet imports.

# STAC.
from unbihexium.io.stac import (
    STACClient,  # STAC API client.
    STACCollection,  # Collection record.
    STACItem,  # Item record.
    filter_items,  # Offline search.
    load_from_stac,  # Read an asset.
    read_stac_item,  # Read an item file.
    search_stac,  # STAC API search.
    walk_catalog,  # Items of a static catalogue.
)  # End of the STAC imports.

# Zarr.
from unbihexium.io.zarr_io import (
    read_raster_zarr,  # Read a georeferenced raster.
    read_zarr,  # Read an array.
    write_raster_zarr,  # Write a georeferenced raster.
    write_zarr,  # Write an array.
    zarr_info,  # Array description.
)  # End of the Zarr imports.

# Public names of the package.
__all__ = [
    "STACClient",  # STAC API client.
    "STACCollection",  # Collection record.
    "STACItem",  # Item record.
    "build_overviews",  # Add overviews.
    "features_to_geojson",  # FeatureCollection from features.
    "filter_items",  # Offline STAC search.
    "geojson_bounds",  # Bounding box of a document.
    "geojson_crs",  # CRS of a document.
    "geojson_problems",  # Structural problems of a document.
    "geometry_to_feature",  # Feature from a geometry.
    "geoparquet_bounds",  # Bounding box of a file.
    "geoparquet_metadata",  # The geo metadata.
    "geotiff_info",  # Metadata without pixels.
    "is_cog",  # COG layout test.
    "load_from_stac",  # Read an asset.
    "read_cog",  # Read a remote COG.
    "read_geojson",  # Read GeoJSON.
    "read_geoparquet",  # Read GeoParquet.
    "read_geotiff",  # Read GeoTIFF.
    "read_raster",  # Read a GeoTIFF into a Raster.
    "read_raster_zarr",  # Read a Zarr raster.
    "read_stac_item",  # Read a STAC item file.
    "read_zarr",  # Read Zarr.
    "reproject_geojson",  # Change the CRS of GeoJSON.
    "rewind",  # Right-hand rule.
    "search_stac",  # STAC API search.
    "validate_geojson",  # GeoJSON validation.
    "walk_catalog",  # Items of a static catalogue.
    "write_cog",  # Write a COG.
    "write_geojson",  # Write GeoJSON.
    "write_geoparquet",  # Write GeoParquet.
    "write_geotiff",  # Write GeoTIFF.
    "write_raster",  # Write a Raster to GeoTIFF.
    "write_raster_zarr",  # Write a Zarr raster.
    "write_zarr",  # Write Zarr.
    "zarr_info",  # Zarr array description.
]  # End of the export list.


# =============================================================================
# End of module src/unbihexium/io/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
