# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/io/geotiff.py
# Title       : GeoTIFF and Cloud Optimized GeoTIFF input and output
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and rasterio (GDAL 3.1
#               or newer for the COG driver)
# =============================================================================
#
# Abstract
# --------
#   read_geotiff     read bands of a file or URL, optionally a pixel window,
#                    a map-coordinate box or an overview level; returns the
#                    array and JSON-friendly metadata whose transform
#                    describes the window that was read
#   write_geotiff    write an array with CRS, transform and no-data value;
#                    tiled, compressed with the matching predictor, with
#                    internal overviews or in the Cloud Optimized GeoTIFF
#                    layout
#   write_cog        write_geotiff with cog=True
#   read_cog         read_geotiff for remote COGs (kept for compatibility)
#   geotiff_info     metadata without reading pixels
#   is_cog           whether a file has the COG layout
#   build_overviews  add internal overviews to an existing file
#   read_raster,     conversion to and from unbihexium.core.raster.Raster
#   write_raster
#
# Transforms are affine coefficients (a, b, c, d, e, f) in the order of the
# affine package and of Raster.metadata.transform:
#
#   x = a * col + b * row + c,   y = d * col + e * row + f
#
# which differs from the GDAL geotransform (c, a, b, f, d, e).
#
# Method
# ------
# Horizontal differencing predictors make neighbouring values similar before
# DEFLATE, LZW or ZSTD compression: predictor 2 for integers and predictor 3
# (floating point) for floats. Overview factors double until the smallest
# overview fits in one block. A COG stores tiles and overviews so that a
# client can fetch any window at any resolution with HTTP range requests.
#
# References
# ----------
# Ritter, N. and Ruth, M. (2000). GeoTIFF Format Specification, revision
# 1.0. Open Geospatial Consortium; OGC GeoTIFF Standard 19-008r4 (2019).
# Open Geospatial Consortium (2023). OGC Cloud Optimized GeoTIFF Standard,
# version 1.0, OGC 21-026.
# Adobe Developers Association (1992). TIFF Revision 6.0, section 14
# (Differencing Predictor); Adobe Photoshop TIFF Technical Note 3 (2005) for
# the floating point predictor.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Pixel window: (row_start, row_stop, col_start, col_stop).
PixelWindow = tuple[int, int, int, int]

# Compression methods whose output benefits from a predictor.
_PREDICTED = ("deflate", "lzw", "zstd", "lzma")


# rasterio, imported lazily with a clear error.
def _rasterio() -> Any:
    # Import on first use.
    try:
        # The GDAL bindings.
        import rasterio
    # Missing installation.
    except ImportError as exc:
        # Explain the requirement.
        raise ImportError("rasterio is required for GeoTIFF support") from exc
    # Return the module.
    return rasterio


# Affine object from coefficients, an Affine, or None (pixel coordinates).
def _to_affine(transform: Any, height: int) -> Any:
    # Affine class of rasterio.
    from affine import Affine

    # Default: pixel coordinates with the origin at the upper-left corner.
    if transform is None:
        # Row numbers grow downwards; y = height - row.
        return Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(height))
    # Affine objects are used as they are.
    if isinstance(transform, Affine):
        # Return it.
        return transform
    # Coefficient sequences (6 or 9 values).
    values = [float(v) for v in transform]
    # Only 6 or 9 coefficients are meaningful.
    if len(values) not in (6, 9):
        # Explain the expected layout.
        raise ValueError(f"transform needs 6 affine coefficients, got {len(values)}")
    # The first six coefficients.
    return Affine(*values[:6])


# JSON-friendly metadata of an open dataset.
def _metadata(src: Any, window: Any = None, bands: list[int] | None = None) -> dict[str, Any]:
    # Transform of the window, or of the whole dataset.
    transform = src.window_transform(window) if window is not None else src.transform
    # Size of the window.
    height = int(window.height) if window is not None else src.height
    # Width of the window.
    width = int(window.width) if window is not None else src.width
    # Bands described.
    indexes = bands or list(range(1, src.count + 1))
    # Structure tags reported by GDAL.
    structure = src.tags(ns="IMAGE_STRUCTURE")
    # Metadata dictionary.
    return {
        "crs": src.crs.to_string() if src.crs else None,  # Coordinate system.
        "transform": tuple(transform)[:6],  # Affine coefficients of the window.
        "bounds": tuple(grid_bounds(transform, width, height)),  # Box of the window.
        "width": width,  # Columns read.
        "height": height,  # Rows read.
        "count": len(indexes),  # Bands read.
        "full_width": src.width,  # Columns of the file.
        "full_height": src.height,  # Rows of the file.
        "band_count": src.count,  # Bands of the file.
        "dtype": str(src.dtypes[0]),  # Data type on disk.
        "nodata": src.nodata,  # No-data value.
        "descriptions": [src.descriptions[i - 1] for i in indexes],  # Band names.
        "overviews": src.overviews(1) if src.count else [],  # Overview factors.
        "compression": src.compression.value if src.compression else None,  # Codec.
        "tiled": bool(src.profile.get("tiled", False)),  # Tiled layout.
        "block_shape": tuple(src.block_shapes[0]) if src.count else None,  # Block size.
        "is_cog": structure.get("LAYOUT", "").upper() == "COG",  # COG layout.
        "driver": src.driver,  # GDAL driver.
    }  # End of the metadata.


# Box (left, bottom, right, top) of a grid with an affine transform.
def grid_bounds(transform: Any, width: int, height: int) -> tuple[float, ...]:
    # Affine coefficients.
    a, b, c, d, e, f = tuple(transform)[:6]
    # Corners of the grid in map coordinates.
    xs, ys = [], []
    # Visit the four corners.
    for col, row in ((0, 0), (width, 0), (0, height), (width, height)):
        # x = a * col + b * row + c.
        xs.append(a * col + b * row + c)
        # y = d * col + e * row + f.
        ys.append(d * col + e * row + f)
    # Box of the corners (works for rotated and north-down grids).
    return (min(xs), min(ys), max(xs), max(ys))


# Read bands of a GeoTIFF file or URL.
def read_geotiff(
    path: str | Path,  # File path or URL (http, s3, ... through GDAL).
    bands: list[int] | None = None,  # Band numbers from 1; None reads all.
    window: PixelWindow | None = None,  # Pixel window (row_start, row_stop, col_start, col_stop).
    bounds: tuple[float, float, float, float] | None = None,  # Box in the raster CRS.
    overview_level: int | None = None,  # 0 is the first overview; None the full resolution.
    masked: bool = False,  # Replace no-data values by NaN (float output only).
    dtype: str | None = "float32",  # Output type; None keeps the type on disk.
) -> tuple[NDArray[Any], dict[str, Any]]:  # (bands, rows, cols) array and metadata.
    # GDAL bindings.
    rasterio = _rasterio()
    # Window helpers.
    from rasterio.windows import Window, from_bounds

    # A pixel window and a box are exclusive.
    if window is not None and bounds is not None:
        # Explain the conflict.
        raise ValueError("give either window or bounds, not both")
    # Options of the dataset.
    options = {} if overview_level is None else {"overview_level": int(overview_level)}
    # Open the dataset.
    with rasterio.open(path, **options) as src:
        # Band numbers.
        indexes = bands or list(range(1, src.count + 1))
        # Reject unknown bands.
        if min(indexes) < 1 or max(indexes) > src.count:
            # Explain the valid range.
            raise ValueError(f"bands must be in [1, {src.count}], got {indexes}")
        # Window of the read, None for the whole dataset.
        win = None
        # Pixel windows.
        if window is not None:
            # Rows and columns.
            r0, r1, c0, c1 = (int(v) for v in window)
            # The window must lie inside the dataset.
            if not (0 <= r0 < r1 <= src.height and 0 <= c0 < c1 <= src.width):
                # Explain the valid range.
                raise ValueError(f"window {window} outside the {src.height}x{src.width} grid")
            # Window object.
            win = Window.from_slices((r0, r1), (c0, c1))
        # Map-coordinate boxes.
        if bounds is not None:
            # Fractional window of the box.
            frac = from_bounds(*bounds, transform=src.transform)
            # First row and column touched by the box (rounded against float noise).
            r0 = max(int(np.floor(round(frac.row_off, 6))), 0)
            # First column.
            c0 = max(int(np.floor(round(frac.col_off, 6))), 0)
            # Row after the last one touched, clipped to the grid.
            r1 = min(int(np.ceil(round(frac.row_off + frac.height, 6))), src.height)
            # Column after the last one touched, clipped to the grid.
            c1 = min(int(np.ceil(round(frac.col_off + frac.width, 6))), src.width)
            # The box must overlap the grid.
            if r0 >= r1 or c0 >= c1:
                # Explain the problem.
                raise ValueError(f"bounds {bounds} do not overlap the raster")
            # Whole pixels that touch the box.
            win = Window.from_slices((r0, r1), (c0, c1))
        # Read the pixels.
        data = src.read(indexes, window=win)
        # Metadata of the read.
        metadata = _metadata(src, win, indexes)
    # Output type.
    out = data if dtype is None else data.astype(dtype, copy=False)
    # No-data values become NaN for float outputs.
    if masked and metadata["nodata"] is not None and np.issubdtype(out.dtype, np.floating):
        # Pixels equal to the no-data value (NaN no-data is already NaN).
        out = np.where(data == metadata["nodata"], np.nan, out).astype(out.dtype)
    # Return the array and the metadata.
    return out, metadata


# Accept (data, path) and the (path, data) order of earlier releases.
def _data_and_path(first: Any, second: Any) -> tuple[Any, Path]:
    # A path in the first position belongs to the earlier order.
    if isinstance(first, (str, Path)) and not isinstance(second, (str, Path)):
        # Swap the arguments.
        return second, Path(first)
    # The documented order.
    return first, Path(second)


# Overview factors 2, 4, 8, ... until the overview fits in one block.
def overview_factors(width: int, height: int, blocksize: int = 256) -> list[int]:
    # Factors found.
    factors = []
    # First factor.
    factor = 2
    # Halve until the largest side fits in one block.
    while max(width, height) / (factor // 2) > blocksize:
        # Keep the factor.
        factors.append(factor)
        # Next factor.
        factor *= 2
    # Return the factors.
    return factors


# Write an array to a GeoTIFF file; returns the path.
def write_geotiff(
    data: NDArray[Any],  # (bands, rows, cols) or (rows, cols) array.
    path: str | Path,  # Output file.
    crs: Any = None,  # CRS as "EPSG:32632", WKT or a rasterio CRS.
    transform: Any = None,  # Affine or 6 coefficients (a, b, c, d, e, f).
    nodata: float | None = None,  # No-data value.
    compress: str = "deflate",  # deflate, lzw, zstd, lzma, packbits or none.
    tiled: bool = True,  # Tiled layout for rasters larger than a block.
    blocksize: int = 256,  # Tile side, a multiple of 16.
    predictor: int | None = None,  # 1, 2 or 3; None chooses from the data type.
    overviews: list[int] | str | None = None,  # Factors, "auto" or None.
    resampling: str = "average",  # Resampling of the overviews.
    descriptions: list[str] | None = None,  # Band names.
    tags: dict[str, str] | None = None,  # Dataset metadata tags.
    cog: bool = False,  # Write the Cloud Optimized GeoTIFF layout.
) -> Path:  # Written file.
    # GDAL bindings.
    rasterio = _rasterio()
    # Support the argument order of earlier releases.
    data, path = _data_and_path(data, path)
    # Array with a band axis.
    array = np.asarray(data)
    # Single bands get a band axis.
    if array.ndim == 2:
        # Add the axis.
        array = array[np.newaxis]
    # Only 3-D arrays are rasters.
    if array.ndim != 3:
        # Explain the expected layout.
        raise ValueError(f"data must be (bands, rows, cols), got shape {array.shape}")
    # GDAL has no boolean type.
    if array.dtype == np.bool_:
        # Store as bytes.
        array = array.astype(np.uint8)
    # Block sizes must be multiples of 16.
    if blocksize % 16 or blocksize < 16:
        # Explain the constraint.
        raise ValueError(f"blocksize must be a positive multiple of 16, got {blocksize}")
    # Size of the array.
    count, height, width = array.shape
    # Normalised compression name.
    method = (compress or "none").lower()
    # Predictor chosen from the type unless given.
    if predictor is None:
        # Floating point predictor for floats, differencing for integers.
        predictor = 3 if np.issubdtype(array.dtype, np.floating) else 2
    # Predictors only help the dictionary coders.
    predictor = predictor if method in _PREDICTED else 1
    # Band descriptions must match the bands.
    if descriptions is not None and len(descriptions) != count:
        # Explain the mismatch.
        raise ValueError(f"{len(descriptions)} descriptions for {count} bands")
    # Destination.
    path = Path(path)
    # Parent directory.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Dataset profile.
    profile: dict[str, Any] = {
        "driver": "GTiff",  # GeoTIFF.
        "dtype": array.dtype.name,  # Data type.
        "width": width,  # Columns.
        "height": height,  # Rows.
        "count": count,  # Bands.
        "crs": crs,  # Coordinate system.
        "transform": _to_affine(transform, height),  # Affine transform.
        "nodata": nodata,  # No-data value.
    }  # End of the profile.
    # Compression options.
    if method != "none":
        # Codec and predictor.
        profile.update(compress=method, predictor=predictor)
    # Tiles only for rasters larger than one block.
    if tiled and max(width, height) > blocksize:
        # Tiled layout.
        profile.update(tiled=True, blockxsize=blocksize, blockysize=blocksize)
    # Write through the COG driver.
    if cog:
        # COG conversion from an in-memory dataset.
        _write_cog(array, profile, path, blocksize, resampling, descriptions, tags)
        # Return the path.
        return path
    # Write the GeoTIFF.
    with rasterio.open(path, "w", **profile) as dst:
        # Pixels.
        dst.write(array)
        # Band names.
        _describe(dst, descriptions, tags)
        # Overview factors.
        factors = overview_factors(width, height, blocksize) if overviews == "auto" else overviews
        # Internal overviews.
        if factors:
            # Resampling enumeration.
            from rasterio.enums import Resampling

            # Build them.
            dst.build_overviews(list(factors), Resampling[resampling])
    # Return the path.
    return path


# Set band descriptions and tags of an open dataset.
def _describe(dst: Any, descriptions: list[str] | None, tags: dict[str, str] | None) -> None:
    # Band names.
    for i, text in enumerate(descriptions or [], start=1):
        # Description of band i.
        dst.set_band_description(i, text)
    # Dataset tags.
    if tags:
        # Store them.
        dst.update_tags(**tags)


# Write a COG through an in-memory GeoTIFF and the GDAL COG driver.
def _write_cog(
    array: NDArray[Any],  # (bands, rows, cols) array.
    profile: dict[str, Any],  # GeoTIFF profile.
    path: Path,  # Output file.
    blocksize: int,  # Tile side.
    resampling: str,  # Overview resampling.
    descriptions: list[str] | None,  # Band names.
    tags: dict[str, str] | None,  # Dataset tags.
) -> None:  # The function writes the file.
    # In-memory files.
    from rasterio.io import MemoryFile

    # Dataset copy.
    from rasterio.shutil import copy

    # Layout options belong to the COG driver, not to the temporary file.
    base = {k: v for k, v in profile.items() if k not in ("tiled", "blockxsize", "blockysize")}
    # Temporary dataset.
    with MemoryFile() as memory:
        # Write the pixels.
        with memory.open(**base) as tmp:
            # Pixels.
            tmp.write(array)
            # Names and tags.
            _describe(tmp, descriptions, tags)
        # Convert with the COG driver.
        with memory.open() as src:
            # COG options.
            options: dict[str, Any] = {
                "driver": "COG",  # Cloud Optimized GeoTIFF.
                "blocksize": blocksize,  # Tile side.
                "overview_resampling": resampling,  # Overview resampling.
                "compress": profile.get("compress", "none"),  # Codec.
            }  # End of the options.
            # Predictor with the codec.
            if "predictor" in profile:
                # PREDICTOR=YES chooses 2 or 3 from the type.
                options["predictor"] = "YES" if profile["predictor"] != 1 else "NO"
            # Write the file.
            copy(src, str(path), **options)


# Write a Cloud Optimized GeoTIFF.
def write_cog(data: NDArray[Any], path: str | Path, **kwargs: Any) -> Path:
    # write_geotiff with the COG layout.
    return write_geotiff(data, path, cog=True, **kwargs)


# Read a (remote) Cloud Optimized GeoTIFF.
def read_cog(
    url: str | Path,  # URL or path.
    bands: list[int] | None = None,  # Band numbers from 1.
    window: PixelWindow | None = None,  # Pixel window.
    **kwargs: Any,  # Other options of read_geotiff.
) -> tuple[NDArray[Any], dict[str, Any]]:  # Array and metadata.
    # Same reader; GDAL fetches only the needed tiles.
    return read_geotiff(url, bands=bands, window=window, **kwargs)


# Metadata of a file without reading pixels.
def geotiff_info(path: str | Path) -> dict[str, Any]:
    # Open the dataset.
    with _rasterio().open(path) as src:
        # Metadata of the whole dataset.
        return _metadata(src)


# Whether a file has the Cloud Optimized GeoTIFF layout.
def is_cog(path: str | Path) -> bool:
    # Layout reported by GDAL.
    return bool(geotiff_info(path)["is_cog"])


# Add internal overviews to an existing GeoTIFF; returns the factors.
def build_overviews(
    path: str | Path,  # GeoTIFF file.
    factors: list[int] | None = None,  # Factors; None chooses them from the size.
    resampling: str = "average",  # Resampling method.
    blocksize: int = 256,  # Block size used to choose the factors.
) -> list[int]:  # Factors built.
    # Resampling enumeration.
    from rasterio.enums import Resampling

    # Open for update.
    with _rasterio().open(path, "r+") as dst:
        # Factors from the size unless given.
        chosen = factors or overview_factors(dst.width, dst.height, blocksize)
        # Build them.
        if chosen:
            # Overviews of every band.
            dst.build_overviews(chosen, Resampling[resampling])
    # Return the factors.
    return list(chosen)


# Read a GeoTIFF into a Raster.
def read_raster(path: str | Path, **kwargs: Any) -> Any:
    # Raster container, imported lazily.
    from unbihexium.core.raster import Raster

    # Pixels and metadata; the type on disk is kept unless requested.
    data, meta = read_geotiff(path, **{"dtype": None, **kwargs})
    # Raster with the georeferencing of the read.
    return Raster.from_array(
        data,  # Pixels.
        crs=meta["crs"] or "EPSG:4326",  # Coordinate system.
        transform=meta["transform"],  # Affine coefficients.
        nodata=meta["nodata"],  # No-data value.
    )  # End of the raster.


# Write a Raster to a GeoTIFF.
def write_raster(raster: Any, path: str | Path, **kwargs: Any) -> Path:
    # Metadata of the raster.
    meta = raster.metadata
    # Pixels with the raster georeferencing.
    return write_geotiff(
        np.asarray(raster.data),  # Pixels.
        path,  # Output file.
        crs=meta.crs if meta else None,  # Coordinate system.
        transform=tuple(meta.transform)[:6] if meta else None,  # Affine coefficients.
        nodata=meta.nodata if meta else None,  # No-data value.
        **kwargs,  # Layout options.
    )  # End of the write.


# =============================================================================
# End of module src/unbihexium/io/geotiff.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
