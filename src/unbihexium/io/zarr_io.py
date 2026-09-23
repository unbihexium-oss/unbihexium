# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/io/zarr_io.py
# Title       : Zarr input and output of arrays and georeferenced rasters
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and zarr (zarr 3 on
#               Python 3.11 and newer, zarr 2.18 on Python 3.10)
# =============================================================================
#
# Abstract
# --------
# Zarr stores N-dimensional arrays as a grid of independently compressed
# chunks, which suits data cubes and cloud object storage:
#
#   write_zarr         write an array, optionally as a named member of a
#                      group, with chunks, compression, attributes and (in
#                      Zarr v3) dimension names
#   read_zarr          read an array or a member of a group, optionally only
#                      a selection of slices
#   zarr_info          shape, chunks, data type, format, codecs and
#                      attributes without reading the chunks
#   list_arrays        names of the arrays of a group
#   write_raster_zarr  write pixels with CRS, affine transform and no-data
#   read_raster_zarr   value as attributes; reading returns a Raster
#
# Format selection
# ----------------
# With zarr 3 the Zarr v3 format is the default (zarr_format=3); v2 can be
# requested for readers that only support v2. zarr 2 writes v2 only. Codecs:
# "zstd" (default), "zlib" (gzip in v3), "blosc" (LZ4 with byte shuffle)
# or None.
#
# References
# ----------
# Zarr Developers (2023). Zarr storage specification version 3.0.
# https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html
# Miles, A. et al. (2024). zarr-developers/zarr-python. Zenodo.
# doi:10.5281/zenodo.3773449
# Collet, Y. and Kucherawy, M. (2021). Zstandard Compression and the
# application/zstd Media Type. IETF RFC 8878. doi:10.17487/RFC8878
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

# Compression codecs accepted by write_zarr.
COMPRESSORS = ("zstd", "zlib", "blosc", None)

# Dimension names of raster arrays.
RASTER_DIMENSIONS = ("band", "y", "x")


# zarr, imported lazily with a clear error.
def _zarr() -> Any:
    # Import on first use.
    try:
        # Chunked arrays.
        import zarr
    # Missing installation.
    except ImportError as exc:
        # Explain the requirement.
        raise ImportError("zarr is required: pip install unbihexium[zarr]") from exc
    # Return the module.
    return zarr


# Major version of the installed zarr package.
def zarr_major_version() -> int:
    # First component of the version string.
    return int(_zarr().__version__.split(".")[0])


# Codec object for a codec name and storage format.
def _codec(name: str | None, zarr_format: int, level: int) -> Any:
    # No compression.
    if name is None:
        # No codec.
        return None
    # Unknown names.
    if name not in COMPRESSORS:
        # Explain the accepted names.
        raise ValueError(f"compressor must be one of zstd, zlib, blosc or None, got {name!r}")
    # Zarr v3 codecs of zarr 3.
    if zarr_format == 3:
        # v3 codec classes.
        from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec

        # Zstandard.
        if name == "zstd":
            # Zstd codec.
            return ZstdCodec(level=level)
        # DEFLATE in a gzip container.
        if name == "zlib":
            # Gzip codec.
            return GzipCodec(level=level)
        # Blosc with LZ4 and byte shuffle.
        return BloscCodec(cname="lz4", clevel=level, shuffle="shuffle")
    # numcodecs for the v2 format.
    import numcodecs

    # Zstandard.
    if name == "zstd":
        # Zstd codec.
        return numcodecs.Zstd(level=level)
    # zlib.
    if name == "zlib":
        # zlib codec.
        return numcodecs.Zlib(level=level)
    # Blosc with LZ4 and byte shuffle.
    return numcodecs.Blosc(cname="lz4", clevel=level, shuffle=numcodecs.Blosc.SHUFFLE)


# Accept (data, path) and the (path, data) order of earlier releases.
def _data_and_path(first: Any, second: Any) -> tuple[Any, Path]:
    # A path in the first position belongs to the earlier order.
    if isinstance(first, (str, Path)) and not isinstance(second, (str, Path)):
        # Swap the arguments.
        return second, Path(first)
    # The documented order.
    return first, Path(second)


# Write an array to a Zarr store; returns the path.
def write_zarr(
    data: NDArray[Any],  # Array to store.
    path: str | Path,  # Store directory.
    chunks: tuple[int, ...] | None = None,  # Chunk shape; None uses up to 256 per axis.
    attrs: dict[str, Any] | None = None,  # JSON-serialisable attributes.
    compressor: str | None = "zstd",  # zstd, zlib, blosc or None.
    level: int = 5,  # Compression level.
    name: str | None = None,  # Member name inside a group; None stores a bare array.
    dimension_names: tuple[str, ...] | None = None,  # Axis names (Zarr v3 only).
    zarr_format: int | None = None,  # 2 or 3; None chooses the newest available.
) -> Path:  # Store directory.
    # Chunked array library.
    zarr = _zarr()
    # Support the argument order of earlier releases.
    data, path = _data_and_path(data, path)
    # Array to store.
    array = np.asarray(data)
    # Storage format.
    fmt = zarr_format or (3 if zarr_major_version() >= 3 else 2)
    # zarr 2 cannot write v3.
    if fmt == 3 and zarr_major_version() < 3:
        # Explain the requirement.
        raise ValueError("Zarr v3 needs zarr 3 or newer")
    # Default chunks.
    chunks = tuple(chunks) if chunks else tuple(min(256, max(n, 1)) for n in array.shape)
    # Chunks need one entry per axis.
    if len(chunks) != array.ndim:
        # Explain the mismatch.
        raise ValueError(f"chunks {chunks} do not match the {array.ndim} axes of the data")
    # Dimension names need one entry per axis.
    if dimension_names is not None and len(dimension_names) != array.ndim:
        # Explain the mismatch.
        raise ValueError(f"{len(dimension_names)} dimension names for {array.ndim} axes")
    # Parent directory.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Codec of the chunks.
    codec = _codec(compressor, fmt, level)
    # zarr 3 API.
    if zarr_major_version() >= 3:
        # Options of create_array.
        options: dict[str, Any] = {
            "store": str(path),  # Store directory.
            "name": name,  # Member name.
            "shape": array.shape,  # Shape.
            "chunks": chunks,  # Chunks.
            "dtype": array.dtype,  # Data type.
            "compressors": [codec] if codec is not None else None,  # Codecs.
            "zarr_format": fmt,  # Format.
            "attributes": dict(attrs or {}),  # Attributes.
            "overwrite": True,  # Replace existing arrays.
        }  # End of the options.
        # Dimension names are a v3 feature.
        if fmt == 3 and dimension_names is not None:
            # Store them.
            options["dimension_names"] = tuple(dimension_names)
        # Create the array.
        target = zarr.create_array(**options)
        # Write the values.
        target[...] = array
        # Return the store.
        return Path(path)
    # zarr 2: bare arrays or group members.
    root = zarr.open_group(str(path), mode="a") if name else None
    # Array options.
    kwargs = {"shape": array.shape, "chunks": chunks, "dtype": array.dtype, "compressor": codec}
    # Create the array.
    target = (
        root.create_dataset(name, overwrite=True, **kwargs)  # Group member.
        if root is not None  # Groups need a name.
        else zarr.open(str(path), mode="w", **kwargs)  # Bare array.
    )  # End of the creation.
    # Write the values.
    target[...] = array
    # Attributes.
    target.attrs.update(dict(attrs or {}))
    # Return the store.
    return Path(path)


# Open an array of a store: a bare array, a named member or the first member.
def _open_array(path: str | Path, variable: str | None) -> Any:
    # Open read-only.
    node = _zarr().open(str(path), mode="r")
    # Bare arrays have a shape.
    if hasattr(node, "shape"):
        # Variables of bare arrays are errors.
        if variable is not None:
            # Explain the problem.
            raise ValueError(f"{path} holds a single array, not a group with {variable!r}")
        # Return the array.
        return node
    # Named members.
    if variable is not None:
        # Unknown names.
        if variable not in list(node.array_keys()):
            # List the members.
            raise KeyError(f"{variable!r} not in {path}; arrays: {sorted(node.array_keys())}")
        # Return the member.
        return node[variable]
    # Members sorted by name.
    names = sorted(node.array_keys())
    # Empty groups.
    if not names:
        # Explain the problem.
        raise ValueError(f"no arrays found in the Zarr store {path}")
    # First member.
    return node[names[0]]


# Read an array (or a selection of it) and its attributes.
def read_zarr(
    path: str | Path,  # Store directory.
    variable: str | None = None,  # Member name of a group; None reads the first.
    selection: tuple[Any, ...] | None = None,  # Slices per axis; None reads everything.
    dtype: Any = None,  # Output type; None keeps the stored type.
) -> tuple[NDArray[Any], dict[str, Any]]:  # Array and attributes.
    # Array of the store.
    array = _open_array(path, variable)
    # Only the selected chunks are decompressed.
    data = np.asarray(array[selection] if selection is not None else array[...])
    # Output type.
    if dtype is not None:
        # Convert.
        data = data.astype(dtype)
    # Return the values and the attributes.
    return data, dict(array.attrs)


# Description of an array without reading its chunks.
def zarr_info(path: str | Path, variable: str | None = None) -> dict[str, Any]:
    # Array of the store.
    array = _open_array(path, variable)
    # Storage format (zarr 3 exposes metadata, zarr 2 is always v2).
    fmt = getattr(getattr(array, "metadata", None), "zarr_format", 2)
    # Codecs (zarr 3) or the compressor (zarr 2).
    codecs = getattr(array, "compressors", None) or (getattr(array, "compressor", None),)
    # Dimension names of v3 arrays.
    names = getattr(getattr(array, "metadata", None), "dimension_names", None)
    # Description.
    return {
        "shape": tuple(int(n) for n in array.shape),  # Shape.
        "chunks": tuple(int(n) for n in array.chunks),  # Chunk shape.
        "dtype": str(array.dtype),  # Data type.
        "zarr_format": int(fmt),  # Format.
        "compressors": [repr(c) for c in codecs if c is not None],  # Codecs.
        "dimension_names": list(names) if names else None,  # Axis names.
        "attrs": dict(array.attrs),  # Attributes.
    }  # End of the description.


# Names of the arrays of a group.
def list_arrays(path: str | Path) -> list[str]:
    # Open read-only.
    node = _zarr().open(str(path), mode="r")
    # Bare arrays have no members.
    if hasattr(node, "shape"):
        # No members.
        return []
    # Sorted member names.
    return sorted(node.array_keys())


# Write pixels with georeferencing attributes.
def write_raster_zarr(
    raster: Any,  # Raster, or a (bands, rows, cols) array.
    path: str | Path,  # Store directory.
    crs: str | None = None,  # CRS for arrays; rasters carry their own.
    transform: tuple[float, ...] | None = None,  # Affine coefficients for arrays.
    nodata: float | None = None,  # No-data value for arrays.
    **kwargs: Any,  # Options of write_zarr.
) -> Path:  # Store directory.
    # Metadata of Raster objects.
    meta = getattr(raster, "metadata", None)
    # Pixels.
    data = np.asarray(raster.data if meta is not None else raster)
    # Single bands get a band axis.
    if data.ndim == 2:
        # Add the axis.
        data = data[np.newaxis]
    # Georeferencing of the raster, used where no value is given.
    own = (meta.crs, tuple(meta.transform)[:6], meta.nodata) if meta else (None, (), None)
    # Unpack the three values.
    own_crs, own_transform, own_nodata = own
    # Attributes with the georeferencing.
    attrs = {
        "crs": crs or own_crs,  # Coordinate system.
        "transform": [float(v) for v in (transform or own_transform)],  # Affine coefficients.
        "nodata": nodata if nodata is not None else own_nodata,  # No-data value.
    }  # End of the attributes.
    # JSON cannot store NaN; record it as a string.
    if isinstance(attrs["nodata"], float) and np.isnan(attrs["nodata"]):
        # NaN marker.
        attrs["nodata"] = "nan"
    # Other attributes given by the caller.
    attrs.update(kwargs.pop("attrs", None) or {})
    # Write with band, y and x axes.
    return write_zarr(data, path, attrs=attrs, dimension_names=RASTER_DIMENSIONS, **kwargs)


# Read a raster written by write_raster_zarr.
def read_raster_zarr(path: str | Path, variable: str | None = None) -> Any:
    # Raster container, imported lazily.
    from unbihexium.core.raster import Raster

    # Pixels and attributes.
    data, attrs = read_zarr(path, variable)
    # No-data value, with the NaN marker decoded.
    nodata = attrs.get("nodata")
    # Decode the marker.
    nodata = float("nan") if nodata == "nan" else nodata
    # Affine coefficients, None for pixel coordinates.
    transform = tuple(attrs["transform"]) if attrs.get("transform") else None
    # Raster with the stored georeferencing.
    return Raster.from_array(
        data,  # Pixels.
        crs=attrs.get("crs") or "EPSG:4326",  # Coordinate system.
        transform=transform,  # Affine transform.
        nodata=nodata,  # No-data value.
    )  # End of the raster.


# =============================================================================
# End of module src/unbihexium/io/zarr_io.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
