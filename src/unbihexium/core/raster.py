# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/core/raster.py
# Title       : Georeferenced raster container
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy; file input and output,
#               resampling, reprojection and clipping require rasterio
# =============================================================================
#
# Abstract
# --------
# Raster holds a (bands, height, width) array with its coordinate reference
# system (CRS), affine transform and no-data value (RasterMetadata):
#
#   input/output   from_array, from_file (optionally lazy, windowed or band
#                  subset), load, to_file (tiled, compressed GeoTIFF), to_cog
#                  (Cloud Optimized GeoTIFF with overviews)
#   geometry       bounds, resolution, xy (pixel to map), rowcol (map to
#                  pixel), sample (values at points)
#   windows        read_window, window, tiles, crop (by bounds), clip (by
#                  geometries, with rasterio.features.geometry_mask)
#   resampling     resample (by scale or resolution), reproject and match (onto
#                  the grid of another raster), with the
#                  GDAL warper through rasterio (nearest, bilinear, cubic,
#                  average, mode, min, max, median, ...)
#   values         valid_mask, masked, statistics, histogram, band_math,
#                  select_bands, stack, astype, apply, set_nodata, mask
#
# Affine transform: the six coefficients (a, b, c, d, e, f) map the column
# and row (col, row) of a pixel corner to map coordinates:
#   x = a * col + b * row + c,   y = d * col + e * row + f
# (the GDAL order is c, a, b, f, d, e). Pixel centres are at col + 0.5 and
# row + 0.5. A north-up raster has b = d = 0 and e < 0.
#
# Validity: a pixel of a band is valid when it is finite and differs from the
# no-data value. Statistics, band math and blending ignore invalid pixels.
#
# Band math: expressions such as "(b4 - b3) / (b4 + b3)" are parsed with the
# Python ast module and evaluated on the bands b1..bN; only arithmetic,
# comparisons, boolean operators and a fixed set of NumPy functions (sqrt,
# log, log10, exp, abs, minimum, maximum, where, clip) are allowed, so no
# arbitrary code is executed.
#
# References
# ----------
#   GDAL/OGR contributors (2025). GDAL/OGR Geospatial Data Abstraction
#     software Library. Open Source Geospatial Foundation. https://gdal.org
#   Cloud Optimized GeoTIFF specification, Open Geospatial Consortium
#     (2023). OGC 21-026.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Parser of band math expressions.
import ast

# Tile iterators and callables.
from collections.abc import Callable, Iterable, Iterator, Sequence

# Metadata container and copies with changed fields.
from dataclasses import dataclass, field, replace

# Data types.
from enum import Enum

# File paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any, Literal

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Tile start positions.
from unbihexium.core.tile import tile_offsets

# Default coordinate reference system of rasters created from arrays.
DEFAULT_CRS = "EPSG:4326"

# Transform of rasters without georeferencing: one unit per pixel, north up.
IDENTITY_TRANSFORM = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

# Resampling methods accepted by resample and reproject.
ResamplingName = Literal[
    "nearest",
    "bilinear",
    "cubic",
    "cubic_spline",
    "lanczos",
    "average",
    "mode",
    "min",
    "max",
    "med",
]


# Data types that GDAL can store.
class RasterDtype(str, Enum):
    # Signed 8-bit integers.
    INT8 = "int8"
    # Unsigned 8-bit integers.
    UINT8 = "uint8"
    # Unsigned 16-bit integers.
    UINT16 = "uint16"
    # Signed 16-bit integers.
    INT16 = "int16"
    # Unsigned 32-bit integers.
    UINT32 = "uint32"
    # Signed 32-bit integers.
    INT32 = "int32"
    # Unsigned 64-bit integers.
    UINT64 = "uint64"
    # Signed 64-bit integers.
    INT64 = "int64"
    # Single precision floats.
    FLOAT32 = "float32"
    # Double precision floats.
    FLOAT64 = "float64"
    # Single precision complex numbers, as in SAR single look complex data.
    COMPLEX64 = "complex64"
    # Double precision complex numbers.
    COMPLEX128 = "complex128"


# Six affine coefficients as floats from a tuple, list or Affine object.
def normalize_transform(transform: Sequence[float] | None) -> tuple[float, ...]:
    # Missing transforms become the identity.
    if transform is None:
        # Default transform.
        return IDENTITY_TRANSFORM
    # Coefficients as floats.
    values = tuple(float(v) for v in tuple(transform)[:6])
    # Six coefficients are needed.
    if len(values) != 6:
        # Explain the problem.
        raise ValueError(f"transform needs 6 coefficients, got {len(values)}")
    # Return the coefficients.
    return values


# Bounds (min x, min y, max x, max y) of a width x height grid.
def transform_bounds(
    transform: Sequence[float],  # Affine coefficients.
    width: int,  # Columns.
    height: int,  # Rows.
) -> tuple[float, float, float, float]:  # Bounds.
    # Affine coefficients.
    a, b, c, d, e, f = normalize_transform(transform)
    # Corner x coordinates.
    xs = (c, c + a * width, c + b * height, c + a * width + b * height)
    # Corner y coordinates.
    ys = (f, f + d * width, f + e * height, f + d * width + e * height)
    # Extremes of the corners.
    return min(xs), min(ys), max(xs), max(ys)


# Metadata of a raster.
@dataclass(frozen=True)
class RasterMetadata:
    # Coordinate reference system, for example "EPSG:32635"; empty if unknown.
    crs: str
    # Affine coefficients (a, b, c, d, e, f).
    transform: tuple[float, ...]
    # Number of columns.
    width: int
    # Number of rows.
    height: int
    # Number of bands.
    count: int
    # Data type of the pixel values.
    dtype: RasterDtype = RasterDtype.FLOAT32
    # Value that marks missing pixels.
    nodata: float | None = None
    # Bounds (min x, min y, max x, max y); computed when not given.
    bounds: tuple[float, float, float, float] | None = None
    # Pixel size (x, y) in CRS units; computed for north-up rasters.
    resolution: tuple[float, float] | None = None
    # Free-form tags, such as band descriptions or acquisition metadata.
    tags: dict[str, str] = field(default_factory=dict)

    # Fill the derived fields.
    def __post_init__(self) -> None:
        # Normalise the transform (the dataclass is frozen).
        object.__setattr__(self, "transform", normalize_transform(self.transform))
        # Name of the dtype, given as an enum member, a string or a NumPy dtype.
        given = self.dtype
        # Enum members keep their value; strings and dtypes are normalised.
        name = given.value if isinstance(given, RasterDtype) else np.dtype(given).name
        # Store the enum member.
        object.__setattr__(self, "dtype", RasterDtype(name))
        # Sizes must be positive.
        if min(self.width, self.height, self.count) <= 0:
            # Explain the problem.
            raise ValueError(f"invalid raster size {self.count} x {self.height} x {self.width}")
        # Compute the bounds from the transform.
        if self.bounds is None:
            # Bounds of the grid.
            bounds = transform_bounds(self.transform, self.width, self.height)
            # Store them.
            object.__setattr__(self, "bounds", bounds)
        # Compute the resolution of rasters without rotation.
        if self.resolution is None and self.transform[1] == 0 and self.transform[3] == 0:
            # Absolute pixel sizes.
            object.__setattr__(self, "resolution", (abs(self.transform[0]), abs(self.transform[4])))

    # Plain dictionary for JSON output.
    def to_dict(self) -> dict[str, Any]:
        # One entry per field.
        return {
            "crs": self.crs,  # Coordinate system.
            "transform": list(self.transform),  # Affine coefficients.
            "width": self.width,  # Columns.
            "height": self.height,  # Rows.
            "count": self.count,  # Bands.
            "dtype": self.dtype.value,  # Data type.
            "nodata": self.nodata,  # No-data value.
            "bounds": list(self.bounds) if self.bounds else None,  # Bounds.
            "resolution": list(self.resolution) if self.resolution else None,  # Pixel size.
            "tags": dict(self.tags),  # Tags.
        }  # End of the dictionary.

    # Metadata from a dictionary written by to_dict.
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RasterMetadata:
        # Optional bounds.
        bounds = tuple(data["bounds"]) if data.get("bounds") else None
        # Optional resolution.
        resolution = tuple(data["resolution"]) if data.get("resolution") else None
        # Build the metadata.
        return cls(
            crs=data["crs"],  # Coordinate system.
            transform=tuple(data["transform"]),  # Affine coefficients.
            width=int(data["width"]),  # Columns.
            height=int(data["height"]),  # Rows.
            count=int(data["count"]),  # Bands.
            dtype=RasterDtype(data.get("dtype", "float32")),  # Data type.
            nodata=data.get("nodata"),  # No-data value.
            bounds=bounds,  # type: ignore[arg-type]  # Bounds.
            resolution=resolution,  # type: ignore[arg-type]  # Pixel size.
            tags=dict(data.get("tags", {})),  # Tags.
        )  # End of the metadata.


# NumPy functions allowed in band math expressions.
_MATH_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": np.sqrt,  # Square root.
    "log": np.log,  # Natural logarithm.
    "log10": np.log10,  # Decimal logarithm.
    "exp": np.exp,  # Exponential.
    "abs": np.abs,  # Absolute value.
    "minimum": np.minimum,  # Element-wise minimum.
    "maximum": np.maximum,  # Element-wise maximum.
    "where": np.where,  # Conditional selection.
    "clip": np.clip,  # Clipping to limits.
}  # End of the functions.

# Binary operators allowed in band math expressions.
_BINARY_OPERATORS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: np.add,  # Addition.
    ast.Sub: np.subtract,  # Subtraction.
    ast.Mult: np.multiply,  # Multiplication.
    ast.Div: np.true_divide,  # Division.
    ast.Pow: np.power,  # Power.
}  # End of the binary operators.

# Comparison operators allowed in band math expressions.
_COMPARISONS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Lt: np.less,  # Less than.
    ast.LtE: np.less_equal,  # Less than or equal.
    ast.Gt: np.greater,  # Greater than.
    ast.GtE: np.greater_equal,  # Greater than or equal.
    ast.Eq: np.equal,  # Equal.
    ast.NotEq: np.not_equal,  # Not equal.
}  # End of the comparisons.


# Evaluate one node of a band math expression.
def _evaluate(node: ast.AST, variables: dict[str, Any]) -> Any:
    # Numbers.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        # The number itself.
        return node.value
    # Band names.
    if isinstance(node, ast.Name):
        # Unknown names are an error.
        if node.id not in variables:
            # Explain the problem with the known names.
            raise ValueError(f"unknown name {node.id!r}; known: {', '.join(sorted(variables))}")
        # The band array.
        return variables[node.id]
    # Binary arithmetic.
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        # Operator on the evaluated operands.
        operator = _BINARY_OPERATORS[type(node.op)]
        # Apply it.
        return operator(_evaluate(node.left, variables), _evaluate(node.right, variables))
    # Unary plus and minus, and logical not.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
        # Operand.
        value = _evaluate(node.operand, variables)
        # Negation.
        if isinstance(node.op, ast.USub):
            # Negative operand.
            return np.negative(value)
        # Logical not.
        if isinstance(node.op, ast.Not):
            # Inverted boolean.
            return np.logical_not(value)
        # Unary plus.
        return value
    # Comparisons, possibly chained as in 0 < b1 < 1.
    if isinstance(node, ast.Compare) and all(type(o) in _COMPARISONS for o in node.ops):
        # Left operand.
        left = _evaluate(node.left, variables)
        # Conjunction of the pairwise comparisons.
        result: Any = True
        # Visit every comparison.
        for op, comparator in zip(node.ops, node.comparators):
            # Right operand.
            right = _evaluate(comparator, variables)
            # Combine with the earlier comparisons.
            result = np.logical_and(result, _COMPARISONS[type(op)](left, right))
            # The right operand is the next left operand.
            left = right
        # Boolean array.
        return result
    # Boolean and / or.
    if isinstance(node, ast.BoolOp):
        # NumPy function of the operator.
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        # Evaluated operands.
        values = [_evaluate(v, variables) for v in node.values]
        # Reduce the operands pairwise.
        result = values[0]
        # Combine the remaining operands.
        for value in values[1:]:
            # Pairwise combination.
            result = combine(result, value)
        # Boolean array.
        return result
    # Calls of allowed functions with positional arguments.
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        # Function by name.
        function = _MATH_FUNCTIONS.get(node.func.id)
        # Unknown functions are an error.
        if function is None:
            # Explain the problem with the allowed functions.
            raise ValueError(
                f"function {node.func.id!r} not allowed; use {sorted(_MATH_FUNCTIONS)}"
            )
        # Call with the evaluated arguments.
        return function(*(_evaluate(a, variables) for a in node.args))
    # Everything else is rejected.
    raise ValueError(f"unsupported syntax in band math: {ast.dump(node)[:60]}")


# Evaluate a band math expression with the given variables.
def evaluate_expression(expression: str, variables: dict[str, Any]) -> Any:
    # Parse as a single expression.
    try:
        # Syntax tree.
        tree = ast.parse(expression, mode="eval")
    # Report syntax errors as value errors.
    except SyntaxError as error:
        # Explain the problem.
        raise ValueError(f"invalid expression {expression!r}: {error.msg}") from error
    # Evaluate with floating point warnings silenced; NaN marks the results.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Value of the expression.
        return _evaluate(tree.body, variables)


# GDAL resampling enumeration value of a method name.
def _resampling(method: str) -> Any:
    # Imported lazily: rasterio is only needed for warping.
    from rasterio.enums import Resampling

    # Look the method up.
    try:
        # Enumeration member.
        return Resampling[method]
    # Report unknown names.
    except KeyError as error:
        # Explain the problem with the known names.
        names = ", ".join(m.name for m in Resampling)
        # Raise a value error.
        raise ValueError(f"unknown resampling method {method!r}; known: {names}") from error


# Georeferenced raster.
@dataclass
class Raster:
    # Pixel values of shape (bands, height, width); None until loaded.
    data: NDArray[Any] | None = None
    # CRS, transform, size, dtype and no-data value.
    metadata: RasterMetadata | None = None
    # File the raster was read from.
    source: str | Path | None = None
    # Whether the data is read from the source on first use.
    lazy: bool = False

    # Normalise the data to three dimensions and derive missing metadata.
    def __post_init__(self) -> None:
        # Nothing to check before loading.
        if self.data is None:
            # Lazy or empty raster.
            return
        # Array view.
        data = np.asarray(self.data)
        # Boolean masks are stored as bytes.
        if data.dtype == np.bool_:
            # 0 and 1.
            data = data.astype(np.uint8)
        # Single bands get a band axis.
        if data.ndim == 2:
            # Shape (1, height, width).
            data = data[np.newaxis]
        # Only two- and three-dimensional arrays are rasters.
        if data.ndim != 3:
            # Explain the problem.
            raise ValueError(f"raster data must be 2-D or 3-D, got shape {data.shape}")
        # Keep the normalised array.
        self.data = data
        # Band, row and column counts.
        count, height, width = data.shape
        # Metadata with the identity transform when none is given.
        if self.metadata is None:
            # Default georeferencing.
            dtype = RasterDtype(data.dtype.name)
            # Metadata of the array.
            self.metadata = RasterMetadata(
                DEFAULT_CRS,  # Default coordinate system.
                IDENTITY_TRANSFORM,  # Pixel coordinates.
                width,  # Columns.
                height,  # Rows.
                count,  # Bands.
                dtype,  # Data type.
            )  # End of the metadata.
            # Done.
            return
        # The metadata must describe the data.
        if (self.metadata.count, self.metadata.height, self.metadata.width) != data.shape:
            # Explain the problem.
            raise ValueError(f"data shape {data.shape} does not match the metadata")
        # Keep the dtype of the metadata in step with the data.
        if self.metadata.dtype.value != data.dtype.name:
            # Metadata with the actual dtype.
            self.metadata = replace(self.metadata, dtype=RasterDtype(data.dtype.name))

    # Raster from an array of shape (bands, height, width) or (height, width).
    @classmethod
    def from_array(
        cls,  # The class.
        data: NDArray[Any],  # Pixel values.
        crs: str = DEFAULT_CRS,  # Coordinate reference system.
        transform: Sequence[float] | None = None,  # Affine coefficients.
        nodata: float | None = None,  # Value that marks missing pixels.
        tags: dict[str, str] | None = None,  # Free-form tags.
    ) -> Raster:  # The raster.
        # Array view.
        values = np.asarray(data)
        # Boolean masks are stored as bytes.
        if values.dtype == np.bool_:
            # 0 and 1.
            values = values.astype(np.uint8)
        # Single bands get a band axis.
        if values.ndim == 2:
            # Shape (1, height, width).
            values = values[np.newaxis]
        # Only two- and three-dimensional arrays are rasters.
        if values.ndim != 3:
            # Explain the problem.
            raise ValueError(f"raster data must be 2-D or 3-D, got shape {values.shape}")
        # Band, row and column counts.
        count, height, width = values.shape
        # Build the metadata.
        metadata = RasterMetadata(
            crs=crs or "",  # Coordinate system.
            transform=normalize_transform(transform),  # Affine coefficients.
            width=width,  # Columns.
            height=height,  # Rows.
            count=count,  # Bands.
            dtype=RasterDtype(values.dtype.name),  # Data type.
            nodata=nodata,  # No-data value.
            tags=dict(tags or {}),  # Tags.
        )  # End of the metadata.
        # Build the raster.
        return cls(data=values, metadata=metadata)

    # Raster from a file readable by GDAL.
    @classmethod
    def from_file(
        cls,  # The class.
        path: str | Path,  # File path or GDAL dataset name.
        lazy: bool = False,  # Defer reading the pixels until load().
        bands: Sequence[int] | None = None,  # 1-based band numbers; all by default.
        window: tuple[int, int, int, int] | None = None,  # (row, col, height, width).
        dtype: str | None = "float32",  # Output dtype; None keeps the stored dtype.
    ) -> Raster:  # The raster.
        # Imported lazily.
        import rasterio

        # Open the dataset.
        with rasterio.open(path) as src:
            # Band numbers to read.
            indexes = list(bands) if bands else list(range(1, src.count + 1))
            # Band numbers must exist.
            if any(not 1 <= i <= src.count for i in indexes):
                # Explain the problem.
                raise ValueError(f"bands {indexes} outside 1..{src.count} of {path}")
            # A window must lie inside the dataset.
            if window is not None and not _window_inside(window, src.height, src.width):
                # Explain the problem.
                raise ValueError(f"window {window} outside the {src.height} x {src.width} raster")
            # Pixel window, or the whole raster.
            win = None if window is None else _window(*window)
            # Affine coefficients of the dataset.
            a, b, c, d, e, f = tuple(src.transform)[:6]
            # Row and column of the window origin.
            row0, col0 = (window[0], window[1]) if window is not None else (0, 0)
            # Transform of the window: the origin moves to its top-left corner.
            transform = (a, b, c + a * col0 + b * row0, d, e, f + d * col0 + e * row0)
            # Height and width of what is read.
            height, width = (src.height, src.width) if window is None else window[2:]
            # Output dtype.
            stored = src.dtypes[indexes[0] - 1]
            # Requested dtype, or the stored one.
            out_dtype = np.dtype(dtype if dtype is not None else stored)
            # Build the metadata.
            metadata = RasterMetadata(
                crs=src.crs.to_string() if src.crs else "",  # Coordinate system.
                transform=tuple(transform)[:6],  # Affine coefficients.
                width=width,  # Columns.
                height=height,  # Rows.
                count=len(indexes),  # Bands.
                dtype=RasterDtype(out_dtype.name),  # Data type after reading.
                nodata=src.nodata,  # No-data value.
                tags={k: str(v) for k, v in src.tags().items()},  # Dataset tags.
            )  # End of the metadata.
            # Read the pixels unless lazy.
            data = None if lazy else src.read(indexes, window=win).astype(out_dtype, copy=False)
        # Remember how to read lazily.
        raster = cls(data=data, metadata=metadata, source=Path(path), lazy=lazy)
        # Keep the read parameters for load().
        raster._read_args = (indexes, window)  # type: ignore[attr-defined]
        # Return the raster.
        return raster

    # Read the pixels of a lazy raster.
    def load(self) -> Raster:
        # Only lazy rasters without data need reading.
        if self.data is not None or self.source is None:
            # Nothing to do.
            return self
        # Imported lazily.
        import rasterio

        # Band numbers and window stored by from_file.
        indexes, window = getattr(self, "_read_args", (None, None))
        # Pixel window.
        win = None if window is None else _window(*window)
        # Open and read.
        with rasterio.open(self.source) as src:
            # All bands when none were chosen.
            indexes = indexes or list(range(1, src.count + 1))
            # Pixels in the dtype of the metadata.
            dtype = self.metadata.dtype.value if self.metadata else "float32"
            # Read.
            self.data = src.read(indexes, window=win).astype(dtype, copy=False)
        # Loaded.
        self.lazy = False
        # Allow chaining.
        return self

    # Data after loading; raises when there is none.
    def require_data(self) -> NDArray[Any]:
        # Load lazy rasters.
        self.load()
        # Rasters without data cannot be processed.
        if self.data is None or self.metadata is None:
            # Explain the problem.
            raise ValueError("raster has no data")
        # The array.
        return self.data

    # Metadata; raises when there is none.
    def _meta(self) -> RasterMetadata:
        # Rasters without metadata cannot be georeferenced.
        if self.metadata is None:
            # Explain the problem.
            raise ValueError("raster has no metadata")
        # The metadata.
        return self.metadata

    # Shape (bands, height, width).
    @property
    def shape(self) -> tuple[int, ...]:
        # From the data when loaded.
        if self.data is not None:
            # Array shape.
            return tuple(self.data.shape)
        # From the metadata otherwise.
        if self.metadata is not None:
            # Metadata sizes.
            return (self.metadata.count, self.metadata.height, self.metadata.width)
        # Empty raster.
        return (0, 0, 0)

    # Number of columns.
    @property
    def width(self) -> int:
        # Last axis.
        return self.shape[-1]

    # Number of rows.
    @property
    def height(self) -> int:
        # Second-last axis.
        return self.shape[-2]

    # Number of bands.
    @property
    def count(self) -> int:
        # First axis.
        return self.shape[0]

    # Data type of the pixels.
    @property
    def dtype(self) -> np.dtype[Any]:
        # From the data when loaded.
        if self.data is not None:
            # Array dtype.
            return self.data.dtype
        # From the metadata otherwise.
        return np.dtype(self.metadata.dtype.value if self.metadata else "float32")

    # Coordinate reference system.
    @property
    def crs(self) -> str:
        # From the metadata.
        return self.metadata.crs if self.metadata else ""

    # Affine coefficients.
    @property
    def transform(self) -> tuple[float, ...]:
        # From the metadata.
        return self.metadata.transform if self.metadata else IDENTITY_TRANSFORM

    # No-data value.
    @property
    def nodata(self) -> float | None:
        # From the metadata.
        return self.metadata.nodata if self.metadata else None

    # Bounds (min x, min y, max x, max y) in CRS units.
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        # Bounds of the grid.
        return transform_bounds(self.transform, self.width, self.height)

    # Pixel size (x, y) of a north-up raster.
    @property
    def resolution(self) -> tuple[float, float]:
        # Affine coefficients.
        a, b, _, d, e, _ = self.transform
        # Rotated rasters have no single pixel size.
        if b != 0 or d != 0:
            # Explain the problem.
            raise ValueError("rotated rasters have no axis-aligned resolution")
        # Absolute pixel sizes.
        return abs(a), abs(e)

    # Map coordinates of pixels (centres by default).
    def xy(
        self,  # This object.
        row: Any,  # Row or rows.
        col: Any,  # Column or columns.
        offset: Literal["center", "ul"] = "center",  # Pixel centre or upper-left corner.
    ) -> tuple[Any, Any]:  # x and y.
        # Affine coefficients.
        a, b, c, d, e, f = self.transform
        # Shift to the pixel centre.
        shift = 0.5 if offset == "center" else 0.0
        # Fractional positions.
        cc = np.asarray(col, dtype=np.float64) + shift
        # Fractional rows.
        rr = np.asarray(row, dtype=np.float64) + shift
        # Apply the transform.
        return a * cc + b * rr + c, d * cc + e * rr + f

    # Row and column of the pixels that contain map coordinates.
    def rowcol(self, x: Any, y: Any) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        # Affine coefficients.
        a, b, c, d, e, f = self.transform
        # Determinant of the linear part.
        det = a * e - b * d
        # Singular transforms cannot be inverted.
        if det == 0:
            # Explain the problem.
            raise ValueError("the transform is not invertible")
        # Offsets from the origin.
        dx, dy = np.asarray(x, dtype=np.float64) - c, np.asarray(y, dtype=np.float64) - f
        # Column from the inverse transform.
        col = (e * dx - b * dy) / det
        # Row from the inverse transform.
        row = (-d * dx + a * dy) / det
        # Pixels containing the points.
        return np.floor(row).astype(np.int64), np.floor(col).astype(np.int64)

    # Pixel values at map coordinates; NaN outside the raster or where invalid.
    def sample(self, x: Any, y: Any) -> NDArray[np.float64]:
        # Pixel data.
        data = self.require_data()
        # Rows and columns of the points.
        rows, cols = self.rowcol(np.atleast_1d(x), np.atleast_1d(y))
        # Points inside the raster.
        inside = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        # Output of shape (points, bands).
        out = np.full((rows.size, self.count), np.nan)
        # Values at the points inside, as float64.
        values = data[:, rows[inside], cols[inside]].T.astype(np.float64)
        # Invalid values become NaN.
        if self.nodata is not None:
            # No-data values.
            values[values == self.nodata] = np.nan
        # Store the values.
        out[inside] = values
        # Return the samples.
        return out

    # Mask of shape (height, width): True where every band is valid.
    def valid_mask(self) -> NDArray[np.bool_]:
        # Pixel data.
        data = self.require_data()
        # Finite values (integers are always finite).
        valid = np.all(np.isfinite(data), axis=0)
        # Values different from the no-data value.
        if self.nodata is not None and not np.isnan(self.nodata):
            # Exclude no-data pixels.
            valid &= np.all(data != self.nodata, axis=0)
        # Return the mask.
        return valid

    # Masked array whose mask marks invalid values per band.
    def masked(self) -> np.ma.MaskedArray[Any, Any]:
        # Pixel data.
        data = self.require_data()
        # Invalid values per band.
        invalid = ~np.isfinite(data)
        # No-data values.
        if self.nodata is not None and not np.isnan(self.nodata):
            # Add them to the mask.
            invalid |= data == self.nodata
        # Masked array.
        return np.ma.MaskedArray(data, mask=invalid)

    # Per-band statistics of the valid pixels.
    def statistics(
        self,  # This object.
        percentiles: Sequence[float] = (2.0, 25.0, 50.0, 75.0, 98.0),  # Percentiles.
    ) -> list[dict[str, float]]:  # One dictionary per band.
        # Masked data.
        masked = self.masked()
        # Results per band.
        results = []
        # Visit the bands.
        for band in range(self.count):
            # Valid values of the band as float64.
            values = masked[band].compressed().astype(np.float64)
            # Bands without valid values give NaN statistics.
            if values.size == 0:
                # Count only.
                stats = {"count": 0.0, "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
                # Percentiles are undefined.
                stats.update({f"p{p:g}": np.nan for p in percentiles})
            # Otherwise compute the statistics.
            else:
                # Moments and extremes.
                stats = {
                    "count": float(values.size),  # Number of valid pixels.
                    "min": float(values.min()),  # Minimum.
                    "max": float(values.max()),  # Maximum.
                    "mean": float(values.mean()),  # Mean.
                    "std": float(values.std()),  # Population standard deviation.
                }  # End of the statistics.
                # Percentiles with linear interpolation.
                stats.update({f"p{p:g}": float(np.percentile(values, p)) for p in percentiles})
            # Keep the band statistics.
            results.append(stats)
        # Return the statistics.
        return results

    # Histogram of the valid values of one band (1-based).
    def histogram(
        self,  # This object.
        band: int = 1,  # Band number.
        bins: int = 256,  # Number of bins.
        value_range: tuple[float, float] | None = None,  # Limits; data range by default.
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:  # Counts and bin edges.
        # Band numbers are 1-based.
        if not 1 <= band <= self.count:
            # Explain the problem.
            raise ValueError(f"band {band} outside 1..{self.count}")
        # Valid values of the band.
        values = self.masked()[band - 1].compressed().astype(np.float64)
        # Counts and edges.
        counts, edges = np.histogram(values, bins=bins, range=value_range)
        # Return both.
        return counts.astype(np.int64), edges

    # Evaluate a band math expression on the bands b1..bN.
    def band_math(
        self,  # This object.
        expression: str,  # Expression such as "(b2 - b1) / (b2 + b1)".
        dtype: str = "float32",  # Output dtype (floating point).
    ) -> Raster:  # Single-band raster with NaN where undefined.
        # Pixel data.
        data = self.require_data()
        # Bands as float64 with invalid values as NaN.
        values = np.ma.filled(self.masked().astype(np.float64), np.nan)
        # Band variables.
        variables = {f"b{i + 1}": values[i] for i in range(data.shape[0])}
        # Evaluate.
        result = np.asarray(evaluate_expression(expression, variables), dtype=np.float64)
        # Constant expressions fill the raster.
        result = np.broadcast_to(result, data.shape[1:]).copy()
        # Undefined results become NaN.
        result[~np.isfinite(result)] = np.nan
        # Output dtype.
        values = result.astype(dtype)
        # Single-band raster on the same grid with NaN as no-data.
        return Raster.from_array(values, crs=self.crs, transform=self.transform, nodata=np.nan)

    # Raster with some bands (1-based numbers) in the given order.
    def select_bands(self, bands: Sequence[int]) -> Raster:
        # Pixel data.
        data = self.require_data()
        # Band numbers must exist.
        if any(not 1 <= b <= self.count for b in bands):
            # Explain the problem.
            raise ValueError(f"bands {list(bands)} outside 1..{self.count}")
        # Selected bands.
        return self.with_data(data[[b - 1 for b in bands]])

    # Raster on the same grid with new data (any band count) and metadata changes.
    def with_data(self, data: NDArray[Any], **changes: Any) -> Raster:
        # Metadata with the new band count and dtype.
        dtype = RasterDtype(data.dtype.name)
        # Metadata of the new data.
        meta = replace(self._meta(), count=data.shape[0], dtype=dtype, **changes)
        # New raster.
        return Raster(data=data, metadata=meta, source=self.source)

    # Stack single- or multi-band rasters on the same grid.
    @classmethod
    def stack(cls, rasters: Sequence[Raster]) -> Raster:
        # At least one raster is needed.
        if not rasters:
            # Explain the problem.
            raise ValueError("stack needs at least one raster")
        # Reference grid.
        first = rasters[0]
        # Every raster must be on the same grid.
        for other in rasters[1:]:
            # Compare size, transform and CRS.
            if not first.same_grid(other):
                # Explain the problem.
                raise ValueError("rasters must share size, transform and CRS to be stacked")
        # Concatenated bands with a common dtype.
        data = np.concatenate([r.require_data() for r in rasters], axis=0)
        # New raster on the reference grid.
        return Raster.from_array(
            data,  # Stacked bands.
            crs=first.crs,  # Coordinate system of the grid.
            transform=first.transform,  # Transform of the grid.
            nodata=first.nodata,  # No-data value of the first raster.
        )  # End of the raster.

    # Whether another raster has the same size, transform and CRS.
    def same_grid(self, other: Raster, tolerance: float = 1e-9) -> bool:
        # Sizes.
        same_size = (self.height, self.width) == (other.height, other.width)
        # Transforms within a tolerance.
        same_transform = np.allclose(self.transform, other.transform, rtol=0, atol=tolerance)
        # All three conditions.
        return same_size and bool(same_transform) and self.crs == other.crs

    # Raster with the pixels converted to another dtype.
    def astype(self, dtype: Any) -> Raster:
        # Converted data.
        return self.with_data(self.require_data().astype(dtype))

    # Raster with a new no-data value (the pixels are unchanged).
    def set_nodata(self, nodata: float | None) -> Raster:
        # Same data, new metadata.
        return self.with_data(self.require_data(), nodata=nodata)

    # Raster with the pixels where mask is True set to the no-data value.
    def mask(self, mask: NDArray[np.bool_], fill: float | None = None) -> Raster:
        # Pixel data.
        data = self.require_data()
        # Boolean mask of shape (height, width).
        where = np.asarray(mask, dtype=bool)
        # The mask must match the grid.
        if where.shape != data.shape[1:]:
            # Explain the problem.
            raise ValueError(f"mask shape {where.shape} does not match {data.shape[1:]}")
        # Value written into the masked pixels.
        value = self._fill_value(fill)
        # Copy with the masked pixels replaced.
        out = np.where(where[np.newaxis], np.asarray(value, dtype=data.dtype), data)
        # New raster with the fill value as no-data.
        return self.with_data(out, nodata=value)

    # Value for missing pixels: the argument, the no-data value or NaN.
    def _fill_value(self, fill: float | None) -> float:
        # Explicit value.
        if fill is not None:
            # As given.
            return fill
        # The no-data value of the raster.
        if self.nodata is not None:
            # As stored.
            return self.nodata
        # Floating point rasters use NaN.
        if self.dtype.kind in "fc":
            # Not a number.
            return float("nan")
        # Integer rasters need an explicit value.
        raise ValueError("integer rasters without nodata need an explicit fill value")

    # Pixels of a window as an array (reads from the source for lazy rasters).
    def read_window(
        self,  # This object.
        row_off: int,  # First row.
        col_off: int,  # First column.
        height: int,  # Rows.
        width: int,  # Columns.
    ) -> NDArray[Any]:  # Array of shape (bands, height, width).
        # The window must lie inside the raster.
        if row_off < 0 or col_off < 0 or height <= 0 or width <= 0:
            # Explain the problem.
            raise ValueError(f"invalid window ({row_off}, {col_off}, {height}, {width})")
        # In-memory rasters are sliced.
        if self.data is not None:
            # Slice (clipped at the edges by NumPy).
            return self.data[:, row_off : row_off + height, col_off : col_off + width]
        # Rasters without data must have a source.
        if self.source is None:
            # Explain the problem.
            raise ValueError("cannot read a window without data or source")
        # Imported lazily.
        import rasterio

        # Band numbers and window of the lazy raster.
        indexes, window = getattr(self, "_read_args", (None, None))
        # Shift by the window offset of the raster itself.
        base_row, base_col = (window[0], window[1]) if window else (0, 0)
        # Read the window.
        with rasterio.open(self.source) as src:
            # Pixels in the dtype of the metadata.
            dtype = self.metadata.dtype.value if self.metadata else "float32"
            # Window of the file.
            win = _window(base_row + row_off, base_col + col_off, height, width)
            # Read the bands.
            return src.read(indexes or list(range(1, src.count + 1)), window=win).astype(dtype)

    # Raster of a pixel window with its transform.
    def window(self, row_off: int, col_off: int, height: int, width: int) -> Raster:
        # Pixels of the window.
        data = self.read_window(row_off, col_off, height, width)
        # Affine coefficients.
        a, b, c, d, e, f = self.transform
        # Origin of the window.
        origin = (a * col_off + b * row_off + c, d * col_off + e * row_off + f)
        # Transform of the window.
        transform = (a, b, origin[0], d, e, origin[1])
        # New raster.
        meta = replace(
            self._meta(),  # Metadata of the raster.
            width=data.shape[2],  # Columns of the window.
            height=data.shape[1],  # Rows of the window.
            transform=transform,  # Transform of the window.
            bounds=None,  # Recomputed from the transform.
            resolution=None,  # Recomputed from the transform.
        )  # End of the metadata.
        # Build the raster.
        return Raster(data=data, metadata=meta, source=self.source)

    # Tiles (row offset, column offset, data) covering the raster.
    def tiles(
        self,  # This object.
        tile_size: int = 256,  # Tile size.
        overlap: int = 0,  # Overlap between neighbours.
    ) -> Iterator[tuple[int, int, NDArray[Any]]]:  # Offsets and data.
        # Pixel data.
        data = self.require_data()
        # Row starts.
        for row_off in tile_offsets(self.height, tile_size, overlap):
            # Column starts.
            for col_off in tile_offsets(self.width, tile_size, overlap):
                # Tile data.
                tile = data[:, row_off : row_off + tile_size, col_off : col_off + tile_size]
                # Offsets and data.
                yield row_off, col_off, tile

    # Raster cropped to map bounds (left, bottom, right, top).
    def crop(self, bounds: tuple[float, float, float, float]) -> Raster:
        # Affine coefficients.
        a, b, c, d, e, f = self.transform
        # Cropping by bounds needs an axis-aligned grid.
        if b != 0 or d != 0:
            # Explain the problem.
            raise ValueError("crop needs a raster without rotation")
        # Bounds.
        left, bottom, right, top = bounds
        # Tolerance for bounds on pixel edges.
        eps = 1e-9
        # Fractional columns of the bounds.
        cols = sorted(((left - c) / a, (right - c) / a))
        # Fractional rows of the bounds.
        rows = sorted(((top - f) / e, (bottom - f) / e))
        # First column, clamped.
        col0 = max(0, int(np.floor(cols[0] + eps)))
        # End column, clamped.
        col1 = min(self.width, int(np.ceil(cols[1] - eps)))
        # First row, clamped.
        row0 = max(0, int(np.floor(rows[0] + eps)))
        # End row, clamped.
        row1 = min(self.height, int(np.ceil(rows[1] - eps)))
        # The bounds must overlap the raster.
        if col1 <= col0 or row1 <= row0:
            # Explain the problem.
            raise ValueError(f"bounds {bounds} do not overlap the raster {self.bounds}")
        # Window of the bounds.
        return self.window(row0, col0, row1 - row0, col1 - col0)

    # Raster with the pixels outside geometries set to the no-data value.
    def clip(
        self,  # This object.
        geometries: Any,  # Shapely geometry, GeoJSON mapping, or a sequence of them.
        crop: bool = True,  # Crop to the bounds of the geometries first.
        all_touched: bool = False,  # Keep every pixel the geometries touch.
        invert: bool = False,  # Set the pixels inside instead.
        fill: float | None = None,  # Value of removed pixels; no-data or NaN by default.
    ) -> Raster:  # Clipped raster.
        # Imported lazily.
        from rasterio.features import geometry_mask  # Rasterised masks.
        from shapely.geometry import mapping, shape  # GeoJSON conversion.
        from shapely.ops import unary_union  # Union of the geometries.

        # A single geometry becomes a list.
        single = isinstance(geometries, dict) or hasattr(geometries, "geom_type")
        # List of geometries.
        items = [geometries] if single else list(geometries)
        # Shapely geometries.
        shapes = [shape(g) if isinstance(g, dict) else g for g in items]
        # At least one geometry is needed.
        if not shapes:
            # Explain the problem.
            raise ValueError("clip needs at least one geometry")
        # Raster to mask: cropped to the geometries unless inverted.
        target = self.crop(unary_union(shapes).bounds) if crop and not invert else self
        # Value of removed pixels.
        value = target._fill_value(fill)
        # Mask that is True outside the geometries (inside when inverted).
        outside = geometry_mask(
            [mapping(s) for s in shapes],  # Geometries as GeoJSON.
            out_shape=(target.height, target.width),  # Grid size.
            transform=_affine(target.transform),  # Grid transform.
            all_touched=all_touched,  # Rasterisation rule.
            invert=invert,  # Which side is masked.
        )  # End of the mask.
        # Masked raster.
        return target.mask(outside, fill=value)

    # Resample to a new pixel size on the same CRS.
    def resample(
        self,  # This object.
        scale: float | None = None,  # Size factor: > 1 upsamples, < 1 downsamples.
        method: ResamplingName = "bilinear",  # Resampling method.
        resolution: float | tuple[float, float] | None = None,  # Target pixel size instead.
    ) -> Raster:  # Resampled raster.
        # Target size from the scale factor.
        if scale is not None and resolution is None:
            # The factor must be positive.
            if scale <= 0:
                # Explain the problem.
                raise ValueError(f"scale must be positive, got {scale}")
            # New size.
            new_h, new_w = max(1, round(self.height * scale)), max(1, round(self.width * scale))
        # Target size from the resolution.
        elif resolution is not None and scale is None:
            # A scalar size applies to both axes.
            size = (resolution, resolution) if isinstance(resolution, (int, float)) else resolution
            # Pixel sizes along x and y.
            rx, ry = size
            # Current pixel sizes.
            px, py = self.resolution
            # New size.
            new_h, new_w = max(1, round(self.height * py / ry)), max(1, round(self.width * px / rx))
        # Neither or both.
        else:
            # Exactly one of scale and resolution is needed.
            raise ValueError("give either scale or resolution")
        # Pixel size factors along the columns and the rows.
        sx, sy = self.width / new_w, self.height / new_h
        # Affine coefficients.
        a, b, c, d, e, f = self.transform
        # Transform of the new grid: the same extent with larger or smaller pixels.
        dst = _affine((a * sx, b * sy, c, d * sx, e * sy, f))
        # Warp onto the new grid.
        return self._warp(self.crs, dst, new_w, new_h, method)

    # Reproject to another CRS.
    def reproject(
        self,  # This object.
        target_crs: str,  # Target coordinate reference system.
        resolution: float | tuple[float, float] | None = None,  # Target pixel size.
        method: ResamplingName = "bilinear",  # Resampling method.
    ) -> Raster:  # Reprojected raster.
        # Imported lazily.
        from rasterio.warp import calculate_default_transform

        # The source CRS must be known.
        if not self.crs:
            # Explain the problem.
            raise ValueError("cannot reproject a raster without a CRS")
        # Output grid covering the input.
        dst, width, height = calculate_default_transform(
            self.crs,  # Source CRS.
            target_crs,  # Target CRS.
            self.width,  # Source width.
            self.height,  # Source height.
            *self.bounds,  # Source bounds.
            resolution=resolution,  # Target pixel size.
        )  # End of the grid.
        # rasterio computes the size whenever the source size is given.
        if width is None or height is None:
            # Explain the problem.
            raise ValueError(f"cannot compute the output grid for {target_crs}")
        # Warp onto the grid.
        return self._warp(target_crs, dst, int(width), int(height), method)

    # Warp onto the grid (CRS, transform and size) of another raster.
    def match(self, other: Raster, method: ResamplingName = "bilinear") -> Raster:
        # Target transform.
        target = _affine(other.transform)
        # Warp onto the grid of the other raster.
        return self._warp(other.crs, target, other.width, other.height, method)

    # Warp onto a target grid with the GDAL warper.
    def _warp(
        self,  # This object.
        dst_crs: str,  # Target CRS.
        dst_transform: Any,  # Target transform.
        width: int,  # Target columns.
        height: int,  # Target rows.
        method: str,  # Resampling method.
    ) -> Raster:  # Warped raster.
        # Imported lazily.
        from rasterio.warp import reproject

        # Pixel data.
        data = self.require_data()
        # No-data value used during warping.
        nodata = self.nodata
        # Floating point rasters without one use NaN.
        if nodata is None and data.dtype.kind == "f":
            # Not a number.
            nodata = np.nan
        # Output filled with the no-data value, or zero for integers without one.
        fill = 0 if nodata is None else nodata
        # Output array.
        out = np.full((self.count, height, width), fill, dtype=data.dtype)
        # Warp every band.
        reproject(
            source=data,  # Source pixels.
            destination=out,  # Output pixels.
            src_transform=_affine(self.transform),  # Source grid.
            src_crs=self.crs or DEFAULT_CRS,  # Source CRS.
            src_nodata=nodata,  # Source no-data value.
            dst_transform=dst_transform,  # Target grid.
            dst_crs=dst_crs or DEFAULT_CRS,  # Target CRS.
            dst_nodata=nodata,  # Target no-data value.
            resampling=_resampling(method),  # Method.
        )  # End of the warp.
        # New raster.
        return Raster.from_array(
            out,  # Warped pixels.
            crs=dst_crs,  # Target CRS.
            transform=tuple(dst_transform)[:6],  # Target transform.
            nodata=nodata,  # No-data value.
            tags=self._meta().tags,  # Tags of the source.
        )  # End of the raster.

    # Apply a function to the data and keep the georeferencing.
    def apply(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Raster:
        # Result of the function.
        result = np.asarray(func(self.require_data(), *args, **kwargs))
        # Single bands get a band axis.
        if result.ndim == 2:
            # Shape (1, height, width).
            result = result[np.newaxis]
        # The spatial size must not change.
        if result.shape[1:] != (self.height, self.width):
            # Explain the problem.
            raise ValueError(f"function changed the raster size to {result.shape}")
        # New raster.
        return self.with_data(result)

    # Write the raster to a file (tiled and compressed GeoTIFF by default).
    def to_file(
        self,  # This object.
        path: str | Path,  # Output file.
        driver: str = "GTiff",  # GDAL driver.
        compress: Literal["lzw", "deflate", "zstd", "none"] = "lzw",  # Compression.
        tiled: bool = True,  # Internal tiling.
        blockxsize: int = 256,  # Tile width, a multiple of 16.
        blockysize: int = 256,  # Tile height, a multiple of 16.
    ) -> Path:  # Path of the written file.
        # Imported lazily.
        import rasterio

        # Output path.
        path = Path(path)
        # Create the parent directory.
        path.parent.mkdir(parents=True, exist_ok=True)
        # Dataset profile.
        profile = self._profile(driver, compress, tiled, blockxsize, blockysize)
        # Write the bands and tags.
        with rasterio.open(path, "w", **profile) as dst:
            # Pixels and tags.
            self._write(dst)
        # Return the path.
        return path

    # Creation profile of a dataset holding this raster.
    def _profile(
        self,  # This object.
        driver: str,  # GDAL driver.
        compress: str,  # Compression.
        tiled: bool,  # Internal tiling.
        blockxsize: int,  # Tile width.
        blockysize: int,  # Tile height.
    ) -> dict[str, Any]:  # Keyword arguments of rasterio.open.
        # Pixel data.
        data = self.require_data()
        # Basic profile.
        profile: dict[str, Any] = {
            "driver": driver,  # Format.
            "dtype": data.dtype.name,  # Data type.
            "width": self.width,  # Columns.
            "height": self.height,  # Rows.
            "count": self.count,  # Bands.
            "crs": self.crs or None,  # Coordinate system.
            "transform": _affine(self.transform),  # Grid.
            "nodata": self.nodata,  # No-data value.
        }  # End of the profile.
        # GeoTIFF creation options.
        if driver == "GTiff":
            # Tiling.
            profile["tiled"] = tiled
            # Block size of tiled files.
            if tiled:
                # Tile sizes.
                profile.update(blockxsize=blockxsize, blockysize=blockysize)
            # Compression.
            if compress != "none":
                # Codec.
                profile["compress"] = compress
        # Return the profile.
        return profile

    # Write the pixels and tags into an open dataset.
    def _write(self, dst: Any) -> None:
        # Pixels.
        dst.write(self.require_data())
        # Tags.
        if self._meta().tags:
            # Dataset tags.
            dst.update_tags(**self._meta().tags)

    # Write a Cloud Optimized GeoTIFF with internal overviews.
    def to_cog(
        self,  # This object.
        path: str | Path,  # Output file.
        compress: Literal["lzw", "deflate", "zstd", "none"] = "deflate",  # Compression.
        blocksize: int = 512,  # Tile size.
        overview_resampling: ResamplingName = "average",  # Overview resampling.
    ) -> Path:  # Path of the written file.
        # Imported lazily.
        from rasterio.io import MemoryFile  # In-memory datasets.
        from rasterio.shutil import copy  # Dataset copies with another driver.

        # Output path.
        path = Path(path)
        # Create the parent directory.
        path.parent.mkdir(parents=True, exist_ok=True)
        # Plain GeoTIFF profile for the in-memory copy.
        profile = self._profile("GTiff", "none", False, 256, 256)
        # Write a plain GeoTIFF in memory, then copy it with the COG driver.
        with MemoryFile() as memory:
            # Write the raster.
            with memory.open(**profile) as dst:
                # Pixels and tags.
                self._write(dst)
            # Copy with the COG driver, which adds overviews and orders the file.
            copy(
                memory.name,  # Source.
                str(path),  # Destination.
                driver="COG",  # Cloud Optimized GeoTIFF.
                COMPRESS=compress.upper(),  # Codec.
                BLOCKSIZE=blocksize,  # Tile size.
                OVERVIEW_RESAMPLING=overview_resampling.upper(),  # Overview method.
            )  # End of the copy.
        # Return the path.
        return path

    # Short description.
    def __repr__(self) -> str:
        # Shape, dtype and CRS.
        return f"Raster(shape={self.shape}, dtype={self.dtype}, crs={self.crs or 'None'})"


# Whether a (row, col, height, width) window lies inside a raster.
def _window_inside(window: tuple[int, int, int, int], height: int, width: int) -> bool:
    # Window position and size.
    row, col, h, w = window
    # Non-negative start, positive size and end inside the raster.
    return row >= 0 and col >= 0 and h > 0 and w > 0 and row + h <= height and col + w <= width


# rasterio Window of a pixel window; from_slices, unlike the attrs constructor
# of Window, is visible to type checkers.
def _window(row_off: int, col_off: int, height: int, width: int) -> Any:
    # Imported lazily.
    from rasterio.windows import Window

    # Rows and columns as start and stop indices.
    return Window.from_slices((row_off, row_off + height), (col_off, col_off + width))


# rasterio Affine object from six coefficients.
def _affine(transform: Sequence[float]) -> Any:
    # Imported lazily.
    from rasterio.transform import Affine

    # Build the Affine object.
    return Affine(*normalize_transform(transform))


# Stack the rasters read from several files (for example one band per file).
def stack_files(paths: Iterable[str | Path]) -> Raster:
    # Read every file.
    rasters = [Raster.from_file(p) for p in paths]
    # Stack them.
    return Raster.stack(rasters)


# =============================================================================
# End of module src/unbihexium/core/raster.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
