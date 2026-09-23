# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/core/tile.py
# Title       : Tiling with overlap, mosaicking with blending, XYZ tiles
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Splits rasters that are too large for one computation into tiles and puts
# the processed tiles back together:
#
#   tile_offsets    start positions along one axis
#   TileIndex       row, column and level of a tile
#   Tile            the data of a tile with its offset, bounds and transform
#   TileGrid        a grid of tiles over a raster; get_tile, tiles, mosaic
#   blend_weights   weights that taper to the tile edges inside the overlap
#   xyz_tile, xyz_bounds, xyz_bounds_mercator
#                   Web Mercator (XYZ, "slippy map") tile arithmetic
#
# Tiling: tiles advance by step = size - overlap. The last tile is shifted
# back so that it ends at the raster edge, so every tile has the full size
# when the raster is at least one tile large; this keeps the input size of
# convolutional networks constant.
#
# Mosaicking: "last" writes the tiles in order, "average" gives equal weight
# to every tile, and "linear" weights each pixel by
#   w(i) = min(1, (i + 0.5) / overlap, (n - i - 0.5) / overlap)
# along each axis (the product of the two axes), so that seams in the
# overlaps are feathered. The result is sum(w * tile) / sum(w), which is the
# exact tile value wherever only one tile covers a pixel.
#
# XYZ tiles: at zoom z the world is 2^z by 2^z tiles; for longitude lon and
# latitude lat, x = floor((lon + 180) / 360 * 2^z) and
# y = floor((1 - asinh(tan(lat)) / pi) / 2 * 2^z), with y growing
# southwards; the projection is EPSG:3857 with the half circumference
# 20037508.342789244 m.
#
# References
# ----------
#   Huang, B., Reichman, D., Collins, L. M., Bradbury, K., Malof, J. M.
#     (2018). Tiling and stitching segmentation output for remote sensing:
#     basic challenges and recommendations. arXiv:1805.12219.
#   OpenStreetMap Wiki. Slippy map tilenames.
#     https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Trigonometry of the Web Mercator projection.
import math

# Tile iterators.
from collections.abc import Iterable, Iterator

# Record containers.
from dataclasses import dataclass

# Types used only for annotations.
from typing import TYPE_CHECKING, Any, Literal

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Imported for annotations only, to avoid an import cycle with raster.py.
if TYPE_CHECKING:
    # Raster container.
    from unbihexium.core.raster import Raster

# Half the equatorial circumference of the Web Mercator sphere in metres.
WEB_MERCATOR_HALF = 20037508.342789244

# Latitude limit of Web Mercator in degrees.
WEB_MERCATOR_MAX_LAT = 85.0511287798066


# Start positions of tiles of the given size along an axis.
def tile_offsets(length: int, size: int, overlap: int = 0) -> list[int]:
    # Sizes must be positive.
    if length <= 0 or size <= 0:
        # Explain the problem.
        raise ValueError(f"length and size must be positive, got {length} and {size}")
    # The overlap must leave a positive step.
    if not 0 <= overlap < size:
        # Explain the problem.
        raise ValueError(f"overlap must be in [0, {size}), got {overlap}")
    # Rasters smaller than a tile give one partial tile.
    if length <= size:
        # Single tile at the origin.
        return [0]
    # Distance between tile starts.
    step = size - overlap
    # Regular starts whose tiles fit inside the raster.
    offsets = list(range(0, length - size + 1, step))
    # Add a last tile flush with the edge when the regular tiles stop short.
    if offsets[-1] + size < length:
        # Start of the last tile.
        offsets.append(length - size)
    # Return the starts.
    return offsets


# Weight ramp along one axis of a tile.
def _ramp(n: int, overlap: int) -> NDArray[np.float64]:
    # No overlap means uniform weights.
    if overlap <= 0:
        # Ones.
        return np.ones(n)
    # Pixel centres.
    i = np.arange(n) + 0.5
    # Distance to the nearer edge relative to the overlap, capped at one.
    return np.minimum(1.0, np.minimum(i, n - i) / overlap)


# Blending weights of a tile of shape (height, width).
def blend_weights(height: int, width: int, overlap: int) -> NDArray[np.float64]:
    # Outer product of the two ramps.
    return np.outer(_ramp(height, overlap), _ramp(width, overlap))


# Position of a tile within a grid.
@dataclass(frozen=True)
class TileIndex:
    # Row of the tile (y of XYZ tiles).
    row: int
    # Column of the tile (x of XYZ tiles).
    col: int
    # Level of detail (z of XYZ tiles).
    level: int = 0

    # Path-like name.
    def __str__(self) -> str:
        # Level, row and column.
        return f"z{self.level}/r{self.row}/c{self.col}"


# One tile of raster data.
@dataclass
class Tile:
    # Position in the grid.
    index: TileIndex
    # Data of shape (bands, height, width).
    data: NDArray[Any]
    # Map bounds (left, bottom, right, top), when the grid is georeferenced.
    bounds: tuple[float, float, float, float] | None = None
    # Row and column of the top-left pixel in the raster.
    offset: tuple[int, int] = (0, 0)
    # Height and width of the tile.
    size: tuple[int, int] = (256, 256)
    # Affine transform of the tile.
    transform: tuple[float, ...] | None = None

    # Height in pixels.
    @property
    def height(self) -> int:
        # Second-last axis.
        return int(self.data.shape[-2])

    # Width in pixels.
    @property
    def width(self) -> int:
        # Last axis.
        return int(self.data.shape[-1])

    # Shape of the data.
    @property
    def shape(self) -> tuple[int, ...]:
        # Shape tuple.
        return tuple(self.data.shape)

    # Copy of the tile with new data of the same height and width.
    def with_data(self, data: NDArray[Any]) -> Tile:
        # Array view.
        values = np.asarray(data)
        # The spatial size must not change.
        if values.shape[-2:] != self.data.shape[-2:]:
            # Explain the problem.
            raise ValueError(f"new data {values.shape} does not match tile {self.data.shape}")
        # New tile with the same position.
        return Tile(self.index, values, self.bounds, self.offset, self.size, self.transform)


# A grid of tiles over a raster.
@dataclass
class TileGrid:
    # Height and width of a tile.
    tile_size: tuple[int, int]
    # Overlap between neighbouring tiles in pixels.
    overlap: int = 0
    # Number of tile rows.
    num_rows: int = 0
    # Number of tile columns.
    num_cols: int = 0
    # Height of the raster.
    raster_height: int = 0
    # Width of the raster.
    raster_width: int = 0
    # Coordinate reference system.
    crs: str = "EPSG:4326"
    # Affine transform of the raster.
    transform: tuple[float, ...] | None = None

    # Grid for a raster shape.
    @classmethod
    def for_shape(
        cls,  # The class.
        height: int,  # Raster height.
        width: int,  # Raster width.
        tile_size: int | tuple[int, int] = 256,  # Tile size.
        overlap: int = 0,  # Overlap in pixels.
        crs: str = "EPSG:4326",  # Coordinate reference system.
        transform: tuple[float, ...] | None = None,  # Affine transform.
    ) -> TileGrid:  # The grid.
        # Square tiles from an integer.
        size = (tile_size, tile_size) if isinstance(tile_size, int) else tuple(tile_size)
        # Starts along the rows.
        rows = tile_offsets(height, size[0], overlap)
        # Starts along the columns.
        cols = tile_offsets(width, size[1], overlap)
        # Build the grid.
        return cls(
            tile_size=(int(size[0]), int(size[1])),  # Tile size.
            overlap=overlap,  # Overlap.
            num_rows=len(rows),  # Rows.
            num_cols=len(cols),  # Columns.
            raster_height=height,  # Height.
            raster_width=width,  # Width.
            crs=crs,  # Coordinate system.
            transform=transform,  # Transform.
        )  # End of the grid.

    # Grid for a raster.
    @classmethod
    def from_raster(
        cls,  # The class.
        raster: Raster,  # Raster to tile.
        tile_size: int | tuple[int, int] = 256,  # Tile size.
        overlap: int = 0,  # Overlap in pixels.
    ) -> TileGrid:  # The grid.
        # Georeferencing of the raster.
        meta = raster.metadata
        # Build the grid.
        return cls.for_shape(
            raster.height,  # Height.
            raster.width,  # Width.
            tile_size,  # Tile size.
            overlap,  # Overlap.
            meta.crs if meta else "EPSG:4326",  # Coordinate system.
            meta.transform if meta else None,  # Transform.
        )  # End of the grid.

    # Number of tiles.
    @property
    def total_tiles(self) -> int:
        # Rows times columns.
        return self.num_rows * self.num_cols

    # Row and column starts of the tiles.
    def _offsets(self) -> tuple[list[int], list[int]]:
        # Row starts.
        rows = tile_offsets(self.raster_height, self.tile_size[0], self.overlap)
        # Column starts.
        cols = tile_offsets(self.raster_width, self.tile_size[1], self.overlap)
        # Both lists.
        return rows, cols

    # Pixel offset (row, column) of a tile.
    def tile_offset(self, index: TileIndex) -> tuple[int, int]:
        # Starts along both axes.
        rows, cols = self._offsets()
        # The index must lie in the grid.
        if not (0 <= index.row < len(rows) and 0 <= index.col < len(cols)):
            # Explain the problem.
            raise IndexError(f"tile {index} outside a {len(rows)} x {len(cols)} grid")
        # Offsets of the tile.
        return rows[index.row], cols[index.col]

    # Every tile index in row-major order.
    def indices(self) -> Iterator[TileIndex]:
        # Rows.
        for row in range(self.num_rows):
            # Columns.
            for col in range(self.num_cols):
                # Index of the tile.
                yield TileIndex(row=row, col=col)

    # Affine transform and bounds of the window at an offset.
    def _georeference(
        self,  # This object.
        row_off: int,  # Row of the top-left pixel.
        col_off: int,  # Column of the top-left pixel.
        height: int,  # Window height.
        width: int,  # Window width.
    ) -> tuple[tuple[float, ...] | None, tuple[float, float, float, float] | None]:  # Both.
        # Grids without a transform have no map coordinates.
        if self.transform is None:
            # Nothing to compute.
            return None, None
        # Affine coefficients.
        a, b, c, d, e, f = (float(v) for v in self.transform[:6])
        # Origin of the window.
        c2, f2 = a * col_off + b * row_off + c, d * col_off + e * row_off + f
        # Corner x coordinates.
        xs = [c2, c2 + a * width, c2 + b * height, c2 + a * width + b * height]
        # Corner y coordinates.
        ys = [f2, f2 + d * width, f2 + e * height, f2 + d * width + e * height]
        # Transform and bounds.
        return (a, b, c2, d, e, f2), (min(xs), min(ys), max(xs), max(ys))

    # Read one tile from a raster.
    def get_tile(self, raster: Raster, index: TileIndex) -> Tile:
        # Offset of the tile.
        row_off, col_off = self.tile_offset(index)
        # Height, clipped at the raster edge.
        height = min(self.tile_size[0], self.raster_height - row_off)
        # Width, clipped at the raster edge.
        width = min(self.tile_size[1], self.raster_width - col_off)
        # Pixel data.
        data = raster.read_window(row_off, col_off, height, width)
        # Map coordinates.
        transform, bounds = self._georeference(row_off, col_off, height, width)
        # Build the tile.
        return Tile(index, data, bounds, (row_off, col_off), (height, width), transform)

    # Every tile of a raster in row-major order.
    def tiles(self, raster: Raster) -> Iterator[Tile]:
        # Visit the indices.
        for index in self.indices():
            # Read the tile.
            yield self.get_tile(raster, index)

    # Put tiles back together into an array of the raster size.
    def mosaic(
        self,  # This object.
        tiles: Iterable[Tile],  # Tiles to combine.
        dtype: Any = None,  # Output dtype; the dtype of the first tile by default.
        nodata: float = 0.0,  # Value of pixels that no tile covers.
        blend: Literal["last", "average", "linear"] = "last",  # Combination of overlaps.
    ) -> NDArray[Any]:  # Array of shape (bands, height, width).
        # Materialise the tiles.
        items = list(tiles)
        # Without tiles there is nothing to combine.
        if not items:
            # Explain the problem.
            raise ValueError("mosaic needs at least one tile")
        # Tile data as three-dimensional arrays.
        arrays = [t.data if t.data.ndim == 3 else t.data[np.newaxis] for t in items]
        # Number of bands.
        count = arrays[0].shape[0]
        # Output dtype.
        out_dtype = np.dtype(dtype) if dtype is not None else arrays[0].dtype
        # Output shape.
        shape = (count, self.raster_height, self.raster_width)
        # Direct copy: later tiles overwrite earlier ones.
        if blend == "last":
            # Output filled with the no-data value.
            output = np.full(shape, nodata, dtype=out_dtype)
            # Write every tile.
            for tile, array in zip(items, arrays):
                # Offset and size.
                (r, c), (h, w) = tile.offset, array.shape[-2:]
                # Copy the tile.
                output[:, r : r + h, c : c + w] = array
            # Return the mosaic.
            return output
        # Weighted blending needs a known mode.
        if blend not in ("average", "linear"):
            # Explain the problem.
            raise ValueError(f"blend must be 'last', 'average' or 'linear', got {blend!r}")
        # Weighted sum of the tiles.
        total = np.zeros(shape, dtype=np.float64)
        # Sum of the weights.
        weight = np.zeros(shape[1:], dtype=np.float64)
        # Accumulate every tile.
        for tile, array in zip(items, arrays):
            # Offset and size.
            (r, c), (h, w) = tile.offset, array.shape[-2:]
            # Weights of the tile.
            wt = blend_weights(h, w, self.overlap if blend == "linear" else 0)
            # Pixels with valid (finite) values in every band.
            valid = np.all(np.isfinite(array), axis=0)
            # Invalid pixels get no weight.
            wt = np.where(valid, wt, 0.0)
            # Weighted values; invalid values are replaced by zero.
            total[:, r : r + h, c : c + w] += np.where(valid, array, 0.0) * wt
            # Weights.
            weight[r : r + h, c : c + w] += wt
        # Pixels covered by at least one tile.
        covered = weight > 0
        # Normalised values; uncovered pixels get the no-data value.
        result = np.where(covered, total / np.where(covered, weight, 1.0), nodata)
        # Round for integer outputs.
        if np.issubdtype(out_dtype, np.integer):
            # Nearest integer.
            result = np.rint(result)
        # Convert to the output dtype.
        return result.astype(out_dtype)


# XYZ tile (x, y) that contains a longitude and latitude at a zoom level.
def xyz_tile(lon: float, lat: float, zoom: int) -> TileIndex:
    # Zoom levels are non-negative.
    if zoom < 0:
        # Explain the problem.
        raise ValueError(f"zoom must be non-negative, got {zoom}")
    # Number of tiles per axis.
    n = 2**zoom
    # Latitudes are limited to the square Web Mercator world.
    lat = max(-WEB_MERCATOR_MAX_LAT, min(WEB_MERCATOR_MAX_LAT, lat))
    # Column from the longitude.
    x = int((lon + 180.0) / 360.0 * n)
    # Row from the Mercator ordinate.
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    # Clamp to the grid (the east and south edges belong to the last tile).
    return TileIndex(row=min(max(y, 0), n - 1), col=min(max(x, 0), n - 1), level=zoom)


# Longitude and latitude bounds (west, south, east, north) of an XYZ tile.
def xyz_bounds(tile: TileIndex) -> tuple[float, float, float, float]:
    # Number of tiles per axis.
    n = 2**tile.level
    # Longitude of the west edge.
    west = tile.col / n * 360.0 - 180.0
    # Longitude of the east edge.
    east = (tile.col + 1) / n * 360.0 - 180.0
    # Latitude of the north edge.
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile.row / n))))
    # Latitude of the south edge.
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (tile.row + 1) / n))))
    # Bounds.
    return west, south, east, north


# EPSG:3857 bounds (left, bottom, right, top) of an XYZ tile in metres.
def xyz_bounds_mercator(tile: TileIndex) -> tuple[float, float, float, float]:
    # Size of a tile in metres.
    size = 2 * WEB_MERCATOR_HALF / 2**tile.level
    # Left edge.
    left = -WEB_MERCATOR_HALF + tile.col * size
    # Top edge.
    top = WEB_MERCATOR_HALF - tile.row * size
    # Bounds.
    return left, top - size, left + size, top


# =============================================================================
# End of module src/unbihexium/core/tile.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
