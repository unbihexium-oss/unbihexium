"""Tile abstraction for tiled raster processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from unbihexium.core.raster import Raster


@dataclass(frozen=True)
class TileIndex:
    """Index of a tile within a grid."""

    row: int
    col: int
    level: int = 0

    def __str__(self) -> str:
        return f"z{self.level}/r{self.row}/c{self.col}"


@dataclass
class Tile:
    """A tile of raster data."""

    index: TileIndex
    data: NDArray[np.floating[Any]]
    bounds: tuple[float, float, float, float] | None = None
    offset: tuple[int, int] = (0, 0)
    size: tuple[int, int] = (256, 256)

    @property
    def height(self) -> int:
        return self.data.shape[-2]

    @property
    def width(self) -> int:
        return self.data.shape[-1]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape


@dataclass
class TileGrid:
    """A grid of tiles for a raster dataset."""

    tile_size: tuple[int, int]
    overlap: int = 0
    num_rows: int = 0
    num_cols: int = 0
    raster_height: int = 0
    raster_width: int = 0
    crs: str = "EPSG:4326"
    transform: tuple[float, ...] | None = None

    @classmethod
    def from_raster(
        cls,
        raster: Raster,
        tile_size: int | tuple[int, int] = 256,
        overlap: int = 0,
    ) -> TileGrid:
        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)

        step_y = tile_size[0] - overlap
        step_x = tile_size[1] - overlap

        num_rows = int(np.ceil(raster.height / step_y))
        num_cols = int(np.ceil(raster.width / step_x))

        return cls(
            tile_size=tile_size,
            overlap=overlap,
            num_rows=num_rows,
            num_cols=num_cols,
            raster_height=raster.height,
            raster_width=raster.width,
            crs=raster.metadata.crs if raster.metadata else "EPSG:4326",
            transform=raster.metadata.transform if raster.metadata else None,
        )

    @property
    def total_tiles(self) -> int:
        return self.num_rows * self.num_cols

    def tile_offset(self, index: TileIndex) -> tuple[int, int]:
        step_y = self.tile_size[0] - self.overlap
        step_x = self.tile_size[1] - self.overlap
        return (index.row * step_y, index.col * step_x)

    def indices(self) -> Iterator[TileIndex]:
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                yield TileIndex(row=row, col=col)

    def get_tile(self, raster: Raster, index: TileIndex) -> Tile:
        row_off, col_off = self.tile_offset(index)
        actual_height = min(self.tile_size[0], self.raster_height - row_off)
        actual_width = min(self.tile_size[1], self.raster_width - col_off)
        data = raster.read_window(row_off, col_off, actual_height, actual_width)
        return Tile(
            index=index,
            data=data,
            offset=(row_off, col_off),
            size=(actual_height, actual_width),
        )

    def tiles(self, raster: Raster) -> Iterator[Tile]:
        for index in self.indices():
            yield self.get_tile(raster, index)

    def mosaic(
        self,
        tiles: list[Tile],
        dtype: np.dtype[Any] | None = None,
        nodata: float = 0.0,
    ) -> NDArray[np.floating[Any]]:
        if not tiles:
            return np.array([])
        count = tiles[0].data.shape[0]
        dtype = dtype or tiles[0].data.dtype
        output = np.full((count, self.raster_height, self.raster_width), nodata, dtype=dtype)
        for tile in tiles:
            row_off, col_off = tile.offset
            h, w = tile.height, tile.width
            output[:, row_off : row_off + h, col_off : col_off + w] = tile.data
        return output
