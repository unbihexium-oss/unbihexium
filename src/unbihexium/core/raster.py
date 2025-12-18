"""Raster abstraction for geospatial imagery.

This module provides the core Raster class for handling geospatial raster data
with support for tiling, chunking, and streaming operations.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


class RasterDtype(str, Enum):
    """Supported raster data types."""

    UINT8 = "uint8"
    UINT16 = "uint16"
    INT16 = "int16"
    UINT32 = "uint32"
    INT32 = "int32"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


@dataclass(frozen=True)
class RasterMetadata:
    """Metadata for a raster dataset.

    Attributes:
        crs: Coordinate reference system (EPSG code or WKT).
        transform: Affine transformation matrix.
        width: Raster width in pixels.
        height: Raster height in pixels.
        count: Number of bands.
        dtype: Data type of the raster.
        nodata: No-data value.
        bounds: Bounding box (left, bottom, right, top).
        resolution: Pixel resolution (x, y).
    """

    crs: str
    transform: tuple[float, ...]
    width: int
    height: int
    count: int
    dtype: RasterDtype = RasterDtype.FLOAT32
    nodata: float | None = None
    bounds: tuple[float, float, float, float] | None = None
    resolution: tuple[float, float] | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "crs": self.crs,
            "transform": self.transform,
            "width": self.width,
            "height": self.height,
            "count": self.count,
            "dtype": self.dtype.value,
            "nodata": self.nodata,
            "bounds": self.bounds,
            "resolution": self.resolution,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RasterMetadata:
        """Create metadata from dictionary."""
        return cls(
            crs=data["crs"],
            transform=tuple(data["transform"]),
            width=data["width"],
            height=data["height"],
            count=data["count"],
            dtype=RasterDtype(data.get("dtype", "float32")),
            nodata=data.get("nodata"),
            bounds=tuple(data["bounds"]) if data.get("bounds") else None,
            resolution=tuple(data["resolution"]) if data.get("resolution") else None,
            tags=data.get("tags", {}),
        )


@dataclass
class Raster:
    """Core raster abstraction for geospatial imagery.

    Supports lazy loading, tiling, and streaming operations for efficient
    processing of large raster datasets.

    Attributes:
        data: The raster data as a numpy array (bands, height, width).
        metadata: Raster metadata including CRS, transform, etc.
        source: Optional source path or URL.
    """

    data: NDArray[np.floating[Any]] | None = None
    metadata: RasterMetadata | None = None
    source: str | Path | None = None
    _lazy: bool = False

    def __post_init__(self) -> None:
        """Validate raster initialization."""
        if self.data is not None and self.metadata is None:
            # Infer basic metadata from data
            if self.data.ndim == 2:
                height, width = self.data.shape
                count = 1
            elif self.data.ndim == 3:
                count, height, width = self.data.shape
            else:
                msg = f"Invalid raster dimensions: {self.data.ndim}"
                raise ValueError(msg)

            self.metadata = RasterMetadata(
                crs="EPSG:4326",
                transform=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
                width=width,
                height=height,
                count=count,
                dtype=RasterDtype(str(self.data.dtype)),
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the raster data."""
        if self.data is not None:
            return self.data.shape
        if self.metadata is not None:
            return (self.metadata.count, self.metadata.height, self.metadata.width)
        return (0, 0, 0)

    @property
    def width(self) -> int:
        """Return the width of the raster."""
        return self.shape[-1]

    @property
    def height(self) -> int:
        """Return the height of the raster."""
        return self.shape[-2]

    @property
    def count(self) -> int:
        """Return the number of bands."""
        if len(self.shape) == 2:
            return 1
        return self.shape[0]

    @property
    def dtype(self) -> np.dtype[Any]:
        """Return the data type of the raster."""
        if self.data is not None:
            return self.data.dtype
        if self.metadata is not None:
            return np.dtype(self.metadata.dtype.value)
        return np.dtype("float32")

    @classmethod
    def from_array(
        cls,
        data: NDArray[np.floating[Any]],
        crs: str = "EPSG:4326",
        transform: tuple[float, ...] | None = None,
        nodata: float | None = None,
    ) -> Raster:
        """Create a raster from a numpy array.

        Args:
            data: Numpy array with shape (bands, height, width) or (height, width).
            crs: Coordinate reference system.
            transform: Affine transformation matrix.
            nodata: No-data value.

        Returns:
            A new Raster instance.
        """
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        count, height, width = data.shape

        if transform is None:
            transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

        metadata = RasterMetadata(
            crs=crs,
            transform=transform,
            width=width,
            height=height,
            count=count,
            dtype=RasterDtype(str(data.dtype)),
            nodata=nodata,
        )

        return cls(data=data, metadata=metadata)

    @classmethod
    def from_file(cls, path: str | Path, lazy: bool = False) -> Raster:
        """Load a raster from a file.

        Args:
            path: Path to the raster file.
            lazy: If True, defer loading data until accessed.

        Returns:
            A new Raster instance.
        """
        import rasterio

        path = Path(path)
        instance = cls(source=path, _lazy=lazy)

        with rasterio.open(path) as src:
            instance.metadata = RasterMetadata(
                crs=str(src.crs),
                transform=tuple(src.transform),
                width=src.width,
                height=src.height,
                count=src.count,
                dtype=RasterDtype(str(src.dtypes[0])),
                nodata=src.nodata,
                bounds=src.bounds,
                resolution=src.res,
                tags=dict(src.tags()),
            )

            if not lazy:
                instance.data = src.read().astype(np.float32)

        return instance

    def load(self) -> None:
        """Load raster data if lazy-loaded."""
        if self._lazy and self.data is None and self.source is not None:
            import rasterio

            with rasterio.open(self.source) as src:
                self.data = src.read().astype(np.float32)
            self._lazy = False

    def to_file(
        self,
        path: str | Path,
        driver: str = "GTiff",
        compress: Literal["lzw", "deflate", "zstd", "none"] = "lzw",
        tiled: bool = True,
        blockxsize: int = 256,
        blockysize: int = 256,
    ) -> Path:
        """Save raster to a file.

        Args:
            path: Output file path.
            driver: GDAL driver name.
            compress: Compression method.
            tiled: Whether to write as tiled GeoTIFF.
            blockxsize: Tile width.
            blockysize: Tile height.

        Returns:
            Path to the saved file.
        """
        import rasterio
        from rasterio.transform import Affine

        if self.data is None:
            msg = "Cannot save raster without data"
            raise ValueError(msg)

        if self.metadata is None:
            msg = "Cannot save raster without metadata"
            raise ValueError(msg)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        profile = {
            "driver": driver,
            "dtype": self.data.dtype,
            "width": self.metadata.width,
            "height": self.metadata.height,
            "count": self.metadata.count,
            "crs": self.metadata.crs,
            "transform": Affine(*self.metadata.transform[:6]),
            "nodata": self.metadata.nodata,
        }

        if driver == "GTiff" and compress != "none":
            profile.update(
                {
                    "compress": compress,
                    "tiled": tiled,
                    "blockxsize": blockxsize,
                    "blockysize": blockysize,
                }
            )

        with rasterio.open(path, "w", **profile) as dst:
            if self.data.ndim == 2:
                dst.write(self.data, 1)
            else:
                dst.write(self.data)

        return path

    def read_window(
        self,
        row_off: int,
        col_off: int,
        height: int,
        width: int,
    ) -> NDArray[np.floating[Any]]:
        """Read a window from the raster.

        Args:
            row_off: Row offset.
            col_off: Column offset.
            height: Window height.
            width: Window width.

        Returns:
            Numpy array with the window data.
        """
        if self.data is not None:
            return self.data[:, row_off : row_off + height, col_off : col_off + width]

        if self.source is None:
            msg = "Cannot read window without data or source"
            raise ValueError(msg)

        import rasterio
        from rasterio.windows import Window

        with rasterio.open(self.source) as src:
            window = Window(col_off, row_off, width, height)
            return src.read(window=window).astype(np.float32)

    def tiles(
        self,
        tile_size: int = 256,
        overlap: int = 0,
    ) -> Iterator[tuple[int, int, NDArray[np.floating[Any]]]]:
        """Iterate over tiles of the raster.

        Args:
            tile_size: Size of each tile (square).
            overlap: Overlap between tiles in pixels.

        Yields:
            Tuples of (row_offset, col_offset, tile_data).
        """
        self.load()

        if self.data is None or self.metadata is None:
            return

        step = tile_size - overlap
        height = self.metadata.height
        width = self.metadata.width

        for row_off in range(0, height, step):
            for col_off in range(0, width, step):
                tile_height = min(tile_size, height - row_off)
                tile_width = min(tile_size, width - col_off)
                tile_data = self.data[
                    :, row_off : row_off + tile_height, col_off : col_off + tile_width
                ]
                yield row_off, col_off, tile_data

    def resample(
        self,
        scale: float,
        method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    ) -> Raster:
        """Resample the raster to a new resolution.

        Args:
            scale: Scale factor (>1 upsamples, <1 downsamples).
            method: Resampling method.

        Returns:
            A new resampled Raster.
        """
        from scipy.ndimage import zoom

        self.load()

        if self.data is None or self.metadata is None:
            msg = "Cannot resample without data"
            raise ValueError(msg)

        order_map = {"nearest": 0, "bilinear": 1, "cubic": 3}
        order = order_map[method]

        # Resample each band
        resampled = zoom(self.data, (1, scale, scale), order=order)

        new_height, new_width = resampled.shape[1], resampled.shape[2]
        transform = list(self.metadata.transform)
        transform[0] /= scale  # x pixel size
        transform[4] /= scale  # y pixel size

        new_metadata = RasterMetadata(
            crs=self.metadata.crs,
            transform=tuple(transform),
            width=new_width,
            height=new_height,
            count=self.metadata.count,
            dtype=self.metadata.dtype,
            nodata=self.metadata.nodata,
        )

        return Raster(data=resampled, metadata=new_metadata)

    def crop(
        self,
        bounds: tuple[float, float, float, float],
    ) -> Raster:
        """Crop the raster to the given bounds.

        Args:
            bounds: Bounding box (left, bottom, right, top).

        Returns:
            A new cropped Raster.
        """
        self.load()

        if self.data is None or self.metadata is None:
            msg = "Cannot crop without data"
            raise ValueError(msg)

        from rasterio.transform import Affine, rowcol

        transform = Affine(*self.metadata.transform[:6])
        left, bottom, right, top = bounds

        # Convert bounds to pixel coordinates
        row_start, col_start = rowcol(transform, left, top)
        row_end, col_end = rowcol(transform, right, bottom)

        # Ensure bounds are within raster
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(self.metadata.height, row_end)
        col_end = min(self.metadata.width, col_end)

        cropped_data = self.data[:, row_start:row_end, col_start:col_end]

        # Update transform for cropped raster
        new_transform = transform * Affine.translation(col_start, row_start)

        new_metadata = RasterMetadata(
            crs=self.metadata.crs,
            transform=tuple(new_transform)[:6],
            width=col_end - col_start,
            height=row_end - row_start,
            count=self.metadata.count,
            dtype=self.metadata.dtype,
            nodata=self.metadata.nodata,
            bounds=bounds,
        )

        return Raster(data=cropped_data, metadata=new_metadata)

    def reproject(
        self,
        target_crs: str,
        resolution: float | None = None,
    ) -> Raster:
        """Reproject the raster to a new CRS.

        Args:
            target_crs: Target coordinate reference system.
            resolution: Target resolution (optional).

        Returns:
            A new reprojected Raster.
        """
        from rasterio.crs import CRS
        from rasterio.enums import Resampling
        from rasterio.transform import Affine
        from rasterio.warp import calculate_default_transform, reproject

        self.load()

        if self.data is None or self.metadata is None:
            msg = "Cannot reproject without data"
            raise ValueError(msg)

        src_crs = CRS.from_string(self.metadata.crs)
        dst_crs = CRS.from_string(target_crs)
        src_transform = Affine(*self.metadata.transform[:6])

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            dst_crs,
            self.metadata.width,
            self.metadata.height,
            *self.metadata.bounds
            if self.metadata.bounds
            else (0, 0, self.metadata.width, self.metadata.height),
            resolution=resolution,
        )

        dst_data = np.zeros(
            (self.metadata.count, dst_height, dst_width),
            dtype=self.data.dtype,
        )

        for i in range(self.metadata.count):
            reproject(
                source=self.data[i],
                destination=dst_data[i],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

        new_metadata = RasterMetadata(
            crs=target_crs,
            transform=tuple(dst_transform)[:6],
            width=dst_width,
            height=dst_height,
            count=self.metadata.count,
            dtype=self.metadata.dtype,
            nodata=self.metadata.nodata,
        )

        return Raster(data=dst_data, metadata=new_metadata)

    def apply(
        self,
        func: callable,
        *args: Any,
        **kwargs: Any,
    ) -> Raster:
        """Apply a function to the raster data.

        Args:
            func: Function to apply.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            A new Raster with the function applied.
        """
        self.load()

        if self.data is None:
            msg = "Cannot apply function without data"
            raise ValueError(msg)

        result = func(self.data, *args, **kwargs)

        return Raster(
            data=result,
            metadata=self.metadata,
            source=self.source,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Raster(shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"crs={self.metadata.crs if self.metadata else 'None'})"
        )
