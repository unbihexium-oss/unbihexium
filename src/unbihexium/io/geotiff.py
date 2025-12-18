"""GeoTIFF and Cloud-Optimized GeoTIFF (COG) IO adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def read_geotiff(
    path: str | Path,
    bands: list[int] | None = None,
    window: tuple[int, int, int, int] | None = None,
) -> tuple[NDArray[np.floating[Any]], dict[str, Any]]:
    """Read a GeoTIFF file.

    Args:
        path: Path to the GeoTIFF file.
        bands: Band indices to read (1-indexed). None reads all bands.
        window: Optional window (row_start, row_stop, col_start, col_stop).

    Returns:
        Tuple of (data array, metadata dict).
    """
    try:
        import rasterio
    except ImportError as e:
        raise ImportError("rasterio is required for GeoTIFF support") from e

    path = Path(path)
    with rasterio.open(path) as src:
        if bands is None:
            bands = list(range(1, src.count + 1))

        if window is not None:
            from rasterio.windows import Window

            win = Window.from_slices(
                (window[0], window[1]),
                (window[2], window[3]),
            )
            data = src.read(bands, window=win)
        else:
            data = src.read(bands)

        metadata = {
            "crs": str(src.crs) if src.crs else None,
            "transform": src.transform,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
        }

    return data.astype(np.float32), metadata


def write_geotiff(
    path: str | Path,
    data: NDArray[np.floating[Any]],
    crs: str | None = None,
    transform: Any = None,
    nodata: float | None = None,
    compress: str = "lzw",
) -> Path:
    """Write data to a GeoTIFF file.

    Args:
        path: Output path.
        data: Data array (bands, height, width) or (height, width).
        crs: Coordinate reference system.
        transform: Affine transform.
        nodata: NoData value.
        compress: Compression method.

    Returns:
        Path to the written file.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError as e:
        raise ImportError("rasterio is required for GeoTIFF support") from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, ...]

    count, height, width = data.shape

    if transform is None:
        transform = from_bounds(0, 0, width, height, width, height)

    profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "transform": transform,
        "compress": compress,
    }

    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

    return path


def read_cog(
    url: str,
    bands: list[int] | None = None,
    window: tuple[int, int, int, int] | None = None,
) -> tuple[NDArray[np.floating[Any]], dict[str, Any]]:
    """Read a Cloud-Optimized GeoTIFF (COG) from URL.

    Args:
        url: URL to the COG file.
        bands: Band indices to read (1-indexed).
        window: Optional window (row_start, row_stop, col_start, col_stop).

    Returns:
        Tuple of (data array, metadata dict).
    """
    try:
        import rasterio
    except ImportError as e:
        raise ImportError("rasterio is required for COG support") from e

    with rasterio.open(url) as src:
        if bands is None:
            bands = list(range(1, src.count + 1))

        if window is not None:
            from rasterio.windows import Window

            win = Window.from_slices(
                (window[0], window[1]),
                (window[2], window[3]),
            )
            data = src.read(bands, window=win)
        else:
            data = src.read(bands)

        metadata = {
            "crs": str(src.crs) if src.crs else None,
            "transform": src.transform,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "is_cog": True,
        }

    return data.astype(np.float32), metadata
