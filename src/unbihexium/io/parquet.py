"""GeoParquet IO adapter for vector data."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_geoparquet(path: str | Path) -> Any:
    """Read a GeoParquet file.

    Args:
        path: Path to GeoParquet file.

    Returns:
        GeoDataFrame.
    """
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError("geopandas is required for GeoParquet support") from e

    path = Path(path)
    return gpd.read_parquet(path)


def write_geoparquet(
    path: str | Path,
    gdf: Any,
    compression: str = "snappy",
) -> Path:
    """Write GeoDataFrame to GeoParquet.

    Args:
        path: Output path.
        gdf: GeoDataFrame to write.
        compression: Compression codec.

    Returns:
        Path to written file.
    """
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError("geopandas is required for GeoParquet support") from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_parquet(path, compression=compression)
    return path
