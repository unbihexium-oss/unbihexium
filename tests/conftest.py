"""Test configuration and fixtures for unbihexium."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def sample_raster_data() -> NDArray[np.floating[Any]]:
    """Create sample raster data for testing."""
    np.random.seed(42)
    return np.random.rand(3, 256, 256).astype(np.float32)


@pytest.fixture
def sample_coordinates() -> NDArray[np.floating[Any]]:
    """Create sample coordinates for geostatistics tests."""
    np.random.seed(42)
    return np.random.rand(50, 2) * 100


@pytest.fixture
def sample_values() -> NDArray[np.floating[Any]]:
    """Create sample values for geostatistics tests."""
    np.random.seed(42)
    coords = np.random.rand(50, 2) * 100
    return np.sin(coords[:, 0] / 10) + np.random.rand(50) * 0.1


@pytest.fixture
def tmp_geotiff(
    tmp_path: Path,
    sample_raster_data: NDArray[np.floating[Any]],
) -> Generator[Path, None, None]:
    """Create a temporary GeoTIFF file."""
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path / "test.tif"
    transform = from_bounds(0, 0, 1, 1, 256, 256)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=3,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(sample_raster_data)

    yield path

    if path.exists():
        path.unlink()


@pytest.fixture
def sample_bands() -> dict[str, NDArray[np.floating[Any]]]:
    """Create sample band data for index calculations."""
    np.random.seed(42)
    return {
        "RED": np.random.rand(256, 256).astype(np.float32) * 0.5,
        "NIR": np.random.rand(256, 256).astype(np.float32) * 0.8,
        "GREEN": np.random.rand(256, 256).astype(np.float32) * 0.4,
        "BLUE": np.random.rand(256, 256).astype(np.float32) * 0.3,
        "SWIR1": np.random.rand(256, 256).astype(np.float32) * 0.6,
        "SWIR2": np.random.rand(256, 256).astype(np.float32) * 0.5,
    }
