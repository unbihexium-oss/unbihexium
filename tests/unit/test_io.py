# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for IO modules (GeoTIFF, Zarr, GeoJSON, Parquet).

This module provides comprehensive tests for:
- GeoTIFF/COG read/write operations
- Zarr chunked array operations
- GeoJSON feature read/write
- GeoParquet operations

Tests use small synthetic fixtures and skip gracefully if optional deps are missing.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from unbihexium.io.geojson import (
    features_to_geojson,
    geometry_to_feature,
    read_geojson,
    write_geojson,
)


def _has_rasterio() -> bool:
    """Check if rasterio is available."""
    try:
        import rasterio

        return True
    except ImportError:
        return False


def _has_zarr() -> bool:
    """Check if zarr is available."""
    try:
        import zarr

        return True
    except ImportError:
        return False


def _has_geopandas() -> bool:
    """Check if geopandas is available."""
    try:
        import geopandas

        return True
    except ImportError:
        return False


# GeoJSON Tests


class TestGeoJSONIO:
    """Tests for GeoJSON I/O functions."""

    def test_geometry_to_feature_point(self) -> None:
        """Test point geometry to feature conversion."""
        geometry = {"type": "Point", "coordinates": [0.0, 0.0]}
        feature = geometry_to_feature(geometry, properties={"name": "test"})

        assert feature["type"] == "Feature"
        assert feature["geometry"] == geometry
        assert feature["properties"]["name"] == "test"

    def test_geometry_to_feature_polygon(self) -> None:
        """Test polygon geometry to feature conversion."""
        geometry = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        }
        feature = geometry_to_feature(geometry, properties={"id": 1})

        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Polygon"

    def test_geometry_to_feature_empty_properties(self) -> None:
        """Test feature creation with no properties."""
        geometry = {"type": "Point", "coordinates": [0.0, 0.0]}
        feature = geometry_to_feature(geometry)

        assert feature["properties"] == {}

    def test_features_to_geojson(self) -> None:
        """Test features to GeoJSON collection conversion."""
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                "properties": {"id": 1},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                "properties": {"id": 2},
            },
        ]

        collection = features_to_geojson(features)

        assert collection["type"] == "FeatureCollection"
        assert len(collection["features"]) == 2

    def test_write_and_read_geojson(self) -> None:
        """Test writing and reading GeoJSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.geojson"

            geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [10.0, 20.0]},
                        "properties": {"name": "test_point"},
                    }
                ],
            }

            write_geojson(geojson, filepath)
            assert filepath.exists()

            loaded = read_geojson(filepath)
            assert loaded["type"] == "FeatureCollection"
            assert len(loaded["features"]) == 1
            assert loaded["features"][0]["properties"]["name"] == "test_point"

    def test_write_geojson_creates_parent_dirs(self) -> None:
        """Test that write_geojson creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "test.geojson"

            geojson = {"type": "FeatureCollection", "features": []}
            write_geojson(geojson, filepath)

            assert filepath.exists()

    def test_read_geojson_invalid_file(self) -> None:
        """Test reading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_geojson(Path("/nonexistent/path.geojson"))


# GeoTIFF Tests


@pytest.mark.skipif(not _has_rasterio(), reason="rasterio not installed")
class TestGeoTIFFIO:
    """Tests for GeoTIFF I/O functions."""

    def test_write_and_read_geotiff(self) -> None:
        """Test writing and reading GeoTIFF files."""
        from unbihexium.io.geotiff import read_geotiff, write_geotiff

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.tif"

            # Create synthetic data
            data = np.random.rand(3, 32, 32).astype(np.float32)

            write_geotiff(
                data,
                filepath,
                crs="EPSG:4326",
                transform=(1.0, 0.0, 0.0, 0.0, -1.0, 32.0),
            )
            assert filepath.exists()

            loaded, meta = read_geotiff(filepath)
            assert loaded.shape == data.shape
            np.testing.assert_array_almost_equal(loaded, data, decimal=5)

    def test_geotiff_preserves_crs(self) -> None:
        """Test that CRS is preserved in round-trip."""
        from unbihexium.io.geotiff import read_geotiff, write_geotiff

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_crs.tif"

            data = np.ones((1, 16, 16), dtype=np.float32)
            write_geotiff(
                data,
                filepath,
                crs="EPSG:32632",
                transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0),
            )

            _, meta = read_geotiff(filepath)
            assert "EPSG:32632" in str(meta.get("crs", ""))


# Zarr Tests


@pytest.mark.skipif(not _has_zarr(), reason="zarr not installed")
class TestZarrIO:
    """Tests for Zarr I/O functions."""

    def test_write_and_read_zarr(self) -> None:
        """Test writing and reading Zarr arrays."""
        from unbihexium.io.zarr_io import read_zarr, write_zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            data = np.random.rand(4, 64, 64).astype(np.float32)

            write_zarr(data, zarr_path, chunks=(1, 32, 32))
            assert zarr_path.exists()

            loaded = read_zarr(zarr_path)
            assert loaded.shape == data.shape
            np.testing.assert_array_almost_equal(loaded[:], data, decimal=5)

    def test_zarr_chunking(self) -> None:
        """Test that Zarr respects chunk specification."""
        from unbihexium.io.zarr_io import read_zarr, write_zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "chunked.zarr"

            data = np.zeros((8, 128, 128), dtype=np.float32)
            chunks = (2, 64, 64)

            write_zarr(data, zarr_path, chunks=chunks)
            loaded = read_zarr(zarr_path)

            assert loaded.chunks == chunks


# Parquet Tests


@pytest.mark.skipif(not _has_geopandas(), reason="geopandas not installed")
class TestGeoParquetIO:
    """Tests for GeoParquet I/O functions."""

    def test_write_and_read_geoparquet(self) -> None:
        """Test writing and reading GeoParquet files."""
        import geopandas as gpd
        from shapely.geometry import Point

        from unbihexium.io.parquet import read_geoparquet, write_geoparquet

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.parquet"

            # Create simple GeoDataFrame
            gdf = gpd.GeoDataFrame(
                {"name": ["A", "B"], "value": [1, 2]},
                geometry=[Point(0, 0), Point(1, 1)],
                crs="EPSG:4326",
            )

            write_geoparquet(gdf, filepath)
            assert filepath.exists()

            loaded = read_geoparquet(filepath)
            assert len(loaded) == 2
            assert "name" in loaded.columns


# Integration Tests


class TestIOIntegration:
    """Integration tests for IO operations."""

    def test_geojson_round_trip_complex(self) -> None:
        """Test GeoJSON round-trip with complex features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "complex.geojson"

            geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                        },
                        "properties": {
                            "name": "polygon",
                            "area": 100.0,
                            "tags": ["a", "b"],
                        },
                    }
                ],
            }

            write_geojson(geojson, filepath)
            loaded = read_geojson(filepath)

            assert loaded["features"][0]["properties"]["area"] == 100.0
            assert loaded["features"][0]["properties"]["tags"] == ["a", "b"]
