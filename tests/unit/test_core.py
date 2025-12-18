"""Tests for core abstractions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


class TestRaster:
    """Tests for Raster class."""

    def test_from_array_creates_raster(
        self, sample_raster_data: NDArray[np.floating[Any]]
    ) -> None:
        from unbihexium.core.raster import Raster

        raster = Raster.from_array(sample_raster_data)

        assert raster.data is not None
        assert raster.shape == (3, 256, 256)
        assert raster.count == 3
        assert raster.height == 256
        assert raster.width == 256

    def test_from_array_2d(self) -> None:
        from unbihexium.core.raster import Raster

        data = np.random.rand(256, 256).astype(np.float32)
        raster = Raster.from_array(data)

        assert raster.count == 1
        assert raster.data.shape == (1, 256, 256)

    def test_tiles_iterator(
        self, sample_raster_data: NDArray[np.floating[Any]]
    ) -> None:
        from unbihexium.core.raster import Raster

        raster = Raster.from_array(sample_raster_data)
        tiles = list(raster.tiles(tile_size=128))

        assert len(tiles) == 4  # 2x2 grid
        for row, col, tile_data in tiles:
            assert tile_data.shape[1] <= 128
            assert tile_data.shape[2] <= 128

    def test_resample(
        self, sample_raster_data: NDArray[np.floating[Any]]
    ) -> None:
        from unbihexium.core.raster import Raster

        raster = Raster.from_array(sample_raster_data)
        resampled = raster.resample(scale=0.5)

        assert resampled.height == 128
        assert resampled.width == 128


class TestVector:
    """Tests for Vector class."""

    def test_from_wkt(self) -> None:
        from unbihexium.core.vector import Vector

        wkt = "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"
        vector = Vector.from_wkt(wkt)

        assert vector.feature_count == 1
        assert vector.crs == "EPSG:4326"

    def test_buffer(self) -> None:
        from unbihexium.core.vector import Vector

        wkt = "POINT(0 0)"
        vector = Vector.from_wkt(wkt)
        buffered = vector.buffer(1.0)

        assert buffered.data is not None


class TestTile:
    """Tests for Tile and TileGrid classes."""

    def test_tile_grid_from_raster(
        self, sample_raster_data: NDArray[np.floating[Any]]
    ) -> None:
        from unbihexium.core.raster import Raster
        from unbihexium.core.tile import TileGrid

        raster = Raster.from_array(sample_raster_data)
        grid = TileGrid.from_raster(raster, tile_size=64)

        assert grid.num_rows == 4
        assert grid.num_cols == 4
        assert grid.total_tiles == 16


class TestSpectralIndices:
    """Tests for spectral index calculations."""

    def test_ndvi(self, sample_bands: dict[str, NDArray[np.floating[Any]]]) -> None:
        from unbihexium.core.index import compute_index

        ndvi = compute_index("NDVI", sample_bands)

        assert ndvi.shape == (256, 256)
        assert ndvi.min() >= -1.0
        assert ndvi.max() <= 1.0

    def test_ndwi(self, sample_bands: dict[str, NDArray[np.floating[Any]]]) -> None:
        from unbihexium.core.index import compute_index

        ndwi = compute_index("NDWI", sample_bands)

        assert ndwi.shape == (256, 256)
        assert ndwi.min() >= -1.0
        assert ndwi.max() <= 1.0

    def test_evi(self, sample_bands: dict[str, NDArray[np.floating[Any]]]) -> None:
        from unbihexium.core.index import compute_index

        evi = compute_index("EVI", sample_bands)

        assert evi.shape == (256, 256)

    def test_unknown_index_raises(
        self, sample_bands: dict[str, NDArray[np.floating[Any]]]
    ) -> None:
        from unbihexium.core.index import compute_index

        with pytest.raises(ValueError, match="Unknown index"):
            compute_index("INVALID", sample_bands)


class TestPipeline:
    """Tests for pipeline execution."""

    def test_pipeline_run(self) -> None:
        from unbihexium.core.pipeline import Pipeline, PipelineConfig, PipelineStatus

        config = PipelineConfig(
            pipeline_id="test",
            name="Test Pipeline",
        )
        pipeline = Pipeline(config)

        def step1(inputs: dict) -> dict:
            return {"value": inputs.get("value", 0) + 1}

        pipeline.add_step(step1)
        run = pipeline.run({"value": 10})

        assert run.status == PipelineStatus.COMPLETED
        assert run.outputs == {"value": "11"}


class TestEvidence:
    """Tests for evidence and provenance."""

    def test_provenance_record(self) -> None:
        from unbihexium.core.evidence import ProvenanceRecord

        record = ProvenanceRecord(
            record_id="test-001",
            pipeline_id="test_pipeline",
            run_id="run-001",
        )

        data = record.to_dict()
        assert data["record_id"] == "test-001"
        assert data["pipeline_id"] == "test_pipeline"
