# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_core.py
# Title       : Tests of the core data model
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest, rasterio, GeoPandas,
#               Shapely and pyproj; the TorchScript test needs PyTorch
# =============================================================================
#
# Abstract
# --------
# Checks the core package against hand-computed or analytically known
# values: raster georeferencing, windows, cropping, clipping, resampling,
# band math, statistics and file round trips (GeoTIFF and COG); vector
# measures (planar and geodesic), rasterisation and polygonisation; tile
# offsets, blending weights, mosaics and XYZ tile arithmetic; every spectral
# index at a reference pixel, sensor aliases and burn severity classes;
# sensor band tables and radiometric conversions; scenes, products (save,
# load, STAC) and the model wrapper; and the package version.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Analytic reference values.
import math

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Test framework.
import pytest

# Array type annotations.
from numpy.typing import NDArray

# Package version.
import unbihexium

# Spectral indices.
from unbihexium.core.index import (
    BURN_SEVERITY_CLASSES,  # Class names of burn severity.
    IndexCategory,  # Index groups.
    IndexRegistry,  # Index registry.
    classify_burn_severity,  # dNBR classes.
    compute_index,  # Index by name.
    compute_indices,  # Several indices.
    dnbr,  # Differenced NBR.
)  # End of the index imports.

# Model wrapper.
from unbihexium.core.model import ModelConfig, ModelTask, ModelWrapper

# Products.
from unbihexium.core.product import Product, ProductType

# Rasters.
from unbihexium.core.raster import Raster, RasterMetadata, evaluate_expression

# Scenes.
from unbihexium.core.scene import Scene, SceneMetadata

# Sensors.
from unbihexium.core.sensor import (
    LANDSAT_C2_SR_OFFSET,  # Landsat surface reflectance offset.
    LANDSAT_C2_SR_SCALE,  # Landsat surface reflectance scale.
    SPEED_OF_LIGHT,  # Speed of light.
    band_from_limits,  # Band from its limits.
    get_sensor,  # Sensor look-up.
    landsat_brightness_temperature,  # Brightness temperature.
    landsat_toa_reflectance,  # Top-of-atmosphere reflectance.
    sentinel2_l2a_offset,  # Sentinel-2 offset by baseline.
    to_reflectance,  # DN conversion.
)  # End of the sensor imports.

# Tiles.
from unbihexium.core.tile import (
    Tile,  # One tile.
    TileGrid,  # Grid of tiles.
    TileIndex,  # Tile position.
    blend_weights,  # Blending weights.
    tile_offsets,  # Tile starts.
    xyz_bounds,  # Longitude and latitude bounds of XYZ tiles.
    xyz_bounds_mercator,  # Web Mercator bounds of XYZ tiles.
    xyz_tile,  # XYZ tile of a point.
)  # End of the tile imports.

# Vectors.
from unbihexium.core.vector import GeometryType, Vector

# 10 m grid of UTM zone 35N whose first column lies on the central meridian.
UTM = (10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)

# Reference reflectances used to check every index formula.
PIXEL = {
    "BLUE": 0.05,  # Blue.
    "GREEN": 0.08,  # Green.
    "RED": 0.1,  # Red.
    "REDEDGE1": 0.2,  # Red edge 1.
    "NIR": 0.5,  # Near infrared.
    "SWIR1": 0.2,  # Shortwave infrared 1.6 um.
    "SWIR2": 0.1,  # Shortwave infrared 2.2 um.
}  # End of the reference pixel.


# 4 x 4 raster with the values 0..15 on the UTM grid.
def ramp(nodata: float | None = None) -> Raster:
    # Row-major values.
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    # Raster on the grid.
    return Raster.from_array(data, crs="EPSG:32635", transform=UTM, nodata=nodata)


# The version tuple holds the numbers of the version string.
def test_version_tuple_matches_string() -> None:
    # Numbers of the string.
    numbers = tuple(int(p) for p in unbihexium.__version__.split("."))
    # Same numbers.
    assert unbihexium.__version_tuple__ == numbers == (2, 0, 1)


# Arrays of three dimensions keep their shape.
def test_raster_from_array(sample_raster_data: NDArray[np.floating[Any]]) -> None:
    # Raster of the fixture.
    raster = Raster.from_array(sample_raster_data)
    # Shape and sizes.
    assert raster.shape == (3, 256, 256) and raster.count == 3
    # Height and width.
    assert raster.height == 256 and raster.width == 256
    # Default georeferencing.
    assert raster.crs == "EPSG:4326" and raster.transform == (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


# Two-dimensional arrays get a band axis; other ranks are rejected.
def test_raster_from_array_2d_and_invalid() -> None:
    # Single-band raster.
    raster = Raster.from_array(np.zeros((5, 7), dtype=np.uint8))
    # Band axis added.
    assert raster.data is not None and raster.data.shape == (1, 5, 7)
    # Metadata dtype follows the data.
    assert raster.metadata is not None and raster.metadata.dtype.value == "uint8"
    # One-dimensional data are rejected.
    with pytest.raises(ValueError, match="2-D or 3-D"):
        # Invalid rank.
        Raster.from_array(np.zeros(4))


# Bounds and resolution follow from the transform.
def test_raster_bounds_and_resolution() -> None:
    # Reference raster.
    raster = ramp()
    # Four 10 m pixels in each direction from the origin.
    assert raster.bounds == (500000.0, 5999960.0, 500040.0, 6000000.0)
    # Pixel size.
    assert raster.resolution == (10.0, 10.0)
    # The metadata holds the same bounds.
    assert raster.metadata is not None and raster.metadata.bounds == raster.bounds


# Metadata dictionaries round-trip.
def test_raster_metadata_round_trip() -> None:
    # Metadata of the reference raster.
    meta = ramp(nodata=-1.0).metadata
    # Round trip through a dictionary.
    assert meta is not None and RasterMetadata.from_dict(meta.to_dict()) == meta


# Pixel centres map to coordinates and back.
def test_raster_xy_and_rowcol() -> None:
    # Reference raster.
    raster = ramp()
    # Centre of the top-left pixel.
    assert tuple(map(float, raster.xy(0, 0))) == (500005.0, 5999995.0)
    # A point in row 2, column 1.
    rows, cols = raster.rowcol(500015.0, 5999975.0)
    # Row and column.
    assert (int(rows), int(cols)) == (2, 1)


# Samples at points; NaN outside the raster.
def test_raster_sample() -> None:
    # Reference raster.
    raster = ramp()
    # One point in pixel (1, 2) and one outside.
    values = raster.sample([500025.0, 400000.0], [5999985.0, 0.0])
    # Value 1 * 4 + 2 = 6 and NaN.
    assert values[0, 0] == 6.0 and np.isnan(values[1, 0])


# Tiles of a 256 x 256 raster with 128-pixel tiles form a 2 x 2 grid.
def test_raster_tiles(sample_raster_data: NDArray[np.floating[Any]]) -> None:
    # Raster of the fixture.
    raster = Raster.from_array(sample_raster_data)
    # Every tile.
    tiles = list(raster.tiles(tile_size=128))
    # Four tiles.
    assert [(r, c) for r, c, _ in tiles] == [(0, 0), (0, 128), (128, 0), (128, 128)]
    # Full-size tiles.
    assert all(t.shape == (3, 128, 128) for _, _, t in tiles)


# Tile starts advance by size - overlap and the last tile ends at the edge.
def test_tile_offsets() -> None:
    # Exact fit: 0, 3, 6 end at 4, 7, 10.
    assert tile_offsets(10, 4, 1) == [0, 3, 6]
    # One pixel left over: an extra tile flush with the edge.
    assert tile_offsets(11, 4, 1) == [0, 3, 6, 7]
    # Rasters smaller than a tile give one tile.
    assert tile_offsets(3, 4) == [0]
    # The overlap must be smaller than the tile.
    with pytest.raises(ValueError, match="overlap"):
        # Invalid overlap.
        tile_offsets(10, 4, 4)


# Averaging 2 x 2 blocks halves the size and doubles the pixel size.
def test_raster_resample_average_and_nearest() -> None:
    # Reference raster.
    raster = ramp()
    # Block averages.
    down = raster.resample(scale=0.5, method="average")
    # Means of the four 2 x 2 blocks of 0..15.
    assert down.data is not None
    # Expected block means.
    assert np.allclose(down.data[0], [[2.5, 4.5], [10.5, 12.5]])
    # 20 m pixels at the same origin.
    assert down.transform == (20.0, 0.0, 500000.0, 0.0, -20.0, 6000000.0)
    # Nearest neighbour upsampling repeats every pixel twice.
    up = raster.resample(scale=2.0, method="nearest")
    # Every value repeated in a 2 x 2 block.
    expected = np.kron(np.arange(16.0).reshape(4, 4), np.ones((2, 2)))
    # Same values.
    assert up.data is not None and np.array_equal(up.data[0], expected)
    # Target resolution gives the same result as the scale factor.
    assert raster.resample(resolution=20.0, method="average").shape == (1, 2, 2)


# Resampling by half gives half the size.
def test_raster_resample_size(sample_raster_data: NDArray[np.floating[Any]]) -> None:
    # Bilinear downsampling.
    resampled = Raster.from_array(sample_raster_data).resample(scale=0.5)
    # Half the size.
    assert (resampled.height, resampled.width) == (128, 128)


# Cropping by bounds selects whole pixels and moves the origin.
def test_raster_crop() -> None:
    # Pixels in rows 1-2 and columns 1-2.
    cropped = ramp().crop((500010.0, 5999970.0, 500030.0, 5999990.0))
    # Values 5, 6, 9, 10.
    assert cropped.data is not None and np.array_equal(cropped.data[0], [[5, 6], [9, 10]])
    # New origin.
    assert cropped.transform == (10.0, 0.0, 500010.0, 0.0, -10.0, 5999990.0)
    # Bounds outside the raster are rejected.
    with pytest.raises(ValueError, match="do not overlap"):
        # Disjoint bounds.
        ramp().crop((0.0, 0.0, 10.0, 10.0))


# Clipping by a polygon keeps the pixels whose centres are inside.
def test_raster_clip() -> None:
    # Imported here: Shapely is only needed for vectors.
    from shapely.geometry import box

    # Top-left 2 x 2 block.
    area = box(500000.0, 5999980.0, 500020.0, 6000000.0)
    # Clip without cropping.
    clipped = ramp().clip(area, crop=False)
    # Pixel values.
    assert clipped.data is not None
    # Four pixels remain, the rest are NaN.
    assert np.isfinite(clipped.data).sum() == 4
    # The kept values.
    assert np.array_equal(clipped.data[0, :2, :2], [[0, 1], [4, 5]])
    # Clipping with cropping returns only the block.
    assert ramp().clip(area).shape == (1, 2, 2)
    # Integer rasters without no-data need a fill value.
    with pytest.raises(ValueError, match="fill"):
        # Integer raster.
        ramp().astype(np.int16).clip(area, crop=False)


# Band math evaluates arithmetic safely; zero denominators give NaN.
def test_band_math() -> None:
    # Two bands.
    data = np.array([[[1, 2], [3, 0]], [[3, 2], [1, 0]]], dtype=np.float32)
    # Normalised difference of band 2 and band 1.
    result = Raster.from_array(data).band_math("(b2 - b1) / (b2 + b1)")
    # Hand-computed values: 2/4, 0/4, -2/4 and 0/0.
    assert result.data is not None
    # Compare with NaN for 0/0.
    assert np.allclose(result.data[0], [[0.5, 0.0], [-0.5, np.nan]], equal_nan=True)
    # Functions and comparisons.
    assert float(evaluate_expression("where(b1 > 1, sqrt(b1), -1)", {"b1": np.array(4.0)})) == 2.0
    # Chained comparisons.
    assert bool(evaluate_expression("0 < b1 < 1", {"b1": np.array(0.5)}))
    # Attribute access and unknown functions are rejected.
    for expression in ("b1.real", "__import__('os')", "open('x')"):
        # Unsupported syntax.
        with pytest.raises(ValueError):
            # Evaluate.
            evaluate_expression(expression, {"b1": np.array(1.0)})


# Statistics ignore no-data values.
def test_raster_statistics_and_histogram() -> None:
    # Values 1, 2, 3 and one no-data pixel.
    raster = Raster.from_array(np.array([[1.0, 2.0], [3.0, -9999.0]]), nodata=-9999.0)
    # Statistics of the single band.
    stats = raster.statistics()[0]
    # Count, extremes and mean.
    assert (stats["count"], stats["min"], stats["max"], stats["mean"]) == (3, 1, 3, 2)
    # Population standard deviation sqrt(2 / 3).
    assert math.isclose(stats["std"], math.sqrt(2.0 / 3.0))
    # Median.
    assert stats["p50"] == 2.0
    # Histogram with three unit bins.
    counts, edges = raster.histogram(bins=3, value_range=(0.5, 3.5))
    # One value per bin.
    assert counts.tolist() == [1, 1, 1] and np.allclose(edges, [0.5, 1.5, 2.5, 3.5])
    # The valid mask excludes the no-data pixel.
    assert raster.valid_mask().tolist() == [[True, True], [True, False]]


# Stacking, band selection and grid comparison.
def test_raster_stack_select_and_grid() -> None:
    # Two single-band rasters on the same grid.
    stacked = Raster.stack([ramp(), ramp().apply(lambda d: d * 2)])
    # Two bands.
    assert stacked.count == 2 and stacked.same_grid(ramp())
    # Second band only.
    second = stacked.select_bands([2])
    # Values doubled.
    assert second.data is not None and second.data[0, 3, 3] == 30.0
    # Rasters on different grids cannot be stacked.
    with pytest.raises(ValueError, match="share"):
        # Different transform.
        Raster.stack([ramp(), Raster.from_array(np.zeros((4, 4), dtype=np.float32))])


# Masking writes the no-data value.
def test_raster_mask_and_set_nodata() -> None:
    # Mask of the first row.
    mask = np.zeros((4, 4), dtype=bool)
    # First row.
    mask[0] = True
    # Masked integer raster with an explicit fill.
    masked = ramp().astype(np.int16).mask(mask, fill=-1)
    # First row filled and no-data recorded.
    assert masked.data is not None and masked.data[0, 0].tolist() == [-1] * 4
    # The fill value is the new no-data value.
    assert masked.nodata == -1
    # set_nodata changes only the metadata.
    assert ramp().set_nodata(0.0).nodata == 0.0


# GeoTIFF round trip keeps values, CRS, transform, no-data and tags.
def test_raster_file_round_trip(tmp_path: Path) -> None:
    # Reference raster with a tag.
    raster = Raster.from_array(ramp().data, "EPSG:32635", UTM, -1.0, {"sensor": "test"})  # type: ignore[arg-type]
    # Write the file.
    path = raster.to_file(tmp_path / "ramp.tif")
    # Read it back.
    back = Raster.from_file(path)
    # Same values and georeferencing.
    assert back.data is not None and np.array_equal(back.data, raster.data)  # type: ignore[arg-type]
    # CRS, transform and no-data.
    assert (back.crs, back.transform, back.nodata) == ("EPSG:32635", UTM, -1.0)
    # Tags.
    assert back.metadata is not None and back.metadata.tags["sensor"] == "test"
    # Lazy rasters read on demand.
    lazy = Raster.from_file(path, lazy=True)
    # No data yet, but the shape is known.
    assert lazy.data is None and lazy.shape == (1, 4, 4)
    # Window read from the file.
    assert lazy.read_window(1, 1, 2, 2)[0].tolist() == [[5, 6], [9, 10]]
    # Load the data.
    assert lazy.load().data is not None
    # Windowed read with the transform of the window.
    window = Raster.from_file(path, window=(1, 1, 2, 2))
    # Origin of the window.
    assert window.transform == (10.0, 0.0, 500010.0, 0.0, -10.0, 5999990.0)
    # Windows outside the file are rejected.
    with pytest.raises(ValueError, match="outside"):
        # Too large.
        Raster.from_file(path, window=(0, 0, 5, 5))


# Cloud Optimized GeoTIFFs have the COG layout and overviews.
def test_raster_to_cog(tmp_path: Path) -> None:
    # Imported here: rasterio reads the result.
    import rasterio

    # 512 x 512 raster.
    raster = Raster.from_array(np.ones((512, 512), dtype=np.float32), "EPSG:32635", UTM)
    # Write the COG with 256-pixel tiles.
    path = raster.to_cog(tmp_path / "cog.tif", blocksize=256)
    # Inspect the file.
    with rasterio.open(path) as src:
        # COG layout reported by GDAL.
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        # One overview at half resolution.
        assert src.overviews(1) == [2]


# Reprojection from UTM to geographic coordinates.
def test_raster_reproject() -> None:
    # Imported here: pyproj gives the reference point.
    from pyproj import Transformer

    # Reference raster reprojected to WGS 84.
    geographic = ramp().reproject("EPSG:4326", method="nearest")
    # Target CRS.
    assert geographic.crs == "EPSG:4326"
    # Centre of the source raster in longitude and latitude.
    lon, lat = Transformer.from_crs("EPSG:32635", "EPSG:4326", always_xy=True).transform(
        500020.0,  # Easting of the centre.
        5999980.0,  # Northing of the centre.
    )  # End of the transformation.
    # The centre lies inside the output bounds.
    west, south, east, north = geographic.bounds
    # Containment.
    assert west < lon < east and south < lat < north
    # Zone 35N has its central meridian at 27 degrees east.
    assert abs(lon - 27.0) < 0.01


# Warping onto another grid takes its size and transform.
def test_raster_match() -> None:
    # 2 x 2 grid of 20 m pixels.
    target = ramp().resample(scale=0.5, method="average")
    # Warp the reference onto it.
    matched = ramp().match(target, method="average")
    # Same grid and values.
    assert matched.same_grid(target) and np.allclose(matched.data, target.data)  # type: ignore[arg-type]


# Point buffers are regular 64-gons with the given radius.
def test_vector_from_wkt_and_buffer() -> None:
    # Unit square.
    vector = Vector.from_wkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    # One feature in WGS 84.
    assert vector.feature_count == 1 and vector.crs == "EPSG:4326"
    # Geometry type recorded.
    assert vector.metadata is not None and vector.metadata.geometry_type is GeometryType.POLYGON
    # Point in a projected CRS buffered by one unit with 16 segments per quarter.
    buffered = Vector.from_wkt("POINT(0 0)", crs="EPSG:3857").buffer(1.0)
    # Area of the inscribed regular 64-gon: 32 sin(pi / 32).
    assert math.isclose(float(buffered.area()[0]), 32 * math.sin(math.pi / 32), rel_tol=1e-9)


# Geodesic area of a 1 km UTM square on the central meridian.
def test_vector_geodesic_area_and_length() -> None:
    # Square of 1000 m centred on the central meridian of zone 35N.
    square = Vector.from_wkt(
        "POLYGON((499500 6000000, 500500 6000000, 500500 6001000, 499500 6001000, 499500 6000000))",
        crs="EPSG:32635",  # UTM zone 35N.
    )  # End of the square.
    # Planar area in square metres.
    assert math.isclose(float(square.area()[0]), 1.0e6)
    # On the central meridian the UTM scale is 0.9996, so the true area is 1e6 / 0.9996^2.
    assert math.isclose(float(square.area(geodesic=True)[0]), 1.0e6 / 0.9996**2, rel_tol=1e-5)
    # One degree of the equator: a * pi / 180 on WGS 84.
    equator = Vector.from_wkt("LINESTRING(0 0, 1 0)")
    # Geodesic length by default in a geographic CRS.
    assert math.isclose(float(equator.length()[0]), 6378137.0 * math.pi / 180.0, rel_tol=1e-9)


# Buffers in metres work in geographic coordinates.
def test_vector_buffer_metres() -> None:
    # Point in Helsinki.
    point = Vector.from_wkt("POINT(24.94 60.17)")
    # 1 km buffer.
    buffered = point.buffer_metres(1000.0)
    # Geodesic area close to the 64-gon of radius 1000 m.
    expected = 32 * math.sin(math.pi / 32) * 1.0e6
    # UTM distortion near the zone edge stays below one percent.
    assert math.isclose(float(buffered.area()[0]), expected, rel_tol=1e-2)
    # The result stays in the original CRS.
    assert buffered.crs == "EPSG:4326"


# GeoJSON round trip, attribute filters and features.
def test_vector_geojson_filter_and_features() -> None:
    # Two points with a class attribute.
    vector = Vector.from_wkt(
        ["POINT(0 0)", "POINT(5 5)"],  # Geometries.
        properties={"kind": ["a", "b"]},  # Attributes.
    )  # End of the vector.
    # GeoJSON round trip.
    back = Vector.from_geojson(vector.to_geojson())
    # Same WKT.
    assert back.to_wkt() == ["POINT (0 0)", "POINT (5 5)"]
    # Attribute query.
    assert vector.filter("kind == 'b'").to_wkt() == ["POINT (5 5)"]
    # Spatial filter by bounds.
    assert vector.filter(bounds=(-1, -1, 1, 1)).feature_count == 1
    # Features with properties.
    assert [f["properties"]["kind"] for f in vector.features()] == ["a", "b"]


# Overlay, clipping, dissolving, exploding and repairing.
def test_vector_geometry_operations() -> None:
    # Two overlapping squares of area 4.
    squares = Vector.from_wkt(
        ["POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))", "POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))"],  # Squares.
        crs="EPSG:3857",  # Planar coordinates.
        properties={"group": [1, 1]},  # Same group.
    )  # End of the squares.
    # Dissolved union of area 4 + 4 - 1.
    assert math.isclose(float(squares.dissolve(by="group").area()[0]), 7.0)
    # Union geometry.
    assert math.isclose(squares.union().area, 7.0)
    # Clip to the unit square around (1.5, 1.5).
    clipped = squares.clip((1.0, 1.0, 2.0, 2.0))
    # Each square contributes the unit square.
    assert np.allclose(clipped.area(), [1.0, 1.0])
    # Multi-part geometry split into parts.
    multi = Vector.from_wkt("MULTIPOINT((0 0), (1 1))", crs="EPSG:3857")
    # Two parts.
    assert multi.explode().feature_count == 2
    # Self-intersecting bow tie.
    bowtie = Vector.from_wkt("POLYGON((0 0, 2 2, 2 0, 0 2, 0 0))", crs="EPSG:3857")
    # Invalid before, valid after repair.
    assert not bowtie.is_valid()[0] and bowtie.make_valid().is_valid()[0]


# Spatial join attaches the attributes of intersecting features.
def test_vector_spatial_join() -> None:
    # Two zones.
    zones = Vector.from_wkt(
        ["POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))", "POLYGON((2 0, 3 0, 3 1, 2 1, 2 0))"],  # Zones.
        crs="EPSG:3857",  # Planar coordinates.
        properties={"zone": ["west", "east"]},  # Names.
    )  # End of the zones.
    # One point in the east zone.
    points = Vector.from_wkt("POINT(2.5 0.5)", crs="EPSG:3857")
    # Join the zone names.
    joined = points.spatial_join(zones)
    # The east zone.
    assert joined.data is not None and joined.data["zone"].tolist() == ["east"]


# Rasterising a square and polygonising it back.
def test_vector_rasterize_and_polygonize() -> None:
    # Square over rows 1-2 and columns 1-2 of the reference grid.
    square = Vector.from_wkt(
        "POLYGON((500010 5999970, 500030 5999970, 500030 5999990, 500010 5999990, 500010 5999970))",
        crs="EPSG:32635",  # Same CRS as the grid.
    )  # End of the square.
    # Burn the square.
    burned = square.rasterize(ramp(), dtype="uint8")
    # Four pixels burned with value 1.
    assert burned.data is not None and burned.data.sum() == 4
    # Exactly the 2 x 2 block.
    assert burned.data[0, 1:3, 1:3].tolist() == [[1, 1], [1, 1]]
    # Polygons of the regions of equal value.
    regions = Vector.from_raster(burned)
    # Region of value 1.
    ones = regions.filter("value == 1")
    # Area of four 10 m pixels.
    assert math.isclose(float(ones.area()[0]), 400.0)


# Tile grids cover a raster.
def test_tile_grid(sample_raster_data: NDArray[np.floating[Any]]) -> None:
    # Grid of 64-pixel tiles.
    grid = TileGrid.from_raster(Raster.from_array(sample_raster_data), tile_size=64)
    # Four by four tiles.
    assert (grid.num_rows, grid.num_cols, grid.total_tiles) == (4, 4, 16)
    # Offset of the last tile.
    assert grid.tile_offset(TileIndex(3, 3)) == (192, 192)
    # Indices outside the grid are rejected.
    with pytest.raises(IndexError):
        # Row 4 does not exist.
        grid.tile_offset(TileIndex(4, 0))


# Tiles carry the transform and bounds of their window.
def test_tile_georeference() -> None:
    # Grid of 2-pixel tiles over the reference raster.
    grid = TileGrid.from_raster(ramp(), tile_size=2)
    # Bottom-right tile.
    tile = grid.get_tile(ramp(), TileIndex(1, 1))
    # Values 10, 11, 14, 15.
    assert tile.data[0].tolist() == [[10, 11], [14, 15]]
    # Bounds of the window.
    assert tile.bounds == (500020.0, 5999960.0, 500040.0, 5999980.0)
    # Replacing data of another size is rejected.
    with pytest.raises(ValueError):
        # Wrong size.
        tile.with_data(np.zeros((1, 3, 3)))


# The linear ramp of a 4-pixel tile with overlap 2.
def test_blend_weights() -> None:
    # Distances to the nearer edge divided by the overlap, capped at one.
    ramp_1d = np.array([0.25, 0.75, 0.75, 0.25])
    # Product of the row and column ramps.
    assert np.allclose(blend_weights(4, 4, 2), np.outer(ramp_1d, ramp_1d))
    # Without overlap the weights are one.
    assert np.array_equal(blend_weights(2, 3, 0), np.ones((2, 3)))


# Mosaics of overlapping tiles.
def test_mosaic_blending() -> None:
    # Grid of a 4 x 6 raster with 4-pixel tiles and overlap 2: column starts 0 and 2.
    grid = TileGrid.for_shape(4, 6, tile_size=4, overlap=2)
    # Tile of zeros at column 0 and tile of ones at column 2.
    tiles = [
        Tile(TileIndex(0, 0), np.zeros((1, 4, 4)), offset=(0, 0)),  # Zeros.
        Tile(TileIndex(0, 1), np.ones((1, 4, 4)), offset=(0, 2)),  # Ones.
    ]  # End of the tiles.
    # Later tiles overwrite earlier ones.
    assert grid.mosaic(tiles)[0, 0].tolist() == [0, 0, 1, 1, 1, 1]
    # Equal weights average the overlap.
    assert np.allclose(grid.mosaic(tiles, blend="average")[0, 0], [0, 0, 0.5, 0.5, 1, 1])
    # Linear weights: column 2 has 0.75 of zeros and 0.25 of ones, column 3 the reverse.
    linear = grid.mosaic(tiles, dtype=np.float64, blend="linear")[0, 0]
    # Hand-computed blend.
    assert np.allclose(linear, [0, 0, 0.25, 0.75, 1, 1])


# Tiling and linear blending reproduce the raster exactly.
def test_mosaic_round_trip(sample_raster_data: NDArray[np.floating[Any]]) -> None:
    # Raster of the fixture.
    raster = Raster.from_array(sample_raster_data)
    # Overlapping tiles.
    grid = TileGrid.from_raster(raster, tile_size=100, overlap=20)
    # Blend the unmodified tiles.
    mosaic = grid.mosaic(grid.tiles(raster), blend="linear")
    # Same values.
    assert np.allclose(mosaic, sample_raster_data, atol=1e-6)


# XYZ tile arithmetic of the Web Mercator grid.
def test_xyz_tiles() -> None:
    # The world is one tile at zoom 0.
    assert xyz_tile(10.0, 20.0, 0) == TileIndex(0, 0, 0)
    # The origin is in the south-east quadrant at zoom 1 (x = 1, y = 1).
    assert xyz_tile(0.0, -0.1, 1) == TileIndex(row=1, col=1, level=1)
    # Bounds of the world tile.
    west, south, east, north = xyz_bounds(TileIndex(0, 0, 0))
    # Longitudes span the globe; latitudes stop at 85.0511 degrees.
    assert (west, east) == (-180.0, 180.0) and math.isclose(north, 85.0511287798066)
    # South bound mirrors the north bound.
    assert math.isclose(south, -north)
    # Web Mercator bounds of the world tile.
    assert xyz_bounds_mercator(TileIndex(0, 0, 0)) == (
        -20037508.342789244,  # Left.
        -20037508.342789244,  # Bottom.
        20037508.342789244,  # Right.
        20037508.342789244,  # Top.
    )  # End of the bounds.
    # The centre of a tile lies in that tile.
    tile = TileIndex(row=297, col=582, level=10)
    # Bounds of the tile.
    w, s, e, n = xyz_bounds(tile)
    # Round trip.
    assert xyz_tile((w + e) / 2, (s + n) / 2, 10) == tile


# Spectral index values at the reference pixel, from the formulas by hand.
@pytest.mark.parametrize(
    ("name", "expected"),  # Index and value.
    [  # One case per index.
        ("NDVI", 0.4 / 0.6),  # (0.5 - 0.1) / (0.5 + 0.1).
        ("GNDVI", 0.42 / 0.58),  # (0.5 - 0.08) / (0.5 + 0.08).
        ("NDRE", 0.3 / 0.7),  # (0.5 - 0.2) / (0.5 + 0.2).
        ("EVI", 2.5 * 0.4 / 1.725),  # 2.5 * 0.4 / (0.5 + 0.6 - 0.375 + 1).
        ("EVI2", 2.5 * 0.4 / 1.74),  # 2.5 * 0.4 / (0.5 + 0.24 + 1).
        ("SAVI", 1.5 * 0.4 / 1.1),  # 1.5 * 0.4 / (0.5 + 0.1 + 0.5).
        ("MSAVI", (2.0 - math.sqrt(0.8)) / 2.0),  # (2 - sqrt(4 - 3.2)) / 2.
        ("OSAVI", 0.4 / 0.76),  # 0.4 / (0.6 + 0.16).
        ("ARVI", 0.35 / 0.65),  # RB = 0.1 - (0.05 - 0.1) = 0.15.
        ("VARI", -0.02 / 0.13),  # (0.08 - 0.1) / (0.08 + 0.1 - 0.05).
        ("SR", 5.0),  # 0.5 / 0.1.
        ("WDRVI", -0.05 / 0.15),  # (0.05 - 0.1) / (0.05 + 0.1).
        ("CIgreen", 5.25),  # 0.5 / 0.08 - 1.
        ("CIre", 1.5),  # 0.5 / 0.2 - 1.
        ("NDWI", -0.42 / 0.58),  # (0.08 - 0.5) / (0.08 + 0.5).
        ("MNDWI", -0.12 / 0.28),  # (0.08 - 0.2) / (0.08 + 0.2).
        ("NDMI", 0.3 / 0.7),  # (0.5 - 0.2) / (0.5 + 0.2).
        ("AWEInsh", -0.88),  # 4 * (-0.12) - (0.125 + 0.275).
        ("AWEIsh", -0.825),  # 0.05 + 0.2 - 1.05 - 0.025.
        ("NDTI", 0.02 / 0.18),  # (0.1 - 0.08) / (0.1 + 0.08).
        ("NDCI", 0.1 / 0.3),  # (0.2 - 0.1) / (0.2 + 0.1).
        ("NBR", 0.4 / 0.6),  # (0.5 - 0.1) / (0.5 + 0.1).
        ("NBR2", 0.1 / 0.3),  # (0.2 - 0.1) / (0.2 + 0.1).
        ("NDBI", -0.3 / 0.7),  # (0.2 - 0.5) / (0.2 + 0.5).
        ("BSI", -0.25 / 0.85),  # (0.3 - 0.55) / (0.3 + 0.55).
        ("NDSI", -0.12 / 0.28),  # (0.08 - 0.2) / (0.08 + 0.2).
        ("MSI", 0.4),  # 0.2 / 0.5.
    ],  # End of the cases.
)  # End of the parametrisation.
def test_index_values(name: str, expected: float) -> None:
    # Reference pixel as 1 x 1 arrays.
    bands = {k: np.array([[v]]) for k, v in PIXEL.items()}
    # Index value.
    value = compute_index(name, bands)
    # Shape of the input and the hand-computed value.
    assert value.shape == (1, 1) and math.isclose(float(value[0, 0]), expected, rel_tol=1e-12)


# Every registered index has a test value above.
def test_every_index_is_tested() -> None:
    # Names of the parametrised test.
    tested = {"NDVI", "GNDVI", "NDRE", "EVI", "EVI2", "SAVI", "MSAVI", "OSAVI", "ARVI", "VARI"}
    # Remaining names.
    tested |= {"SR", "WDRVI", "CIgreen", "CIre", "NDWI", "MNDWI", "NDMI", "AWEInsh", "AWEIsh"}
    # Last names.
    tested |= {"NDTI", "NDCI", "NBR", "NBR2", "NDBI", "BSI", "NDSI", "MSI"}
    # Same set as the registry.
    assert set(IndexRegistry.list_all()) == tested


# Normalised differences stay in [-1, 1] for random reflectances.
def test_index_value_ranges(sample_bands: dict[str, NDArray[np.floating[Any]]]) -> None:
    # Indices with finite limits whose bands the fixture provides.
    for name in IndexRegistry.available_for(sample_bands):
        # Index definition.
        index = IndexRegistry.get(name)
        # Limits of the index.
        low, high = index.value_range  # type: ignore[union-attr]
        # Values on the fixture.
        values = compute_index(name, sample_bands)
        # Finite values lie within the limits.
        assert np.nanmin(values) >= low - 1e-12 and np.nanmax(values) <= high + 1e-12


# Shapes of the fixture indices and unknown names.
def test_index_shapes_and_unknown(sample_bands: dict[str, NDArray[np.floating[Any]]]) -> None:
    # Three indices at once.
    values = compute_indices(["NDVI", "NDWI", "EVI"], sample_bands)
    # Shape of the bands.
    assert all(v.shape == (256, 256) for v in values.values())
    # Unknown names are rejected.
    with pytest.raises(ValueError, match="Unknown index"):
        # Invalid name.
        compute_index("INVALID", sample_bands)
    # Missing bands are reported.
    with pytest.raises(ValueError, match="Missing bands"):
        # No SWIR2 band.
        compute_index("NBR", {"NIR": np.ones(2)})


# Zero denominators, NaN inputs and no-data values give NaN.
def test_index_invalid_pixels() -> None:
    # Zero sum, NaN, no-data and a valid pixel.
    nir = np.array([0.0, np.nan, -9999.0, 0.3])
    # Red band.
    red = np.array([0.0, 0.1, 0.1, 0.1])
    # NDVI with a no-data value.
    values = compute_index("NDVI", {"NIR": nir, "RED": red}, nodata=-9999.0)
    # Three NaN and 0.2 / 0.4.
    assert np.allclose(values, [np.nan, np.nan, np.nan, 0.5], equal_nan=True)


# Product band names of Sentinel-2 and Landsat are resolved.
def test_index_sensor_aliases() -> None:
    # Sentinel-2 names.
    s2 = compute_index("NDVI", {"B04": np.array([0.1]), "B8": np.array([0.5])}, sensor="S2")
    # Landsat Collection 2 names.
    l8 = compute_index("ndvi", {"SR_B4": np.array([0.1]), "SR_B5": np.array([0.5])}, sensor="l8")
    # Same value as the common names.
    assert np.allclose(s2, 0.4 / 0.6) and np.allclose(l8, 0.4 / 0.6)
    # Unknown sensors are rejected.
    with pytest.raises(ValueError, match="unknown sensor"):
        # Invalid sensor.
        compute_index("NDVI", {"B4": np.ones(1)}, sensor="nosuchsat")


# Parameters override the defaults; unknown parameters are rejected.
def test_index_parameters() -> None:
    # Bands of the reference pixel.
    bands = {"NIR": np.array([0.5]), "RED": np.array([0.1])}
    # SAVI with L = 0 is NDVI.
    assert np.allclose(compute_index("SAVI", bands, L=0.0), compute_index("NDVI", bands))
    # Unknown parameters.
    with pytest.raises(ValueError, match="no parameters"):
        # NDVI has none.
        compute_index("NDVI", bands, L=0.5)


# Registry look-up is case-insensitive and grouped by category.
def test_index_registry() -> None:
    # Lower-case name.
    assert IndexRegistry.get("cigreen") is IndexRegistry.get("CIgreen")
    # Burn indices.
    assert {i.name for i in IndexRegistry.by_category(IndexCategory.BURN)} == {"NBR", "NBR2"}
    # Indices computable from red and near infrared.
    assert set(IndexRegistry.available_for(["NIR", "RED"])) == {
        "NDVI",  # Normalised difference.
        "EVI2",  # Two-band EVI.
        "SAVI",  # Soil adjusted.
        "MSAVI",  # Modified soil adjusted.
        "OSAVI",  # Optimised soil adjusted.
        "SR",  # Simple ratio.
        "WDRVI",  # Wide dynamic range.
    }  # End of the names.


# dNBR and the burn severity classes of Key and Benson (2006).
def test_burn_severity() -> None:
    # NBR before 0.6 and after 0.0 give dNBR 0.6.
    pre = {"NIR": np.array([0.8]), "SWIR2": np.array([0.2])}
    # After the fire.
    post = {"NIR": np.array([0.3]), "SWIR2": np.array([0.3])}
    # dNBR.
    assert np.allclose(dnbr(pre, post), 0.6)
    # One value per class, a limit value and NaN.
    values = np.array([-0.3, -0.25, 0.0, 0.1, 0.3, 0.5, 0.7, np.nan])
    # Expected classes; limits belong to the upper class.
    assert classify_burn_severity(values).tolist() == [0, 1, 2, 3, 4, 5, 6, 255]
    # Seven class names.
    assert len(BURN_SEVERITY_CLASSES) == 7 and BURN_SEVERITY_CLASSES[6] == "high severity"


# Sentinel-2 and Landsat band tables.
def test_sensor_bands() -> None:
    # Sentinel-2 MSI.
    s2 = get_sensor("sentinel-2")
    # Red band by product name and by common name.
    assert s2 is not None and s2.band("B04") is s2.band("red") is s2.band("B4")
    # Centre, width and pixel of B04 (Sentinel-2A).
    red = s2.band("B04")
    # Values of the band table.
    assert (red.center_nm, red.bandwidth_nm, red.resolution_m) == (664.6, 31.0, 10.0)
    # Limits from the centre and width.
    assert s2.get_band_wavelength("B04") == (649.1, 680.1)
    # Ten-metre bands.
    assert s2.bands_at_resolution(10.0) == ["B02", "B03", "B04", "B08"]
    # Landsat 8 NIR from its limits 850-880 nm.
    l8 = get_sensor("landsat8")
    # Collection 2 alias.
    assert l8 is not None and l8.band("SR_B5").center_nm == 865.0
    # Band construction checks the limits.
    with pytest.raises(ValueError):
        # Upper below lower.
        band_from_limits("X", "X", 500.0, 400.0, 30.0)


# Sentinel-1 radar wavelength and modes.
def test_sensor_sar() -> None:
    # Sentinel-1 C-SAR.
    s1 = get_sensor("s1")
    # Wavelength c / f at 5.405 GHz, about 5.55 cm.
    assert s1 is not None and math.isclose(s1.wavelength_m or 0.0, SPEED_OF_LIGHT / 5.405e9)
    # Interferometric Wide swath mode.
    assert (s1.mode("iw").swath_km, s1.mode("IW").azimuth_resolution_m) == (250.0, 20.0)
    # Optical sensors have no wavelength.
    assert get_sensor("s2").wavelength_m is None  # type: ignore[union-attr]


# Digital numbers to reflectance and temperature.
def test_radiometric_conversions() -> None:
    # Landsat Collection 2: 10000 DN is 0.275 - 0.2 = 0.075; 0 is no-data.
    values = to_reflectance(np.array([10000, 0]), LANDSAT_C2_SR_SCALE, LANDSAT_C2_SR_OFFSET, 0)
    # Hand-computed reflectance and NaN.
    assert np.allclose(values, [0.075, np.nan], equal_nan=True)
    # Sentinel-2 offset from baseline 04.00 on.
    assert (sentinel2_l2a_offset("04.00"), sentinel2_l2a_offset("N0213")) == (-0.1, 0.0)
    # TOA reflectance (2e-5 * 10000 - 0.1) / sin(30 degrees) = 0.2.
    toa = landsat_toa_reflectance(np.array([10000]), 2e-5, -0.1, 30.0)
    # Hand-computed value.
    assert np.allclose(toa, 0.2)
    # Radiance K1 / (e - 1) gives the temperature K2.
    k1, k2 = 774.8853, 1321.0789
    # Brightness temperature with ML = 1 and AL = 0.
    bt = landsat_brightness_temperature(np.array([k1 / (math.e - 1.0)]), 1.0, 0.0, k1, k2)
    # Equals K2.
    assert np.allclose(bt, k2)


# Scenes split, stack, harmonise and compute indices.
def test_scene() -> None:
    # Red and near-infrared bands on the reference grid.
    bands = np.stack([np.full((4, 4), 0.1), np.full((4, 4), 0.5)])
    # Two-band raster on the reference grid.
    stack = Raster.from_array(bands, "EPSG:32635", UTM)
    # Sentinel-2 scene with product band names.
    meta = SceneMetadata("S2A_TEST", "sentinel2_msi", cloud_cover=12.5, sun_elevation=40.0)
    # Split into bands.
    scene = Scene.from_raster(stack, ["B04", "B08"], meta)
    # Band names and shape.
    assert scene.bands == ["B04", "B08"] and scene.shape == (2, 4, 4)
    # NDVI through the sensor aliases.
    ndvi = scene.compute_index("NDVI")
    # Hand-computed value on the scene grid.
    assert np.allclose(ndvi.data, 0.4 / 0.6) and ndvi.same_grid(stack)  # type: ignore[arg-type]
    # Stack keeps the band names.
    assert scene.stack().metadata.tags["band_names"] == "B04,B08"  # type: ignore[union-attr]
    # A 20 m band.
    scene["B11"] = ramp().resample(scale=0.5, method="average")
    # Bands of different size cannot be stacked into an array.
    with pytest.raises(ValueError, match="harmonize"):
        # Mixed grids.
        scene.to_array()
    # Harmonised onto the 10 m grid.
    harmonized = scene.harmonize(method="nearest")
    # All bands on one grid.
    assert harmonized.is_aligned() and harmonized.to_array().shape == (3, 4, 4)
    # Sun zenith is the complement of the elevation.
    assert meta.sun_zenith == 50.0
    # Metadata round trip.
    assert SceneMetadata.from_dict(meta.to_dict()) == meta
    # Cloud cover is a percentage.
    with pytest.raises(ValueError, match="cloud_cover"):
        # Invalid value.
        SceneMetadata("x", "s2", cloud_cover=120.0)


# Products save, load with checksum verification, and describe themselves in STAC.
def test_product_save_load_and_stac(tmp_path: Path) -> None:
    # Index product on the reference grid.
    product = Product.create("ndvi-test", "index", ramp(), license="CC-BY-4.0")
    # Georeferencing from the raster.
    assert product.metadata is not None and product.metadata.resolution == 10.0
    # Record a processing step.
    product.add_processing_step("ndvi")
    # Save as COG and product.json.
    json_path = product.save(tmp_path / "ndvi")
    # Load back.
    loaded = Product.load(json_path.parent)
    # Same values and metadata.
    assert np.array_equal(loaded.data.data, ramp().data)  # type: ignore[union-attr]
    # Processing chain kept.
    assert loaded.metadata is not None and loaded.metadata.processing_chain == ["ndvi"]
    # STAC item of the product.
    item = product.to_stac_item(asset_href="ndvi-test.tif")
    # Projection extension with the EPSG code.
    assert item["properties"]["proj:epsg"] == 32635 and item["stac_version"] == "1.0.0"
    # Footprint near 27 degrees east, 54.1 degrees north.
    west, south, east, north = item["bbox"]
    # The west edge is the central meridian of zone 35N.
    assert math.isclose(west, 27.0, abs_tol=1e-9) and east > 27.0 and 54.0 < south < north < 54.2
    # Cloud Optimized GeoTIFF asset.
    assert item["assets"]["data"]["type"].endswith("profile=cloud-optimized")
    # A modified data file fails the checksum.
    (tmp_path / "ndvi" / "ndvi-test.tif").write_bytes(b"corrupted")
    # Loading detects it.
    with pytest.raises(ValueError, match="SHA-256"):
        # Verify the digest.
        Product.load(json_path)


# Array products and metadata validation.
def test_product_array_and_validation(tmp_path: Path) -> None:
    # Array product.
    product = Product.create("mask", ProductType.SEGMENTATION, np.eye(3, dtype=np.uint8))
    # Save and load.
    loaded = Product.load(product.save(tmp_path))
    # Same array.
    assert np.array_equal(loaded.data, np.eye(3))
    # Scores are fractions.
    with pytest.raises(ValueError, match="quality_score"):
        # Invalid score.
        Product.create("x", "raster", None, quality_score=1.5)


# Standardisation and binary segmentation of a callable model.
def test_model_wrapper_segmentation() -> None:
    # One-channel segmentation with unit statistics.
    config = ModelConfig(
        "m",  # Identifier.
        "Mask",  # Name.
        ModelTask.SEGMENTATION,  # Task.
        "custom",  # Framework.
        1,  # Input channels.
        1,  # Output channels.
        mean=(0.5,),  # Mean.
        std=(0.25,),  # Standard deviation.
    )  # End of the configuration.
    # Identity model: the output is the standardised input.
    wrapper = ModelWrapper(config, model=lambda batch: batch)
    # Values standardise to (x - 0.5) / 0.25 = -2, 0, 2.
    assert np.allclose(wrapper.preprocess(np.array([[[0.0, 0.5, 1.0]]])), [[[-2.0, 0.0, 2.0]]])
    # Thresholded at 0.5 after standardisation.
    assert wrapper.predict(np.array([[[0.0, 0.5, 1.0]]])).tolist() == [[0, 0, 1]]
    # Wrong channel counts are rejected.
    with pytest.raises(ValueError, match="channels"):
        # Two channels.
        wrapper.predict(np.zeros((2, 1, 1)))
    # Models without a model object cannot predict.
    with pytest.raises(RuntimeError, match="not loaded"):
        # No model.
        ModelWrapper(config).predict(np.zeros((1, 1, 1)))


# Multi-class maps, class probabilities and pixel estimators.
def test_model_wrapper_tasks() -> None:
    # Three-class segmentation without standardisation.
    seg = ModelConfig("s", "S", "segmentation", "custom", 3, 3, normalize=False)
    # Arg max over the channels of the input.
    labels = ModelWrapper(seg, model=lambda b: b).predict(np.array([[[1.0]], [[3.0]], [[2.0]]]))
    # Class 1 wins.
    assert labels.tolist() == [[1]]
    # Classification with logits 0 and ln 3.
    cls_config = ModelConfig(
        "c",  # Identifier.
        "C",  # Name.
        "classification",  # Task.
        "custom",  # Framework.
        1,  # Input channels.
        2,  # Classes.
        normalize=False,  # Raw inputs.
        logits=True,  # Scores are logits.
    )  # End of the configuration.
    # Model that returns two logits per image.
    wrapper = ModelWrapper(cls_config, model=lambda b: np.array([[0.0, math.log(3.0)]]))
    # Softmax probabilities 1/4 and 3/4.
    assert np.allclose(wrapper.predict(np.zeros((1, 2, 2))), [0.25, 0.75])

    # Estimator that labels pixels whose first feature exceeds 0.5.
    class Estimator:
        # scikit-learn style prediction on (samples, features).
        def predict(self, samples: NDArray[Any]) -> NDArray[Any]:
            # One label per sample.
            return (samples[:, 0] > 0.5).astype(int)

    # Pixel regression with the estimator.
    reg = ModelConfig("r", "R", "regression", "sklearn", 2, 1, normalize=False)
    # Pixels (0.2, x) and (0.9, x).
    out = ModelWrapper(reg, model=Estimator()).predict(np.array([[[0.2, 0.9]], [[0.0, 0.0]]]))
    # Labels in the layout (1, H, W).
    assert out.tolist() == [[[0, 1]]]


# TorchScript files load and run on the CPU.
def test_model_wrapper_torchscript(tmp_path: Path) -> None:
    # PyTorch is optional.
    torch = pytest.importorskip("torch")
    # Scripted identity network.
    path = tmp_path / "identity.pt"
    # Save it.
    torch.jit.save(torch.jit.script(torch.nn.Identity()), str(path))
    # Regression model loaded from the file.
    config = ModelConfig("t", "T", "regression", "pytorch", 3, 3)
    # Wrapper with the weights.
    wrapper = ModelWrapper(config, weights_path=path)
    # Input image.
    image = np.random.default_rng(0).random((3, 4, 4), dtype=np.float32)
    # The identity returns the standardised input.
    assert np.allclose(wrapper.predict(image), wrapper.preprocess(image), atol=1e-6)


# =============================================================================
# End of module tests/unit/test_core.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
