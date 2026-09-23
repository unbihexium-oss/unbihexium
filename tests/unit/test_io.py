# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_io.py
# Title       : Tests of GeoJSON, GeoTIFF, GeoParquet, STAC and Zarr I/O
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest; rasterio, zarr,
#               geopandas and pyproj tests are skipped when missing
# =============================================================================
#
# Abstract
# --------
# Round trips of every format with exact comparisons, georeferencing of
# windows and boxes (hand-computed affine coefficients), COG layout and
# overviews, GeoJSON validation, bounds, shoelace areas, the right-hand
# rule and UTM to WGS 84 reprojection (the central meridian of zone 32 is
# 9 degrees east), GeoParquet metadata and box filters, STAC parsing,
# offline search, static catalogues and API pagination with a fake
# transport (no network), and Zarr v2 and v3 stores with selections.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Optional dependency detection.
import importlib.util

# JSON files.
import json

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Test framework.
import pytest

# GeoJSON functions under test.
from unbihexium.io.geojson import (
    features_to_geojson,  # Collections.
    geojson_bounds,  # Bounds.
    geojson_crs,  # CRS member.
    geojson_problems,  # Validation problems.
    geometry_to_feature,  # Features.
    read_geojson,  # Reading.
    reproject_geojson,  # Reprojection.
    rewind,  # Right-hand rule.
    ring_area,  # Shoelace area.
    validate_geojson,  # Validation.
    write_geojson,  # Writing.
)  # End of the GeoJSON imports.

# STAC functions under test.
from unbihexium.io.stac import (
    STACClient,  # API client.
    STACCollection,  # Collections.
    STACItem,  # Items.
    bbox_intersects,  # Box test.
    filter_items,  # Offline search.
    parse_datetime,  # Times.
    parse_datetime_range,  # Intervals.
    read_stac_item,  # Item files.
    walk_catalog,  # Static catalogues.
)  # End of the STAC imports.

# Whether optional packages are installed.
HAS_RASTERIO = importlib.util.find_spec("rasterio") is not None
# zarr.
HAS_ZARR = importlib.util.find_spec("zarr") is not None
# geopandas.
HAS_GEOPANDAS = importlib.util.find_spec("geopandas") is not None
# pyproj.
HAS_PYPROJ = importlib.util.find_spec("pyproj") is not None

# Unit square polygon, counterclockwise.
SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}


# Minimal STAC item dictionary.
def make_item(
    item_id: str,  # Id.
    bbox: list[float],  # Box.
    when: str | None,  # Datetime or None.
    collection: str = "s2",  # Collection id.
    **props: Any,  # Extra properties.
) -> dict[str, Any]:  # Item dictionary.
    # One Sentinel-2 red band asset with a relative href.
    assets = {"B04": {"href": "B04.tif", "roles": ["data"], "type": "image/tiff"}}
    # Feature with the required members.
    return {
        "type": "Feature",  # GeoJSON type.
        "stac_version": "1.0.0",  # Version.
        "id": item_id,  # Id.
        "bbox": bbox,  # Box.
        "geometry": None,  # No geometry.
        "properties": {"datetime": when, **props},  # Properties.
        "collection": collection,  # Collection.
        "links": [],  # Links.
        "assets": assets,  # Assets.
    }  # End of the item.


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------


# Features and collections have the RFC 7946 members.
def test_feature_construction() -> None:
    # Point feature with properties and an id.
    feature = geometry_to_feature({"type": "Point", "coordinates": [1.0, 2.0]}, {"a": 1}, 7)
    # Members of the feature.
    assert feature == {
        "type": "Feature",  # Object type.
        "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},  # Geometry.
        "properties": {"a": 1},  # Attributes.
        "id": 7,  # Id.
    }  # End of the expected feature.
    # Missing properties become an empty object.
    assert geometry_to_feature(SQUARE)["properties"] == {}
    # Collection of two features without a CRS member.
    collection = features_to_geojson([feature, feature])
    # Type and size.
    assert collection["type"] == "FeatureCollection" and len(collection["features"]) == 2
    # No CRS member for the default CRS.
    assert "crs" not in collection
    # A projected CRS is stored as a legacy named member.
    assert geojson_crs(features_to_geojson([], crs="EPSG:32632")) == "EPSG:32632"


# Write and read keep the document, create directories and accept both orders.
def test_geojson_round_trip(tmp_path: Path) -> None:
    # Collection with a polygon and nested properties.
    doc = features_to_geojson([geometry_to_feature(SQUARE, {"area": 100.0, "tags": ["a", "b"]})])
    # Output in a new directory.
    path = tmp_path / "sub" / "square.geojson"
    # Data first, path second.
    write_geojson(doc, path)
    # The file equals the document.
    assert read_geojson(path) == doc
    # The earlier (path, data) order still works.
    write_geojson(tmp_path / "old.geojson", doc)
    # Same document.
    assert read_geojson(tmp_path / "old.geojson") == doc
    # Rounding to two decimals.
    point = {"type": "Point", "coordinates": [24.9384, 60.1699]}
    # Write with precision 2.
    write_geojson(point, tmp_path / "p.geojson", precision=2)
    # Coordinates of Helsinki rounded to 0.01 degree.
    assert read_geojson(tmp_path / "p.geojson")["coordinates"] == [24.94, 60.17]


# Invalid documents and files are rejected with clear errors.
def test_geojson_validation(tmp_path: Path) -> None:
    # Missing files.
    with pytest.raises(FileNotFoundError):
        # Nothing to read.
        read_geojson(tmp_path / "missing.geojson")
    # Open ring.
    open_ring = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}
    # Reported as not closed.
    assert any("closed" in p for p in geojson_problems(open_ring))
    # One-position line string.
    assert geojson_problems({"type": "LineString", "coordinates": [[0, 0]]})
    # Booleans are not coordinates.
    assert geojson_problems({"type": "Point", "coordinates": [True, 1]})
    # Unknown types.
    with pytest.raises(ValueError, match="unknown GeoJSON type"):
        # Validation fails.
        validate_geojson({"type": "Circle"})
    # Writing an invalid document leaves no file.
    with pytest.raises(ValueError):
        # Invalid feature collection.
        write_geojson({"type": "FeatureCollection", "features": [{}]}, tmp_path / "x.json")
    # No file was written.
    assert not (tmp_path / "x.json").exists()
    # Malformed JSON.
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    # Reported with the path.
    with pytest.raises(ValueError, match="not valid JSON"):
        # Parse the file.
        read_geojson(tmp_path / "bad.json")


# Bounds, shoelace areas and ring orientation.
def test_geojson_geometry_helpers() -> None:
    # Two features; the bounds cover both.
    point = geometry_to_feature({"type": "Point", "coordinates": [-3.0, 5.0]})
    # Collection of the point and the unit square.
    doc = features_to_geojson([point, geometry_to_feature(SQUARE)])
    # Minimum and maximum of all positions.
    assert geojson_bounds(doc) == (-3.0, 0.0, 1.0, 5.0)
    # Counterclockwise unit square: +1.
    assert ring_area(SQUARE["coordinates"][0]) == pytest.approx(1.0)
    # Clockwise 2 x 3 rectangle: -6.
    assert ring_area([[0, 0], [0, 3], [2, 3], [2, 0], [0, 0]]) == pytest.approx(-6.0)
    # Clockwise exterior and counterclockwise hole.
    polygon = {
        "type": "Polygon",  # Type.
        "coordinates": [  # Rings.
            [[0, 0], [0, 4], [4, 4], [4, 0], [0, 0]],  # Clockwise exterior, area -16.
            [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]],  # Counterclockwise hole, area +1.
        ],  # End of the rings.
    }  # End of the polygon.
    # Orientation after the right-hand rule.
    fixed = rewind(polygon)["coordinates"]
    # Exterior counterclockwise.
    assert ring_area(fixed[0]) == pytest.approx(16.0)
    # Hole clockwise.
    assert ring_area(fixed[1]) == pytest.approx(-1.0)
    # The input is unchanged.
    assert ring_area(polygon["coordinates"][0]) == pytest.approx(-16.0)


# UTM zone 32N to longitude and latitude.
@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
def test_geojson_reprojection() -> None:
    # Point on the central meridian (easting 500 000 m) at the equator.
    point = geometry_to_feature({"type": "Point", "coordinates": [500000.0, 0.0, 12.5]})
    # Collection with the OGC URN of UTM zone 32N.
    doc = features_to_geojson([point], crs="urn:ogc:def:crs:EPSG::32632")
    # The URN is understood.
    assert geojson_crs(doc) == "EPSG:32632"
    # Reproject to OGC:CRS84.
    out = reproject_geojson(doc)
    # Longitude 9 degrees east, latitude 0, elevation kept.
    assert out["features"][0]["geometry"]["coordinates"] == pytest.approx([9.0, 0.0, 12.5])
    # The CRS member is removed for the default CRS.
    assert "crs" not in out


# ---------------------------------------------------------------------------
# GeoTIFF
# ---------------------------------------------------------------------------


# Round trip with CRS, transform, no-data and band names.
@pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")
def test_geotiff_round_trip(tmp_path: Path) -> None:
    # GeoTIFF functions.
    from unbihexium.io.geotiff import geotiff_info, read_geotiff, write_geotiff

    # Three bands of distinct values.
    data = np.arange(3 * 32 * 32, dtype=np.float32).reshape(3, 32, 32)
    # Output file.
    path = tmp_path / "test.tif"
    # Write with a UTM grid of 10 m pixels.
    write_geotiff(
        data,  # Pixels.
        path,  # File.
        crs="EPSG:32632",  # UTM zone 32N.
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0),  # 10 m pixels.
        nodata=-9999.0,  # No-data value.
        descriptions=["red", "green", "blue"],  # Band names.
    )  # End of the write.
    # Read everything.
    loaded, meta = read_geotiff(path)
    # Values are exact.
    np.testing.assert_array_equal(loaded, data)
    # CRS as an authority string.
    assert meta["crs"] == "EPSG:32632"
    # Affine coefficients.
    assert meta["transform"] == (10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0)
    # Bounds: 32 pixels of 10 m.
    assert meta["bounds"] == (500000.0, 4999680.0, 500320.0, 5000000.0)
    # No-data and band names.
    assert meta["nodata"] == -9999.0 and meta["descriptions"] == ["red", "green", "blue"]
    # Float data uses the floating point predictor with DEFLATE.
    assert geotiff_info(path)["compression"] == "DEFLATE"
    # The earlier (path, data) order still works.
    write_geotiff(tmp_path / "old.tif", data[0])
    # Default transform: pixel coordinates with y = height - row.
    assert read_geotiff(tmp_path / "old.tif")[1]["transform"] == (1, 0, 0, 0, -1, 32)


# Pixel windows, map boxes and no-data masking.
@pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")
def test_geotiff_windows(tmp_path: Path) -> None:
    # GeoTIFF functions.
    from unbihexium.io.geotiff import read_geotiff, write_geotiff

    # One band of distinct values with a no-data pixel.
    data = np.arange(20 * 30, dtype=np.int16).reshape(1, 20, 30)
    # Pixel (0, 0) is missing.
    data[0, 0, 0] = -1
    # Output file.
    path = tmp_path / "w.tif"
    # Grid with 10 m pixels and origin (1000, 2000).
    write_geotiff(data, path, crs="EPSG:32632", transform=(10, 0, 1000, 0, -10, 2000), nodata=-1)
    # Rows 5 to 9, columns 10 to 14.
    part, meta = read_geotiff(path, window=(5, 10, 10, 15))
    # The values of the window.
    np.testing.assert_array_equal(part, data[:, 5:10, 10:15])
    # Origin of the window: x = 1000 + 10 * 10, y = 2000 - 10 * 5.
    assert meta["transform"] == (10, 0, 1100, 0, -10, 1950)
    # Box x in [1105, 1125], y in [1905, 1925]: columns 10..12, rows 7..9.
    boxed, bmeta = read_geotiff(path, bounds=(1105, 1905, 1125, 1925))
    # Pixels that touch the box.
    np.testing.assert_array_equal(boxed, data[:, 7:10, 10:13])
    # Origin of the box window.
    assert bmeta["transform"][2] == 1100 and bmeta["transform"][5] == 1930
    # Masked read: the no-data pixel is NaN.
    masked, _ = read_geotiff(path, masked=True)
    # Only that pixel.
    assert np.isnan(masked[0, 0, 0]) and np.isfinite(masked).sum() == 20 * 30 - 1
    # The native type is kept on request.
    assert read_geotiff(path, dtype=None)[0].dtype == np.int16
    # Windows outside the grid are errors.
    with pytest.raises(ValueError, match="outside"):
        # 21 rows do not exist.
        read_geotiff(path, window=(0, 21, 0, 5))


# COG layout, overviews and Raster conversion.
@pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")
def test_cog_and_overviews(tmp_path: Path) -> None:
    # GeoTIFF functions.
    from unbihexium.io.geotiff import (
        geotiff_info,  # Metadata.
        is_cog,  # Layout test.
        overview_factors,  # Factors.
        read_geotiff,  # Reading.
        read_raster,  # Raster reading.
        write_cog,  # COG writing.
        write_geotiff,  # Writing.
        write_raster,  # Raster writing.
    )  # End of the imports.

    # 600 x 500 pixels with 256 blocks: overviews 2 (300 x 250) and 4 (150 x 125).
    assert overview_factors(500, 600, 256) == [2, 4]
    # Small images need none.
    assert overview_factors(100, 100, 256) == []
    # Gradient image.
    data = np.add.outer(np.arange(600), np.arange(500)).astype(np.uint16)[None]
    # COG with default options.
    write_cog(data, tmp_path / "c.tif", crs="EPSG:3067", transform=(2, 0, 0, 0, -2, 1200))
    # Layout reported by GDAL.
    assert is_cog(tmp_path / "c.tif")
    # Overviews built by the COG driver.
    assert geotiff_info(tmp_path / "c.tif")["overviews"] == [2, 4]
    # The first overview has half the size and twice the pixel size.
    ov, meta = read_geotiff(tmp_path / "c.tif", overview_level=0)
    # Shape of the overview.
    assert ov.shape == (1, 300, 250) and meta["transform"][0] == 4.0
    # Plain GeoTIFF with automatic overviews is not a COG.
    write_geotiff(data, tmp_path / "g.tif", overviews="auto")
    # Tiled, with overviews, but not the COG layout.
    info = geotiff_info(tmp_path / "g.tif")
    # Check the flags.
    assert info["tiled"] and info["overviews"] == [2, 4] and not info["is_cog"]
    # Raster conversion keeps pixels and georeferencing.
    raster = read_raster(tmp_path / "c.tif")
    # Native type and transform.
    assert raster.data.dtype == np.uint16 and tuple(raster.metadata.transform)[:2] == (2.0, 0.0)
    # Write it back.
    write_raster(raster, tmp_path / "r.tif")
    # Same pixels.
    np.testing.assert_array_equal(read_geotiff(tmp_path / "r.tif", dtype=None)[0], data)


# ---------------------------------------------------------------------------
# GeoParquet
# ---------------------------------------------------------------------------


# Round trip, metadata, column selection and box filter.
@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
def test_geoparquet(tmp_path: Path) -> None:
    # Data frames and geometries.
    import geopandas as gpd

    # Points.
    from shapely.geometry import Point

    # GeoParquet functions.
    from unbihexium.io.parquet import (
        geojson_to_geoparquet,  # From GeoJSON.
        geoparquet_bounds,  # Bounds.
        geoparquet_metadata,  # Metadata.
        geoparquet_to_geojson,  # To GeoJSON.
        read_geoparquet,  # Reading.
        write_geoparquet,  # Writing.
    )  # End of the imports.

    # Two points with attributes.
    gdf = gpd.GeoDataFrame(
        {"name": ["A", "B"], "value": [1, 2]},  # Attributes.
        geometry=[Point(0, 0), Point(10, 10)],  # Geometries.
        crs="EPSG:4326",  # WGS 84.
    )  # End of the frame.
    # Output file.
    path = tmp_path / "test.parquet"
    # Data first, with the bounding box covering column.
    write_geoparquet(gdf, path, write_bbox=True)
    # Everything is read back.
    loaded = read_geoparquet(path)
    # Same rows and attributes.
    assert loaded["name"].tolist() == ["A", "B"] and loaded["value"].tolist() == [1, 2]
    # Same CRS.
    assert loaded.crs.to_epsg() == 4326
    # Metadata of the file.
    meta = geoparquet_metadata(path)
    # Primary column and encoding.
    assert meta["primary_column"] == "geometry"
    # WKB encoding and point geometries.
    assert meta["columns"]["geometry"]["encoding"] == "WKB"
    # Box of the two points.
    assert geoparquet_bounds(path) == (0.0, 0.0, 10.0, 10.0)
    # Only the features that intersect the box.
    assert read_geoparquet(path, bbox=(-1, -1, 1, 1))["name"].tolist() == ["A"]
    # Column selection keeps the geometry.
    assert list(read_geoparquet(path, columns=["name"]).columns) == ["name", "geometry"]
    # The earlier (path, frame) order still works.
    write_geoparquet(tmp_path / "old.parquet", gdf)
    # Same number of rows.
    assert len(read_geoparquet(tmp_path / "old.parquet")) == 2
    # GeoJSON to GeoParquet and back.
    doc = features_to_geojson([geometry_to_feature(SQUARE, {"k": "v"})])
    # Write the document.
    geojson_to_geoparquet(doc, tmp_path / "doc.parquet")
    # Read it as GeoJSON.
    back = geoparquet_to_geojson(tmp_path / "doc.parquet")
    # Geometry and properties survive.
    assert geojson_bounds(back) == (0.0, 0.0, 1.0, 1.0)
    # Properties.
    assert back["features"][0]["properties"] == {"k": "v"}


# Plain Parquet files have no geo metadata.
@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
def test_geoparquet_rejects_plain_parquet(tmp_path: Path) -> None:
    # Tables.
    import pandas as pd

    # GeoParquet metadata reader.
    from unbihexium.io.parquet import geoparquet_metadata, write_geoparquet

    # Plain table.
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "plain.parquet")
    # No geo key.
    with pytest.raises(ValueError, match="no GeoParquet metadata"):
        # Read the metadata.
        geoparquet_metadata(tmp_path / "plain.parquet")
    # Only GeoDataFrames can be written.
    with pytest.raises(TypeError):
        # A list is not a frame.
        write_geoparquet([1, 2], tmp_path / "x.parquet")


# ---------------------------------------------------------------------------
# STAC
# ---------------------------------------------------------------------------


# RFC 3339 times and intervals.
def test_stac_times() -> None:
    # Z suffix and seven fraction digits (cut to microseconds).
    t = parse_datetime("2024-06-01T10:20:30.1234567Z")
    # Components of the time.
    assert (t.year, t.month, t.second, t.microsecond) == (2024, 6, 30, 123456)
    # UTC.
    assert t.utcoffset().total_seconds() == 0
    # Dates are midnight UTC.
    assert parse_datetime("2024-06-01").hour == 0
    # Open start.
    start, end = parse_datetime_range("../2024-01-01T00:00:00Z")
    # Only the end is set.
    assert start is None and end == parse_datetime("2024-01-01")
    # Reversed intervals are errors.
    with pytest.raises(ValueError, match="starts after"):
        # End before start.
        parse_datetime_range("2024-02-01/2024-01-01")


# Box intersection, including boxes that cross the antimeridian.
def test_bbox_intersects() -> None:
    # Overlapping boxes.
    assert bbox_intersects((0, 0, 2, 2), (1, 1, 3, 3))
    # Touching boxes intersect.
    assert bbox_intersects((0, 0, 1, 1), (1, 0, 2, 1))
    # Disjoint boxes.
    assert not bbox_intersects((0, 0, 1, 1), (2, 2, 3, 3))
    # Box from 170 E to 170 W crosses the antimeridian and contains 179 E.
    assert bbox_intersects((170, -10, -170, 10), (179, 0, 179.5, 1))
    # It does not contain 0 E.
    assert not bbox_intersects((170, -10, -170, 10), (-1, -1, 1, 1))
    # 3-D boxes use their horizontal extent.
    assert bbox_intersects((0, 0, -5, 2, 2, 5), (1, 1, 3, 3))


# Item parsing, validation and offline search.
def test_stac_items_and_filter(tmp_path: Path) -> None:
    # Items of two dates and collections.
    raw = [
        make_item("a", [0, 0, 1, 1], "2024-01-10T00:00:00Z", **{"eo:cloud_cover": 5}),  # Clear.
        make_item("b", [5, 5, 6, 6], "2024-03-10T00:00:00Z", **{"eo:cloud_cover": 50}),  # Cloudy.
        make_item("c", [0, 0, 1, 1], "2024-02-10T00:00:00Z", platform="sentinel-2b"),  # No cover.
    ]  # End of the dictionaries.
    # Parsed items.
    items = [STACItem.from_dict(d) for d in raw]
    # Box filter.
    assert [i.id for i in filter_items(items, bbox=(0.5, 0.5, 2, 2))] == ["a", "c"]
    # Time filter with an open end.
    assert [i.id for i in filter_items(items, datetime_range="2024-02-01/..")] == ["b", "c"]
    # Cloud cover filter keeps items without cloud cover.
    assert [i.id for i in filter_items(items, max_cloud_cover=10)] == ["a", "c"]
    # Query extension operators.
    assert [i.id for i in filter_items(items, query={"eo:cloud_cover": {"gte": 10}})] == ["b"]
    # String operator.
    assert [i.id for i in filter_items(items, query={"platform": {"startsWith": "sentinel"}})] == [
        "c"
    ]
    # Limit.
    assert len(filter_items(items, limit=1)) == 1
    # Unknown operators are errors.
    with pytest.raises(ValueError, match="unknown query operator"):
        # Misspelt operator.
        filter_items(items, query={"platform": {"like": "x"}})
    # A null datetime needs an interval.
    with pytest.raises(ValueError, match="datetime is null"):
        # Invalid item.
        STACItem.from_dict(make_item("x", [0, 0, 1, 1], None))
    # Interval items match overlapping ranges.
    span = {"start_datetime": "2024-01-01", "end_datetime": "2024-12-31"}
    # Item without an instant.
    year = make_item("r", [0, 0, 1, 1], None, **span)
    # Parsed item.
    ranged = STACItem.from_dict(year)
    # The range overlaps June.
    assert filter_items([ranged], datetime_range="2024-06-01/2024-06-30") == [ranged]
    # Relative hrefs are resolved against the item file.
    path = tmp_path / "items" / "a.json"
    # Directory of the item.
    path.parent.mkdir()
    # Write the item.
    path.write_text(json.dumps(make_item("a", [0, 0, 1, 1], "2024-01-01T00:00:00Z")))
    # Parse the file.
    item = read_stac_item(path)
    # Absolute path next to the item.
    assert item.assets["B04"] == str(tmp_path / "items" / "B04.tif")
    # Asset lookup by role and media type.
    assert item.find_assets(role="data", media_type="image/tiff") == ["B04"]
    # Round trip through the dictionary.
    assert STACItem.from_dict(item.to_dict()).id == "a"


# Static catalogues are walked through child and item links.
def test_walk_catalog(tmp_path: Path) -> None:
    # Root catalogue linking to a collection (and back to itself: a cycle).
    root = {
        "type": "Catalog",  # Type.
        "id": "root",  # Id.
        "links": [  # Links.
            {"rel": "child", "href": "col/collection.json"},  # Child.
            {"rel": "self", "href": "catalog.json"},  # Self.
        ],  # End of the links.
    }  # End of the catalogue.
    # Collection with two items and a link back to the root.
    collection = {
        "type": "Collection",  # Type.
        "id": "col",  # Id.
        "description": "test",  # Description.
        "license": "CC-BY-4.0",  # Licence.
        "extent": {  # Extent.
            "spatial": {"bbox": [[0, 0, 2, 2]]},  # Box.
            "temporal": {"interval": [["2024-01-01T00:00:00Z", None]]},  # Open interval.
        },  # End of the extent.
        "links": [  # Links.
            {"rel": "item", "href": "i1.json"},  # First item.
            {"rel": "item", "href": "i2.json"},  # Second item.
            {"rel": "child", "href": "../catalog.json"},  # Cycle.
        ],  # End of the links.
    }  # End of the collection.
    # Collection directory.
    (tmp_path / "col").mkdir()
    # Root file.
    (tmp_path / "catalog.json").write_text(json.dumps(root))
    # Collection file.
    (tmp_path / "col" / "collection.json").write_text(json.dumps(collection))
    # Item files.
    for name in ("i1", "i2"):
        # Item with the file name as id.
        text = json.dumps(make_item(name, [0, 0, 1, 1], "2024-01-01T00:00:00Z"))
        # Write the item file.
        (tmp_path / "col" / f"{name}.json").write_text(text)
    # Items in link order; the cycle does not repeat them.
    assert [i.id for i in walk_catalog(tmp_path / "catalog.json")] == ["i1", "i2"]
    # Collection extent.
    parsed = STACCollection.from_dict(collection)
    # Open end of the interval.
    assert parsed.temporal_extent == (parse_datetime("2024-01-01"), None)


# API search follows next links with a fake transport (no network).
def test_stac_client_pagination() -> None:
    # Requests seen by the transport.
    calls: list[tuple[str, str, Any]] = []

    # Two pages: the first links to the second with a POST body.
    def transport(method: str, url: str, body: dict[str, Any] | None) -> dict[str, Any]:
        # Record the request.
        calls.append((method, url, body))
        # First page.
        if len(calls) == 1:
            # Two items and a next link.
            features = [make_item(f"p1-{i}", [0, 0, 1, 1], "2024-01-01T00:00:00Z") for i in (0, 1)]
            # POST link whose body is merged into the previous one.
            nxt = {
                "rel": "next",  # Link relation.
                "href": url,  # Same endpoint.
                "method": "POST",  # Method.
                "body": {"token": "t2"},  # Page token.
                "merge": True,  # Merge with the previous body.
            }  # End of the link.
            # First page.
            return {"features": features, "links": [nxt]}
        # Second and last page.
        return {"features": [make_item("p2-0", [0, 0, 1, 1], "2024-01-02")], "links": []}

    # Client with the fake transport.
    client = STACClient("https://example.test/stac/", transport=transport)
    # Search with a box, an interval and a collection.
    search = client.search((0, 0, 1, 1), ("2024-01-01", None), ["s2"])
    # Collect the items.
    found = list(search)
    # Items of both pages.
    assert [i.id for i in found] == ["p1-0", "p1-1", "p2-0"]
    # First request: POST /search with the STAC API body.
    assert calls[0][0] == "POST" and calls[0][1] == "https://example.test/stac/search"
    # Interval text with an open end.
    assert calls[0][2]["datetime"] == "2024-01-01T00:00:00Z/.."
    # The merged second body keeps the filters and adds the token.
    assert calls[1][2]["token"] == "t2" and calls[1][2]["collections"] == ["s2"]
    # The limit stops the search early.
    calls.clear()
    # Only one item.
    assert len(list(client.search(limit=1))) == 1 and len(calls) == 1


# ---------------------------------------------------------------------------
# Zarr
# ---------------------------------------------------------------------------


# Round trip, chunks, attributes, selections and both argument orders.
@pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")
def test_zarr_round_trip(tmp_path: Path) -> None:
    # Zarr functions.
    from unbihexium.io.zarr_io import read_zarr, write_zarr, zarr_info

    # Distinct values.
    data = np.arange(4 * 64 * 64, dtype=np.float32).reshape(4, 64, 64)
    # Store.
    path = tmp_path / "test.zarr"
    # Data first.
    write_zarr(data, path, chunks=(1, 32, 32), attrs={"units": "reflectance"})
    # Read everything.
    loaded, attrs = read_zarr(path)
    # Exact values and attributes.
    np.testing.assert_array_equal(loaded, data)
    # Attributes.
    assert attrs == {"units": "reflectance"}
    # Chunks as requested.
    assert zarr_info(path)["chunks"] == (1, 32, 32)
    # Selections read only a part.
    part, _ = read_zarr(path, selection=(slice(2, 3), slice(0, 2), slice(10, 12)))
    # Values of the selection.
    np.testing.assert_array_equal(part, data[2:3, 0:2, 10:12])
    # The earlier (path, data) order still works.
    write_zarr(tmp_path / "old.zarr", data, chunks=(2, 64, 64))
    # Chunks of the second store.
    assert zarr_info(tmp_path / "old.zarr")["chunks"] == (2, 64, 64)
    # Chunks must match the axes.
    with pytest.raises(ValueError, match="axes"):
        # Two chunk sizes for three axes.
        write_zarr(data, tmp_path / "bad.zarr", chunks=(1, 2))


# Groups, formats, codecs and georeferenced rasters.
@pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")
def test_zarr_groups_and_rasters(tmp_path: Path) -> None:
    # Zarr functions.
    from unbihexium.io.zarr_io import (
        list_arrays,  # Group members.
        read_raster_zarr,  # Raster reading.
        read_zarr,  # Reading.
        write_raster_zarr,  # Raster writing.
        write_zarr,  # Writing.
        zarr_info,  # Description.
        zarr_major_version,  # Library version.
    )  # End of the imports.

    # Two members of one group.
    write_zarr(np.ones((2, 2), dtype=np.int32), tmp_path / "g.zarr", name="b")
    # Second member.
    write_zarr(np.zeros((3,), dtype=np.uint8), tmp_path / "g.zarr", name="a")
    # Sorted names.
    assert list_arrays(tmp_path / "g.zarr") == ["a", "b"]
    # Named member.
    assert read_zarr(tmp_path / "g.zarr", "b")[0].tolist() == [[1, 1], [1, 1]]
    # Without a name the first member is read.
    assert read_zarr(tmp_path / "g.zarr")[0].dtype == np.uint8
    # Unknown members.
    with pytest.raises(KeyError):
        # No such array.
        read_zarr(tmp_path / "g.zarr", "c")
    # Version 2 format with Blosc.
    write_zarr(np.arange(6), tmp_path / "v2.zarr", zarr_format=2, compressor="blosc")
    # Format of the store.
    assert zarr_info(tmp_path / "v2.zarr")["zarr_format"] == 2
    # Raster with georeferencing and a NaN no-data value.
    pixels = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    # Write the array with explicit georeferencing.
    write_raster_zarr(
        pixels,  # Pixels.
        tmp_path / "r.zarr",  # Store.
        crs="EPSG:32632",  # UTM zone 32N.
        transform=(10, 0, 5, 0, -10, 7),  # 10 m pixels.
        nodata=float("nan"),  # NaN marks missing values.
    )  # End of the write.
    # Attributes of the store.
    info = zarr_info(tmp_path / "r.zarr")
    # Georeferencing attributes.
    assert info["attrs"]["crs"] == "EPSG:32632" and info["attrs"]["nodata"] == "nan"
    # Dimension names in Zarr v3.
    if zarr_major_version() >= 3:
        # Band, y and x.
        assert info["dimension_names"] == ["band", "y", "x"]
    # Read it as a Raster.
    raster = read_raster_zarr(tmp_path / "r.zarr")
    # Pixels and transform.
    np.testing.assert_array_equal(raster.data, pixels)
    # Transform coefficients.
    assert tuple(raster.metadata.transform) == (10, 0, 5, 0, -10, 7)
    # No-data decoded to NaN.
    assert np.isnan(raster.metadata.nodata)


# =============================================================================
# End of module tests/unit/test_io.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
