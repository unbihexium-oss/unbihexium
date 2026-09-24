# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/io/geojson.py
# Title       : GeoJSON reading, writing, validation and reprojection
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy; reprojection needs
#               pyproj and data frames need geopandas
# =============================================================================
#
# Abstract
# --------
# GeoJSON documents are handled as plain dictionaries, which keeps them
# JSON-serialisable and free of heavy dependencies:
#
#   read_geojson, write_geojson   file input and output; writing validates
#                                 the document, can round the coordinates
#                                 and replaces the file atomically
#   geojson_problems,             structural validation against RFC 7946
#   validate_geojson              (object types, positions, rings);
#                                 GeometryCollections nested deeper than
#                                 MAX_COLLECTION_DEPTH are reported as a
#                                 problem instead of exhausting the stack
#   geometry_to_feature,          construction of features and collections
#   features_to_geojson
#   geojson_crs                   CRS of a document: the legacy "crs" member
#                                 of the 2008 specification, else OGC:CRS84
#   geojson_bounds                bounding box of every position
#   ring_area, rewind             signed ring area and the right-hand rule
#   map_coordinates,              coordinate transforms, for example between
#   reproject_geojson             UTM and longitude/latitude
#   to_geodataframe,              conversion to and from geopandas
#   from_geodataframe
#
# Method
# ------
# RFC 7946 fixes the coordinate reference system to WGS 84 longitude and
# latitude (OGC:CRS84) and removed the "crs" member of the 2008 draft.
# Files written by older software, and intermediate products in projected
# coordinates, still carry it; geojson_crs() reads it and
# reproject_geojson() converts such documents to OGC:CRS84. The signed area
# of a ring follows the shoelace formula
#
#   A = 1/2 * sum_i (x_i * y_{i+1} - x_{i+1} * y_i)
#
# which is positive for counterclockwise rings. RFC 7946 section 3.1.6
# requires exterior rings to be counterclockwise and holes clockwise
# (the right-hand rule); rewind() enforces it.
#
# References
# ----------
# Butler, H., Daly, M., Doyle, A., Gillies, S., Hagen, S. and Schaub, T.
# (2016). The GeoJSON Format. IETF RFC 7946. doi:10.17487/RFC7946
# Butler, H. et al. (2008). The GeoJSON Format Specification, revision 1.0.
# https://geojson.org/geojson-spec.html
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Deep copies of documents.
import copy

# JSON parsing and serialisation.
import json

# Numbers.
import math

# Represent file paths.
from pathlib import Path

# Types of loosely structured values and callables.
from typing import Any, Callable, Iterator

# Arrays.
import numpy as np

# Atomic file writes.
from unbihexium.utils.files import atomic_write_text

# Geometry object types of RFC 7946.
GEOMETRY_TYPES = (
    "Point",  # One position.
    "MultiPoint",  # Positions.
    "LineString",  # Two or more positions.
    "MultiLineString",  # Line strings.
    "Polygon",  # Linear rings.
    "MultiPolygon",  # Polygons.
    "GeometryCollection",  # Geometries.
)  # End of the geometry types.

# Every GeoJSON object type.
OBJECT_TYPES = (*GEOMETRY_TYPES, "Feature", "FeatureCollection")

# Default CRS of RFC 7946: WGS 84 longitude and latitude.
DEFAULT_CRS = "OGC:CRS84"

# Deepest nesting of GeometryCollections that is validated; RFC 7946 section
# 3.1.8 advises against nesting them at all, and deeper documents would
# exhaust the Python stack of the recursive functions of this module.
MAX_COLLECTION_DEPTH = 64

# Nesting depth of the coordinates of each geometry type.
_DEPTH = {
    "Point": 0,  # Position.
    "MultiPoint": 1,  # List of positions.
    "LineString": 1,  # List of positions.
    "MultiLineString": 2,  # List of lists of positions.
    "Polygon": 2,  # List of rings.
    "MultiPolygon": 3,  # List of polygons.
}  # End of the depths.


# Whether a value is a position: two or more finite numbers.
def _is_position(value: Any) -> bool:
    # A list of numbers (booleans are not numbers).
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        # Not a position.
        return False
    # Numbers only; booleans are a subclass of int but not coordinates.
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    # Every element must be a finite number that fits in a float.
    return numbers and all(_is_finite(v) for v in value)


# Whether a number is finite as a float; integers too large for a float are not.
def _is_finite(value: float) -> bool:
    # Convert and test.
    try:
        # Finite double precision value.
        return math.isfinite(float(value))
    # Integers beyond the float range.
    except OverflowError:
        # Not representable.
        return False


# Problems of a coordinate array of the given nesting depth.
def _coordinate_problems(coords: Any, depth: int, where: str) -> list[str]:
    # Positions.
    if depth == 0:
        # Report invalid positions.
        return [] if _is_position(coords) else [f"{where}: invalid position {coords!r}"]
    # Arrays must be lists.
    if not isinstance(coords, (list, tuple)):
        # Report the type.
        return [f"{where}: coordinates must be an array"]
    # Problems of the elements.
    found: list[str] = []
    # Check every element one level down.
    for i, item in enumerate(coords):
        # Recurse.
        found += _coordinate_problems(item, depth - 1, f"{where}[{i}]")
    # Return the problems.
    return found


# Problems of the rings of one polygon.
def _ring_problems(rings: Any, where: str) -> list[str]:
    # Problems found.
    found: list[str] = []
    # Check every ring.
    for i, ring in enumerate(rings):
        # Linear rings have at least four positions.
        if len(ring) < 4:
            # Report it.
            found.append(f"{where}[{i}]: a linear ring needs at least 4 positions")
        # The first and last positions are equal.
        elif list(ring[0]) != list(ring[-1]):
            # Report it.
            found.append(f"{where}[{i}]: a linear ring must be closed")
    # Return the problems.
    return found


# Problems of a geometry object; level counts the enclosing GeometryCollections.
def _geometry_problems(geometry: Any, where: str, level: int = 0) -> list[str]:
    # Geometries are objects.
    if not isinstance(geometry, dict):
        # Report the type.
        return [f"{where}: geometry must be an object"]
    # Geometry type; only strings name a type.
    kind = geometry.get("type") if isinstance(geometry.get("type"), str) else None
    # Collections hold geometries.
    if kind == "GeometryCollection":
        # Member list.
        members = geometry.get("geometries")
        # It must be an array.
        if not isinstance(members, list):
            # Report it.
            return [f"{where}: GeometryCollection needs a geometries array"]
        # Members would be nested deeper than the limit.
        if members and level >= MAX_COLLECTION_DEPTH:
            # Report it without descending further.
            return [f"{where}: GeometryCollection nested deeper than {MAX_COLLECTION_DEPTH} levels"]
        # Next nesting level.
        down = level + 1
        # Problems of the members.
        return [
            p  # One problem.
            for i, g in enumerate(members)  # Every member.
            for p in _geometry_problems(g, f"{where}[{i}]", down)  # Its problems.
        ]  # End of the problems.
    # Unknown types.
    if kind not in _DEPTH:
        # Report it.
        return [f"{where}: unknown geometry type {kind!r}"]
    # Coordinates of the geometry.
    coords: Any = geometry.get("coordinates")
    # Structure of the coordinates.
    found = _coordinate_problems(coords, _DEPTH[kind], f"{where}.coordinates")
    # Stop at structural problems.
    if found:
        # Return them.
        return found
    # Line strings need two positions.
    if kind == "LineString" and len(coords) < 2:
        # Report it.
        found.append(f"{where}: a LineString needs at least 2 positions")
    # Rings of a polygon.
    if kind == "Polygon":
        # Check them.
        found += _ring_problems(coords, f"{where}.coordinates")
    # Rings of every polygon of a multipolygon.
    if kind == "MultiPolygon":
        # Check every polygon.
        for i, polygon in enumerate(coords):
            # Rings of this polygon.
            found += _ring_problems(polygon, f"{where}.coordinates[{i}]")
    # Return the problems.
    return found


# Problems of a feature.
def _feature_problems(feature: Any, where: str) -> list[str]:
    # Features are objects of type Feature.
    if not isinstance(feature, dict) or feature.get("type") != "Feature":
        # Report it.
        return [f"{where}: not a Feature object"]
    # Problems found.
    found: list[str] = []
    # The geometry member is required and may be null.
    if "geometry" not in feature:
        # Report it.
        found.append(f"{where}: missing geometry member")
    # Non-null geometries are checked.
    elif feature["geometry"] is not None:
        # Geometry problems.
        found += _geometry_problems(feature["geometry"], f"{where}.geometry")
    # Properties must be an object or null.
    if not isinstance(feature.get("properties", {}), (dict, type(None))):
        # Report it.
        found.append(f"{where}: properties must be an object or null")
    # Ids are strings or numbers.
    if "id" in feature and not isinstance(feature["id"], (str, int, float)):
        # Report it.
        found.append(f"{where}: id must be a string or a number")
    # Return the problems.
    return found


# Every structural problem of a GeoJSON object; empty when valid.
def geojson_problems(obj: Any) -> list[str]:
    # Documents are objects.
    if not isinstance(obj, dict):
        # Report it.
        return ["document must be a JSON object"]
    # Type of the document; only strings name a type.
    kind = obj.get("type") if isinstance(obj.get("type"), str) else None
    # Collections of features.
    if kind == "FeatureCollection":
        # Feature list.
        features = obj.get("features")
        # It must be an array.
        if not isinstance(features, list):
            # Report it.
            return ["FeatureCollection needs a features array"]
        # Problems of the features.
        return [p for i, f in enumerate(features) for p in _feature_problems(f, f"features[{i}]")]
    # Single features.
    if kind == "Feature":
        # Problems of the feature.
        return _feature_problems(obj, "feature")
    # Geometries.
    if kind in GEOMETRY_TYPES:
        # Problems of the geometry.
        return _geometry_problems(obj, "geometry")
    # Anything else.
    return [f"unknown GeoJSON type {kind!r}"]


# Raise ValueError when a document is not valid GeoJSON.
def validate_geojson(obj: Any) -> dict[str, Any]:
    # Problems of the document.
    found = geojson_problems(obj)
    # Report the first problems.
    if found:
        # Up to five problems in the message.
        shown = "; ".join(found[:5]) + ("; ..." if len(found) > 5 else "")
        # Explain the problems.
        raise ValueError(f"invalid GeoJSON ({len(found)} problems): {shown}")
    # Return the valid document.
    return obj


# Read a GeoJSON file.
def read_geojson(path: str | Path, validate: bool = True) -> dict[str, Any]:
    # Path object.
    path = Path(path)
    # Parse the file; missing files raise FileNotFoundError.
    with open(path, encoding="utf-8") as handle:
        # Parse the JSON text.
        try:
            # Document.
            data = json.load(handle)
        # Malformed JSON.
        except json.JSONDecodeError as exc:
            # Explain the problem with the path.
            raise ValueError(f"{path}: not valid JSON: {exc}") from exc
    # Validate on request.
    return validate_geojson(data) if validate else data


# Accept (data, path) and the (path, data) order of earlier releases.
def _data_and_path(first: Any, second: Any) -> tuple[Any, Path]:
    # Paths in the first position belong to the earlier order.
    if isinstance(first, (str, Path)) and not isinstance(second, (str, Path)):
        # Swap the arguments.
        return second, Path(first)
    # The documented order.
    return first, Path(second)


# Write a GeoJSON document; returns the path.
def write_geojson(
    data: dict[str, Any],  # GeoJSON document.
    path: str | Path,  # Output file.
    indent: int | None = 2,  # JSON indentation; None for compact output.
    validate: bool = True,  # Reject invalid documents.
    precision: int | None = None,  # Decimals of the coordinates; None keeps them.
) -> Path:  # Written file.
    # Support the argument order of earlier releases.
    data, path = _data_and_path(data, path)
    # Validate first so that no invalid file is written.
    if validate:
        # Raises ValueError.
        validate_geojson(data)
    # Rounded copy on request (RFC 7946 suggests 6 decimals, about 10 cm).
    if precision is not None:
        # Round every coordinate array.
        data = map_coordinates(data, lambda xy: np.round(xy, precision), keep_crs=True)
    # Serialise; NaN is not valid JSON.
    text = json.dumps(data, indent=indent, ensure_ascii=False, allow_nan=False)
    # Write atomically.
    return atomic_write_text(path, text + "\n")


# Feature object from a geometry.
def geometry_to_feature(
    geometry: dict[str, Any] | None,  # Geometry object, or None for unlocated features.
    properties: dict[str, Any] | None = None,  # Attributes.
    feature_id: str | int | None = None,  # Optional id.
) -> dict[str, Any]:  # Feature.
    # Required members.
    feature: dict[str, Any] = {
        "type": "Feature",  # Object type.
        "geometry": geometry,  # Geometry.
        "properties": dict(properties or {}),  # Attributes.
    }  # End of the feature.
    # Optional id.
    if feature_id is not None:
        # Store it.
        feature["id"] = feature_id
    # Return the feature.
    return feature


# FeatureCollection from features; a non-default CRS is stored as a legacy member.
def features_to_geojson(
    features: list[dict[str, Any]],  # Features.
    crs: str | None = None,  # CRS of the coordinates, None for OGC:CRS84.
) -> dict[str, Any]:  # FeatureCollection.
    # Collection.
    collection: dict[str, Any] = {"type": "FeatureCollection", "features": list(features)}
    # The 2008 named CRS member for other systems.
    if crs:
        # Store the CRS name.
        collection["crs"] = {"type": "name", "properties": {"name": crs}}
    # Return the collection.
    return collection


# CRS of a document as an authority string, for example "EPSG:32632".
def geojson_crs(obj: dict[str, Any]) -> str:
    # Legacy member.
    member = obj.get("crs")
    # Documents without it use the RFC 7946 default.
    if not isinstance(member, dict):
        # WGS 84 longitude and latitude.
        return DEFAULT_CRS
    # Name of the CRS.
    name = str(member.get("properties", {}).get("name", "")).strip()
    # Missing names.
    if not name:
        # Fall back to the default.
        return DEFAULT_CRS
    # OGC URNs such as urn:ogc:def:crs:EPSG::32632 or urn:ogc:def:crs:OGC:1.3:CRS84.
    if name.lower().startswith("urn:ogc:def:crs:"):
        # Fields of the URN.
        parts = name.split(":")
        # Authority and code (the version field may be empty).
        authority, code = parts[4], parts[-1]
        # CRS84 is written OGC:CRS84.
        return f"{authority.upper()}:{code}"
    # Plain names are returned unchanged.
    return name


# Every position of a document (features, collections and geometries).
def iter_positions(obj: Any) -> Iterator[list[float]]:
    # Collections of features.
    if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
        # Every feature.
        for feature in obj.get("features", []):
            # Positions of the feature.
            yield from iter_positions(feature)
    # Features.
    elif isinstance(obj, dict) and obj.get("type") == "Feature":
        # Positions of the geometry.
        yield from iter_positions(obj.get("geometry"))
    # Geometry collections.
    elif isinstance(obj, dict) and obj.get("type") == "GeometryCollection":
        # Every member.
        for geometry in obj.get("geometries", []):
            # Positions of the member.
            yield from iter_positions(geometry)
    # Geometries.
    elif isinstance(obj, dict):
        # Positions of the coordinates.
        yield from iter_positions(obj.get("coordinates"))
    # Positions.
    elif _is_position(obj):
        # One position.
        yield list(obj)
    # Arrays of coordinates.
    elif isinstance(obj, (list, tuple)):
        # Every element.
        for item in obj:
            # Positions of the element.
            yield from iter_positions(item)


# Bounding box (min x, min y, max x, max y) of every position.
def geojson_bounds(obj: dict[str, Any]) -> tuple[float, float, float, float]:
    # Horizontal coordinates of all positions.
    xy = np.array([p[:2] for p in iter_positions(obj)], dtype=np.float64).reshape(-1, 2)
    # Empty documents have no bounds.
    if xy.shape[0] == 0:
        # Explain the problem.
        raise ValueError("the document has no positions")
    # Minimum and maximum per axis.
    low, high = xy.min(axis=0), xy.max(axis=0)
    # Box.
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


# Signed area of a ring with the shoelace formula; positive when counterclockwise.
def ring_area(ring: Any) -> float:
    # Horizontal coordinates; positions may mix two and three dimensions.
    xy = np.asarray([p[:2] for p in ring], dtype=np.float64).reshape(-1, 2)
    # Next vertex of every vertex (closed rings repeat the first vertex).
    nxt = np.roll(xy, -1, axis=0)
    # Half the sum of the cross products.
    return float(0.5 * np.sum(xy[:, 0] * nxt[:, 1] - nxt[:, 0] * xy[:, 1]))


# Orientation of a ring: 1 counterclockwise, -1 clockwise, 0 without area.
def _orientation(ring: Any) -> int:
    # Horizontal coordinates; positions may mix two and three dimensions.
    xy = np.asarray([p[:2] for p in ring], dtype=np.float64).reshape(-1, 2)
    # Next vertex of every vertex.
    nxt = np.roll(xy, -1, axis=0)
    # Cross products of the shoelace formula.
    cross = xy[:, 0] * nxt[:, 1] - nxt[:, 0] * xy[:, 1]
    # Twice the signed area.
    total = float(np.sum(cross))
    # Rounding error of the sum, relative to the size of its terms.
    tolerance = 1e-9 * float(np.sum(np.abs(cross)))
    # Collinear and repeated vertices give an area within the rounding error,
    # whose sign would flip at random when the ring is reversed.
    return 0 if abs(total) <= tolerance else (1 if total > 0 else -1)


# Orient the rings of one polygon: exterior counterclockwise, holes clockwise.
def _rewind_polygon(rings: list[Any]) -> list[Any]:
    # Oriented rings.
    out = []
    # Visit every ring.
    for i, ring in enumerate(rings):
        # The exterior ring is the first one.
        want_ccw = i == 0
        # Orientation; zero for degenerate rings, which have no orientation.
        sign = _orientation(ring)
        # Reverse rings with the wrong orientation and keep degenerate rings.
        out.append(list(ring)[::-1] if sign != 0 and (sign > 0) != want_ccw else list(ring))
    # Return the rings.
    return out


# Copy of a document that follows the right-hand rule of RFC 7946.
def rewind(obj: dict[str, Any]) -> dict[str, Any]:
    # Work on a copy.
    out = copy.deepcopy(obj)
    # Visit every geometry.
    for geometry in _geometries(out):
        # Polygons.
        if geometry.get("type") == "Polygon":
            # Orient the rings.
            geometry["coordinates"] = _rewind_polygon(geometry["coordinates"])
        # Multipolygons.
        elif geometry.get("type") == "MultiPolygon":
            # Orient the rings of every polygon.
            geometry["coordinates"] = [_rewind_polygon(p) for p in geometry["coordinates"]]
    # Return the copy.
    return out


# Every geometry object with coordinates in a document.
def _geometries(obj: Any) -> Iterator[dict[str, Any]]:
    # Only objects hold geometries.
    if not isinstance(obj, dict):
        # Nothing to yield.
        return
    # Type of the object.
    kind = obj.get("type")
    # Collections of features.
    if kind == "FeatureCollection":
        # Every feature.
        for feature in obj.get("features", []):
            # Geometries of the feature.
            yield from _geometries(feature)
    # Features.
    elif kind == "Feature":
        # The geometry.
        yield from _geometries(obj.get("geometry"))
    # Geometry collections.
    elif kind == "GeometryCollection":
        # Every member.
        for geometry in obj.get("geometries", []):
            # Geometries of the member.
            yield from _geometries(geometry)
    # Geometries with coordinates.
    elif kind in _DEPTH:
        # The geometry itself.
        yield obj


# Apply a function to the coordinates of a positions array of the given depth.
def _map_array(coords: Any, depth: int, func: Callable[[Any], Any]) -> Any:
    # Positions of one level: transform them together.
    if depth <= 1:
        # (N, dims) array; a single position becomes (1, dims).
        arr = np.atleast_2d(np.asarray(coords, dtype=np.float64))
        # Transformed positions.
        result = np.asarray(func(arr), dtype=np.float64).tolist()
        # Single positions stay single.
        return result[0] if depth == 0 else result
    # Deeper arrays are processed element by element.
    return [_map_array(item, depth - 1, func) for item in coords]


# Copy of a document with a function applied to every array of positions.
def map_coordinates(
    obj: dict[str, Any],  # GeoJSON document.
    func: Callable[[Any], Any],  # Maps an (N, dims) array to an (N, dims) array.
    keep_crs: bool = False,  # Keep the legacy crs member.
) -> dict[str, Any]:  # Transformed copy.
    # Work on a copy.
    out = copy.deepcopy(obj)
    # Visit every geometry.
    for geometry in _geometries(out):
        # Depth of its coordinates.
        depth = _DEPTH[geometry["type"]]
        # Transformed coordinates.
        geometry["coordinates"] = _map_array(geometry["coordinates"], depth, func)
    # Bounding boxes are no longer valid.
    for item in [out, *out.get("features", [])]:
        # Remove stale boxes.
        item.pop("bbox", None)
    # The CRS member no longer applies unless the transform keeps the CRS.
    if not keep_crs:
        # Remove it.
        out.pop("crs", None)
    # Return the copy.
    return out


# Reproject a document; the result is in OGC:CRS84 unless dst_crs is given.
def reproject_geojson(
    obj: dict[str, Any],  # GeoJSON document.
    src_crs: str | None = None,  # Source CRS; default the document CRS.
    dst_crs: str = DEFAULT_CRS,  # Target CRS.
) -> dict[str, Any]:  # Reprojected copy.
    # Imported lazily because pyproj is only needed here.
    from pyproj import Transformer

    # Source CRS of the document.
    source = src_crs or geojson_crs(obj)
    # Transformer in x/y (longitude/latitude) axis order.
    transformer = Transformer.from_crs(source, dst_crs, always_xy=True)

    # Transform an (N, dims) array of positions.
    def transform(arr: Any) -> Any:
        # New horizontal coordinates.
        x, y = transformer.transform(arr[:, 0], arr[:, 1])
        # Copy that keeps extra dimensions such as elevation.
        out = arr.copy()
        # Store x.
        out[:, 0] = x
        # Store y.
        out[:, 1] = y
        # Return the positions.
        return out

    # Transformed copy without the old CRS member.
    result = map_coordinates(obj, transform)
    # Record a non-default target CRS as a legacy member.
    if dst_crs not in (DEFAULT_CRS, "EPSG:4326") and result.get("type") == "FeatureCollection":
        # Named CRS member.
        result["crs"] = {"type": "name", "properties": {"name": dst_crs}}
    # Return the copy.
    return result


# GeoDataFrame of a document.
def to_geodataframe(obj: dict[str, Any], crs: str | None = None) -> Any:
    # Imported lazily because geopandas is only needed here.
    import geopandas as gpd

    # Features of the document.
    if obj.get("type") == "FeatureCollection":
        # Every feature.
        features = obj["features"]
    # A single feature.
    elif obj.get("type") == "Feature":
        # One feature.
        features = [obj]
    # A bare geometry.
    else:
        # Wrap it.
        features = [geometry_to_feature(obj)]
    # Frame with the document CRS unless one is given.
    return gpd.GeoDataFrame.from_features(features, crs=crs or geojson_crs(obj))


# FeatureCollection of a GeoDataFrame, with plain lists as coordinates.
def from_geodataframe(frame: Any) -> dict[str, Any]:
    # geopandas serialisation, parsed back into a document.
    data = json.loads(frame.to_json(drop_id=False))
    # Frames in another CRS keep it as a legacy member.
    if frame.crs is not None and not frame.crs.equals("OGC:CRS84") and frame.crs.to_epsg() != 4326:
        # Authority string of the CRS.
        authority = frame.crs.to_authority()
        # Store the name.
        name = f"{authority[0]}:{authority[1]}" if authority else frame.crs.to_string()
        # Named CRS member.
        data["crs"] = {"type": "name", "properties": {"name": name}}
    # Return the collection.
    return data


# =============================================================================
# End of module src/unbihexium/io/geojson.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
