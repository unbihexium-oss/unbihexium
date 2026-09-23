# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/io/stac.py
# Title       : SpatioTemporal Asset Catalog items, catalogues and search
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library; STAC API search
#               needs requests, loading assets needs rasterio
# =============================================================================
#
# Abstract
# --------
# STAC describes Earth observation scenes as GeoJSON Features ("items") with
# acquisition times, properties and links to their files ("assets").
#
#   STACItem, STACCollection   parsed records with validation, relative
#                              asset hrefs resolved against the item
#   parse_datetime,            RFC 3339 times and the open or closed
#   parse_datetime_range       intervals of the STAC API ("a/b", "../b")
#   bbox_intersects            box test that handles boxes crossing the
#                              antimeridian (west > east, RFC 7946 5.2)
#   filter_items               offline search: box, time, collections, ids,
#                              cloud cover and the query extension operators
#   read_stac_item,            local JSON files
#   read_stac_collection
#   walk_catalog               items of a static catalogue, following child
#                              and item links recursively
#   STACClient, search_stac    STAC API item search with pagination through
#                              "next" links; the HTTP transport can be
#                              replaced, for example in tests
#   load_from_stac             read an asset with the GeoTIFF reader
#
# Method
# ------
# An item matches a time interval when its own interval (datetime, or
# start_datetime and end_datetime) overlaps it, as the STAC API item search
# specifies. Missing ends of an interval are unbounded.
#
# References
# ----------
# STAC contributors (2021). SpatioTemporal Asset Catalog specification,
# version 1.0.0 (item, collection and catalog specifications).
# https://github.com/radiantearth/stac-spec
# STAC API contributors (2023). STAC API specification 1.0.0, item search
# and the query extension. https://github.com/radiantearth/stac-api-spec
# Klyne, G. and Newman, C. (2002). Date and Time on the Internet:
# Timestamps. IETF RFC 3339. doi:10.17487/RFC3339
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON files.
import json

# Check that numbers are finite.
import math

# Fractional seconds of RFC 3339 times.
import re

# Iterators.
from collections.abc import Iterable, Iterator

# Records.
from dataclasses import dataclass, field

# Times.
from datetime import datetime, timezone

# Represent file paths.
from pathlib import Path

# Types of loosely structured values and callables.
from typing import Any, Callable

# Relative references.
from urllib.parse import urljoin, urlparse

# Box of a geometry.
from unbihexium.io.geojson import geojson_bounds, geojson_problems

# HTTP transport: (method, url, json body or None) -> decoded JSON response.
Transport = Callable[[str, str, "dict[str, Any] | None"], "dict[str, Any]"]

# Operators of the STAC API query extension.
QUERY_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda v, t: v == t,  # Equal.
    "neq": lambda v, t: v != t,  # Not equal.
    "lt": lambda v, t: v < t,  # Less than.
    "lte": lambda v, t: v <= t,  # Less or equal.
    "gt": lambda v, t: v > t,  # Greater than.
    "gte": lambda v, t: v >= t,  # Greater or equal.
    "startsWith": lambda v, t: str(v).startswith(str(t)),  # Prefix.
    "endsWith": lambda v, t: str(v).endswith(str(t)),  # Suffix.
    "contains": lambda v, t: str(t) in str(v),  # Substring.
    "in": lambda v, t: v in t,  # Membership.
}  # End of the operators.


# Parse an RFC 3339 time; times without a zone are taken as UTC.
def parse_datetime(value: str | datetime) -> datetime:
    # Datetime objects only get a zone.
    if isinstance(value, datetime):
        # Aware copy.
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    # Normalise the zone designator for fromisoformat on Python 3.10.
    text = value.strip().replace("z", "Z").replace("Z", "+00:00")
    # Dates without a time are midnight.
    if "T" not in text and " " not in text:
        # Add the time.
        text += "T00:00:00+00:00"
    # Python 3.10 accepts only 0, 3 or 6 fraction digits; pad or cut to 6.
    match = re.match(r"^(.*?[T ]\d{2}:\d{2}:\d{2})\.(\d+)(.*)$", text)
    # Rewrite the fraction when present.
    if match:
        # Time, fraction digits and zone.
        head, digits, zone = match.groups()
        # Six digits.
        text = f"{head}.{(digits + '000000')[:6]}{zone}"
    # Parse the text.
    try:
        # ISO 8601 parser of the standard library.
        parsed = datetime.fromisoformat(text)
    # Malformed times.
    except ValueError as exc:
        # Explain the problem.
        raise ValueError(f"not an RFC 3339 time: {value!r}") from exc
    # Aware result.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# Interval (start, end) from "a/b", "../b", "a/..", a single time or a pair.
def parse_datetime_range(value: Any) -> tuple[datetime | None, datetime | None]:
    # Pairs of values.
    if isinstance(value, (tuple, list)):
        # Exactly two ends.
        if len(value) != 2:
            # Explain the expected form.
            raise ValueError(f"a datetime range needs two ends, got {value!r}")
        # Parse each end; None, "" and ".." are open.
        start, end = (None if v in (None, "", "..") else parse_datetime(v) for v in value)
    # Text intervals.
    elif isinstance(value, str) and "/" in value:
        # Split at the slash.
        first, _, second = value.partition("/")
        # Parse both ends.
        return parse_datetime_range((first, second))
    # Single instants.
    else:
        # Both ends at the instant.
        start = end = parse_datetime(value)
    # Ordered interval.
    if start and end and start > end:
        # Explain the problem.
        raise ValueError(f"the range starts after it ends: {value!r}")
    # Return the interval.
    return start, end


# Raise ValueError when a member of an item dictionary has the wrong JSON type.
def _check_item_members(data: dict[str, Any]) -> None:
    # Name of the item for the messages.
    name = data["id"]
    # Properties are an object or null.
    if not isinstance(data.get("properties") or {}, dict):
        # Explain the problem.
        raise ValueError(f"item {name}: properties must be an object")
    # Acquisition times are strings or null.
    props = data.get("properties") or {}
    # Check the three time members.
    for key in ("datetime", "start_datetime", "end_datetime"):
        # Strings or null only.
        if props.get(key) is not None and not isinstance(props[key], str):
            # Explain the problem.
            raise ValueError(f"item {name}: {key} must be a string or null")
    # Links are an array of objects.
    links = data.get("links") or []
    # Check the container and every link.
    if not isinstance(links, list) or not all(isinstance(lk, dict) for lk in links):
        # Explain the problem.
        raise ValueError(f"item {name}: links must be an array of objects")
    # Assets are an object of objects.
    assets = data.get("assets") or {}
    # Check the container and every asset.
    if not isinstance(assets, dict) or not all(isinstance(a, dict) for a in assets.values()):
        # Explain the problem.
        raise ValueError(f"item {name}: assets must be an object of objects")
    # Hrefs of links and assets are strings.
    for entry in [*links, *assets.values()]:
        # Missing hrefs are allowed; present ones are strings.
        if "href" in entry and not isinstance(entry["href"], str):
            # Explain the problem.
            raise ValueError(f"item {name}: every href must be a string")
    # Extension lists are arrays of strings.
    extensions = data.get("stac_extensions") or []
    # Check the container and every entry.
    if not isinstance(extensions, list) or not all(isinstance(e, str) for e in extensions):
        # Explain the problem.
        raise ValueError(f"item {name}: stac_extensions must be an array of strings")
    # Geometries are valid GeoJSON geometries or null.
    geometry = data.get("geometry")
    # Validate a present geometry.
    if geometry is not None:
        # Problems of the geometry.
        found = geojson_problems(geometry)
        # Only geometries are allowed, not features or collections.
        if found or geometry.get("type") in ("Feature", "FeatureCollection"):
            # First problems, or the type of a feature given as geometry.
            detail = "; ".join(found[:3]) or geometry.get("type")
            # Explain the problem.
            raise ValueError(f"item {name}: invalid geometry: {detail}")


# Raise ValueError unless a box has 4 or 6 finite numbers.
def _check_bbox(name: str, box: Any) -> None:
    # An array of 4 or 6 numbers; booleans are not numbers.
    if not isinstance(box, list) or len(box) not in (4, 6):
        # Explain the problem.
        raise ValueError(f"item {name}: bbox must be an array of 4 or 6 numbers")
    # Every value is a finite number.
    for value in box:
        # Numbers only.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            # Explain the problem.
            raise ValueError(f"item {name}: bbox values must be numbers")
        # Finite numbers only, including integers too large for a float.
        try:
            # Convert to a float.
            finite = math.isfinite(float(value))
        # Integers beyond the float range.
        except OverflowError:
            # Not representable.
            finite = False
        # Reject infinities and NaN.
        if not finite:
            # Explain the problem.
            raise ValueError(f"item {name}: bbox values must be finite")


# Split a box that crosses the antimeridian into boxes with west <= east.
def _split_box(box: Any) -> list[tuple[float, float, float, float]]:
    # Horizontal extent of 2-D and 3-D boxes.
    values = [float(v) for v in box]
    # 3-D boxes: (west, south, low, east, north, high).
    if len(values) == 6:
        # Drop the vertical extent.
        values = [values[0], values[1], values[3], values[4]]
    # Only 2-D and 3-D boxes are valid.
    if len(values) != 4:
        # Explain the problem.
        raise ValueError(f"a bbox needs 4 or 6 values, got {len(values)}")
    # Corners.
    west, south, east, north = values
    # Boxes crossing the antimeridian have west > east.
    if west > east:
        # Eastern and western parts.
        return [(west, south, 180.0, north), (-180.0, south, east, north)]
    # Ordinary box.
    return [(west, south, east, north)]


# Whether two boxes intersect (touching boxes intersect).
def bbox_intersects(a: Any, b: Any) -> bool:
    # Any pair of parts that overlaps.
    return any(
        p[0] <= q[2] and q[0] <= p[2] and p[1] <= q[3] and q[1] <= p[3]  # Overlap on both axes.
        for p in _split_box(a)  # Parts of the first box.
        for q in _split_box(b)  # Parts of the second box.
    )  # End of the test.


# Resolve an href against a base (file path or URL).
def resolve_href(href: str, base: str | None) -> str:
    # Absolute URLs and missing bases need no resolution.
    if not base or urlparse(href).scheme in ("http", "https", "s3", "gs", "file"):
        # Return the href.
        return href
    # Absolute local paths.
    if Path(href).is_absolute():
        # Return the href.
        return href
    # URL bases.
    if urlparse(base).scheme in ("http", "https", "s3", "gs"):
        # Join like a browser.
        return urljoin(base, href)
    # Local paths: relative to the directory of the base file.
    return str((Path(base).parent / href).resolve())


# One STAC item.
@dataclass
class STACItem:
    # Item id.
    id: str
    # Box (west, south, east, north) in longitude and latitude.
    bbox: tuple[float, float, float, float]
    # Acquisition time; None when only an interval is given.
    datetime: datetime | None
    # Item properties.
    properties: dict[str, Any] = field(default_factory=dict)
    # Asset hrefs by key, resolved against the item location.
    assets: dict[str, str] = field(default_factory=dict)
    # Collection id.
    collection: str | None = None
    # GeoJSON geometry.
    geometry: dict[str, Any] | None = None
    # Complete asset objects (type, roles, title, bands, ...).
    asset_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Links of the item.
    links: list[dict[str, Any]] = field(default_factory=list)
    # STAC version.
    stac_version: str = "1.0.0"
    # Extension schema URLs.
    stac_extensions: list[str] = field(default_factory=list)

    # Parse an item dictionary; relative hrefs are resolved against base.
    @classmethod
    def from_dict(cls, data: dict[str, Any], base: str | None = None) -> STACItem:
        # Items are GeoJSON Features.
        if data.get("type") != "Feature":
            # Explain the problem.
            raise ValueError(f"a STAC item must be a GeoJSON Feature, got {data.get('type')!r}")
        # Ids are required.
        if not isinstance(data.get("id"), str) or not data["id"]:
            # Explain the problem.
            raise ValueError("a STAC item needs a non-empty string id")
        # Members that must have a JSON object, array or string type.
        _check_item_members(data)
        # Properties object.
        props = dict(data.get("properties") or {})
        # Acquisition time.
        instant = props.get("datetime")
        # A null datetime requires an interval.
        if instant is None and not (props.get("start_datetime") and props.get("end_datetime")):
            # Explain the rule of the specification.
            raise ValueError(f"item {data['id']}: datetime is null without start/end_datetime")
        # Self link, used as the base of relative hrefs.
        self_links = [lk.get("href") for lk in data.get("links", []) if lk.get("rel") == "self"]
        # The first self link, if any.
        self_href = self_links[0] if self_links else None
        # Base of relative hrefs.
        root = base or self_href
        # Geometry of the item.
        geometry = data.get("geometry")
        # Box from the item or from the geometry.
        box = data.get("bbox") or (geojson_bounds(geometry) if geometry else None)
        # Items with a geometry need a box.
        if box is None:
            # Explain the problem.
            raise ValueError(f"item {data['id']}: no bbox and no geometry")
        # Only 2-D and 3-D boxes of finite numbers are valid.
        _check_bbox(data["id"], box)
        # Horizontal box.
        west, south, east, north = box if len(box) == 4 else (box[0], box[1], box[3], box[4])
        # Full asset objects with resolved hrefs.
        info = {
            key: {**asset, "href": resolve_href(asset.get("href", ""), root)}  # Resolved asset.
            for key, asset in (data.get("assets") or {}).items()  # Every asset.
        }  # End of the assets.
        # Item record.
        return cls(
            id=data["id"],  # Id.
            bbox=(float(west), float(south), float(east), float(north)),  # Box.
            datetime=parse_datetime(instant) if instant else None,  # Time.
            properties=props,  # Properties.
            assets={key: asset["href"] for key, asset in info.items()},  # Hrefs.
            collection=data.get("collection"),  # Collection.
            geometry=geometry,  # Geometry.
            asset_info=info,  # Asset objects.
            links=list(data.get("links") or []),  # Links.
            stac_version=str(data.get("stac_version", "1.0.0")),  # Version.
            stac_extensions=list(data.get("stac_extensions") or []),  # Extensions.
        )  # End of the item.

    # Item dictionary.
    def to_dict(self) -> dict[str, Any]:
        # GeoJSON Feature with the STAC members.
        data: dict[str, Any] = {
            "type": "Feature",  # GeoJSON type.
            "stac_version": self.stac_version,  # Version.
            "stac_extensions": list(self.stac_extensions),  # Extensions.
            "id": self.id,  # Id.
            "geometry": self.geometry,  # Geometry.
            "bbox": list(self.bbox),  # Box.
            "properties": dict(self.properties),  # Properties.
            "links": list(self.links),  # Links.
            "assets": {k: dict(v) for k, v in self.asset_info.items()},  # Assets.
        }  # End of the item.
        # Optional collection.
        if self.collection:
            # Store it.
            data["collection"] = self.collection
        # Return the dictionary.
        return data

    # Time interval of the item (start, end).
    @property
    def interval(self) -> tuple[datetime | None, datetime | None]:
        # Explicit start and end times win.
        start = self.properties.get("start_datetime")
        # End time.
        end = self.properties.get("end_datetime")
        # Interval items.
        if start or end:
            # Parse the ends.
            return (parse_datetime(start) if start else None, parse_datetime(end) if end else None)
        # Instant items.
        return (self.datetime, self.datetime)

    # Cloud cover in percent from the eo extension, None when absent.
    @property
    def cloud_cover(self) -> float | None:
        # Property of the eo extension.
        value = self.properties.get("eo:cloud_cover")
        # Convert numbers.
        return float(value) if value is not None else None

    # Asset keys with a role (for example "data" or "thumbnail") and/or media type.
    def find_assets(self, role: str | None = None, media_type: str | None = None) -> list[str]:
        # Keys found.
        keys = []
        # Visit every asset.
        for key, asset in self.asset_info.items():
            # Role filter.
            role_ok = role is None or role in asset.get("roles", [])
            # Media type filter (prefix, so "image/tiff" matches its profiles).
            type_ok = media_type is None or str(asset.get("type", "")).startswith(media_type)
            # Keep matching assets.
            if role_ok and type_ok:
                # Store the key.
                keys.append(key)
        # Return the keys.
        return keys


# One STAC collection.
@dataclass
class STACCollection:
    # Collection id.
    id: str
    # Description.
    description: str = ""
    # Licence identifier.
    license: str = "proprietary"
    # Spatial extent boxes.
    bboxes: list[list[float]] = field(default_factory=list)
    # Temporal extent intervals (RFC 3339 texts or None).
    intervals: list[list[str | None]] = field(default_factory=list)
    # Title.
    title: str = ""
    # Links.
    links: list[dict[str, Any]] = field(default_factory=list)
    # Summaries of item properties.
    summaries: dict[str, Any] = field(default_factory=dict)

    # Parse a collection dictionary.
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> STACCollection:
        # Collections have type Collection.
        if data.get("type") != "Collection":
            # Explain the problem.
            raise ValueError(f"not a STAC collection: type {data.get('type')!r}")
        # Extent object.
        extent = data.get("extent") or {}
        # Collection record.
        return cls(
            id=str(data["id"]),  # Id.
            description=str(data.get("description", "")),  # Description.
            license=str(data.get("license", "proprietary")),  # Licence.
            bboxes=list(extent.get("spatial", {}).get("bbox", [])),  # Boxes.
            intervals=list(extent.get("temporal", {}).get("interval", [])),  # Intervals.
            title=str(data.get("title", "")),  # Title.
            links=list(data.get("links") or []),  # Links.
            summaries=dict(data.get("summaries") or {}),  # Summaries.
        )  # End of the collection.

    # Overall temporal extent (start, end); open ends are None.
    @property
    def temporal_extent(self) -> tuple[datetime | None, datetime | None]:
        # The first interval is the overall extent.
        if not self.intervals:
            # Unbounded.
            return (None, None)
        # Parse it.
        return parse_datetime_range(tuple(self.intervals[0]))


# Compare a property value with the query extension operators.
def _matches(value: Any, condition: dict[str, Any]) -> bool:
    # Every operator must hold.
    for op, target in condition.items():
        # Unknown operators are errors.
        if op not in QUERY_OPERATORS:
            # Explain the accepted operators.
            raise ValueError(f"unknown query operator {op!r}; use {', '.join(QUERY_OPERATORS)}")
        # Missing properties never match except neq.
        if value is None:
            # Only inequality holds for missing values.
            if op != "neq":
                # No match.
                return False
            # Next operator.
            continue
        # Result of the operator.
        ok = QUERY_OPERATORS[op](value, target)
        # One failing operator rejects the item.
        if not ok:
            # No match.
            return False
    # Every operator held.
    return True


# Whether two intervals overlap; None ends are unbounded.
def _overlaps(a: tuple[Any, Any], b: tuple[Any, Any]) -> bool:
    # a starts before b ends and b starts before a ends.
    before = a[0] is None or b[1] is None or a[0] <= b[1]
    # Symmetric condition.
    after = b[0] is None or a[1] is None or b[0] <= a[1]
    # Both must hold.
    return before and after


# Offline item search with the semantics of the STAC API.
def filter_items(
    items: Iterable[STACItem],  # Items to search.
    bbox: Any = None,  # Box (west, south, east, north).
    datetime_range: Any = None,  # "a/b", "../b", a pair or an instant.
    collections: list[str] | None = None,  # Collection ids.
    ids: list[str] | None = None,  # Item ids.
    query: dict[str, dict[str, Any]] | None = None,  # Query extension conditions.
    max_cloud_cover: float | None = None,  # Maximum eo:cloud_cover in percent.
    limit: int | None = None,  # Maximum number of items.
) -> list[STACItem]:  # Matching items in input order.
    # Parsed interval.
    interval = parse_datetime_range(datetime_range) if datetime_range is not None else None
    # Matching items.
    found: list[STACItem] = []
    # Visit every item.
    for item in items:
        # Box filter.
        if bbox is not None and not bbox_intersects(item.bbox, bbox):
            # Skip the item.
            continue
        # Time filter.
        if interval is not None and not _overlaps(item.interval, interval):
            # Skip the item.
            continue
        # Collection filter.
        if collections is not None and item.collection not in collections:
            # Skip the item.
            continue
        # Id filter.
        if ids is not None and item.id not in ids:
            # Skip the item.
            continue
        # Cloud cover filter; items without cloud cover are kept.
        cover = item.cloud_cover
        # Compare with the maximum.
        if max_cloud_cover is not None and cover is not None and cover > max_cloud_cover:
            # Skip the item.
            continue
        # Property conditions.
        conditions = (query or {}).items()
        # Every condition must hold.
        if not all(_matches(item.properties.get(k), c) for k, c in conditions):
            # Skip the item.
            continue
        # Keep the item.
        found.append(item)
        # Stop at the limit.
        if limit is not None and len(found) >= limit:
            # Enough items.
            break
    # Return the items.
    return found


# Read a JSON file.
def _read_json(path: str | Path) -> dict[str, Any]:
    # Parse the file.
    with open(path, encoding="utf-8") as handle:
        # Document.
        return json.load(handle)


# Read an item from a JSON file.
def read_stac_item(path: str | Path) -> STACItem:
    # Parse and resolve hrefs against the file.
    return STACItem.from_dict(_read_json(path), base=str(Path(path).resolve()))


# Read a collection from a JSON file.
def read_stac_collection(path: str | Path) -> STACCollection:
    # Parse the file.
    return STACCollection.from_dict(_read_json(path))


# Items of a local static catalogue or collection, following child and item links.
def walk_catalog(path: str | Path, max_depth: int = 16) -> Iterator[STACItem]:
    # Files already visited, to stop cycles.
    seen: set[Path] = set()
    # Files to visit with their depth.
    stack = [(Path(path).resolve(), 0)]
    # Depth-first traversal.
    while stack:
        # Next file.
        current, depth = stack.pop()
        # Skip visited files.
        if current in seen:
            # Next file.
            continue
        # Remember it.
        seen.add(current)
        # Document of the file.
        doc = _read_json(current)
        # Items are yielded.
        if doc.get("type") == "Feature":
            # Parse the item.
            yield STACItem.from_dict(doc, base=str(current))
            # Items have no children.
            continue
        # Stop descending at the maximum depth.
        if depth >= max_depth:
            # Next file.
            continue
        # Child catalogues and items, in reverse so that the stack keeps file order.
        for link in reversed(doc.get("links", [])):
            # Only local child and item links.
            if link.get("rel") in ("child", "item") and "://" not in str(link.get("href", "")):
                # Queue the target.
                stack.append((Path(resolve_href(link["href"], str(current))), depth + 1))


# Default HTTP transport with requests.
def _requests_transport(headers: dict[str, str], timeout: float) -> Transport:
    # Send one request and decode the JSON response.
    def send(method: str, url: str, body: dict[str, Any] | None) -> dict[str, Any]:
        # Imported lazily.
        import requests

        # Perform the request.
        response = requests.request(method, url, json=body, headers=headers, timeout=timeout)
        # Raise for HTTP errors.
        response.raise_for_status()
        # Decoded JSON.
        return response.json()

    # Return the transport.
    return send


# Client of a STAC API.
@dataclass
class STACClient:
    # Root URL of the API.
    url: str
    # HTTP headers, for example authorisation.
    headers: dict[str, str] = field(default_factory=dict)
    # Timeout per request in seconds.
    timeout: float = 30.0
    # Transport; None uses requests.
    transport: Transport | None = None
    # Maximum number of result pages.
    max_pages: int = 100

    # Send a request with the configured transport.
    def _send(self, method: str, url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        # Transport of the client.
        send = self.transport or _requests_transport(self.headers, self.timeout)
        # Response document.
        return send(method, url, body)

    # Search items with POST /search, following "next" links.
    def search(
        self,  # The client.
        bbox: tuple[float, float, float, float] | None = None,  # Box.
        datetime_range: Any = None,  # Interval.
        collections: list[str] | None = None,  # Collection ids.
        limit: int = 100,  # Maximum number of items.
        query: dict[str, Any] | None = None,  # Query extension conditions.
        ids: list[str] | None = None,  # Item ids.
    ) -> Iterator[STACItem]:  # Items.
        # Request body.
        body: dict[str, Any] = {"limit": min(limit, 1000)}
        # Box.
        if bbox is not None:
            # Store it.
            body["bbox"] = [float(v) for v in bbox]
        # Interval in the "a/b" form with open ends as "..".
        if datetime_range is not None:
            # Parsed interval.
            start, end = parse_datetime_range(datetime_range)
            # Text of an end.
            text = [t.isoformat().replace("+00:00", "Z") if t else ".." for t in (start, end)]
            # Store it.
            body["datetime"] = f"{text[0]}/{text[1]}"
        # Collections.
        if collections:
            # Store them.
            body["collections"] = list(collections)
        # Ids.
        if ids:
            # Store them.
            body["ids"] = list(ids)
        # Query conditions.
        if query:
            # Store them.
            body["query"] = query
        # First request.
        method, url, payload = "POST", f"{self.url.rstrip('/')}/search", body
        # Items returned so far.
        returned = 0
        # Follow the pages.
        for _ in range(self.max_pages):
            # Page of results.
            page = self._send(method, url, payload)
            # Items of the page.
            for feature in page.get("features", []):
                # Parse the item, resolving against the page URL.
                yield STACItem.from_dict(feature, base=url)
                # Count it.
                returned += 1
                # Stop at the limit.
                if returned >= limit:
                    # Done.
                    return
            # Link to the next page.
            nxt = next((lk for lk in page.get("links", []) if lk.get("rel") == "next"), None)
            # Last page.
            if nxt is None:
                # Done.
                return
            # Next request as described by the link.
            method = str(nxt.get("method", "GET")).upper()
            # URL of the next page.
            url = nxt["href"]
            # Body of POST links; merge requests extend the previous body.
            payload = {**payload, **nxt.get("body", {})} if nxt.get("merge") else nxt.get("body")

    # Collections of the API.
    def collections(self) -> list[STACCollection]:
        # GET /collections.
        doc = self._send("GET", f"{self.url.rstrip('/')}/collections")
        # Parse every collection.
        return [STACCollection.from_dict(c) for c in doc.get("collections", [])]


# Search a STAC API and return a list of items.
def search_stac(
    url: str,  # Root URL of the API.
    bbox: tuple[float, float, float, float] | None = None,  # Box.
    datetime_range: Any = None,  # Interval.
    collections: list[str] | None = None,  # Collection ids.
    limit: int = 100,  # Maximum number of items.
    **kwargs: Any,  # Other options of STACClient.search.
) -> list[STACItem]:  # Items.
    # Client of the API.
    client = STACClient(url=url)
    # Collect the results.
    return list(client.search(bbox, datetime_range, collections, limit, **kwargs))


# Read a raster asset of an item with the GeoTIFF reader.
def load_from_stac(item: STACItem, asset_key: str = "visual", **kwargs: Any) -> Any:
    # GeoTIFF reader.
    from unbihexium.io.geotiff import read_cog

    # Href of the asset.
    href = item.assets.get(asset_key)
    # Unknown keys.
    if not href:
        # List the available keys.
        raise KeyError(f"asset {asset_key!r} not found; available: {sorted(item.assets)}")
    # Read the raster.
    return read_cog(href, **kwargs)


# =============================================================================
# End of module src/unbihexium/io/stac.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
