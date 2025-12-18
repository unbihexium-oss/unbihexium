"""GeoJSON IO adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_geojson(path: str | Path) -> dict[str, Any]:
    """Read a GeoJSON file.

    Args:
        path: Path to GeoJSON file.

    Returns:
        GeoJSON dict (FeatureCollection or Feature).
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_geojson(
    path: str | Path,
    data: dict[str, Any],
    indent: int | None = 2,
) -> Path:
    """Write data to GeoJSON file.

    Args:
        path: Output path.
        data: GeoJSON dict.
        indent: JSON indentation.

    Returns:
        Path to written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)

    return path


def features_to_geojson(
    features: list[dict[str, Any]],
    crs: str | None = None,
) -> dict[str, Any]:
    """Convert list of features to GeoJSON FeatureCollection.

    Args:
        features: List of GeoJSON features.
        crs: Optional CRS string.

    Returns:
        GeoJSON FeatureCollection.
    """
    fc: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": features,
    }

    if crs:
        fc["crs"] = {
            "type": "name",
            "properties": {"name": crs},
        }

    return fc


def geometry_to_feature(
    geometry: dict[str, Any],
    properties: dict[str, Any] | None = None,
    feature_id: str | None = None,
) -> dict[str, Any]:
    """Create a GeoJSON feature from geometry.

    Args:
        geometry: GeoJSON geometry dict.
        properties: Feature properties.
        feature_id: Optional feature ID.

    Returns:
        GeoJSON Feature.
    """
    feature: dict[str, Any] = {
        "type": "Feature",
        "geometry": geometry,
        "properties": properties or {},
    }

    if feature_id:
        feature["id"] = feature_id

    return feature
