"""STAC (SpatioTemporal Asset Catalog) client and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator


@dataclass
class STACItem:
    """A STAC item representation."""

    id: str
    bbox: tuple[float, float, float, float]
    datetime: datetime | None
    properties: dict[str, Any] = field(default_factory=dict)
    assets: dict[str, str] = field(default_factory=dict)
    collection: str | None = None
    geometry: dict[str, Any] | None = None


@dataclass
class STACClient:
    """Client for STAC API endpoints."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)

    def search(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        datetime_range: tuple[str, str] | None = None,
        collections: list[str] | None = None,
        limit: int = 100,
        query: dict[str, Any] | None = None,
    ) -> Iterator[STACItem]:
        """Search STAC catalog.

        Args:
            bbox: Bounding box (west, south, east, north).
            datetime_range: Date range (start, end) in ISO format.
            collections: Collection IDs to search.
            limit: Maximum items to return.
            query: Additional query parameters.

        Yields:
            STACItem objects matching the search.
        """
        try:
            import requests
        except ImportError as e:
            raise ImportError("requests is required for STAC support") from e

        search_url = f"{self.url.rstrip('/')}/search"

        payload: dict[str, Any] = {"limit": limit}
        if bbox:
            payload["bbox"] = list(bbox)
        if datetime_range:
            payload["datetime"] = f"{datetime_range[0]}/{datetime_range[1]}"
        if collections:
            payload["collections"] = collections
        if query:
            payload["query"] = query

        response = requests.post(search_url, json=payload, headers=self.headers)
        response.raise_for_status()
        data = response.json()

        for feature in data.get("features", []):
            yield STACItem(
                id=feature["id"],
                bbox=tuple(feature.get("bbox", [0, 0, 0, 0])),
                datetime=datetime.fromisoformat(feature["properties"].get("datetime", "").replace("Z", "+00:00"))
                if feature["properties"].get("datetime")
                else None,
                properties=feature.get("properties", {}),
                assets={k: v.get("href", "") for k, v in feature.get("assets", {}).items()},
                collection=feature.get("collection"),
                geometry=feature.get("geometry"),
            )


def search_stac(
    url: str,
    bbox: tuple[float, float, float, float] | None = None,
    datetime_range: tuple[str, str] | None = None,
    collections: list[str] | None = None,
    limit: int = 100,
) -> list[STACItem]:
    """Convenience function to search a STAC catalog.

    Args:
        url: STAC API URL.
        bbox: Bounding box.
        datetime_range: Date range.
        collections: Collection IDs.
        limit: Max results.

    Returns:
        List of STAC items.
    """
    client = STACClient(url=url)
    return list(client.search(bbox=bbox, datetime_range=datetime_range, collections=collections, limit=limit))


def load_from_stac(item: STACItem, asset_key: str = "visual") -> Any:
    """Load raster data from a STAC item asset.

    Args:
        item: STAC item.
        asset_key: Asset key to load (e.g., 'visual', 'B04').

    Returns:
        Loaded data (uses COG reader).
    """
    from unbihexium.io.geotiff import read_cog

    asset_url = item.assets.get(asset_key)
    if not asset_url:
        available = list(item.assets.keys())
        raise KeyError(f"Asset '{asset_key}' not found. Available: {available}")

    return read_cog(asset_url)
