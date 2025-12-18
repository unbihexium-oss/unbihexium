"""Product type definitions for geospatial products."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ProductType(str, Enum):
    """Types of geospatial products."""

    RASTER = "raster"
    VECTOR = "vector"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    INDEX = "index"
    CHANGE = "change"
    DEM = "dem"
    DSM = "dsm"
    MOSAIC = "mosaic"
    COMPOSITE = "composite"


@dataclass
class ProductMetadata:
    """Metadata for a geospatial product."""

    product_id: str
    product_type: ProductType
    name: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    crs: str = "EPSG:4326"
    bounds: tuple[float, float, float, float] | None = None
    resolution: float | None = None
    source_scenes: list[str] = field(default_factory=list)
    processing_chain: list[str] = field(default_factory=list)
    quality_score: float | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class Product:
    """Geospatial product container."""

    data: Any = None
    metadata: ProductMetadata | None = None
    source: str | Path | None = None

    @property
    def product_type(self) -> ProductType | None:
        if self.metadata:
            return self.metadata.product_type
        return None

    @property
    def product_id(self) -> str:
        if self.metadata:
            return self.metadata.product_id
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_type": self.product_type.value if self.product_type else None,
            "metadata": {
                "name": self.metadata.name if self.metadata else "",
                "description": self.metadata.description if self.metadata else "",
                "crs": self.metadata.crs if self.metadata else "EPSG:4326",
                "bounds": self.metadata.bounds if self.metadata else None,
            },
        }

    @classmethod
    def create(
        cls,
        product_id: str,
        product_type: ProductType,
        data: Any,
        **kwargs: Any,
    ) -> Product:
        metadata = ProductMetadata(
            product_id=product_id,
            product_type=product_type,
            **kwargs,
        )
        return cls(data=data, metadata=metadata)
