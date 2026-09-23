# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/core/product.py
# Title       : Derived products with metadata, checksums and STAC items
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy; saving rasters and
#               vectors requires rasterio and GeoPandas, STAC bounds pyproj
# =============================================================================
#
# Abstract
# --------
# A Product is the output of a processing chain (a classification map, an
# index, a mosaic, detected objects, ...) together with its metadata:
#
#   ProductMetadata   identifier, type, CRS, bounds, resolution, source
#                     scenes, processing chain, quality score, licence
#   Product.create    wraps data and fills CRS, bounds and resolution from a
#                     Raster or Vector
#   Product.save      writes the data (Cloud Optimized GeoTIFF for rasters,
#                     GeoJSON or GeoParquet for vectors, .npy for arrays)
#                     and product.json with the SHA-256 digest of the data
#   Product.load      reads a saved product back and verifies the digest
#   to_stac_item      a SpatioTemporal Asset Catalog (STAC) 1.0.0 Item with
#                     the projection extension
#
# STAC geometry and bbox are in WGS 84 longitude and latitude (RFC 7946);
# bounds in other CRSs are transformed with pyproj, densified along the
# edges so that curved edges are enclosed.
#
# References
# ----------
#   STAC contributors (2021). SpatioTemporal Asset Catalog specification,
#     version 1.0.0. https://stacspec.org
#   Butler, H., Daly, M., Doyle, A., Gillies, S., Hagen, S., Schaub, T.
#     (2016). The GeoJSON format. IETF RFC 7946.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON serialisation.
import json

# Metadata dictionaries.
from collections.abc import Mapping

# Record containers.
from dataclasses import dataclass, field

# Timestamps.
from datetime import datetime, timezone

# Product types.
from enum import Enum

# File paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Data digests.
from unbihexium.core.evidence import sha256_file

# Raster products.
from unbihexium.core.raster import Raster

# Vector products.
from unbihexium.core.vector import Vector

# STAC version written by to_stac_item.
STAC_VERSION = "1.0.0"

# Schema of the STAC projection extension.
STAC_PROJECTION = "https://stac-extensions.github.io/projection/v1.1.0/schema.json"

# Media type of Cloud Optimized GeoTIFF assets.
COG_MEDIA_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"


# Kinds of products.
class ProductType(str, Enum):
    # Generic raster.
    RASTER = "raster"
    # Generic vector.
    VECTOR = "vector"
    # Detected objects.
    DETECTION = "detection"
    # Segmentation map.
    SEGMENTATION = "segmentation"
    # Classification map.
    CLASSIFICATION = "classification"
    # Spectral index.
    INDEX = "index"
    # Change map.
    CHANGE = "change"
    # Digital elevation model (bare earth).
    DEM = "dem"
    # Digital surface model (with objects).
    DSM = "dsm"
    # Mosaic of several scenes.
    MOSAIC = "mosaic"
    # Temporal composite.
    COMPOSITE = "composite"


# Current time in UTC.
def _now() -> datetime:
    # Timezone-aware timestamp.
    return datetime.now(timezone.utc)


# Metadata of a product.
@dataclass
class ProductMetadata:
    # Identifier.
    product_id: str
    # Kind of product.
    product_type: ProductType
    # Human-readable name.
    name: str = ""
    # Description.
    description: str = ""
    # Creation time.
    created_at: datetime = field(default_factory=_now)
    # Version of the product.
    version: str = "1.0.0"
    # Coordinate reference system.
    crs: str = "EPSG:4326"
    # Bounds (min x, min y, max x, max y) in the CRS.
    bounds: tuple[float, float, float, float] | None = None
    # Pixel size in CRS units.
    resolution: float | None = None
    # Identifiers of the input scenes.
    source_scenes: list[str] = field(default_factory=list)
    # Processing steps in order.
    processing_chain: list[str] = field(default_factory=list)
    # Quality score in [0, 1], for example an accuracy.
    quality_score: float | None = None
    # Licence of the product, as an SPDX identifier.
    license: str = ""
    # Free-form tags.
    tags: dict[str, str] = field(default_factory=dict)

    # Validate the fields.
    def __post_init__(self) -> None:
        # Accept the type as a string.
        self.product_type = ProductType(self.product_type)
        # The identifier names the product.
        if not self.product_id:
            # Explain the problem.
            raise ValueError("product_id must not be empty")
        # Scores are fractions.
        if self.quality_score is not None and not 0.0 <= self.quality_score <= 1.0:
            # Explain the problem.
            raise ValueError(f"quality_score must be in [0, 1], got {self.quality_score}")

    # Plain dictionary for JSON output.
    def to_dict(self) -> dict[str, Any]:
        # One entry per field.
        return {
            "product_id": self.product_id,  # Identifier.
            "product_type": self.product_type.value,  # Kind.
            "name": self.name,  # Name.
            "description": self.description,  # Description.
            "created_at": self.created_at.isoformat(),  # Creation time.
            "version": self.version,  # Version.
            "crs": self.crs,  # Coordinate system.
            "bounds": list(self.bounds) if self.bounds else None,  # Bounds.
            "resolution": self.resolution,  # Pixel size.
            "source_scenes": list(self.source_scenes),  # Inputs.
            "processing_chain": list(self.processing_chain),  # Steps.
            "quality_score": self.quality_score,  # Quality.
            "license": self.license,  # Licence.
            "tags": dict(self.tags),  # Tags.
        }  # End of the dictionary.

    # Metadata from a dictionary written by to_dict.
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProductMetadata:
        # Creation time, when present.
        created = data.get("created_at")
        # Bounds, when present.
        bounds = data.get("bounds")
        # Build the metadata.
        return cls(
            product_id=data["product_id"],  # Identifier.
            product_type=ProductType(data["product_type"]),  # Kind.
            name=data.get("name", ""),  # Name.
            description=data.get("description", ""),  # Description.
            created_at=datetime.fromisoformat(created) if created else _now(),  # Time.
            version=data.get("version", "1.0.0"),  # Version.
            crs=data.get("crs", "EPSG:4326"),  # Coordinate system.
            bounds=tuple(bounds) if bounds else None,  # type: ignore[arg-type]
            resolution=data.get("resolution"),  # Pixel size.
            source_scenes=list(data.get("source_scenes", [])),  # Inputs.
            processing_chain=list(data.get("processing_chain", [])),  # Steps.
            quality_score=data.get("quality_score"),  # Quality.
            license=data.get("license", ""),  # Licence.
            tags=dict(data.get("tags", {})),  # Tags.
        )  # End of the metadata.


# Georeferencing of raster and vector data: CRS, bounds and resolution.
def _georeference(data: Any) -> dict[str, Any]:
    # Rasters know all three.
    if isinstance(data, Raster):
        # Pixel size of north-up rasters.
        a, b, _, d, e, _ = data.transform
        # Rotated rasters have no single pixel size.
        resolution = abs(a) if b == 0 and d == 0 and abs(a) == abs(e) else None
        # Values from the raster.
        return {"crs": data.crs, "bounds": data.bounds, "resolution": resolution}
    # Vectors know the CRS and bounds.
    if isinstance(data, Vector):
        # Values from the vector.
        return {"crs": data.crs, "bounds": data.bounds}
    # Other data carry no georeferencing.
    return {}


# A derived product: data and metadata.
@dataclass
class Product:
    # Raster, Vector, NumPy array or other data.
    data: Any = None
    # Metadata.
    metadata: ProductMetadata | None = None
    # File or directory the product was read from.
    source: str | Path | None = None

    # Kind of product.
    @property
    def product_type(self) -> ProductType | None:
        # From the metadata.
        return self.metadata.product_type if self.metadata else None

    # Identifier.
    @property
    def product_id(self) -> str:
        # From the metadata.
        return self.metadata.product_id if self.metadata else ""

    # Metadata; raises when there is none.
    def _meta(self) -> ProductMetadata:
        # Products without metadata cannot be described.
        if self.metadata is None:
            # Explain the problem.
            raise ValueError("product has no metadata")
        # The metadata.
        return self.metadata

    # Append a step to the processing chain.
    def add_processing_step(self, step: str) -> None:
        # Record the step.
        self._meta().processing_chain.append(step)

    # Plain dictionary for JSON output.
    def to_dict(self) -> dict[str, Any]:
        # Product type as text.
        kind = self.product_type.value if self.product_type else None
        # Identifier, type and metadata.
        return {
            "product_id": self.product_id,  # Identifier.
            "product_type": kind,  # Kind.
            "metadata": self.metadata.to_dict() if self.metadata else None,  # Metadata.
        }  # End of the dictionary.

    # Product with metadata filled from the data where not given.
    @classmethod
    def create(
        cls,  # The class.
        product_id: str,  # Identifier.
        product_type: ProductType | str,  # Kind.
        data: Any,  # Product data.
        **kwargs: Any,  # Further metadata fields.
    ) -> Product:  # The product.
        # Georeferencing of the data, overridden by explicit arguments.
        fields = {**_georeference(data), **kwargs}
        # Kind as an enum member.
        kind = ProductType(product_type)
        # Metadata.
        metadata = ProductMetadata(product_id=product_id, product_type=kind, **fields)
        # Build the product.
        return cls(data=data, metadata=metadata)

    # Write the data and product.json into a directory; returns the JSON path.
    def save(self, directory: str | Path, vector_format: str = "geojson") -> Path:
        # Metadata to write.
        meta = self._meta()
        # Output directory.
        folder = Path(directory)
        # Create it.
        folder.mkdir(parents=True, exist_ok=True)
        # Rasters become Cloud Optimized GeoTIFFs.
        if isinstance(self.data, Raster):
            # Write the raster.
            data_path = self.data.to_cog(folder / f"{meta.product_id}.tif")
            # Kind of data.
            kind = "raster"
        # Vectors become GeoJSON or GeoParquet.
        elif isinstance(self.data, Vector):
            # File extension.
            suffix = ".parquet" if vector_format == "parquet" else ".geojson"
            # Write the vector.
            data_path = self.data.to_file(folder / f"{meta.product_id}{suffix}")
            # Kind of data.
            kind = "vector"
        # Arrays become NumPy files.
        elif isinstance(self.data, np.ndarray):
            # Output file.
            data_path = folder / f"{meta.product_id}.npy"
            # Write the array without pickling.
            np.save(data_path, self.data, allow_pickle=False)
            # Kind of data.
            kind = "array"
        # Other data cannot be saved.
        else:
            # Explain the problem.
            raise TypeError(f"cannot save product data of type {type(self.data).__name__}")
        # Description of the data file.
        entry = {"path": data_path.name, "kind": kind, "sha256": sha256_file(data_path)}
        # Product document.
        document = {"metadata": meta.to_dict(), "data": entry}
        # Metadata file.
        json_path = folder / "product.json"
        # Write it.
        json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        # Return its path.
        return json_path

    # Read a product written by save; verifies the SHA-256 digest of the data.
    @classmethod
    def load(cls, path: str | Path, verify: bool = True) -> Product:
        # product.json itself or its directory.
        json_path = Path(path) / "product.json" if Path(path).is_dir() else Path(path)
        # Parse the document.
        document = json.loads(json_path.read_text(encoding="utf-8"))
        # Data description.
        entry = document["data"]
        # Data file next to the JSON file.
        data_path = json_path.parent / entry["path"]
        # Compare the digest.
        if verify and sha256_file(data_path) != entry["sha256"]:
            # The data changed after saving.
            raise ValueError(f"{data_path} does not match its recorded SHA-256 digest")
        # Read by kind.
        if entry["kind"] == "raster":
            # Raster in its stored dtype.
            data: Any = Raster.from_file(data_path, dtype=None)
        # Vectors.
        elif entry["kind"] == "vector":
            # Read the file.
            data = Vector.from_file(data_path)
        # Arrays.
        else:
            # Read without unpickling.
            data = np.load(data_path, allow_pickle=False)
        # Metadata of the product.
        metadata = ProductMetadata.from_dict(document["metadata"])
        # Product with its data.
        return cls(data=data, metadata=metadata, source=json_path)

    # WGS 84 bounds (west, south, east, north) of the product.
    def wgs84_bounds(self) -> tuple[float, float, float, float] | None:
        # Metadata with the bounds.
        meta = self._meta()
        # Products without bounds have no footprint.
        if meta.bounds is None:
            # Unknown footprint.
            return None
        # Bounds already in longitude and latitude.
        if meta.crs.upper() in ("EPSG:4326", "OGC:CRS84", ""):
            # As stored.
            return tuple(float(v) for v in meta.bounds)  # type: ignore[return-value]
        # Imported lazily.
        from pyproj import Transformer

        # Transformer with longitude first.
        transformer = Transformer.from_crs(meta.crs, "EPSG:4326", always_xy=True)
        # Densified transform of the edges.
        return tuple(transformer.transform_bounds(*meta.bounds, densify_pts=21))  # type: ignore[return-value]

    # STAC 1.0.0 Item of the product.
    def to_stac_item(self, asset_href: str | None = None) -> dict[str, Any]:
        # Metadata.
        meta = self._meta()
        # Footprint.
        bbox = self.wgs84_bounds()
        # Polygon of the footprint, closed counter-clockwise (RFC 7946).
        geometry = None
        # Build it when the bounds are known.
        if bbox is not None:
            # Corners.
            w, s, e, n = bbox
            # Closed ring.
            ring = [[w, s], [e, s], [e, n], [w, n], [w, s]]
            # GeoJSON polygon.
            geometry = {"type": "Polygon", "coordinates": [ring]}
        # Item properties.
        properties: dict[str, Any] = {
            "datetime": meta.created_at.isoformat(),  # Nominal time.
            "created": meta.created_at.isoformat(),  # Creation time.
            "title": meta.name or meta.product_id,  # Title.
            "description": meta.description,  # Description.
            "unbihexium:product_type": meta.product_type.value,  # Kind.
            "unbihexium:processing_chain": list(meta.processing_chain),  # Steps.
        }  # End of the properties.
        # Ground sample distance.
        if meta.resolution is not None:
            # Pixel size.
            properties["gsd"] = meta.resolution
        # Licence.
        if meta.license:
            # SPDX identifier.
            properties["license"] = meta.license
        # Extensions in use.
        extensions = []
        # EPSG code of the projection.
        if meta.crs.upper().startswith("EPSG:"):
            # Projection extension.
            extensions.append(STAC_PROJECTION)
            # Code as an integer.
            properties["proj:epsg"] = int(meta.crs.split(":")[1])
        # Assets.
        assets = {}
        # The data asset, when a location is given.
        if asset_href is not None:
            # Media type of the data.
            media = COG_MEDIA_TYPE if isinstance(self.data, Raster) else "application/geo+json"
            # Asset description.
            assets["data"] = {"href": asset_href, "type": media, "roles": ["data"]}
        # Item.
        return {
            "type": "Feature",  # GeoJSON feature.
            "stac_version": STAC_VERSION,  # Specification version.
            "stac_extensions": extensions,  # Extensions.
            "id": meta.product_id,  # Identifier.
            "geometry": geometry,  # Footprint.
            "bbox": list(bbox) if bbox else None,  # Bounding box.
            "properties": properties,  # Properties.
            "links": [],  # Links.
            "assets": assets,  # Assets.
        }  # End of the item.


# =============================================================================
# End of module src/unbihexium/core/product.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
