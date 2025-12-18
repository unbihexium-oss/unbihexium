"""Vector abstraction for geospatial vector data.

This module provides the core Vector class for handling geospatial vector data
with support for GeoJSON, Parquet, and various geometry operations.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


class GeometryType(str, Enum):
    """Supported geometry types."""

    POINT = "Point"
    MULTIPOINT = "MultiPoint"
    LINESTRING = "LineString"
    MULTILINESTRING = "MultiLineString"
    POLYGON = "Polygon"
    MULTIPOLYGON = "MultiPolygon"
    GEOMETRYCOLLECTION = "GeometryCollection"


@dataclass(frozen=True)
class VectorMetadata:
    """Metadata for a vector dataset.

    Attributes:
        crs: Coordinate reference system (EPSG code or WKT).
        geometry_type: Type of geometry in the dataset.
        feature_count: Number of features.
        bounds: Bounding box (minx, miny, maxx, maxy).
        columns: List of attribute column names.
        schema: Schema definition for attributes.
    """

    crs: str
    geometry_type: GeometryType | None = None
    feature_count: int = 0
    bounds: tuple[float, float, float, float] | None = None
    columns: list[str] = field(default_factory=list)
    schema: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "crs": self.crs,
            "geometry_type": self.geometry_type.value if self.geometry_type else None,
            "feature_count": self.feature_count,
            "bounds": self.bounds,
            "columns": self.columns,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorMetadata:
        """Create metadata from dictionary."""
        return cls(
            crs=data["crs"],
            geometry_type=GeometryType(data["geometry_type"])
            if data.get("geometry_type")
            else None,
            feature_count=data.get("feature_count", 0),
            bounds=tuple(data["bounds"]) if data.get("bounds") else None,
            columns=data.get("columns", []),
            schema=data.get("schema", {}),
        )


@dataclass
class Vector:
    """Core vector abstraction for geospatial vector data.

    Supports loading from GeoJSON, Shapefile, GeoParquet, and other formats.
    Provides spatial operations, filtering, and export capabilities.

    Attributes:
        data: GeoDataFrame or similar structure containing the vector data.
        metadata: Vector metadata including CRS, bounds, schema.
        source: Optional source path or URL.
    """

    data: Any | None = None  # GeoDataFrame
    metadata: VectorMetadata | None = None
    source: str | Path | None = None

    def __post_init__(self) -> None:
        """Validate and infer metadata if needed."""
        if self.data is not None and self.metadata is None:
            self._infer_metadata()

    def _infer_metadata(self) -> None:
        """Infer metadata from the data."""
        import geopandas as gpd

        if isinstance(self.data, gpd.GeoDataFrame):
            geom_type = None
            if not self.data.empty and self.data.geometry is not None:
                first_geom = self.data.geometry.iloc[0]
                if first_geom is not None:
                    geom_type = GeometryType(first_geom.geom_type)

            self.metadata = VectorMetadata(
                crs=str(self.data.crs) if self.data.crs else "EPSG:4326",
                geometry_type=geom_type,
                feature_count=len(self.data),
                bounds=tuple(self.data.total_bounds) if not self.data.empty else None,
                columns=[c for c in self.data.columns if c != "geometry"],
                schema={c: str(self.data[c].dtype) for c in self.data.columns if c != "geometry"},
            )

    @property
    def crs(self) -> str:
        """Return the CRS of the vector data."""
        if self.metadata is not None:
            return self.metadata.crs
        return "EPSG:4326"

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        """Return the bounding box of the vector data."""
        if self.metadata is not None:
            return self.metadata.bounds
        return None

    @property
    def feature_count(self) -> int:
        """Return the number of features."""
        if self.data is not None:
            return len(self.data)
        if self.metadata is not None:
            return self.metadata.feature_count
        return 0

    @classmethod
    def from_geodataframe(cls, gdf: Any) -> Vector:
        """Create a Vector from a GeoDataFrame.

        Args:
            gdf: A GeoDataFrame instance.

        Returns:
            A new Vector instance.
        """
        return cls(data=gdf)

    @classmethod
    def from_file(cls, path: str | Path) -> Vector:
        """Load vector data from a file.

        Args:
            path: Path to the vector file (GeoJSON, Shapefile, GeoParquet, etc.).

        Returns:
            A new Vector instance.
        """
        import geopandas as gpd

        path = Path(path)

        if path.suffix.lower() == ".parquet":
            gdf = gpd.read_parquet(path)
        else:
            gdf = gpd.read_file(path)

        return cls(data=gdf, source=path)

    @classmethod
    def from_geojson(cls, geojson: dict[str, Any] | str) -> Vector:
        """Create a Vector from GeoJSON.

        Args:
            geojson: GeoJSON dictionary or string.

        Returns:
            A new Vector instance.
        """
        import json

        import geopandas as gpd

        if isinstance(geojson, str):
            geojson = json.loads(geojson)

        gdf = gpd.GeoDataFrame.from_features(geojson["features"])
        if "crs" in geojson:
            gdf.set_crs(geojson["crs"]["properties"]["name"], inplace=True)
        else:
            gdf.set_crs("EPSG:4326", inplace=True)

        return cls(data=gdf)

    @classmethod
    def from_wkt(
        cls,
        wkt: str | list[str],
        crs: str = "EPSG:4326",
        properties: dict[str, list[Any]] | None = None,
    ) -> Vector:
        """Create a Vector from WKT geometry strings.

        Args:
            wkt: Single WKT string or list of WKT strings.
            crs: Coordinate reference system.
            properties: Optional dictionary of property arrays.

        Returns:
            A new Vector instance.
        """
        import geopandas as gpd
        from shapely import wkt as shapely_wkt

        if isinstance(wkt, str):
            wkt = [wkt]

        geometries = [shapely_wkt.loads(w) for w in wkt]
        gdf = gpd.GeoDataFrame(properties or {}, geometry=geometries, crs=crs)

        return cls(data=gdf)

    def to_file(
        self,
        path: str | Path,
        driver: str | None = None,
    ) -> Path:
        """Save vector data to a file.

        Args:
            path: Output file path.
            driver: GDAL driver name (auto-detected from extension if None).

        Returns:
            Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.data is None:
            msg = "Cannot save vector without data"
            raise ValueError(msg)

        if path.suffix.lower() == ".parquet":
            self.data.to_parquet(path)
        else:
            driver = driver or self._detect_driver(path)
            self.data.to_file(path, driver=driver)

        return path

    def _detect_driver(self, path: Path) -> str:
        """Detect GDAL driver from file extension."""
        ext = path.suffix.lower()
        drivers = {
            ".geojson": "GeoJSON",
            ".json": "GeoJSON",
            ".shp": "ESRI Shapefile",
            ".gpkg": "GPKG",
            ".fgb": "FlatGeobuf",
        }
        return drivers.get(ext, "GeoJSON")

    def to_geojson(self) -> dict[str, Any]:
        """Convert to GeoJSON dictionary.

        Returns:
            GeoJSON dictionary.
        """
        import json

        if self.data is None:
            return {"type": "FeatureCollection", "features": []}

        return json.loads(self.data.to_json())

    def to_wkt(self) -> list[str]:
        """Convert geometries to WKT strings.

        Returns:
            List of WKT strings.
        """
        if self.data is None:
            return []

        return [geom.wkt for geom in self.data.geometry]

    def filter(
        self,
        expression: str | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        geometry: BaseGeometry | None = None,
    ) -> Vector:
        """Filter vector data.

        Args:
            expression: Pandas query expression for attribute filtering.
            bounds: Bounding box for spatial filtering.
            geometry: Geometry for spatial filtering (intersects).

        Returns:
            A new filtered Vector.
        """
        if self.data is None:
            return Vector()

        filtered = self.data.copy()

        if expression:
            filtered = filtered.query(expression)

        if bounds:
            from shapely.geometry import box

            bbox = box(*bounds)
            filtered = filtered[filtered.intersects(bbox)]

        if geometry:
            filtered = filtered[filtered.intersects(geometry)]

        return Vector(data=filtered)

    def buffer(self, distance: float, resolution: int = 16) -> Vector:
        """Create buffer around geometries.

        Args:
            distance: Buffer distance in CRS units.
            resolution: Number of segments per quarter circle.

        Returns:
            A new Vector with buffered geometries.
        """
        if self.data is None:
            return Vector()

        buffered = self.data.copy()
        buffered["geometry"] = buffered.geometry.buffer(distance, resolution=resolution)

        return Vector(data=buffered)

    def simplify(self, tolerance: float, preserve_topology: bool = True) -> Vector:
        """Simplify geometries.

        Args:
            tolerance: Simplification tolerance.
            preserve_topology: Whether to preserve topology.

        Returns:
            A new Vector with simplified geometries.
        """
        if self.data is None:
            return Vector()

        simplified = self.data.copy()
        simplified["geometry"] = simplified.geometry.simplify(
            tolerance, preserve_topology=preserve_topology
        )

        return Vector(data=simplified)

    def centroid(self) -> Vector:
        """Get centroids of geometries.

        Returns:
            A new Vector with centroid points.
        """
        if self.data is None:
            return Vector()

        centroids = self.data.copy()
        centroids["geometry"] = centroids.geometry.centroid

        return Vector(data=centroids)

    def union(self) -> BaseGeometry:
        """Compute union of all geometries.

        Returns:
            Union geometry.
        """
        if self.data is None or self.data.empty:
            from shapely.geometry import Point

            return Point()

        return self.data.geometry.unary_union

    def intersection(self, other: Vector) -> Vector:
        """Compute intersection with another vector.

        Args:
            other: Another Vector to intersect with.

        Returns:
            A new Vector with intersection geometries.
        """
        if self.data is None or other.data is None:
            return Vector()

        import geopandas as gpd

        result = gpd.overlay(self.data, other.data, how="intersection")
        return Vector(data=result)

    def difference(self, other: Vector) -> Vector:
        """Compute difference with another vector.

        Args:
            other: Another Vector to subtract.

        Returns:
            A new Vector with difference geometries.
        """
        if self.data is None or other.data is None:
            return Vector()

        import geopandas as gpd

        result = gpd.overlay(self.data, other.data, how="difference")
        return Vector(data=result)

    def reproject(self, target_crs: str) -> Vector:
        """Reproject to a new CRS.

        Args:
            target_crs: Target coordinate reference system.

        Returns:
            A new reprojected Vector.
        """
        if self.data is None:
            return Vector()

        reprojected = self.data.to_crs(target_crs)
        return Vector(data=reprojected)

    def dissolve(
        self,
        by: str | list[str] | None = None,
        aggfunc: str = "first",
    ) -> Vector:
        """Dissolve geometries.

        Args:
            by: Column(s) to group by.
            aggfunc: Aggregation function for attributes.

        Returns:
            A new dissolved Vector.
        """
        if self.data is None:
            return Vector()

        dissolved = self.data.dissolve(by=by, aggfunc=aggfunc)
        dissolved = dissolved.reset_index()

        return Vector(data=dissolved)

    def area(self) -> NDArray[np.floating[Any]]:
        """Calculate area of geometries.

        Returns:
            Array of areas.
        """
        if self.data is None:
            return np.array([])

        return self.data.geometry.area.values

    def length(self) -> NDArray[np.floating[Any]]:
        """Calculate length of geometries.

        Returns:
            Array of lengths.
        """
        if self.data is None:
            return np.array([])

        return self.data.geometry.length.values

    def features(self) -> Iterator[dict[str, Any]]:
        """Iterate over features as dictionaries.

        Yields:
            Feature dictionaries with geometry and properties.
        """
        if self.data is None:
            return

        for idx, row in self.data.iterrows():
            properties = {k: v for k, v in row.items() if k != "geometry"}
            yield {
                "type": "Feature",
                "id": idx,
                "geometry": row.geometry.__geo_interface__,
                "properties": properties,
            }

    def __len__(self) -> int:
        """Return the number of features."""
        return self.feature_count

    def __repr__(self) -> str:
        """Return string representation."""
        geom_type = (
            self.metadata.geometry_type.value
            if self.metadata and self.metadata.geometry_type
            else "Unknown"
        )
        return f"Vector(features={self.feature_count}, geometry_type={geom_type}, crs={self.crs})"
