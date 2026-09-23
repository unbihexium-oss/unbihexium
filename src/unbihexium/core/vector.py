# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/core/vector.py
# Title       : Georeferenced vector container
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires GeoPandas, Shapely and pyproj;
#               rasterize and from_raster require rasterio
# =============================================================================
#
# Abstract
# --------
# Vector wraps a GeoDataFrame with its metadata (CRS, geometry type, feature
# count, bounds and attribute schema):
#
#   input/output   from_file (GeoJSON, GeoPackage, Shapefile, FlatGeobuf,
#                  GeoParquet), from_geojson, from_features, from_wkt,
#                  to_file, to_geojson, to_wkt, features
#   geometry       buffer, buffer_metres, simplify, centroid, union,
#                  intersection, difference, overlay, clip, dissolve,
#                  explode, make_valid, reproject, spatial_join, filter
#   measures       area and length, planar or geodesic on the WGS 84
#                  ellipsoid
#   raster links   rasterize (burn geometries into a raster grid) and
#                  from_raster (polygonise connected regions of equal value)
#
# Geodesic measures use the algorithms of Karney (2013) through pyproj.Geod;
# they are exact to round-off for any geometry size, whereas planar measures
# in a geographic CRS are in square degrees and meaningless. With
# geodesic=None the geodesic measure is used when the CRS is geographic.
#
# buffer_metres buffers geometries of any CRS by a distance in metres: the
# data are projected to the UTM zone of their centre (estimate_utm_crs),
# buffered there and projected back.
#
# References
# ----------
#   Karney, C. F. F. (2013). Algorithms for geodesics. Journal of Geodesy
#     87(1), 43-55.
#   Butler, H., Daly, M., Doyle, A., Gillies, S., Hagen, S., Schaub, T.
#     (2016). The GeoJSON format. IETF RFC 7946.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON text of GeoJSON documents.
import json

# Feature iterators.
from collections.abc import Iterable, Iterator, Mapping

# Metadata container.
from dataclasses import dataclass, field

# Geometry types.
from enum import Enum

# File paths.
from pathlib import Path

# Types used only for annotations.
from typing import TYPE_CHECKING, Any, Literal

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Imported for annotations only.
if TYPE_CHECKING:
    # Raster grid for rasterize and from_raster.
    from unbihexium.core.raster import Raster

# GDAL drivers by file extension.
DRIVERS = {
    ".geojson": "GeoJSON",  # GeoJSON.
    ".json": "GeoJSON",  # GeoJSON.
    ".shp": "ESRI Shapefile",  # Shapefile.
    ".gpkg": "GPKG",  # GeoPackage.
    ".fgb": "FlatGeobuf",  # FlatGeobuf.
}  # End of the drivers.


# Simple Features geometry types.
class GeometryType(str, Enum):
    # Single point.
    POINT = "Point"
    # Several points.
    MULTIPOINT = "MultiPoint"
    # Single line.
    LINESTRING = "LineString"
    # Several lines.
    MULTILINESTRING = "MultiLineString"
    # Single polygon.
    POLYGON = "Polygon"
    # Several polygons.
    MULTIPOLYGON = "MultiPolygon"
    # Mixed collection.
    GEOMETRYCOLLECTION = "GeometryCollection"


# Metadata of a vector dataset.
@dataclass(frozen=True)
class VectorMetadata:
    # Coordinate reference system, for example "EPSG:4326".
    crs: str
    # Geometry type when every feature has the same one, else None.
    geometry_type: GeometryType | None = None
    # Number of features.
    feature_count: int = 0
    # Bounds (min x, min y, max x, max y).
    bounds: tuple[float, float, float, float] | None = None
    # Attribute column names.
    columns: list[str] = field(default_factory=list)
    # Attribute dtypes by column.
    schema: dict[str, str] = field(default_factory=dict)

    # Plain dictionary for JSON output.
    def to_dict(self) -> dict[str, Any]:
        # Geometry type as text.
        geometry_type = self.geometry_type.value if self.geometry_type else None
        # One entry per field.
        return {
            "crs": self.crs,  # Coordinate system.
            "geometry_type": geometry_type,  # Geometry type.
            "feature_count": self.feature_count,  # Features.
            "bounds": list(self.bounds) if self.bounds else None,  # Bounds.
            "columns": list(self.columns),  # Columns.
            "schema": dict(self.schema),  # Dtypes.
        }  # End of the dictionary.

    # Metadata from a dictionary written by to_dict.
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VectorMetadata:
        # Geometry type, when present.
        kind = data.get("geometry_type")
        # Bounds, when present.
        bounds = data.get("bounds")
        # Build the metadata.
        return cls(
            crs=data["crs"],  # Coordinate system.
            geometry_type=GeometryType(kind) if kind else None,  # Geometry type.
            feature_count=int(data.get("feature_count", 0)),  # Features.
            bounds=tuple(bounds) if bounds else None,  # type: ignore[arg-type]
            columns=list(data.get("columns", [])),  # Columns.
            schema=dict(data.get("schema", {})),  # Dtypes.
        )  # End of the metadata.


# Metadata of a GeoDataFrame.
def _infer_metadata(gdf: Any) -> VectorMetadata:
    # Distinct geometry types, ignoring missing geometries.
    kinds = {str(k) for k in gdf.geom_type.dropna().unique()}
    # A single type is recorded.
    kind = GeometryType(kinds.pop()) if len(kinds) == 1 else None
    # Attribute columns.
    columns = [str(c) for c in gdf.columns if c != gdf.geometry.name]
    # Bounds of non-empty data.
    bounds = tuple(float(v) for v in gdf.total_bounds) if len(gdf) else None
    # Build the metadata.
    return VectorMetadata(
        crs=gdf.crs.to_string() if gdf.crs else "",  # Coordinate system.
        geometry_type=kind,  # Geometry type.
        feature_count=len(gdf),  # Features.
        bounds=bounds,  # type: ignore[arg-type]
        columns=columns,  # Columns.
        schema={c: str(gdf[c].dtype) for c in columns},  # Dtypes.
    )  # End of the metadata.


# Georeferenced vector data backed by a GeoDataFrame.
@dataclass
class Vector:
    # GeoDataFrame with the features.
    data: Any | None = None
    # CRS, geometry type, count, bounds and schema.
    metadata: VectorMetadata | None = None
    # File the data were read from.
    source: str | Path | None = None

    # Derive the metadata from the data.
    def __post_init__(self) -> None:
        # Metadata is derived whenever data are given without it.
        if self.data is not None and self.metadata is None:
            # Inferred metadata.
            self.metadata = _infer_metadata(self.data)

    # GeoDataFrame; raises when there is none.
    def _gdf(self) -> Any:
        # Vectors without data cannot be processed.
        if self.data is None:
            # Explain the problem.
            raise ValueError("vector has no data")
        # The GeoDataFrame.
        return self.data

    # Coordinate reference system.
    @property
    def crs(self) -> str:
        # From the metadata.
        return self.metadata.crs if self.metadata else ""

    # Bounds (min x, min y, max x, max y).
    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        # From the metadata.
        return self.metadata.bounds if self.metadata else None

    # Number of features.
    @property
    def feature_count(self) -> int:
        # From the data when present.
        if self.data is not None:
            # Rows of the GeoDataFrame.
            return len(self.data)
        # From the metadata otherwise.
        return self.metadata.feature_count if self.metadata else 0

    # Whether the CRS is geographic (longitude and latitude).
    @property
    def is_geographic(self) -> bool:
        # CRS object of the data.
        crs = self._gdf().crs
        # Unknown CRS counts as not geographic.
        return bool(crs is not None and crs.is_geographic)

    # Number of features.
    def __len__(self) -> int:
        # Feature count.
        return self.feature_count

    # Short description.
    def __repr__(self) -> str:
        # Geometry type or "Mixed".
        kind = self.metadata.geometry_type if self.metadata else None
        # Text of the type.
        text = kind.value if kind else "Mixed"
        # Count, type and CRS.
        return f"Vector(features={self.feature_count}, geometry_type={text}, crs={self.crs})"

    # Vector from a GeoDataFrame.
    @classmethod
    def from_geodataframe(cls, gdf: Any) -> Vector:
        # Wrap the data.
        return cls(data=gdf)

    # Vector from a file (GeoParquet by the .parquet extension).
    @classmethod
    def from_file(
        cls,  # The class.
        path: str | Path,  # File path.
        layer: str | None = None,  # Layer of multi-layer formats.
        bbox: tuple[float, float, float, float] | None = None,  # Spatial filter.
    ) -> Vector:  # The vector.
        # Imported lazily.
        import geopandas as gpd

        # Path object.
        path = Path(path)
        # GeoParquet.
        if path.suffix.lower() == ".parquet":
            # Read the table.
            gdf = gpd.read_parquet(path, bbox=bbox)
        # Every other format through GDAL.
        else:
            # Read the layer.
            gdf = gpd.read_file(path, layer=layer, bbox=bbox)
        # Wrap the data.
        return cls(data=gdf, source=path)

    # Vector from GeoJSON features (RFC 7946 coordinates are WGS 84).
    @classmethod
    def from_features(cls, features: Iterable[Mapping[str, Any]], crs: str = "EPSG:4326") -> Vector:
        # Imported lazily.
        import geopandas as gpd

        # GeoDataFrame from the features.
        gdf = gpd.GeoDataFrame.from_features(list(features), crs=crs)
        # Wrap the data.
        return cls(data=gdf)

    # Vector from a GeoJSON FeatureCollection given as a dictionary or text.
    @classmethod
    def from_geojson(cls, geojson: Mapping[str, Any] | str) -> Vector:
        # Parse text.
        document = json.loads(geojson) if isinstance(geojson, str) else geojson
        # Named CRS of pre-RFC 7946 documents, else WGS 84.
        crs = document.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
        # Build from the features.
        return cls.from_features(document.get("features", []), crs=crs)

    # Vector from WKT geometries with optional attribute columns.
    @classmethod
    def from_wkt(
        cls,  # The class.
        wkt: str | list[str],  # One or more WKT strings.
        crs: str = "EPSG:4326",  # Coordinate reference system.
        properties: Mapping[str, list[Any]] | None = None,  # Attribute columns.
    ) -> Vector:  # The vector.
        # Imported lazily.
        import geopandas as gpd
        from shapely import wkt as shapely_wkt  # WKT parser.

        # A single string becomes a list.
        texts = [wkt] if isinstance(wkt, str) else list(wkt)
        # Parsed geometries.
        geometries = [shapely_wkt.loads(t) for t in texts]
        # GeoDataFrame.
        gdf = gpd.GeoDataFrame(dict(properties or {}), geometry=geometries, crs=crs)
        # Wrap the data.
        return cls(data=gdf)

    # Polygons of connected regions of equal value in a raster band.
    @classmethod
    def from_raster(
        cls,  # The class.
        raster: Raster,  # Raster to polygonise.
        band: int = 1,  # 1-based band number.
        connectivity: Literal[4, 8] = 4,  # Pixel neighbourhood.
        include_nodata: bool = False,  # Also polygonise no-data regions.
    ) -> Vector:  # Polygons with a "value" column.
        # Imported lazily.
        import geopandas as gpd
        from rasterio.features import shapes  # Polygonisation.
        from shapely.geometry import shape  # GeoJSON to Shapely.

        # Band values.
        values = raster.select_bands([band]).require_data()[0]
        # rasterio polygonises only some dtypes; float32 keeps every value exactly.
        if values.dtype.kind == "f" or values.dtype.itemsize > 4:
            # Convert.
            values = values.astype(np.float32)
        # Mask of the pixels to polygonise.
        mask = None if include_nodata else raster.valid_mask()
        # Grid transform.
        transform = _affine(raster)
        # Polygons and values of the regions.
        pairs = list(shapes(values, mask=mask, connectivity=connectivity, transform=transform))
        # Attribute table and geometries.
        gdf = gpd.GeoDataFrame(
            {"value": [v for _, v in pairs]},  # Region values.
            geometry=[shape(g) for g, _ in pairs],  # Region polygons.
            crs=raster.crs or None,  # Coordinate system of the raster.
        )  # End of the table.
        # Wrap the data.
        return cls(data=gdf)

    # Write the data; the format follows the extension unless a driver is given.
    def to_file(self, path: str | Path, driver: str | None = None) -> Path:
        # Data to write.
        gdf = self._gdf()
        # Output path.
        path = Path(path)
        # Create the parent directory.
        path.parent.mkdir(parents=True, exist_ok=True)
        # GeoParquet.
        if path.suffix.lower() == ".parquet":
            # Write the table.
            gdf.to_parquet(path)
        # Every other format through GDAL.
        else:
            # Write with the driver of the extension.
            gdf.to_file(path, driver=driver or DRIVERS.get(path.suffix.lower(), "GeoJSON"))
        # Return the path.
        return path

    # GeoJSON FeatureCollection as a dictionary.
    def to_geojson(self) -> dict[str, Any]:
        # Empty collections for vectors without data.
        if self.data is None:
            # Empty collection.
            return {"type": "FeatureCollection", "features": []}
        # Serialise through GeoPandas.
        return json.loads(self.data.to_json())

    # Geometries as WKT strings.
    def to_wkt(self) -> list[str]:
        # Empty list for vectors without data.
        if self.data is None:
            # No geometries.
            return []
        # WKT of every geometry.
        return [g.wkt for g in self.data.geometry]

    # GeoJSON features one by one.
    def features(self) -> Iterator[dict[str, Any]]:
        # Nothing to yield without data.
        if self.data is None:
            # Stop.
            return
        # Name of the geometry column.
        name = self.data.geometry.name
        # Visit the rows.
        for index, row in self.data.iterrows():
            # Attributes without the geometry.
            properties = {k: v for k, v in row.items() if k != name}
            # Geometry as a GeoJSON mapping.
            geometry = row[name].__geo_interface__ if row[name] is not None else None
            # GeoJSON feature.
            yield {
                "type": "Feature",  # Feature object.
                "id": index,  # Row index.
                "geometry": geometry,  # Geometry.
                "properties": properties,  # Attributes.
            }  # End of the feature.

    # Copy with a new geometry column.
    def _with_geometry(self, geometry: Any) -> Vector:
        # Copy of the data.
        gdf = self._gdf().copy()
        # Replace the geometries.
        gdf = gdf.set_geometry(geometry)
        # New vector.
        return Vector(data=gdf, source=self.source)

    # Features selected by an attribute query, bounds or a geometry.
    def filter(
        self,  # This object.
        expression: str | None = None,  # pandas query on the attributes.
        bounds: tuple[float, float, float, float] | None = None,  # Intersecting box.
        geometry: Any = None,  # Intersecting Shapely geometry.
    ) -> Vector:  # Selected features.
        # Imported lazily.
        from shapely.geometry import box

        # Copy of the data.
        gdf = self._gdf()
        # Attribute query.
        if expression:
            # Rows that satisfy the query.
            gdf = gdf.query(expression)
        # Bounding box.
        if bounds is not None:
            # Rows that intersect the box.
            gdf = gdf[gdf.intersects(box(*bounds))]
        # Geometry.
        if geometry is not None:
            # Rows that intersect the geometry.
            gdf = gdf[gdf.intersects(geometry)]
        # New vector.
        return Vector(data=gdf.copy(), source=self.source)

    # Buffer by a distance in CRS units.
    def buffer(self, distance: float, resolution: int = 16) -> Vector:
        # Buffered geometries (resolution is the segments per quarter circle).
        return self._with_geometry(self._gdf().geometry.buffer(distance, resolution=resolution))

    # Buffer by a distance in metres, whatever the CRS.
    def buffer_metres(self, distance: float, resolution: int = 16) -> Vector:
        # Data in their own CRS.
        gdf = self._gdf()
        # The CRS must be known to project.
        if gdf.crs is None:
            # Explain the problem.
            raise ValueError("buffer_metres needs a CRS")
        # UTM zone of the centre of the data.
        utm = gdf.estimate_utm_crs()
        # Buffer in UTM metres and project back.
        buffered = gdf.to_crs(utm).geometry.buffer(distance, resolution=resolution).to_crs(gdf.crs)
        # New vector.
        return self._with_geometry(buffered)

    # Simplify with the Douglas-Peucker algorithm.
    def simplify(self, tolerance: float, preserve_topology: bool = True) -> Vector:
        # Simplified geometries.
        simplified = self._gdf().geometry.simplify(tolerance, preserve_topology=preserve_topology)
        # New vector.
        return self._with_geometry(simplified)

    # Centroids, or points guaranteed to lie inside each geometry.
    def centroid(self, inside: bool = False) -> Vector:
        # Geometries.
        geometry = self._gdf().geometry
        # Representative points or centroids.
        points = geometry.representative_point() if inside else geometry.centroid
        # New vector.
        return self._with_geometry(points)

    # Union of every geometry.
    def union(self) -> Any:
        # Imported lazily.
        from shapely.geometry import GeometryCollection

        # Empty data give an empty geometry.
        if self.data is None or self.data.empty:
            # Empty collection.
            return GeometryCollection()
        # Union of all geometries.
        return self.data.geometry.union_all()

    # Overlay with another vector ("intersection", "union", "difference", ...).
    def overlay(self, other: Vector, how: str = "intersection") -> Vector:
        # Imported lazily.
        import geopandas as gpd

        # Other data in this CRS.
        right = other._gdf().to_crs(self._gdf().crs) if other.crs != self.crs else other._gdf()
        # Overlay.
        result = gpd.overlay(self._gdf(), right, how=how, keep_geom_type=False)
        # New vector.
        return Vector(data=result)

    # Intersection with another vector.
    def intersection(self, other: Vector) -> Vector:
        # Overlay with "intersection".
        return self.overlay(other, "intersection")

    # Parts of this vector outside another vector.
    def difference(self, other: Vector) -> Vector:
        # Overlay with "difference".
        return self.overlay(other, "difference")

    # Features clipped to a geometry, bounds or another vector.
    def clip(self, mask: Any) -> Vector:
        # Vectors clip by their data.
        target = mask._gdf() if isinstance(mask, Vector) else mask
        # Clipped features.
        return Vector(data=self._gdf().clip(target))

    # Reproject to another CRS.
    def reproject(self, target_crs: str) -> Vector:
        # Transformed coordinates.
        return Vector(data=self._gdf().to_crs(target_crs), source=self.source)

    # Alias of reproject with the GeoPandas name.
    def to_crs(self, target_crs: str) -> Vector:
        # Same as reproject.
        return self.reproject(target_crs)

    # Merge geometries that share attribute values.
    def dissolve(self, by: str | list[str] | None = None, aggfunc: str = "first") -> Vector:
        # Dissolved features with the group keys as columns.
        dissolved = self._gdf().dissolve(by=by, aggfunc=aggfunc).reset_index()
        # New vector.
        return Vector(data=dissolved)

    # Split multi-part geometries into single parts.
    def explode(self) -> Vector:
        # One row per part.
        return Vector(data=self._gdf().explode(index_parts=False).reset_index(drop=True))

    # Repair invalid geometries (self-intersections and the like).
    def make_valid(self) -> Vector:
        # Imported lazily.
        import shapely

        # Valid geometries.
        return self._with_geometry(shapely.make_valid(self._gdf().geometry.values))

    # Whether each geometry is valid.
    def is_valid(self) -> NDArray[np.bool_]:
        # Validity per feature.
        return np.asarray(self._gdf().geometry.is_valid, dtype=bool)

    # Attributes of another vector joined by a spatial predicate.
    def spatial_join(
        self,  # This object.
        other: Vector,  # Vector whose attributes are joined.
        predicate: str = "intersects",  # Spatial relation.
        how: Literal["left", "right", "inner"] = "inner",  # Join type.
    ) -> Vector:  # Joined features.
        # Imported lazily.
        import geopandas as gpd

        # Other data in this CRS.
        right = other._gdf().to_crs(self._gdf().crs) if other.crs != self.crs else other._gdf()
        # Spatial join.
        joined = gpd.sjoin(self._gdf(), right, how=how, predicate=predicate)
        # New vector.
        return Vector(data=joined)

    # Geometries in WGS 84 and the ellipsoid for geodesic measures.
    def _geodesic(self) -> tuple[Any, Any]:
        # Imported lazily.
        from pyproj import Geod

        # Data in their own CRS.
        gdf = self._gdf()
        # The CRS must be known.
        if gdf.crs is None:
            # Explain the problem.
            raise ValueError("geodesic measures need a CRS")
        # Geographic coordinates on WGS 84.
        geometries = gdf.geometry.to_crs("EPSG:4326")
        # WGS 84 ellipsoid.
        return geometries, Geod(ellps="WGS84")

    # Area of every geometry: square metres when geodesic, CRS units squared otherwise.
    def area(self, geodesic: bool | None = None) -> NDArray[np.float64]:
        # Nothing to measure without data.
        if self.data is None:
            # Empty array.
            return np.array([], dtype=np.float64)
        # Geodesic by default for geographic data.
        if geodesic is None:
            # Decide from the CRS.
            geodesic = self.is_geographic
        # Planar area.
        if not geodesic:
            # Shapely area.
            return np.asarray(self.data.geometry.area, dtype=np.float64)
        # Geometries and ellipsoid.
        geometries, geod = self._geodesic()
        # Absolute signed area (the sign depends on the ring orientation).
        return np.array([abs(geod.geometry_area_perimeter(g)[0]) for g in geometries])

    # Length of every geometry: metres when geodesic, CRS units otherwise.
    def length(self, geodesic: bool | None = None) -> NDArray[np.float64]:
        # Nothing to measure without data.
        if self.data is None:
            # Empty array.
            return np.array([], dtype=np.float64)
        # Geodesic by default for geographic data.
        if geodesic is None:
            # Decide from the CRS.
            geodesic = self.is_geographic
        # Planar length.
        if not geodesic:
            # Shapely length (perimeter for polygons).
            return np.asarray(self.data.geometry.length, dtype=np.float64)
        # Geometries and ellipsoid.
        geometries, geod = self._geodesic()
        # Geodesic length of lines and perimeter of polygons.
        return np.array([geod.geometry_length(g) for g in geometries])

    # Burn the geometries into a raster on the grid of another raster.
    def rasterize(
        self,  # This object.
        like: Raster,  # Raster that defines the grid.
        attribute: str | None = None,  # Column burned as value; 1 by default.
        fill: float = 0,  # Value of pixels outside the geometries.
        all_touched: bool = False,  # Burn every pixel the geometries touch.
        dtype: str = "float32",  # Output dtype.
    ) -> Raster:  # Single-band raster.
        # Imported lazily.
        from rasterio.features import rasterize

        # Import at run time to avoid an import cycle.
        from unbihexium.core.raster import Raster

        # Data in the CRS of the grid.
        gdf = self._gdf()
        # Reproject when the CRS differs.
        if like.crs and gdf.crs is not None and gdf.crs != like.crs:
            # Transformed coordinates.
            gdf = gdf.to_crs(like.crs)
        # Burn values.
        values = gdf[attribute] if attribute else np.ones(len(gdf))
        # Pairs of geometry and value, skipping missing geometries.
        pairs = [(g, v) for g, v in zip(gdf.geometry, values) if g is not None and not g.is_empty]
        # Burned array.
        burned = rasterize(
            pairs,  # Shapes and values.
            out_shape=(like.height, like.width),  # Grid size.
            transform=_affine(like),  # Grid transform.
            fill=fill,  # Background.
            all_touched=all_touched,  # Rasterisation rule.
            dtype=dtype,  # Output dtype.
        )  # End of the rasterisation.
        # Raster on the grid.
        return Raster.from_array(burned, crs=like.crs, transform=like.transform)


# rasterio Affine object of a raster.
def _affine(raster: Raster) -> Any:
    # Imported lazily.
    from rasterio.transform import Affine

    # Build from the six coefficients.
    return Affine(*raster.transform)


# =============================================================================
# End of module src/unbihexium/core/vector.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
