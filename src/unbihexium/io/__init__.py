"""IO adapters for various geospatial file formats."""

from unbihexium.io.geojson import read_geojson, write_geojson
from unbihexium.io.geotiff import read_cog, read_geotiff, write_geotiff
from unbihexium.io.parquet import read_geoparquet, write_geoparquet
from unbihexium.io.stac import STACClient, load_from_stac, search_stac
from unbihexium.io.zarr_io import read_zarr, write_zarr

__all__ = [
    # GeoTIFF/COG
    "read_geotiff",
    "write_geotiff",
    "read_cog",
    # Zarr
    "read_zarr",
    "write_zarr",
    # STAC
    "STACClient",
    "search_stac",
    "load_from_stac",
    # GeoJSON
    "read_geojson",
    "write_geojson",
    # GeoParquet
    "read_geoparquet",
    "write_geoparquet",
]
