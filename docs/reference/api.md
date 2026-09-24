<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/reference/api.md
Title       : Python API Reference
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Python API Reference

| Field | Value |
| --- | --- |
| Document | UBX-DOC-REF-API |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch |

## Abstract

This document is the reference of the public Python interface of Unbihexium. For each subpackage it lists every name exported in `__all__` with its kind, its call signature and a one-line description, all taken from the source code of the main branch, and it gives runnable examples of the main entry points together with the output they produced. Selected modules that are public but not re-exported by their package, such as `unbihexium.ai.training` and `unbihexium.zoo.export`, are documented as well. It is written for developers who build applications on the library, for contributors who need an overview of the package structure, and for reviewers who check what the public interface promises under the versioning policy. Scientific methods are cited with their primary sources; the formulas themselves are documented in the module headers of the source code.

## Contents

- [1. Scope and conventions](#1-scope-and-conventions)
- [2. Package structure](#2-package-structure)
- [3. unbihexium](#3-unbihexium)
- [4. unbihexium.core](#4-unbihexiumcore)
- [5. unbihexium.indices](#5-unbihexiumindices)
- [6. unbihexium.io](#6-unbihexiumio)
- [7. unbihexium.preprocessing](#7-unbihexiumpreprocessing)
- [8. unbihexium.sar](#8-unbihexiumsar)
- [9. unbihexium.terrain](#9-unbihexiumterrain)
- [10. unbihexium.geostat](#10-unbihexiumgeostat)
- [11. unbihexium.analysis](#11-unbihexiumanalysis)
- [12. unbihexium.postprocessing](#12-unbihexiumpostprocessing)
- [13. unbihexium.metrics](#13-unbihexiummetrics)
- [14. unbihexium.visualization](#14-unbihexiumvisualization)
- [15. unbihexium.ai](#15-unbihexiumai)
- [16. unbihexium.zoo](#16-unbihexiumzoo)
- [17. unbihexium.registry](#17-unbihexiumregistry)
- [18. unbihexium.serving](#18-unbihexiumserving)
- [19. unbihexium.config](#19-unbihexiumconfig)
- [20. unbihexium.utils](#20-unbihexiumutils)
- [21. unbihexium.cli](#21-unbihexiumcli)
- [22. Stability of the interface](#22-stability-of-the-interface)
- [References](#references)

## 1. Scope and conventions

### 1.1 Scope

The public interface consists of the names listed in `__all__` of the package `unbihexium` and of its subpackages, the members of the classes they export, and the modules documented in the subsections titled "Further public modules". Names that start with an underscore, and modules not mentioned here, are internal and can change in any release. The document describes the main branch; the release 1.0.1 on PyPI predates large parts of it (see [installation.md](../getting_started/installation.md#3-choosing-an-installation-method)).

### 1.2 Reading the tables

- **Signature** shows the parameters with their default values; type annotations are omitted for readability, and a return type is shown when it is short. Defaults that are mutable containers created by a factory are written as `...`. The complete annotations are in the source and are checked by pyright in strict mode.
- **Description** is the comment or docstring that precedes the definition in the source code.
- Arrays are NumPy arrays. Image arrays are band-first, `(bands, rows, cols)`, or single-band `(rows, cols)`; missing values are NaN unless a `nodata` argument says otherwise.
- Affine transforms are six coefficients `(a, b, c, d, e, f)` in the order of the `affine` package used by rasterio, or an `affine.Affine` object.
- Every example was executed in a fresh working directory with `UNBIHEXIUM_CACHE` pointing to a temporary directory, with the extras `torch`, `onnx`, `serving`, `zarr` and `stac` installed; the output block below each example is the output of that run.

### 1.3 Conventions

The key words MUST, MUST NOT, SHOULD and MAY are used as described in RFC 2119 [1] and RFC 8174 [2] when they appear in capitals.

## 2. Package structure

Importing `unbihexium` loads only the version. Each subpackage is imported on first use, and optional dependencies are imported inside the functions that need them, so a missing extra shows up as an `ImportError` at the first call, not at import time. Importing `unbihexium.ai` or `unbihexium.zoo` does not import PyTorch.

| Subpackage | Purpose | Optional dependencies |
| --- | --- | --- |
| `unbihexium.core` | Data model: rasters, vectors, tiles, scenes, sensors, products, pipelines, provenance, index registry | none (rasterio and GeoPandas are core dependencies) |
| `unbihexium.indices` | Spectral indices as array functions | none |
| `unbihexium.io` | GeoTIFF and COG, Zarr, GeoJSON, GeoParquet, STAC | `zarr`, `parquet`, `stac` |
| `unbihexium.preprocessing` | Radiometry, cloud masks, enhancement, pansharpening, resampling, transforms | none |
| `unbihexium.sar` | SAR calibration, speckle filters, interferometry, polarimetry | none |
| `unbihexium.terrain` | Terrain derivatives, hydrology, viewshed | none |
| `unbihexium.geostat` | Variograms, kriging, spatial autocorrelation | none |
| `unbihexium.analysis` | Zonal statistics, suitability analysis, networks and cost surfaces | none |
| `unbihexium.postprocessing` | Activations, morphology, vectorisation, tile stitching | none |
| `unbihexium.metrics` | Accuracy assessment, area estimation, regression and image-quality metrics | none |
| `unbihexium.visualization` | Colour maps, composites, relief shading, PNG output | matplotlib only for `legend_figure` |
| `unbihexium.ai` | Task APIs, prediction, tiled inference, training and evaluation | `torch` (or `onnx` for ONNX files) |
| `unbihexium.zoo` | Model catalogue, local model store, verification, checkpoints, ONNX export | `torch` for building, `onnx` for export checks |
| `unbihexium.registry` | Capability, model and pipeline registries | none |
| `unbihexium.serving` | FastAPI REST service | `serving` |
| `unbihexium.config` | Layered, validated settings | none |
| `unbihexium.utils` | Logging, hashing, seeding, timing, tiling, atomic files | none |
| `unbihexium.cli` | The `unbihexium` command | none |

## 3. unbihexium

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `__version__` | constant | `'1.0.1'` | Version string. |
| `__version_tuple__` | constant | `(1, 0, 1)` | Version as a tuple of integers. |

```python
import unbihexium

print(unbihexium.__version__, unbihexium.__version_tuple__)
```

```text
1.0.1 (1, 0, 1)
```

## 4. unbihexium.core

### 4.1 Overview

`unbihexium.core` contains the building blocks shared by the other packages. `Raster` is a georeferenced array with metadata (CRS, affine transform, no-data value) and methods for windows, statistics, band math, resampling, reprojection, clipping and output. `Vector` wraps a GeoDataFrame, `Tile`, `TileGrid` and `TileIndex` describe tilings, `Scene` holds one acquisition as named bands, `SensorModel` and `get_sensor` describe sensors and their bands, `Product` describes derived products, `Pipeline` and `PipelineRun` execute and record processing steps, and `Evidence` and `ProvenanceRecord` form a SHA-256 audit trail of a run. `IndexRegistry` and `compute_index` give access to 27 spectral index definitions by name. Importing the package imports NumPy only.

### 4.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `Evidence` | class | `Evidence(source, checksum='', evidence_type=EvidenceType.INPUT, description='', timestamp=..., size_bytes=None, metadata=..., evidence_id='')` | One artefact identified by its SHA-256 digest. |
| `EvidenceType` | enum | `input`, `output`, `intermediate`, `model`, `config`, `log` | Kinds of artefacts recorded as evidence. |
| `IndexCategory` | enum | `vegetation`, `water`, `soil`, `burn`, `urban`, `snow`, `moisture` | Thematic groups of indices. |
| `IndexRegistry` | class | `IndexRegistry()` | Registry of spectral indices by case-insensitive name. |
| `ModelConfig` | class | `ModelConfig(model_id, name, task, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1, input_size=None, normalize=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), threshold=0.5, logits=False, version='1.0.0', tags=...)` | Configuration of a model. |
| `ModelFramework` | enum | `pytorch`, `onnx`, `sklearn`, `custom` | Frameworks a model can come from. |
| `ModelTask` | enum | `detection`, `segmentation`, `classification`, `regression`, `super_resolution`, `change_detection` | Tasks a model can solve. |
| `ModelWrapper` | class | `ModelWrapper(config, model=None, weights_path=None)` | Model with framework-neutral pre- and postprocessing. |
| `Pipeline` | class | `Pipeline(config)` | Ordered list of named steps. |
| `PipelineConfig` | class | `PipelineConfig(pipeline_id, name, description='', version='1.0.0', steps=..., parameters=..., seed=None, deterministic=True)` | Configuration of a pipeline. |
| `PipelineRun` | class | `PipelineRun(run_id=..., pipeline_id='', status=PipelineStatus.PENDING, start_time=None, end_time=None, config_snapshot=..., inputs=..., outputs=..., logs=..., metrics=..., error=None, seed=None, steps=..., provenance=None, results=...)` | Record of one execution of a pipeline. |
| `PipelineStatus` | enum | `pending`, `running`, `completed`, `failed`, `cancelled` | States of a run or a step. |
| `Product` | class | `Product(data=None, metadata=None, source=None)` | A derived product: data and metadata. |
| `ProductMetadata` | class | `ProductMetadata(product_id, product_type, name='', description='', created_at=..., version='1.0.0', crs='EPSG:4326', bounds=None, resolution=None, source_scenes=..., processing_chain=..., quality_score=None, license='', tags=...)` | Metadata of a product. |
| `ProductType` | enum | `raster`, `vector`, `detection`, `segmentation`, `classification`, `index`, `change`, `dem`, `dsm`, `mosaic`, `composite` | Kinds of products. |
| `ProvenanceRecord` | class | `ProvenanceRecord(run_id, pipeline_id, record_id=..., inputs=..., outputs=..., model_ids=..., models=..., config=..., environment=..., parent_records=..., created_at=...)` | Provenance of one pipeline run. |
| `Raster` | class | `Raster(data=None, metadata=None, source=None, lazy=False)` | Georeferenced raster. |
| `RasterMetadata` | class | `RasterMetadata(crs, transform, width, height, count, dtype=RasterDtype.FLOAT32, nodata=None, bounds=None, resolution=None, tags=...)` | Metadata of a raster. |
| `Scene` | class | `Scene(rasters=..., metadata=None, source=None)` | One acquisition as a set of named band rasters. |
| `SceneMetadata` | class | `SceneMetadata(scene_id, sensor, acquisition_date=None, acquisition_mode=AcquisitionMode.MONO, cloud_cover=0.0, sun_azimuth=None, sun_elevation=None, resolution=1.0, bands=..., crs='EPSG:4326', bounds=None, processing_level='L1', tags=...)` | Metadata of one acquisition. |
| `SensorModel` | class | `SensorModel(name, sensor_type, platform='', resolution=1.0, swath_width=0.0, bands=..., band_wavelengths=..., revisit_time_days=0.0, altitude_km=0.0, inclination_deg=0.0, launch_date='', operator='', spectral_bands=(), frequency_ghz=None, sar_modes=(), alias_prefixes=())` | A satellite sensor with its bands and orbit. |
| `SensorType` | enum | `optical`, `sar`, `multispectral`, `hyperspectral`, `panchromatic`, `thermal` | Families of sensors. |
| `SpectralBand` | class | `SpectralBand(name, common_name, center_nm, bandwidth_nm, resolution_m)` | One spectral band of an optical sensor. |
| `SpectralIndex` | class | `SpectralIndex(name, formula, category, bands_required, function, value_range=(-1.0, 1.0), description='', reference='', parameters=...)` | Definition of a spectral index. |
| `Tile` | class | `Tile(index, data, bounds=None, offset=(0, 0), size=(256, 256), transform=None)` | One tile of raster data. |
| `TileGrid` | class | `TileGrid(tile_size, overlap=0, num_rows=0, num_cols=0, raster_height=0, raster_width=0, crs='EPSG:4326', transform=None)` | A grid of tiles over a raster. |
| `TileIndex` | class | `TileIndex(row, col, level=0)` | Position of a tile within a grid. |
| `Vector` | class | `Vector(data=None, metadata=None, source=None)` | Georeferenced vector data backed by a GeoDataFrame. |
| `VectorMetadata` | class | `VectorMetadata(crs, geometry_type=None, feature_count=0, bounds=None, columns=..., schema=...)` | Metadata of a vector dataset. |
| `compute_index` | function | `compute_index(name, bands, sensor=None, nodata=None, **parameters) -> Array` | Compute a registered index by name. |
| `get_sensor` | function | `get_sensor(sensor_id) -> SensorModel \| None` | Sensor by identifier or short name, or None when unknown. |

`unbihexium.core.compute_index` takes a mapping of band arrays keyed by common band names (for example `{"RED": red, "NIR": nir}`) and optionally a sensor, whereas `unbihexium.indices.compute_index` takes the bands as keyword arguments; both evaluate the same formulas.

### 4.3 Members of Raster

| Member of `Raster` | Signature | Description |
| --- | --- | --- |
| `from_array` | classmethod `from_array(data, crs='EPSG:4326', transform=None, nodata=None, tags=None) -> Raster` | Raster from an array of shape (bands, height, width) or (height, width). |
| `from_file` | classmethod `from_file(path, lazy=False, bands=None, window=None, dtype='float32') -> Raster` | Raster from a file readable by GDAL. |
| `load` | `load() -> Raster` | Read the pixels of a lazy raster. |
| `require_data` | `require_data() -> ndarray` | Data after loading; raises when there is none. |
| `shape` | property | Shape (bands, height, width). |
| `width` | property | Number of columns. |
| `height` | property | Number of rows. |
| `count` | property | Number of bands. |
| `dtype` | property | Data type of the pixels. |
| `crs` | property | Coordinate reference system. |
| `transform` | property | Affine coefficients. |
| `nodata` | property | No-data value. |
| `bounds` | property | Bounds (min x, min y, max x, max y) in CRS units. |
| `resolution` | property | Pixel size (x, y) of a north-up raster. |
| `xy` | `xy(row, col, offset='center') -> tuple[Any, Any]` | Map coordinates of pixels (centres by default). |
| `rowcol` | `rowcol(x, y) -> tuple[ndarray[int64], ndarray[int64]]` | Row and column of the pixels that contain map coordinates. |
| `sample` | `sample(x, y) -> ndarray[float64]` | Pixel values at map coordinates; NaN outside the raster or where invalid. |
| `valid_mask` | `valid_mask() -> ndarray[bool_]` | Mask of shape (height, width): True where every band is valid. |
| `masked` | `masked() -> ma.MaskedArray[Any, Any]` | Masked array whose mask marks invalid values per band. |
| `statistics` | `statistics(percentiles=(2.0, 25.0, 50.0, 75.0, 98.0)) -> list[dict[str, float]]` | Per-band statistics of the valid pixels. |
| `histogram` | `histogram(band=1, bins=256, value_range=None) -> tuple[ndarray[int64], ndarray[float64]]` | Histogram of the valid values of one band (1-based). |
| `band_math` | `band_math(expression, dtype='float32') -> Raster` | Evaluate a band math expression on the bands b1..bN. |
| `select_bands` | `select_bands(bands) -> Raster` | Raster with some bands (1-based numbers) in the given order. |
| `with_data` | `with_data(data, **changes) -> Raster` | Raster on the same grid with new data (any band count) and metadata changes. |
| `stack` | classmethod `stack(rasters) -> Raster` | Stack single- or multi-band rasters on the same grid. |
| `same_grid` | `same_grid(other, tolerance=1e-09) -> bool` | Whether another raster has the same size, transform and CRS. |
| `astype` | `astype(dtype) -> Raster` | Raster with the pixels converted to another dtype. |
| `set_nodata` | `set_nodata(nodata) -> Raster` | Raster with a new no-data value (the pixels are unchanged). |
| `mask` | `mask(mask, fill=None) -> Raster` | Raster with the pixels where mask is True set to the no-data value. |
| `read_window` | `read_window(row_off, col_off, height, width) -> ndarray` | Pixels of a window as an array (reads from the source for lazy rasters). |
| `window` | `window(row_off, col_off, height, width) -> Raster` | Raster of a pixel window with its transform. |
| `tiles` | `tiles(tile_size=256, overlap=0) -> Iterator[tuple[int, int, ndarray]]` | Tiles (row offset, column offset, data) covering the raster. |
| `crop` | `crop(bounds) -> Raster` | Raster cropped to map bounds (left, bottom, right, top). |
| `clip` | `clip(geometries, crop=True, all_touched=False, invert=False, fill=None) -> Raster` | Raster with the pixels outside geometries set to the no-data value. |
| `resample` | `resample(scale=None, method='bilinear', resolution=None) -> Raster` | Resample to a new pixel size on the same CRS. |
| `reproject` | `reproject(target_crs, resolution=None, method='bilinear') -> Raster` | Reproject to another CRS. |
| `match` | `match(other, method='bilinear') -> Raster` | Warp onto the grid (CRS, transform and size) of another raster. |
| `apply` | `apply(func, *args, **kwargs) -> Raster` | Apply a function to the data and keep the georeferencing. |
| `to_file` | `to_file(path, driver='GTiff', compress='lzw', tiled=True, blockxsize=256, blockysize=256) -> Path` | Write the raster to a file (tiled and compressed GeoTIFF by default). |
| `to_cog` | `to_cog(path, compress='deflate', blocksize=512, overview_resampling='average') -> Path` | Write a Cloud Optimized GeoTIFF with internal overviews. |

`Raster.from_file(window=...)` takes `(row, col, height, width)`, whereas `unbihexium.io.read_geotiff(window=...)` takes `(row_start, row_stop, col_start, col_stop)`.

### 4.4 Members of IndexRegistry

| Member of `IndexRegistry` | Signature | Description |
| --- | --- | --- |
| `register` | classmethod `register(index) -> SpectralIndex` | Register an index; an existing index of the same name is replaced. |
| `unregister` | classmethod `unregister(name)` | Remove an index. |
| `get` | classmethod `get(name) -> SpectralIndex \| None` | Index by name, or None when unknown. |
| `list_all` | classmethod `list_all() -> list[str]` | Names of the registered indices. |
| `by_category` | classmethod `by_category(category) -> list[SpectralIndex]` | Indices of one category. |
| `available_for` | classmethod `available_for(band_names) -> list[str]` | Indices that can be computed from the given common band names. |

### 4.5 Example

```python
import numpy as np

from unbihexium.core import IndexRegistry, Raster, compute_index, get_sensor

data = np.random.default_rng(0).uniform(0.02, 0.5, size=(4, 64, 64)).astype("float32")
raster = Raster.from_array(data, crs="EPSG:32635", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6700000.0))
print(raster.shape, raster.bounds, raster.resolution)

ndvi = raster.band_math("(b4 - b3) / (b4 + b3)")          # bands are b1..bN
coarse = ndvi.resample(scale=0.5)                            # 20 m pixels
print(ndvi.count, coarse.shape, coarse.resolution)
print(raster.window(0, 0, 16, 16).bounds)

bands = {"RED": data[2], "NIR": data[3]}
print(np.allclose(compute_index("NDVI", bands), ndvi.data[0], atol=1e-6))
print(IndexRegistry.available_for(["red", "nir"])[:6])
s2 = get_sensor("sentinel-2")
print(s2.name if s2 else None, len(s2.bands) if s2 else 0)
```

Output:

```text
(4, 64, 64) (500000.0, 6699360.0, 500640.0, 6700000.0) (10.0, 10.0)
1 (1, 32, 32) (20.0, 20.0)
(500000.0, 6699840.0, 500160.0, 6700000.0)
True
['NDVI', 'EVI2', 'SAVI', 'MSAVI', 'OSAVI', 'SR']
Sentinel-2 MSI 13
```

## 5. unbihexium.indices

### 5.1 Overview

Spectral indices as functions of reflectance arrays of any shape. The functions compute in float64 and return NaN where a denominator is zero or an input is not finite (`safe_divide`). `INDEX_FUNCTIONS` maps lower-case names to the functions, and `compute_index(name, **bands)` evaluates an index by name. Examples of the implemented formulas are the Normalized Difference Vegetation Index [3], the Normalized Difference Water Index [4], the Enhanced Vegetation Index [5] and the Soil Adjusted Vegetation Index [6]:

$$
\mathrm{NDVI} = \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{red}}}{\rho_{\mathrm{NIR}} + \rho_{\mathrm{red}}}, \qquad
\mathrm{EVI} = G \, \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{red}}}{\rho_{\mathrm{NIR}} + C_1 \rho_{\mathrm{red}} - C_2 \rho_{\mathrm{blue}} + L}, \qquad
\mathrm{SAVI} = (1 + L) \, \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{red}}}{\rho_{\mathrm{NIR}} + \rho_{\mathrm{red}} + L}
$$

with $G = 2.5$, $C_1 = 6$, $C_2 = 7.5$ and $L = 1$ for EVI and $L = 0.5$ for SAVI by default. The reference of every other index is given in its description below.

### 5.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `BURN_SEVERITY_BREAKS` | constant | `(-0.25, -0.1, 0.1, 0.27, 0.44, 0.66)` | dNBR class limits. |
| `BURN_SEVERITY_CLASSES` | constant | tuple, 7 items | dNBR class names. |
| `INDEX_FUNCTIONS` | constant | dict, 25 entries | Index functions by name. |
| `arvi` | function | `arvi(nir, red, blue, gamma=1.0) -> ndarray[float64]` | Atmospherically Resistant Vegetation Index (Kaufman and Tanre, 1992). |
| `awei_nsh` | function | `awei_nsh(green, nir, swir1, swir2) -> ndarray[float64]` | Automated Water Extraction Index without shadows (Feyisa et al., 2014). |
| `awei_sh` | function | `awei_sh(blue, green, nir, swir1, swir2) -> ndarray[float64]` | Automated Water Extraction Index with shadows (Feyisa et al., 2014). |
| `bsi` | function | `bsi(blue, red, nir, swir1) -> ndarray[float64]` | Bare Soil Index (Rikimaru et al., 2002). |
| `burn_severity` | function | `burn_severity(dnbr_values) -> ndarray[int64]` | Burn severity class 0 to 6 from dNBR, -1 for NaN. |
| `ci_green` | function | `ci_green(nir, green) -> ndarray[float64]` | Green chlorophyll index (Gitelson et al., 2003). |
| `ci_rededge` | function | `ci_rededge(nir, red_edge) -> ndarray[float64]` | Red-edge chlorophyll index (Gitelson et al., 2003). |
| `compute_index` | function | `compute_index(name, **bands) -> ndarray` | Evaluate an index by name with bands passed as keyword arguments. |
| `cross_pol_ratio` | function | `cross_pol_ratio(sigma_cross, sigma_co) -> ndarray[float64]` | Cross-polarisation ratio of linear backscatter, for example VH / VV. |
| `dnbr` | function | `dnbr(nbr_pre, nbr_post) -> ndarray[float64]` | Differenced NBR, pre-fire minus post-fire (Key and Benson, 2006). |
| `evi` | function | `evi(nir, red, blue, g=2.5, c1=6.0, c2=7.5, l=1.0) -> ndarray[float64]` | Enhanced Vegetation Index (Huete et al., 2002). |
| `evi2` | function | `evi2(nir, red) -> ndarray[float64]` | Two-band EVI without the blue band (Jiang et al., 2008). |
| `gndvi` | function | `gndvi(nir, green) -> ndarray[float64]` | Green NDVI (Gitelson et al., 1996). |
| `kndvi` | function | `kndvi(nir, red) -> ndarray[float64]` | Kernel NDVI with the RBF kernel and sigma = (NIR + Red) / 2 (Camps-Valls et al., 2021). |
| `mndwi` | function | `mndwi(green, swir1) -> ndarray[float64]` | Modified NDWI (Xu, 2006). |
| `msavi` | function | `msavi(nir, red) -> ndarray[float64]` | Modified SAVI, MSAVI2 (Qi et al., 1994). |
| `msi` | function | `msi(swir, nir) -> ndarray[float64]` | Moisture Stress Index (Hunt and Rock, 1989). |
| `nbr` | function | `nbr(nir, swir) -> ndarray[float64]` | Normalized Burn Ratio with the long SWIR band (Key and Benson, 2006). |
| `nbr2` | function | `nbr2(swir1, swir2) -> ndarray[float64]` | Normalized Burn Ratio 2 from the two SWIR bands. |
| `ndbi` | function | `ndbi(swir1, nir) -> ndarray[float64]` | Normalized Difference Built-up Index (Zha et al., 2003). |
| `ndmi` | function | `ndmi(nir, swir1) -> ndarray[float64]` | Normalized Difference Moisture Index, the NDWI of Gao (1996). |
| `ndre` | function | `ndre(nir, red_edge) -> ndarray[float64]` | Normalized Difference Red Edge index (Barnes et al., 2000). |
| `ndsi` | function | `ndsi(green, swir1) -> ndarray[float64]` | Normalized Difference Snow Index (Hall et al., 1995). |
| `ndvi` | function | `ndvi(nir, red) -> ndarray[float64]` | Normalized Difference Vegetation Index (Rouse et al., 1974). |
| `ndwi` | function | `ndwi(green, nir) -> ndarray[float64]` | Normalized Difference Water Index for open water (McFeeters, 1996). |
| `normalized_difference` | function | `normalized_difference(a, b) -> ndarray[float64]` | Normalised difference (a - b) / (a + b). |
| `osavi` | function | `osavi(nir, red) -> ndarray[float64]` | Optimised SAVI (Rondeaux et al., 1996). |
| `rdnbr` | function | `rdnbr(nbr_pre, nbr_post) -> ndarray[float64]` | Relative dNBR, dNBR / sqrt(\|NBR_pre\|) (Miller and Thode, 2007). |
| `rvi` | function | `rvi(sigma_hh, sigma_hv, sigma_vv) -> ndarray[float64]` | Radar Vegetation Index of quad-pol backscatter (Kim and van Zyl, 2009). |
| `safe_divide` | function | `safe_divide(num, den) -> ndarray[float64]` | Ratio num / den with NaN where the denominator is zero or not finite. |
| `savi` | function | `savi(nir, red, l=0.5) -> ndarray[float64]` | Soil Adjusted Vegetation Index (Huete, 1988). |
| `vari` | function | `vari(green, red, blue) -> ndarray[float64]` | Visible Atmospherically Resistant Index (Gitelson et al., 2002). |

### 5.3 Example

```python
import numpy as np

from unbihexium.indices import INDEX_FUNCTIONS, burn_severity, compute_index, dnbr, evi, nbr, ndvi

red = np.array([0.05, 0.10, 0.20])
nir = np.array([0.45, 0.30, 0.20])
blue = np.array([0.03, 0.05, 0.10])
print(np.round(ndvi(nir=nir, red=red), 3))
print(np.round(evi(nir=nir, red=red, blue=blue), 3))
print(np.round(compute_index("savi", nir=nir, red=red), 3), len(INDEX_FUNCTIONS))

pre = nbr(nir=np.array([0.40, 0.40]), swir=np.array([0.10, 0.10]))    # before a fire
post = nbr(nir=np.array([0.35, 0.15]), swir=np.array([0.12, 0.30]))   # after a fire
print(np.round(dnbr(pre, post), 3), burn_severity(dnbr(pre, post)))
print(ndvi(nir=np.array([0.0]), red=np.array([0.0])))                 # 0 / 0 gives NaN
```

Output:

```text
[0.8 0.5 0. ]
[0.656 0.328 0.   ]
[0.6   0.333 0.   ] 25
[0.111 0.933] [3 6]
[nan]
```

## 6. unbihexium.io

### 6.1 Overview

Adapters for the file formats of Earth observation workflows: GeoTIFF and Cloud Optimized GeoTIFF [7] through rasterio and GDAL, Zarr v2 and v3 stores [8], GeoJSON documents [9] as dictionaries (validation, reprojection, right-hand rule), GeoParquet [10] through GeoPandas and pyarrow, and SpatioTemporal Asset Catalog [11] items, static catalogues and API search. Writers take the data first and the path second; the path-first order of earlier releases is still accepted. Pixel windows of `read_geotiff` and `read_cog` are `(row_start, row_stop, col_start, col_stop)`.

### 6.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `STACClient` | class | `STACClient(url, headers=..., timeout=30.0, transport=None, max_pages=100)` | Client of a STAC API. |
| `STACCollection` | class | `STACCollection(id, description='', license='proprietary', bboxes=..., intervals=..., title='', links=..., summaries=...)` | One STAC collection. |
| `STACItem` | class | `STACItem(id, bbox, datetime, properties=..., assets=..., collection=None, geometry=None, asset_info=..., links=..., stac_version='1.0.0', stac_extensions=...)` | One STAC item. |
| `build_overviews` | function | `build_overviews(path, factors=None, resampling='average', blocksize=256) -> list[int]` | Add internal overviews to an existing GeoTIFF; returns the factors. |
| `features_to_geojson` | function | `features_to_geojson(features, crs=None) -> dict[str, Any]` | FeatureCollection from features; a non-default CRS is stored as a legacy member. |
| `filter_items` | function | `filter_items(items, bbox=None, datetime_range=None, collections=None, ids=None, query=None, max_cloud_cover=None, limit=None) -> list[STACItem]` | Offline item search with the semantics of the STAC API. |
| `geojson_bounds` | function | `geojson_bounds(obj) -> tuple[float, float, float, float]` | Bounding box (min x, min y, max x, max y) of every position. |
| `geojson_crs` | function | `geojson_crs(obj) -> str` | CRS of a document as an authority string, for example "EPSG:32632". |
| `geometry_to_feature` | function | `geometry_to_feature(geometry, properties=None, feature_id=None) -> dict[str, Any]` | Feature object from a geometry. |
| `geoparquet_bounds` | function | `geoparquet_bounds(path) -> tuple[float, float, float, float]` | Bounding box (min x, min y, max x, max y) of the primary geometry column. |
| `geoparquet_metadata` | function | `geoparquet_metadata(path) -> dict[str, Any]` | Decoded "geo" metadata of a GeoParquet file. |
| `geotiff_info` | function | `geotiff_info(path) -> dict[str, Any]` | Metadata of a file without reading pixels. |
| `is_cog` | function | `is_cog(path) -> bool` | Whether a file has the Cloud Optimized GeoTIFF layout. |
| `load_from_stac` | function | `load_from_stac(item, asset_key='visual', **kwargs) -> Any` | Read a raster asset of an item with the GeoTIFF reader. |
| `read_cog` | function | `read_cog(url, bands=None, window=None, **kwargs) -> tuple[ndarray, dict[str, Any]]` | Read a (remote) Cloud Optimized GeoTIFF. |
| `read_geojson` | function | `read_geojson(path, validate=True) -> dict[str, Any]` | Read a GeoJSON file. |
| `read_geoparquet` | function | `read_geoparquet(path, columns=None, bbox=None, to_crs=None) -> Any` | Read a GeoParquet file into a GeoDataFrame. |
| `read_geotiff` | function | `read_geotiff(path, bands=None, window=None, bounds=None, overview_level=None, masked=False, dtype='float32') -> tuple[ndarray, dict[str, Any]]` | Read bands of a GeoTIFF file or URL. |
| `read_raster` | function | `read_raster(path, **kwargs) -> Any` | Read a GeoTIFF into a Raster. |
| `read_raster_zarr` | function | `read_raster_zarr(path, variable=None) -> Any` | Read a raster written by write_raster_zarr. |
| `read_stac_item` | function | `read_stac_item(path) -> STACItem` | Read an item from a JSON file. |
| `read_zarr` | function | `read_zarr(path, variable=None, selection=None, dtype=None) -> tuple[ndarray, dict[str, Any]]` | Read an array (or a selection of it) and its attributes. |
| `reproject_geojson` | function | `reproject_geojson(obj, src_crs=None, dst_crs='OGC:CRS84') -> dict[str, Any]` | Reproject a document; the result is in OGC:CRS84 unless dst_crs is given. |
| `rewind` | function | `rewind(obj) -> dict[str, Any]` | Copy of a document that follows the right-hand rule of RFC 7946. |
| `search_stac` | function | `search_stac(url, bbox=None, datetime_range=None, collections=None, limit=100, **kwargs) -> list[STACItem]` | Search a STAC API and return a list of items. |
| `validate_geojson` | function | `validate_geojson(obj) -> dict[str, Any]` | Raise ValueError when a document is not valid GeoJSON. |
| `walk_catalog` | function | `walk_catalog(path, max_depth=16) -> Iterator[STACItem]` | Items of a local static catalogue or collection, following child and item links. |
| `write_cog` | function | `write_cog(data, path, **kwargs) -> Path` | Write a Cloud Optimized GeoTIFF. |
| `write_geojson` | function | `write_geojson(data, path, indent=2, validate=True, precision=None) -> Path` | Write a GeoJSON document; returns the path. |
| `write_geoparquet` | function | `write_geoparquet(gdf, path, compression='snappy', write_bbox=False, index=None) -> Path` | Write a GeoDataFrame as GeoParquet; returns the path. |
| `write_geotiff` | function | `write_geotiff(data, path, crs=None, transform=None, nodata=None, compress='deflate', tiled=True, blocksize=256, predictor=None, overviews=None, resampling='average', descriptions=None, tags=None, cog=False) -> Path` | Write an array to a GeoTIFF file; returns the path. |
| `write_raster` | function | `write_raster(raster, path, **kwargs) -> Path` | Write a Raster to a GeoTIFF. |
| `write_raster_zarr` | function | `write_raster_zarr(raster, path, crs=None, transform=None, nodata=None, **kwargs) -> Path` | Write pixels with georeferencing attributes. |
| `write_zarr` | function | `write_zarr(data, path, chunks=None, attrs=None, compressor='zstd', level=5, name=None, dimension_names=None, zarr_format=None) -> Path` | Write an array to a Zarr store; returns the path. |
| `zarr_info` | function | `zarr_info(path, variable=None) -> dict[str, Any]` | Description of an array without reading its chunks. |

### 6.3 Example

```python
import numpy as np
from rasterio.transform import from_origin

from unbihexium.io import (
    features_to_geojson, geometry_to_feature, read_geojson, read_geotiff,
    read_raster, write_geojson, write_geotiff, write_zarr, read_zarr,
)

data = np.arange(2 * 32 * 32, dtype="float32").reshape(2, 32, 32)
path = write_geotiff(data, "two_bands.tif", crs="EPSG:3067", transform=from_origin(385000, 6672000, 10, 10),
                     nodata=-9999.0, descriptions=["a", "b"])
subset, meta = read_geotiff(path, bands=[2], window=(0, 8, 0, 8))  # rows 0-7, columns 0-7
print(subset.shape, meta["crs"], meta["nodata"])

raster = read_raster(path)
print(type(raster).__name__, raster.shape)

point = geometry_to_feature({"type": "Point", "coordinates": [24.94, 60.17]}, {"name": "Helsinki"})
write_geojson(features_to_geojson([point]), "points.geojson")
print(read_geojson("points.geojson")["features"][0]["properties"])

write_zarr(data, "stack.zarr", chunks=(1, 16, 16), attrs={"units": "counts"})
array, attrs = read_zarr("stack.zarr")
print(array.shape, attrs["units"])
```

Output:

```text
(1, 8, 8) EPSG:3067 -9999.0
Raster (2, 32, 32)
{'name': 'Helsinki'}
(2, 32, 32) counts
```

## 7. unbihexium.preprocessing

### 7.1 Overview

Preparation of optical imagery before analysis or inference: conversion of digital numbers to radiance, reflectance and brightness temperature for Sentinel-2 and Landsat 8/9 (including the offset of the Sentinel-2 processing baseline 04.00 and later), the Earth-Sun distance and dark object subtraction; Sentinel-2 SCL and Landsat QA_PIXEL masks; linear and percentile stretches, gamma correction, histogram equalisation and matching; Brovey, IHS and Gram-Schmidt adaptive pansharpening; interpolation and block aggregation with missing values; and array transforms used by the models.

### 7.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `AGGREGATIONS` | constant | tuple, 6 items | Aggregation methods. |
| `LANDSAT_C2L2_SR_OFFSET` | constant | `-0.2` | Level-2 reflectance offset. |
| `LANDSAT_C2L2_SR_SCALE` | constant | `2.75e-05` | Level-2 reflectance scale. |
| `LANDSAT_C2L2_ST_OFFSET` | constant | `149.0` | Level-2 temperature offset. |
| `LANDSAT_C2L2_ST_SCALE` | constant | `0.00341802` | Level-2 temperature scale. |
| `LANDSAT_QA_BITS` | constant | dict, 8 entries | QA_PIXEL flag bits. |
| `LANDSAT_QA_CONFIDENCE` | constant | dict, 4 entries | QA_PIXEL confidence fields. |
| `RESAMPLING_ORDERS` | constant | dict, 3 entries | Interpolation methods. |
| `S2_OFFSET_PB04` | constant | `-1000.0` | Sentinel-2 offset from baseline 04.00. |
| `S2_QUANTIFICATION` | constant | `10000.0` | Sentinel-2 quantification value. |
| `SCL_CLASSES` | constant | dict, 12 entries | Sentinel-2 SCL legend. |
| `SCL_DEFAULT_INVALID` | constant | `(0, 1, 3, 8, 9, 10)` | SCL classes masked by default. |
| `Compose` | class | `Compose(transforms)` | Apply several transforms in order. |
| `Normalize` | class | `Normalize(mean=None, std=None, min_val=0.0, max_val=1.0)` | Per-band standardisation with fixed statistics, or min-max scaling. |
| `Pad` | class | `Pad(target_size, mode='constant')` | Pad an (H, W) or (C, H, W) array at the bottom and right to a target size. |
| `Resize` | class | `Resize(size, interpolation='bilinear')` | Resize an (H, W) or (C, H, W) array to a target (height, width). |
| `aggregate` | function | `aggregate(image, factor, method='mean', nodata=None, trim=True) -> ndarray[float64]` | Reduce an (H, W) or (C, H, W) array by an integer factor. |
| `apply_mask` | function | `apply_mask(image, mask) -> ndarray[float64]` | Set pixels of an image where the mask is True to NaN. |
| `brovey` | function | `brovey(pan, ms, weights=None, match=True) -> ndarray[float64]` | Brovey (ratio) transform. |
| `buffer_mask` | function | `buffer_mask(mask, pixels) -> ndarray[bool_]` | Grow a boolean mask by a number of pixels with a round structuring element. |
| `dark_object_subtraction` | function | `dark_object_subtraction(reflectance, dark_percentile=0.01, dark_reflectance=0.01, clip=True) -> tuple[ndarray[float64], ndarray[float64]]` | DOS1 haze removal of TOA reflectance, band by band. |
| `earth_sun_distance` | function | `earth_sun_distance(day_of_year) -> float \| ndarray[float64]` | Earth-Sun distance in astronomical units for a day of the year. |
| `from_tensor` | function | `from_tensor(tensor) -> ndarray` | Convert a band-first (C, H, W) array to the channel-last (H, W, C) layout. |
| `gamma_correction` | function | `gamma_correction(image, gamma=1.0) -> ndarray[float64]` | Power-law adjustment out = in ** (1 / gamma) of values in [0, 1]. |
| `gram_schmidt` | function | `gram_schmidt(pan, ms, weights=None) -> ndarray[float64]` | Gram-Schmidt adaptive (GSA) component substitution. |
| `histogram_equalize` | function | `histogram_equalize(image, nodata=None) -> ndarray[float64]` | Histogram equalisation of every band to [0, 1] through the empirical CDF. |
| `histogram_match` | function | `histogram_match(source, reference, nodata=None) -> ndarray[float64]` | Match the histogram of every source band to the matching reference band. |
| `ihs` | function | `ihs(pan, ms, weights=None, match=True) -> ndarray[float64]` | Fast generalised IHS fusion. |
| `landsat_brightness_temperature` | function | `landsat_brightness_temperature(radiance, k1, k2) -> ndarray[float64]` | Thermal radiance to at-sensor brightness temperature in kelvin. |
| `landsat_c2l2_reflectance` | function | `landsat_c2l2_reflectance(dn, nodata=0) -> ndarray[float64]` | Landsat Collection 2 Level-2 DN to surface reflectance. |
| `landsat_c2l2_temperature` | function | `landsat_c2l2_temperature(dn, nodata=0) -> ndarray[float64]` | Landsat Collection 2 Level-2 DN to surface temperature in kelvin. |
| `landsat_cloud_confidence` | function | `landsat_cloud_confidence(qa, field='cloud') -> ndarray[int64]` | Two-bit confidence of a QA_PIXEL field (0 none, 1 low, 2 medium, 3 high). |
| `landsat_qa_mask` | function | `landsat_qa_mask(qa, fill=True, dilated_cloud=True, cirrus=True, cloud=True, cloud_shadow=True, snow=False, water=False) -> ndarray[bool_]` | Flag mask of the Landsat Collection 2 QA_PIXEL band. |
| `landsat_radiance` | function | `landsat_radiance(dn, mult, add, nodata=0) -> ndarray[float64]` | Landsat Level-1 DN to at-sensor spectral radiance in W / (m^2 sr um). |
| `landsat_toa_reflectance` | function | `landsat_toa_reflectance(dn, mult, add, sun_elevation, nodata=0) -> ndarray[float64]` | Landsat Level-1 DN to sun-corrected top-of-atmosphere reflectance. |
| `landsat_toa_reflectance_from_mtl` | function | `landsat_toa_reflectance_from_mtl(dn, mtl, band, nodata=0) -> ndarray[float64]` | Landsat Level-1 DN of one band to TOA reflectance with MTL metadata. |
| `linear_stretch` | function | `linear_stretch(image, low, high, clip=True, nodata=None) -> ndarray[float64]` | Linear stretch of [low, high] onto [0, 1], per band. |
| `minmax_normalize` | function | `minmax_normalize(image, nodata=None) -> ndarray[float64]` | Per-band min-max scaling to [0, 1] over finite values. |
| `parse_landsat_mtl` | function | `parse_landsat_mtl(text) -> dict[str, Any]` | Parse a Landsat MTL.txt file into a flat dictionary of values. |
| `percentile_bounds` | function | `percentile_bounds(image, low=2.0, high=98.0, nodata=None) -> tuple[ndarray[float64], ndarray[float64]]` | Percentile bounds of every band over finite values. |
| `percentile_stretch` | function | `percentile_stretch(image, low=2.0, high=98.0, nodata=None) -> ndarray[float64]` | Linear stretch between two percentiles of every band. |
| `qa_bits` | function | `qa_bits(qa, start, length=1) -> ndarray[int64]` | Extract a bit field of a given length from an integer QA band. |
| `radiance_to_reflectance` | function | `radiance_to_reflectance(radiance, esun, sun_zenith, day_of_year) -> ndarray[float64]` | At-sensor radiance to TOA reflectance with the solar exoatmospheric irradiance. |
| `resample` | function | `resample(image, shape, method='bilinear', nodata=None) -> ndarray` | Interpolate an (H, W) or (C, H, W) array to a new (height, width). |
| `scaled_transform` | function | `scaled_transform(transform, old_shape, new_shape)` | Geotransform of a grid resampled from (H, W) to a new shape. |
| `scl_valid_mask` | function | `scl_valid_mask(scl, invalid=(0, 1, 3, 8, 9, 10)) -> ndarray[bool_]` | Mask of usable pixels from a Sentinel-2 SCL band. |
| `sentinel2_reflectance` | function | `sentinel2_reflectance(dn, offset=None, quantification=10000.0, processing_baseline=None, nodata=0) -> ndarray[float64]` | Sentinel-2 L1C or L2A digital numbers to reflectance. |
| `standardize` | function | `standardize(image, nodata=None) -> ndarray[float64]` | Per-band z-scores over finite values. |
| `to_tensor` | function | `to_tensor(image) -> ndarray` | Convert an (H, W) or (H, W, C) image to the band-first (C, H, W) layout. |
| `upsample_to_pan` | function | `upsample_to_pan(ms, shape, order=3) -> ndarray[float64]` | Interpolate multispectral bands onto the grid of the panchromatic band. |

### 7.3 Example

```python
import numpy as np

from unbihexium.preprocessing import (
    apply_mask, percentile_stretch, resample, scl_valid_mask, sentinel2_reflectance,
)

rng = np.random.default_rng(0)
dn = rng.integers(1000, 4000, size=(4, 64, 64)).astype("uint16")   # Sentinel-2 L2A digital numbers
reflectance = sentinel2_reflectance(dn, processing_baseline="05.11")  # offset -1000 since baseline 04.00
scl = rng.choice([4, 5, 8, 9], size=(64, 64))                        # scene classification layer
clear = apply_mask(reflectance, ~scl_valid_mask(scl))                 # clouds become NaN
print(round(float(np.nanmin(reflectance)), 4), round(float(np.isnan(clear[0]).mean()), 2))

display = percentile_stretch(clear, low=2, high=98)
half = resample(display, (32, 32), method="bilinear")
print(float(np.nanmin(display)), float(np.nanmax(display)), half.shape)
```

Output:

```text
0.0 0.49
0.0 1.0 (4, 32, 32)
```

## 8. unbihexium.sar

### 8.1 Overview

Array-level processing of synthetic aperture radar images. Radiometric calibration follows the look-up-table convention of Sentinel-1 products, $\sigma^0 = (|DN|^2 - N) / A_\sigma^2$ with an optional thermal noise power $N$, and $\gamma^0 = \sigma^0 / \cos\theta$. The speckle filters are the Lee filter [12], the refined Lee filter, the enhanced Lee filter [13], and the Frost [14], Kuan and Gamma MAP filters. Interferometric functions form interferograms and sample coherence, apply the Goldstein filter [15], unwrap phase and convert it to line-of-sight displacement $\Delta R = \lambda \phi / (4 \pi)$. Polarimetric functions compute covariance and coherency matrices and the Pauli, Freeman-Durden [16], Yamaguchi and Cloude-Pottier H/A/alpha [17] decompositions.

### 8.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `InterferometricResult` | class | `InterferometricResult(coherence, phase, unwrapped_phase=None)` | Products of interferometric processing. |
| `PolarimetricResult` | class | `PolarimetricResult(decomposition_type, components=..., entropy=None, anisotropy=None, alpha=None)` | Result of a polarimetric decomposition. |
| `amplitude_to_db` | function | `amplitude_to_db(amplitude, floor=-40.0) -> ndarray[float64]` | Amplitude to decibels, 20 log10(A), with a floor for zero amplitudes. |
| `calibrate_amplitude` | function | `calibrate_amplitude(data, calibration_factor=1.0) -> ndarray[float64]` | Scale amplitudes (or the modulus of complex samples) by a constant. |
| `coherency_matrix` | function | `coherency_matrix(hh, hv, vv, vh=None, window_size=5) -> ndarray[complex128]` | Pauli coherency matrix T3, shape (..., 3, 3). |
| `compute_beta0` | function | `compute_beta0(amplitude, calibration_lut=None) -> ndarray[float64]` | Radar brightness beta0 from calibrated amplitude. |
| `compute_coherence` | function | `compute_coherence(master, slave, window_size=5) -> ndarray[float64]` | Sample coherence \|<s1 s2*>\| / sqrt(<\|s1\|^2> <\|s2\|^2>) over a square window. |
| `compute_displacement` | function | `compute_displacement(unwrapped_phase, wavelength, incidence_angle=None, vertical=False) -> ndarray[float64]` | Line-of-sight range change R2 - R1 = lambda * phi / (4 pi). |
| `compute_gamma0` | function | `compute_gamma0(sigma0, incidence_angle, degrees=True) -> ndarray[float64]` | Backscatter coefficient gamma0 = sigma0 / cos(theta). |
| `compute_interferogram` | function | `compute_interferogram(master, slave, multilook=(1, 1)) -> tuple[ndarray[float64], ndarray[complex128]]` | Interferogram s1 * conj(s2) and its wrapped phase. |
| `compute_polarimetric_decomposition` | function | `compute_polarimetric_decomposition(hh, hv, vv, decomposition='pauli', vh=None, window_size=5) -> PolarimetricResult` | Decomposition selected by name. |
| `compute_sigma0` | function | `compute_sigma0(amplitude, incidence_angle, calibration_lut=None, degrees=True) -> ndarray[float64]` | Backscatter coefficient sigma0 = A^2 / K * sin(theta). |
| `covariance_matrix` | function | `covariance_matrix(hh, hv, vv, vh=None, window_size=5) -> ndarray[complex128]` | Lexicographic covariance matrix C3, shape (..., 3, 3). |
| `db_to_power` | function | `db_to_power(db) -> ndarray[float64]` | Decibels to linear intensity, 10^(dB / 10). |
| `enhanced_lee_filter` | function | `enhanced_lee_filter(data, window_size=5, looks=1.0, damping=1.0, amplitude=False) -> ndarray[float64]` | Enhanced Lee filter of Lopes, Touzi and Nezry (1990). |
| `equivalent_number_of_looks` | function | `equivalent_number_of_looks(intensity) -> float` | Equivalent number of looks, mean^2 / variance of a homogeneous intensity area. |
| `freeman_durden_decomposition` | function | `freeman_durden_decomposition(hh, hv, vv, vh=None, window_size=5) -> PolarimetricResult` | Freeman-Durden (1998) three-component decomposition. |
| `frost_filter` | function | `frost_filter(data, window_size=5, damping=2.0) -> ndarray[float64]` | Frost et al. (1982) exponentially weighted filter. |
| `gamma_map_filter` | function | `gamma_map_filter(data, window_size=5, looks=1.0) -> ndarray[float64]` | Gamma maximum a posteriori filter of Lopes et al. (1990) for intensities. |
| `goldstein_filter` | function | `goldstein_filter(interferogram, alpha=0.5, patch_size=32, smoothing=3) -> ndarray[complex128]` | Goldstein and Werner (1998) adaptive filter of a complex interferogram. |
| `h_a_alpha` | function | `h_a_alpha(t3) -> dict[str, ndarray[float64]]` | Entropy, anisotropy, mean alpha and eigenvalues of coherency matrices (..., 3, 3). |
| `h_alpha_decomposition` | function | `h_alpha_decomposition(hh, hv, vv, vh=None, window_size=5) -> PolarimetricResult` | Cloude-Pottier H / A / alpha decomposition of scattering amplitudes. |
| `h_alpha_zones` | function | `h_alpha_zones(entropy, alpha) -> ndarray[int64]` | Zones 1 to 9 of the H / alpha plane (Lee and Pottier, 2009), 0 for NaN. |
| `height_of_ambiguity` | function | `height_of_ambiguity(wavelength, slant_range, incidence_angle, perpendicular_baseline) -> float` | Height of ambiguity lambda R sin(theta) / (2 B_perp) of a repeat-pass pair. |
| `kuan_filter` | function | `kuan_filter(data, window_size=5, looks=1.0, amplitude=False) -> ndarray[float64]` | Kuan et al. (1985) linear MMSE filter. |
| `lee_filter` | function | `lee_filter(data, window_size=5, looks=1.0, amplitude=False) -> ndarray[float64]` | Lee (1980) minimum mean square error filter for multiplicative noise. |
| `los_to_vertical` | function | `los_to_vertical(range_change, incidence_angle) -> ndarray[float64]` | Vertical motion (positive up) from range change, assuming purely vertical motion. |
| `multilook` | function | `multilook(intensity, looks=(1, 1)) -> ndarray[float64]` | Average intensities over non-overlapping blocks of (rows, cols) pixels. |
| `pauli_decomposition` | function | `pauli_decomposition(hh, hv, vv, vh=None) -> PolarimetricResult` | Pauli decomposition powers of each pixel. |
| `pauli_rgb` | function | `pauli_rgb(hh, hv, vv, vh=None, percentile=98.0) -> ndarray[float64]` | Pauli RGB composite: red \|HH - VV\|, green 2 \|HV\|, blue \|HH + VV\|, scaled to [0, 1]. |
| `phase_residues` | function | `phase_residues(phase) -> ndarray[int64]` | Residue charges (-1, 0, +1) of the 2 x 2 loops of a wrapped phase image. |
| `phase_unwrapping` | function | `phase_unwrapping(phase, method='least_squares', coherence=None, coherence_threshold=0.3) -> ndarray[float64]` | Unwrap a wrapped phase image. |
| `power_to_db` | function | `power_to_db(power, floor=None) -> ndarray[float64]` | Intensity to decibels, 10 log10(p); non-positive values become NaN or the floor. |
| `radiometric_calibration` | function | `radiometric_calibration(dn, lut, noise=None, clip_negative=True) -> ndarray[float64]` | Calibrate a detected or complex image with look-up tables (Sentinel-1 style). |
| `refined_lee_filter` | function | `refined_lee_filter(data, looks=1.0) -> ndarray[float64]` | Refined Lee (1981) filter with edge-aligned windows (7 x 7). |
| `speckle_filter` | function | `speckle_filter(data, filter_type='lee', window_size=5, looks=1.0) -> ndarray[float64]` | Moving-window speckle filter selected by name. |
| `wrap_phase` | function | `wrap_phase(phase) -> ndarray[float64]` | Wrap phase values to [-pi, pi). |
| `yamaguchi_decomposition` | function | `yamaguchi_decomposition(hh, hv, vv, vh=None, window_size=5) -> PolarimetricResult` | Yamaguchi et al. (2005) four-component decomposition. |

### 8.3 Example

```python
import numpy as np

from unbihexium.sar import compute_coherence, lee_filter, power_to_db, pauli_decomposition

rng = np.random.default_rng(1)
intensity = rng.gamma(shape=1.0, scale=0.05, size=(128, 128))   # single-look speckle
filtered = lee_filter(intensity, window_size=7, looks=1.0)
print(round(float(intensity.std() / intensity.mean()), 2), round(float(filtered.std() / filtered.mean()), 2))
print(round(float(np.nanmean(power_to_db(filtered))), 2))

s1 = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
s2 = s1 * np.exp(1j * 0.3) + 0.5 * (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64)))
print(round(float(np.nanmean(compute_coherence(s1, s2, window_size=5))), 2))

hh, hv, vv = (rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32)) for _ in range(3))
print(sorted(pauli_decomposition(hh, hv, vv).components))
```

Output:

```text
1.0 0.22
-13.1
0.9
['dihedral', 'surface', 'volume']
```

## 9. unbihexium.terrain

### 9.1 Overview

Array functions for gridded, north-up digital elevation models with NaN as no-data: the Horn gradient [18] and the slope, aspect and hillshade derived from it, profile and plan curvature after Zevenbergen and Thorne [19], topographic position, ruggedness and vector ruggedness; priority-flood depression filling, D8 flow direction and accumulation, stream extraction, watersheds and the topographic wetness index $\ln(a / \tan\beta)$ of Beven and Kirkby [20]; and a line-of-sight viewshed with optional Earth curvature and refraction.

### 9.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `D8_NEIGHBOURS` | constant | tuple, 8 items | D8 offsets and codes. |
| `aspect` | function | `aspect(dem, resolution=1.0) -> ndarray[float64]` | Aspect: compass direction the slope faces (downslope), NaN on flat cells. |
| `curvature` | function | `curvature(dem, resolution=1.0) -> tuple[ndarray[float64], ndarray[float64]]` | Profile and plan curvature (Zevenbergen and Thorne, 1987), in 1 / length units. |
| `extract_streams` | function | `extract_streams(accumulation, threshold) -> ndarray[bool_]` | Stream cells: accumulation at or above a threshold. |
| `fill_depressions` | function | `fill_depressions(dem, epsilon=0.0) -> ndarray[float64]` | Fill depressions so that every valid cell drains to the border or to nodata. |
| `flow_accumulation` | function | `flow_accumulation(flow_dir, weights=None) -> ndarray[float64]` | Upstream cell count (or summed weights) of every cell. |
| `flow_direction_d8` | function | `flow_direction_d8(dem, resolution=1.0) -> ndarray[uint8]` | D8 flow direction codes of steepest descent. |
| `gradient` | function | `gradient(dem, resolution=1.0, z_factor=1.0) -> tuple[ndarray[float64], ndarray[float64]]` | Horn (1981) gradient: dz/dx towards east and dz/dy towards north. |
| `hillshade` | function | `hillshade(dem, resolution=1.0, azimuth=315.0, altitude=45.0, z_factor=1.0) -> ndarray[float64]` | Hillshade for a light source at the given azimuth and altitude (Burrough and McDonnell, 1998). |
| `roughness` | function | `roughness(dem) -> ndarray[float64]` | Roughness: largest minus smallest elevation of the 3 x 3 window (Wilson et al., 2007). |
| `slope` | function | `slope(dem, resolution=1.0, units='degrees', z_factor=1.0) -> ndarray[float64]` | Slope of the surface. |
| `total_curvature` | function | `total_curvature(dem, resolution=1.0) -> ndarray[float64]` | Total curvature -2 (D + E): positive on convex (upward bulging) cells. |
| `tpi` | function | `tpi(dem, radius=1) -> ndarray[float64]` | Topographic position index: elevation minus the mean of its neighbourhood (Weiss, 2001). |
| `tri` | function | `tri(dem, method='riley') -> ndarray[float64]` | Terrain ruggedness index of the 3 x 3 window. |
| `twi` | function | `twi(dem, resolution=1.0, fill=True, min_slope=0.1) -> ndarray[float64]` | Topographic wetness index ln(a / tan(beta)) (Beven and Kirkby, 1979). |
| `viewshed` | function | `viewshed(dem, observer, resolution=1.0, observer_height=1.7, target_height=0.0, max_distance=None, earth_curvature=False, refraction=0.13) -> ndarray[bool_]` | Visibility of every cell from an observer cell. |
| `vrm` | function | `vrm(dem, resolution=1.0, window_size=3) -> ndarray[float64]` | Vector ruggedness measure (Sappington et al., 2007): 0 flat or planar, 1 maximally rugged. |
| `watershed` | function | `watershed(flow_dir, outlet) -> ndarray[bool_]` | Cells that drain to an outlet cell. |

### 9.3 Example

```python
import numpy as np

from unbihexium.terrain import fill_depressions, flow_accumulation, flow_direction_d8, hillshade, slope

y, x = np.mgrid[0:100, 0:100]
dem = 200.0 + 0.5 * x + 20.0 * np.sin(y / 15.0)          # 30 m synthetic elevation model
print(round(float(np.nanmax(slope(dem, resolution=30.0))), 1))
shade = hillshade(dem, resolution=30.0, azimuth=315.0, altitude=45.0)
filled = fill_depressions(dem)
accumulation = flow_accumulation(flow_direction_d8(filled, resolution=30.0))
print(shade.shape, int(accumulation.max()))
```

Output:

```text
2.7
(100, 100) 7599
```

## 10. unbihexium.geostat

### 10.1 Overview

Analysis and interpolation of point data. `Variogram` estimates an isotropic empirical semivariogram with the Matheron estimator [21] or the robust Cressie-Hawkins estimator and fits a spherical, exponential, Gaussian, Matern, linear or power model by weighted least squares. `OrdinaryKriging` and `UniversalKriging` solve the kriging system with predictions and kriging variances [22], optionally with local neighbourhoods and cross-validation; `idw` is inverse distance weighting [23]. The spatial statistics are global Moran's I [24] and Geary's C [25] with analytical or permutation inference, local Moran's I [26] and the Getis-Ord Gi* statistic [27].

### 10.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `GearysC` | class | `GearysC(distance_threshold=None, row_standardized=False, assumption='randomization', permutations=0, seed=None)` | Geary's C with weights built from coordinates. |
| `KrigingResult` | class | `KrigingResult(predictions, variance, coordinates)` | Predictions of a kriging model. |
| `LocalStatisticResult` | class | `LocalStatisticResult(statistic, z_score, p_value, statistic_name, quadrant=None)` | Result of a local statistic. |
| `MoransI` | class | `MoransI(distance_threshold=None, row_standardized=True, assumption='randomization', permutations=0, seed=None)` | Moran's I with weights built from coordinates. |
| `OrdinaryKriging` | class | `OrdinaryKriging(variogram=None, n_neighbors=None)` | Ordinary kriging with an unknown constant mean. |
| `SpatialAutocorrelationResult` | class | `SpatialAutocorrelationResult(statistic, expected, variance, z_score, p_value, statistic_name, p_value_permutation=None)` | Result of a global autocorrelation statistic. |
| `UniversalKriging` | class | `UniversalKriging(variogram=None, drift_terms=1, n_neighbors=None)` | Universal kriging with a polynomial trend in the coordinates. |
| `Variogram` | class | `Variogram(n_lags=15, max_lag=None, model=VariogramModel.SPHERICAL, estimator='matheron', shape=0.5)` | Semivariogram estimation and model fitting. |
| `VariogramModel` | enum | `spherical`, `exponential`, `gaussian`, `matern`, `linear`, `power` | Variogram model families. |
| `VariogramResult` | class | `VariogramResult(lags, semivariance, model, nugget, sill, range_param, fitted_values=None, counts=None, shape=0.5)` | Result of fitting a variogram. |
| `contiguity_weights` | function | `contiguity_weights(shape, contiguity='rook') -> ndarray[float64]` | Binary contiguity weights of the cells of a grid (row-major order). |
| `distance_band_weights` | function | `distance_band_weights(coordinates, threshold=None, binary=True, power=1.0) -> ndarray[float64]` | Distance band weights: binary, or inverse distance to a power, within a threshold. |
| `empirical_variogram` | function | `empirical_variogram(coordinates, values, n_lags=15, max_lag=None, estimator='matheron')` | Binned empirical semivariogram. |
| `gearys_c` | function | `gearys_c(values, weights, row_standardized=False, assumption='randomization', permutations=0, seed=None) -> SpatialAutocorrelationResult` | Global Geary's C of values with a weight matrix. |
| `getis_ord_gi_star` | function | `getis_ord_gi_star(values, weights) -> LocalStatisticResult` | Getis-Ord Gi* z-scores of every location (Ord and Getis, 1995). |
| `grid_morans_i` | function | `grid_morans_i(array, contiguity='queen', assumption='randomization') -> SpatialAutocorrelationResult` | Moran's I of a raster with binary rook or queen contiguity; NaN cells are excluded. |
| `idw` | function | `idw(coordinates, values, targets, power=2.0, n_neighbors=None) -> ndarray[float64]` | Inverse distance weighting (Shepard, 1968). |
| `knn_weights` | function | `knn_weights(coordinates, k) -> ndarray[float64]` | k-nearest-neighbour binary weights (not symmetric in general). |
| `local_morans_i` | function | `local_morans_i(values, weights, row_standardized=True, permutations=999, seed=None) -> LocalStatisticResult` | Local Moran's I with conditional permutation inference (Anselin, 1995). |
| `morans_i` | function | `morans_i(values, weights, row_standardized=True, assumption='randomization', permutations=0, seed=None) -> SpatialAutocorrelationResult` | Global Moran's I of values with a weight matrix. |
| `row_standardize` | function | `row_standardize(weights) -> ndarray[float64]` | Divide each row by its sum; rows without neighbours stay zero. |
| `variogram_function` | function | `variogram_function(model, h, nugget, sill, range_param, shape=0.5) -> ndarray[float64]` | Semivariance of a model at distances h (gamma(0) = 0). |
| `weight_sums` | function | `weight_sums(weights) -> tuple[float, float, float]` | Weight sums S0, S1 and S2. |

`Variogram.fit(coordinates, values)` returns a `VariogramResult` and keeps the fitted model in the `Variogram` object, which is then passed to a kriging class; a kriging object fits an unfitted variogram itself.

### 10.3 Example

```python
import numpy as np

from unbihexium.geostat import OrdinaryKriging, Variogram, grid_morans_i, idw

rng = np.random.default_rng(1)
coords = rng.uniform(0, 1000, size=(60, 2))
values = np.sin(coords[:, 0] / 200.0) + rng.normal(0, 0.1, size=60)

variogram = Variogram(n_lags=10, model="spherical")
fitted = variogram.fit(coords, values)                 # VariogramResult
print(fitted.model.value, len(fitted.lags), fitted.nugget >= 0)

kriging = OrdinaryKriging(variogram=variogram).fit(coords, values)
result = kriging.predict(np.array([[500.0, 500.0], [100.0, 900.0]]))
print(np.round(result.predictions, 2), np.round(result.variance, 3))
print(np.round(idw(coords, values, np.array([[500.0, 500.0]])), 2))

field = np.add.outer(np.arange(20.0), np.arange(20.0))  # smooth gradient
print(round(grid_morans_i(field).statistic, 3))
```

Output:

```text
spherical 10 True
[0.5  0.71] [0.135 0.085]
[0.23]
0.922
```

## 11. unbihexium.analysis

### 11.1 Overview

GIS analysis tools: zonal statistics of rasters with integer or polygon zones; suitability analysis with Analytic Hierarchy Process weights [28], linear and fuzzy standardisation, reclassification and weighted overlay with Boolean constraints; raster cost distance and least-cost paths; and graph routing with Dijkstra and A*, service areas and accessibility in `NetworkAnalyzer`.

### 11.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `AHP` | class | `AHP(criteria=None, method='eigenvector')` | Analytic Hierarchy Process weights from a pairwise comparison matrix. |
| `AStarPathfinder` | class | `AStarPathfinder(default_heuristic=Heuristic.EUCLIDEAN)` | A* on a graph built node by node. |
| `AccessibilityResult` | class | `AccessibilityResult(travel_times, origin, threshold, node_ids=...)` | Costs from one origin to every node. |
| `NetworkAnalyzer` | class | `NetworkAnalyzer()` | Weighted graph with coordinates and routing queries. |
| `Route` | class | `Route(nodes, distance, time=None, geometry=...)` | A path through the network. |
| `SuitabilityResult` | class | `SuitabilityResult(suitability, weights=..., consistency_ratio=None, raster=None)` | Result of a weighted overlay. |
| `WeightedOverlay` | class | `WeightedOverlay(rescale=False)` | Weighted linear combination of factor layers with Boolean constraints. |
| `ZonalResult` | class | `ZonalResult(zone_id, count, sum, mean, std, min, max, median, majority=None, minority=None, variety=0, percentiles=...)` | Statistics of one zone. |
| `ZonalStatistics` | class | `ZonalStatistics(nodata=None, zone_nodata=None)` | Zonal statistics as a table of dictionaries. |
| `cost_distance` | function | `cost_distance(cost, sources, resolution=1.0, connectivity=8) -> tuple[ndarray[float64], ndarray[int64]]` | Accumulated least cost from the nearest source to every cell. |
| `fuzzy_membership` | function | `fuzzy_membership(values, a, b, shape='sigmoidal') -> ndarray[float64]` | Fuzzy membership between control points a (0) and b (1) (Eastman, 2009). |
| `least_cost_path` | function | `least_cost_path(cost, start, end, resolution=1.0, connectivity=8) -> tuple[list[tuple[int, int]], float]` | Cheapest path between two cells. |
| `rasterize_zones` | function | `rasterize_zones(geometries, shape, transform, ids=None, all_touched=False) -> ndarray[int32]` | Burn polygons into an integer zone raster (0 = outside every polygon). |
| `reclassify` | function | `reclassify(values, breaks, scores) -> ndarray[float64]` | Map value ranges to scores: [breaks[i-1], breaks[i]) -> scores[i]. |
| `rescale_linear` | function | `rescale_linear(values, low=None, high=None, increasing=True) -> ndarray[float64]` | Linear rescaling to [0, 1] between low and high. |
| `weighted_overlay` | function | `weighted_overlay(layers, weights, normalize=True, constraints=None, names=None) -> SuitabilityResult` | Weighted overlay of arrays or rasters, returning weights and a raster. |
| `zonal_statistics` | function | `zonal_statistics(raster, zones, stats=None, percentiles=None, nodata=None) -> list[ZonalResult]` | Zonal statistics as one record per zone. |
| `zonal_table` | function | `zonal_table(values, zones, stats=None, percentiles=None, nodata=None, zone_nodata=None) -> dict[str, dict[Any, float]]` | Statistics of all zones at once. |

### 11.3 Further public names of unbihexium.analysis.network

| Name in `unbihexium.analysis.network` | Kind | Signature | Description |
| --- | --- | --- | --- |
| `AStarResult` | class | `AStarResult(path, cost, nodes_explored, success)` | Result of an A* search. |
| `Heuristic` | class | `Heuristic(*values)` | Heuristics available to A*. |
| `astar` | function | `astar(nodes, adj, start, goal, heuristic=Heuristic.EUCLIDEAN) -> AStarResult` | A* search on an adjacency list. |
| `euclidean_distance` | function | `euclidean_distance(x1, y1, x2, y2) -> float` | Straight-line distance between (x1, y1) and (x2, y2). |
| `haversine_distance` | function | `haversine_distance(lat1, lon1, lat2, lon2, earth_radius=6371.0088) -> float` | Great-circle distance of two (latitude, longitude) points in degrees. |
| `manhattan_distance` | function | `manhattan_distance(x1, y1, x2, y2) -> float` | Manhattan distance \|x2 - x1\| + \|y2 - y1\|. |

### 11.4 Example

```python
import numpy as np

from unbihexium.analysis import AHP, least_cost_path, weighted_overlay, zonal_table

rng = np.random.default_rng(3)
values = rng.uniform(0, 1, size=(40, 40))
zones = np.repeat(np.arange(1, 5), 400).reshape(40, 40)            # four horizontal strips
table = zonal_table(values, zones, stats=["count", "mean"])
print(sorted(table), table["count"][1], round(table["mean"][1], 3))

ahp = AHP(criteria=["slope", "distance", "soil"])
ahp.fit(np.array([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]]))
weights = ahp.weights_dict()
print({k: round(v, 3) for k, v in weights.items()}, round(ahp.consistency_ratio(), 3))

layers = [rng.uniform(0, 1, size=(40, 40)) for _ in range(3)]
result = weighted_overlay(layers, list(weights.values()), names=list(weights))
print(result.suitability.shape, round(float(np.nanmean(result.suitability)), 3))

cost = np.ones((40, 40)); cost[10:30, 20] = 100.0                 # a barrier
path, total = least_cost_path(cost, (20, 0), (20, 39), resolution=10.0)
print(len(path), round(total, 1))
```

Output:

```text
['count', 'mean'] 400.0 0.507
{'slope': 0.637, 'distance': 0.258, 'soil': 0.105} 0.033
(40, 40) 0.495
40 472.8
```

## 12. unbihexium.postprocessing

### 12.1 Overview

Turns scores and class maps into clean map products: activations (sigmoid, softmax, argmax), thresholds, confidence masks, prediction entropy and margin; binary morphology, removal of small objects and holes, a minimum mapping unit sieve, a majority filter and connected components with statistics; raster to polygon vectorisation with Douglas-Peucker simplification and GeoDataFrame output; and tile positions, blending weights and stitching of tiled predictions.

### 12.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `argmax` | function | `argmax(predictions, axis=0) -> ndarray` | Class map from scores along an axis. |
| `as_affine` | function | `as_affine(transform) -> Affine` | Affine transform from any accepted representation. |
| `blend_weights` | function | `blend_weights(tile_shape, overlap=0, blend='mean') -> ndarray[float64]` | Per-pixel weights of a tile. |
| `component_statistics` | function | `component_statistics(labels, values=None, pixel_area=1.0, source=None) -> list[dict[str, Any]]` | Statistics of every labelled region. |
| `confidence_mask` | function | `confidence_mask(probabilities, min_confidence=0.5, min_margin=0.0, nodata=255) -> ndarray[uint8]` | Class map with pixels below a confidence or margin set to nodata. |
| `connected_components` | function | `connected_components(image, connectivity=8, background=0) -> tuple[ndarray[int32], int]` | Label connected regions of a mask, or of equal values in a class map. |
| `fill_small_holes` | function | `fill_small_holes(mask, max_size=100, connectivity=4) -> ndarray` | Fill holes of the background with fewer than max_size pixels. |
| `majority_filter` | function | `majority_filter(labels, size=3, nodata=None) -> ndarray` | Modal filter of a class map in a square window. |
| `margin` | function | `margin(probabilities) -> ndarray[float64]` | Difference between the largest and second largest probability. |
| `morphology_clean` | function | `morphology_clean(mask, operation='open', kernel_size=3, shape='square', iterations=1) -> ndarray` | Binary morphology of a mask. |
| `polygons_to_geodataframe` | function | `polygons_to_geodataframe(image, transform=None, crs=None, connectivity=4, skip_values=(0,), simplify_tolerance=None) -> Any` | GeoDataFrame of the polygons of a class map. |
| `prediction_entropy` | function | `prediction_entropy(probabilities) -> ndarray[float64]` | Normalised Shannon entropy of class probabilities, in [0, 1]. |
| `raster_to_polygons` | function | `raster_to_polygons(image, transform=None, mask=None, connectivity=4, skip_values=(0,)) -> list[tuple[Any, float]]` | Polygons of connected regions of equal value. |
| `remove_small_objects` | function | `remove_small_objects(mask, min_size=100, connectivity=8) -> ndarray` | Remove connected components with fewer than min_size pixels. |
| `sieve` | function | `sieve(labels, min_size, connectivity=4, nodata=None) -> ndarray` | Minimum mapping unit: merge small regions into their largest neighbour. |
| `sigmoid` | function | `sigmoid(x) -> ndarray[float32]` | Numerically stable logistic function. |
| `simplify_polygons` | function | `simplify_polygons(polygons, tolerance, preserve_topology=True) -> list[Any]` | Douglas-Peucker simplification of polygons. |
| `softmax` | function | `softmax(x, axis=0) -> ndarray[float32]` | Numerically stable softmax along one axis. |
| `stitch_tiles` | function | `stitch_tiles(tiles, positions, output_shape, overlap=0, blend='mean') -> ndarray[float64]` | Weighted average of overlapping tiles. |
| `structuring_element` | function | `structuring_element(size=3, shape='square') -> ndarray[bool_]` | Footprint of a given shape and size. |
| `threshold` | function | `threshold(predictions, threshold=0.5, above=True) -> ndarray[uint8]` | Binary map of values above (or below) a threshold. |
| `tile_positions` | function | `tile_positions(shape, tile_size, overlap=0) -> list[tuple[int, int]]` | Origins of overlapping tiles that cover an image. |

### 12.3 Example

```python
import numpy as np

from unbihexium.postprocessing import (
    confidence_mask, connected_components, sieve, softmax, stitch_tiles, tile_positions,
)

rng = np.random.default_rng(4)
logits = rng.normal(size=(3, 64, 64))                  # three classes
probabilities = softmax(logits, axis=0)
labels = confidence_mask(probabilities, min_confidence=0.5, nodata=255)
print(labels.dtype, sorted(np.unique(labels).tolist()))

cleaned = sieve(np.where(labels == 255, 0, labels), min_size=8, nodata=0)
components, count = connected_components(cleaned > 0, connectivity=8)
print(components.shape, count > 0)

positions = tile_positions((100, 100), tile_size=64, overlap=16)
tiles = [np.ones((64, 64)) for _ in positions]
mosaic = stitch_tiles(tiles, positions, (100, 100), overlap=16, blend="mean")
print(positions, float(mosaic.min()), float(mosaic.max()))
```

Output:

```text
uint8 [0, 1, 2, 255]
(64, 64) True
[(0, 0), (0, 36), (36, 0), (36, 36)] 1.0 1.0
```

## 13. unbihexium.metrics

### 13.1 Overview

Accuracy and quality measures of map products. Every error matrix has the reference classes in rows and the map classes in columns. `accuracy_assessment` reports overall, producer's and user's accuracy, F1, IoU, Cohen's kappa [29] and quantity and allocation disagreement; `stratified_area_estimate` gives unbiased area estimates with confidence intervals under stratified random sampling following Olofsson et al. [30], and `sample_allocation` the stratum sample sizes. Image quality is measured with PSNR, SSIM [31], the spectral angle, ERGAS and the Q index. Streaming metrics used during training are in `unbihexium.ai.evaluation`.

### 13.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `AccuracyAssessment` | class | `AccuracyAssessment(classes, overall_accuracy, producers_accuracy, users_accuracy, f1, iou, kappa, quantity_disagreement, allocation_disagreement, total)` | Accuracy measures of an error matrix. |
| `AreaEstimate` | class | `AreaEstimate(classes, proportion, proportion_se, area, area_se, area_ci, mapped_area, overall_accuracy, overall_accuracy_se, users_accuracy, users_accuracy_se, producers_accuracy, producers_accuracy_se, z)` | Unbiased area and accuracy estimates with uncertainty. |
| `accuracy` | function | `accuracy(pred, target) -> float` | Fraction of equal labels. |
| `accuracy_assessment` | function | `accuracy_assessment(matrix, classes=None) -> AccuracyAssessment` | Accuracy measures of an error matrix (rows reference, columns map). |
| `bias` | function | `bias(pred, target) -> float` | Mean error, estimate minus reference. |
| `calculate_ergas` | function | `calculate_ergas(reference, estimate, ratio=4.0) -> float` | Relative dimensionless global error in synthesis (ERGAS). |
| `calculate_qindex` | function | `calculate_qindex(reference, estimate, block_size=8) -> float` | Universal image quality index Q, averaged over bands. |
| `calculate_sam` | function | `calculate_sam(reference, estimate) -> float` | Mean spectral angle in degrees over pixels where it is defined. |
| `change_detection_metrics` | function | `change_detection_metrics(reference, predicted, valid=None) -> dict[str, float]` | Accuracy of a binary change map against a reference change map. |
| `change_map` | function | `change_map(before, after, nodata=None) -> ndarray[bool_]` | Pixels whose class differs between two dates. |
| `cohen_kappa` | function | `cohen_kappa(matrix) -> float` | Cohen's kappa of an error matrix. |
| `confusion_matrix` | function | `confusion_matrix(reference, predicted, labels=None, ignore=None, weights=None) -> ndarray` | Error matrix of reference and map labels; rows are reference classes. |
| `dice` | function | `dice(pred, target, smooth=1e-06) -> float` | Dice coefficient (F1) of binary masks. |
| `ergas` | function | `ergas(reference, estimate, ratio=4.0) -> float` | Relative dimensionless global error in synthesis (ERGAS). |
| `estimated_error_matrix` | function | `estimated_error_matrix(matrix, mapped_area) -> ndarray[float64]` | Estimated population error matrix of proportions (rows reference, columns map). |
| `f1_score` | function | `f1_score(pred, target) -> float` | F1 score of a binary prediction. |
| `iou` | function | `iou(pred, target, smooth=1e-06) -> float` | Intersection over union of binary masks. |
| `mae` | function | `mae(pred, target) -> float` | Mean absolute error. |
| `mean_iou` | function | `mean_iou(pred, target, num_classes) -> float` | Mean IoU over the classes present in the target. |
| `pearson_r` | function | `pearson_r(pred, target) -> float` | Pearson correlation coefficient. |
| `precision` | function | `precision(pred, target) -> float` | Precision of a binary prediction. |
| `psnr` | function | `psnr(pred, target, max_val=1.0) -> float` | Peak signal-to-noise ratio in decibels, ignoring NaN. |
| `q_index` | function | `q_index(reference, estimate, block_size=8) -> float` | Universal image quality index Q, averaged over bands. |
| `r_squared` | function | `r_squared(pred, target) -> float` | Coefficient of determination against the 1:1 line. |
| `recall` | function | `recall(pred, target) -> float` | Recall of a binary prediction. |
| `regression_report` | function | `regression_report(pred, target) -> dict[str, float]` | All error statistics of a set of pairs. |
| `rmse` | function | `rmse(pred, target) -> float` | Root mean square error. |
| `sam` | function | `sam(reference, estimate) -> float` | Mean spectral angle in degrees over pixels where it is defined. |
| `sample_allocation` | function | `sample_allocation(mapped_area, expected_users_accuracy, target_se=0.01, rare_minimum=50) -> ndarray[int64]` | Stratum sample sizes for a target standard error of overall accuracy. |
| `spectral_angle` | function | `spectral_angle(reference, estimate) -> ndarray[float64]` | Spectral angle of every pixel in degrees, NaN where undefined. |
| `ssim` | function | `ssim(pred, target, data_range=1.0, sigma=1.5) -> float` | Mean structural similarity of (H, W) or (C, H, W) images. |
| `stratified_area_estimate` | function | `stratified_area_estimate(matrix, mapped_area, confidence=0.95, classes=None) -> AreaEstimate` | Area and accuracy estimates of Olofsson et al. (2014). |
| `transition_matrix` | function | `transition_matrix(before, after, labels=None, nodata=None) -> ndarray` | From-to counts of classes; rows are the first date, columns the second. |
| `transition_summary` | function | `transition_summary(matrix, classes=None) -> dict[str, dict[str, float]]` | Gross gains, losses, net change and swap of every class. |
| `ubrmse` | function | `ubrmse(pred, target) -> float` | Unbiased root mean square error. |

### 13.3 Example

```python
import numpy as np

from unbihexium.metrics import (
    accuracy_assessment, confusion_matrix, psnr, rmse, ssim, stratified_area_estimate,
)

rng = np.random.default_rng(1)
reference = rng.integers(0, 3, size=(64, 64))
predicted = np.where(rng.random((64, 64)) < 0.9, reference, (reference + 1) % 3)
matrix = confusion_matrix(reference, predicted)          # rows reference, columns map
report = accuracy_assessment(matrix, classes=["water", "forest", "urban"])
print(round(report.overall_accuracy, 3), round(report.kappa, 3))

# Olofsson et al. (2014): sample counts per stratum and mapped areas in hectares.
sample = np.array([[97, 0, 3], [3, 279, 18], [2, 1, 97]])
areas = stratified_area_estimate(sample, mapped_area=[20000.0, 360000.0, 18000.0])
print(np.round(areas.area), np.round(areas.area_ci))

image = rng.random((3, 32, 32))
noisy = np.clip(image + rng.normal(0, 0.05, size=image.shape), 0, 1)
print(round(psnr(noisy, image), 1), round(ssim(noisy, image), 3), round(rmse(noisy, image), 3))
```

Output:

```text
0.898 0.848
[ 19477. 362048.  16474.] [ 986. 2857. 2863.]
26.3 0.986 0.048
```

## 14. unbihexium.visualization

### 14.1 Overview

Rendering without a plotting library: colour maps and look-up tables (including palettes for common indices and the ESA WorldCover legend), class breaks, RGB composites of Sentinel-2 and Landsat 8/9 band combinations with stretches and gamma, mask overlays, hillshade and multidirectional hillshade, shaded images, PNG output with world files, quicklooks and legends. Only `legend_figure` needs matplotlib.

### 14.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `COLORMAPS` | constant | dict, 6 entries | Named colour maps. |
| `DEFAULT_PALETTE` | constant | dict, 6 entries | Default class colours. |
| `INDEX_COLORMAPS` | constant | dict, 7 entries | Colour maps of common indices. |
| `LANDSAT89_COMPOSITES` | constant | dict, 5 entries | Landsat band combinations. |
| `SENTINEL2_COMPOSITES` | constant | dict, 6 entries | Sentinel-2 band combinations. |
| `WORLDCOVER_PALETTE` | constant | dict, 11 entries | ESA WorldCover legend. |
| `alpha_composite` | function | `alpha_composite(background, layer, opacity=1.0) -> ndarray[uint8]` | Composite an RGBA layer over an RGB or RGBA image ("over" operator). |
| `apply_colormap` | function | `apply_colormap(values, cmap='viridis', vmin=None, vmax=None, nodata=None, reverse=False, n=256) -> ndarray[uint8]` | Continuous values to RGBA through a colour map. |
| `classify_colors` | function | `classify_colors(values, breaks, colors) -> ndarray[uint8]` | Values to RGBA by class breaks. |
| `color_to_rgb` | function | `color_to_rgb(color) -> tuple[int, int, int]` | Colour as an (r, g, b) tuple from a hex string or a tuple. |
| `colorize_classes` | function | `colorize_classes(labels, palette=..., nodata=None) -> ndarray[uint8]` | Class map to RGBA with transparent nodata and unknown classes. |
| `colorize_mask` | function | `colorize_mask(mask, colormap=None) -> ndarray[uint8]` | Class map to RGB. |
| `colormap_lut` | function | `colormap_lut(name, n=256, reverse=False) -> ndarray[uint8]` | Lookup table of a colour map. |
| `create_legend` | function | `create_legend(labels, colors, size=(200, 20)) -> ndarray[uint8]` | Legend array of colour patches and labels, with the geometry of earlier releases. |
| `hex_to_rgb` | function | `hex_to_rgb(color) -> tuple[int, int, int]` | "#rrggbb" to an (r, g, b) tuple of integers. |
| `hillshade` | function | `hillshade(dem, cellsize=1.0, azimuth=315.0, altitude=45.0, z_factor=1.0, as_uint8=False) -> ndarray` | Illumination of a DEM for one sun position. |
| `legend_figure` | function | `legend_figure(labels, colors, title=None) -> Any` | Matplotlib figure with a legend of colour patches. |
| `legend_image` | function | `legend_image(labels, colors, patch=(24, 16), padding=6, width=None, background=(255, 255, 255), text_color=(0, 0, 0)) -> ndarray[uint8]` | Legend of colour patches and text labels drawn with Pillow. |
| `multidirectional_hillshade` | function | `multidirectional_hillshade(dem, cellsize=1.0, azimuths=(225.0, 270.0, 315.0, 360.0), weights=None, altitude=45.0, z_factor=1.0) -> ndarray[float64]` | Weighted blend of hillshades from several directions. |
| `normalize_for_display` | function | `normalize_for_display(image, percentile=(2, 98)) -> ndarray[uint8]` | Percentile stretch of an image to uint8 for display. |
| `overlay_mask` | function | `overlay_mask(image, mask, alpha=0.5, color=(255, 0, 0)) -> ndarray[uint8]` | Colour a binary mask over an RGB image. |
| `quicklook` | function | `quicklook(path, image, bands=(0, 1, 2), band_names=None, max_size=1024, cmap='viridis', transform=None, **stretch) -> Path` | Write a reduced-size quicklook of a band stack or a single band. |
| `rgb_composite` | function | `rgb_composite(stack, bands=(0, 1, 2), band_names=None, stretch='percentile', low=2.0, high=98.0, gamma=1.0, per_band=True, nodata=None, alpha=False) -> ndarray[uint8]` | Three bands of a (C, H, W) stack to an 8-bit colour image. |
| `save_png` | function | `save_png(path, image, transform=None) -> Path` | Write an 8-bit image as PNG. |
| `shade_image` | function | `shade_image(rgb, shade, strength=0.6) -> ndarray[uint8]` | Multiply an RGB image by a hillshade. |
| `slope_aspect` | function | `slope_aspect(dem, cellsize=1.0, z_factor=1.0) -> tuple[ndarray[float64], ndarray[float64]]` | Slope and aspect in radians from Horn's gradients. |
| `to_uint8` | function | `to_uint8(image) -> ndarray[uint8]` | Values in [0, 1] to uint8, NaN as zero. |
| `world_file_lines` | function | `world_file_lines(transform) -> list[str]` | World file lines for an affine transform (a, b, c, d, e, f). |

### 14.3 Example

```python
import numpy as np

from unbihexium.visualization import apply_colormap, colorize_classes, rgb_composite, save_png

rng = np.random.default_rng(5)
stack = rng.uniform(0.0, 0.3, size=(4, 64, 64))
rgb = rgb_composite(stack, bands=(2, 1, 0), stretch="percentile")     # true colour
print(rgb.shape, rgb.dtype)
save_png("true_colour.png", rgb, transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6700000.0))

ndvi = rng.uniform(-1, 1, size=(64, 64))
rgba = apply_colormap(ndvi, cmap="viridis", vmin=-1, vmax=1)
print(rgba.shape, rgba.dtype)
classes = colorize_classes(rng.choice([10, 50, 80], size=(64, 64)))   # ESA WorldCover codes
print(classes.shape)
```

Output:

```text
(64, 64, 3) uint8
(64, 64, 4) uint8
(64, 64, 4)
```

## 15. unbihexium.ai

### 15.1 Overview

Task APIs, inference, training and evaluation for the model zoo. A task API opens a model, checks the bands of the input, runs tiled inference and returns a result object with georeferencing:

| Base class | Task | Result | Specialised classes (default family) |
| --- | --- | --- | --- |
| `ObjectDetector` (`object_detector`) | detection | `DetectionResult` | `ShipDetector` (`ship_detector`), `BuildingDetector` (`building_detector`), `AircraftDetector` (`aircraft_detector`), `VehicleDetector` (`vehicle_detector`), `SARShipDetector` (`sar_ship_detector`), `CropDetector` (`crop_detector`), `GreenhouseDetector` (`greenhouse_detector`), `PivotDetector` (`pivot_inventory`), `FireDetector` (`fire_monitor`) |
| `SemanticSegmenter` (`lulc_classifier`) | segmentation | `SegmentationResult` | `LandCoverClassifier` (`lulc_classifier`), `WaterDetector` (`water_surface_detector`), `CloudMasker` (`cloud_mask`), `CropClassifier` (`crop_classifier`), `FloodMapper` (`sar_flood_detector`), `OilSpillDetector` (`sar_oil_spill_detector`) |
| `ChangeDetector` (`change_detector`) | change detection | `SegmentationResult` | method `predict_pair(before, after)` |
| `DenseRegressor` (`tree_height_estimator`) | dense regression, spectral index | `RegressionResult` | `TreeHeightEstimator` (`tree_height_estimator`), `LandSurfaceTemperature` (`land_surface_temperature`), `NDVICalculator` (`ndvi_calculator`) |
| `SceneRegressor` (`yield_predictor`) | scene regression | `RegressionResult` | `YieldPredictor` (`yield_predictor`) |
| `SuperResolution` (`super_resolution`) | super-resolution | `SuperResolutionResult` | methods `predict` and `enhance` |
| `Enhancer` (`pansharpening`) | enhancement | `EnhancementResult` | |

Every task API accepts `model` (a catalogue family, model id, checkpoint, ONNX file or `ZooModel`; default the family in parentheses), `variant` (default `base`), `weights` (a checkpoint that replaces `model`), `device` (default `cpu`), `backend` (`auto`, `torch` or `onnx`), `tile_size`, `overlap` (default 0.25) and `batch_size` (default 4); detection and segmentation classes add their thresholds. Inputs are a `Raster`, an array or a raster file. The detectors are CenterNet networks [32], the segmentation, change detection and dense regression models U-Nets [33], and the super-resolution models EDSR-style residual networks [34]. Apart from the 28 spectral index models, all 520 catalogue models are untrained starter models; their predictions are not meaningful before training. Importing `unbihexium.ai` does not import PyTorch.

### 15.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `AircraftDetector` | class | `AircraftDetector(model=None, threshold=0.5, iou_threshold=0.5, max_detections=1000, **kwargs)` | Aircraft on airfields. |
| `BuildingDetector` | class | `BuildingDetector(model=None, threshold=0.5, iou_threshold=0.5, max_detections=1000, **kwargs)` | Buildings in very high resolution imagery. |
| `ChangeDetector` | class | `ChangeDetector(model=None, threshold=0.5, return_probabilities=False, **kwargs)` | Bi-temporal change detection. |
| `CloudMasker` | class | `CloudMasker(model=None, threshold=0.5, return_probabilities=False, **kwargs)` | Clouds and cloud shadows. |
| `CropClassifier` | class | `CropClassifier(model=None, threshold=0.5, return_probabilities=False, **kwargs)` | Crop types. |
| `CropDetector` | class | `CropDetector(model=None, threshold=0.5, iou_threshold=0.5, max_detections=1000, **kwargs)` | Crop fields and parcels. |
| `DenseRegressor` | class | `DenseRegressor(model=None, variant=None, weights=None, device='cpu', backend='auto', tile_size=None, overlap=0.25, batch_size=4)` | Per-pixel regression with a model zoo U-Net or a spectral index formula. |
| `Detection` | class | `Detection(bbox, confidence, class_id, class_name, geo_bbox=None)` | A single detected object. |
| `DetectionResult` | class | `DetectionResult(detections=..., source='', model_id='', crs='EPSG:4326')` | Result of object detection on one image. |
| `EnhancementResult` | class | `EnhancementResult(raster=None, bands=..., source='', model_id='')` | Result of image enhancement (image-to-image models). |
| `Enhancer` | class | `Enhancer(model=None, variant=None, weights=None, device='cpu', backend='auto', tile_size=None, overlap=0.25, batch_size=4)` | Image-to-image models on the input grid. |
| `FireDetector` | class | `FireDetector(model=None, threshold=0.5, iou_threshold=0.5, max_detections=1000, **kwargs)` | Active fires and burn scars. |
| `FloodMapper` | class | `FloodMapper(model=None, threshold=0.5, return_probabilities=False, **kwargs)` | Floods in SAR imagery. |
| `GreenhouseDetector` | class | `GreenhouseDetector(model=None, threshold=0.5, iou_threshold=0.5, max_detections=1000, **kwargs)` | Greenhouses. |
| `LandCoverClassifier` | class | `LandCoverClassifier(model=None, threshold=0.5, return_probabilities=False, **kwargs)` | Land use and land cover. |
| `LandSurfaceTemperature` | class | `LandSurfaceTemperature(model=None, variant=None, weights=None, device='cpu', backend='auto', tile_size=None, overlap=0.25, batch_size=4)` | Land surface temperature in kelvin. |
| `NDVICalculator` | class | `NDVICalculator(model=None, variant=None, weights=None, device='cpu', backend='auto', tile_size=None, overlap=0.25, batch_size=4)` | Normalised difference vegetation index, exact formula. |
| `ObjectDetector` | class | `ObjectDetector(model=None, threshold=0.5, iou_threshold=0.5, max_detections=1000, **kwargs)` | Object detector backed by a model zoo CenterNet. |
| `OilSpillDetector` | class | `OilSpillDetector(model=None, threshold=0.5, return_probabilities=False, **kwargs)` | Oil spills in SAR imagery. |
| `PivotDetector` | class | `PivotDetector(model=None, threshold=0.5, iou_threshold=0.5, max_detections=1000, **kwargs)` | Centre-pivot irrigation fields. |
| `Predictor` | class | `Predictor(source, variant=None, device='cpu', backend='auto', tile_size=None, overlap=0.25, batch_size=4, normalization=None)` | Tiled inference for a model zoo model. |
| `RegressionResult` | class | `RegressionResult(values, names=..., units=..., model_id='', source='', crs='EPSG:4326', transform=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0))` | Result of dense or scene-level regression on one image. |
| `SARShipDetector` | class | `SARShipDetector(model=None, threshold=0.5, iou_threshold=0.5, max_detections=1000, **kwargs)` | Ships in SAR amplitude imagery (VV, VH). |
| `SceneRegressor` | class | `SceneRegressor(model=None, variant=None, weights=None, device='cpu', backend='auto', tile_size=None, overlap=0.25, batch_size=4)` | Scene-level regression with a pooled encoder. |
| `SegmentationResult` | class | `SegmentationResult(mask, classes=..., model_id='', probabilities=None, source='', crs='EPSG:4326', transform=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0), nodata=255)` | Result of semantic segmentation or change detection on one image. |
| `SemanticSegmenter` | class | `SemanticSegmenter(model=None, threshold=0.5, return_probabilities=False, **kwargs)` | Per-pixel classification with a model zoo U-Net. |
| `ShipDetector` | class | `ShipDetector(model=None, threshold=0.5, iou_threshold=0.5, max_detections=1000, **kwargs)` | Ships in optical imagery. |
| `SuperResolution` | class | `SuperResolution(model=None, scale_factor=None, tile_size=256, **kwargs)` | Super-resolution of multispectral imagery. |
| `SuperResolutionResult` | class | `SuperResolutionResult(raster=None, scale_factor=2, source='', model_id='')` | Result of super-resolution. |
| `TreeHeightEstimator` | class | `TreeHeightEstimator(model=None, variant=None, weights=None, device='cpu', backend='auto', tile_size=None, overlap=0.25, batch_size=4)` | Canopy height in metres. |
| `VehicleDetector` | class | `VehicleDetector(model=None, threshold=0.5, iou_threshold=0.5, max_detections=1000, **kwargs)` | Cars, trucks and buses. |
| `WaterDetector` | class | `WaterDetector(model=None, threshold=0.5, return_probabilities=False, **kwargs)` | Water surfaces. |
| `YieldPredictor` | class | `YieldPredictor(model=None, variant=None, weights=None, device='cpu', backend='auto', tile_size=None, overlap=0.25, batch_size=4)` | Crop yield per field. |
| `predict` | function | `predict(model, image, **options) -> Result` | Run a model on an image, raster or raster file. |
| `task_api` | function | `task_api(model, **options) -> ZooTask` | Task API for a model, with the model already opened. |
| `write_result` | function | `write_result(result, path) -> Path` | Write a result to a file in the natural format of its task. |

### 15.3 Members of the result classes

| Member of `DetectionResult` | Signature | Description |
| --- | --- | --- |
| `count` | property | Number of detections. |
| `filter_by_confidence` | `filter_by_confidence(threshold) -> DetectionResult` | Keep only detections with at least the given confidence. |
| `filter_by_class` | `filter_by_class(*names) -> DetectionResult` | Keep only detections of the given classes. |
| `counts_by_class` | `counts_by_class() -> dict[str, int]` | Number of detections per class name. |
| `as_arrays` | `as_arrays()` | Boxes, scores and class ids as arrays, as used by the metrics. |
| `to_geojson` | `to_geojson(pixel_coordinates=False) -> dict[str, Any]` | Export the detections as a GeoJSON FeatureCollection. |

| Member of `SegmentationResult` | Signature | Description |
| --- | --- | --- |
| `num_classes` | property | Number of classes. |
| `class_mask` | `class_mask(cls) -> ndarray[bool_]` | Binary mask of one class, by index or name. |
| `class_fractions` | `class_fractions() -> dict[str, float]` | Fraction of the valid pixels per class. |
| `class_areas` | `class_areas() -> dict[str, float]` | Area per class in square map units (square metres for projected CRS). |
| `to_raster` | `to_raster() -> Raster` | Export the mask as a single-band raster. |

| Member of `RegressionResult` | Signature | Description |
| --- | --- | --- |
| `is_dense` | property | Whether the values form a map. |
| `output` | `output(name) -> ndarray[float32]` | Values of one output, by index or name. |
| `summary` | `summary() -> dict[str, dict[str, float]]` | Summary statistics per output, ignoring NaN. |
| `to_dict` | `to_dict() -> dict[str, Any]` | Plain dictionary of a scene-level result. |
| `to_raster` | `to_raster() -> Raster` | Export dense values as a multi-band float32 raster. |

`SegmentationResult.class_mask(cls)` takes a class index or name. `EnhancementResult` and `SuperResolutionResult` carry the output bands as a `Raster` in their attribute `raster`.

### 15.4 Members shared by the task APIs

| Member of `ZooTask` | Signature | Description |
| --- | --- | --- |
| `model_id` | property | Model id, known without loading for catalogue names. |
| `predictor` | property | Predictor of the model, opened on first use. |
| `outputs` | property | Class, target or band names of the model. |
| `prepare` | staticmethod `prepare(image) -> Prepared` | Convert an input to an array with its georeferencing. |

### 15.5 Further public modules

`unbihexium.ai.inference.Predictor` runs any model on arrays of any size: images are normalised with the statistics stored at training time, larger images are cut into overlapping tiles padded by reflection at the border, dense outputs are blended with weights that fall towards the tile edges, and detections pass class-aware non-maximum suppression.

| Member of `Predictor` | Signature | Description |
| --- | --- | --- |
| `model_id` | property | Model id of the loaded model. |
| `prepare` | `prepare(image) -> tuple[ndarray[float32], ndarray[bool_]]` | Validate and normalise an image; returns the input and the invalid mask. |
| `windows` | `windows(height, width) -> list[tuple[int, int]]` | Tile windows (top, left) for an image of the given size. |
| `dense` | `dense(image) -> ndarray[float32]` | Dense prediction: probabilities, values or bands on the output grid. |
| `detect` | `detect(image, threshold=0.3, max_detections=1000, iou_threshold=0.5)` | Object detection: boxes (N, 4) in pixels, scores (N,) and class ids (N,). |
| `scene` | `scene(image) -> ndarray[float32]` | Scene-level prediction: one value per output for the whole image. |

`unbihexium.ai.training` trains and evaluates models (PyTorch). `TrainConfig` has the fields `epochs` (50), `batch_size` (8), `learning_rate` (0.001), `weight_decay` (0.0001), `warmup_epochs` (1.0), `chip_size`, `samples_per_epoch`, `num_workers` (0), `device` (`auto`), `seed` (0), `amp` (False), `grad_clip` (10.0), `patience`, `regression_loss` (`l1`), `augment` (True), `photometric`, `output_dir` (`runs`), `detection_threshold` (0.3), `class_weights` and `verbose` (True).

| Name in `unbihexium.ai.training` | Kind | Signature | Description |
| --- | --- | --- | --- |
| `TrainConfig` | class | `TrainConfig(epochs=50, batch_size=8, learning_rate=0.001, weight_decay=0.0001, warmup_epochs=1.0, chip_size=None, samples_per_epoch=None, num_workers=0, device='auto', seed=0, amp=False, grad_clip=10.0, patience=None, regression_loss='l1', augment=True, photometric=None, output_dir='runs', detection_threshold=0.3, class_weights=None, verbose=True)` | Hyperparameters of a training run. |
| `TrainingResult` | class | `TrainingResult(history=..., best_epoch=0, best_metrics=..., best_checkpoint=None, last_checkpoint=None)` | Outcome of a training run. |
| `Trainer` | class | `Trainer(model, config=None)` | Optimisation loop with validation and checkpoints. |
| `ChipDataset` | class | `ChipDataset(source, config, chip_size, mode='random', normalization=None, augmenter=None, length=None, seed=0)` | PyTorch dataset of encoded training or validation chips. |
| `train` | function | `train(model, data=None, config=None, variant=None, synthetic=None, callback=None) -> TrainingResult` | Train a model on a dataset folder or on synthetic data. |
| `evaluate` | function | `evaluate(model, data, split='val', chip_size=None, batch_size=8, device='auto', threshold=0.3) -> dict[str, Any]` | Metrics of a model on a split of a dataset folder. |
| `resolve_device` | function | `resolve_device(device) -> torch.device` | Resolve "auto" to the best available device. |

`unbihexium.ai.data` reads dataset folders (`train/`, `val/`, `test/` with `images/` and `labels/`) and generates synthetic data; the layout is described in [docs/model_zoo/training.md](../model_zoo/training.md).

| Name in `unbihexium.ai.data` | Kind | Signature | Description |
| --- | --- | --- | --- |
| `FolderDataset` | class | `FolderDataset(root, split, config)` | Labelled dataset in the folder layout described above. |
| `SyntheticDataset` | class | `SyntheticDataset(config, length=64, size=64, seed=0, noise=0.02)` | Learnable toy data for a model configuration. |
| `DatasetError` | exception | subclass of `ValueError` | Raised for missing or malformed dataset files. |

`unbihexium.ai.evaluation` provides the streaming metrics of training and `unbihexium evaluate`.

| Name in `unbihexium.ai.evaluation` | Kind | Signature | Description |
| --- | --- | --- | --- |
| `DetectionAccumulator` | class | `DetectionAccumulator(num_classes, class_names=None)` | Accumulates detections and ground truth boxes to compute average precision. |
| `ConfusionMatrix` | class | `ConfusionMatrix(num_classes, class_names=None, ignore_index=255)` | Confusion matrix for semantic segmentation and change detection. |
| `RegressionAccumulator` | class | `RegressionAccumulator(names)` | Streaming regression errors per output, ignoring NaN references. |
| `ImageQuality` | class | `ImageQuality(data_range=1.0)` | Streaming PSNR and SSIM over a set of images. |
| `TaskEvaluator` | class | `TaskEvaluator(config, threshold=0.3, data_range=1.0)` | Evaluates raw network outputs against batch targets for any task. |

`unbihexium.ai.models` contains the network architectures and the model factory (exported names in its `__all__`); the most important entry points are:

| Name in `unbihexium.ai.models` | Kind | Signature | Description |
| --- | --- | --- | --- |
| `build_model` | function | `build_model(name, variant=None, channel_names=None, outputs=None, initialise=True) -> ZooModel` | Build a model zoo model by name. |
| `build_from_config` | function | `build_from_config(config, initialise=True) -> ZooModel` | Build a model from a configuration, with deterministic starter weights. |
| `ZooModel` | class | `ZooModel(network, config)` | A network together with the configuration that describes it. |
| `UNet` | class | `UNet(in_channels, out_channels, base_channels, depth, blocks_per_stage, value_range=None, residual=False)` | U-Net for dense per-pixel outputs. |
| `CenterNet` | class | `CenterNet(in_channels, num_classes, base_channels, depth, blocks_per_stage, head_channels)` | CenterNet object detector with output stride 4. |
| `SceneRegressor` | class | `SceneRegressor(in_channels, out_channels, base_channels, depth, blocks_per_stage, head_channels, value_range=None)` | Scene regressor: encoder, global average pooling and a two-layer head. |
| `SuperResolutionNet` | class | `SuperResolutionNet(in_channels, out_channels, channels, num_blocks, scale)` | Super-resolution network with sub-pixel up-sampling and a global skip. |
| `SpectralIndex` | class | `SpectralIndex(formula)` | Spectral index as a parameter-free network module. |
| `weights_digest` | function | `weights_digest(model_or_state) -> str` | Compute the SHA-256 digest of a model's weights. |
| `seed_for` | function | `seed_for(model_id) -> int` | Derive the 32-bit seed of a model id. |
| `initialize` | function | `initialize(model, seed) -> nn.Module` | Initialise every layer of a model deterministically from a seed. |

### 15.6 Example

```python
import numpy as np

from unbihexium.ai import WaterDetector, predict, task_api, write_result
from unbihexium.ai.inference import Predictor

# One call: open the model, run the matching task API and return a result object.
result = predict("water_surface_detector_tiny", "scene.tif")
print(type(result).__name__, result.mask.shape, result.crs)
print(write_result(result, "water.tif"))

# The task API directly, with options.
api = task_api("water_surface_detector_tiny", tile_size=64, overlap=0.5)
print(type(api).__name__, api.model_id, api.outputs)

# Low level: tiled prediction on a plain (bands, rows, cols) array.
image = np.random.default_rng(0).uniform(0, 0.3, size=(4, 100, 100)).astype("float32")
probabilities = Predictor("water_surface_detector_tiny", tile_size=64).dense(image)
print(probabilities.shape, probabilities.dtype)
```

Output:

```text
SegmentationResult (128, 128) EPSG:32635
water.tif
SemanticSegmenter water_surface_detector_tiny ['background', 'water']
(2, 100, 100) float32
```

A complete training example is given in [docs/getting_started/quickstart.md](../getting_started/quickstart.md#8-training-briefly-and-predicting-again).

## 16. unbihexium.zoo

### 16.1 Overview

The model zoo offers 130 model families in the variants `tiny`, `base`, `large` and `mega` (520 models), defined in `src/unbihexium/zoo/catalog.yaml` (catalogue version 2.0.0). Listing and describing models works without PyTorch; building, training and exporting require the `torch` extra. Starter weights are generated locally and deterministically from the model id and verified against the published SHA-256 digests in `src/unbihexium/zoo/digests.json`; no weights are downloaded for catalogue models. Only the 28 models of the 7 spectral index families compute meaningful output without training. Built models are kept in `$UNBIHEXIUM_CACHE/models/<model_id>/` (see [configuration.md](../getting_started/configuration.md#5-model-store)).

### 16.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `DETECTION_STRIDE` | constant | `4` | Output stride of the detectors. |
| `BuildConfig` | class | `BuildConfig(model_id, family, variant, task, channel_names, outputs, units=(), value_range=None, scale=1, formula=None, tile_size=256, customised=False, extra=...)` | Effective configuration of a built model. |
| `CatalogError` | exception | subclass of `ValueError` | Raised when catalog.yaml or a lookup is invalid. |
| `ModelSpec` | class | `ModelSpec(family, name, task, domain, description, bands, dates, outputs, units=(), value_range=None, scale=1, formula=None, labels='', sources=...)` | Specification of one model family, as read from catalog.yaml. |
| `ModelZooEntry` | class | `ModelZooEntry(model_id, spec, variant, weights_digest='', num_parameters=0, source='build', download_url=None, local_path=None, version='2.0.0', license='MPL-2.0', tags=...)` | Description of one model of the zoo. |
| `Task` | enum | `detection`, `segmentation`, `change_detection`, `dense_regression`, `scene_regression`, `enhancement`, `super_resolution`, `spectral_index` | Tasks supported by the model zoo. The value is the name used in YAML files. |
| `Variant` | enum | `tiny`, `base`, `large`, `mega` | The four size variants of every model family. |
| `VariantSpec` | class | `VariantSpec(variant, base_channels, depth, blocks_per_stage, tile_size, head_channels)` | Architecture hyperparameters of a size variant. |
| `VerificationError` | exception | subclass of `RuntimeError` | Raised when a file or model does not match its expected checksum. |
| `all_model_ids` | function | `all_model_ids() -> list[str]` | Return every model id of the zoo: families times variants. |
| `catalog_version` | function | `catalog_version() -> str` | Version of the catalogue format and content. |
| `clear_cache` | function | `clear_cache(model_id=None, cache_dir=None) -> int` | Remove one model or the whole cache; returns the number of removed models. |
| `compute_sha256` | function | `compute_sha256(path) -> str` | Compute the SHA-256 of a file. |
| `download_model` | function | `download_model(model_id, cache_dir=None, force=False) -> Path` | Backwards-compatible name: obtain a model and return its checkpoint path. |
| `ensure_model` | function | `ensure_model(model_id, cache_dir=None, onnx=False, force=False) -> Path` | Make sure a model exists in the cache and return its directory. |
| `get_cache_dir` | function | `get_cache_dir() -> Path` | Root directory of the model cache. |
| `get_cached_model_path` | function | `get_cached_model_path(model_id, cache_dir=None) -> Path \| None` | Return the checkpoint path of a cached model, or None. |
| `get_model` | function | `get_model(model_id) -> ModelZooEntry \| None` | Return the entry of a model id, or None when it is unknown. |
| `get_spec` | function | `get_spec(name) -> ModelSpec` | Return the specification of a model family or model id. |
| `get_variant` | function | `get_variant(variant) -> VariantSpec` | Return the hyperparameters of a size variant. |
| `is_model_cached` | function | `is_model_cached(model_id, cache_dir=None) -> bool` | Whether a model's checkpoint exists in the cache. |
| `list_cached` | function | `list_cached(cache_dir=None) -> list[str]` | List the model ids present in the cache. |
| `list_models` | function | `list_models(task=None, domain=None, variant=None) -> list[ModelZooEntry]` | List entries, optionally filtered by task, domain and variant. |
| `list_specs` | function | `list_specs(task=None, domain=None) -> list[ModelSpec]` | Return the specifications of all model families, optionally filtered. |
| `load_model` | function | `load_model(name, variant=None, verify=True) -> ZooModel` | Load a model into memory. |
| `model_dir` | function | `model_dir(model_id, cache_dir=None) -> Path` | Directory of one model inside the cache. |
| `parse_model_id` | function | `parse_model_id(model_id) -> tuple[str, Variant]` | Split a model id such as "ship_detector_base" into family and variant. |
| `read_sha256_file` | function | `read_sha256_file(path) -> dict[str, str]` | Read a sha256sum-compatible checksum file into {name: digest}. |
| `register_model` | function | `register_model(entry)` | Register a user model, for example a fine-tuned checkpoint. |
| `unregister_model` | function | `unregister_model(model_id) -> bool` | Remove a user model from the registry. |
| `verify_directory` | function | `verify_directory(directory, filename='model.sha256') -> dict[str, bool]` | Verify every file listed in a model.sha256 file. |
| `verify_file` | function | `verify_file(path, expected) -> bool` | Verify a file against an expected SHA-256 digest. |
| `verify_model` | function | `verify_model(model_id, cache_dir=None) -> bool` | Verify a cached model: file checksums and the published weights digest. |
| `write_sha256_file` | function | `write_sha256_file(directory, names, filename='model.sha256') -> Path` | Write a sha256sum-compatible checksum file for files in one directory. |

### 16.3 Members of ModelZooEntry and ModelSpec

| Member of `ModelZooEntry` | Signature | Description |
| --- | --- | --- |
| `family` | property | Model family. |
| `task` | property | Task of the model. |
| `domain` | property | Capability domain of the model. |
| `name` | property | Human-readable name including the variant. |
| `requires_training` | property | Whether the model needs training before its predictions are meaningful. |
| `to_dict` | `to_dict() -> dict[str, Any]` | Serialise to plain Python types. |

| Member of `ModelSpec` | Signature | Description |
| --- | --- | --- |
| `in_channels` | property | Total number of input channels: bands times acquisitions. |
| `channel_names` | property | Names of all input channels in the order the model expects them. |
| `out_channels` | property | Number of output channels of the network. |
| `sigmoid_output` | property | True when a regression output is bounded to [0, 1] and uses a sigmoid. |
| `model_id` | `model_id(variant) -> str` | Build the model identifier of a variant of this family. |
| `to_dict` | `to_dict() -> dict[str, Any]` | Serialise the specification to plain Python types. |

### 16.4 Further public modules

`unbihexium.zoo.export` exports a model to ONNX [35] with dynamic batch size, height and width, stores the model configuration in the ONNX metadata and compares ONNX Runtime with PyTorch outputs. `unbihexium.zoo.checkpoint` reads and writes checkpoints that contain plain data only and are loaded with `torch.load(weights_only=True)`, so that loading a checkpoint cannot execute code.

| Name in `unbihexium.zoo.export` | Kind | Signature | Description |
| --- | --- | --- | --- |
| `export_onnx` | function | `export_onnx(model, path, verify=True, tolerance=0.001) -> Path` | Export a model to ONNX and return the path. |
| `verify_onnx` | function | `verify_onnx(model, path, tolerance=0.001) -> float` | Compare ONNX Runtime and PyTorch outputs on a random input. |
| `read_onnx_config` | function | `read_onnx_config(path) -> dict[str, object]` | Read the model configuration stored in an ONNX file. |
| `ExportError` | exception | subclass of `RuntimeError` | Raised when an exported model does not reproduce the PyTorch outputs. |

| Name in `unbihexium.zoo.checkpoint` | Kind | Signature | Description |
| --- | --- | --- | --- |
| `save_checkpoint` | function | `save_checkpoint(model, path, training=None) -> str` | Save a model to a checkpoint file and return the weights digest. |
| `load_checkpoint` | function | `load_checkpoint(path, verify=True) -> ZooModel` | Load a model from a checkpoint file. |
| `read_checkpoint` | function | `read_checkpoint(path) -> dict[str, Any]` | Read a checkpoint file and validate its structure. |
| `CheckpointError` | exception | subclass of `ValueError` | Raised when a file is not a valid Unbihexium checkpoint. |

`python -m unbihexium.zoo.sync --root . [--check] [--skip-digests]` regenerates or checks the derived files of the model zoo (digests, inventory, manifests, model cards, checksums); it is a maintainer tool used by CI.

### 16.5 Example

```python
from unbihexium.zoo import (
    ensure_model, get_model, list_models, list_specs, load_model, parse_model_id, verify_model,
)
from unbihexium.zoo.export import export_onnx, read_onnx_config

print(len(list_specs()), len(list_models()), len(list_models(task="detection", variant="tiny")))
entry = get_model("ship_detector_tiny")
print(entry.model_id, entry.task.value, entry.spec.bands, entry.num_parameters, entry.requires_training)
print(parse_model_id("ship_detector_large"))

model = load_model("ship_detector_tiny")               # in memory, digest verified
print(model.digest() == entry.weights_digest)

directory = ensure_model("ship_detector_tiny")         # $UNBIHEXIUM_CACHE/models/ship_detector_tiny
print(sorted(p.name for p in directory.iterdir()), verify_model("ship_detector_tiny"))

path = export_onnx(model, "ship_detector_tiny.onnx")   # verified against PyTorch
print(path, read_onnx_config(path)["model_id"])
```

Output:

```text
130 520 19
ship_detector_tiny detection ('red', 'green', 'blue') 730581 True
('ship_detector', <Variant.LARGE: 'large'>)
True
['config.json', 'model.pt', 'model.sha256'] True
ship_detector_tiny.onnx ship_detector_tiny
```

## 17. unbihexium.registry

### 17.1 Overview

Class-level registries: `CapabilityRegistry` lists what the library can do (library algorithms and one capability per model zoo family, 147 in total), `ModelRegistry` is a flat view of the model zoo with input validation, and `PipelineRegistry` holds the pipeline factories used by `unbihexium pipeline`. The task APIs register their pipelines when `unbihexium.ai` is imported.

### 17.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `Capability` | class | `Capability(capability_id, name, domain, description='', maturity=CapabilityMaturity.STABLE, entry_points=..., pipeline_id=None, cli_command=None, example_path=None, test_path=None, docs_path=None, model_family=None, task=None, bands=..., tags=...)` | One capability of the library. |
| `CapabilityDomain` | enum | `ai`, `tourism`, `analysis`, `indices`, `water`, `environment`, `forestry`, `imaging`, `assets`, `energy`, ... (16 members) | Capability domains; the model catalogue uses the same names. |
| `CapabilityMaturity` | enum | `stable`, `beta`, `research`, `deprecated` | Maturity of a capability. |
| `CapabilityRegistry` | class | `CapabilityRegistry()` | Class-level registry of capabilities. |
| `ModelEntry` | class | `ModelEntry(model_id, name='', task='', family='', domain='', variant='', channels=..., outputs=..., units=..., sha256='', download_url=None, num_parameters=0, license='MPL-2.0', source='build', requires_training=True, tile_size=256, tags=...)` | Flat description of one model. |
| `ModelRegistry` | class | `ModelRegistry()` | Registry of the models of the zoo and of user descriptions. |
| `PipelineEntry` | class | `PipelineEntry(pipeline_id, name, description='', config_class=None, factory=None, domains=..., tags=...)` | Registry entry of one pipeline. |
| `PipelineRegistry` | class | `PipelineRegistry()` | Class-level registry of pipeline factories. |
| `get_capability` | function | `get_capability(capability_id) -> Capability \| None` | Capability of an id, None when unknown. |
| `list_capabilities` | function | `list_capabilities(domain=None) -> list[Capability]` | Every capability, optionally of one domain. |
| `register_capability` | function | `register_capability(capability, replace=False) -> Capability` | Register a capability. |

### 17.3 Example

```python
from unbihexium.registry import CapabilityRegistry, ModelRegistry, PipelineRegistry, get_capability, list_capabilities
import unbihexium.ai  # task APIs register their pipelines on import

print(len(CapabilityRegistry.ids()), len(list_capabilities("sar")))
first = list_capabilities("sar")[0]
print(first.capability_id, first.domain.value, first.maturity.value)
print(PipelineRegistry.ids())
print(len(ModelRegistry.list_all(task="detection", variant="tiny")))
```

Output:

```text
147 9
ground_displacement sar beta
['building_detection', 'change_detection', 'ship_detection', 'super_resolution', 'water_detection']
19
```

## 18. unbihexium.serving

### 18.1 Overview

`create_app()` builds the FastAPI REST service (extra `serving`) with the routes `GET /health`, `GET /capabilities`, `GET /capabilities/{capability_id}`, `GET /models`, `GET /models/{model_id}`, `GET /pipelines`, `POST /predict/{model_id}` and the earlier routes `POST /infer/{model_id}`, `/detect/{model_id}` and `/segment/{model_id}`. The command `unbihexium serve` starts the service with the configured settings (see [cli.md](cli.md#12-unbihexium-serve)), and the module `unbihexium.serving.app` also provides the instance `app` for `uvicorn unbihexium.serving.app:app`. Settings (request limits, API key, rate limit, CORS) come from `unbihexium.config`; see [configuration.md](../getting_started/configuration.md#8-rest-service-settings). Images are sent as nested JSON lists `(bands, rows, cols)` or as a base64-encoded NumPy `.npy` file; the OpenAPI document is served at `/openapi.json` and `/docs`.

### 18.2 Exported names

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `CapabilitiesResponse` | class | `CapabilitiesResponse(*, count, capabilities)` | List of capabilities. |
| `DetectionRequest` | class | `DetectionRequest(*, image_data=None, threshold=0.5)` | Detection request of earlier releases. |
| `DetectionResponse` | class | `DetectionResponse(*, model_id, count, detections)` | Detection response. |
| `HealthResponse` | class | `HealthResponse(*, status, version, ready, models_available=0, models_loaded=0)` | Health check. |
| `InferenceRequest` | class | `InferenceRequest(*, data=None, parameters=...)` | Generic inference request of earlier releases. |
| `InferenceResponse` | class | `InferenceResponse(*, model_id, success, result, error=None)` | Generic inference response of earlier releases. |
| `ModelInferenceService` | class | `ModelInferenceService(max_pixels=4194304, max_values=16777216, cache_size=4, device='cpu', backend='auto', batch_size=4)` | Inference service for the REST API. |
| `ModelInfo` | class | `ModelInfo(*, model_id, task, description, name='', domain='', variant='', in_channels=0, channels=..., outputs=..., units=..., requires_training=True)` | One model. |
| `ModelsResponse` | class | `ModelsResponse(*, count, total=0, offset=0, models)` | List of models. |
| `PayloadTooLargeError` | exception | subclass of `ValueError` | Input larger than the limits of the service. |
| `PredictRequest` | class | `PredictRequest(*, image=None, image_npy_base64=None, crs=None, transform=None, nodata=None, parameters=...)` | Prediction request. |
| `PredictResponse` | class | `PredictResponse(*, model_id, task, success=True, input_shape, elapsed_ms, requires_training, result)` | Prediction response. |
| `UnknownModelError` | exception | subclass of `KeyError` | Unknown model id. |
| `create_app` | function | `create_app(title='Unbihexium API', version=None, enable_cors=True, config=None, service=None) -> FastAPI` | Build the FastAPI application. |

### 18.3 Example

```python
from fastapi.testclient import TestClient

from unbihexium.config import ServingConfig
from unbihexium.serving import create_app

app = create_app(config=ServingConfig(max_pixels=512 * 512))
client = TestClient(app)
print(client.get("/health").json()["status"])

body = {"image": [[[0.05, 0.06], [0.04, 0.05]], [[0.40, 0.45], [0.30, 0.35]]]}   # bands red, nir
response = client.post("/predict/ndvi_calculator_tiny", json=body)
data = response.json()
print(response.status_code, data["task"], data["input_shape"], data["requires_training"])
print(sorted(data["result"]))
print(client.post("/predict/no_such_model", json=body).status_code)
```

Output:

```text
healthy
200 spectral_index [2, 2, 2] False
['outputs', 'shape', 'statistics', 'units']
404
```

## 19. unbihexium.config

`unbihexium.config` holds validated settings in the sections `model`, `processing` and `serving` plus `log_level`, loaded in layers from defaults, a YAML file, `UNBIHEXIUM_<SECTION>__<KEY>` environment variables and explicit overrides. Within the package only the REST service reads them. Every key, its default and its consumer are listed in [configuration.md](../getting_started/configuration.md#4-layered-settings-in-unbihexiumconfig).

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `CONFIG_ENV` | constant | `'UNBIHEXIUM_CONFIG'` | Variable with the path of a YAML file. |
| `ENV_PREFIX` | constant | `'UNBIHEXIUM_'` | Prefix of the environment variables. |
| `Config` | class | `Config(model=..., processing=..., serving=..., log_level='WARNING')` | Complete configuration of the library. |
| `ModelConfig` | class | `ModelConfig(variant='base', device='cpu', backend='auto', batch_size=8, num_workers=4)` | Model selection and execution. |
| `ProcessingConfig` | class | `ProcessingConfig(tile_size=512, overlap=64, output_format='GTiff', compression='DEFLATE', nodata=None, seed=None)` | Raster processing. |
| `ServingConfig` | class | `ServingConfig(host='127.0.0.1', port=8000, max_request_bytes=10485760, max_pixels=4194304, max_values=16777216, api_key=None, cors_origins=..., rate_limit_per_minute=0, model_cache_size=4)` | REST service. |
| `get_default_config` | function | `get_default_config() -> Config` | Default configuration. |
| `get_settings` | function | `get_settings() -> Config` | Cached process-wide settings. |
| `load_config` | function | `load_config(path=None, env=True, overrides=None, environ=None) -> Config` | Layered configuration: defaults, YAML file, environment, overrides. |
| `reset_settings` | function | `reset_settings()` | Forget the cached settings, for example after changing the environment. |

```python
from unbihexium.config import load_config

config = load_config(env=False, overrides={"model": {"device": "cpu", "batch_size": 2}})
print(config.model.batch_size, config.serving.port, config.log_level)
```

Output:

```text
2 8000 WARNING
```

## 20. unbihexium.utils

Small helpers shared by the library: SHA-256 digests of files, bytes, arrays and canonical JSON; the `unbihexium` logger hierarchy and `configure_logging`; timing; seeding of Python, NumPy and PyTorch; tile windows and mosaicking; atomic file writes. `count_parameters` counts the parameters of a PyTorch module without importing PyTorch itself.

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `Timer` | class | `Timer(name='block', logger=None, level=20, clock=perf_counter)` | Elapsed-time measurement with laps. |
| `array_digest` | function | `array_digest(array) -> str` | SHA-256 hex digest of an array, covering dtype, shape and values. |
| `atomic_write_bytes` | function | `atomic_write_bytes(path, data) -> Path` | Write bytes atomically. |
| `atomic_write_text` | function | `atomic_write_text(path, text, encoding='utf-8') -> Path` | Write text atomically. |
| `bytes_to_human` | function | `bytes_to_human(size, precision=1) -> str` | Byte count as text with binary prefixes, for example 1536 -> "1.5 KiB". |
| `canonical_json` | function | `canonical_json(document) -> str` | Canonical JSON text of a document: sorted keys, compact separators. |
| `compute_sha256` | function | `compute_sha256(path, chunk_size=1048576) -> str` | SHA-256 hex digest of a file. |
| `configure_logging` | function | `configure_logging(level=None, stream=None, fmt=...) -> logging.Logger` | Install one stream handler on the library logger and set its level. |
| `count_parameters` | function | `count_parameters(model, trainable_only=True) -> int` | Number of parameters of a PyTorch module. |
| `derive_seed` | function | `derive_seed(seed, *keys) -> int` | Child seed derived from a base seed and keys, independent of call order. |
| `ensure_dir` | function | `ensure_dir(path) -> Path` | Create a directory with its parents and return it. |
| `get_logger` | function | `get_logger(name=None) -> logging.Logger` | Logger in the library hierarchy. |
| `json_digest` | function | `json_digest(document) -> str` | SHA-256 hex digest of a JSON document in canonical form. |
| `merge_tiles` | function | `merge_tiles(tiles, windows, shape, weights=None) -> ndarray[float32]` | Mosaic of tiles; overlapping pixels are the (weighted) mean of the tiles. |
| `parse_level` | function | `parse_level(level) -> int` | Convert a level name ("info") or number (20) to a logging level. |
| `set_seed` | function | `set_seed(seed, torch=None, deterministic=False) -> random.Generator` | Seed Python, NumPy and optionally PyTorch; return a NumPy generator. |
| `sha256_bytes` | function | `sha256_bytes(data) -> str` | SHA-256 hex digest of a byte string. |
| `spawn_generators` | function | `spawn_generators(seed, count) -> list[random.Generator]` | Independent generators for parallel workers. |
| `tile_image` | function | `tile_image(image, tile_size=512, overlap=64) -> Iterator[tuple[ndarray, int, int]]` | Generator of (tile, row, col) for (bands, H, W) or (H, W) images. |
| `tile_starts` | function | `tile_starts(size, tile, overlap=0) -> list[int]` | Start offsets of tiles of length `tile` along an axis of length `size`. |
| `tile_windows` | function | `tile_windows(height, width, tile_size, overlap=0) -> list[Window]` | Windows (row, col, height, width) of the tiles of an image. |
| `timed` | function | `timed(logger=None, level=10)` | Decorator that logs the duration of every call. |

```python
import numpy as np

from unbihexium.utils import Timer, array_digest, compute_sha256, derive_seed, set_seed, tile_windows

rng = set_seed(42)                               # Python, NumPy (and PyTorch when installed)
print(rng.integers(0, 100, size=3), derive_seed(42, "tile", 3) == derive_seed(42, "tile", 3))
print(array_digest(np.zeros((2, 2), dtype="float32"))[:16])
print(tile_windows(100, 100, tile_size=64, overlap=16))
with Timer("sum") as timer:
    total = float(np.ones(1_000_000).sum())
print(total, timer.elapsed >= 0)
with open("hello.txt", "w") as handle:
    handle.write("hello\n")
print(compute_sha256("hello.txt"))
```

Output:

```text
[ 8 77 65] True
deccdf73aab4e5c4
[(0, 0, 64, 64), (0, 36, 64, 64), (36, 0, 64, 64), (36, 36, 64, 64)]
1000000.0 True
5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03
```

## 21. unbihexium.cli

`unbihexium.cli` exports the Click group of the `unbihexium` command as `main` (the console script entry point) and under its earlier name `cli`. The commands are documented in [cli.md](cli.md).

| Name | Kind | Signature or value | Description |
| --- | --- | --- | --- |
| `cli` | Click group | `cli(args=None, prog_name=None, complete_var=None, standalone_mode=True, windows_expand_args=True, **extra)` | The same group as `main`, under the name used by earlier releases. |
| `main` | Click group | `main(args=None, prog_name=None, complete_var=None, standalone_mode=True, windows_expand_args=True, **extra)` | Root command group and console script entry point `unbihexium`. |

Calling `main()` parses `sys.argv` and exits the interpreter, as Click commands do. In tests, the group is invoked in-process with Click's `CliRunner`:

```python
from click.testing import CliRunner

from unbihexium.cli import main

result = CliRunner().invoke(main, ["zoo", "list", "--task", "spectral_index", "--variant", "tiny", "--json"])
print(result.exit_code, len(result.output) > 0)
```

Output:

```text
0 True
```

## 22. Stability of the interface

The public interface defined in Section 1.1 follows the versioning and deprecation policy of [VERSIONING.md](../../VERSIONING.md): incompatible changes to exported names, signatures or documented behaviour require a new major version, and a deprecated name SHOULD emit a `DeprecationWarning`, SHOULD remain available for at least two minor releases and MUST NOT be removed before the next major release. Aliases kept for compatibility with earlier releases, for example `calculate_ergas`, `calculate_qindex`, `calculate_sam`, `create_legend`, `read_cog` and `download_model`, are part of the interface until they are deprecated. Changes are recorded in [CHANGELOG.md](../../CHANGELOG.md).

## References

[1] S. Bradner. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF, 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] B. Leiba. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF, 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] J. W. Rouse, R. H. Haas, J. A. Schell and D. W. Deering. Monitoring vegetation systems in the Great Plains with ERTS. Third Earth Resources Technology Satellite-1 Symposium, NASA SP-351, 309-317. 1974. <https://ntrs.nasa.gov/citations/19740022614>

[4] S. K. McFeeters. The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. International Journal of Remote Sensing 17(7), 1425-1432. 1996. <https://doi.org/10.1080/01431169608948714>

[5] A. Huete, K. Didan, T. Miura, E. P. Rodriguez, X. Gao and L. G. Ferreira. Overview of the radiometric and biophysical performance of the MODIS vegetation indices. Remote Sensing of Environment 83(1-2), 195-213. 2002. <https://doi.org/10.1016/S0034-4257(02)00096-2>

[6] A. R. Huete. A soil-adjusted vegetation index (SAVI). Remote Sensing of Environment 25(3), 295-309. 1988. <https://doi.org/10.1016/0034-4257(88)90106-X>

[7] Open Geospatial Consortium. OGC Cloud Optimized GeoTIFF Standard, OGC 21-026. 2023. <https://docs.ogc.org/is/21-026/21-026.html>

[8] Zarr Developers. Zarr specifications. 2026. <https://github.com/zarr-developers/zarr-specs>

[9] H. Butler, M. Daly, A. Doyle, S. Gillies, S. Hagen and T. Schaub. RFC 7946: The GeoJSON Format. IETF, 2016. <https://www.rfc-editor.org/rfc/rfc7946>

[10] Open Geospatial Consortium. GeoParquet specification. 2026. <https://github.com/opengeospatial/geoparquet>

[11] Radiant Earth Foundation. SpatioTemporal Asset Catalog (STAC) specification. 2026. <https://github.com/radiantearth/stac-spec>

[12] J.-S. Lee. Digital image enhancement and noise filtering by use of local statistics. IEEE Transactions on Pattern Analysis and Machine Intelligence PAMI-2(2), 165-168. 1980. <https://doi.org/10.1109/TPAMI.1980.4766994>

[13] A. Lopes, R. Touzi and E. Nezry. Adaptive speckle filters and scene heterogeneity. IEEE Transactions on Geoscience and Remote Sensing 28(6), 992-1000. 1990. <https://doi.org/10.1109/36.62623>

[14] V. S. Frost, J. A. Stiles, K. S. Shanmugan and J. C. Holtzman. A model for radar images and its application to adaptive digital filtering of multiplicative noise. IEEE Transactions on Pattern Analysis and Machine Intelligence PAMI-4(2), 157-166. 1982. <https://doi.org/10.1109/TPAMI.1982.4767223>

[15] R. M. Goldstein and C. L. Werner. Radar interferogram filtering for geophysical applications. Geophysical Research Letters 25(21), 4035-4038. 1998. <https://doi.org/10.1029/1998GL900033>

[16] A. Freeman and S. L. Durden. A three-component scattering model for polarimetric SAR data. IEEE Transactions on Geoscience and Remote Sensing 36(3), 963-973. 1998. <https://doi.org/10.1109/36.673687>

[17] S. R. Cloude and E. Pottier. An entropy based classification scheme for land applications of polarimetric SAR. IEEE Transactions on Geoscience and Remote Sensing 35(1), 68-78. 1997. <https://doi.org/10.1109/36.551935>

[18] B. K. P. Horn. Hill shading and the reflectance map. Proceedings of the IEEE 69(1), 14-47. 1981. <https://doi.org/10.1109/PROC.1981.11918>

[19] L. W. Zevenbergen and C. R. Thorne. Quantitative analysis of land surface topography. Earth Surface Processes and Landforms 12(1), 47-56. 1987. <https://doi.org/10.1002/esp.3290120107>

[20] K. J. Beven and M. J. Kirkby. A physically based, variable contributing area model of basin hydrology. Hydrological Sciences Bulletin 24(1), 43-69. 1979. <https://doi.org/10.1080/02626667909491834>

[21] G. Matheron. Principles of geostatistics. Economic Geology 58(8), 1246-1266. 1963. <https://doi.org/10.2113/gsecongeo.58.8.1246>

[22] N. Cressie. Statistics for Spatial Data, revised edition. Wiley. 1993. <https://doi.org/10.1002/9781119115151>

[23] D. Shepard. A two-dimensional interpolation function for irregularly-spaced data. Proceedings of the 23rd ACM National Conference, 517-524. 1968. <https://doi.org/10.1145/800186.810616>

[24] P. A. P. Moran. Notes on continuous stochastic phenomena. Biometrika 37(1/2), 17-23. 1950. <https://doi.org/10.2307/2332142>

[25] R. C. Geary. The contiguity ratio and statistical mapping. The Incorporated Statistician 5(3), 115-145. 1954. <https://doi.org/10.2307/2986645>

[26] L. Anselin. Local indicators of spatial association: LISA. Geographical Analysis 27(2), 93-115. 1995. <https://doi.org/10.1111/j.1538-4632.1995.tb00338.x>

[27] J. K. Ord and A. Getis. Local spatial autocorrelation statistics: distributional issues and an application. Geographical Analysis 27(4), 286-306. 1995. <https://doi.org/10.1111/j.1538-4632.1995.tb00912.x>

[28] T. L. Saaty. A scaling method for priorities in hierarchical structures. Journal of Mathematical Psychology 15(3), 234-281. 1977. <https://doi.org/10.1016/0022-2496(77)90033-5>

[29] J. Cohen. A coefficient of agreement for nominal scales. Educational and Psychological Measurement 20(1), 37-46. 1960. <https://doi.org/10.1177/001316446002000104>

[30] P. Olofsson, G. M. Foody, M. Herold, S. V. Stehman, C. E. Woodcock and M. A. Wulder. Good practices for estimating area and assessing accuracy of land change. Remote Sensing of Environment 148, 42-57. 2014. <https://doi.org/10.1016/j.rse.2014.02.015>

[31] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600-612. 2004. <https://doi.org/10.1109/TIP.2003.819861>

[32] X. Zhou, D. Wang and P. Kraehenbuehl. Objects as points. arXiv:1904.07850. 2019. <https://arxiv.org/abs/1904.07850>

[33] O. Ronneberger, P. Fischer and T. Brox. U-Net: convolutional networks for biomedical image segmentation. MICCAI 2015, LNCS 9351, 234-241. 2015. <https://arxiv.org/abs/1505.04597>

[34] B. Lim, S. Son, H. Kim, S. Nah and K. M. Lee. Enhanced deep residual networks for single image super-resolution. CVPR Workshops. 2017. <https://arxiv.org/abs/1707.02921>

[35] ONNX Project Contributors. Open Neural Network Exchange (ONNX). 2026. <https://onnx.ai/>

<!--
=============================================================================
End of file docs/reference/api.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
