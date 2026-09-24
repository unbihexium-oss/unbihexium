<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/capabilities/11_resolution_metadata_qa.md
Title       : Capability 11: Resolution, Metadata and Quality Assurance
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Capability 11: Resolution, Metadata and Quality Assurance

| Field | Value |
| --- | --- |
| Document | UBX-DOC-CAP-11-RESOLUTION-QA |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch |

## Abstract

This document describes what Unbihexium implements to read and record the metadata of geospatial data, to handle the pixel size (ground sample distance) of rasters, and to check the quality of inputs and products: GeoTIFF and Cloud Optimized GeoTIFF metadata, affine transforms and windows, band alignment and resampling, tiling and mosaicking, SpatioTemporal Asset Catalog (STAC) metadata and product records, validity and integrity checks, and the accuracy and image quality metrics of `unbihexium.metrics`. It also lists the five model families intended for tiling, mosaics, relief displacement, map digitisation and super-resolution. It is written for users who prepare and validate data, for operators who need reproducible and verifiable outputs, and for reviewers who need to know which formulas are implemented and where they come from. Every example was executed against the current code. The model families are untrained starter models, and several quality measures described in earlier versions of this document (MTF, edge response, signal-to-noise estimation, positional accuracy standards) have no implementation and have been removed.

## Contents

1. [Scope and status](#1-scope-and-status)
2. [Components of the capability](#2-components-of-the-capability)
3. [Raster metadata](#3-raster-metadata)
4. [Pixel size and resampling](#4-pixel-size-and-resampling)
5. [Tiling and mosaicking](#5-tiling-and-mosaicking)
6. [Catalogue and product metadata](#6-catalogue-and-product-metadata)
7. [Validity and integrity checks](#7-validity-and-integrity-checks)
8. [Accuracy and image quality metrics](#8-accuracy-and-image-quality-metrics)
9. [Model families](#9-model-families)
10. [Command line](#10-command-line)
11. [Limitations](#11-limitations)
12. [Related documents](#12-related-documents)
13. [References](#references)

## 1. Scope and status

### 1.1 What the capability covers

The capability registry (`unbihexium.registry.CapabilityRegistry`) has no single domain for this topic; the document collects components of several registered library capabilities, all with maturity `stable`:

- `io_geotiff`, `io_stac` and the other input and output capabilities of domain `io` (package `unbihexium.io`);
- `image_preprocessing` (domain `imaging`), of which the resampling functions of `unbihexium.preprocessing.resample` are described here;
- `accuracy_metrics` (domain `analysis`), implemented by `unbihexium.metrics` and `unbihexium.ai.evaluation`;
- the core records `Raster`, `Scene`, `TileGrid`, `Product` and `Evidence` of `unbihexium.core`.

It also lists five model families of the catalogue domain `imaging`: `raster_tiler`, `mosaic_processor`, `ortho_processor`, `digitization_2d` and `super_resolution`. The last one is also covered by [01_ai_products.md](01_ai_products.md), because it has a registered processing pipeline (`super_resolution`).

### 1.2 Status

The functions and records are deterministic and covered by the unit tests in `tests/`. The 20 models of the five families (four size variants each) are untrained starter models: each has a complete, trainable network with deterministically initialised weights whose SHA-256 digest is published in `src/unbihexium/zoo/digests.json`, but none has been trained on Earth observation data, so their outputs carry no information until the model is trained (see [docs/model_zoo/training.md](../model_zoo/training.md)). The domain describes intended applications, not validated products. Of the 520 models of the zoo, only the 28 models of the 7 spectral index families compute exact formulas without training, and none of them is listed here. No accuracy figures are published for any model; see section 2 of [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md).

### 1.3 Terms

- **Pixel size** or ground sample distance (GSD): the size of one pixel in the units of the coordinate reference system (CRS), given by the affine transform. `Raster.resolution` returns it as `(x, y)`, both positive.
- **Affine transform**: the six coefficients $(a, b, c, d, e, f)$ with $x = a \cdot col + b \cdot row + c$ and $y = d \cdot col + e \cdot row + f$ for the pixel corner $(col, row)$, in the order of the `affine` package. The GDAL geotransform orders the same numbers as $(c, a, b, f, d, e)$. A north-up raster has $b = d = 0$ and $e < 0$.
- **Valid pixel**: a pixel that is finite and differs from the no-data value.

## 2. Components of the capability

| Component | Public names | Purpose |
| --- | --- | --- |
| GeoTIFF (`unbihexium.io.geotiff`) | `read_geotiff`, `write_geotiff`, `write_cog`, `geotiff_info`, `is_cog`, `build_overviews`, `read_raster`, `write_raster` | Read pixels and metadata (optionally a window, a box or an overview level), write tiled and compressed files and COGs |
| Raster (`unbihexium.core.raster`) | `Raster`, `RasterMetadata` | Georeferenced arrays: `bounds`, `resolution`, `xy`, `rowcol`, `sample`, `valid_mask`, `statistics`, `histogram`, `resample`, `reproject`, `match`, `same_grid` |
| Scene (`unbihexium.core.scene`) | `Scene`, `SceneMetadata` | Named bands of one acquisition; `is_aligned` and `harmonize` put bands of different pixel sizes on one grid |
| Resampling (`unbihexium.preprocessing.resample`) | `resample`, `aggregate`, `scaled_transform` | Array-level change of pixel size with missing values |
| Tiling (`unbihexium.core.tile`, `unbihexium.postprocessing`) | `TileGrid`, `Tile`, `TileIndex`, `xyz_tile`, `xyz_bounds`, `tile_positions`, `blend_weights`, `stitch_tiles` | Overlapping tiles, feathered mosaics and Web Mercator tile arithmetic |
| STAC (`unbihexium.io.stac`) | `STACItem`, `STACCollection`, `filter_items`, `read_stac_item`, `walk_catalog`, `STACClient`, `search_stac`, `load_from_stac` | Parse, filter and search catalogue metadata |
| Products (`unbihexium.core.product`) | `Product`, `ProductMetadata`, `ProductType` | Derived products with CRS, bounds, resolution, lineage, SHA-256 digest and a STAC item |
| Evidence (`unbihexium.core.evidence`) | `Evidence`, `ProvenanceRecord`, `sha256_file`, `sha256_array` | SHA-256 audit trail of inputs, outputs and models |
| GeoJSON checks (`unbihexium.io.geojson`) | `validate_geojson`, `geojson_problems`, `rewind` | Structural validation against RFC 7946 |
| Metrics (`unbihexium.metrics`) | `psnr`, `ssim`, `sam`, `spectral_angle`, `ergas`, `q_index`, `confusion_matrix`, `accuracy_assessment`, `stratified_area_estimate`, `sample_allocation`, `regression_report`, `change_detection_metrics` and others | Image quality and map accuracy |

## 3. Raster metadata

`geotiff_info(path)` returns the metadata of a GeoTIFF without reading pixels: driver, data type, band count, width and height (of the full resolution and of the level read), CRS, transform, bounds, no-data value, band descriptions, block shape, tiling, compression, overview factors and whether the file has the Cloud Optimized GeoTIFF layout. `read_geotiff(path, bands=None, window=None, bounds=None, overview_level=None)` returns the pixels as a `(bands, rows, cols)` array and the same metadata, with a transform that describes the window that was read. `write_geotiff` writes tiled files compressed with DEFLATE and the matching predictor (2 for integers, 3 for floating point) and, with `cog=True`, the COG layout with internal overviews [1], [2].

```python
import numpy as np
from rasterio.transform import from_origin

from unbihexium.io import geotiff_info, is_cog, read_geotiff, write_geotiff

rng = np.random.default_rng(0)
b04 = rng.uniform(0.02, 0.2, size=(1, 120, 120)).astype("float32")
b11 = rng.uniform(0.05, 0.3, size=(1, 60, 60)).astype("float32")
write_geotiff(b04, "B04.tif", crs="EPSG:32635", transform=from_origin(500000, 6700020, 10, 10), nodata=-9999.0, descriptions=["B04"])
write_geotiff(b11, "B11.tif", crs="EPSG:32635", transform=from_origin(500000, 6700020, 20, 20), cog=True)

info = geotiff_info("B04.tif")
print(sorted(info))
print(info["crs"], info["transform"], info["width"], info["height"], info["nodata"])
print(is_cog("B04.tif"), is_cog("B11.tif"))

# Rows 50 to 70 and columns 30 to 50: the returned transform describes the window.
window, meta = read_geotiff("B04.tif", window=(50, 70, 30, 50))
print(window.shape, meta["transform"])
```

Output:

```text
['band_count', 'block_shape', 'bounds', 'compression', 'count', 'crs', 'descriptions', 'driver', 'dtype', 'full_height', 'full_width', 'height', 'is_cog', 'nodata', 'overviews', 'tiled', 'transform', 'width']
EPSG:32635 (10.0, 0.0, 500000.0, 0.0, -10.0, 6700020.0) 120 120 -9999.0
False True
(1, 20, 20) (10.0, 0.0, 500300.0, 0.0, -10.0, 6699520.0)
```

The window starts 30 columns (300 m) east and 50 rows (500 m) south of the file origin, which the returned transform reflects. The same metadata are available on a `Raster` (`Raster.from_file`, `raster.crs`, `raster.transform`, `raster.nodata`, `raster.bounds`, `raster.resolution`).

## 4. Pixel size and resampling

### 4.1 Aligning bands of different pixel sizes

Sentinel-2 delivers bands at 10, 20 and 60 m (see [10_satellite_imagery_features.md](10_satellite_imagery_features.md)). `Scene.is_aligned()` tests whether all bands share one grid (`Raster.same_grid`: same shape, CRS and transform), and `Scene.harmonize(reference=None, method="bilinear")` warps every band onto the grid of a reference band, by default the band with the most pixels. Warping uses the GDAL warper through rasterio (`Raster.match`), with the methods nearest, bilinear, cubic, average, mode, min, max, median and others [3].

### 4.2 Array-level resampling

`resample(image, shape, method="bilinear", nodata=None)` interpolates `(H, W)` or `(C, H, W)` arrays to a new shape with nearest, bilinear or cubic splines. It uses pixel-area alignment, so the outer edges of the first and last pixels stay fixed as in GDAL. With missing values it applies normalised convolution [4]: the data with missing values set to zero and the validity weights are interpolated separately and divided, so NaN does not spread; output pixels with a weight below one half are missing. `aggregate(image, factor, method="mean")` reduces the size by an integer factor with `mean`, `sum`, `min`, `max`, `median` or `mode` (majority, for class maps), ignoring NaN. `scaled_transform(transform, old_shape, new_shape)` returns the affine transform of the resampled grid, which keeps the footprint of the image.

### 4.3 Example

```python
from unbihexium.core import Raster, Scene
from unbihexium.preprocessing import aggregate, resample, scaled_transform

red, swir = Raster.from_file("B04.tif"), Raster.from_file("B11.tif")
print(red.resolution, swir.resolution, red.bounds == swir.bounds)

# Warp every band onto the grid of the finest band (20 m onto 10 m).
scene = Scene(rasters={"B04": red, "B11": swir})
print(scene.is_aligned())
aligned = scene.harmonize(method="bilinear")
print(aligned.is_aligned(), aligned.to_array().shape, aligned["B11"].resolution)

# Array-level change of pixel size, with the matching geotransform.
coarse = aggregate(red.require_data()[0], 2, method="mean")
fine = resample(swir.require_data()[0], (120, 120), method="cubic")
print(coarse.shape, fine.shape)
print(scaled_transform(red.transform, (120, 120), coarse.shape))
```

Output:

```text
(10.0, 10.0) (20.0, 20.0) True
False
True (2, 120, 120) (10.0, 10.0)
(60, 60) (120, 120)
(20.0, 0.0, 500000.0, 0.0, -20.0, 6700020.0)
```

Resampling to a finer grid does not add spatial detail; it only changes the sampling. Increasing the information content of an image is the purpose of the `super_resolution` family (section 9), which must be trained first.

## 5. Tiling and mosaicking

`TileGrid.for_shape` and `TileGrid.from_raster` divide a raster into tiles of a fixed size that advance by `tile_size - overlap`; the last tile of each row and column is shifted back to end at the raster edge, so every tile has the full size. `TileGrid.mosaic(tiles, blend=...)` recombines tiles: `last` writes them in order, `average` weights all tiles equally, and `linear` weights each pixel by

$$
w(i) = \min\left(1,\ \frac{i + 0.5}{o},\ \frac{n - i - 0.5}{o}\right)
$$

along each axis (the product of both axes), where $o$ is the overlap and $n$ the tile size, so that seams in the overlaps are feathered [5], [6]. The result is exact wherever only one tile covers a pixel. The functions `tile_positions`, `blend_weights` and `stitch_tiles` of `unbihexium.postprocessing` do the same for predictions of the model zoo. `xyz_tile(lon, lat, zoom)` and `xyz_bounds(tile)` convert between geographic coordinates and the Web Mercator (XYZ) tile scheme [7].

```python
import numpy as np

from unbihexium.core import Raster, TileGrid
from unbihexium.core.tile import xyz_bounds, xyz_tile

red = Raster.from_file("B04.tif")
grid = TileGrid.from_raster(red, tile_size=64, overlap=16)
print(grid.num_rows, grid.num_cols, [t.offset for t in grid.tiles(red)])
restored = grid.mosaic(grid.tiles(red), blend="linear")
print(np.allclose(restored, red.require_data(), atol=1e-6))

tile = xyz_tile(24.94, 60.17, 12)  # Web Mercator tile of a point in Helsinki.
print(tile, [round(v, 4) for v in xyz_bounds(tile)])
```

Output:

```text
3 3 [(0, 0), (0, 48), (0, 56), (48, 0), (48, 48), (48, 56), (56, 0), (56, 48), (56, 56)]
True
z12/r1185/c2331 [24.873, 60.1524, 24.9609, 60.1962]
```

## 6. Catalogue and product metadata

### 6.1 STAC items

`STACItem.from_dict` parses a STAC 1.0.0 item [8], validates its required members and bounding box and resolves relative asset links. `filter_items(items, bbox=None, datetime_range=None, collections=None, ids=None, query=None, max_cloud_cover=None, limit=None)` selects items offline: boxes may cross the antimeridian (RFC 7946 section 5.2 [9]), time intervals follow the STAC API item search (an item matches when its own interval overlaps the requested one), and `query` supports the operators of the STAC API query extension (`eq`, `neq`, `lt`, `lte`, `gt`, `gte`, `in`, `startsWith`, `endsWith`, `contains`) on any property, for example the `gsd` of the common metadata [10]. `STACClient` and `search_stac` apply the same criteria to a STAC API.

```python
from unbihexium.io import STACItem, filter_items


def item(item_id, day, cloud, gsd):
    return STACItem.from_dict({
        "type": "Feature", "stac_version": "1.0.0", "id": item_id,
        "bbox": [24.0, 60.0, 25.0, 61.0],
        "geometry": {"type": "Polygon", "coordinates": [[[24, 60], [25, 60], [25, 61], [24, 61], [24, 60]]]},
        "properties": {"datetime": f"2026-06-{day:02d}T09:50:00Z", "eo:cloud_cover": cloud, "gsd": gsd},
        "assets": {"visual": {"href": f"{item_id}.tif", "roles": ["visual"]}},
        "links": [],
    })


items = [item("a", 3, 62.0, 10), item("b", 8, 4.5, 10), item("c", 13, 12.0, 30)]
selected = filter_items(items, bbox=[24.5, 60.2, 24.6, 60.3], datetime_range="2026-06-01/2026-06-30",
                        max_cloud_cover=20.0, query={"gsd": {"lte": 10}})
print([(i.id, i.cloud_cover, i.properties["gsd"]) for i in selected])
```

Output:

```text
[('b', 4.5, 10)]
```

### 6.2 Product records

`Product.create(product_id, product_type, data, **metadata)` wraps a derived raster, vector or array and fills its CRS, bounds and resolution from the data. `ProductMetadata` also records source scenes, the processing chain, an optional quality score in $[0, 1]$, a licence and free-form tags. `Product.save(directory)` writes the data (a COG for rasters, GeoJSON or GeoParquet for vectors, `.npy` for arrays) and `product.json` with the SHA-256 digest of the data file; `Product.load` verifies the digest; and `to_stac_item()` returns a STAC 1.0.0 item with the projection extension whose geometry and bounding box are transformed to WGS 84 longitude and latitude.

```python
import json

from unbihexium.core import Product, Raster

# Package a derived raster as a product with metadata, digest and a STAC item.
red = Raster.from_file("B04.tif")
product = Product.create("red_b04_demo", "index", red, source_scenes=["B04.tif"],
                         processing_chain=["write_geotiff"], license="CC-BY-4.0")
print(product.metadata.resolution, product.metadata.crs, product.metadata.bounds)
path = product.save("products")
reloaded = Product.load(path)  # Verifies the SHA-256 digest of the data file.
stac = reloaded.to_stac_item()
print(stac["properties"]["proj:epsg"], [round(v, 4) for v in stac["bbox"]])
print(sorted(json.loads(path.read_text())["data"]))
```

Output:

```text
10.0 EPSG:32635 (500000.0, 6698820.0, 501200.0, 6700020.0)
32635 [27.0, 60.4257, 27.0218, 60.4365]
['kind', 'path', 'sha256']
```

## 7. Validity and integrity checks

### 7.1 Conventions

In this section, MUST, SHOULD and MAY are used as described in RFC 2119 [11] and RFC 8174 [12]. The statements are recommendations of this document for workflows built with the library; the library does not enforce them.

### 7.2 Checks

The library provides the following checks, which a quality control step of a workflow SHOULD apply before a product is released:

| Check | Function | What it reports |
| --- | --- | --- |
| Valid pixels | `Raster.valid_mask()`, `Raster.statistics()` | Pixels that are finite and differ from no-data; count, minimum, maximum, mean, standard deviation and percentiles of the valid pixels per band |
| Cloud and quality flags | `scl_valid_mask`, `landsat_qa_mask` (see [10_satellite_imagery_features.md](10_satellite_imagery_features.md)) | Usable pixels of Sentinel-2 and Landsat quality layers |
| Grid alignment | `Raster.same_grid`, `Scene.is_aligned` | Whether rasters share shape, CRS and transform |
| COG layout | `is_cog`, `geotiff_info` | Whether a file is tiled with overviews in the COG layout |
| Vector structure | `validate_geojson`, `geojson_problems`, `rewind` | RFC 7946 violations (object types, positions, closed rings with at least four positions); `rewind` enforces counterclockwise exterior rings [9] |
| File integrity | `Evidence.from_file`, `Evidence.verify`, `ProvenanceRecord.verify_outputs`, `Product.load` | SHA-256 digests [13] of inputs and outputs, recomputed and compared |
| Model integrity | `unbihexium zoo verify`, `unbihexium.zoo.verify_model` | Files and weights digest of a cached model against the published digest |

A product that fails an integrity check MUST NOT be used as if it were the recorded artefact.

```python
from unbihexium.core import Evidence, Raster
from unbihexium.io import rewind
from unbihexium.io.geojson import geojson_problems

# Share of valid pixels and per-band statistics (no-data and NaN are ignored).
red = Raster.from_file("B04.tif")
data = red.require_data().copy()
data[0, :10, :] = -9999.0  # Ten rows of no data.
patched = red.with_data(data)
print(round(float(patched.valid_mask().mean()), 4), patched.statistics(percentiles=(50,))[0]["count"])

# RFC 7946 structure of a vector product: an open ring and a clockwise exterior ring.
bad = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}
cw = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}
print(geojson_problems(bad))
print(rewind(cw)["coordinates"][0])

# SHA-256 evidence of a file, verified again later.
evidence = Evidence.from_file("B04.tif")
print(len(evidence.checksum), evidence.verify("B04.tif"))
```

Output:

```text
0.9167 13200.0
['geometry.coordinates[0]: a linear ring must be closed']
[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
64 True
```

## 8. Accuracy and image quality metrics

### 8.1 Image quality

The full-reference measures of `unbihexium.metrics.image_quality` compare an estimated image $\hat{x}$ with a reference $x$ (band-first arrays):

| Function | Definition | Source |
| --- | --- | --- |
| `psnr(pred, target, max_val=1.0)` | $10 \log_{10}(\mathrm{MAX}^2 / \mathrm{MSE})$ in dB, infinite for identical images | none (standard definition) |
| `ssim(pred, target, data_range=1.0, sigma=1.5)` | Mean over pixels and bands of $\frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$ with Gaussian-weighted local statistics ($\sigma = 1.5$ pixels), $C_1 = (0.01 L)^2$ and $C_2 = (0.03 L)^2$ | Wang et al. [14] |
| `spectral_angle`, `sam` | Per-pixel angle $\arccos\left(\langle x, \hat{x}\rangle / (\lVert x\rVert \lVert\hat{x}\rVert)\right)$ between spectra in degrees, and its mean | Yuhas, Goetz and Boardman [15] |
| `ergas(reference, estimate, ratio=4.0)` | $\frac{100}{r}\sqrt{\frac{1}{K}\sum_k \left(\mathrm{RMSE}_k / \mu_k\right)^2}$, with $r$ the ratio of the low to the high resolution pixel size and $\mu_k$ the mean of reference band $k$ | Wald [16] |
| `q_index(reference, estimate, block_size=8)` | $Q = \frac{4\sigma_{xy}\mu_x\mu_y}{(\sigma_x^2 + \sigma_y^2)(\mu_x^2 + \mu_y^2)}$ over sliding windows, averaged over windows and bands (`block_size=None` for the whole band) | Wang and Bovik [17] |

`psnr` and `ssim` are the functions of `unbihexium.ai.evaluation`, so training and product validation report the same numbers. An example of SAM, ERGAS and Q for pansharpening is in [10_satellite_imagery_features.md](10_satellite_imagery_features.md).

### 8.2 Thematic accuracy and area estimation

Every error matrix of `unbihexium.metrics` has the reference classes in its rows and the map classes in its columns. With proportions $p_{ij}$ and row and column totals $p_{i+}$ and $p_{+j}$, `accuracy_assessment` reports the overall accuracy $OA = \sum_i p_{ii}$, producer's accuracy $p_{ii}/p_{i+}$, user's accuracy $p_{ii}/p_{+i}$, F1 and IoU per class, Cohen's kappa $(OA - p_e)/(1 - p_e)$ with $p_e = \sum_i p_{i+} p_{+i}$ [18], and the quantity and allocation disagreement of Pontius and Millones [19], which sum to $1 - OA$.

Counting map pixels gives biased class areas when the map has errors. `stratified_area_estimate(matrix, mapped_area, confidence=0.95)` implements the stratified estimator of Olofsson et al. [20] for a stratified random sample whose strata are the map classes: with the mapped area proportions $W_i$ and the sample counts $n_{ij}$ of map class $i$ and reference class $j$, the area proportion of reference class $j$ is $\hat{p}_{\cdot j} = \sum_i W_i n_{ij}/n_{i\cdot}$, and its standard error, the accuracies and their standard errors follow the same reference; confidence intervals are $\pm z \cdot SE$. `sample_allocation` computes stratum sample sizes for a target standard error of the overall accuracy.

### 8.3 Continuous variables

`regression_report(pred, target)` returns the number of pairs, bias, MAE, RMSE, unbiased RMSE $\sqrt{\mathrm{RMSE}^2 - \mathrm{bias}^2}$ [21], relative RMSE, $R^2 = 1 - SS_{res}/SS_{tot}$ computed against the 1:1 line (the Nash-Sutcliffe efficiency [22], which can be negative), Pearson's $r$, and the slope and intercept of the least-squares line. Errors are estimate minus reference, and pairs with NaN are ignored.

### 8.4 Example

```python
import numpy as np

from unbihexium.metrics import (
    accuracy_assessment,
    confusion_matrix,
    psnr,
    regression_report,
    ssim,
    stratified_area_estimate,
)

rng = np.random.default_rng(11)

# Full-reference image quality of a degraded image against its reference.
reference = rng.uniform(0.0, 1.0, size=(3, 64, 64))
degraded = np.clip(reference + rng.normal(0.0, 0.05, size=reference.shape), 0.0, 1.0)
print(round(psnr(degraded, reference, max_val=1.0), 2), round(ssim(degraded, reference, data_range=1.0), 3))

# Error matrix of a three-class map against reference labels (rows reference, columns map).
ref_labels = rng.integers(0, 3, size=2000)
map_labels = np.where(rng.random(2000) < 0.85, ref_labels, rng.integers(0, 3, size=2000))
matrix = confusion_matrix(ref_labels, map_labels, labels=[0, 1, 2])
report = accuracy_assessment(matrix, classes=["water", "forest", "urban"])
print(round(report.overall_accuracy, 3), round(report.kappa, 3), np.round(report.users_accuracy, 3))

# Stratified estimator of class areas with 95 % confidence intervals (areas in ha).
sample = np.array([[97, 3, 0], [2, 180, 8], [1, 17, 92]])  # Reference rows, map columns.
estimate = stratified_area_estimate(sample, mapped_area=[1200.0, 22000.0, 4800.0], classes=["water", "forest", "urban"])
for name, row in estimate.to_dict()["classes"].items():
    print(f"{name:>7}: mapped {row['mapped_area']:8.1f} ha, estimated {row['area']:8.1f} +/- {row['area_ci']:6.1f} ha")

# Regression statistics of estimated against reference values.
truth = rng.uniform(5.0, 30.0, size=500)
estimated = truth + rng.normal(0.5, 2.0, size=500)
stats = regression_report(estimated, truth)
print({k: round(stats[k], 3) for k in ("bias", "rmse", "ubrmse", "r2")})
```

Output:

```text
26.29 0.985
0.904 0.857 [0.896 0.914 0.904]
  water: mapped   1200.0 ha, estimated   1494.0 +/-  373.7 ha
 forest: mapped  22000.0 ha, estimated  20208.0 +/-  952.8 ha
  urban: mapped   4800.0 ha, estimated   6298.0 +/-  890.5 ha
{'bias': 0.63, 'rmse': 2.037, 'ubrmse': 1.937, 'r2': 0.922}
```

For water, the estimate is $28000 \times (0.04286 \times 97/100 + 0.78571 \times 3/200) = 1494$ ha, larger than the mapped 1200 ha because some forest pixels of the sample are water in the reference.

## 9. Model families

### 9.1 Families

The tables below were generated from the model catalogue with the script of section 9.2. The networks are those of `unbihexium.ai.models.networks`: a U-Net with a residual encoder for dense outputs and a super-resolution network with residual blocks and sub-pixel convolution [23], [24]. Parameter counts are those of the built models.

| Family | Domain | Task | Network | Input bands | Outputs | Parameters (tiny / base / large / mega) |
| --- | --- | --- | --- | --- | --- | --- |
| `raster_tiler` | imaging | enhancement | U-Net | red, green, blue | red, green, blue | 732,963 / 7,058,499 / 22,059,171 / 60,450,691 |
| `mosaic_processor` | imaging | enhancement | U-Net | red, green, blue | red, green, blue | 732,963 / 7,058,499 / 22,059,171 / 60,450,691 |
| `ortho_processor` | imaging | enhancement | U-Net | red, green, blue, elevation | dx, dy (px, px) | 733,090 / 7,058,754 / 22,059,554 / 60,451,202 |
| `digitization_2d` | imaging | segmentation | U-Net | red, green, blue | background, building, road, water, vegetation | 732,997 / 7,058,565 / 22,059,269 / 60,450,821 |
| `super_resolution` | imaging | super_resolution | SuperResolutionNet | red, green, blue | red, green, blue | 134,992 / 657,264 / 2,784,528 / 6,109,872 |

| Family | Intended application (once trained) | Reference data needed for training | Suitable input data |
| --- | --- | --- | --- |
| `raster_tiler` | Normalises the radiometry of individual tiles before tiling into a web map. | Radiometrically normalised reference tiles. | RGB tiles |
| `mosaic_processor` | Harmonises the radiometry of a scene to a reference for seamless mosaics. | Radiometrically harmonised reference scenes. | RGB scenes of overlapping areas |
| `ortho_processor` | Estimates the displacement field that removes relief displacement using a DEM. | Displacement fields from rigorous orthorectification. | Raw imagery with a co-registered DEM |
| `digitization_2d` | Segments buildings, roads, water and vegetation for map digitisation. | Masks of the four map feature classes. | Aerial or satellite RGB imagery |
| `super_resolution` | Increases the spatial resolution of RGB imagery by a factor of four. | High resolution reference images. | Pairs of low and high resolution RGB imagery |

The `super_resolution` family upscales by a factor of 4 (`scale: 4` in the catalogue); its output pixel size is a quarter of the input pixel size, and `SuperResolutionResult` records the scale factor. The recommended tile size is 256 pixels for the `tiny` and `base` variants and 512 pixels for `large` and `mega`. The deterministic functions of sections 4 and 5 (resampling, tiling, feathered mosaics, `histogram_match` for radiometric harmonisation) need no training and can produce training references for these families.

### 9.2 Generating the tables

```python
from unbihexium.zoo import list_models
from unbihexium.zoo.catalog import get_spec

FAMILIES = ["raster_tiler", "mosaic_processor", "ortho_processor", "digitization_2d", "super_resolution"]
NETWORK = {"detection": "CenterNet", "scene_regression": "SceneRegressor",
           "super_resolution": "SuperResolutionNet", "spectral_index": "formula"}
params = {(e.family, e.variant.variant.value): e.num_parameters for e in list_models()}
for family in FAMILIES:
    spec = get_spec(family)
    outputs = ", ".join(spec.outputs) + (f" ({', '.join(spec.units)})" if spec.units else "")
    counts = " / ".join(f"{params[family, v]:,}" for v in ("tiny", "base", "large", "mega"))
    print(f"| `{family}` | {spec.domain} | {spec.task.value} | {NETWORK.get(spec.task.value, 'U-Net')} "
          f"| {', '.join(spec.bands)} | {outputs} | {counts} |")
for family in FAMILIES:
    spec = get_spec(family)
    print(f"| `{family}` | {spec.description} | {spec.labels} | {'; '.join(spec.sources)} |")
```

### 9.3 Running a tiny model

`predict` builds a catalogue model in memory with its deterministic starter weights and checks them against the published digest; `unbihexium zoo build` stores a model in the directory named by `UNBIHEXIUM_CACHE` (default `~/.cache/unbihexium`). The class map of the untrained model is meaningless; the example only shows the input layout and the per-class summaries that a trained model would report. For models with more than two classes, pixels whose winning class has a probability below `threshold` (default 0.5) receive the no-data label 255. `class_fractions` is relative to the labelled pixels, and `class_areas` multiplies the pixel count of each class by the pixel area of the affine transform, in square CRS units (here 0.25 m2 per pixel).

```python
import numpy as np
from rasterio.transform import from_origin

from unbihexium.ai import predict
from unbihexium.io import write_geotiff

rng = np.random.default_rng(1)
rgb = rng.uniform(0.0, 0.3, size=(3, 64, 64)).astype("float32")
write_geotiff(rgb, "rgb.tif", crs="EPSG:32635", transform=from_origin(500000, 6700000, 0.5, 0.5))
seg = predict("digitization_2d_tiny", "rgb.tif")
print(type(seg).__name__, seg.classes, seg.mask.shape)
print(int((seg.mask == seg.nodata).sum()), round(sum(seg.class_fractions().values()), 6), round(sum(seg.class_areas().values()), 2))
```

Output:

```text
SegmentationResult ['background', 'building', 'road', 'water', 'vegetation'] (64, 64)
963 1.0 783.25
```

## 10. Command line

The command line has no metadata or quality control commands; the functions of sections 3 to 8 are used from Python. Models are built, verified and run with the generic commands of [docs/reference/cli.md](../reference/cli.md), for example (with `rgb.tif` from section 9.3):

```bash
unbihexium zoo build digitization_2d_tiny
unbihexium zoo verify digitization_2d_tiny
unbihexium predict digitization_2d_tiny rgb.tif map.tif
```

`zoo verify` prints `Verified: digitization_2d_tiny` when the cached files and the weights digest match, and `predict` writes the class map as a GeoTIFF. `unbihexium evaluate` computes the accuracy measures of a trained model on a dataset split (see [docs/model_zoo/training.md](../model_zoo/training.md)).

## 11. Limitations

Earlier versions of this document described metrics, standards and commands that have no implementation. The current release does not provide:

- estimation of the modulation transfer function, relative edge response, signal-to-noise ratio, dynamic range or bit-depth use of an image;
- positional accuracy assessment against ground control points (for example CE90 or the ASPRS positional accuracy standards);
- validation or generation of ISO 19115 metadata; product metadata are recorded as `product.json` and STAC items;
- orthorectification or mosaicking of raw scenes other than the tile mosaics of section 5 and the reprojection of `Raster.reproject` and `Raster.match`;
- conformance testing against OGC standards; the GeoTIFF and COG layouts are produced with GDAL through rasterio, and `is_cog` checks the tiled layout with overviews.

The five model families are untrained (section 1.2).

## 12. Related documents

- [README.md](../../README.md): project overview and installation.
- [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md): limits of the starter models.
- [10_satellite_imagery_features.md](10_satellite_imagery_features.md): sensor tables, radiometry, quality masks and pansharpening.
- [09_benefits_narrative.md](09_benefits_narrative.md): reporting results with uncertainty and provenance.
- [docs/model_zoo/inference.md](../model_zoo/inference.md) and [docs/model_zoo/model_catalog.md](../model_zoo/model_catalog.md): tiled inference and the complete catalogue.
- [index.md](index.md): overview of all capability documents.

## References

[1] Open Geospatial Consortium. OGC GeoTIFF Standard, version 1.1, OGC 19-008r4. 2019. <https://docs.ogc.org/is/19-008r4/19-008r4.html>

[2] Open Geospatial Consortium. OGC Cloud Optimized GeoTIFF Standard, version 1.0, OGC 21-026. 2023. <https://docs.ogc.org/is/21-026/21-026.html>

[3] GDAL/OGR contributors. GDAL/OGR Geospatial Data Abstraction software Library. Open Source Geospatial Foundation. 2025. <https://gdal.org>

[4] Knutsson, H., Westin, C.-F. Normalized and differential convolution. Proceedings of IEEE CVPR, 515-523. 1993.

[5] Burt, P. J., Adelson, E. H. A multiresolution spline with application to image mosaics. ACM Transactions on Graphics 2(4), 217-236. 1983. <https://doi.org/10.1145/245.247>

[6] Huang, B., Reichman, D., Collins, L. M., Bradbury, K., Malof, J. M. Tiling and stitching segmentation output for remote sensing: basic challenges and recommendations. arXiv:1805.12219. 2018. <https://arxiv.org/abs/1805.12219>

[7] OpenStreetMap Wiki contributors. Slippy map tilenames. <https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames>

[8] STAC contributors. SpatioTemporal Asset Catalog specification, version 1.0.0. 2021. <https://github.com/radiantearth/stac-spec>

[9] Butler, H., Daly, M., Doyle, A., Gillies, S., Hagen, S., Schaub, T. The GeoJSON Format. IETF RFC 7946. 2016. <https://doi.org/10.17487/RFC7946>

[10] STAC API contributors. STAC API specification 1.0.0, item search and the query extension. 2023. <https://github.com/radiantearth/stac-api-spec>

[11] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. IETF RFC 2119. 1997. <https://doi.org/10.17487/RFC2119>

[12] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF RFC 8174. 2017. <https://doi.org/10.17487/RFC8174>

[13] National Institute of Standards and Technology. Secure Hash Standard (SHS), FIPS PUB 180-4. 2015. <https://doi.org/10.6028/NIST.FIPS.180-4>

[14] Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600-612. 2004. <https://doi.org/10.1109/TIP.2003.819861>

[15] Yuhas, R. H., Goetz, A. F. H., Boardman, J. W. Discrimination among semi-arid landscape endmembers using the spectral angle mapper (SAM) algorithm. Summaries of the Third Annual JPL Airborne Geoscience Workshop, JPL Publication 92-14, 147-149. 1992.

[16] Wald, L. Data Fusion: Definitions and Architectures. Fusion of Images of Different Spatial Resolutions. Presses de l'Ecole, Ecole des Mines de Paris. 2002.

[17] Wang, Z., Bovik, A. C. A universal image quality index. IEEE Signal Processing Letters 9(3), 81-84. 2002. <https://doi.org/10.1109/97.995823>

[18] Cohen, J. A coefficient of agreement for nominal scales. Educational and Psychological Measurement 20(1), 37-46. 1960. <https://doi.org/10.1177/001316446002000104>

[19] Pontius, R. G., Millones, M. Death to kappa: birth of quantity disagreement and allocation disagreement for accuracy assessment. International Journal of Remote Sensing 32(15), 4407-4429. 2011. <https://doi.org/10.1080/01431161.2011.552923>

[20] Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., Wulder, M. A. Good practices for estimating area and assessing accuracy of land change. Remote Sensing of Environment 148, 42-57. 2014. <https://doi.org/10.1016/j.rse.2014.02.015>

[21] Entekhabi, D., Reichle, R. H., Koster, R. D., Crow, W. T. Performance metrics for soil moisture retrievals and application requirements. Journal of Hydrometeorology 11(3), 832-840. 2010. <https://doi.org/10.1175/2010JHM1223.1>

[22] Nash, J. E., Sutcliffe, J. V. River flow forecasting through conceptual models part I: a discussion of principles. Journal of Hydrology 10(3), 282-290. 1970. <https://doi.org/10.1016/0022-1694(70)90255-6>

[23] Lim, B., Son, S., Kim, H., Nah, S., Lee, K. M. Enhanced deep residual networks for single image super-resolution. CVPR Workshops. 2017. <https://arxiv.org/abs/1707.02921>

[24] Shi, W., Caballero, J., Huszar, F., Totz, J., Aitken, A. P., Bishop, R., Rueckert, D., Wang, Z. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. CVPR. 2016. <https://arxiv.org/abs/1609.05158>

<!--
=============================================================================
End of file docs/capabilities/11_resolution_metadata_qa.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
