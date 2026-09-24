<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/capabilities/08_value_added_imagery.md
Title       : Capability Domain 08: Value-Added Imagery, Elevation and 3D Products
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Capability Domain 08: Value-Added Imagery, Elevation and 3D Products

| Field | Value |
| --- | --- |
| Document | UBX-DOC-608 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 2.0.0 and the main branch |

## Abstract

This document describes what Unbihexium provides for value-added elevation and 3D products derived from imagery: the seven model families of the `imaging` capability domain that estimate disparities, surface and terrain models, object heights and landforms, the related building height family of the `urban` domain, and the library functions that turn elevation grids into derived products, namely terrain derivatives, relief shading, resampling and aggregation, vertical accuracy statistics and the height sensitivity of SAR interferometry. It is written for analysts who produce or use digital elevation, surface and terrain models, and for reviewers who need to know which parts are validated algorithms and which are untrained starter models. For every function it states the purpose, inputs, outputs and the implemented formula with its primary source, and every example has been executed against the current code. The families describe intended applications; they are not validated products.

## Contents

- [1. Scope and Status](#1-scope-and-status)
- [2. Inventory](#2-inventory)
- [3. Model Families: Inputs, Outputs and Architecture](#3-model-families-inputs-outputs-and-architecture)
- [4. Library Functions for Elevation Products](#4-library-functions-for-elevation-products)
- [5. Worked Examples](#5-worked-examples)
- [6. Running a Starter Model](#6-running-a-starter-model)
- [7. Validation and Responsible Use](#7-validation-and-responsible-use)
- [References](#references)

## 1. Scope and Status

### 1.1 Purpose

The capability registry (`unbihexium.registry.CapabilityRegistry`) has no domain called "value-added imagery". This document covers the elevation and 3D subset of the `imaging` domain:

- `stereo_processor`, `dem_generator`, `dsm_generator` and `tri_stereo_processor`: disparity, terrain and surface elevation from panchromatic stereo pairs and triplets;
- `dtm_generator` and `model_3d`: bare-earth terrain and object heights above ground from a surface model;
- `topography_mapper`: landform classes from a DEM;

and, for completeness, the `digitization_3d` family (building heights) of the `urban` domain, which is listed in [06_urban_agriculture.md](06_urban_agriculture.md). The other families of the `imaging` domain (cloud masks, pansharpening, mosaicking, orthorectification, co-registration, denoising, super-resolution and thematic mapping) are described in [04_environment_forestry_image_processing.md](04_environment_forestry_image_processing.md), [10_satellite_imagery_features.md](10_satellite_imagery_features.md) and [11_resolution_metadata_qa.md](11_resolution_metadata_qa.md). The library functions of Section 4 are registered as the `terrain_analysis` capability (domain `analysis`), and as parts of `image_preprocessing` and `visualization` (domain `imaging`), `accuracy_metrics` (domain `analysis`) and `sar_processing` (domain `sar`).

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when they appear in all capitals. A digital elevation model (DEM) is used as the general term; a digital surface model (DSM) includes buildings and vegetation, a digital terrain model (DTM) represents the bare ground, and a normalised surface model (nDSM) is the difference DSM minus DTM. Module paths are given relative to the `unbihexium` package, for example `terrain.derivatives` for `src/unbihexium/terrain/derivatives.py`.

### 1.3 Status of the Models

All eight families of this document are untrained starter models. Each has a complete, trainable network with the input and output layout of its task and deterministic starter weights derived from the model identifier, but none has been trained on Earth observation data. **Until a family is trained on reference elevation data, its outputs carry no information about the input image: an elevation map produced by a starter model is not an elevation model.** The only families of the model zoo that need no training are the 7 spectral index families (28 models) of the `indices` domain, and none of them belongs to this document. The registry records the status: the capabilities of these families have maturity `beta` and the tag `requires_training: "true"`.

The library functions of Section 4 are deterministic implementations of published methods and are covered by the test suite in `tests/`. The project publishes no accuracy figures for any model (see [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md), Section 2).

### 1.4 What the Library Does Not Provide

Earlier versions of this document described capabilities that have no implementation. The library does not contain:

- photogrammetric processing: sensor models, collinearity equations, epipolar rectification, dense image matching (such as semi-global matching), triangulation or bundle adjustment; the stereo families expect epipolar-rectified pairs prepared with other software;
- interferometric DEM generation (the SAR functions compute coherence, phase, unwrapping and displacement, see [12_radar_sar.md](12_radar_sar.md), but no phase-to-height conversion);
- filtering of point clouds or DSMs into DTMs other than the learned `dtm_generator` family;
- robust vertical accuracy statistics such as the normalised median absolute deviation (NMAD) or LE90, and accuracy specifications for any product.

## 2. Inventory

### 2.1 Registry Query

```python
from unbihexium.registry import CapabilityRegistry

FAMILIES = ["stereo_processor", "dem_generator", "dsm_generator", "tri_stereo_processor",
            "dtm_generator", "model_3d", "topography_mapper", "digitization_3d"]
for family in FAMILIES:
    c = CapabilityRegistry.require(family)
    print(f"{c.capability_id:22s} {c.domain.value:8s} {c.task:18s} {c.maturity.value:5s} {c.tags['requires_training']}")
print(CapabilityRegistry.require("terrain_analysis").entry_points)
```

```text
stereo_processor       imaging  dense_regression   beta  true
dem_generator          imaging  dense_regression   beta  true
dsm_generator          imaging  dense_regression   beta  true
tri_stereo_processor   imaging  dense_regression   beta  true
dtm_generator          imaging  dense_regression   beta  true
model_3d               imaging  dense_regression   beta  true
topography_mapper      imaging  segmentation       beta  true
digitization_3d        urban    dense_regression   beta  true
['unbihexium.terrain']
```

Each family has four size variants, `<family>_tiny`, `_base`, `_large` and `_mega` (32 models in total). None of these families has a registered pipeline; they run through the generic `unbihexium.ai.predict.predict` function and the `unbihexium predict` command (Section 6).

### 2.2 Model Families

The two tables below are generated from the model catalogue, `src/unbihexium/zoo/catalog.yaml`, by the script in Section 2.3. The first gives the input channels, the number of stacked acquisitions (dates), the outputs and their units; the second gives the reference data a user needs to train each family, the input data the catalogue suggests, and the range of trainable parameters from the `tiny` to the `mega` variant.

<!-- BEGIN GENERATED TABLES (Section 2.3); do not edit by hand -->

| Family | Name | Domain | Task | Input channels | Dates | Outputs | Units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `stereo_processor` | Stereo Disparity Estimator | imaging | dense_regression | 2: pan_t1, pan_t2 | 1 | disparity | px |
| `dem_generator` | DEM from Stereo | imaging | dense_regression | 2: pan_t1, pan_t2 | 1 | elevation | m |
| `dsm_generator` | DSM from Stereo | imaging | dense_regression | 2: pan_t1, pan_t2 | 1 | surface_elevation | m |
| `tri_stereo_processor` | DSM from Tri-Stereo | imaging | dense_regression | 3: pan_forward, pan_nadir, pan_backward | 1 | surface_elevation | m |
| `dtm_generator` | DTM from DSM | imaging | dense_regression | 1: surface_height | 1 | ground_elevation | m |
| `model_3d` | Normalised Surface Model | imaging | dense_regression | 4: red, green, blue, surface_height | 1 | object_height | m |
| `topography_mapper` | Landform Mapper | imaging | segmentation | 1: elevation | 1 | flat, slope, ridge, valley, peak, pit | - |
| `digitization_3d` | Building Height Estimator | urban | dense_regression | 4: red, green, blue, surface_height | 1 | building_height | m |

| Family | Reference data needed for training | Suggested input data | Parameters, tiny to mega |
| --- | --- | --- | --- |
| `stereo_processor` | Reference disparity maps. | Epipolar-rectified stereo pairs | 732,785 to 60,449,985 |
| `dem_generator` | Reference DEMs, for example LiDAR. | Epipolar-rectified panchromatic stereo pairs | 732,785 to 60,449,985 |
| `dsm_generator` | Reference DSMs, for example LiDAR. | Epipolar-rectified panchromatic stereo pairs | 732,785 to 60,449,985 |
| `tri_stereo_processor` | Reference DSMs. | Epipolar-rectified tri-stereo panchromatic imagery | 732,929 to 60,450,561 |
| `dtm_generator` | Reference DTMs, for example LiDAR ground returns. | Photogrammetric or radar DSMs | 732,641 to 60,449,409 |
| `model_3d` | nDSM from LiDAR. | RGB orthophotos with a DSM | 733,073 to 60,451,137 |
| `topography_mapper` | Landform masks, for example geomorphons. | Copernicus DEM GLO-30; national DEMs | 732,726 to 60,449,734 |
| `digitization_3d` | Building heights from LiDAR or cadastral 3D models. | RGB orthophotos with a DSM | 733,073 to 60,451,137 |

<!-- END GENERATED TABLES -->

### 2.3 Generating the Tables

```python
# GENERATED-TABLES: prints the tables of Section 2.2.
from unbihexium.zoo import get_model, get_spec


def family_tables(families):
    io = ["| Family | Name | Domain | Task | Input channels | Dates | Outputs | Units |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    data = ["| Family | Reference data needed for training | Suggested input data | Parameters, tiny to mega |",
            "| --- | --- | --- | --- |"]
    for s in (get_spec(f) for f in families):
        units = ", ".join(s.units) if s.units else "-"
        io.append(f"| `{s.family}` | {s.name} | {s.domain} | {s.task.value} | "
                  f"{len(s.bands)}: {', '.join(s.bands)} | {s.dates} | {', '.join(s.outputs)} | {units} |")
        tiny, mega = (get_model(f"{s.family}_{v}").num_parameters for v in ("tiny", "mega"))
        data.append(f"| `{s.family}` | {s.labels} | {'; '.join(s.sources)} | {tiny:,} to {mega:,} |")
    return "\n".join(io) + "\n\n" + "\n".join(data)


print(family_tables(FAMILIES))
```

## 3. Model Families: Inputs, Outputs and Architecture

### 3.1 Architecture by Task

The network of a family is chosen by its task (`ai.models.networks`). All networks share a residual convolutional encoder with group normalisation [3] whose input width equals the number of catalogue channels:

| Task | Families in this document | Network | Output of the task API |
| --- | --- | --- | --- |
| `dense_regression` | `stereo_processor`, `dem_generator`, `dsm_generator`, `tri_stereo_processor`, `dtm_generator`, `model_3d`, `digitization_3d` | U-Net [4] with one unbounded output channel (disparity in pixels or height in metres) | `RegressionResult`: one float32 map, written as a GeoTIFF |
| `segmentation` | `topography_mapper` | U-Net with six classes (flat, slope, ridge, valley, peak, pit) | `SegmentationResult`: class map (`uint8`), written as a single-band GeoTIFF |

The stereo images enter as channels of one input (`pan_t1`, `pan_t2`, or `pan_forward`, `pan_nadir`, `pan_backward`); the catalogue therefore records one date. The four variants differ in width and depth (`zoo.get_variant`): `tiny` has 16 base channels and 3 levels, `base` 32 and 4, `large` 48 and 4 with two residual blocks per stage, `mega` 64 and 5 with two blocks per stage. The default inference tile is 256 pixels for `tiny` and `base` and 512 pixels for `large` and `mega`, with overlapping tiles blended by weights that fall towards the tile edges (`ai.inference`). The receptive field of the network limits the disparities and height differences a trained model can resolve; tile size and variant SHOULD be chosen with the expected parallax in mind.

### 3.2 Inputs

Input rasters MUST contain the channels of the family in catalogue order, band first. Stereo pairs MUST be epipolar-rectified so that disparities are horizontal; the library does not rectify images. Height inputs (`surface_height`, `elevation`) are in metres on the grid of the image. Training records per-band normalisation statistics that inference applies; a different band layout is possible with a customised model, see [docs/model_zoo/training.md](../model_zoo/training.md).

### 3.3 Application Map

| Product | Model families | Library functions (Section 4) |
| --- | --- | --- |
| Disparity, DSM and DEM from stereo | `stereo_processor`, `dsm_generator`, `dem_generator`, `tri_stereo_processor` | `metrics.regression_report` for vertical accuracy; `preprocessing.resample`, `aggregate` |
| DTM, nDSM and building heights | `dtm_generator`, `model_3d`, `digitization_3d` | array arithmetic (nDSM = DSM - DTM); `metrics.regression_report` |
| Terrain derivatives | none (deterministic) | `terrain.gradient`, `slope`, `aspect`, `hillshade`, `curvature`, `total_curvature`, `tpi`, `tri`, `roughness`, `vrm`, `twi` |
| Landform classes | `topography_mapper` | `terrain.tpi`, `curvature` as explanatory layers |
| Shaded relief | none (deterministic) | `visualization.hillshade`, `multidirectional_hillshade`, `shade_image` |
| InSAR height sensitivity | none (deterministic) | `sar.height_of_ambiguity` |

## 4. Library Functions for Elevation Products

### 4.1 Window Derivatives

Module `terrain.derivatives` computes derivatives of a north-up DEM (first row north, NaN for nodata) in the 3 x 3 window

$$\begin{matrix} z_1 & z_2 & z_3 \\ z_4 & z_5 & z_6 \\ z_7 & z_8 & z_9 \end{matrix}$$

with $x$ pointing east and $y$ north. `resolution` is the cell size in the units of the elevations (one number or `(x size, y size)`), `z_factor` converts elevation units when they differ. The grid is extended by one cell with linear extrapolation ($2 z_{\mathrm{edge}} - z_{\mathrm{inner}}$), so a plane keeps its exact slope up to the border; NaN propagates to every result that uses it.

| Function | Definition | Source |
| --- | --- | --- |
| `gradient(dem, resolution, z_factor)` | $\partial z / \partial x = \big((z_3 + 2z_6 + z_9) - (z_1 + 2z_4 + z_7)\big) / (8\Delta x)$, $\partial z / \partial y = \big((z_1 + 2z_2 + z_3) - (z_7 + 2z_8 + z_9)\big) / (8\Delta y)$ | [5] |
| `slope(dem, resolution, units, z_factor)` | $\arctan \lVert \nabla z \rVert$ in `"degrees"`, `"radians"`, or $100 \lVert \nabla z \rVert$ in `"percent"` | [5] |
| `aspect(dem, resolution)` | $\operatorname{atan2}(-\partial z/\partial x, -\partial z/\partial y) \bmod 360$, degrees clockwise from north, NaN on flat cells | [6] |
| `hillshade(dem, resolution, azimuth=315, altitude=45, z_factor)` | $255 \max\big(0, \cos Z \cos S + \sin Z \sin S \cos(\phi - A)\big)$ with zenith $Z = 90^\circ - \text{altitude}$, slope $S$, aspect $A$ and light azimuth $\phi$ | [5], [6] |
| `curvature(dem, resolution)` | profile $-2(D g^2 + E h^2 + F g h) / (g^2 + h^2)$ and plan $-2(D h^2 + E g^2 - F g h) / (g^2 + h^2)$, 0 where the gradient vanishes | [7] |
| `total_curvature(dem, resolution)` | $-2(D + E)$ | [7] |
| `tpi(dem, radius=1)` | $z_5$ minus the mean of the square neighbourhood of half-width `radius` (centre excluded) | [8] |
| `tri(dem, method="riley")` | $\sqrt{\sum_{k \ne 5} (z_k - z_5)^2}$; `method="wilson"`: $\frac{1}{8} \sum_{k \ne 5} \lvert z_k - z_5 \rvert$ | [9], [10] |
| `roughness(dem)` | $\max_k z_k - \min_k z_k$ | [10] |
| `vrm(dem, resolution, window_size=3)` | $1 - \lVert \bar{\mathbf{n}} \rVert$, the dispersion of the unit surface normals $\mathbf{n} \propto (-\partial z/\partial x, -\partial z/\partial y, 1)$ averaged over the window | [11] |

The curvature coefficients are those of the Zevenbergen and Thorne polynomial $z = D x^2 + E y^2 + F x y + G x + H y + z_5$: $D = ((z_4 + z_6)/2 - z_5) / \Delta x^2$, $E = ((z_2 + z_8)/2 - z_5) / \Delta y^2$, $F = (-z_1 + z_3 + z_7 - z_9) / (4 \Delta x \Delta y)$, $g = G = (z_6 - z_4) / (2\Delta x)$ and $h = H = (z_2 - z_8) / (2\Delta y)$. All three curvatures are in 1 / length units, without the $(1 + \lVert \nabla z \rVert^2)$ normalisation, and positive on convex forms (a slope that steepens downhill, diverging contours of a ridge, a hilltop).

`terrain.twi` (topographic wetness index) and the flow routing functions of `terrain.hydrology` are documented with their formulas in [05_asset_management_energy.md](05_asset_management_energy.md), Section 4.4, and [07_risk_defense_neutral.md](07_risk_defense_neutral.md), Section 4.2; `terrain.viewshed` in [05_asset_management_energy.md](05_asset_management_energy.md), Section 4.2.

### 4.2 Relief Shading for Maps

Module `visualization.relief` produces shaded relief for display. `hillshade(dem, cellsize, azimuth=315.0, altitude=45.0, z_factor=1.0, as_uint8=False)` uses the same Horn gradients and illumination model as `terrain.hillshade` but returns values in [0, 1] (or bytes with `as_uint8=True`) and replicates the border pixels instead of extrapolating them. `multidirectional_hillshade(dem, cellsize, azimuths=(225, 270, 315, 360), weights=None)` blends several light directions, which reduces the loss of detail on slopes that face away from a single light source [12]. `shade_image(rgb, shade, strength=0.6)` multiplies an RGB image by $1 - s + s \cdot \mathrm{shade}$ with strength $s$.

### 4.3 Resampling and Aggregation of Elevation Grids

Module `preprocessing.resample` changes the grid of elevation products. `resample(image, shape, method="bilinear", nodata=None)` interpolates an `(H, W)` or `(C, H, W)` array to a new shape with `"nearest"`, `"bilinear"` or `"cubic"` spline interpolation; `aggregate(image, factor, method="mean", nodata=None, trim=True)` reduces non-overlapping blocks of `factor x factor` cells with `mean`, `sum`, `min`, `max`, `median` or `mode`, which is the appropriate way to coarsen a DEM (a block mean preserves the volume); `scaled_transform(transform, old_shape, new_shape)` returns the affine transform of the new grid. Derivatives depend on the cell size, so slope and curvature SHOULD be computed at the resolution at which they are interpreted.

### 4.4 Vertical Accuracy

`metrics.regression_report(pred, target)` compares estimated heights with reference heights (for example check points or a LiDAR DTM), ignoring pairs with NaN. With errors $e_i = \hat z_i - z_i$ (estimate minus reference) it reports the number of pairs, bias $\bar e$, MAE, RMSE $\sqrt{\overline{e^2}}$, the unbiased RMSE $\sqrt{\mathrm{RMSE}^2 - \bar e^2}$ [13], the relative RMSE, $R^2 = 1 - SS_{\mathrm{res}} / SS_{\mathrm{tot}}$ against the 1:1 line (the Nash-Sutcliffe efficiency, which can be negative) [14], the Pearson correlation and the least-squares line of estimate on reference. RMSE assumes normally distributed errors; elevation errors of photogrammetric and learned products are often heavy-tailed, and robust measures such as the NMAD [15] SHOULD be reported as well (they are computed with NumPy; the library does not implement them).

### 4.5 Height Sensitivity of SAR Interferometry

`sar.height_of_ambiguity(wavelength, slant_range, incidence_angle, perpendicular_baseline)` returns the height difference that produces one $2\pi$ phase cycle in a repeat-pass interferogram [16]:

$$h_a = \frac{\lambda R \sin \theta}{2 B_\perp},$$

with wavelength $\lambda$, slant range $R$, incidence angle $\theta$ (degrees) and perpendicular baseline $B_\perp$ (metres). A small $h_a$ (long baseline) makes the phase more sensitive to topography, which matters both for topographic mapping and for removing the topographic phase in displacement studies ([12_radar_sar.md](12_radar_sar.md)).

## 5. Worked Examples

The examples below were executed in sequence in one Python session (CPython 3.13, NumPy 2, CPU only); each output block is the real output. They use small synthetic arrays so that the results can be checked by hand.

### 5.1 Derivatives of a Hill

A Gaussian hill of 30 m height on a 7 x 7 grid with 10 m cells; the printed rows run west to east through the summit.

```python
import numpy as np
from unbihexium.terrain import (aspect, curvature, gradient, hillshade, roughness, slope, total_curvature,
                                tpi, tri, vrm)

rows, cols = np.mgrid[0:7, 0:7].astype(float)
dem = 500.0 + 30.0 * np.exp(-((rows - 3) ** 2 + (cols - 3) ** 2) / 4.0)
dzdx, dzdy = gradient(dem, 10.0)
print(round(float(dzdx[3, 1]), 3), round(float(dzdy[1, 3]), 3))
print(np.round(slope(dem, 10.0)[3], 1))
print(np.round(aspect(dem, 10.0)[3], 1))
print(np.round(hillshade(dem, 10.0)[3]).astype(int))
profile, plan = curvature(dem, 10.0)
print(float(profile[3, 3]), float(plan[3, 3]), round(float(total_curvature(dem, 10.0)[3, 3]), 4))
print(round(float(tpi(dem)[3, 3]), 2), round(float(tri(dem)[3, 3]), 2), round(float(tri(dem, method="wilson")[3, 3]), 2),
      round(float(roughness(dem)[3, 3]), 2), round(float(vrm(dem, 10.0)[3, 3]), 4))
```

```text
0.898 -0.898
[35.  41.9 40.1  0.  40.1 41.9 35. ]
[270. 270. 270.  nan  90.  90.  90.]
[221 219 220 180  56  49  75]
0.0 0.0 0.2654
9.22 27.08 9.22 11.8 0.2318
```

West of the summit the terrain rises towards the east ($\partial z / \partial x > 0$) and faces west (aspect 270); north of the summit it rises towards the south ($\partial z / \partial y < 0$). The summit has no slope and an undefined aspect (NaN), profile and plan curvature 0 because the gradient vanishes, a positive total curvature (convex) and a positive topographic position. With the light from the north-west, the western slope is bright and the eastern slope dark.

### 5.2 Multidirectional Shaded Relief

```python
from unbihexium.visualization import hillshade as relief, multidirectional_hillshade, shade_image

print(np.round(relief(dem, 10.0)[3], 3))
print(np.round(multidirectional_hillshade(dem, 10.0)[3], 3))
grey = np.full((7, 7, 3), 200, dtype=np.uint8)
print(shade_image(grey, multidirectional_hillshade(dem, 10.0), strength=0.6)[3, :, 0])
```

```text
[0.833 0.86  0.863 0.707 0.218 0.192 0.502]
[0.808 0.811 0.816 0.707 0.265 0.241 0.526]
[177 177 178 165 112 109 143]
```

The first row equals `terrain.hillshade` divided by 255 in the interior; the border cells differ because the visualisation function replicates border pixels.

### 5.3 Coarsening a DEM

```python
from unbihexium.preprocessing import aggregate, resample, scaled_transform

fine = np.arange(36, dtype=float).reshape(6, 6)
coarse = aggregate(fine, factor=3, method="mean")
print(coarse)
print(scaled_transform((10.0, 0.0, 500000.0, 0.0, -10.0, 6700060.0), fine.shape, coarse.shape))
print(resample(coarse, (6, 6), method="bilinear").shape)
```

```text
[[ 7. 10.]
 [25. 28.]]
(30.0, 0.0, 500000.0, 0.0, -30.0, 6700060.0)
(6, 6)
```

### 5.4 Normalised Surface Model and Vertical Accuracy

A 3 x 3 DSM with a building in the north-west corner and a DTM of the ground, followed by the comparison of six estimated heights with check points.

```python
from unbihexium.metrics import regression_report

dsm = np.array([[212.0, 214.5, 203.1], [211.8, 202.9, 203.0], [203.2, 203.1, 202.8]])
dtm = np.array([[202.9, 203.0, 203.0], [202.8, 202.9, 202.9], [203.1, 203.0, 202.8]])
ndsm = np.clip(dsm - dtm, 0.0, None)
print(np.round(ndsm, 1))

estimate = np.array([101.2, 99.4, 103.9, 98.7, 100.8, 102.2])
reference = np.array([100.0, 100.1, 103.0, 99.5, 100.2, 101.1])
print({k: round(v, 3) for k, v in regression_report(estimate, reference).items()})
```

```text
[[ 9.1 11.5  0.1]
 [ 9.   0.   0.1]
 [ 0.1  0.1  0. ]]
{'n': 6, 'bias': 0.383, 'mae': 0.883, 'rmse': 0.908, 'ubrmse': 0.823, 'relative_rmse': 0.009, 'r2': 0.379, 'r': 0.91, 'slope': 1.359, 'intercept': -35.775}
```

### 5.5 Height of Ambiguity of a C-Band Pair

Sentinel-1 wavelength (5.547 cm), a slant range of 850 km, an incidence angle of 39 degrees and a perpendicular baseline of 150 m:

```python
from unbihexium.sar import height_of_ambiguity

print(round(height_of_ambiguity(wavelength=0.05547, slant_range=850e3, incidence_angle=39.0,
                                perpendicular_baseline=150.0), 1), "m per fringe")
```

```text
98.9 m per fringe
```

### 5.6 Writing Derivatives as a GeoTIFF

```python
from unbihexium.core.raster import Raster

transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 6700070.0)
stack = np.stack([dem, slope(dem, 10.0), hillshade(dem, 10.0)]).astype("float32")
path = Raster.from_array(stack, crs="EPSG:3067", transform=transform).to_file("terrain_derivatives.tif")
check = Raster.from_file(path)
print(path, check.data.shape, check.crs)
```

```text
terrain_derivatives.tif (3, 7, 7) EPSG:3067
```

## 6. Running a Starter Model

This section runs the smallest variant of `dsm_generator` on a synthetic stereo pair to show the input and output contract. Because the model is untrained, the "surface elevation" it returns is a meaningless field of values within a few metres of zero, not heights; the example demonstrates the mechanics only.

```python
from unbihexium.ai.predict import predict, write_result
from unbihexium.zoo import get_model

entry = get_model("dsm_generator_tiny")
print(entry.requires_training, entry.spec.bands, entry.spec.outputs, entry.spec.units)
rng = np.random.default_rng(3)
pair = rng.uniform(0.0, 1.0, size=(2, 64, 64)).astype("float32")
raster = Raster.from_array(pair, crs="EPSG:32635", transform=(0.7, 0.0, 385000.0, 0.0, -0.7, 6672000.0))
raster.to_file("stereo_pair.tif")
result = predict("dsm_generator_tiny", raster)
print({k: round(v, 2) for k, v in result.summary()["surface_elevation"].items()})
print(write_result(result, "dsm.tif"))
```

```text
True ('pan_t1', 'pan_t2') ('surface_elevation',) ('m',)
{'mean': -1.13, 'min': -5.58, 'max': 4.25, 'std': 1.23}
dsm.tif
```

The same run on the command line, with a family name and a variant:

```bash
unbihexium predict dsm_generator --variant tiny stereo_pair.tif dsm_cli.tif
```

```text
Wrote: dsm_cli.tif (dsm_generator_tiny)
```

After training (`unbihexium train`, see [docs/model_zoo/training.md](../model_zoo/training.md)), the checkpoint path is passed instead of the model identifier. Inference options are described in [docs/model_zoo/inference.md](../model_zoo/inference.md).

## 7. Validation and Responsible Use

Users of this document:

- MUST NOT publish or use elevation, height or landform maps produced by the starter models as if they were measurements;
- MUST validate a trained model against independent reference heights (for example GNSS check points or LiDAR) that represent the terrain, land cover and sensor of use, and SHOULD report bias, RMSE and a robust measure of the vertical error together with the number and distribution of check points;
- SHOULD state the vertical datum, the units, the cell size and the date of the source imagery with every elevation product, and whether it is a DSM, a DTM or an nDSM;
- SHOULD consider that building heights and 3D models of individual properties can be personal data (see [PRIVACY.md](../../PRIVACY.md)).

Elevation products serve the intended uses of [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md) (Section 3), for example hydrology, hazard mapping and planning. This document is not legal advice.

## References

[1] S. Bradner. Key words for use in RFCs to Indicate Requirement Levels (RFC 2119). 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] B. Leiba. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words (RFC 8174). 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Y. Wu, K. He. Group Normalization. European Conference on Computer Vision. 2018. <https://arxiv.org/abs/1803.08494>

[4] O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI. 2015. <https://arxiv.org/abs/1505.04597>

[5] B. K. P. Horn. Hill shading and the reflectance map. Proceedings of the IEEE 69(1), 14-47. 1981. <https://doi.org/10.1109/PROC.1981.11918>

[6] P. A. Burrough, R. A. McDonnell. Principles of Geographical Information Systems. Oxford University Press. 1998.

[7] L. W. Zevenbergen, C. R. Thorne. Quantitative analysis of land surface topography. Earth Surface Processes and Landforms 12(1), 47-56. 1987. <https://doi.org/10.1002/esp.3290120107>

[8] A. Weiss. Topographic position and landforms analysis. Poster presentation, ESRI User Conference, San Diego. 2001.

[9] S. J. Riley, S. D. DeGloria, R. Elliot. A terrain ruggedness index that quantifies topographic heterogeneity. Intermountain Journal of Sciences 5(1-4), 23-27. 1999.

[10] M. F. J. Wilson, B. O'Connell, C. Brown, J. C. Guinan, A. J. Grehan. Multiscale terrain analysis of multibeam bathymetry data for habitat mapping on the continental slope. Marine Geodesy 30(1-2), 3-35. 2007. <https://doi.org/10.1080/01490410701295962>

[11] J. M. Sappington, K. M. Longshore, D. B. Thompson. Quantifying landscape ruggedness for animal habitat analysis: a case study using bighorn sheep in the Mojave Desert. Journal of Wildlife Management 71(5), 1419-1426. 2007. <https://doi.org/10.2193/2005-723>

[12] R. K. Mark. Multidirectional, oblique-weighted, shaded-relief image of the Island of Hawaii. U.S. Geological Survey Open-File Report 92-422. 1992. <https://doi.org/10.3133/ofr92422>

[13] D. Entekhabi, R. H. Reichle, R. D. Koster, W. T. Crow. Performance metrics for soil moisture retrievals and application requirements. Journal of Hydrometeorology 11(3), 832-840. 2010. <https://doi.org/10.1175/2010JHM1223.1>

[14] J. E. Nash, J. V. Sutcliffe. River flow forecasting through conceptual models part I: a discussion of principles. Journal of Hydrology 10(3), 282-290. 1970. <https://doi.org/10.1016/0022-1694(70)90255-6>

[15] J. Hoehle, M. Hoehle. Accuracy assessment of digital elevation models by means of robust statistical methods. ISPRS Journal of Photogrammetry and Remote Sensing 64(4), 398-406. 2009. <https://doi.org/10.1016/j.isprsjprs.2009.02.003>

[16] R. F. Hanssen. Radar Interferometry: Data Interpretation and Error Analysis. Kluwer Academic Publishers, Dordrecht. 2001. <https://doi.org/10.1007/0-306-47633-9>

<!--
=============================================================================
End of file docs/capabilities/08_value_added_imagery.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
