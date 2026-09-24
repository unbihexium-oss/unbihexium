<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/capabilities/02_tourism_data_processing.md
Title       : Capability Domain 02: Tourism and Data Processing
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Capability Domain 02: Tourism and Data Processing

| Field | Value |
| --- | --- |
| Document | UBX-DOC-602 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 2.0.0 and the main branch (model catalogue 2.0.0) |

## Abstract

This document describes capability domain 02, "tourism and data processing". It covers the model families of the catalogue domains `tourism` and `analysis` and the deterministic spatial analysis functions that the registry files under the domain `analysis`: terrain derivatives and visibility (`unbihexium.terrain`), geostatistics (`unbihexium.geostat`), multi-criteria suitability, cost surfaces, network routing and zonal statistics (`unbihexium.analysis`), and accuracy assessment (`unbihexium.metrics`). It is intended for analysts who need to know which of these tools compute a published method exactly and which are learned starter models, for contributors, and for reviewers. For every function it states the inputs, the outputs and the formula implemented, with its primary source; the twelve model families are listed in tables generated from the catalogue; all examples were executed. The twelve model families are untrained starter models, and the domain describes intended applications, not validated products.

## Contents

1. [Scope and status](#1-scope-and-status)
2. [Place in the registry and the catalogue](#2-place-in-the-registry-and-the-catalogue)
3. [Model families](#3-model-families)
4. [Terrain analysis](#4-terrain-analysis)
5. [Geostatistics](#5-geostatistics)
6. [Spatial analysis](#6-spatial-analysis)
7. [Accuracy assessment](#7-accuracy-assessment)
8. [Examples](#8-examples)
9. [Limitations and responsible use](#9-limitations-and-responsible-use)
10. [Related documents](#10-related-documents)
11. [References](#references)

## 1. Scope and status

### 1.1 What the domain covers

Domain 02 combines two groups of capabilities that serve the same kind of question, "where is something accessible, visible, suitable or dense, and how certain is the map":

- **deterministic functions** that implement published algorithms: slope, aspect, hillshade, curvature and ruggedness measures, line-of-sight viewsheds, empirical variograms, ordinary and universal kriging, inverse distance weighting, global and local spatial autocorrelation, the Analytic Hierarchy Process with weighted overlay, raster cost distance and least-cost paths, graph routing, zonal statistics, and error matrix and area estimation measures;
- **learned model families** of the catalogue domains `tourism` (4 families) and `analysis` (8 families) that are designed to learn a surface or a scene value (travel time, visibility, suitability, density, economic indicators) from imagery and covariates.

### 1.2 Status of the models and the functions

The functions of Sections 4 to 7 are deterministic, need no training and are covered by the unit tests in `tests/`. The twelve model families of Section 3 are **untrained starter models**: complete, trainable networks with deterministic initial weights that have not been fitted to any data. Their output is meaningless until they are trained on reference data (Section 8.5 shows the output of an untrained model). Several families are learned counterparts of a function of this document, for example `viewshed_analyzer` (Section 4.3), `geostatistical_analyzer` (Section 5) and `zonal_statistics` (Section 6.4); the function is exact for its inputs, whereas the family is meant to approximate its result from imagery after training. The only models of the zoo that need no training are the spectral index families of [domain 03](03_indices_flood_water.md).

### 1.3 Changes from the previous version

Version 1 of this document listed ten "production" models with accuracy tables, an MLP architecture, a Siamese change detector, gravity and distance-decay accessibility models and a time series model. There is no code for gravity models or distance-decay functions; the architectures are U-Net regressors and encoder regressors (Section 3.2); the time series family `timeseries_analyzer` belongs to the agriculture domain ([domain 06](06_urban_agriculture.md)) and `mobility_analyzer` to the urban domain. These parts were removed and replaced by the functions that the library implements.

## 2. Place in the registry and the catalogue

The enumeration `unbihexium.registry.CapabilityDomain` has the members `TOURISM = "tourism"` and `ANALYSIS = "analysis"`. The capability registry holds:

| Registry domain | Model capabilities (families) | Library capabilities | Total |
| --- | --- | --- | --- |
| `tourism` | 4, maturity `beta` | none | 4 |
| `analysis` | 8, maturity `beta` | `terrain_analysis` (`unbihexium.terrain`), `geostatistics` (`unbihexium.geostat`), `spatial_analysis` (`unbihexium.analysis`), `accuracy_metrics` (`unbihexium.metrics`, `unbihexium.ai.evaluation`), all `stable` | 12 |

None of the twelve families has a registered pipeline; each is run with `unbihexium predict <family>_<variant> INPUT OUTPUT` or `unbihexium.ai.predict`. The hydrological part of `terrain_analysis` (depression filling, flow direction, flow accumulation, watersheds, streams and the topographic wetness index) is described with the water capabilities in [domain 03](03_indices_flood_water.md), and the image quality measures of `unbihexium.metrics` with the imaging capabilities in [domain 04](04_environment_forestry_image_processing.md).

## 3. Model families

### 3.1 Inventory

The tables were generated from the catalogue and the model registry of the installed package with the script of [index.md, Section 4](index.md#4-regenerating-the-family-tables), run with the arguments `tourism analysis`. Units are given in brackets after the outputs; `1` means dimensionless.

| Family | Domain | Task | Input bands | Outputs | Parameters (tiny / base / large / mega) |
| --- | --- | --- | --- | --- | --- |
| `tourist_destination_monitor` | tourism | segmentation | red, green, blue | background, beach, built_up, vegetation, water | 732,997 / 7,058,565 / 22,059,269 / 60,450,821 |
| `accessibility_analyzer` | tourism | dense_regression | B02, B03, B04, B08, B11, B12, elevation, slope, population, road_distance | travel_time [min] | 733,937 / 7,060,449 / 22,062,097 / 60,454,593 |
| `route_planner` | tourism | dense_regression | B02, B03, B04, B08, B11, B12, elevation, slope, population, road_distance | travel_cost [s m-1] | 733,937 / 7,060,449 / 22,062,097 / 60,454,593 |
| `viewshed_analyzer` | tourism | dense_regression | elevation | visible_fraction [1] | 732,641 / 7,057,857 / 22,058,209 / 60,449,409 |
| `site_suitability` | analysis | dense_regression | B02, B03, B04, B08, B11, B12, elevation, slope, population, road_distance | suitability [1] | 733,937 / 7,060,449 / 22,062,097 / 60,454,593 |
| `spatial_analyzer` | analysis | dense_regression | blue, green, red, nir | density [ha-1] | 733,073 / 7,058,721 / 22,059,505 / 60,451,137 |
| `geostatistical_analyzer` | analysis | dense_regression | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, elevation, slope | value [user defined] | 734,225 / 7,061,025 / 22,062,961 / 60,455,745 |
| `spatial_relationship` | analysis | dense_regression | blue, green, red, nir | distance [m] | 733,073 / 7,058,721 / 22,059,505 / 60,451,137 |
| `business_valuation` | analysis | scene_regression | blue, green, red, nir | activity_index [1] | 494,625 / 3,745,345 / 14,606,497 / 37,762,945 |
| `economic_spatial_assessor` | analysis | scene_regression | red, green, blue | median_value [currency m-2] | 494,481 / 3,745,057 / 14,606,065 / 37,762,369 |
| `resource_allocation` | analysis | scene_regression | blue, green, red, nir | population, service_demand [persons, 1] | 494,658 / 3,745,410 / 14,606,594 / 37,763,074 |
| `zonal_statistics` | analysis | scene_regression | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | vegetation_fraction, water_fraction, built_up_fraction [1, 1, 1] | 495,555 / 3,747,203 / 14,609,283 / 37,766,659 |

| Family | Name | Intended application | Reference data needed for training |
| --- | --- | --- | --- |
| `tourist_destination_monitor` | Tourist Destination Monitor | Maps beaches, built-up areas, vegetation and water around tourist destinations. | Masks of the land cover classes. |
| `accessibility_analyzer` | Accessibility Analyser | Estimates travel time to the nearest service centre per pixel. | Travel time rasters from network analysis or surveys. |
| `route_planner` | Route Cost Surface | Estimates a travel cost surface for least-cost routing. | Travel cost rasters from speed models. |
| `viewshed_analyzer` | Visibility Estimator | Estimates the fraction of the surrounding area visible from each pixel. | Visibility rasters from viewshed analysis. |
| `site_suitability` | Site Suitability | Scores general site suitability for development. | Suitability scores from multi-criteria analysis. |
| `spatial_analyzer` | Spatial Density Estimator | Estimates the density of a mapped phenomenon, for example buildings per hectare. | Density rasters derived from reference vector data. |
| `geostatistical_analyzer` | Geostatistical Surface Estimator | Learns a continuous surface of a sampled variable from covariates, as a learned alternative to kriging. | Point samples rasterised as sparse targets. |
| `spatial_relationship` | Proximity Estimator | Estimates the distance to the nearest mapped feature of interest. | Distance rasters derived from reference vector data. |
| `business_valuation` | Economic Activity Estimator | Estimates an economic activity index of an area from imagery. | Economic indicators aggregated to chips. |
| `economic_spatial_assessor` | Property Value Estimator | Estimates median property value of an area. | Property transaction statistics aggregated to chips. |
| `resource_allocation` | Service Demand Estimator | Estimates population and service demand of an area for resource allocation. | Census population and service statistics per chip. |
| `zonal_statistics` | Zonal Cover Estimator | Estimates the fractional cover of vegetation, water and built-up area in a chip. | Cover fractions per chip from reference land cover maps. |

"Intended application" is the catalogue description of what a model does once it has been trained. Several families take auxiliary layers (population, road distance, elevation, slope) as input bands; the user supplies them co-registered with the imagery. The deterministic functions of Sections 4 and 6 can produce several of the training targets, for example travel cost rasters with `cost_distance`, visibility rasters with `viewshed` and suitability scores with `weighted_overlay`.

### 3.2 Architectures and outputs

Dense regression families (`unet_regression`) return a tensor $(N, K, H, W)$ with one channel per target in the listed units; families whose catalogue entry has a value range of $[0, 1]$ (`viewshed_analyzer`, `site_suitability`, `zonal_statistics`) end in a sigmoid. Scene regression families (`encoder_regressor`) return $(N, K)$, one value per target and image chip, and are written as JSON by `unbihexium predict`. The segmentation family `tourist_destination_monitor` (`unet`) returns class logits $(N, K, H, W)$.

## 4. Terrain analysis

`unbihexium.terrain` works on gridded digital elevation models (DEM) whose first row is the northern edge; NaN marks nodata. `resolution` is the cell size (one number or `(x size, y size)`) in the units of the elevations, and `z_factor` converts elevation units when they differ.

### 4.1 Derivatives

With the $3 \times 3$ window numbered $z_1, z_2, z_3$ (north row), $z_4, z_5, z_6$, $z_7, z_8, z_9$ (south row) and cell sizes $\Delta x$, $\Delta y$, the gradient is the finite difference of Horn [1]:

$$
\frac{\partial z}{\partial x} = \frac{(z_3 + 2 z_6 + z_9) - (z_1 + 2 z_4 + z_7)}{8\,\Delta x}, \qquad
\frac{\partial z}{\partial y} = \frac{(z_1 + 2 z_2 + z_3) - (z_7 + 2 z_8 + z_9)}{8\,\Delta y}
$$

with $y$ pointing north. The DEM is extended by one cell with linear extrapolation, so a plane keeps its exact slope up to the border.

| Function | Output | Formula and source |
| --- | --- | --- |
| `gradient(dem, resolution, z_factor)` | $(\partial z/\partial x, \partial z/\partial y)$ | Horn [1] |
| `slope(dem, resolution, units, z_factor)` | slope in `degrees`, `percent` or `radians` | $S = \arctan \lvert \nabla z \rvert$ |
| `aspect(dem, resolution)` | downslope direction, degrees clockwise from north, NaN on flat cells | $A = \operatorname{atan2}(-\partial z/\partial x, -\partial z/\partial y)$ |
| `hillshade(dem, resolution, azimuth=315, altitude=45, z_factor)` | brightness 0 to 255 | $255 \max(0, \cos Z \cos S + \sin Z \sin S \cos(\phi - A))$ with zenith $Z = 90$ degrees minus the altitude and light azimuth $\phi$ [1] |
| `curvature(dem, resolution)` | profile and plan curvature | quadratic surface of Zevenbergen and Thorne [2], positive on convex forms |
| `total_curvature(dem, resolution)` | total curvature | $-2(D + E)$ of the same surface |
| `tpi(dem, radius=1)` | topographic position index | $z_5$ minus the mean of the square neighbourhood without the centre [3] |
| `tri(dem, method="riley")` | terrain ruggedness index | $\sqrt{\sum_{i \ne 5} (z_i - z_5)^2}$ (Riley et al. [4]) or the mean absolute difference (`"wilson"`, [5]) |
| `roughness(dem)` | range of elevations in the window | [5] |
| `vrm(dem, resolution, window_size=3)` | vector ruggedness measure, 0 (flat) to 1 | one minus the length of the mean unit surface normal (Sappington et al. [6]) |

### 4.2 Hydrology

`fill_depressions`, `flow_direction_d8`, `flow_accumulation`, `watershed`, `extract_streams` and `twi` belong to the same registry capability but are documented, with their formulas, in [domain 03, Section 5](03_indices_flood_water.md#5-hydrological-terrain-analysis).

### 4.3 Visibility

`viewshed(dem, observer, resolution, observer_height=1.7, target_height=0.0, max_distance=None, earth_curvature=False, refraction=0.13)` returns a boolean grid of the cells visible from the observer cell `(row, column)`. For every target cell the line of sight from the observer's eye to the target is sampled once per crossed row or column, the terrain is interpolated bilinearly, and the target is visible when the tangent of its elevation angle is at least the largest tangent of the intermediate samples (the exact "R3" test of Franklin and Ray [7] on an interpolated surface). With `earth_curvature=True` elevations are lowered by $\Delta z = (1 - k)\, d^2 / (2R)$ with the refraction coefficient $k$ and $R = 6\,371\,000$ m. The cost is $O(N D)$ for $N$ cells and lines of $D$ cells; `max_distance` limits it.

## 5. Geostatistics

`unbihexium.geostat` analyses and interpolates point data given as coordinates `(n, 2)` and values `(n,)`.

### 5.1 Variograms

`empirical_variogram(coordinates, values, n_lags=15, max_lag=None, estimator="matheron")` bins the pairs by distance and estimates the semivariance with the classical estimator of Matheron [8] or the robust estimator of Cressie and Hawkins [9]:

$$
\hat\gamma_M(h) = \frac{1}{2 N(h)} \sum_{N(h)} (z_i - z_j)^2, \qquad
\hat\gamma_{CH}(h) = \frac{\left(\frac{1}{N(h)} \sum_{N(h)} \lvert z_i - z_j \rvert^{1/2}\right)^4}{2\,(0.457 + 0.494 / N(h))} .
$$

`Variogram(n_lags, max_lag, model, estimator, shape).fit(coordinates, values)` fits a model by least squares weighted with the pair counts $N(h)$ [10] and returns a `VariogramResult` (lags, semivariance, nugget $c_0$, partial sill $c$, range parameter $a$, fitted values, counts). The models of `variogram_function` are, for $h > 0$ and $r = h/a$ ($\gamma(0) = 0$):

| Model | $\gamma(h)$ |
| --- | --- |
| `spherical` | $c_0 + c\,(1.5 r - 0.5 r^3)$ for $r < 1$, else $c_0 + c$ |
| `exponential` | $c_0 + c\,(1 - e^{-r})$, practical range $3a$ |
| `gaussian` | $c_0 + c\,(1 - e^{-r^2})$, practical range $\sqrt{3}\,a$ |
| `matern` | $c_0 + c\left(1 - \frac{2^{1-\nu}}{\Gamma(\nu)} r^\nu K_\nu(r)\right)$, smoothness $\nu$ (`shape`) |
| `linear` | $c_0 + c\,r$ (no sill) |
| `power` | $c_0 + c\,r^\nu$, $0 < \nu < 2$ (no sill) |

### 5.2 Kriging and inverse distance weighting

`OrdinaryKriging(variogram, n_neighbors=None)` and `UniversalKriging(variogram, drift_terms=1, n_neighbors=None)` solve the kriging system [11]

$$
\begin{bmatrix} \Gamma & F \\ F^\top & 0 \end{bmatrix}
\begin{bmatrix} \lambda \\ \mu \end{bmatrix} =
\begin{bmatrix} \gamma_0 \\ f_0 \end{bmatrix}, \qquad
\hat z(x_0) = \lambda^\top z, \qquad
\sigma^2_K(x_0) = \lambda^\top \gamma_0 + \mu^\top f_0 ,
$$

where $\Gamma_{ij} = \gamma(\lvert x_i - x_j \rvert)$, $F$ is a column of ones (ordinary kriging) or the linear ($1, x, y$) or quadratic drift terms (universal kriging). The predictor is exact at data locations. For universal kriging the variogram is fitted to the residuals of an ordinary least squares trend. `predict` returns a `KrigingResult` (predictions and kriging variances), `predict_grid(x, y)` a surface, and `cross_validate(k_folds=None)` leave-one-out or k-fold statistics (RMSE, MAE, bias and the mean standardised squared error `msse`). With `n_neighbors` only the nearest points enter each prediction. `idw(coordinates, values, targets, power=2, n_neighbors=None)` is inverse distance weighting after Shepard [12].

### 5.3 Spatial autocorrelation

With deviations $z_i = x_i - \bar x$, weights $w_{ij}$ and $S_0 = \sum_{ij} w_{ij}$:

$$
I = \frac{n}{S_0} \frac{\sum_{ij} w_{ij} z_i z_j}{\sum_i z_i^2} \;\;\text{(Moran [13])}, \qquad
C = \frac{(n - 1) \sum_{ij} w_{ij} (x_i - x_j)^2}{2 S_0 \sum_i z_i^2} \;\;\text{(Geary [14])} .
$$

`morans_i` and `gearys_c` return the statistic with its expectation, the variance under normality or randomisation (Cliff and Ord [15]), a z-score, a two-sided normal p-value and, with `permutations > 0`, a pseudo p-value. `grid_morans_i` computes Moran's I of a raster with rook or queen contiguity, `getis_ord_gi_star` the local Gi* z-scores of hot and cold spots [16], and `local_morans_i` the local indicators of spatial association [17]. Weights come from `distance_band_weights`, `knn_weights`, `contiguity_weights` and `row_standardize`.

## 6. Spatial analysis

### 6.1 Multi-criteria suitability

`unbihexium.analysis.suitability` follows the three steps of GIS multi-criteria evaluation [18]:

1. standardise the factor layers to $[0, 1]$ with `rescale_linear(values, low, high, increasing)`, `fuzzy_membership(values, a, b, shape)` (linear or sine-squared sigmoidal) or `reclassify(values, breaks, scores)`;
2. derive criterion weights with the Analytic Hierarchy Process, `AHP` [19]: from a reciprocal pairwise comparison matrix $A$ on the 1 to 9 scale, the weights are its normalised principal eigenvector $w$ (or the row geometric means with `method="geometric_mean"`, Crawford and Williams [29]), the consistency index is $CI = (\lambda_{\max} - n)/(n - 1)$ and the consistency ratio $CR = CI / RI$ with the random index $RI$ of matrices of the same size; judgements with $CR < 0.1$ are usually accepted (`is_consistent`);
3. combine the layers by weighted linear combination $S = \sum_k w_k x_k$ and set cells that fail a Boolean constraint to zero: `weighted_overlay(layers, weights, normalize=True, constraints=None)` returns a `SuitabilityResult`.

### 6.2 Cost surfaces

A friction raster gives the cost of crossing one unit of distance in each cell. Moving between neighbouring cells $a$ and $b$ costs $\tfrac{1}{2}(c_a + c_b)$ times the step length (one cell size, or the diagonal). `cost_distance(cost, sources, resolution, connectivity=8)` runs Dijkstra's algorithm [20] and returns the least accumulated cost to the nearest source and that source's index; `least_cost_path(cost, start, end, resolution, connectivity)` returns the cells of the cheapest path and its cost [21]. Cells with NaN, infinite or negative cost are barriers.

### 6.3 Network analysis

`NetworkAnalyzer` holds a weighted graph with node coordinates, for example a road network with lengths or travel times as edge costs. It provides `shortest_path` (Dijkstra [20], or A* [22] with a Euclidean, Manhattan or haversine heuristic), `shortest_costs`, `service_area`, `accessibility`, `closest_facility`, `od_cost_matrix` and `nearest_node`. Edge costs must be non-negative.

### 6.4 Zonal statistics

`zonal_table(values, zones, stats, percentiles, nodata, zone_nodata)`, `zonal_statistics(raster, zones, ...)` and `ZonalStatistics.calculate` compute, for every zone of a zone raster of the same shape, the statistics `count`, `sum`, `mean`, `std` (population), `min`, `max`, `range`, `median`, `majority`, `minority`, `variety` and arbitrary percentiles [23]. Values that are NaN or equal to `nodata` are ignored. `rasterize_zones` burns polygons into a zone raster.

## 7. Accuracy assessment

`unbihexium.metrics` implements the accuracy measures used to validate map products, including the outputs of the families of Section 3 once they are trained. Every error matrix has the **reference** classes in its rows and the **map** classes in its columns.

- `confusion_matrix`, `accuracy_assessment` and `cohen_kappa`: with proportions $p_{ij}$, overall accuracy $OA = \sum_i p_{ii}$, producer's and user's accuracies, F1, IoU, Cohen's $\kappa = (OA - p_e)/(1 - p_e)$ with $p_e = \sum_i p_{i+} p_{+i}$ [24], and the quantity and allocation disagreement of Pontius and Millones, $Q + A = 1 - OA$ [25].
- `stratified_area_estimate`, `estimated_error_matrix` and `sample_allocation`: unbiased class areas and accuracies with standard errors and confidence intervals under stratified random sampling with the map classes as strata, $\hat p_{\cdot j} = \sum_i W_i\, n_{ij} / n_{i\cdot}$ with the mapped area proportions $W_i$ (Olofsson et al. [26]).
- `bias`, `mae`, `rmse`, `ubrmse`, `r_squared` (against the 1:1 line), `pearson_r` and `regression_report` for continuous products.
- `change_detection_metrics`, `transition_matrix` and `transition_summary` for change maps.

## 8. Examples

The examples were executed on 24 September 2026 against the main branch with CPython 3.13 on a CPU. They share one Python session and use small synthetic data.

### 8.1 Terrain derivatives and a viewshed

A cone with a slope of 2 m per 10 m cell ($\arctan 0.2 = 11.3$ degrees) is observed from its western edge; the summit hides the eastern flank.

```python
import numpy as np
from unbihexium.terrain import aspect, hillshade, slope, viewshed

y, x = np.mgrid[0:21, 0:21]
dem = 100.0 - 2.0 * np.hypot(x - 10, y - 10)  # cone, 10 m cells

print(round(float(slope(dem, 10.0)[10, 15]), 2), round(float(aspect(dem, 10.0)[10, 15]), 1))
print(round(float(hillshade(dem, 10.0)[10, 15]), 1))
visible = viewshed(dem, observer=(10, 0), resolution=10.0, observer_height=1.7)
print(int(visible.sum()), bool(visible[10, 10]), bool(visible[10, 20]))
```

```text
11.2 90.0
152.1
59 True False
```

### 8.2 Variogram, kriging and Moran's I

```python
import numpy as np
from unbihexium.geostat import OrdinaryKriging, Variogram, knn_weights, morans_i

rng = np.random.default_rng(42)
xy = rng.uniform(0, 100, size=(80, 2))
z = np.sin(xy[:, 0] / 12.0) + np.cos(xy[:, 1] / 12.0) + rng.normal(0, 0.1, 80)

variogram = Variogram(n_lags=10, max_lag=50.0, model="spherical")
fit = variogram.fit(xy, z)
print(round(fit.nugget, 3), round(fit.sill, 3), round(fit.range_param, 1))

kriging = OrdinaryKriging(variogram).fit(xy, z)
result = kriging.predict(np.array([[50.0, 50.0], xy[0]]))
print(result.predictions.round(3), result.variance.round(4), round(float(z[0]), 3))
print({k: round(v, 3) for k, v in kriging.cross_validate().items()})

moran = morans_i(z, knn_weights(xy, k=6))
print(round(moran.statistic, 3), round(moran.expected, 4), round(moran.z_score, 2))
```

```text
0.0 1.444 58.1
[-1.166 -0.606] [0.1042 0.    ] -0.606
{'rmse': 0.187, 'mae': 0.131, 'bias': 0.001, 'msse': 0.108}
0.75 -0.0127 13.13
```

The prediction at the first sample reproduces the datum with zero variance, as expected of an exact interpolator.

### 8.3 Suitability, cost paths, zonal statistics and routing

```python
import numpy as np
from unbihexium.analysis import (
    AHP, NetworkAnalyzer, least_cost_path, rescale_linear, weighted_overlay, zonal_table,
)

ahp = AHP.from_judgements(
    ["slope", "access", "view"],
    {("access", "slope"): 3, ("access", "view"): 5, ("slope", "view"): 2},
)
weights = ahp.weights_dict()
print({k: round(v, 3) for k, v in weights.items()}, round(ahp.consistency_ratio(), 4))

rng = np.random.default_rng(1)
slope_deg = rng.uniform(0, 30, (4, 4))
road_m = rng.uniform(0, 2000, (4, 4))
view = rng.uniform(0, 1, (4, 4))
layers = [
    rescale_linear(slope_deg, 0, 30, increasing=False),
    rescale_linear(road_m, 0, 2000, increasing=False),
    view,
]
result = weighted_overlay(
    layers, [weights["slope"], weights["access"], weights["view"]], constraints=[slope_deg < 25]
)
print(result.suitability.round(2))

cost = np.ones((5, 5))
cost[1:4, 2] = 10.0  # expensive wall with a gap at both ends
path, total = least_cost_path(cost, (2, 0), (2, 4))
print(path, round(total, 3))

values = np.arange(16, dtype=float).reshape(4, 4)
zones = np.array([[1, 1, 2, 2]] * 4)
print(zonal_table(values, zones, stats=["count", "mean", "max"]))

net = NetworkAnalyzer()
for node, (nx, ny) in enumerate([(0, 0), (1, 0), (2, 0), (1, 1)]):
    net.add_node(node, nx, ny)
net.add_edge(0, 1, 1.0)
net.add_edge(1, 2, 1.0)
net.add_edge(0, 3, 1.5)
net.add_edge(3, 2, 1.5)
print(net.shortest_path(0, 2).nodes, net.service_area(0, 1.5), net.od_cost_matrix([0], [2, 3]))
```

```text
{'slope': 0.23, 'access': 0.648, 'view': 0.122} 0.0032
[[0.83 0.   0.86 0.  ]
 [0.33 0.72 0.46 0.14]
 [0.2  0.53 0.46 0.66]
 [0.88 0.11 0.57 0.87]]
[(2, 0), (3, 1), (4, 2), (3, 3), (2, 4)] 5.657
{'count': {1: 8.0, 2: 8.0}, 'mean': {1: 6.5, 2: 8.5}, 'max': {1: 13.0, 2: 15.0}}
[0, 1, 2] [0, 1, 3] [[2.  1.5]]
```

The two cells with a slope of 25 degrees or more fail the constraint and score 0; the cheapest path goes around the wall with four diagonal steps ($4\sqrt{2} = 5.657$).

### 8.4 Accuracy and area estimation

The error matrix below is illustrative (reference classes in rows, map classes in columns); the mapped areas are 3000 ha of forest and 7000 ha of non-forest.

```python
import numpy as np
from unbihexium.metrics import accuracy_assessment, stratified_area_estimate

counts = np.array([[90, 10], [5, 95]])
acc = accuracy_assessment(counts, classes=["forest", "non_forest"])
print(round(acc.overall_accuracy, 3), round(acc.kappa, 3),
      round(acc.quantity_disagreement, 3), round(acc.allocation_disagreement, 3))

estimate = stratified_area_estimate(counts, mapped_area=[3000.0, 7000.0],
                                    classes=["forest", "non_forest"])
print(estimate.area.round(1), estimate.area_ci.round(1))
```

```text
0.925 0.85 0.025 0.05
[3508.8 6491.2] [417.5 417.5]
```

The map shows 3000 ha of forest, but the sample-based estimate is $10\,000 \times (0.3 \cdot 90/95 + 0.7 \cdot 10/105) = 3508.8$ ha, with a 95 % confidence interval of $\pm 417.5$ ha.

### 8.5 A tiny starter model

The untrained `viewshed_analyzer` ends in a sigmoid and, before training, returns values close to 0.5 everywhere, unrelated to the terrain. Compare it with the exact `viewshed` of Section 8.1.

```python
import numpy as np
from unbihexium.ai import predict

y, x = np.mgrid[0:64, 0:64]
dem = (100.0 - 2.0 * np.hypot(x - 32, y - 32)).astype("float32")
result = predict("viewshed_analyzer_tiny", dem)
print(result.names, result.units, result.values.shape)
print(round(float(result.values.min()), 3), round(float(result.values.max()), 3))
```

```text
['visible_fraction'] ['1'] (1, 64, 64)
0.498 0.503
```

## 9. Limitations and responsible use

### 9.1 Conventions

The key words MUST, SHOULD and MAY in this section are to be interpreted as described in RFC 2119 and RFC 8174 [27], [28] when, and only when, they appear in capitals.

### 9.2 Limitations

- The twelve model families are untrained. Their outputs MUST NOT be used before the model has been trained and validated against independent reference data; Section 7 provides the measures.
- Estimates of property values, economic activity, population or service demand (`economic_spatial_assessor`, `business_valuation`, `resource_allocation`) can affect people. Users MUST read [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md) before deploying such a model, and SHOULD publish the uncertainty of the results.
- The deterministic functions are exact for their inputs, but their results inherit the errors of those inputs (DEM accuracy, sample design, friction values, AHP judgements). Viewsheds on a DEM without buildings and vegetation overestimate visibility; a digital surface model SHOULD be used where obstacles matter.
- Kriging variances assume that the fitted variogram is correct; the cross-validation statistics (`msse` close to 1) SHOULD be checked before the variances are reported.

## 10. Related documents

- [Capability index](index.md), [domain 03](03_indices_flood_water.md) (hydrology) and [domain 04](04_environment_forestry_image_processing.md) (image quality measures).
- [Model catalogue](../model_zoo/model_catalog.md) and [training](../model_zoo/training.md).
- [API reference](../reference/api.md) and [command line reference](../reference/cli.md).
- [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md).

## References

[1] Horn, B. K. P. Hill shading and the reflectance map. Proceedings of the IEEE 69(1), 14-47. 1981. <https://doi.org/10.1109/PROC.1981.11918>

[2] Zevenbergen, L. W., Thorne, C. R. Quantitative analysis of land surface topography. Earth Surface Processes and Landforms 12(1), 47-56. 1987. <https://doi.org/10.1002/esp.3290120107>

[3] Weiss, A. Topographic position and landforms analysis. Poster, ESRI User Conference, San Diego. 2001.

[4] Riley, S. J., DeGloria, S. D., Elliot, R. A terrain ruggedness index that quantifies topographic heterogeneity. Intermountain Journal of Sciences 5(1-4), 23-27. 1999.

[5] Wilson, M. F. J., O'Connell, B., Brown, C., Guinan, J. C., Grehan, A. J. Multiscale terrain analysis of multibeam bathymetry data for habitat mapping on the continental slope. Marine Geodesy 30(1-2), 3-35. 2007. <https://doi.org/10.1080/01490410701295962>

[6] Sappington, J. M., Longshore, K. M., Thompson, D. B. Quantifying landscape ruggedness for animal habitat analysis: a case study using bighorn sheep in the Mojave Desert. Journal of Wildlife Management 71(5), 1419-1426. 2007. <https://doi.org/10.2193/2005-723>

[7] Franklin, W. R., Ray, C. K. Higher isn't necessarily better: visibility algorithms and experiments. Proceedings of the 6th International Symposium on Spatial Data Handling, Edinburgh, 751-770. 1994.

[8] Matheron, G. Principles of geostatistics. Economic Geology 58(8), 1246-1266. 1963. <https://doi.org/10.2113/gsecongeo.58.8.1246>

[9] Cressie, N., Hawkins, D. M. Robust estimation of the variogram: I. Mathematical Geology 12(2), 115-125. 1980. <https://doi.org/10.1007/BF01035243>

[10] Cressie, N. Fitting variogram models by weighted least squares. Mathematical Geology 17(5), 563-586. 1985. <https://doi.org/10.1007/BF01032109>

[11] Cressie, N. Statistics for Spatial Data, revised edition. Wiley, New York. 1993. <https://doi.org/10.1002/9781119115151>

[12] Shepard, D. A two-dimensional interpolation function for irregularly-spaced data. Proceedings of the 23rd ACM National Conference, 517-524. 1968. <https://doi.org/10.1145/800186.810616>

[13] Moran, P. A. P. Notes on continuous stochastic phenomena. Biometrika 37(1-2), 17-23. 1950. <https://doi.org/10.2307/2332142>

[14] Geary, R. C. The contiguity ratio and statistical mapping. The Incorporated Statistician 5(3), 115-145. 1954. <https://doi.org/10.2307/2986645>

[15] Cliff, A. D., Ord, J. K. Spatial Processes: Models and Applications. Pion, London. 1981.

[16] Getis, A., Ord, J. K. The analysis of spatial association by use of distance statistics. Geographical Analysis 24(3), 189-206. 1992. <https://doi.org/10.1111/j.1538-4632.1992.tb00261.x>

[17] Anselin, L. Local indicators of spatial association: LISA. Geographical Analysis 27(2), 93-115. 1995. <https://doi.org/10.1111/j.1538-4632.1995.tb00338.x>

[18] Malczewski, J. GIS-based land-use suitability analysis: a critical overview. Progress in Planning 62(1), 3-65. 2004. <https://doi.org/10.1016/j.progress.2003.09.002>

[19] Saaty, T. L. A scaling method for priorities in hierarchical structures. Journal of Mathematical Psychology 15(3), 234-281. 1977. <https://doi.org/10.1016/0022-2496(77)90033-5>

[20] Dijkstra, E. W. A note on two problems in connexion with graphs. Numerische Mathematik 1(1), 269-271. 1959. <https://doi.org/10.1007/BF01386390>

[21] Douglas, D. H. Least-cost path in GIS using an accumulated cost surface and slopelines. Cartographica 31(3), 37-51. 1994. <https://doi.org/10.3138/D327-0323-2JUT-016M>

[22] Hart, P. E., Nilsson, N. J., Raphael, B. A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics 4(2), 100-107. 1968. <https://doi.org/10.1109/TSSC.1968.300136>

[23] Tomlin, C. D. Geographic Information Systems and Cartographic Modeling. Prentice Hall, Englewood Cliffs NJ. 1990.

[24] Cohen, J. A coefficient of agreement for nominal scales. Educational and Psychological Measurement 20(1), 37-46. 1960. <https://doi.org/10.1177/001316446002000104>

[25] Pontius, R. G., Millones, M. Death to kappa: birth of quantity disagreement and allocation disagreement for accuracy assessment. International Journal of Remote Sensing 32(15), 4407-4429. 2011. <https://doi.org/10.1080/01431161.2011.552923>

[26] Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., Wulder, M. A. Good practices for estimating area and assessing accuracy of land change. Remote Sensing of Environment 148, 42-57. 2014. <https://doi.org/10.1016/j.rse.2014.02.015>

[27] Bradner, S. Key words for use in RFCs to indicate requirement levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[28] Leiba, B. Ambiguity of uppercase vs lowercase in RFC 2119 key words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[29] Crawford, G., Williams, C. A note on the analysis of subjective judgment matrices. Journal of Mathematical Psychology 29(4), 387-405. 1985. <https://doi.org/10.1016/0022-2496(85)90002-1>

<!--
=============================================================================
End of file docs/capabilities/02_tourism_data_processing.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
