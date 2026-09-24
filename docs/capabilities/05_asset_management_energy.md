<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/capabilities/05_asset_management_energy.md
Title       : Capability Domain 05: Asset Management and Energy
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Capability Domain 05: Asset Management and Energy

| Field | Value |
| --- | --- |
| Document | UBX-DOC-CAP-05-ASSETS-ENERGY |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch |

## Abstract

This document describes what Unbihexium provides for infrastructure asset management and energy siting: the twelve model families of the `assets` and `energy` capability domains, and the library functions that serve these applications, namely multi-criteria suitability analysis with the Analytic Hierarchy Process, terrain derivatives, least-cost routing over friction surfaces, surface hydrology for reservoir catchments, Landsat surface temperature scaling, change-map validation and the cleaning of segmentation masks. It is written for analysts who want to apply the library to corridors, pipelines, utilities and renewable energy sites, and for reviewers who need to know which parts are validated algorithms and which are untrained starter models. For every function it states the purpose, inputs, outputs and the implemented formula with its primary source, and every example has been executed against the current code. The domains describe intended applications; they are not validated products.

## Contents

- [1. Scope and Status](#1-scope-and-status)
- [2. Domain Inventory](#2-domain-inventory)
- [3. Model Families: Inputs, Outputs and Architecture](#3-model-families-inputs-outputs-and-architecture)
- [4. Library Functions for the Domain](#4-library-functions-for-the-domain)
- [5. Worked Examples](#5-worked-examples)
- [6. Running a Starter Model](#6-running-a-starter-model)
- [7. Validation and Responsible Use](#7-validation-and-responsible-use)
- [References](#references)

## 1. Scope and Status

### 1.1 Purpose

The capability registry (`unbihexium.registry.CapabilityRegistry`) assigns every model family of the model zoo to one capability domain. Two domains are covered here:

- `assets`: monitoring of linear infrastructure, utilities and their rights of way (7 families);
- `energy`: siting and monitoring of energy installations (5 families).

The registry assigns no library capability to these two domains. The algorithms that serve asset and energy applications are registered in the `analysis`, `indices` and `imaging` domains; Section 4 documents the ones that are relevant here.

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when they appear in all capitals. Module paths are given relative to the `unbihexium` package, for example `analysis.suitability` for `src/unbihexium/analysis/suitability.py`.

### 1.3 Status of the Models

All twelve families of this document are untrained starter models. Each has a complete, trainable network with the input and output layout of its task and deterministic starter weights derived from the model identifier, but none has been trained on Earth observation data. **Until a family is trained on labelled data for the area and sensor of use, its outputs carry no information about the input image.** The only families of the model zoo that need no training are the 7 spectral index families (28 models) of the `indices` domain, and none of them belongs to this document. The registry records the status: the capabilities of these families have maturity `beta` and the tag `requires_training: "true"`.

The library functions of Section 4 are deterministic implementations of published methods and are covered by the test suite in `tests/`. The domain names describe the applications the families are designed for; they are not validated products, and the project publishes no accuracy figures for any model (see [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md), Section 2).

### 1.4 What the Library Does Not Provide

Earlier versions of this document described capabilities that have no implementation. The library does not contain, and this document therefore does not describe:

- solar geometry, irradiance or photovoltaic yield models, capacity factors, wind resource statistics (Weibull fits, shear profiles, power curves) or hydropower equations;
- grid integration, power flow, hosting capacity or storage sizing;
- economic measures (levelised cost of energy or storage, net present value, internal rate of return);
- asset degradation, failure rate or remaining useful life models;
- plug-ins for desktop GIS software;
- measured accuracies, validation datasets or hardware requirements for any model.

Users who need these quantities compute them with specialised tools and MAY use the rasters produced by Unbihexium (for example slope, aspect or a suitability surface) as inputs.

## 2. Domain Inventory

### 2.1 Registry Query

The registry lists the capabilities of a domain. The query below runs without building any model.

```python
from unbihexium.registry import CapabilityRegistry

for domain in ("assets", "energy"):
    capabilities = CapabilityRegistry.by_domain(domain)
    print(domain, len(capabilities))
    for c in capabilities:
        print(f"  {c.capability_id:24s} {c.task:18s} {c.maturity.value:5s} {c.tags['requires_training']}")
```

```text
assets 7
  asset_condition_change   change_detection   beta  true
  corridor_monitor         segmentation       beta  true
  encroachment_detector    detection          beta  true
  infrastructure_monitor   segmentation       beta  true
  leakage_detector         detection          beta  true
  pipeline_route_planner   dense_regression   beta  true
  utility_mapper           segmentation       beta  true
energy 5
  energy_potential         dense_regression   beta  true
  hydroelectric_monitor    dense_regression   beta  true
  onshore_monitor          dense_regression   beta  true
  solar_site_selector      dense_regression   beta  true
  wind_site_selector       dense_regression   beta  true
```

Each family has four size variants, `<family>_tiny`, `_base`, `_large` and `_mega`, so the two domains contain 48 models. None of these families has a registered pipeline; they run through the generic `unbihexium.ai.predict.predict` function and the `unbihexium predict` command (Section 6).

### 2.2 Model Families

The two tables below are generated from the model catalogue, `src/unbihexium/zoo/catalog.yaml`, by the script in Section 2.3. The first gives the input channels, the number of stacked acquisitions (dates), the outputs and their units; the second gives the reference data a user needs to train each family, the input data the catalogue suggests, and the range of trainable parameters from the `tiny` to the `mega` variant.

<!-- BEGIN GENERATED TABLES (Section 2.3); do not edit by hand -->

| Family | Name | Domain | Task | Input channels | Dates | Outputs | Units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `asset_condition_change` | Asset Condition Change | assets | change_detection | 3: red, green, blue | 2 | no_change, deterioration, repair | - |
| `corridor_monitor` | Corridor Monitor | assets | segmentation | 3: red, green, blue | 1 | background, vegetation_encroachment, structure | - |
| `encroachment_detector` | Encroachment Detector | assets | detection | 3: red, green, blue | 1 | structure, vehicle, excavation | - |
| `infrastructure_monitor` | Infrastructure Monitor | assets | segmentation | 3: red, green, blue | 1 | background, road, railway, building, bridge | - |
| `leakage_detector` | Leakage Detector | assets | detection | 5: blue, green, red, nir, swir1 | 1 | leak_signature | - |
| `pipeline_route_planner` | Pipeline Route Cost Surface | assets | dense_regression | 10: B02, B03, B04, B08, B11, B12, elevation, slope, population, road_distance | 1 | cost | cost units per metre |
| `utility_mapper` | Utility Mapper | assets | segmentation | 3: red, green, blue | 1 | background, power_line, pipeline_corridor, substation | - |
| `energy_potential` | Energy Potential | energy | dense_regression | 12: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, elevation, slope | 1 | solar_potential | kWh m-2 a-1 |
| `hydroelectric_monitor` | Hydroelectric Reservoir Level Estimator | energy | dense_regression | 5: blue, green, red, nir, elevation | 1 | water_surface_elevation | m |
| `onshore_monitor` | Surface Temperature Anomaly | energy | dense_regression | 7: SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7, ST_B10 | 1 | temperature_anomaly | K |
| `solar_site_selector` | Solar Site Selector | energy | dense_regression | 12: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, elevation, slope | 1 | suitability | 1 |
| `wind_site_selector` | Wind Site Selector | energy | dense_regression | 12: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, elevation, slope | 1 | suitability | 1 |

| Family | Reference data needed for training | Suggested input data | Parameters, tiny to mega |
| --- | --- | --- | --- |
| `asset_condition_change` | Change masks. | Co-registered very high resolution RGB image pairs | 733,395 to 60,452,419 |
| `corridor_monitor` | Masks of vegetation encroachment and structures. | Aerial or satellite RGB imagery; LiDAR-derived orthophotos | 732,963 to 60,450,691 |
| `encroachment_detector` | Bounding boxes of structures, vehicles and excavations. | Very high resolution satellite or aerial RGB imagery | 730,647 to 60,413,767 |
| `infrastructure_monitor` | Infrastructure masks. | Aerial or satellite RGB imagery | 732,997 to 60,450,821 |
| `leakage_detector` | Bounding boxes of confirmed leak locations. | Multispectral satellite or aerial imagery with a short-wave infrared band | 730,869 to 60,414,661 |
| `pipeline_route_planner` | Cost rasters from engineering estimates. | Sentinel-2 L2A; Copernicus DEM; road distance rasters | 733,937 to 60,454,593 |
| `utility_mapper` | Utility masks. | Aerial RGB imagery | 732,980 to 60,450,756 |
| `energy_potential` | Solar potential rasters from irradiance models. | Sentinel-2 L2A; Copernicus DEM | 734,225 to 60,455,745 |
| `hydroelectric_monitor` | Reservoir level records. | Sentinel-2 10 m bands; Copernicus DEM | 733,217 to 60,451,713 |
| `onshore_monitor` | Surface temperature anomaly rasters. | Landsat 8/9 Collection 2 Level 2 | 733,505 to 60,452,865 |
| `solar_site_selector` | Suitability scores or existing plant locations. | Sentinel-2 L2A; Copernicus DEM | 734,225 to 60,455,745 |
| `wind_site_selector` | Suitability scores or existing turbine locations. | Sentinel-2 L2A; Copernicus DEM | 734,225 to 60,455,745 |

<!-- END GENERATED TABLES -->

### 2.3 Generating the Tables

The tables are produced from the catalogue with the public API of `unbihexium.zoo`; the parameter counts come from the catalogue entries and require no network build.

```python
# GENERATED-TABLES: prints the tables of Section 2.2.
from unbihexium.zoo import get_model, list_specs


def family_tables(*domains):
    specs = sorted((s for d in domains for s in list_specs(domain=d)), key=lambda s: (s.domain, s.family))
    io = ["| Family | Name | Domain | Task | Input channels | Dates | Outputs | Units |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    data = ["| Family | Reference data needed for training | Suggested input data | Parameters, tiny to mega |",
            "| --- | --- | --- | --- |"]
    for s in specs:
        units = ", ".join(s.units) if s.units else "-"
        io.append(f"| `{s.family}` | {s.name} | {s.domain} | {s.task.value} | "
                  f"{len(s.bands)}: {', '.join(s.bands)} | {s.dates} | {', '.join(s.outputs)} | {units} |")
        tiny, mega = (get_model(f"{s.family}_{v}").num_parameters for v in ("tiny", "mega"))
        data.append(f"| `{s.family}` | {s.labels} | {'; '.join(s.sources)} | {tiny:,} to {mega:,} |")
    return "\n".join(io) + "\n\n" + "\n".join(data)


print(family_tables("assets", "energy"))
```

## 3. Model Families: Inputs, Outputs and Architecture

### 3.1 Architecture by Task

The network of a family is chosen by its task (`ai.models.networks`). All networks share a residual convolutional encoder with group normalisation [3] whose input width equals the number of catalogue channels:

| Task | Families in this document | Network | Output of the task API |
| --- | --- | --- | --- |
| `detection` | `encroachment_detector`, `leakage_detector` | CenterNet-style detector [4] on the encoder and a U-Net decoder to 1/4 resolution | `DetectionResult`: boxes, scores and class names, written as a GeoJSON FeatureCollection |
| `segmentation` | `corridor_monitor`, `infrastructure_monitor`, `utility_mapper` | U-Net [5] | `SegmentationResult`: class map (`uint8`) and optional probabilities, written as a single-band GeoTIFF |
| `change_detection` | `asset_condition_change` | U-Net on both dates stacked on the channel axis (6 input channels: `red_t1` to `blue_t2`) | `SegmentationResult` of the change classes |
| `dense_regression` | `pipeline_route_planner`, `energy_potential`, `hydroelectric_monitor`, `onshore_monitor`, `solar_site_selector`, `wind_site_selector` | U-Net with one output channel per target; a scaled sigmoid bounds targets that have a catalogue range (the three suitability scores lie in [0, 1]) | `RegressionResult`: one float32 map per target, written as a multi-band GeoTIFF |

The four variants differ in width and depth (`zoo.get_variant`): `tiny` has 16 base channels and 3 levels, `base` 32 and 4, `large` 48 and 4 with two residual blocks per stage, `mega` 64 and 5 with two blocks per stage. The default inference tile is 256 pixels for `tiny` and `base` and 512 pixels for `large` and `mega`; larger images are cut into overlapping tiles whose outputs are blended (`ai.inference`).

### 3.2 Inputs

Input rasters MUST contain the channels of the family in catalogue order, band first, as `numpy` arrays, `Raster` objects or raster files. Channels named after Sentinel-2 (`B02` to `B12`) or Landsat Collection 2 Level-2 (`SR_B2` to `ST_B10`) bands expect those products; generic names (`red`, `nir`, `swir1`) accept any sensor with such bands. Auxiliary channels (`elevation`, `slope`, `population`, `road_distance`) are rasters the user prepares on the same grid, for example slope with `terrain.slope` (Section 4.2). Before training, an image is used as given apart from NaN, which becomes zero; training records per-band normalisation statistics in the checkpoint, and inference applies them. A different band layout is possible by building a customised model (`ai.models.build_model(..., channel_names=...)`), as described in [docs/model_zoo/training.md](../model_zoo/training.md).

### 3.3 Application Map

| Application | Model families | Library functions (Section 4) |
| --- | --- | --- |
| Solar and wind site screening | `solar_site_selector`, `wind_site_selector`, `energy_potential` | `analysis.AHP`, `rescale_linear`, `fuzzy_membership`, `reclassify`, `weighted_overlay`; `terrain.slope`, `terrain.aspect`; `terrain.viewshed` for visibility studies |
| Pipeline and line routing | `pipeline_route_planner` | `analysis.cost_distance`, `analysis.least_cost_path`, `reclassify` |
| Reservoirs and hydropower | `hydroelectric_monitor` | `terrain.fill_depressions`, `flow_direction_d8`, `flow_accumulation`, `watershed`, `extract_streams`; water indices of [03_indices_flood_water.md](03_indices_flood_water.md) |
| Thermal monitoring of onshore facilities | `onshore_monitor` | `preprocessing.landsat_c2l2_temperature`, `landsat_c2l2_reflectance`, `landsat_qa_mask` |
| Corridor, utility and infrastructure mapping | `corridor_monitor`, `utility_mapper`, `infrastructure_monitor`, `encroachment_detector` | `postprocessing.remove_small_objects`, `connected_components`, `component_statistics`, `raster_to_polygons` |
| Leak indications | `leakage_detector` | vegetation and moisture indices (`indices.ndvi`, `indices.ndmi`) as context layers |
| Condition change of assets | `asset_condition_change` | `ChangeDetector.predict_pair`, `metrics.change_detection_metrics` |

## 4. Library Functions for the Domain

### 4.1 Multi-Criteria Suitability Analysis

Module `analysis.suitability` implements GIS multi-criteria evaluation in three steps [6]: standardise factor layers to [0, 1], derive criterion weights, combine the layers and exclude cells with Boolean constraints.

**Standardisation.** `rescale_linear(values, low, high, increasing=True)` maps $x$ to $t = \min(\max((x - \mathit{low}) / (\mathit{high} - \mathit{low}), 0), 1)$, or $1 - t$ when `increasing=False`; `low` and `high` default to the minimum and maximum of the layer. `fuzzy_membership(values, a, b, shape)` returns $t$ (`shape="linear"`) or $\sin^2(t \pi / 2)$ (`shape="sigmoidal"`) with membership 0 at `a` and 1 at `b`; `a > b` gives a decreasing membership. `reclassify(values, breaks, scores)` assigns `scores[k]` to the values between `breaks[k-1]` and `breaks[k]` (`numpy.digitize` semantics, one more score than breaks). NaN is preserved by all three.

**Analytic Hierarchy Process.** `AHP` takes a reciprocal pairwise comparison matrix $A$ ($a_{ij}$ is the importance of criterion $i$ over $j$ on Saaty's 1 to 9 scale, $a_{ji} = 1 / a_{ij}$), either directly (`set_comparison_matrix`, alias `fit`) or as judgements (`AHP.from_judgements(criteria, {(a, b): value})`). With `method="eigenvector"` (default) the weights are the principal eigenvector of $A$, $A w = \lambda_{\max} w$, normalised to $\sum_i w_i = 1$ [7]; with `method="geometric_mean"` they are the normalised row geometric means [8] and $\lambda_{\max}$ is estimated from $A w$. Consistency is measured by

$$\mathrm{CI} = \frac{\lambda_{\max} - n}{n - 1}, \qquad \mathrm{CR} = \frac{\mathrm{CI}}{\mathrm{RI}_n},$$

with Saaty's random index $\mathrm{RI}_n$ for $n = 1$ to 15 criteria (0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59) [9]. `is_consistent(threshold=0.1)` applies the customary acceptance rule $\mathrm{CR} < 0.1$. The matrix MUST be positive and reciprocal (a tolerance of 2 % admits rounded fractions such as 0.33).

**Weighted linear combination.** `weighted_overlay(layers, weights, normalize=True, constraints=None, names=None)` normalises the weights to sum 1, optionally rescales each layer to [0, 1] (`normalize=True`; switch it off for layers already standardised as above) and returns

$$S(x) = \Big(\sum_i w_i\, f_i(x)\Big) \prod_j c_j(x),$$

where $c_j(x) \in \{0, 1\}$ are the constraint masks (`False` excludes a cell by setting its score to 0). A cell that is NaN in any layer stays NaN. The result is a `SuitabilityResult` with the surface (`suitability`), the weights by name and, when the first layer is a georeferenced `Raster`, a raster of the surface. `WeightedOverlay(rescale=...).calculate(...)` is the array-only variant.

### 4.2 Terrain Derivatives for Siting

Module `terrain.derivatives` computes derivatives of a north-up DEM (first row north, NaN for nodata) in a 3 x 3 window. The gradient is Horn's estimator [10]:

$$\frac{\partial z}{\partial x} = \frac{(z_3 + 2 z_6 + z_9) - (z_1 + 2 z_4 + z_7)}{8\,\Delta x}, \qquad \frac{\partial z}{\partial y} = \frac{(z_1 + 2 z_2 + z_3) - (z_7 + 2 z_8 + z_9)}{8\,\Delta y},$$

with $z_1 \dots z_9$ numbered row by row from the north-west cell, $x$ pointing east and $y$ north. `slope(dem, resolution, units="degrees", z_factor=1.0)` returns $\arctan \lVert \nabla z \rVert$ in degrees, radians or percent ($100 \lVert \nabla z \rVert$). `aspect(dem, resolution)` returns the downslope direction in degrees clockwise from north, $\operatorname{atan2}(-\partial z / \partial x, -\partial z / \partial y) \bmod 360$, and NaN on flat cells. `resolution` is the cell size in elevation units, a number or `(x size, y size)`; the grid is extended by one cell with linear extrapolation, so a plane keeps its slope up to the border. For ground-mounted photovoltaics in the northern hemisphere, south-facing cells have an aspect near 180 degrees.

`terrain.viewshed(dem, observer, resolution, observer_height=1.7, target_height=0.0, max_distance=None, earth_curvature=False, refraction=0.13)` marks the cells visible from an observer cell with the R3 line-of-sight test on a bilinearly interpolated surface [11]; with `earth_curvature=True` elevations are lowered by $(1 - k) d^2 / (2R)$ with $R = 6\,371\,000$ m. It supports visual impact studies of turbines or masts (set `observer_height` to the hub height).

### 4.3 Least-Cost Routing

Module `analysis.network.cost_surface` routes over a friction raster whose cells give the cost of crossing one unit of distance. Every cell is connected to its 4 or 8 neighbours (`connectivity`), and a move from cell $a$ to cell $b$ costs

$$\frac{c_a + c_b}{2}\, \ell_{ab},$$

where $\ell_{ab}$ is the cell size for orthogonal moves and $\sqrt{2}$ times the cell size for diagonal moves (`resolution` is the cell size). Dijkstra's algorithm [12] on this graph gives:

- `cost_distance(cost, sources, resolution=1.0, connectivity=8)`: the least accumulated cost from the nearest source to every cell, and the index of that source;
- `least_cost_path(cost, start, end, resolution=1.0, connectivity=8)`: the list of `(row, col)` cells of the cheapest path and its total cost [13].

Cells with NaN, infinite or negative cost are barriers. A friction surface is usually built by reclassifying slope, land cover and constraint layers (Section 4.1) and adding them; the `pipeline_route_planner` family is designed to learn such a surface from engineering cost estimates once it is trained.

### 4.4 Surface Hydrology for Reservoir Catchments

Module `terrain.hydrology` delineates the catchments that feed reservoirs and run-of-river sites:

- `fill_depressions(dem, epsilon=0.0)`: priority-flood filling of pits [14]; a small `epsilon` adds a gradient across flats so that they drain;
- `flow_direction_d8(dem, resolution)`: steepest descent to one of eight neighbours [15], with drops to diagonal neighbours divided by the diagonal distance; codes are the ESRI powers of two (1 east, 2 south-east, 4 south, 8 south-west, 16 west, 32 north-west, 64 north, 128 north-east) and 0 marks cells without a lower neighbour (pits, flats, nodata and border cells that would drain off the grid);
- `flow_accumulation(flow_dir, weights=None)`: the number (or weighted sum) of upstream cells, excluding the cell itself;
- `watershed(flow_dir, outlet)`: the Boolean mask of cells that drain to `outlet`, including the outlet;
- `extract_streams(accumulation, threshold)`: cells whose accumulation reaches `threshold`.

### 4.5 Landsat Surface Temperature

Module `preprocessing.radiometry` converts Landsat 8 and 9 Collection 2 Level-2 digital numbers with the fixed scale factors of the product [16]:

$$T_s = 0.00341802\,\mathrm{DN} + 149.0 \ \mathrm{K}, \qquad \rho = 2.75 \times 10^{-5}\,\mathrm{DN} - 0.2,$$

in `landsat_c2l2_temperature(dn, nodata=0)` (band `ST_B10`) and `landsat_c2l2_reflectance(dn, nodata=0)` (bands `SR_B1` to `SR_B7`); pixels equal to `nodata` become NaN. `preprocessing.landsat_qa_mask(qa, ...)` flags fill, cloud, cloud shadow, cirrus, snow and dilated cloud from the `QA_PIXEL` band (True means unusable). These are the inputs of the `onshore_monitor` family, whose target is a temperature anomaly in kelvin; a simple non-learned anomaly is the difference from the median of the surroundings (Section 5.5).

### 4.6 Validation of Change Maps

`metrics.change_detection_metrics(reference, predicted, valid=None)` compares a binary change map with a reference, with changed pixels as the positive class, and returns the counts TP, FP, FN and TN and

$$\text{detection rate} = \frac{TP}{TP + FN}, \quad \text{false alarm rate} = \frac{FP}{FP + TN}, \quad \text{missed detection rate} = \frac{FN}{TP + FN},$$

together with precision, F1, IoU, overall accuracy and Cohen's kappa, following the conventions of the change detection literature [17]. It is the measure to report when `asset_condition_change` has been trained (convert its `deterioration` class to a binary map first).

### 4.7 Cleaning and Vectorising Masks

Module `postprocessing` turns class maps of the corridor, utility and infrastructure families into map objects: `remove_small_objects(mask, min_size, connectivity=8)` drops connected regions smaller than `min_size` pixels; `connected_components(image, connectivity=8)` labels regions and returns the label image and their number; `component_statistics(labels, values=None, pixel_area=1.0)` reports pixel count, area, bounding box and centroid per region; `raster_to_polygons(image, transform)` returns `(shapely polygon, value)` pairs in map coordinates. Detections of `encroachment_detector` and `leakage_detector` are `DetectionResult` objects whose `to_geojson()` gives map coordinates when the input was georeferenced.

## 5. Worked Examples

The examples below were executed in sequence in one Python session (CPython 3.13, NumPy 2, CPU only); each output block is the real output. They use small synthetic arrays so that the results can be checked by hand.

### 5.1 Criterion Weights with AHP

Irradiance is judged moderately more important than slope (3) and strongly more important than the distance to the grid (5); slope is slightly more important than grid distance (2).

```python
from unbihexium.analysis import AHP

ahp = AHP.from_judgements(
    ["irradiance", "slope", "grid_distance"],
    {("irradiance", "slope"): 3, ("irradiance", "grid_distance"): 5, ("slope", "grid_distance"): 2},
)
print({name: round(w, 3) for name, w in ahp.weights_dict().items()})
print(round(ahp.lambda_max(), 4), round(ahp.consistency_ratio(), 4), ahp.is_consistent())
```

```text
{'irradiance': 0.648, 'slope': 0.23, 'grid_distance': 0.122}
3.0037 0.0032 True
```

### 5.2 Slope and Aspect of a South-Facing Plane

The elevation falls by 2 m per 10 m cell towards the south (the first row is north).

```python
import numpy as np
from unbihexium.terrain import aspect, slope

rows = np.mgrid[0:6, 0:6][0]
dem = 100.0 - 2.0 * rows
print(round(float(slope(dem, resolution=10.0)[2, 2]), 2), "degrees")
print(float(aspect(dem, resolution=10.0)[2, 2]), "degrees from north")
```

```text
11.31 degrees
180.0 degrees from north
```

The slope is $\arctan(0.2) = 11.31$ degrees and the aspect 180 degrees (south-facing), as expected.

### 5.3 Suitability Surface with a Constraint

Three factor layers are standardised, combined with the AHP weights and masked by an exclusion zone (the centre cell).

```python
from unbihexium.analysis import fuzzy_membership, reclassify, weighted_overlay

irradiation = np.array([[1100.0, 1250.0, 1400.0], [1150.0, 1300.0, 1450.0], [1200.0, 1350.0, 1500.0]])
slope_deg = np.array([[2.0, 4.0, 12.0], [1.0, 6.0, 20.0], [3.0, 8.0, 30.0]])
grid_km = np.array([[0.5, 2.0, 8.0], [1.0, 3.0, 9.0], [1.5, 4.0, 12.0]])

f_irr = fuzzy_membership(irradiation, a=1100.0, b=1500.0, shape="linear")
f_slope = reclassify(slope_deg, breaks=[5.0, 10.0, 15.0], scores=[1.0, 0.6, 0.2, 0.0])
f_grid = fuzzy_membership(grid_km, a=10.0, b=1.0, shape="sigmoidal")
allowed = np.array([[True, True, True], [True, False, True], [True, True, True]])

result = weighted_overlay([f_irr, f_slope, f_grid], ahp.calculate_weights(), normalize=False,
                          constraints=[allowed], names=ahp.criteria)
print(np.round(result.suitability, 3))
print({name: round(w, 3) for name, w in result.weights.items()})
```

```text
[[0.352 0.591 0.546]
 [0.433 0.    0.571]
 [0.513 0.635 0.648]]
{'irradiance': 0.648, 'slope': 0.23, 'grid_distance': 0.122}
```

### 5.4 Least-Cost Route around a High-Friction Zone

A 5 x 7 friction grid with 100 m cells contains a barrier-like zone of cost 50 in column 3; the route from the west edge to the east edge passes below it.

```python
from unbihexium.analysis import cost_distance, least_cost_path

friction = np.ones((5, 7))
friction[1:4, 3] = 50.0
path, total = least_cost_path(friction, start=(2, 0), end=(2, 6), resolution=100.0)
print(path)
print(round(total, 1))
accumulated, nearest = cost_distance(friction, sources=[(2, 0)], resolution=100.0)
print(np.round(accumulated[2], 1))
```

```text
[(2, 0), (2, 1), (3, 2), (4, 3), (4, 4), (3, 5), (2, 6)]
765.7
[   0.   100.   200.  2750.   624.3  665.7  765.7]
```

The detour uses four diagonal and two orthogonal moves at unit cost, $4 \times 100\sqrt{2} + 2 \times 100 = 765.7$, which is cheaper than crossing the zone ($2750$ at column 3 of the direct row).

### 5.5 Catchment of a Reservoir Outlet

```python
from unbihexium.terrain import fill_depressions, flow_accumulation, flow_direction_d8, watershed

valley = np.array([
    [9.0, 8.0, 7.0, 8.0, 9.0],
    [8.0, 6.0, 5.0, 6.0, 8.0],
    [7.0, 5.0, 3.5, 5.0, 7.0],
    [6.0, 4.0, 2.0, 4.0, 6.0],
    [5.0, 3.0, 1.0, 3.0, 5.0],
])
flow_dir = flow_direction_d8(fill_depressions(valley), resolution=30.0)
print(flow_dir)
print(flow_accumulation(flow_dir).astype(int))
basin = watershed(flow_dir, outlet=(3, 2))
print(int(basin.sum()), "cells,", float(basin.sum() * 30.0 * 30.0), "m2")
```

```text
[[ 2  2  4  8  8]
 [ 2  2  4  8  8]
 [ 2  2  4  8  8]
 [ 2  2  4  8  8]
 [ 1  1  0 16 16]]
[[ 0  0  0  0  0]
 [ 0  1  3  1  0]
 [ 0  1  8  1  0]
 [ 0  1 13  1  0]
 [ 0  2 24  2  0]]
14 cells, 12600.0 m2
```

The lowest border cell (row 4, column 2) has code 0 because it would drain off the grid; it collects the other 24 cells. The catchment of the outlet in row 3 consists of the outlet and its 13 upstream cells.

### 5.6 Surface Temperature Anomaly from Landsat Level-2

```python
from unbihexium.preprocessing import landsat_c2l2_temperature

st_b10 = np.array([[43000, 43500, 44000], [43200, 47000, 43800], [0, 43300, 43600]])
kelvin = landsat_c2l2_temperature(st_b10)
print(np.round(kelvin, 2))
print(np.round(kelvin - np.nanmedian(kelvin), 2))
```

```text
[[295.97 297.68 299.39]
 [296.66 309.65 298.71]
 [   nan 297.   298.03]]
[[-1.88 -0.17  1.54]
 [-1.2  11.79  0.85]
 [  nan -0.85  0.17]]
```

The fill value 0 becomes NaN, and the centre pixel is about 11.8 K warmer than the median of the window. This non-learned anomaly is a baseline against which a trained `onshore_monitor` model can be compared.

### 5.7 Validating a Change Map and Summarising Regions

```python
from unbihexium.metrics import change_detection_metrics
from unbihexium.postprocessing import component_statistics, connected_components, remove_small_objects

reference = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 0]])
predicted = np.array([[0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
scores = change_detection_metrics(reference, predicted)
print({k: round(v, 3) for k, v in scores.items()})

mask = np.zeros((6, 8), dtype=bool)
mask[0:2, 0:3] = True   # 6 pixels
mask[4:6, 5:8] = True   # 6 pixels
mask[3, 3] = True       # isolated pixel
labels, count = connected_components(remove_small_objects(mask, min_size=2))
print(count)
for region in component_statistics(labels, pixel_area=0.25):
    print(region)
```

```text
{'tp': 4.0, 'fp': 1.0, 'fn': 1.0, 'tn': 10.0, 'detection_rate': 0.8, 'false_alarm_rate': 0.091, 'missed_detection_rate': 0.2, 'precision': 0.8, 'f1': 0.8, 'iou': 0.667, 'overall_accuracy': 0.875, 'kappa': 0.709}
2
{'label': 1, 'pixels': 6, 'area': 1.5, 'bbox': (0, 0, 2, 3), 'centroid': (0.5, 1.0)}
{'label': 2, 'pixels': 6, 'area': 1.5, 'bbox': (4, 5, 6, 8), 'centroid': (4.5, 6.0)}
```

## 6. Running a Starter Model

This section runs the smallest variant of `solar_site_selector` on a synthetic 12-channel stack to show the input and output contract. Because the model is untrained, its output is close to 0.5 everywhere (the midpoint of the sigmoid that bounds the score) and carries no information; the example demonstrates the mechanics only.

```python
from unbihexium.ai.predict import predict, write_result
from unbihexium.core.raster import Raster
from unbihexium.zoo import get_model

entry = get_model("solar_site_selector_tiny")
print(entry.requires_training, entry.spec.bands, entry.spec.outputs, entry.spec.value_range)

rng = np.random.default_rng(42)
stack = rng.uniform(0.0, 0.4, size=(12, 64, 64)).astype("float32")
raster = Raster.from_array(stack, crs="EPSG:32635", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6700000.0))
raster.to_file("site_stack.tif")

result = predict("solar_site_selector_tiny", raster)
print(result.output("suitability").shape)
print({k: round(v, 3) for k, v in result.summary()["suitability"].items()})
print(write_result(result, "suitability.tif"))
```

```text
True ('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'elevation', 'slope') ('suitability',) (0.0, 1.0)
(64, 64)
{'mean': 0.5, 'min': 0.497, 'max': 0.502, 'std': 0.001}
suitability.tif
```

The same run on the command line, with the file written above:

```bash
unbihexium predict solar_site_selector_tiny site_stack.tif suitability_cli.tif
```

```text
Wrote: suitability_cli.tif (solar_site_selector_tiny)
```

`--variant` selects a variant when a family name is given, `--tile-size` and `--overlap` control tiling, `--backend` chooses PyTorch or ONNX Runtime, and `--second` supplies the second date of a change detection family such as `asset_condition_change`. After training (`unbihexium train`, see [docs/model_zoo/training.md](../model_zoo/training.md)), the checkpoint path is passed instead of the model identifier. Inference options are described in [docs/model_zoo/inference.md](../model_zoo/inference.md).

## 7. Validation and Responsible Use

Users of this domain:

- MUST NOT present outputs of the starter models as information about assets, sites or risks; the families of this document produce meaningful results only after training;
- MUST validate a trained model on independent reference data that represent the area, sensor and period of use, and SHOULD report the method and the results (for example with `metrics.change_detection_metrics`, `metrics.regression_report` or the accuracy and area estimators of `unbihexium.metrics`) together with any map used for decisions;
- SHOULD document the criteria, standardisation functions, weights and consistency ratio of every suitability analysis, because the result depends on these expert choices as much as on the data;
- SHOULD treat maps of critical infrastructure (pipelines, substations, power lines) as potentially sensitive information and follow the rules that apply to their publication.

Asset and energy applications fall under the intended uses of [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md) (Section 3). A system that uses a trained model as a safety component of critical infrastructure can be a high-risk AI system under the EU Artificial Intelligence Act; the obligations of deployers are summarised in [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md) (Section 6) and [COMPLIANCE.md](../../COMPLIANCE.md). This document is not legal advice.

## References

[1] S. Bradner. Key words for use in RFCs to Indicate Requirement Levels (RFC 2119). 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] B. Leiba. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words (RFC 8174). 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Y. Wu, K. He. Group Normalization. European Conference on Computer Vision. 2018. <https://arxiv.org/abs/1803.08494>

[4] X. Zhou, D. Wang, P. Kraehenbuehl. Objects as Points. 2019. <https://arxiv.org/abs/1904.07850>

[5] O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI. 2015. <https://arxiv.org/abs/1505.04597>

[6] J. Malczewski. GIS-based land-use suitability analysis: a critical overview. Progress in Planning 62(1), 3-65. 2004. <https://doi.org/10.1016/j.progress.2003.09.002>

[7] T. L. Saaty. A scaling method for priorities in hierarchical structures. Journal of Mathematical Psychology 15(3), 234-281. 1977. <https://doi.org/10.1016/0022-2496(77)90033-5>

[8] G. Crawford, C. Williams. A note on the analysis of subjective judgment matrices. Journal of Mathematical Psychology 29(4), 387-405. 1985. <https://doi.org/10.1016/0022-2496(85)90002-1>

[9] T. L. Saaty. The Analytic Hierarchy Process. McGraw-Hill, New York. 1980.

[10] B. K. P. Horn. Hill shading and the reflectance map. Proceedings of the IEEE 69(1), 14-47. 1981. <https://doi.org/10.1109/PROC.1981.11918>

[11] W. R. Franklin, C. K. Ray. Higher isn't necessarily better: visibility algorithms and experiments. Proceedings of the 6th International Symposium on Spatial Data Handling, Edinburgh, 751-770. 1994.

[12] E. W. Dijkstra. A note on two problems in connexion with graphs. Numerische Mathematik 1(1), 269-271. 1959. <https://doi.org/10.1007/BF01386390>

[13] D. H. Douglas. Least-cost path in GIS using an accumulated cost surface and slopelines. Cartographica 31(3), 37-51. 1994. <https://doi.org/10.3138/D327-0323-2JUT-016M>

[14] R. Barnes, C. Lehman, D. Mulla. Priority-flood: an optimal depression-filling and watershed-labeling algorithm for digital elevation models. Computers and Geosciences 62, 117-127. 2014. <https://doi.org/10.1016/j.cageo.2013.04.024>

[15] J. F. O'Callaghan, D. M. Mark. The extraction of drainage networks from digital elevation data. Computer Vision, Graphics, and Image Processing 28(3), 323-344. 1984. <https://doi.org/10.1016/S0734-189X(84)80011-0>

[16] U.S. Geological Survey. Landsat Collection 2 Level-2 Science Products. 2020. <https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products>

[17] L. Bruzzone, D. F. Prieto. Automatic analysis of the difference image for unsupervised change detection. IEEE Transactions on Geoscience and Remote Sensing 38(3), 1171-1182. 2000. <https://doi.org/10.1109/36.843009>

<!--
=============================================================================
End of file docs/capabilities/05_asset_management_energy.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
