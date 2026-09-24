<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/capabilities/07_risk_defense_neutral.md
Title       : Capability Domain 07: Risk Assessment and Neutral Monitoring
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Capability Domain 07: Risk Assessment and Neutral Monitoring

| Field | Value |
| --- | --- |
| Document | UBX-DOC-607 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch |

## Abstract

This document describes what Unbihexium provides for natural hazard risk assessment, disaster response and the neutral monitoring of objects in very high resolution imagery: the 15 model families of the `risk` and `defense` capability domains, and the library functions that serve these applications, namely burn ratios and burn severity classes, terrain and wetness indices for susceptibility mapping, multi-criteria overlays, kriging of point measurements, hot spot analysis and the evaluation of object detectors. It is written for analysts in civil protection, humanitarian response, insurance and verification, and for reviewers who need to know which parts are validated algorithms and which are untrained starter models. It frames the `defense` domain in civilian and neutral terms, states the uses the project does not support and refers to the [Responsible Use Policy](../../RESPONSIBLE_USE.md) for the binding details. Every example has been executed against the current code. The domains describe intended applications; they are not validated products.

## Contents

- [1. Scope and Status](#1-scope-and-status)
- [2. Domain Inventory](#2-domain-inventory)
- [3. Model Families: Inputs, Outputs and Architecture](#3-model-families-inputs-outputs-and-architecture)
- [4. Library Functions for the Domain](#4-library-functions-for-the-domain)
- [5. Worked Examples](#5-worked-examples)
- [6. Running a Starter Model](#6-running-a-starter-model)
- [7. Responsible Use](#7-responsible-use)
- [References](#references)

## 1. Scope and Status

### 1.1 Purpose

The capability registry (`unbihexium.registry.CapabilityRegistry`) assigns every model family of the model zoo to one capability domain. Two domains are covered here:

- `risk`: susceptibility, vulnerability and risk scores for wildfires, landslides, earthquakes and combined environmental hazards, rapid building damage assessment, and scene-level estimates of disaster impact, needs and preparedness (10 families);
- `defense`: object detectors for neutral monitoring, described in the registry as "Defence and security (neutral monitoring)" (5 families). Their civilian uses are the verification of international commitments, humanitarian monitoring, the protection of critical sites and maritime safety.

The registry assigns no library capability to these two domains. The algorithms that serve them are registered in the `indices`, `analysis` and `ai` domains; Section 4 documents the ones that are relevant here. Flood mapping is covered by the `water` domain in [03_indices_flood_water.md](03_indices_flood_water.md) and by the SAR families in [12_radar_sar.md](12_radar_sar.md).

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when they appear in all capitals. Module paths are given relative to the `unbihexium` package, for example `geostat.kriging` for `src/unbihexium/geostat/kriging.py`.

### 1.3 Status of the Models

All 15 families of this document are untrained starter models. Each has a complete, trainable network with the input and output layout of its task and deterministic starter weights derived from the model identifier, but none has been trained on Earth observation data. **Until a family is trained on labelled data for the area and sensor of use, its outputs carry no information about the input image.** A risk score, damage map or detection produced by a starter model MUST NOT be used for any decision. The only families of the model zoo that need no training are the 7 spectral index families (28 models) of the `indices` domain, and none of them belongs to this document. The registry records the status: the capabilities of these families have maturity `beta` and the tag `requires_training: "true"`.

The project ships no trained weights, no labelled data of military objects and no interface to any weapon or command system. The catalogue contains no model that detects or identifies people, faces or other biometric features.

The library functions of Section 4 are deterministic implementations of published methods and are covered by the test suite in `tests/`. The domain names describe intended applications; they are not validated products, and the project publishes no accuracy figures for any model.

### 1.4 What the Library Does Not Provide

Earlier versions of this document described capabilities that have no implementation. The library does not contain:

- hazard, vulnerability or loss models (multi-hazard risk formulas, damage indices, economic loss or seismic intensity models);
- correlation with vessel tracking data (AIS) or the identification of individual vessels, aircraft or vehicles;
- alerting, emergency response integration or response time targets;
- measured accuracies, validation datasets or hardware requirements for any model.

## 2. Domain Inventory

### 2.1 Registry Query

```python
from unbihexium.registry import CapabilityRegistry

for domain in ("risk", "defense"):
    capabilities = CapabilityRegistry.by_domain(domain)
    print(domain, len(capabilities))
    for c in capabilities:
        print(f"  {c.capability_id:28s} {c.task:18s} {c.maturity.value:5s} {c.tags['requires_training']}")
```

```text
risk 10
  damage_assessor              detection          beta  true
  disaster_management          scene_regression   beta  true
  emergency_disaster_manager   scene_regression   beta  true
  environmental_risk           dense_regression   beta  true
  hazard_vulnerability         dense_regression   beta  true
  insurance_underwriting       scene_regression   beta  true
  landslide_risk               dense_regression   beta  true
  preparedness_manager         scene_regression   beta  true
  seismic_risk                 dense_regression   beta  true
  wildfire_risk                dense_regression   beta  true
defense 5
  border_monitor               detection          beta  true
  maritime_awareness           detection          beta  true
  military_objects_detector    detection          beta  true
  security_monitor             detection          beta  true
  target_detector              detection          beta  true
```

Each family has four size variants, `<family>_tiny`, `_base`, `_large` and `_mega`, so the two domains contain 60 models. None of these families has a registered pipeline; they run through the generic `unbihexium.ai.predict.predict` function and the `unbihexium predict` command (Section 6).

### 2.2 Model Families

The two tables below are generated from the model catalogue, `src/unbihexium/zoo/catalog.yaml`, by the script in Section 2.3. The first gives the input channels, the number of stacked acquisitions (dates), the outputs and their units; the second gives the reference data a user needs to train each family, the input data the catalogue suggests, and the range of trainable parameters from the `tiny` to the `mega` variant.

<!-- BEGIN GENERATED TABLES (Section 2.3); do not edit by hand -->

| Family | Name | Domain | Task | Input channels | Dates | Outputs | Units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `border_monitor` | Border Area Monitor | defense | detection | 3: red, green, blue | 1 | vehicle, vessel, structure | - |
| `maritime_awareness` | Maritime Awareness Detector | defense | detection | 3: red, green, blue | 1 | vessel, offshore_platform | - |
| `military_objects_detector` | Military Objects Detector | defense | detection | 3: red, green, blue | 1 | vehicle, aircraft, vessel, fortified_structure | - |
| `security_monitor` | Security Monitor | defense | detection | 3: red, green, blue | 1 | vehicle, vessel, temporary_structure | - |
| `target_detector` | Target Detector | defense | detection | 3: red, green, blue | 1 | object_of_interest | - |
| `damage_assessor` | Building Damage Assessor | risk | detection | 3: red, green, blue | 1 | damaged_building, destroyed_building | - |
| `disaster_management` | Disaster Impact Estimator | risk | scene_regression | 10: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | 1 | affected_fraction | 1 |
| `emergency_disaster_manager` | Emergency Needs Estimator | risk | scene_regression | 4: blue, green, red, nir | 1 | affected_population, shelter_need | persons, persons |
| `environmental_risk` | Environmental Risk | risk | dense_regression | 10: B02, B03, B04, B08, B11, B12, elevation, slope, population, road_distance | 1 | risk_score | 1 |
| `hazard_vulnerability` | Hazard Vulnerability | risk | dense_regression | 10: B02, B03, B04, B08, B11, B12, elevation, slope, population, road_distance | 1 | vulnerability | 1 |
| `insurance_underwriting` | Insurance Risk Scorer | risk | scene_regression | 10: B02, B03, B04, B08, B11, B12, elevation, slope, population, road_distance | 1 | hazard_score, exposure_score | 1, 1 |
| `landslide_risk` | Landslide Susceptibility | risk | dense_regression | 12: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, elevation, slope | 1 | susceptibility | 1 |
| `preparedness_manager` | Preparedness Scorer | risk | scene_regression | 10: B02, B03, B04, B08, B11, B12, elevation, slope, population, road_distance | 1 | preparedness | 1 |
| `seismic_risk` | Seismic Risk | risk | dense_regression | 10: B02, B03, B04, B08, B11, B12, elevation, slope, population, road_distance | 1 | risk_score | 1 |
| `wildfire_risk` | Wildfire Risk | risk | dense_regression | 12: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, elevation, slope | 1 | susceptibility | 1 |

| Family | Reference data needed for training | Suggested input data | Parameters, tiny to mega |
| --- | --- | --- | --- |
| `border_monitor` | Bounding boxes of vehicles, vessels and structures. | Very high resolution satellite or aerial RGB imagery; 0.3 to 1 m | 730,647 to 60,413,767 |
| `maritime_awareness` | Bounding boxes of vessels and platforms. | Very high resolution satellite RGB imagery | 730,614 to 60,413,638 |
| `military_objects_detector` | Bounding boxes of the four object classes. | Very high resolution satellite RGB imagery | 730,680 to 60,413,896 |
| `security_monitor` | Bounding boxes of the three object classes. | Very high resolution satellite or aerial RGB imagery | 730,647 to 60,413,767 |
| `target_detector` | Bounding boxes of the object of interest. | Very high resolution satellite or aerial RGB imagery | 730,581 to 60,413,509 |
| `damage_assessor` | Bounding boxes of buildings graded as damaged or destroyed. | Very high resolution post-event satellite or aerial RGB imagery | 730,614 to 60,413,638 |
| `disaster_management` | Affected area fractions from damage assessments. | Sentinel-2 L2A | 495,489 to 37,766,401 |
| `emergency_disaster_manager` | Affected population and shelter statistics per chip. | Sentinel-2 10 m bands; very high resolution imagery | 494,658 to 37,763,074 |
| `environmental_risk` | Environmental risk scores. | Sentinel-2 L2A; Copernicus DEM; population and road distance rasters | 733,937 to 60,454,593 |
| `hazard_vulnerability` | Vulnerability scores from exposure models or loss data. | Sentinel-2 L2A; Copernicus DEM; population and road distance rasters | 733,937 to 60,454,593 |
| `insurance_underwriting` | Claims history or risk model scores per site. | Sentinel-2 L2A; Copernicus DEM; population and road distance rasters | 495,522 to 37,766,530 |
| `landslide_risk` | Landslide inventories converted to susceptibility targets. | Sentinel-2 L2A; Copernicus DEM | 734,225 to 60,455,745 |
| `preparedness_manager` | Preparedness scores from surveys. | Sentinel-2 L2A; Copernicus DEM; population and road distance rasters | 495,489 to 37,766,401 |
| `seismic_risk` | Seismic risk scores from hazard and exposure models. | Sentinel-2 L2A; Copernicus DEM; population rasters | 733,937 to 60,454,593 |
| `wildfire_risk` | Burned area history converted to susceptibility targets. | Sentinel-2 L2A; Copernicus DEM | 734,225 to 60,455,745 |

<!-- END GENERATED TABLES -->

### 2.3 Generating the Tables

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


print(family_tables("defense", "risk"))
```

## 3. Model Families: Inputs, Outputs and Architecture

### 3.1 Architecture by Task

The network of a family is chosen by its task (`ai.models.networks`). All networks share a residual convolutional encoder with group normalisation [3] whose input width equals the number of catalogue channels:

| Task | Families in this document | Network | Output of the task API |
| --- | --- | --- | --- |
| `detection` | `damage_assessor` and the five `defense` families | CenterNet-style detector [4] with a U-Net decoder to 1/4 resolution: per-class centre heatmaps, box sizes and offsets, decoded with non-maximum suppression | `DetectionResult`: boxes in pixel and map coordinates, scores and class names, written as a GeoJSON FeatureCollection |
| `dense_regression` | `environmental_risk`, `hazard_vulnerability`, `landslide_risk`, `seismic_risk`, `wildfire_risk` | U-Net [5] with a sigmoid output bounded to [0, 1] | `RegressionResult`: one float32 map per target, written as a multi-band GeoTIFF |
| `scene_regression` | `disaster_management`, `emergency_disaster_manager`, `insurance_underwriting`, `preparedness_manager` | encoder, global average pooling and a two-layer head | `RegressionResult` with one value per target for the whole chip, written as JSON |

The four variants differ in width and depth (`zoo.get_variant`): `tiny` has 16 base channels and 3 levels, `base` 32 and 4, `large` 48 and 4 with two residual blocks per stage, `mega` 64 and 5 with two blocks per stage. The default inference tile is 256 pixels for `tiny` and `base` and 512 pixels for `large` and `mega`; detections in the overlap of tiles are merged by class-aware non-maximum suppression (`ai.inference`).

### 3.2 Inputs

Input rasters MUST contain the channels of the family in catalogue order, band first. The detectors take three-band RGB imagery; the catalogue suggests very high resolution imagery (about 0.3 to 1 m) because the objects are only a few metres in size. The dense risk families take Sentinel-2 Level-2A bands (six or ten) with `elevation` and `slope`, and some also `population` and `road_distance`; these auxiliary rasters are prepared by the user on the same grid (slope with `terrain.slope`, distances with `analysis.cost_distance`). Training records per-band normalisation statistics that inference applies; a different band layout is possible with a customised model, see [docs/model_zoo/training.md](../model_zoo/training.md).

### 3.3 Application Map

| Application | Model families | Library functions (Section 4) |
| --- | --- | --- |
| Wildfire susceptibility and burn severity | `wildfire_risk` | `indices.nbr`, `dnbr`, `rdnbr`, `burn_severity`; `terrain.slope`, `aspect` |
| Landslide susceptibility | `landslide_risk` | `terrain.slope`, `curvature`, `tpi`, `twi`; `analysis.fuzzy_membership`, `weighted_overlay`, `AHP` |
| Exposure, vulnerability and combined risk | `hazard_vulnerability`, `environmental_risk`, `seismic_risk`, `insurance_underwriting`, `preparedness_manager` | `geostat.OrdinaryKriging`, `Variogram`, `idw`; `analysis.zonal_table` |
| Post-event damage and impact | `damage_assessor`, `disaster_management`, `emergency_disaster_manager` | `geostat.getis_ord_gi_star`, `contiguity_weights`; `ai.evaluation.DetectionAccumulator` |
| Neutral monitoring and maritime safety | `border_monitor`, `maritime_awareness`, `military_objects_detector`, `security_monitor`, `target_detector` | `DetectionResult.filter_by_confidence`, `counts_by_class`, `to_geojson`; `ai.evaluation.DetectionAccumulator` |

## 4. Library Functions for the Domain

### 4.1 Burn Ratios and Burn Severity

Module `indices.spectral` implements the normalised burn ratio and its differences [6], [7]:

$$\mathrm{NBR} = \frac{\rho_{NIR} - \rho_{SWIR}}{\rho_{NIR} + \rho_{SWIR}}, \qquad \mathrm{dNBR} = \mathrm{NBR}_{\mathrm{pre}} - \mathrm{NBR}_{\mathrm{post}}, \qquad \mathrm{RdNBR} = \frac{\mathrm{dNBR}}{\sqrt{\lvert \mathrm{NBR}_{\mathrm{pre}} \rvert}},$$

in `nbr(nir, swir)`, `dnbr(nbr_pre, nbr_post)` and `rdnbr(nbr_pre, nbr_post)`, where `swir` is the band near 2.2 um (Sentinel-2 B12, Landsat 8 and 9 band 7). The functions work on unscaled reflectance; Miller and Thode [7] define RdNBR on values multiplied by 1000, so thresholds from that paper apply to 1000 times the result of `rdnbr`. `nbr2(swir1, swir2)` is the ratio of the two short-wave infrared bands.

`burn_severity(dnbr_values)` classifies dNBR with the breaks of Key and Benson [6], `BURN_SEVERITY_BREAKS = (-0.25, -0.1, 0.1, 0.27, 0.44, 0.66)`, into the class indices 0 to 6 whose names are in `BURN_SEVERITY_CLASSES` (enhanced regrowth high and low, unburned, low, moderate-low, moderate-high and high severity); NaN input gives -1. The breaks were derived for particular ecosystems and SHOULD be checked against field data (for example Composite Burn Index plots) before use elsewhere.

### 4.2 Terrain Factors for Susceptibility

Module `terrain` provides the terrain factors that most susceptibility studies combine. The gradient is Horn's 3 x 3 estimator [8] and `slope` returns $\arctan \lVert \nabla z \rVert$ (degrees by default). `curvature(dem, resolution)` returns profile and plan curvature from the quadratic surface of Zevenbergen and Thorne [9], positive on convex forms; `tpi(dem, radius=1)` is the topographic position index $z - \bar z_{\mathrm{neighbourhood}}$ [10], positive on ridges and negative in valleys. The topographic wetness index [11] is

$$\mathrm{TWI} = \ln \frac{a}{\tan \beta}, \qquad a = \frac{(\mathrm{accumulation} + 1)\, \Delta x\, \Delta y}{\sqrt{\Delta x\, \Delta y}},$$

in `twi(dem, resolution, fill=True, min_slope=0.1)`, where the accumulation comes from D8 flow directions on the depression-filled DEM [12], $a$ is the specific catchment area and $\beta$ the local slope, floored at `min_slope` degrees so that flat cells stay finite.

Knowledge-driven susceptibility maps combine such factors with the standardisation and weighting functions of `analysis.suitability` (`fuzzy_membership`, `reclassify`, `AHP`, `weighted_overlay`), which are documented with their formulas in [05_asset_management_energy.md](05_asset_management_energy.md), Section 4.1. Data-driven maps are what the `landslide_risk` and `wildfire_risk` families are designed to learn from event inventories once they are trained.

### 4.3 Interpolation of Point Measurements

Module `geostat` interpolates measurements taken at stations, such as ground motion, rainfall or water levels. `Variogram` fits one of the models spherical, exponential, gaussian, Matern, linear or power to an empirical semivariogram (Matheron or Cressie-Hawkins estimator) by pair-count weighted least squares, or holds given parameters (`Variogram.from_parameters(model, nugget, sill, range_param)`); the exponential model is $\gamma(h) = c_0 + c\,(1 - e^{-h/a})$ for $h > 0$ [13]. `OrdinaryKriging(variogram, n_neighbors=None)` solves, for each target $x_0$, the system

$$\begin{bmatrix} \Gamma & \mathbf{1} \\ \mathbf{1}^{\mathsf{T}} & 0 \end{bmatrix} \begin{bmatrix} \lambda \\ \mu \end{bmatrix} = \begin{bmatrix} \gamma_0 \\ 1 \end{bmatrix}, \qquad \hat z(x_0) = \lambda^{\mathsf{T}} z, \qquad \sigma^2(x_0) = \lambda^{\mathsf{T}} \gamma_0 + \mu,$$

with $\Gamma_{ij} = \gamma(\lVert x_i - x_j \rVert)$ and $(\gamma_0)_i = \gamma(\lVert x_i - x_0 \rVert)$ [14]. `fit(coordinates, values)` fits the variogram when it has no parameters; `predict(targets)` returns a `KrigingResult` with predictions, variances and standard deviations (`std`); `predict_grid(x, y)` works on a grid; `cross_validate(k_folds=None)` performs leave-one-out (or k-fold) cross-validation and reports RMSE, MAE, bias and the mean standardised squared error. `UniversalKriging` adds a linear or quadratic trend, and `idw(coordinates, values, targets, power=2.0)` is inverse distance weighting.

### 4.4 Hot Spots of Damage

`geostat.getis_ord_gi_star(values, weights)` computes the local statistic $G_i^*$ of Getis and Ord [15], [16], a z-score that is large where high values cluster:

$$G_i^* = \frac{\sum_j w_{ij} x_j - \bar x \sum_j w_{ij}}{S \sqrt{\dfrac{n \sum_j w_{ij}^2 - \left(\sum_j w_{ij}\right)^2}{n - 1}}},$$

with $w_{ii} = 1$ (the star), $\bar x$ the mean and $S$ the population standard deviation of the $n$ values; p-values are two-sided normal approximations. `contiguity_weights(shape, contiguity="rook" | "queen")` builds the weights of a grid, and `distance_band_weights` and `knn_weights` those of points. For counts of damaged buildings per grid cell, significant positive $G_i^*$ marks clusters of damage. Because each cell is tested separately, a correction for multiple testing SHOULD be considered when many cells are examined.

### 4.5 Evaluation of Detectors

`ai.evaluation.DetectionAccumulator(num_classes, class_names)` accumulates predicted and reference boxes image by image (`update(boxes, scores, classes, gt_boxes, gt_classes)`, boxes as `x1, y1, x2, y2` in pixels) and `compute()` returns the mean average precision at IoU 0.5 (`map50`), the COCO-style mean over IoU thresholds 0.5 to 0.95 (`map50_95`), precision, recall and the AP per class, using all-point interpolation of the precision-recall curve [17], [18]. This is the measure to report for a trained `damage_assessor` or any detector of the `defense` domain; `unbihexium evaluate` computes it for a labelled dataset split.

`DetectionResult`, the output of every detector, provides `filter_by_confidence(threshold)`, `filter_by_class(*names)`, `counts_by_class()`, `as_arrays()` and `to_geojson()`; boxes are in map coordinates when the input was a georeferenced raster.

## 5. Worked Examples

The examples below were executed in sequence in one Python session (CPython 3.13, NumPy 2, CPU only); each output block is the real output. They use small synthetic arrays so that the results can be checked by hand.

### 5.1 Burn Severity from Two Dates

Four pixels before and after a fire, from unburned (first) to severely burned (last).

```python
import numpy as np
from unbihexium.indices import BURN_SEVERITY_CLASSES, burn_severity, dnbr, nbr, rdnbr

nir_pre, swir_pre = np.array([0.32, 0.30, 0.28, 0.31]), np.array([0.12, 0.12, 0.13, 0.12])
nir_post, swir_post = np.array([0.31, 0.22, 0.14, 0.10]), np.array([0.12, 0.16, 0.22, 0.28])
pre, post = nbr(nir_pre, swir_pre), nbr(nir_post, swir_post)
d = dnbr(pre, post)
print(np.round(d, 3))
print(np.round(rdnbr(pre, post), 3))
print([BURN_SEVERITY_CLASSES[c] for c in burn_severity(d)])
```

```text
[0.013 0.271 0.588 0.916]
[0.019 0.413 0.972 1.377]
['unburned', 'moderate-low severity', 'moderate-high severity', 'high severity']
```

### 5.2 Knowledge-Driven Landslide Susceptibility

A synthetic slope that steepens towards the south (30 m cells) is scored with a sigmoidal membership of slope (0 at 5 degrees, 1 at 35 degrees) and a linear membership of wetness, weighted 0.7 and 0.3. The weights are illustrative; in practice they come from an AHP or from expert judgement.

```python
from unbihexium.analysis import fuzzy_membership, weighted_overlay
from unbihexium.terrain import slope, twi

x = np.arange(8) * 30.0
dem = np.tile(400.0 - 0.25 * x, (6, 1)) + np.linspace(0.0, 10.0, 6)[:, None] ** 2 / 2
slope_deg = slope(dem, resolution=30.0)
wetness = twi(dem, resolution=30.0)
print(np.round(slope_deg[:, 3], 1))
print(np.round(wetness[:, 3], 2))

susceptibility = weighted_overlay(
    [fuzzy_membership(slope_deg, a=5.0, b=35.0), fuzzy_membership(wetness, a=4.0, b=10.0, shape="linear")],
    [0.7, 0.3], normalize=False, names=["slope", "wetness"],
)
print(np.round(susceptibility.suitability[:, 3], 3), susceptibility.weights)
```

```text
[14.5 15.8 20.1 25.3 30.5 33. ]
[6.14 6.96 5.79 5.25 4.62 3.83]
[0.267 0.35  0.443 0.596 0.693 0.693] {'slope': 0.7, 'wetness': 0.3}
```

### 5.3 Kriging of Ground Motion Records

Peak ground acceleration (in g) recorded at five stations on a 10 km square is interpolated at two sites with an exponential variogram of given parameters.

```python
from unbihexium.geostat import OrdinaryKriging, Variogram

stations = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [5.0, 5.0]])
pga = np.array([0.12, 0.18, 0.10, 0.22, 0.16])
variogram = Variogram.from_parameters("exponential", nugget=0.0, sill=0.002, range_param=8.0)
kriging = OrdinaryKriging(variogram=variogram).fit(stations, pga)
result = kriging.predict(np.array([[5.0, 0.0], [2.5, 7.5]]))
print(np.round(result.predictions, 4), np.round(result.std, 4))
```

```text
[0.1529 0.1348] [0.0319 0.0286]
```

### 5.4 Hot Spots of Damaged Buildings

Counts of damaged buildings per grid cell, tested with queen contiguity.

```python
from unbihexium.geostat import contiguity_weights, getis_ord_gi_star

damaged = np.array([
    [0, 1, 0, 0, 0],
    [1, 4, 5, 0, 0],
    [0, 5, 6, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
], dtype=float)
gi = getis_ord_gi_star(damaged.ravel(), contiguity_weights(damaged.shape, contiguity="queen"))
print(np.round(gi.z_score.reshape(damaged.shape), 2))
print((gi.p_value.reshape(damaged.shape) < 0.05).astype(int))
```

```text
[[ 0.59  1.27  1.01 -0.25 -1.18]
 [ 1.27  2.93  2.93  0.68 -1.27]
 [ 1.01  2.93  2.93  0.9  -1.27]
 [-0.25  0.68  0.9   0.   -1.01]
 [-1.18 -1.27 -1.27 -1.01 -0.89]]
[[0 0 0 0 0]
 [0 1 1 0 0]
 [0 1 1 0 0]
 [0 0 0 0 0]
 [0 0 0 0 0]]
```

The four central cells form a significant hot spot at the 5 % level (without correction for multiple testing).

### 5.5 Evaluating a Detector against Reference Boxes

Three predicted and three reference boxes of one image; the third prediction misses the third reference box.

```python
from unbihexium.ai.evaluation import DetectionAccumulator

accumulator = DetectionAccumulator(num_classes=2, class_names=["damaged_building", "destroyed_building"])
accumulator.update(
    boxes=[[10, 10, 30, 30], [50, 50, 70, 70], [80, 10, 95, 25]], scores=[0.9, 0.8, 0.3], classes=[0, 1, 0],
    gt_boxes=[[12, 11, 31, 29], [50, 52, 69, 70], [5, 80, 20, 95]], gt_classes=[0, 1, 0],
)
print(accumulator.compute())
```

```text
{'map50': 0.75, 'map50_95': 0.55, 'precision': 0.75, 'recall': 0.75, 'ap50_per_class': {'damaged_building': 0.5, 'destroyed_building': 1.0}, 'images': 1}
```

## 6. Running a Starter Model

This section runs the smallest variant of `damage_assessor` on a synthetic RGB image with a 0.5 m grid to show the input and output contract. The untrained detector finds nothing at the default score threshold of 0.5; this shows the mechanics only and says nothing about the image.

```python
from unbihexium.ai.predict import predict, write_result
from unbihexium.core.raster import Raster
from unbihexium.zoo import get_model

print(get_model("damage_assessor_tiny").requires_training, get_model("damage_assessor_tiny").spec.outputs)
rng = np.random.default_rng(7)
image = rng.uniform(0.0, 1.0, size=(3, 64, 64)).astype("float32")
raster = Raster.from_array(image, crs="EPSG:32633", transform=(0.5, 0.0, 400000.0, 0.0, -0.5, 5000000.0))
raster.to_file("post_event.tif")
detections = predict("damage_assessor_tiny", raster)
print(detections.count, detections.counts_by_class())
print(write_result(detections, "damage.geojson"))
```

```text
True ('damaged_building', 'destroyed_building')
0 {}
damage.geojson
```

The same run on the command line, with the file written above:

```bash
unbihexium predict damage_assessor_tiny post_event.tif damage_cli.geojson
cat damage_cli.geojson
```

```text
Wrote: damage_cli.geojson (damage_assessor_tiny)
{
  "type": "FeatureCollection",
  "model_id": "damage_assessor_tiny",
  "crs": "EPSG:32633",
  "features": []
}
```

`--threshold` changes the minimum detection score. After training (`unbihexium train`, see [docs/model_zoo/training.md](../model_zoo/training.md)), the checkpoint path is passed instead of the model identifier. Inference options are described in [docs/model_zoo/inference.md](../model_zoo/inference.md).

## 7. Responsible Use

### 7.1 Supported and Unsupported Uses

Natural hazards, disaster management and humanitarian damage assessment, maritime safety and environmental protection, and verification and transparency in accordance with applicable law are intended uses of the project ([RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md), Section 3). The maintainers do not support, will not help with and will not accept contributions for the uses listed in [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md), Section 4, which include the surveillance of individuals without a lawful basis, the selection or engagement of targets for weapons, any use in violation of international humanitarian law or international human rights law, and the breach of export controls or sanctions (see also [COMPLIANCE.md](../../COMPLIANCE.md)).

### 7.2 Dual-Use Components

The five `defense` families, the generic object detectors and the change detection and damage assessment families have both civilian and security applications. Their dual-use considerations are set out in [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md), Section 5. In summary: they are shipped as untrained architectures; what a detector finds depends entirely on the data it is trained on; `target_detector` is a single-class detector for a user-defined object of interest whose name does not imply any military purpose; and those who use these components in a security, defence or law enforcement context MUST comply with international humanitarian law and international human rights law and MUST keep a qualified human responsible for every decision that affects people.

### 7.3 Obligations of Users

Users of this domain:

- MUST NOT present outputs of the starter models, or of models not validated for the purpose, as evidence or as reliable information for decisions about people, property or safety;
- MUST validate a trained model on independent reference data that represent the area, sensor, event type and period of use, and SHOULD report the validation (for example with `DetectionAccumulator` or the regression and classification measures of `unbihexium.metrics`) together with any map or score;
- SHOULD communicate the uncertainty of risk scores and interpolated surfaces (for example the kriging standard deviation) to those who act on them;
- SHOULD consider that damage maps and building-level scores can relate to identifiable households and are then personal data (see [PRIVACY.md](../../PRIVACY.md)).

A trained model used for decisions about insurance, emergency services or critical infrastructure can fall into a high-risk category of the EU Artificial Intelligence Act; the obligations of deployers are summarised in [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md), Section 6. Concerns about misuse are reported as described in [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md), Section 8. This document is not legal advice.

## References

[1] S. Bradner. Key words for use in RFCs to Indicate Requirement Levels (RFC 2119). 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] B. Leiba. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words (RFC 8174). 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Y. Wu, K. He. Group Normalization. European Conference on Computer Vision. 2018. <https://arxiv.org/abs/1803.08494>

[4] X. Zhou, D. Wang, P. Kraehenbuehl. Objects as Points. 2019. <https://arxiv.org/abs/1904.07850>

[5] O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI. 2015. <https://arxiv.org/abs/1505.04597>

[6] C. H. Key, N. C. Benson. Landscape Assessment: ground measure of severity, the Composite Burn Index, and remote sensing of severity, the Normalized Burn Ratio. In FIREMON: Fire Effects Monitoring and Inventory System, USDA Forest Service General Technical Report RMRS-GTR-164-CD. 2006. <https://www.fs.usda.gov/research/treesearch/24066>

[7] J. D. Miller, A. E. Thode. Quantifying burn severity in a heterogeneous landscape with a relative version of the delta Normalized Burn Ratio (dNBR). Remote Sensing of Environment 109(1), 66-80. 2007. <https://doi.org/10.1016/j.rse.2006.12.006>

[8] B. K. P. Horn. Hill shading and the reflectance map. Proceedings of the IEEE 69(1), 14-47. 1981. <https://doi.org/10.1109/PROC.1981.11918>

[9] L. W. Zevenbergen, C. R. Thorne. Quantitative analysis of land surface topography. Earth Surface Processes and Landforms 12(1), 47-56. 1987. <https://doi.org/10.1002/esp.3290120107>

[10] A. Weiss. Topographic position and landforms analysis. Poster presentation, ESRI User Conference, San Diego. 2001.

[11] K. J. Beven, M. J. Kirkby. A physically based, variable contributing area model of basin hydrology. Hydrological Sciences Bulletin 24(1), 43-69. 1979. <https://doi.org/10.1080/02626667909491834>

[12] J. F. O'Callaghan, D. M. Mark. The extraction of drainage networks from digital elevation data. Computer Vision, Graphics, and Image Processing 28(3), 323-344. 1984. <https://doi.org/10.1016/S0734-189X(84)80011-0>

[13] G. Matheron. Principles of geostatistics. Economic Geology 58(8), 1246-1266. 1963. <https://doi.org/10.2113/gsecongeo.58.8.1246>

[14] N. Cressie. Statistics for Spatial Data, revised edition. Wiley, New York. 1993. <https://doi.org/10.1002/9781119115151>

[15] A. Getis, J. K. Ord. The analysis of spatial association by use of distance statistics. Geographical Analysis 24(3), 189-206. 1992. <https://doi.org/10.1111/j.1538-4632.1992.tb00261.x>

[16] J. K. Ord, A. Getis. Local spatial autocorrelation statistics: distributional issues and an application. Geographical Analysis 27(4), 286-306. 1995. <https://doi.org/10.1111/j.1538-4632.1995.tb00912.x>

[17] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman. The PASCAL Visual Object Classes (VOC) Challenge. International Journal of Computer Vision 88, 303-338. 2010. <https://doi.org/10.1007/s11263-009-0275-4>

[18] T.-Y. Lin et al. Microsoft COCO: Common Objects in Context. European Conference on Computer Vision. 2014. <https://arxiv.org/abs/1405.0312>

<!--
=============================================================================
End of file docs/capabilities/07_risk_defense_neutral.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
