<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/capabilities/06_urban_agriculture.md
Title       : Capability Domain 06: Urban Planning and Agriculture
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Capability Domain 06: Urban Planning and Agriculture

| Field | Value |
| --- | --- |
| Document | UBX-DOC-CAP-06-URBAN-AGRICULTURE |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch |

## Abstract

This document describes what Unbihexium provides for urban planning and agriculture: the 26 model families of the `urban` and `agriculture` capability domains and the library functions that serve these applications, namely Sentinel-2 radiometric scaling and scene classification masks, vegetation, moisture and built-up indices, zonal statistics per field or district, cleaning and vectorisation of class maps, accuracy assessment and sample-based area estimation, land change transition analysis and routing on road networks. It is written for analysts who map crops, fields, buildings and urban growth, and for reviewers who need to know which parts are validated algorithms and which are untrained starter models. For every function it states the purpose, inputs, outputs and the implemented formula with its primary source, and every example has been executed against the current code. The domains describe intended applications; they are not validated products.

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

- `urban`: buildings, built-up areas, roads and transport, land use, construction and urban growth, traffic and building heights (10 families);
- `agriculture`: crop parcels and types, field boundaries, crop condition and growth, yield, phenology, grazing, livestock, soil salinity and land suitability for perennial crops and apiaries (16 families).

The registry assigns no library capability to these two domains. The algorithms that serve them are registered in the `indices`, `analysis` and `imaging` domains; Section 4 documents the ones that are relevant here. The seven spectral index families of the model zoo (`ndvi_calculator`, `evi_calculator`, `savi_calculator` and others) belong to the `indices` domain and are described in [03_indices_flood_water.md](03_indices_flood_water.md).

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when they appear in all capitals. Module paths are given relative to the `unbihexium` package, for example `analysis.zonal` for `src/unbihexium/analysis/zonal.py`.

### 1.3 Status of the Models

All 26 families of this document are untrained starter models. Each has a complete, trainable network with the input and output layout of its task and deterministic starter weights derived from the model identifier, but none has been trained on Earth observation data. **Until a family is trained on labelled data for the area and sensor of use, its outputs carry no information about the input image.** The only families of the model zoo that need no training are the 7 spectral index families (28 models) of the `indices` domain, which compute exact published formulas. The registry records the status: the capabilities of the families in this document have maturity `beta` and the tag `requires_training: "true"`.

The library functions of Section 4 are deterministic implementations of published methods and are covered by the test suite in `tests/`. The domain names describe the applications the families are designed for; they are not validated products, and the project publishes no accuracy figures for any model (see [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md), Section 2).

### 1.4 What the Library Does Not Provide

Earlier versions of this document described capabilities that have no implementation. The library does not contain:

- crop growth or yield models based on growing degree days, crop water stress, biomass or harvest indices, or crop calendars;
- urban density or land use indicators beyond what a user computes from the class maps with the functions of Section 4;
- time series processing of image stacks (compositing, gap filling, phenology extraction); the `timeseries_analyzer` family expects six NDVI observations prepared by the user;
- measured accuracies, validation datasets or hardware requirements for any model.

## 2. Domain Inventory

### 2.1 Registry Query

```python
from unbihexium.registry import CapabilityRegistry

for domain in ("urban", "agriculture"):
    capabilities = CapabilityRegistry.by_domain(domain)
    print(domain, len(capabilities))
    for c in capabilities:
        print(f"  {c.capability_id:30s} {c.task:18s} {c.maturity.value:5s} {c.pipeline_id}")
```

```text
urban 10
  building_detector              detection          beta  building_detection
  builtup_detector               detection          beta  None
  construction_monitor           change_detection   beta  None
  digitization_3d                dense_regression   beta  None
  mobility_analyzer              dense_regression   beta  None
  network_analyzer               dense_regression   beta  None
  road_network_analyzer          segmentation       beta  None
  transportation_mapper          segmentation       beta  None
  urban_growth_assessor          change_detection   beta  None
  urban_planner                  segmentation       beta  None
agriculture 16
  beekeeping_suitability         dense_regression   beta  None
  crop_boundary_delineation      segmentation       beta  None
  crop_classifier                segmentation       beta  None
  crop_detector                  detection          beta  None
  crop_growth_monitor            dense_regression   beta  None
  crop_health_assessor           dense_regression   beta  None
  field_surveyor                 dense_regression   beta  None
  grazing_potential              dense_regression   beta  None
  greenhouse_detector            detection          beta  None
  livestock_estimator            scene_regression   beta  None
  perennial_garden_suitability   dense_regression   beta  None
  pivot_inventory                detection          beta  None
  plowed_land_detector           segmentation       beta  None
  salinity_detector              segmentation       beta  None
  timeseries_analyzer            scene_regression   beta  None
  yield_predictor                scene_regression   beta  None
```

Each family has four size variants, `<family>_tiny`, `_base`, `_large` and `_mega`, so the two domains contain 104 models. One family has a registered pipeline: `building_detector` runs through the `building_detection` pipeline (`unbihexium pipeline run building_detection`, Section 6.3). All families run through the generic `unbihexium.ai.predict.predict` function and the `unbihexium predict` command.

### 2.2 Model Families

The two tables below are generated from the model catalogue, `src/unbihexium/zoo/catalog.yaml`, by the script in Section 2.3. The first gives the input channels, the number of stacked acquisitions (dates), the outputs and their units; the second gives the reference data a user needs to train each family, the input data the catalogue suggests, and the range of trainable parameters from the `tiny` to the `mega` variant.

<!-- BEGIN GENERATED TABLES (Section 2.3); do not edit by hand -->

| Family | Name | Domain | Task | Input channels | Dates | Outputs | Units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `beekeeping_suitability` | Beekeeping Suitability | agriculture | dense_regression | 12: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, elevation, slope | 1 | suitability | 1 |
| `crop_boundary_delineation` | Crop Boundary Delineation | agriculture | segmentation | 4: blue, green, red, nir | 1 | background, field_interior, field_boundary | - |
| `crop_classifier` | Crop Type Classifier | agriculture | segmentation | 10: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | 1 | background, wheat, maize, rice, soybean, sunflower, other_crop | - |
| `crop_detector` | Crop Parcel Detector | agriculture | detection | 4: blue, green, red, nir | 1 | crop_parcel, orchard | - |
| `crop_growth_monitor` | Crop Growth Monitor | agriculture | dense_regression | 10: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | 1 | leaf_area_index | m2 m-2 |
| `crop_health_assessor` | Crop Health Assessor | agriculture | dense_regression | 10: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | 1 | health_score | 1 |
| `field_surveyor` | Field Productivity Estimator | agriculture | dense_regression | 10: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | 1 | relative_yield | 1 |
| `grazing_potential` | Grazing Potential | agriculture | dense_regression | 10: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | 1 | forage_biomass | kg ha-1 |
| `greenhouse_detector` | Greenhouse Detector | agriculture | detection | 3: red, green, blue | 1 | greenhouse | - |
| `livestock_estimator` | Livestock Estimator | agriculture | scene_regression | 3: red, green, blue | 1 | head_count | animals |
| `perennial_garden_suitability` | Perennial Garden Suitability | agriculture | dense_regression | 12: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, elevation, slope | 1 | suitability | 1 |
| `pivot_inventory` | Centre Pivot Inventory | agriculture | detection | 4: blue, green, red, nir | 1 | centre_pivot | - |
| `plowed_land_detector` | Ploughed Land Detector | agriculture | segmentation | 4: blue, green, red, nir | 1 | background, ploughed | - |
| `salinity_detector` | Soil Salinity Detector | agriculture | segmentation | 10: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | 1 | non_saline, slight, moderate, strong | - |
| `timeseries_analyzer` | Phenology Estimator | agriculture | scene_regression | 6: ndvi_t1, ndvi_t2, ndvi_t3, ndvi_t4, ndvi_t5, ndvi_t6 | 1 | start_of_season, peak_of_season, end_of_season | day of year, day of year, day of year |
| `yield_predictor` | Yield Predictor | agriculture | scene_regression | 10: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | 1 | yield | t ha-1 |
| `building_detector` | Building Detector | urban | detection | 3: red, green, blue | 1 | building | - |
| `builtup_detector` | Built-up Area Detector | urban | detection | 4: blue, green, red, nir | 1 | built_up_area | - |
| `construction_monitor` | Construction Monitor | urban | change_detection | 3: red, green, blue | 2 | no_change, new_construction, demolition | - |
| `digitization_3d` | Building Height Estimator | urban | dense_regression | 4: red, green, blue, surface_height | 1 | building_height | m |
| `mobility_analyzer` | Mobility Analyser | urban | dense_regression | 10: B02, B03, B04, B08, B11, B12, elevation, slope, population, road_distance | 1 | traffic_intensity | vehicles h-1 |
| `network_analyzer` | Road Network Density Estimator | urban | dense_regression | 4: blue, green, red, nir | 1 | road_density | km km-2 |
| `road_network_analyzer` | Road Network Extractor | urban | segmentation | 3: red, green, blue | 1 | background, road | - |
| `transportation_mapper` | Transportation Mapper | urban | segmentation | 3: red, green, blue | 1 | background, road, railway, airport, port | - |
| `urban_growth_assessor` | Urban Growth Assessor | urban | change_detection | 10: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | 2 | no_change, urban_expansion | - |
| `urban_planner` | Urban Land Use Mapper | urban | segmentation | 3: red, green, blue | 1 | background, residential, commercial, industrial, green_space | - |

| Family | Reference data needed for training | Suggested input data | Parameters, tiny to mega |
| --- | --- | --- | --- |
| `beekeeping_suitability` | Suitability scores from expert assessment or productivity records. | Sentinel-2 L2A; Copernicus DEM | 734,225 to 60,455,745 |
| `crop_boundary_delineation` | Field polygons rasterised into interiors and boundaries. | Sentinel-2 10 m bands; PlanetScope | 733,107 to 60,451,267 |
| `crop_classifier` | Crop type masks, for example from farmer declarations. | Sentinel-2 L2A | 734,039 to 60,454,983 |
| `crop_detector` | Bounding boxes of crop parcels and orchards. | PlanetScope; SPOT; aerial RGBN imagery | 730,758 to 60,414,214 |
| `crop_growth_monitor` | Leaf area index measurements or reference products. | Sentinel-2 L2A | 733,937 to 60,454,593 |
| `crop_health_assessor` | Crop health scores from field inspection. | Sentinel-2 L2A | 733,937 to 60,454,593 |
| `field_surveyor` | Yield monitor data normalised by the field mean. | Sentinel-2 L2A | 733,937 to 60,454,593 |
| `grazing_potential` | Biomass samples from rangeland surveys. | Sentinel-2 L2A | 733,937 to 60,454,593 |
| `greenhouse_detector` | Bounding boxes of greenhouses. | Very high resolution satellite or aerial RGB imagery | 730,581 to 60,413,509 |
| `livestock_estimator` | Livestock counts per chip. | Very high resolution RGB imagery | 494,481 to 37,762,369 |
| `perennial_garden_suitability` | Suitability scores from land evaluation. | Sentinel-2 L2A; Copernicus DEM | 734,225 to 60,455,745 |
| `pivot_inventory` | Bounding boxes of centre pivot fields. | Sentinel-2 10 m bands; Landsat 8/9; PlanetScope | 730,725 to 60,414,085 |
| `plowed_land_detector` | Masks of ploughed fields. | Sentinel-2 10 m bands; PlanetScope | 733,090 to 60,451,202 |
| `salinity_detector` | Salinity class masks from soil sampling. | Sentinel-2 L2A | 733,988 to 60,454,788 |
| `timeseries_analyzer` | Phenology dates from field observation or reference products. | Sentinel-2 L2A time series; Landsat 8/9 time series | 494,979 to 37,764,355 |
| `yield_predictor` | Field-level yield records. | Sentinel-2 L2A | 495,489 to 37,766,401 |
| `building_detector` | Bounding boxes of building footprints. | Very high resolution satellite or aerial RGB imagery; 0.3 to 1 m | 730,581 to 60,413,509 |
| `builtup_detector` | Bounding boxes of built-up areas. | PlanetScope; SPOT; Sentinel-2 10 m bands | 730,725 to 60,414,085 |
| `construction_monitor` | Construction change masks. | Co-registered very high resolution RGB image pairs | 733,395 to 60,452,419 |
| `digitization_3d` | Building heights from LiDAR or cadastral 3D models. | RGB orthophotos with a DSM | 733,073 to 60,451,137 |
| `mobility_analyzer` | Traffic counts interpolated along roads. | Sentinel-2 L2A; road distance and population rasters | 733,937 to 60,454,593 |
| `network_analyzer` | Road density rasters derived from road vector data. | Sentinel-2 10 m bands; PlanetScope | 733,073 to 60,451,137 |
| `road_network_analyzer` | Road masks. | Aerial or satellite RGB imagery | 732,946 to 60,450,626 |
| `transportation_mapper` | Transport infrastructure masks. | Aerial or satellite RGB imagery | 732,997 to 60,450,821 |
| `urban_growth_assessor` | Urban expansion masks. | Sentinel-2 L2A image pairs; Landsat 8/9 image pairs | 735,394 to 60,460,418 |
| `urban_planner` | Urban land use masks. | Aerial or satellite RGB imagery | 732,997 to 60,450,821 |

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


print(family_tables("agriculture", "urban"))
```

## 3. Model Families: Inputs, Outputs and Architecture

### 3.1 Architecture by Task

The network of a family is chosen by its task (`ai.models.networks`). All networks share a residual convolutional encoder with group normalisation [3] whose input width equals the number of catalogue channels:

| Task | Families in this document | Network | Output of the task API |
| --- | --- | --- | --- |
| `detection` | `building_detector`, `builtup_detector`, `crop_detector`, `greenhouse_detector`, `pivot_inventory` | CenterNet-style detector [4] with a U-Net decoder to 1/4 resolution | `DetectionResult`, written as a GeoJSON FeatureCollection |
| `segmentation` | `road_network_analyzer`, `transportation_mapper`, `urban_planner`, `crop_boundary_delineation`, `crop_classifier`, `plowed_land_detector`, `salinity_detector` | U-Net [5] | `SegmentationResult`: class map (`uint8`), written as a single-band GeoTIFF |
| `change_detection` | `construction_monitor`, `urban_growth_assessor` | U-Net on both dates stacked on the channel axis | `SegmentationResult` of the change classes |
| `dense_regression` | `digitization_3d`, `mobility_analyzer`, `network_analyzer`, `beekeeping_suitability`, `crop_growth_monitor`, `crop_health_assessor`, `field_surveyor`, `grazing_potential`, `perennial_garden_suitability` | U-Net with one output channel per target; a scaled sigmoid bounds targets with a catalogue range (for example leaf area index in [0, 8]) | `RegressionResult`: one float32 map per target, written as a multi-band GeoTIFF |
| `scene_regression` | `livestock_estimator`, `timeseries_analyzer`, `yield_predictor` | encoder, global average pooling and a two-layer head | `RegressionResult` with one value per target for the whole chip, written as JSON |

The four variants differ in width and depth (`zoo.get_variant`): `tiny` has 16 base channels and 3 levels, `base` 32 and 4, `large` 48 and 4 with two residual blocks per stage, `mega` 64 and 5 with two blocks per stage. The default inference tile is 256 pixels for `tiny` and `base` and 512 pixels for `large` and `mega`. Scene regression families produce one value per call, so their input SHOULD be a chip that covers one field or one area of interest.

### 3.2 Inputs

Input rasters MUST contain the channels of the family in catalogue order, band first. Sentinel-2 channels (`B02` to `B12`) expect Level-2A surface reflectance, for example scaled with `preprocessing.sentinel2_reflectance` (Section 4.1); generic names (`blue`, `green`, `red`, `nir`) accept any sensor with such bands. Auxiliary channels (`elevation`, `slope`, `population`, `road_distance`, `surface_height`) are rasters the user prepares on the same grid. The `timeseries_analyzer` family takes six NDVI observations of one season (`ndvi_t1` to `ndvi_t6`), which the user computes with `indices.ndvi` and stacks in time order. Two-date families take the second acquisition through `ChangeDetector.predict_pair` or `unbihexium predict --second`. Training records per-band normalisation statistics that inference applies; a different band layout is possible with a customised model, see [docs/model_zoo/training.md](../model_zoo/training.md).

### 3.3 Application Map

| Application | Model families | Library functions (Section 4) |
| --- | --- | --- |
| Crop type and condition mapping | `crop_classifier`, `crop_health_assessor`, `crop_growth_monitor`, `salinity_detector`, `plowed_land_detector` | `preprocessing.sentinel2_reflectance`, `scl_valid_mask`, `apply_mask`; `indices.ndvi`, `evi`, `savi`, `msavi`, `ndre`, `ci_rededge`, `ndmi` |
| Field boundaries and parcels | `crop_boundary_delineation`, `crop_detector`, `pivot_inventory`, `greenhouse_detector` | `postprocessing.sieve`, `majority_filter`, `raster_to_polygons`, `polygons_to_geodataframe` |
| Field and district statistics | `yield_predictor`, `field_surveyor`, `grazing_potential`, `livestock_estimator` | `analysis.zonal_table`, `zonal_statistics`, `rasterize_zones` |
| Seasonal dynamics | `timeseries_analyzer` | `indices.ndvi` per date |
| Land suitability | `perennial_garden_suitability`, `beekeeping_suitability` | `analysis.AHP`, `weighted_overlay` (see [05_asset_management_energy.md](05_asset_management_energy.md), Section 4.1) |
| Buildings, built-up areas and land use | `building_detector`, `builtup_detector`, `urban_planner`, `digitization_3d` | `indices.ndbi`, `bsi`; `metrics.confusion_matrix`, `accuracy_assessment` |
| Urban growth and construction | `urban_growth_assessor`, `construction_monitor` | `metrics.transition_matrix`, `transition_summary`, `stratified_area_estimate` |
| Roads, transport and mobility | `road_network_analyzer`, `transportation_mapper`, `network_analyzer`, `mobility_analyzer` | `analysis.NetworkAnalyzer` |

## 4. Library Functions for the Domain

### 4.1 Sentinel-2 Reflectance and Clear-Sky Masks

`preprocessing.sentinel2_reflectance(dn, offset=None, quantification=10000.0, processing_baseline=None, nodata=0)` converts Level-2A (or Level-1C) digital numbers to reflectance,

$$\rho = \frac{\mathrm{DN} + \mathrm{offset}}{\mathrm{quantification}},$$

where the offset is $-1000$ for products of processing baseline 04.00 and later and 0 before [6]. When neither `offset` nor `processing_baseline` is given, the function assumes a current product and applies $-1000$; pass `processing_baseline` (for example `"05.10"`) from the product metadata to be explicit. Pixels equal to `nodata` become NaN.

`preprocessing.scl_valid_mask(scl, invalid=(0, 1, 3, 8, 9, 10))` returns True for usable pixels of the Level-2A scene classification band; the default excludes no data, saturated or defective pixels, cloud shadows, medium and high probability clouds and thin cirrus (the legend is in `preprocessing.SCL_CLASSES`). `preprocessing.apply_mask(image, mask)` sets the pixels where `mask` is True to NaN in every band, so a clear-sky mask is applied as `apply_mask(image, ~scl_valid_mask(scl))`.

### 4.2 Vegetation, Moisture and Built-Up Indices

Module `indices.spectral` computes indices from reflectance bands in [0, 1] and returns float64 arrays; ratios return NaN where the denominator is zero or an input is NaN. The indices most used in agriculture and urban mapping are:

| Function | Formula | Primary source |
| --- | --- | --- |
| `ndvi(nir, red)` | $(\rho_{NIR} - \rho_{R}) / (\rho_{NIR} + \rho_{R})$ | [7] |
| `savi(nir, red, l=0.5)` | $(1 + L)(\rho_{NIR} - \rho_{R}) / (\rho_{NIR} + \rho_{R} + L)$ | [8] |
| `msavi(nir, red)` | $\big(2\rho_{NIR} + 1 - \sqrt{(2\rho_{NIR} + 1)^2 - 8(\rho_{NIR} - \rho_{R})}\big) / 2$ | [9] |
| `evi(nir, red, blue, g=2.5, c1=6.0, c2=7.5, l=1.0)` | $G(\rho_{NIR} - \rho_{R}) / (\rho_{NIR} + C_1\rho_{R} - C_2\rho_{B} + L)$ | [10] |
| `ndre(nir, red_edge)` | $(\rho_{NIR} - \rho_{RE}) / (\rho_{NIR} + \rho_{RE})$ | [11] |
| `ci_rededge(nir, red_edge)`, `ci_green(nir, green)` | $\rho_{NIR} / \rho_{RE} - 1$, $\rho_{NIR} / \rho_{G} - 1$ | [12] |
| `ndmi(nir, swir1)` | $(\rho_{NIR} - \rho_{SWIR1}) / (\rho_{NIR} + \rho_{SWIR1})$ | [13] |
| `ndbi(swir1, nir)` | $(\rho_{SWIR1} - \rho_{NIR}) / (\rho_{SWIR1} + \rho_{NIR})$ | [14] |
| `bsi(blue, red, nir, swir1)` | $\big((\rho_{SWIR1} + \rho_{R}) - (\rho_{NIR} + \rho_{B})\big) / \big((\rho_{SWIR1} + \rho_{R}) + (\rho_{NIR} + \rho_{B})\big)$ | [15] |
| `kndvi(nir, red)` | $\tanh(\mathrm{NDVI}^2)$ | [16] |

For Sentinel-2, `red_edge` is usually B05, `nir` B08, `swir1` B11. `compute_index(name, **bands)` evaluates an index by name (see `indices.INDEX_FUNCTIONS` for the 25 names it accepts); the complete list of indices and the water, fire and radar indices are documented in [03_indices_flood_water.md](03_indices_flood_water.md).

### 4.3 Statistics per Field or District

Module `analysis.zonal` computes statistics of a value raster within the zones of a zone raster of the same shape [17]. `zonal_table(values, zones, stats=None, percentiles=None, nodata=None, zone_nodata=None)` returns `{statistic: {zone: value}}`; `zonal_statistics(...)` returns one `ZonalResult` record per zone. The statistics are `count`, `sum`, `mean`, `std` (population), `min`, `max`, `range`, `median`, `majority`, `minority` and `variety`, plus arbitrary percentiles (linear interpolation, key `p<q>`). NaN values and values equal to `nodata` are ignored. `rasterize_zones(geometries, shape, transform, ids=None, all_touched=False)` burns field or district polygons into a zone raster.

### 4.4 Cleaning and Vectorising Class Maps

Module `postprocessing` turns class maps into parcels: `sieve(labels, min_size, connectivity=4)` replaces regions smaller than `min_size` pixels with their largest neighbour (GDAL sieve through rasterio), which implements a minimum mapping unit [18]; `majority_filter(labels, size=3)` applies a modal filter; `raster_to_polygons(image, transform=None, connectivity=4, skip_values=(0,))` returns `(shapely polygon, value)` pairs in map coordinates; `polygons_to_geodataframe(image, transform, crs, ...)` returns a GeoDataFrame, which requires the optional `geopandas` dependency.

### 4.5 Accuracy Assessment and Area Estimation

Module `metrics` uses one convention for every error matrix: reference classes in rows, map classes in columns. `confusion_matrix(reference, predicted, labels=None)` builds the matrix, and `accuracy_assessment(matrix, classes)` returns overall accuracy $OA = \sum_i p_{ii}$, producer's and user's accuracies, F1, IoU, Cohen's kappa $(OA - p_e) / (1 - p_e)$ with $p_e = \sum_i p_{i+} p_{+i}$ [19], and quantity and allocation disagreement, which add up to $1 - OA$ [20].

Pixel counting of a classified map is biased by its errors. `stratified_area_estimate(matrix, mapped_area, confidence=0.95, classes=None)` implements the good-practice estimator for a stratified random sample whose strata are the map classes [21], [22]. With $W_i$ the mapped area proportion of map class $i$, $n_{ij}$ the sample count of map class $i$ and reference class $j$ and $n_{i\cdot}$ the sample size of stratum $i$:

$$\hat p_{ij} = W_i \frac{n_{ij}}{n_{i\cdot}}, \qquad \hat A_j = A_{\mathrm{tot}} \sum_i \hat p_{ij}, \qquad S(\hat p_{\cdot j}) = \sqrt{\sum_i W_i^2 \frac{\frac{n_{ij}}{n_{i\cdot}}\left(1 - \frac{n_{ij}}{n_{i\cdot}}\right)}{n_{i\cdot} - 1}},$$

and the confidence interval of the area is $\hat A_j \pm z\, A_{\mathrm{tot}}\, S(\hat p_{\cdot j})$. The result (`AreaEstimate`) also holds user's, producer's and overall accuracy with standard errors. `sample_allocation(mapped_area, expected_users_accuracy, target_se=0.01, rare_minimum=50)` proposes stratum sample sizes for a target standard error of overall accuracy.

### 4.6 Land Change Transitions

`metrics.transition_matrix(before, after, labels=None, nodata=None)` counts the from-to transitions of two class maps (rows: first date, columns: second date), and `metrics.transition_summary(matrix, classes)` reports, per class, persistence, gross gain, gross loss, net change, swap and total change [23]. `metrics.change_map(before, after)` marks the pixels whose class differs. These describe urban growth from two land use maps, whether produced by `urban_planner` after training or by any other classifier.

### 4.7 Routing on Road Networks

`analysis.NetworkAnalyzer` holds a weighted graph with node coordinates (for example a road network with lengths or travel times as edge costs, added with `add_node(node_id, x, y)` and `add_edge(from_node, to_node, cost, bidirectional=True)`). It answers `shortest_path(origin, destination, heuristic=None)` with Dijkstra's algorithm [24], or with A* [25] when a heuristic (`"euclidean"`, `"manhattan"`, `"haversine"`) is given; `service_area(origin, max_cost)` lists the nodes reachable within a budget; `accessibility(origin, threshold)` returns the costs to all nodes; `closest_facility(origin, facilities)` and `od_cost_matrix(origins, destinations)` compute costs between sets of nodes; `nearest_node(x, y)` snaps a coordinate to the network. Edge costs MUST be non-negative. A* returns optimal paths only when the heuristic is expressed in the unit of the edge costs (for travel times, divide distances by the highest speed, or use no heuristic).

## 5. Worked Examples

The examples below were executed in sequence in one Python session (CPython 3.13, NumPy 2, CPU only); each output block is the real output. They use small synthetic arrays so that the results can be checked by hand.

### 5.1 Reflectance, Cloud Masking and NDVI

Two Sentinel-2 bands (B04 and B08) of a 2 x 3 window with processing baseline 05.10; the scene classification marks one cloud (9) and one water pixel (6).

```python
import numpy as np
from unbihexium.indices import ndvi, savi
from unbihexium.preprocessing import apply_mask, scl_valid_mask, sentinel2_reflectance

dn = np.array([
    [[1350, 1400, 1380], [1360, 3500, 1370]],   # B04, red
    [[4800, 5200, 5000], [4900, 3700, 1250]],   # B08, near infrared
])
scl = np.array([[4, 4, 4], [4, 9, 6]])
reflectance = apply_mask(sentinel2_reflectance(dn, processing_baseline="05.10"), ~scl_valid_mask(scl))
red, nir = reflectance
print(np.round(red, 3))
print(np.round(ndvi(nir, red), 3))
print(np.round(savi(nir, red), 3))
```

```text
[[0.035 0.04  0.038]
 [0.036   nan 0.037]]
[[ 0.831  0.826  0.826]
 [ 0.831    nan -0.194]]
[[ 0.566  0.594  0.579]
 [ 0.573    nan -0.032]]
```

The cloud pixel is NaN in every result; the water pixel has a negative NDVI.

### 5.2 Indices of a Crop Canopy and a Roof

```python
from unbihexium.indices import compute_index, evi, ndbi, ndmi, ndre

crop = dict(blue=0.03, green=0.06, red=0.04, red_edge=0.18, nir=0.42, swir1=0.20)
roof = dict(blue=0.12, green=0.13, red=0.15, red_edge=0.17, nir=0.20, swir1=0.28)
print("        NDVI    EVI   NDRE   NDMI   NDBI")
for label, b in (("crop", crop), ("roof", roof)):
    values = (
        ndvi(b["nir"], b["red"]),
        evi(b["nir"], b["red"], b["blue"]),
        ndre(b["nir"], b["red_edge"]),
        ndmi(b["nir"], b["swir1"]),
        ndbi(b["swir1"], b["nir"]),
    )
    print(f"{label:5s}", " ".join(f"{float(v):6.3f}" for v in values))
print(round(float(compute_index("ndvi", nir=0.42, red=0.04)), 3))
```

```text
        NDVI    EVI   NDRE   NDMI   NDBI
crop   0.826  0.662  0.400  0.355 -0.355
roof   0.143  0.104  0.081 -0.167  0.167
0.826
```

### 5.3 Statistics per Field

```python
from unbihexium.analysis import zonal_table

ndvi_map = np.array([
    [0.81, 0.79, 0.30, 0.28],
    [0.83, 0.77, 0.32, np.nan],
    [0.60, 0.62, 0.31, 0.29],
])
fields = np.array([
    [1, 1, 2, 2],
    [1, 1, 2, 2],
    [3, 3, 2, 2],
])
table = zonal_table(ndvi_map, fields, stats=["count", "mean", "std"], percentiles=[50])
for statistic, per_field in table.items():
    print(statistic, {int(field): round(value, 3) for field, value in per_field.items()})
```

```text
count {1: 4.0, 2: 5.0, 3: 2.0}
mean {1: 0.8, 2: 0.3, 3: 0.61}
std {1: 0.022, 2: 0.014, 3: 0.01}
p50 {1: 0.8, 2: 0.3, 3: 0.61}
```

The NaN pixel of field 2 is ignored, so the field has 5 valid pixels.

### 5.4 Minimum Mapping Unit and Parcel Polygons

A class map with one isolated pixel of class 3 inside field 1 is sieved with a minimum size of 2 pixels and vectorised on a 10 m grid.

```python
from unbihexium.postprocessing import raster_to_polygons, sieve

classes = np.array([
    [1, 1, 1, 2, 2, 2],
    [1, 1, 1, 2, 2, 2],
    [1, 3, 1, 2, 2, 2],
    [1, 1, 1, 2, 2, 2],
])
clean = sieve(classes, min_size=2)
print(clean)
transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 6700040.0)
for polygon, value in raster_to_polygons(clean, transform=transform):
    print(int(value), polygon.area, polygon.bounds)
```

```text
[[1 1 1 2 2 2]
 [1 1 1 2 2 2]
 [1 1 1 2 2 2]
 [1 1 1 2 2 2]]
1 1200.0 (500000.0, 6700000.0, 500030.0, 6700040.0)
2 1200.0 (500030.0, 6700000.0, 500060.0, 6700040.0)
```

### 5.5 Accuracy of a Built-Up Map

Twelve reference points compared with a two-class map (1 built-up, 2 other).

```python
from unbihexium.metrics import accuracy_assessment, confusion_matrix

reference = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2])
mapped = np.array([1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2])
matrix = confusion_matrix(reference, mapped, labels=[1, 2])
print(matrix)
assessment = accuracy_assessment(matrix, classes=["built_up", "other"])
print(round(assessment.overall_accuracy, 3), round(assessment.kappa, 3))
print(np.round(assessment.producers_accuracy, 3), np.round(assessment.users_accuracy, 3))
```

```text
[[4 1]
 [1 6]]
0.833 0.657
[0.8   0.857] [0.8   0.857]
```

### 5.6 Area of Urban Expansion from a Stratified Sample

A change map shows 1 200 ha of urban expansion in a 50 000 ha study area. A stratified random sample of 50 points in the expansion stratum and 150 points in the no-change stratum was labelled with reference data (rows: reference, columns: map).

```python
from unbihexium.metrics import stratified_area_estimate

sample = np.array([
    [45, 4],     # reference: urban expansion
    [5, 146],    # reference: no change
])
estimate = stratified_area_estimate(sample, mapped_area=[1200.0, 48800.0], classes=["expansion", "no_change"])
print(np.round(estimate.area, 1), np.round(estimate.area_ci, 1))
print(np.round(estimate.users_accuracy, 3), np.round(estimate.producers_accuracy, 3),
      round(estimate.overall_accuracy, 3))
```

```text
[ 2381.3 47618.7] [1266.4 1266.4]
[0.9   0.973] [0.454 0.997] 0.972
```

Although 90 % of the mapped expansion is correct, the four omitted expansion points in the large no-change stratum double the estimated expansion area to about 2 381 ha, with a 95 % confidence half-width of about 1 266 ha. The producer's accuracy of the expansion class is 0.454. Reporting the pixel count (1 200 ha) would have underestimated the change.

### 5.7 Land Use Transitions

Classes 1 built-up, 2 cropland and 3 grassland on two dates.

```python
from unbihexium.metrics import transition_matrix, transition_summary

before = np.array([[1, 1, 2, 2], [1, 2, 2, 2], [3, 3, 2, 2]])
after = np.array([[1, 1, 1, 2], [1, 1, 2, 2], [3, 1, 2, 2]])
transitions = transition_matrix(before, after, labels=[1, 2, 3])
print(transitions)
print(transition_summary(transitions, classes=["built_up", "cropland", "grassland"])["built_up"])
```

```text
[[3 0 0]
 [2 5 0]
 [1 0 1]]
{'persistence': 3.0, 'gain': 3.0, 'loss': 0.0, 'net_change': 3.0, 'swap': 0.0, 'total_change': 3.0}
```

### 5.8 Travel Times on a Road Network

Five junctions with travel times in minutes on the edges.

```python
from unbihexium.analysis import NetworkAnalyzer

network = NetworkAnalyzer()
for node, (x, y) in {1: (0, 0), 2: (400, 0), 3: (800, 0), 4: (400, 300), 5: (800, 300)}.items():
    network.add_node(node, x, y)
for a, b, minutes in [(1, 2, 2.0), (2, 3, 2.5), (2, 4, 1.5), (4, 5, 2.0), (3, 5, 1.0)]:
    network.add_edge(a, b, minutes)
route = network.shortest_path(1, 5)
print(route.nodes, route.distance)
print(network.service_area(1, max_cost=4.0))
print(network.od_cost_matrix([1, 4], [3, 5]))
```

```text
[1, 2, 4, 5] 5.5
[1, 2, 4]
[[4.5 5.5]
 [3.  2. ]]
```

## 6. Running a Starter Model

### 6.1 Segmentation: Crop Type Classifier

This section runs the smallest variant of `crop_classifier` on a synthetic 10-band chip to show the input and output contract. Because the model is untrained, the class map is arbitrary; the example demonstrates the mechanics only.

```python
from unbihexium.ai.predict import predict, write_result
from unbihexium.core.raster import Raster
from unbihexium.zoo import get_model

print(get_model("crop_classifier_tiny").requires_training)
rng = np.random.default_rng(0)
chip = rng.uniform(0.0, 0.5, size=(10, 64, 64)).astype("float32")
chip_raster = Raster.from_array(chip, crs="EPSG:32634", transform=(10.0, 0.0, 400000.0, 0.0, -10.0, 5000000.0))
chip_raster.to_file("field_chip.tif")
crops = predict("crop_classifier_tiny", chip_raster)
print(type(crops).__name__, crops.mask.shape, crops.mask.dtype)
print(crops.classes)
print(write_result(crops, "crop_types.tif"))
```

```text
True
SegmentationResult (64, 64) uint8
['background', 'wheat', 'maize', 'rice', 'soybean', 'sunflower', 'other_crop']
crop_types.tif
```

`SegmentationResult.class_fractions()` and `class_areas()` summarise a class map, `class_mask(name)` extracts one class, and `to_raster()` returns a georeferenced raster when the input was georeferenced.

### 6.2 Scene Regression: Yield Predictor

A scene regression family returns one value per chip and writes JSON. The command below uses the chip written in Section 6.1. The untrained `yield_predictor_tiny` returns a value near zero, which is not a yield: the output layer starts with small weights, and the value only becomes meaningful after training on field-level yield records.

```bash
unbihexium predict yield_predictor_tiny field_chip.tif field_yield.json
cat field_yield.json
```

```text
Wrote: field_yield.json (yield_predictor_tiny)
{
  "model_id": "yield_predictor_tiny",
  "units": [
    "t ha-1"
  ],
  "values": {
    "yield": -0.0013565990375354886
  }
}
```

### 6.3 The Building Detection Pipeline

The `building_detector` family is also available as a registered pipeline. The command below runs its `tiny` variant on a synthetic RGB image; the untrained detector finds no buildings, and the output is an empty FeatureCollection in the coordinate reference system of the input.

```python
rgb = rng.uniform(0.0, 1.0, size=(3, 64, 64)).astype("float32")
Raster.from_array(rgb, crs="EPSG:32635", transform=(0.5, 0.0, 385000.0, 0.0, -0.5, 6672000.0)).to_file("rgb.tif")
```

```bash
unbihexium pipeline run building_detection -i rgb.tif -o buildings.geojson -p variant=tiny > /dev/null
cat buildings.geojson
```

```text
{
  "type": "FeatureCollection",
  "model_id": "building_detector_tiny",
  "crs": "EPSG:32635",
  "features": []
}
```

After training (`unbihexium train`, see [docs/model_zoo/training.md](../model_zoo/training.md)), the checkpoint path is passed instead of the model identifier. Inference options are described in [docs/model_zoo/inference.md](../model_zoo/inference.md).

## 7. Validation and Responsible Use

Users of this domain:

- MUST NOT present outputs of the starter models as information about crops, yields, buildings or land use; the families of this document produce meaningful results only after training;
- MUST validate a trained model on independent reference data that represent the area, sensor, season and period of use, and SHOULD report class areas with the sample-based estimator of Section 4.5 rather than by pixel counting;
- SHOULD state the minimum mapping unit, the class legend and the date of the imagery with every map;
- SHOULD consider that building-level maps, livestock counts and field-level yields can relate to identifiable owners and are then personal data under data protection law (see [PRIVACY.md](../../PRIVACY.md) and [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md), Section 4).

Urban planning and agriculture are intended uses of the project ([RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md), Section 3). This document is not legal advice.

## References

[1] S. Bradner. Key words for use in RFCs to Indicate Requirement Levels (RFC 2119). 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] B. Leiba. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words (RFC 8174). 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Y. Wu, K. He. Group Normalization. European Conference on Computer Vision. 2018. <https://arxiv.org/abs/1803.08494>

[4] X. Zhou, D. Wang, P. Kraehenbuehl. Objects as Points. 2019. <https://arxiv.org/abs/1904.07850>

[5] O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI. 2015. <https://arxiv.org/abs/1505.04597>

[6] European Space Agency. Sentinel-2 Products Specification Document, S2-PDGS-TAS-DI-PSD (radiometric offset introduced with processing baseline 04.00). 2022.

[7] J. W. Rouse, R. H. Haas, J. A. Schell, D. W. Deering. Monitoring vegetation systems in the Great Plains with ERTS. Third ERTS Symposium, NASA SP-351, 309-317. 1974. <https://ntrs.nasa.gov/citations/19740022614>

[8] A. R. Huete. A soil-adjusted vegetation index (SAVI). Remote Sensing of Environment 25(3), 295-309. 1988. <https://doi.org/10.1016/0034-4257(88)90106-X>

[9] J. Qi, A. Chehbouni, A. R. Huete, Y. H. Kerr, S. Sorooshian. A modified soil adjusted vegetation index. Remote Sensing of Environment 48(2), 119-126. 1994. <https://doi.org/10.1016/0034-4257(94)90134-1>

[10] A. Huete, K. Didan, T. Miura, E. P. Rodriguez, X. Gao, L. G. Ferreira. Overview of the radiometric and biophysical performance of the MODIS vegetation indices. Remote Sensing of Environment 83(1-2), 195-213. 2002. <https://doi.org/10.1016/S0034-4257(02)00096-2>

[11] E. M. Barnes et al. Coincident detection of crop water stress, nitrogen status and canopy density using ground-based multispectral data. Proceedings of the 5th International Conference on Precision Agriculture. 2000.

[12] A. A. Gitelson, Y. Gritz, M. N. Merzlyak. Relationships between leaf chlorophyll content and spectral reflectance and algorithms for non-destructive chlorophyll assessment in higher plant leaves. Journal of Plant Physiology 160(3), 271-282. 2003. <https://doi.org/10.1078/0176-1617-00887>

[13] B.-C. Gao. NDWI: a normalized difference water index for remote sensing of vegetation liquid water from space. Remote Sensing of Environment 58(3), 257-266. 1996. <https://doi.org/10.1016/S0034-4257(96)00067-3>

[14] Y. Zha, J. Gao, S. Ni. Use of normalized difference built-up index in automatically mapping urban areas from TM imagery. International Journal of Remote Sensing 24(3), 583-594. 2003. <https://doi.org/10.1080/01431160304987>

[15] A. Rikimaru, P. S. Roy, S. Miyatake. Tropical forest cover density mapping. Tropical Ecology 43(1), 39-47. 2002.

[16] G. Camps-Valls et al. A unified vegetation index for quantifying the terrestrial biosphere. Science Advances 7(9), eabc7447. 2021. <https://doi.org/10.1126/sciadv.abc7447>

[17] C. D. Tomlin. Geographic Information Systems and Cartographic Modeling. Prentice Hall, Englewood Cliffs NJ. 1990.

[18] S. Saura. Effects of minimum mapping unit on land cover data spatial configuration and composition. International Journal of Remote Sensing 23(22), 4853-4880. 2002. <https://doi.org/10.1080/01431160110114493>

[19] J. Cohen. A coefficient of agreement for nominal scales. Educational and Psychological Measurement 20(1), 37-46. 1960. <https://doi.org/10.1177/001316446002000104>

[20] R. G. Pontius, M. Millones. Death to kappa: birth of quantity disagreement and allocation disagreement for accuracy assessment. International Journal of Remote Sensing 32(15), 4407-4429. 2011. <https://doi.org/10.1080/01431161.2011.552923>

[21] P. Olofsson, G. M. Foody, S. V. Stehman, C. E. Woodcock. Making better use of accuracy data in land change studies: estimating accuracy and area and quantifying uncertainty using stratified estimation. Remote Sensing of Environment 129, 122-131. 2013. <https://doi.org/10.1016/j.rse.2012.10.031>

[22] P. Olofsson, G. M. Foody, M. Herold, S. V. Stehman, C. E. Woodcock, M. A. Wulder. Good practices for estimating area and assessing accuracy of land change. Remote Sensing of Environment 148, 42-57. 2014. <https://doi.org/10.1016/j.rse.2014.02.015>

[23] R. G. Pontius, E. Shusas, M. McEachern. Detecting important categorical land changes while accounting for persistence. Agriculture, Ecosystems and Environment 101(2-3), 251-268. 2004. <https://doi.org/10.1016/j.agee.2003.09.008>

[24] E. W. Dijkstra. A note on two problems in connexion with graphs. Numerische Mathematik 1(1), 269-271. 1959. <https://doi.org/10.1007/BF01386390>

[25] P. E. Hart, N. J. Nilsson, B. Raphael. A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics 4(2), 100-107. 1968. <https://doi.org/10.1109/TSSC.1968.300136>

<!--
=============================================================================
End of file docs/capabilities/06_urban_agriculture.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
