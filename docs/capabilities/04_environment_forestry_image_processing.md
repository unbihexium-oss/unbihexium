<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/capabilities/04_environment_forestry_image_processing.md
Title       : Capability Domain 04: Environment, Forestry and Image Processing
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Capability Domain 04: Environment, Forestry and Image Processing

| Field | Value |
| --- | --- |
| Document | UBX-DOC-604 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 2.0.0 and the main branch (model catalogue 2.0.0) |

## Abstract

This document describes capability domain 04, "environment, forestry and image processing". It covers the model families of the catalogue domains `environment`, `forestry` and `imaging`, and the library capabilities that the registry files under the domain `imaging`: radiometric preprocessing, quality masks, resampling and pansharpening (`unbihexium.preprocessing`), postprocessing of predictions (`unbihexium.postprocessing`) and visualisation (`unbihexium.visualization`), together with the image quality measures of `unbihexium.metrics` that validate image products. It is written for users who prepare imagery, map land cover, forests and environmental condition, or produce derived imagery, for contributors and for reviewers. Formulas are given as implemented, with their primary sources; the 35 model families are listed in tables generated from the catalogue; all examples were executed. All 35 families are untrained starter models, and the domain describes intended applications, not validated products.

## Contents

1. [Scope and status](#1-scope-and-status)
2. [Place in the registry and the catalogue](#2-place-in-the-registry-and-the-catalogue)
3. [Model families](#3-model-families)
4. [Preprocessing](#4-preprocessing)
5. [Postprocessing](#5-postprocessing)
6. [Visualisation](#6-visualisation)
7. [Image quality measures](#7-image-quality-measures)
8. [Examples](#8-examples)
9. [Limitations and responsible use](#9-limitations-and-responsible-use)
10. [Related documents](#10-related-documents)
11. [References](#references)

## 1. Scope and status

### 1.1 What the domain covers

- **Environment and forestry.** Fifteen learned families for land cover (the eleven classes of ESA WorldCover), land degradation, desertification, erosion, drought, habitat suitability, biomass, ecological condition, land surface temperature, active fires, change in protected areas, forest type, deforestation, canopy cover and canopy height.
- **Image processing.** Twenty learned families that produce or improve imagery and elevation: cloud and shadow masks, map digitisation, thematic and landform mapping, elevation models from stereo, surface-to-terrain conversion, object heights, co-registration, orthorectification, mosaicking, denoising, pansharpening, tile normalisation and fourfold super-resolution.
- **Deterministic image processing.** Conversion of Sentinel-2 and Landsat 8/9 digital numbers to reflectance and temperature, image-based haze removal, Sentinel-2 and Landsat quality masks, contrast stretches and histogram matching, resampling and aggregation with missing values, three pansharpening methods, the cleaning, vectorisation and tiling of predictions, RGB composites, colour maps, relief shading and PNG quicklooks, and full-reference image quality measures.

The burn and moisture indices used in environmental work (NBR, dNBR, MSI, NDMI) are described in [domain 03](03_indices_flood_water.md).

### 1.2 Status

The functions of Sections 4 to 7 are deterministic and covered by the unit tests in `tests/`. Every one of the 35 model families is an **untrained starter model**: a complete, trainable network with deterministic initial weights, not fitted to any Earth observation data. The predictions of these models are meaningless until the model has been trained on reference data (Section 8.5 shows an example). Several image processing families have a deterministic counterpart in this document, for example `pansharpening` (Section 4.5) and `cloud_mask` (the quality masks of Section 4.2). The deterministic function is exact for its inputs; the family is meant to learn a better result from training data. The only models of the zoo that need no training are the spectral index families of [domain 03](03_indices_flood_water.md).

### 1.3 Changes from the previous version

Version 1 of this document listed fourteen "production" models (including `watershed_manager` and `environmental_risk`, which belong to the water and risk domains), accuracy tables against named datasets, carbon stock and allometric equations, forest fragmentation and Shannon diversity indices, a Siamese change detection network and hardware requirements. There is no code for carbon stocks, allometry, fragmentation or diversity indices; the change detectors are early-fusion U-Nets; no accuracy or hardware figure was measured. These parts were removed. The imaging families, previously scattered over several documents, are listed here, in the document of their catalogue domain.

## 2. Place in the registry and the catalogue

The enumeration `unbihexium.registry.CapabilityDomain` has the members `ENVIRONMENT = "environment"`, `FORESTRY = "forestry"` and `IMAGING = "imaging"`.

| Registry domain | Model capabilities (families) | Library capabilities | Total |
| --- | --- | --- | --- |
| `environment` | 11, maturity `beta` | none | 11 |
| `forestry` | 4, maturity `beta` | none | 4 |
| `imaging` | 20, maturity `beta` | `image_preprocessing` (`unbihexium.preprocessing`), `prediction_postprocessing` (`unbihexium.postprocessing`), `visualization` (`unbihexium.visualization`), all `stable` | 23 |

The family `super_resolution` runs in the registered pipeline `super_resolution`. Task classes of `unbihexium.ai` preset families of this domain: `LandCoverClassifier` (`lulc_classifier`), `CloudMasker` (`cloud_mask`), `FireDetector` (`fire_monitor`), `LandSurfaceTemperature` (`land_surface_temperature`), `TreeHeightEstimator` (`tree_height_estimator`), `Enhancer` (`pansharpening`) and `SuperResolution` (`super_resolution`). The image quality measures belong to the library capability `accuracy_metrics` of the domain `analysis` ([domain 02](02_tourism_data_processing.md)).

The capability documents [08](08_value_added_imagery.md), [10](10_satellite_imagery_features.md) and [11](11_resolution_metadata_qa.md) have no catalogue domain of their own; they discuss products built from imaging families and functions of this document.

## 3. Model families

### 3.1 Inventory

The tables were generated from the catalogue and the model registry of the installed package with the script of [index.md, Section 4](index.md#4-regenerating-the-family-tables), run with the arguments `environment forestry imaging`. Input bands are listed in channel order; `(x 2 dates)` means that two co-registered acquisitions are stacked on the channel axis; units are given in brackets.

| Family | Domain | Task | Input bands | Outputs | Parameters (tiny / base / large / mega) |
| --- | --- | --- | --- | --- | --- |
| `fire_monitor` | environment | detection | B12, B8A, B04 | active_fire | 730,581 / 7,049,125 / 22,038,197 / 60,413,509 |
| `desertification_monitor` | environment | segmentation | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | not_degraded, low, moderate, severe | 733,988 / 7,060,548 / 22,062,244 / 60,454,788 |
| `erosion_detector` | environment | segmentation | blue, green, red, nir, elevation | stable, sheet_erosion, gully | 733,251 / 7,059,075 / 22,060,035 / 60,451,843 |
| `land_degradation_detector` | environment | segmentation | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | not_degraded, degraded | 733,954 / 7,060,482 / 22,062,146 / 60,454,658 |
| `lulc_classifier` | environment | segmentation | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | tree_cover, shrubland, grassland, cropland, built_up, bare_sparse, snow_ice, water, herbaceous_wetland, mangroves, moss_lichen | 734,107 / 7,060,779 / 22,062,587 / 60,455,243 |
| `protected_area_change_detector` | environment | change_detection | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 (x 2 dates) | no_change, vegetation_loss, new_structure, other_change | 735,428 / 7,063,428 / 22,066,564 / 60,460,548 |
| `drought_monitor` | environment | dense_regression | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | drought_severity [1] | 733,937 / 7,060,449 / 22,062,097 / 60,454,593 |
| `wildlife_habitat_analyzer` | environment | dense_regression | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, elevation, slope | habitat_suitability [1] | 734,225 / 7,061,025 / 22,062,961 / 60,455,745 |
| `natural_resources_monitor` | environment | dense_regression | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, VV, VH | above_ground_biomass [Mg ha-1] | 734,225 / 7,061,025 / 22,062,961 / 60,455,745 |
| `environmental_monitor` | environment | dense_regression | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | condition [1] | 733,937 / 7,060,449 / 22,062,097 / 60,454,593 |
| `land_surface_temperature` | environment | dense_regression | SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7, ST_B10 | surface_temperature [K] | 733,505 / 7,059,585 / 22,060,801 / 60,452,865 |
| `forest_monitor` | forestry | segmentation | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | non_forest, broadleaf, coniferous, mixed | 733,988 / 7,060,548 / 22,062,244 / 60,454,788 |
| `deforestation_detector` | forestry | change_detection | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 (x 2 dates) | no_change, forest_loss | 735,394 / 7,063,362 / 22,066,466 / 60,460,418 |
| `forest_density_estimator` | forestry | dense_regression | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | canopy_cover [1] | 733,937 / 7,060,449 / 22,062,097 / 60,454,593 |
| `tree_height_estimator` | forestry | dense_regression | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, VV, VH | canopy_height [m] | 734,225 / 7,061,025 / 22,062,961 / 60,455,745 |
| `cloud_mask` | imaging | segmentation | B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12 | clear, thick_cloud, thin_cloud, cloud_shadow | 734,420 / 7,061,412 / 22,063,540 / 60,456,516 |
| `digitization_2d` | imaging | segmentation | red, green, blue | background, building, road, water, vegetation | 732,997 / 7,058,565 / 22,059,269 / 60,450,821 |
| `thematic_mapper` | imaging | segmentation | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | artificial, agricultural, forest, grassland, wetland, water, bare, snow_ice | 734,056 / 7,060,680 / 22,062,440 / 60,455,048 |
| `topography_mapper` | imaging | segmentation | elevation | flat, slope, ridge, valley, peak, pit | 732,726 / 7,058,022 / 22,058,454 / 60,449,734 |
| `dem_generator` | imaging | dense_regression | pan_t1, pan_t2 | elevation [m] | 732,785 / 7,058,145 / 22,058,641 / 60,449,985 |
| `dsm_generator` | imaging | dense_regression | pan_t1, pan_t2 | surface_elevation [m] | 732,785 / 7,058,145 / 22,058,641 / 60,449,985 |
| `dtm_generator` | imaging | dense_regression | surface_height | ground_elevation [m] | 732,641 / 7,057,857 / 22,058,209 / 60,449,409 |
| `model_3d` | imaging | dense_regression | red, green, blue, surface_height | object_height [m] | 733,073 / 7,058,721 / 22,059,505 / 60,451,137 |
| `stereo_processor` | imaging | dense_regression | pan_t1, pan_t2 | disparity [px] | 732,785 / 7,058,145 / 22,058,641 / 60,449,985 |
| `tri_stereo_processor` | imaging | dense_regression | pan_forward, pan_nadir, pan_backward | surface_elevation [m] | 732,929 / 7,058,433 / 22,059,073 / 60,450,561 |
| `coregistration` | imaging | enhancement | red, green, blue (x 2 dates) | dx, dy [px, px] | 733,378 / 7,059,330 / 22,060,418 / 60,452,354 |
| `mosaic_processor` | imaging | enhancement | red, green, blue | red, green, blue | 732,963 / 7,058,499 / 22,059,171 / 60,450,691 |
| `mosaicking` | imaging | enhancement | red, green, blue (x 2 dates) | red, green, blue | 733,395 / 7,059,363 / 22,060,467 / 60,452,419 |
| `multispectral_processor` | imaging | enhancement | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | 734,090 / 7,060,746 / 22,062,538 / 60,455,178 |
| `ortho_processor` | imaging | enhancement | red, green, blue, elevation | dx, dy [px, px] | 733,090 / 7,058,754 / 22,059,554 / 60,451,202 |
| `orthorectification` | imaging | enhancement | red, green, blue, elevation | red, green, blue | 733,107 / 7,058,787 / 22,059,603 / 60,451,267 |
| `panchromatic_processor` | imaging | enhancement | pan | pan | 732,641 / 7,057,857 / 22,058,209 / 60,449,409 |
| `pansharpening` | imaging | enhancement | ms_blue, ms_green, ms_red, ms_nir, pan | blue, green, red, nir | 733,268 / 7,059,108 / 22,060,084 / 60,451,908 |
| `raster_tiler` | imaging | enhancement | red, green, blue | red, green, blue | 732,963 / 7,058,499 / 22,059,171 / 60,450,691 |
| `super_resolution` | imaging | super_resolution | red, green, blue | red, green, blue | 134,992 / 657,264 / 2,784,528 / 6,109,872 |

| Family | Name | Intended application | Reference data needed for training |
| --- | --- | --- | --- |
| `fire_monitor` | Active Fire Detector | Detects active fire fronts and hotspots in short-wave infrared composites. | Bounding boxes of active fire areas. |
| `desertification_monitor` | Desertification Monitor | Classifies land degradation severity in drylands. | Degradation severity masks from field surveys. |
| `erosion_detector` | Erosion Detector | Maps eroded soil surfaces and gullies using spectral and terrain inputs. | Erosion masks. |
| `land_degradation_detector` | Land Degradation Detector | Separates degraded from non-degraded land. | Degradation masks. |
| `lulc_classifier` | Land Use and Land Cover Classifier | Maps land cover in the eleven classes of ESA WorldCover. | Land cover masks, for example ESA WorldCover. |
| `protected_area_change_detector` | Protected Area Change Detector | Detects land cover change inside protected areas. | Change masks by change type. |
| `drought_monitor` | Drought Monitor | Estimates a drought severity index. | Drought index rasters, for example from soil moisture products. |
| `wildlife_habitat_analyzer` | Wildlife Habitat Suitability | Scores habitat suitability for a target species. | Species occurrence data converted to suitability targets. |
| `natural_resources_monitor` | Natural Resources Monitor | Estimates above-ground biomass. | Biomass plots or LiDAR-derived biomass rasters. |
| `environmental_monitor` | Environmental Condition Monitor | Scores the ecological condition of land. | Ecological condition scores from field assessment. |
| `land_surface_temperature` | Land Surface Temperature | Estimates land surface temperature from reflective and thermal bands. | Land surface temperature products or in situ radiometry. |
| `forest_monitor` | Forest Monitor | Maps forest cover by forest type. | Forest type masks. |
| `deforestation_detector` | Deforestation Detector | Detects forest loss between two dates. | Forest loss masks. |
| `forest_density_estimator` | Forest Density Estimator | Estimates tree canopy cover. | Canopy cover fraction from LiDAR or high resolution imagery. |
| `tree_height_estimator` | Canopy Height Estimator | Estimates canopy height from optical and radar data. | Canopy height from airborne or spaceborne LiDAR. |
| `cloud_mask` | Cloud and Shadow Mask | Masks clear sky, thick cloud, thin cloud and cloud shadow in Sentinel-2 imagery. | Cloud and shadow masks. |
| `digitization_2d` | 2D Digitisation | Segments buildings, roads, water and vegetation for map digitisation. | Masks of the four map feature classes. |
| `thematic_mapper` | Thematic Mapper | Eight-class thematic mapping for general land cover mapping projects. | Thematic masks. |
| `topography_mapper` | Landform Mapper | Classifies landforms from a digital elevation model. | Landform masks, for example geomorphons. |
| `dem_generator` | DEM from Stereo | Estimates terrain elevation from a panchromatic stereo pair. | Reference DEMs, for example LiDAR. |
| `dsm_generator` | DSM from Stereo | Estimates surface elevation including buildings and trees from a stereo pair. | Reference DSMs, for example LiDAR. |
| `dtm_generator` | DTM from DSM | Removes buildings and vegetation from a surface model to estimate bare ground elevation. | Reference DTMs, for example LiDAR ground returns. |
| `model_3d` | Normalised Surface Model | Estimates the height of objects above ground (nDSM) from imagery and a surface model. | nDSM from LiDAR. |
| `stereo_processor` | Stereo Disparity Estimator | Estimates disparity between the two images of an epipolar stereo pair. | Reference disparity maps. |
| `tri_stereo_processor` | DSM from Tri-Stereo | Estimates surface elevation from a tri-stereo acquisition. | Reference DSMs. |
| `coregistration` | Co-registration Flow Estimator | Estimates the pixel displacement field that aligns a moving image to a reference image. | Displacement fields, for example from synthetic warps of a reference image. |
| `mosaic_processor` | Mosaic Colour Harmoniser | Harmonises the radiometry of a scene to a reference for seamless mosaics. | Radiometrically harmonised reference scenes. |
| `mosaicking` | Seamline Blender | Blends two overlapping scenes into one seamless image. | Seamlessly blended reference mosaics. |
| `multispectral_processor` | Multispectral Denoiser | Reduces noise and striping in multispectral imagery. | Clean reference images, for example temporal composites. |
| `ortho_processor` | Orthorectification Flow Estimator | Estimates the displacement field that removes relief displacement using a DEM. | Displacement fields from rigorous orthorectification. |
| `orthorectification` | Learned Orthorectification | Produces an orthorectified image from a raw image and a DEM. | Orthorectified reference images. |
| `panchromatic_processor` | Panchromatic Denoiser | Reduces noise in panchromatic imagery. | Clean panchromatic references. |
| `pansharpening` | Pansharpening | Fuses upsampled multispectral bands with the panchromatic band into a sharpened multispectral image. | Sharpened references, for example by Wald's protocol. |
| `raster_tiler` | Tile Radiometric Normaliser | Normalises the radiometry of individual tiles before tiling into a web map. | Radiometrically normalised reference tiles. |
| `super_resolution` | Super-Resolution | Increases the spatial resolution of RGB imagery by a factor of four. | High resolution reference images. |

"Intended application" is the catalogue description of what a model does once it has been trained.

### 3.2 Architectures and outputs

| Task | Architecture id | Output |
| --- | --- | --- |
| detection (`fire_monitor`) | `centernet` | $(N, K + 4, H/4, W/4)$: class heat maps, box sizes and centre offsets, decoded as described in [domain 01](01_ai_products.md#42-decoding-of-detector-outputs) |
| segmentation | `unet` | $(N, K, H, W)$ class logits |
| change detection (`protected_area_change_detector`, `deforestation_detector`) | `unet_early_fusion` | $(N, K, H, W)$ change class logits; both dates stacked on the channel axis |
| dense regression | `unet_regression` | $(N, K, H, W)$ values in the listed units; a sigmoid for $[0, 1]$ targets |
| enhancement | `unet_image_to_image` | $(N, K, H, W)$ output bands or displacement components (`dx`, `dy` in pixels) |
| super-resolution (`super_resolution`) | `residual_subpixel` | $(N, K, sH, sW)$ with $s = 4$: residual blocks without normalisation [1] and sub-pixel convolution [2] |

`SuperResolution(model, scale_factor=None)` builds a variant with another factor when `scale_factor` differs from the catalogue value; such a model is a new, untrained network.

## 4. Preprocessing

Arrays are $(H, W)$ or band-first $(C, H, W)$; missing values are NaN or an explicit `nodata` value.

### 4.1 Radiometry

| Function | Converts | Formula and source |
| --- | --- | --- |
| `sentinel2_reflectance(dn, offset=None, quantification=10000, processing_baseline=None, nodata=0)` | Sentinel-2 L1C or L2A digital numbers to reflectance | $\rho = (DN + O)/Q$; from processing baseline 04.00 (25 January 2022) the offset is $O = -1000$, before it 0 [3]; DN 0 is no data |
| `landsat_radiance(dn, mult, add)` | Landsat 8/9 Collection 2 Level-1 DN to radiance | $L = M_L Q + A_L$ [4] |
| `landsat_toa_reflectance(dn, mult, add, sun_elevation)`, `landsat_toa_reflectance_from_mtl(dn, mtl, band)` | DN to sun-corrected top-of-atmosphere reflectance | $\rho = (M_\rho Q + A_\rho)/\sin\theta_{SE}$ with the factors and sun elevation of the MTL file (`parse_landsat_mtl`) [4] |
| `landsat_brightness_temperature(radiance, k1, k2)` | thermal radiance to brightness temperature | $T = K_2 / \ln(K_1/L + 1)$ [4] |
| `landsat_c2l2_reflectance(dn)`, `landsat_c2l2_temperature(dn)` | Collection 2 Level-2 products | $\rho = 2.75 \times 10^{-5}\,DN - 0.2$; $T = 0.00341802\,DN + 149.0$ K [5] |
| `earth_sun_distance(day_of_year)` | Earth-Sun distance in astronomical units | Fourier series of Spencer [6] |
| `radiance_to_reflectance(radiance, esun, sun_zenith, day_of_year)` | radiance to TOA reflectance | $\rho = \pi L d^2 / (E_{sun} \cos\theta_s)$ [7] |
| `dark_object_subtraction(reflectance, dark_percentile=0.01, dark_reflectance=0.01)` | haze removal (DOS1) | subtracts the path reflectance $\rho_{dark} - 0.01$ from every pixel of a band, $\rho_{dark}$ being a low percentile [8], [9] |

### 4.2 Quality masks

`scl_valid_mask(scl, invalid=(0, 1, 3, 8, 9, 10))` returns `True` for usable pixels of the Sentinel-2 L2A scene classification (by default it rejects no data, saturated or defective, cloud shadow, medium and high probability cloud and thin cirrus) [10]. `landsat_qa_mask(qa, fill, dilated_cloud, cirrus, cloud, cloud_shadow, snow=False, water=False)` returns `True` for flagged pixels of the Landsat Collection 2 `QA_PIXEL` band (bits 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow, 7 water) [5]; `landsat_cloud_confidence` and `qa_bits` read the two-bit confidence fields and arbitrary bit fields. `buffer_mask(mask, pixels)` grows a mask and `apply_mask(image, mask)` sets masked pixels to NaN.

### 4.3 Enhancement

`linear_stretch`, `percentile_stretch` (default 2 % to 98 %), `percentile_bounds`, `gamma_correction`, `minmax_normalize`, `standardize`, `histogram_equalize` and `histogram_match` work band by band and ignore NaN. Histogram matching maps every source value $x$ to $F_{ref}^{-1}(F_{src}(x))$, with linear interpolation between quantiles [11]; it serves the relative radiometric normalisation of multi-date images.

### 4.4 Resampling

`resample(image, shape, method="bilinear", nodata=None)` interpolates with nearest neighbour, bilinear or cubic splines using pixel-area alignment (the outer pixel edges stay fixed, as in GDAL). With missing values, the interpolated data are divided by the interpolated validity weights (normalised convolution [12]); output pixels with a weight below one half are missing. `aggregate(image, factor, method="mean")` reduces by an integer factor with `mean`, `sum`, `min`, `max`, `median` or `mode`, ignoring NaN, and `scaled_transform` gives the affine transform of the new grid.

### 4.5 Pansharpening

`upsample_to_pan(ms, shape, order=3)` brings the multispectral bands onto the panchromatic grid. The three fusion methods are component substitution: an intensity $I = \sum_k w_k MS_k (+ b)$ is computed, the panchromatic band $P$ is matched to the mean and standard deviation of $I$, and the detail is injected as $F_k = MS_k + g_k (P - I)$ [13]:

- `brovey(pan, ms)`: $F_k = MS_k\, P / I$ with $I$ the weighted mean of the bands [14];
- `ihs(pan, ms)`: $g_k = 1$ with $I$ the weighted mean (fast generalised IHS [15]);
- `gram_schmidt(pan, ms)`: weights $w_k$ and $b$ from a least-squares regression of $P$ on the bands and $g_k = \mathrm{cov}(MS_k, I)/\mathrm{var}(I)$ (Gram-Schmidt adaptive, GSA [16]), which reproduces Gram-Schmidt spectral sharpening [17] with an adaptive intensity.

### 4.6 Model input transforms

`Normalize`, `Resize`, `Pad`, `Compose`, `to_tensor` and `from_tensor` prepare arrays for inference; the task classes of `unbihexium.ai` apply the normalisation statistics stored with a trained checkpoint automatically.

## 5. Postprocessing

`unbihexium.postprocessing` turns scores and class maps into map products.

| Group | Functions | Method and source |
| --- | --- | --- |
| Activations | `sigmoid`, `softmax`, `threshold`, `argmax`, `confidence_mask(probabilities, min_confidence, min_margin, nodata=255)` | class map with low-confidence pixels set to `nodata` |
| Uncertainty | `prediction_entropy`, `margin` | normalised entropy $H = -\sum_k p_k \ln p_k / \ln K$ [18]; margin $p_{(1)} - p_{(2)}$ [19] |
| Morphology | `structuring_element`, `morphology_clean`, `remove_small_objects`, `fill_small_holes`, `majority_filter`, `connected_components`, `component_statistics` | opening, closing, erosion and dilation [20]; 4- or 8-connectivity |
| Minimum mapping unit | `sieve(labels, min_size, connectivity=4, nodata=None)` | regions below `min_size` pixels take the class of their largest neighbour (GDAL sieve through rasterio) [21] |
| Vectorisation | `raster_to_polygons`, `simplify_polygons`, `polygons_to_geodataframe` | polygons of connected regions in map coordinates (GDAL polygonize through rasterio), Douglas-Peucker simplification that keeps polygons valid [22] |
| Tiling | `tile_positions`, `blend_weights`, `stitch_tiles` | overlapping tiles blended with uniform or linear (feathered) weights [23] |

## 6. Visualisation

`unbihexium.visualization` renders imagery and map products without a plotting library:

- `rgb_composite(stack, bands, band_names, stretch="percentile", low=2, high=98, gamma=1.0, ...)` with the named composites of `SENTINEL2_COMPOSITES` (`true_color`, `color_infrared`, `swir`, `agriculture`, `geology`, `bathymetric`) and `LANDSAT89_COMPOSITES`;
- `apply_colormap`, `colormap_lut`, `classify_colors`, `colorize_classes` (by default the ESA WorldCover palette `WORLDCOVER_PALETTE`), `colorize_mask`, `overlay_mask`, `alpha_composite`, with the colour maps `greys`, `viridis`, `rdylgn`, `rdbu`, `brbg` and `blues`;
- `hillshade` and `slope_aspect` (Horn's method [24]), `multidirectional_hillshade` and `shade_image`;
- `save_png(path, image, transform)` writes a PNG with a world file (`.pgw`), `quicklook` a stretched preview, and `legend_image` or `legend_figure` a legend.

## 7. Image quality measures

`unbihexium.metrics` provides full-reference measures for super-resolution, enhancement and pansharpening products (reference first, estimate second; band-first arrays):

| Function | Measure | Source |
| --- | --- | --- |
| `psnr(pred, target, max_val=1.0)` | $10 \log_{10}(\mathrm{MAX}^2/\mathrm{MSE})$ in dB | |
| `ssim(pred, target, data_range=1.0, sigma=1.5)` | structural similarity with a Gaussian window, mean over pixels and bands | [25] |
| `spectral_angle`, `sam` | $\arccos\left(\langle x, y\rangle / (\lVert x \rVert \lVert y \rVert)\right)$ per pixel, mean in degrees | [26] |
| `ergas(reference, estimate, ratio=4)` | $\frac{100}{r} \sqrt{\frac{1}{K}\sum_k (\mathrm{RMSE}_k/\mu_k)^2}$ with the resolution ratio $r = l/h$ | [27] |
| `q_index(reference, estimate, block_size=8)` | $Q = \frac{4 \sigma_{xy} \mu_x \mu_y}{(\sigma_x^2 + \sigma_y^2)(\mu_x^2 + \mu_y^2)}$ over sliding windows, mean over windows and bands | [28] |

## 8. Examples

The examples were executed on 24 September 2026 against the main branch with CPython 3.13 on a CPU, with `UNBIHEXIUM_CACHE` set to a temporary directory. They share one Python session and one working directory.

### 8.1 Reflectance and quality masks

```python
import numpy as np
from unbihexium.preprocessing import (
    landsat_c2l2_reflectance, landsat_c2l2_temperature, landsat_qa_mask,
    scl_valid_mask, sentinel2_reflectance,
)

dn = np.array([[0, 1000, 2000, 11000]], dtype=np.uint16)
print(sentinel2_reflectance(dn, processing_baseline="05.10"))
print(sentinel2_reflectance(dn, processing_baseline="03.01"))
print(scl_valid_mask(np.array([[4, 5, 8, 9, 3, 6]])))
print(landsat_c2l2_reflectance(np.array([7273, 20000, 0])).round(4),
      landsat_c2l2_temperature(np.array([44177])).round(2))
print(landsat_qa_mask(np.array([21824, 22280, 1])))  # clear land, cloud, fill
```

```text
[[nan 0.  0.1 1. ]]
[[nan 0.1 0.2 1.1]]
[[ True  True False False False  True]]
[0.   0.35  nan] [300.]
[False  True  True]
```

The same digital numbers give different reflectances before and after processing baseline 04.00; the baseline MUST therefore be read from the product metadata. In the scene classification, vegetation (4), bare soil (5) and water (6) are valid, clouds (8, 9) and cloud shadow (3) are not.

### 8.2 Pansharpening and image quality

A synthetic four-band image at full resolution serves as reference; its panchromatic band is the band mean, and the multispectral input is the reference degraded by a factor of four (Wald's protocol [27]).

```python
import numpy as np
from scipy.ndimage import gaussian_filter
from unbihexium.metrics import ergas, q_index, sam
from unbihexium.preprocessing import aggregate, brovey, gram_schmidt, ihs, upsample_to_pan

rng = np.random.default_rng(0)
base = gaussian_filter(rng.random((64, 64)), 2)
reference = np.stack([base * f + o for f, o in [(0.8, 0.05), (0.9, 0.07), (1.0, 0.06), (1.4, 0.2)]])
pan = reference.mean(axis=0)
ms_up = upsample_to_pan(aggregate(reference, 4, method="mean"), pan.shape)

for name, method in [("upsampled", None), ("brovey", brovey), ("ihs", ihs), ("gram_schmidt", gram_schmidt)]:
    estimate = ms_up if method is None else method(pan, ms_up)
    print(f"{name:13} ERGAS {ergas(reference, estimate, ratio=4):6.3f}  "
          f"SAM {sam(reference, estimate):6.3f}  Q {q_index(reference, estimate):.4f}")
```

```text
upsampled     ERGAS  0.569  SAM  0.064  Q 0.8495
brovey        ERGAS  0.282  SAM  0.064  Q 0.9841
ihs           ERGAS  0.290  SAM  0.166  Q 0.9823
gram_schmidt  ERGAS  0.280  SAM  0.032  Q 0.9848
```

All three methods halve ERGAS against the upsampled bands; Brovey preserves the spectral angle of the input by construction, IHS distorts it, and the regression-based Gram-Schmidt variant gives the smallest error on this synthetic scene. The numbers describe this toy example only.

### 8.3 Cleaning and vectorising a class map

```python
import numpy as np
from unbihexium.postprocessing import confidence_mask, prediction_entropy, raster_to_polygons, sieve

p = np.zeros((3, 6, 6))
p[0], p[1], p[2] = 0.8, 0.1, 0.1
p[:, 1:4, 1:4] = np.array([0.1, 0.8, 0.1])[:, None, None]
p[:, 5, 5] = [0.4, 0.35, 0.25]          # an uncertain pixel
labels = confidence_mask(p, min_confidence=0.5)
print(labels)
print(prediction_entropy(p)[[0, 5], [0, 5]].round(3))

labels[0, 5] = 2                        # an isolated pixel of class 2
cleaned = sieve(labels, min_size=2, nodata=255)
print(cleaned[0, 5])
polygons = raster_to_polygons(
    cleaned, transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6700000.0), skip_values=(0, 255)
)
print([(value, geometry.area) for geometry, value in polygons])
```

```text
[[  0   0   0   0   0   0]
 [  0   1   1   1   0   0]
 [  0   1   1   1   0   0]
 [  0   1   1   1   0   0]
 [  0   0   0   0   0   0]
 [  0   0   0   0   0 255]]
[0.582 0.984]
0
[(1.0, 900.0)]
```

The uncertain pixel (normalised entropy 0.984) becomes no data, the isolated pixel is absorbed by its neighbours, and the remaining class 1 region is one polygon of 900 square metres (nine 10 m pixels).

### 8.4 A quicklook

```python
import numpy as np
from unbihexium.visualization import rgb_composite, save_png

stack = np.random.default_rng(0).random((3, 16, 16))
rgb = rgb_composite(stack, (0, 1, 2), gamma=1.2)
print(rgb.shape, rgb.dtype, save_png("quicklook.png", rgb, transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6700000.0)))
```

```text
(16, 16, 3) uint8 quicklook.png
```

The command also writes the world file `quicklook.pgw` next to the PNG.

### 8.5 A tiny starter model

The untrained land cover classifier assigns random noise to snow and ice, water and mangroves. The class fractions have no meaning; they show why the model must be trained.

```python
import numpy as np
from unbihexium.ai import LandCoverClassifier

image = np.random.default_rng(0).uniform(0.0, 0.4, (10, 64, 64)).astype("float32")
result = LandCoverClassifier("lulc_classifier_tiny", threshold=None).predict(image)
print({k: round(v, 3) for k, v in result.class_fractions().items() if v > 0})
```

```text
{'tree_cover': 0.002, 'shrubland': 0.022, 'cropland': 0.097, 'snow_ice': 0.418, 'water': 0.303, 'herbaceous_wetland': 0.004, 'mangroves': 0.118, 'moss_lichen': 0.036}
```

## 9. Limitations and responsible use

### 9.1 Conventions

The key words MUST, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 and RFC 8174 [29], [30] when, and only when, they appear in capitals.

### 9.2 Limitations

- The 35 model families are untrained. Their outputs MUST NOT be used for environmental reporting, forest monitoring, enforcement or any other decision before the model has been trained and validated on independent reference data; the accuracy and area estimation measures of [domain 02](02_tourism_data_processing.md#7-accuracy-assessment) SHOULD be used for the validation.
- Dark object subtraction is a first-order haze correction. Where surface reflectance products (Sentinel-2 L2A, Landsat Collection 2 Level-2) exist, they SHOULD be preferred.
- Pansharpened and super-resolved images contain synthesised detail. They MAY be used for visual interpretation, but measurements from them SHOULD be validated against true high resolution data.
- Maps of deforestation, protected areas or land degradation can affect people and land rights. Read [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md).

## 10. Related documents

- [Capability index](index.md), [domain 03](03_indices_flood_water.md) (spectral indices) and the imaging product documents [08](08_value_added_imagery.md), [10](10_satellite_imagery_features.md) and [11](11_resolution_metadata_qa.md).
- [Model catalogue](../model_zoo/model_catalog.md), [training](../model_zoo/training.md) and [inference](../model_zoo/inference.md).
- [API reference](../reference/api.md) and [command line reference](../reference/cli.md).

## References

[1] Lim, B., Son, S., Kim, H., Nah, S., Lee, K. M. Enhanced deep residual networks for single image super-resolution. CVPR Workshops 2017. 2017. <https://doi.org/10.1109/CVPRW.2017.151>

[2] Shi, W., et al. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. CVPR 2016. 2016. <https://doi.org/10.1109/CVPR.2016.207>

[3] European Space Agency. Sentinel-2 products specification document, S2-PDGS-TAS-DI-PSD (radiometric offset of processing baseline 04.00). 2022.

[4] U.S. Geological Survey. Landsat 8-9 Collection 2 Level 1 data format control book, LSDS-1822. <https://www.usgs.gov/landsat-missions/landsat-collection-2-level-1-data>

[5] U.S. Geological Survey. Landsat 8-9 Collection 2 Level 2 science product guide, LSDS-1619. <https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products>

[6] Spencer, J. W. Fourier series representation of the position of the sun. Search 2(5), 172. 1971.

[7] Chander, G., Markham, B. L., Helder, D. L. Summary of current radiometric calibration coefficients for Landsat MSS, TM, ETM+, and EO-1 ALI sensors. Remote Sensing of Environment 113(5), 893-903. 2009. <https://doi.org/10.1016/j.rse.2009.01.007>

[8] Chavez, P. S. An improved dark-object subtraction technique for atmospheric scattering correction of multispectral data. Remote Sensing of Environment 24(3), 459-479. 1988. <https://doi.org/10.1016/0034-4257(88)90019-3>

[9] Chavez, P. S. Image-based atmospheric corrections, revisited and improved. Photogrammetric Engineering and Remote Sensing 62(9), 1025-1036. 1996.

[10] Main-Knorn, M., Pflug, B., Louis, J., Debaecker, V., Mueller-Wilm, U., Gascon, F. Sen2Cor for Sentinel-2. Proceedings of SPIE 10427, Image and Signal Processing for Remote Sensing XXIII. 2017. <https://doi.org/10.1117/12.2278218>

[11] Gonzalez, R. C., Woods, R. E. Digital Image Processing, 4th edition, chapter 3. Pearson. 2018.

[12] Knutsson, H., Westin, C.-F. Normalized and differential convolution. Proceedings of IEEE CVPR, 515-523. 1993. <https://doi.org/10.1109/CVPR.1993.341081>

[13] Vivone, G., et al. A critical comparison among pansharpening algorithms. IEEE Transactions on Geoscience and Remote Sensing 53(5), 2565-2586. 2015. <https://doi.org/10.1109/TGRS.2014.2361734>

[14] Gillespie, A. R., Kahle, A. B., Walker, R. E. Color enhancement of highly correlated images. II. Channel ratio and "chromaticity" transformation techniques. Remote Sensing of Environment 22(3), 343-365. 1987. <https://doi.org/10.1016/0034-4257(87)90088-5>

[15] Tu, T.-M., Su, S.-C., Shyu, H.-C., Huang, P. S. A new look at IHS-like image fusion methods. Information Fusion 2(3), 177-186. 2001. <https://doi.org/10.1016/S1566-2535(01)00036-7>

[16] Aiazzi, B., Baronti, S., Selva, M. Improving component substitution pansharpening through multivariate regression of MS+Pan data. IEEE Transactions on Geoscience and Remote Sensing 45(10), 3230-3239. 2007. <https://doi.org/10.1109/TGRS.2007.901007>

[17] Laben, C. A., Brower, B. V. Process for enhancing the spatial resolution of multispectral imagery using pan-sharpening. US Patent 6,011,875. 2000. <https://patents.google.com/patent/US6011875A>

[18] Shannon, C. E. A mathematical theory of communication. Bell System Technical Journal 27(3), 379-423. 1948. <https://doi.org/10.1002/j.1538-7305.1948.tb01338.x>

[19] Scheffer, T., Decomain, C., Wrobel, S. Active hidden Markov models for information extraction. Advances in Intelligent Data Analysis, LNCS 2189, 309-318. 2001. <https://doi.org/10.1007/3-540-44816-0_31>

[20] Soille, P. Morphological Image Analysis: Principles and Applications, 2nd edition. Springer. 2003. <https://doi.org/10.1007/978-3-662-05088-0>

[21] Saura, S. Effects of minimum mapping unit on land cover data spatial configuration and composition. International Journal of Remote Sensing 23(22), 4853-4880. 2002. <https://doi.org/10.1080/01431160110114493>

[22] Douglas, D. H., Peucker, T. K. Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. Cartographica 10(2), 112-122. 1973. <https://doi.org/10.3138/FM57-6770-U75U-7727>

[23] Burt, P. J., Adelson, E. H. A multiresolution spline with application to image mosaics. ACM Transactions on Graphics 2(4), 217-236. 1983. <https://doi.org/10.1145/245.247>

[24] Horn, B. K. P. Hill shading and the reflectance map. Proceedings of the IEEE 69(1), 14-47. 1981. <https://doi.org/10.1109/PROC.1981.11918>

[25] Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600-612. 2004. <https://doi.org/10.1109/TIP.2003.819861>

[26] Yuhas, R. H., Goetz, A. F. H., Boardman, J. W. Discrimination among semi-arid landscape endmembers using the spectral angle mapper (SAM) algorithm. Summaries of the Third Annual JPL Airborne Geoscience Workshop, JPL Publication 92-14, 147-149. 1992. <https://ntrs.nasa.gov/citations/19940012238>

[27] Wald, L. Data Fusion: Definitions and Architectures. Fusion of Images of Different Spatial Resolutions. Presses de l'Ecole, Ecole des Mines de Paris. 2002.

[28] Wang, Z., Bovik, A. C. A universal image quality index. IEEE Signal Processing Letters 9(3), 81-84. 2002. <https://doi.org/10.1109/97.995823>

[29] Bradner, S. Key words for use in RFCs to indicate requirement levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[30] Leiba, B. Ambiguity of uppercase vs lowercase in RFC 2119 key words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

<!--
=============================================================================
End of file docs/capabilities/04_environment_forestry_image_processing.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
