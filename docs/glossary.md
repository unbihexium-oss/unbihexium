<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/glossary.md
Title       : Glossary
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Glossary

| Field | Value |
| --- | --- |
| Document | UBX-DOC-310 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../MAINTAINERS.md)) |
| Applies to | Unbihexium 2.0.0 and the main branch (model catalogue 2.0.0) |

## Abstract

This glossary defines the technical terms and abbreviations that appear in the source code, the command line help and the documentation of Unbihexium. It is written for users who meet an unfamiliar remote sensing, radar, geostatistical or machine learning term, for contributors who need the project's precise meaning of words such as "starter model", "variant" or "digest", and for auditors who read the supply-chain documents. Each entry gives a short definition and, where the term corresponds to code, the module, function or file of Unbihexium that implements it. Formulas are given exactly as the code evaluates them, with the primary source of each method. The glossary covers terms used in the repository only; it is not a general dictionary of Earth observation.

## Contents

1. [Scope and use](#1-scope-and-use)
2. [Earth observation and sensors](#2-earth-observation-and-sensors)
3. [Geospatial data and formats](#3-geospatial-data-and-formats)
4. [Spectral indices](#4-spectral-indices)
5. [Synthetic aperture radar](#5-synthetic-aperture-radar)
6. [Terrain and hydrology](#6-terrain-and-hydrology)
7. [Geostatistics and spatial analysis](#7-geostatistics-and-spatial-analysis)
8. [Accuracy and image quality metrics](#8-accuracy-and-image-quality-metrics)
9. [Machine learning and the model zoo](#9-machine-learning-and-the-model-zoo)
10. [Software, releases and supply chain](#10-software-releases-and-supply-chain)
11. [References](#references)

## 1. Scope and use

### 1.1 Organisation

Terms are grouped by subject and listed alphabetically within each group. The column "In Unbihexium" names the implementing module (relative to `src/unbihexium/`), function, command or file, so that each definition can be checked against the code. Abbreviations are expanded at their entry; common abbreviations such as NIR (near infrared) and SWIR (short-wave infrared) are used freely.

### 1.2 Status of the models

Several entries concern the model zoo. The 520 models of the zoo (130 families in the four variants tiny, base, large and mega) are untrained starter models with deterministic weights, with the exception of the 28 models of the 7 spectral index families, which compute exact formulas. Where an entry describes a learned model, its predictions are not meaningful until the model has been trained on labelled data. See [README.md, Section 2.2](../README.md#22-the-models-are-starter-models).

## 2. Earth observation and sensors

| Term | Definition | In Unbihexium |
| --- | --- | --- |
| Band | One spectral channel of a multispectral image, identified by its wavelength interval. Band arrays are stacked as `(bands, rows, columns)`. Band numbers on the command line count from 1, as in GDAL; Python indices count from 0. | `core.sensor.SpectralBand`, `unbihexium index --red 3 ...` |
| Brightness temperature | Temperature of a black body that would emit the observed thermal radiance, `T = K2 / ln(K1 / L + 1)` in kelvin, with the calibration constants K1 and K2 of the Landsat metadata file. | `preprocessing.landsat_brightness_temperature` |
| Cloud mask | Boolean raster that marks pixels covered by clouds, cloud shadows or other invalid observations, used to exclude them from analysis. | `preprocessing.scl_valid_mask`, `preprocessing.landsat_qa_mask`, the `cloud_mask` model family |
| Collection 2 Level-2 (Landsat) | Landsat surface reflectance and surface temperature products, stored as scaled integers; reflectance is `DN * 0.0000275 - 0.2`. | `preprocessing.landsat_c2l2_reflectance`, `LANDSAT_C2L2_SR_SCALE` |
| Dark object subtraction (DOS) | Image-based haze correction (DOS1) that estimates the path reflectance of each band from a low percentile of its TOA reflectance, assuming a dark object reflectance of 1 %, and subtracts it [1]. | `preprocessing.dark_object_subtraction` |
| Digital number (DN) | Raw integer value stored by a sensor or product before scaling to radiance or reflectance. | `preprocessing.radiometry` |
| EO (Earth observation) | Gathering of information about the Earth's surface and atmosphere by remote sensing, in particular from satellites. The subject of the library. | whole package |
| GSD (ground sampling distance) | Distance on the ground between the centres of adjacent pixels, for example 10 m for the Sentinel-2 visible and NIR bands. The model catalogue lists suitable resolutions for each family under `sources`. | `core.sensor`, `zoo/catalog.yaml` |
| L2A (Sentinel-2 Level-2A) | Sentinel-2 bottom-of-atmosphere reflectance product with the scene classification layer; reflectance is `(DN + offset) / 10000`, with an offset of -1000 for processing baseline 04.00 and later. | `preprocessing.sentinel2_reflectance`, `S2_QUANTIFICATION`, `S2_OFFSET_PB04` |
| Multispectral | Imagery with several spectral bands, typically visible, NIR and SWIR. Panchromatic imagery has one broad band of higher resolution. | `core.sensor` |
| Pansharpening | Fusion of a high-resolution panchromatic band with lower-resolution multispectral bands. Implemented as the Brovey (ratio) transform, IHS and Gram-Schmidt adaptive component substitution. | `preprocessing.brovey`, `ihs`, `gram_schmidt` |
| QA_PIXEL | Landsat Collection 2 bit-packed quality band with flags for fill, dilated cloud, cirrus, cloud, cloud shadow, snow, clear and water, and confidence bits. | `preprocessing.landsat_qa_mask`, `LANDSAT_QA_BITS` |
| Radiance | Radiant flux per unit area, solid angle and wavelength measured at the sensor, in W m^-2 sr^-1 um^-1. | `preprocessing.landsat_radiance` |
| Reflectance | Ratio of reflected to incident radiation, between 0 and 1. Top-of-atmosphere (TOA) reflectance is computed from radiance and sun geometry; surface or bottom-of-atmosphere (BOA) reflectance is corrected for the atmosphere. Spectral indices expect reflectance, not DN. | `preprocessing.landsat_toa_reflectance`, `radiance_to_reflectance` |
| Remote sensing | Measurement of an object without contact, here by optical, thermal or radar sensors on satellites, aircraft or drones. | whole package |
| SCL (scene classification layer) | Sentinel-2 L2A band that assigns every pixel one of 12 classes (for example cloud medium probability, cloud shadow, vegetation, water). | `preprocessing.SCL_CLASSES`, `scl_valid_mask` |
| Sensor model | Description of a sensor's bands, wavelengths and resolutions. Sentinel-1, Sentinel-2 and Landsat 8 and 9 are defined. | `core.SensorModel`, `core.get_sensor` |

## 3. Geospatial data and formats

| Term | Definition | In Unbihexium |
| --- | --- | --- |
| Affine transform (geotransform) | Six coefficients `(a, b, c, d, e, f)` that map pixel column and row to map coordinates: `x = a * col + b * row + c`, `y = d * col + e * row + f`. `read_geotiff` returns them as a tuple; `affine.Affine(*coefficients)` rebuilds the rasterio object. | `io.read_geotiff`, metadata key `transform` |
| COG (Cloud Optimized GeoTIFF) | GeoTIFF with internal tiling and overviews, ordered so that clients can read parts of the file with HTTP range requests [2]. | `io.write_geotiff(..., cog=True)`, `io.write_cog`, `io.is_cog` |
| CRS (coordinate reference system) | Definition of how coordinates relate to locations on the Earth, given as an EPSG code (for example `EPSG:32635`, UTM zone 35N), WKT or a rasterio CRS. Rasters carry their CRS as a string. | `io`, `core.Raster` |
| EPSG code | Identifier of a CRS in the EPSG Geodetic Parameter Dataset [3]. | CRS arguments |
| GeoJSON | JSON format for vector features, defined by RFC 7946 [4]: WGS 84 longitude and latitude, and counterclockwise exterior rings. Malformed input raises `ValueError`. | `io.read_geojson`, `validate_geojson`, `rewind`, `reproject_geojson` |
| GeoParquet | Apache Parquet files with a geometry column in well-known binary and GeoParquet metadata [5]. Requires the `parquet` extra. | `io.read_geoparquet`, `write_geoparquet` |
| GeoTIFF | TIFF image with georeferencing tags [6]. Written with DEFLATE compression and a tiled layout by default. | `io.read_geotiff`, `io.write_geotiff` |
| No-data value | Pixel value that marks missing data. With `masked=True`, `read_geotiff` replaces it by NaN. | `nodata` arguments |
| Overview | Reduced-resolution copy of a raster stored in the same file, used for fast display and required by the COG layout. | `io.build_overviews`, `read_geotiff(overview_level=...)` |
| Raster | Gridded data with georeferencing; in the library a NumPy array `(bands, rows, columns)` with CRS, transform and no-data value. | `core.Raster`, `core.RasterMetadata` |
| STAC (SpatioTemporal Asset Catalog) | Specification for describing geospatial assets as JSON items, collections and catalogues, with a search API [7]. | `io.STACClient`, `search_stac`, `walk_catalog`, extra `stac` |
| Tile | Rectangular part of a larger raster, processed independently; also a cell of the XYZ web map tile scheme. | `core.Tile`, `core.TileGrid`, `core.TileIndex` |
| Vector | Points, lines and polygons with attributes. | `core.Vector`, `io.geojson` |
| Window | Pixel range `(row_start, row_stop, col_start, col_stop)` read from a raster without reading the rest. | `read_geotiff(window=...)` |
| Zarr | Chunked, compressed N-dimensional array storage format [8]; version 3 is written. Requires the `zarr` extra. | `io.read_zarr`, `write_zarr`, `write_raster_zarr` |

## 4. Spectral indices

### 4.1 General terms

| Term | Definition | In Unbihexium |
| --- | --- | --- |
| Normalized difference | `(a - b) / (a + b)` of two bands, the form of NDVI, NDWI, NBR and many others; the result lies in [-1, 1] for non-negative inputs. | `indices.normalized_difference` |
| Spectral index | Arithmetic combination of bands that emphasises a surface property such as vegetation, water, burn or built-up area. | `unbihexium.indices`, `core.IndexRegistry` |
| Zero denominator | Where the denominator of an index is zero the result is NaN; no small constant is added. | `indices.safe_divide` |

### 4.2 Implemented indices

Two interfaces exist. The functions of `unbihexium.indices` take bands as keyword arguments (for example `ndvi(nir=..., red=...)`, or `indices.compute_index("ndvi", nir=..., red=...)`). The registry of `unbihexium.core` evaluates indices by name from a mapping of upper-case band names (`core.compute_index("NDVI", {"NIR": nir, "RED": red})`) and is used by `unbihexium index`. The column "Name" gives the function name and, in parentheses, the registry name where it differs or where only the registry has the index.

| Index | Name | Formula as implemented | Source |
| --- | --- | --- | --- |
| Normalized Difference Vegetation Index | `ndvi` (NDVI) | `(NIR - RED) / (NIR + RED)` | [9] |
| Green NDVI | `gndvi` (GNDVI) | `(NIR - GREEN) / (NIR + GREEN)` | [10] |
| Normalized Difference Red Edge | `ndre` (NDRE) | `(NIR - RE) / (NIR + RE)` | [11] |
| Enhanced Vegetation Index | `evi` (EVI) | `2.5 (NIR - RED) / (NIR + 6 RED - 7.5 BLUE + 1)` | [12] |
| Two-band EVI | `evi2` (EVI2) | `2.5 (NIR - RED) / (NIR + 2.4 RED + 1)` | [13] |
| Soil Adjusted Vegetation Index | `savi` (SAVI) | `(1 + L) (NIR - RED) / (NIR + RED + L)`, `L = 0.5` | [14] |
| Optimised SAVI | `osavi` (OSAVI) | `(NIR - RED) / (NIR + RED + 0.16)` | [15] |
| Modified SAVI (MSAVI2) | `msavi` (MSAVI) | `(2 NIR + 1 - sqrt((2 NIR + 1)^2 - 8 (NIR - RED))) / 2` | [16] |
| Atmospherically Resistant VI | `arvi` (ARVI) | `(NIR - RB) / (NIR + RB)`, `RB = RED - gamma (BLUE - RED)`, `gamma = 1` | [17] |
| Visible Atmospherically Resistant Index | `vari` (VARI) | `(GREEN - RED) / (GREEN + RED - BLUE)` | [18] |
| Kernel NDVI | `kndvi` | `tanh(NDVI^2)` (RBF kernel with sigma = (NIR + RED) / 2) | [19] |
| Green chlorophyll index | `ci_green` (CIgreen) | `NIR / GREEN - 1` | [20] |
| Red-edge chlorophyll index | `ci_rededge` (CIre) | `NIR / RE - 1` | [20] |
| Simple ratio | (SR) | `NIR / RED` | registry only |
| Wide Dynamic Range VI | (WDRVI) | `(alpha NIR - RED) / (alpha NIR + RED)` | registry only |
| Normalized Difference Water Index | `ndwi` (NDWI) | `(GREEN - NIR) / (GREEN + NIR)` | [21] |
| Modified NDWI | `mndwi` (MNDWI) | `(GREEN - SWIR1) / (GREEN + SWIR1)` | [22] |
| Normalized Difference Moisture Index | `ndmi` (NDMI) | `(NIR - SWIR1) / (NIR + SWIR1)` (the NDWI of Gao) | [23] |
| Automated Water Extraction Index, no shadow | `awei_nsh` (AWEInsh) | `4 (GREEN - SWIR1) - (0.25 NIR + 2.75 SWIR2)` | [24] |
| Automated Water Extraction Index, shadow | `awei_sh` (AWEIsh) | `BLUE + 2.5 GREEN - 1.5 (NIR + SWIR1) - 0.25 SWIR2` | [24] |
| Normalized Difference Turbidity Index | (NDTI) | `(RED - GREEN) / (RED + GREEN)` | registry only |
| Normalized Difference Chlorophyll Index | (NDCI) | `(RE - RED) / (RE + RED)` | registry only |
| Normalized Difference Built-up Index | `ndbi` (NDBI) | `(SWIR1 - NIR) / (SWIR1 + NIR)` | [25] |
| Bare Soil Index | `bsi` (BSI) | `((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))` | [26] |
| Normalized Difference Snow Index | `ndsi` (NDSI) | `(GREEN - SWIR1) / (GREEN + SWIR1)` | [27] |
| Normalized Burn Ratio | `nbr` (NBR) | `(NIR - SWIR2) / (NIR + SWIR2)`; the function takes the long SWIR band as `swir` | [28] |
| Normalized Burn Ratio 2 | `nbr2` (NBR2) | `(SWIR1 - SWIR2) / (SWIR1 + SWIR2)` | [28] |
| Differenced NBR | `dnbr` | `NBR_pre - NBR_post` | [28] |
| Relative dNBR | `rdnbr` | `dNBR / sqrt(abs(NBR_pre))` | [29] |
| Moisture Stress Index | `msi` (MSI) | `SWIR1 / NIR` | [30] |
| Radar Vegetation Index | `rvi` | `8 sigma_HV / (sigma_HH + sigma_VV + 2 sigma_HV)` (quad-polarisation) | [31] |
| Cross-polarisation ratio | `cross_pol_ratio` | `sigma_cross / sigma_co` (for example VH / VV) | not a published index |

`RE` is the first red-edge band (`REDEDGE1`, Sentinel-2 B05). The registry has 27 indices; the 7 spectral index families of the model zoo (`ndvi_calculator`, `evi_calculator`, `savi_calculator`, `ndwi_calculator`, `nbr_calculator`, `msi_calculator` and `vegetation_condition`, the last computing the Vegetation Condition Index `(NDVI - NDVImin) / (NDVImax - NDVImin)`) wrap such formulas as exact models.

### 4.3 Burn severity classes

`indices.burn_severity` classifies dNBR with the lower limits `BURN_SEVERITY_BREAKS = (-0.25, -0.1, 0.1, 0.27, 0.44, 0.66)` of Key and Benson [28] into the seven classes of `BURN_SEVERITY_CLASSES`: enhanced regrowth high, enhanced regrowth low, unburned, low, moderate-low, moderate-high and high severity (codes 0 to 6; -1 for NaN input).

## 5. Synthetic aperture radar

| Term | Definition | In Unbihexium |
| --- | --- | --- |
| Amplitude and intensity | Amplitude A is the magnitude of the complex radar signal; intensity (power) is A squared. | `sar.calibrate_amplitude` |
| beta0 | Radar brightness, backscatter normalised to the slant-range pixel area: `beta0 = abs(A)^2 / K`, with the calibration constant K [32]. | `sar.compute_beta0` |
| Coherence | Magnitude of the normalised complex correlation of two SLC images in a moving window, between 0 (decorrelated) and 1. | `sar.compute_coherence` |
| Decibel (dB) | Logarithmic unit of backscatter, `10 log10(p)`; non-positive powers give NaN unless a floor is set. | `sar.power_to_db`, `db_to_power`, `amplitude_to_db` |
| ENL (equivalent number of looks) | `mean^2 / variance` of intensity in a homogeneous area; a measure of speckle strength. | `sar.equivalent_number_of_looks` |
| gamma0 | Backscatter normalised to the area perpendicular to the look direction: `gamma0 = sigma0 / cos(theta)`. | `sar.compute_gamma0` |
| GRD (Ground Range Detected) | Sentinel-1 Level-1 product of detected (amplitude) data, multilooked and projected to ground range; phase is not preserved. The input to calibration and speckle filtering. | `sar.compute_sigma0` (real amplitude input) |
| H/A/alpha decomposition | Eigenvalue decomposition of the coherency matrix into entropy H, anisotropy A and mean scattering angle alpha [33]. | `sar.h_a_alpha`, `h_alpha_zones` |
| Height of ambiguity | Topographic height difference that produces one full phase cycle in an interferogram, depending on the perpendicular baseline. | `sar.height_of_ambiguity` |
| InSAR (interferometric SAR) | Use of the phase difference of two SLC acquisitions to measure topography or surface displacement; the interferogram is `s1 * conj(s2)`. | `sar.compute_interferogram`, `compute_displacement`, `los_to_vertical` |
| Incidence angle | Angle between the radar beam and the ellipsoid normal. Angles are in degrees by default; pass `degrees=False` for radians. | `degrees` argument of `sar` functions |
| Multilooking | Averaging of neighbouring pixels to reduce speckle at the cost of resolution. | `sar.multilook` |
| Phase unwrapping | Recovery of the absolute phase from the wrapped phase in (-pi, pi], implemented as least-squares and quality-guided methods [34]. | `sar.phase_unwrapping`, `wrap_phase`, `phase_residues` |
| Polarimetric decomposition | Separation of polarimetric backscatter into scattering mechanisms: Pauli, Freeman-Durden [35], Yamaguchi [36] and H/A/alpha. | `sar.pauli_decomposition`, `freeman_durden_decomposition`, `yamaguchi_decomposition` |
| Polarisation | Orientation of the transmitted and received waves: HH, HV, VH, VV. Co-polarised channels are HH and VV, cross-polarised HV and VH. | `indices.rvi`, `indices.cross_pol_ratio` |
| SAR (synthetic aperture radar) | Active microwave imaging radar that synthesises a long antenna from the motion of the platform, independent of daylight and clouds. | `unbihexium.sar` |
| sigma0 | Backscatter coefficient normalised to the ground area, the standard quantity for land and sea applications. As implemented: `sigma0 = abs(A)^2 / K * sin(theta)`, with the incidence angle theta [32]. | `sar.compute_sigma0`, `radiometric_calibration` |
| SLC (Single Look Complex) | Level-1 SAR product in slant-range geometry that keeps amplitude and phase as complex numbers; required for interferometry and polarimetry. | `sar.interferometry`, `sar.covariance_matrix` |
| Speckle | Granular multiplicative noise of coherent imaging. Reduced with the Lee [37], refined Lee, enhanced Lee, Frost [38], Kuan [39] and Gamma MAP filters. | `sar.lee_filter`, `refined_lee_filter`, `enhanced_lee_filter`, `frost_filter`, `kuan_filter`, `gamma_map_filter` |
| Goldstein filter | Adaptive frequency-domain filter of interferometric phase [40]. | `sar.goldstein_filter` |

## 6. Terrain and hydrology

| Term | Definition | In Unbihexium |
| --- | --- | --- |
| Aspect | Compass direction of the steepest downslope, in degrees clockwise from north (0 to 360). | `terrain.aspect` |
| Curvature | Second derivative of the surface: profile, plan and total curvature [41]. | `terrain.curvature`, `total_curvature` |
| D8 flow direction | Assignment of each cell's flow to the steepest of its eight neighbours [42]. | `terrain.flow_direction_d8`, `D8_NEIGHBOURS` |
| DEM (digital elevation model) | Raster of terrain heights. | input of `unbihexium.terrain` |
| Depression filling | Raising of closed depressions so that every cell drains to the edge (priority-flood) [43]. | `terrain.fill_depressions` |
| Flow accumulation | Number of upstream cells (or area) draining through each cell. | `terrain.flow_accumulation`, `extract_streams`, `watershed` |
| Hillshade | Illumination of the surface by a light source at a given azimuth and altitude, as floating-point values in [0, 255] [44]. | `terrain.hillshade` |
| Slope | Steepness of the surface in degrees, from the Horn gradient [44]. | `terrain.slope`, `gradient` |
| TPI (topographic position index) | Elevation of a cell minus the mean elevation of its neighbourhood. | `terrain.tpi` |
| TRI (terrain ruggedness index) | Ruggedness of the 3 x 3 window: by default the square root of the summed squared elevation differences to the eight neighbours (Riley et al.); `method="wilson"` gives their mean absolute difference. | `terrain.tri` |
| TWI (topographic wetness index) | `ln(a / tan(beta))` with specific catchment area a and slope beta [45]. | `terrain.twi` |
| Viewshed | Cells visible from an observer point at a given height. | `terrain.viewshed` |
| VRM (vector ruggedness measure) | Dispersion of the unit normal vectors of the surface in a window [46]. | `terrain.vrm` |

## 7. Geostatistics and spatial analysis

| Term | Definition | In Unbihexium |
| --- | --- | --- |
| AHP (analytic hierarchy process) | Derivation of criterion weights from a pairwise comparison matrix, with a consistency ratio (CR below 0.1 is usually accepted) [47]. | `analysis.AHP` |
| Cost distance | Accumulated cost of travelling from sources over a cost surface; the least-cost path follows its minimum. | `analysis.cost_distance`, `least_cost_path` |
| Cross-validation (kriging) | Leave-one-out or k-fold prediction of held-out points, reporting RMSE, MAE, bias and the mean standardised squared error (MSSE, near 1 when the kriging variance is well calibrated). | `geostat.OrdinaryKriging.cross_validate` |
| Geary's C | Global spatial autocorrelation statistic based on squared differences between neighbours [48]. | `geostat.gearys_c`, `GearysC` |
| Getis-Ord Gi* | Local statistic that identifies hot and cold spots as z-scores [49]. | `geostat.getis_ord_gi_star` |
| IDW (inverse distance weighting) | Interpolation by a weighted mean of observations with weights `1 / d^p` [50]. | `geostat.idw` |
| Kriging | Best linear unbiased prediction of a spatial variable from observations and a variogram [51]. Ordinary kriging assumes an unknown constant mean; universal kriging a polynomial trend in the coordinates. Each prediction comes with the kriging variance. | `geostat.OrdinaryKriging`, `UniversalKriging`, `KrigingResult` |
| LISA (local indicators of spatial association) | Local decomposition of Moran's I into one value per location [52]. | `geostat.local_morans_i` |
| Moran's I | Global spatial autocorrelation coefficient of values under a spatial weights matrix [53]. | `geostat.morans_i`, `MoransI`, `grid_morans_i` |
| Nugget, sill, range | Parameters of a variogram model: the nugget c0 is the discontinuity at the origin, the partial sill c the additional variance, the total sill c0 + c the plateau, and the range parameter a the distance scale. | `geostat.VariogramResult` |
| Spatial weights | Matrix of neighbour relations (contiguity, distance band, k nearest neighbours), optionally row-standardised. | `geostat.contiguity_weights`, `distance_band_weights`, `knn_weights`, `row_standardize` |
| Variogram (semivariogram) | Half the expected squared difference of values as a function of their separation distance h. The empirical variogram is estimated in distance bins with the Matheron or Cressie-Hawkins [54] estimator and fitted by a spherical, exponential, Gaussian, Matern, linear or power model. | `geostat.Variogram`, `empirical_variogram`, `variogram_function`, `VariogramModel` |
| Weighted overlay | Suitability analysis by a weighted sum of reclassified criterion rasters. | `analysis.WeightedOverlay`, `weighted_overlay`, `fuzzy_membership`, `reclassify` |
| Zonal statistics | Statistics of a value raster within each zone of a zone raster or polygon layer. | `analysis.zonal_statistics`, `ZonalStatistics`, `zonal_table` |

## 8. Accuracy and image quality metrics

| Term | Definition | In Unbihexium |
| --- | --- | --- |
| AP and mAP (mean average precision) | Area under the precision-recall curve of a detector for one class (AP), averaged over classes (mAP). `map50` uses an IoU threshold of 0.5 as in PASCAL VOC [55]; `map50_95` averages over IoU thresholds 0.5 to 0.95 in steps of 0.05 as in COCO [56]. | `ai.evaluation`, `unbihexium evaluate` |
| Confusion matrix | Table of counts of reference class against predicted class. `confusion_matrix` puts reference classes in rows; the area estimation functions follow Olofsson et al. [61] and put map classes in rows. | `metrics.confusion_matrix` |
| Dice coefficient | `2 TP / (2 TP + FP + FN)`, equal to F1 for binary masks. | `metrics.dice` |
| ERGAS | Relative dimensionless global error in synthesis, a pansharpening quality index [57]. | `metrics.ergas` |
| F1 score | Harmonic mean of precision and recall. | `metrics.f1_score` |
| IoU (intersection over union, Jaccard index) | `TP / (TP + FP + FN)` for masks, or the overlap area over the union area of two boxes. mIoU is the mean over classes. | `metrics.iou`, `mean_iou` |
| Kappa (Cohen's kappa) | Agreement between map and reference corrected for chance agreement [58]. | `metrics.cohen_kappa` |
| MAE, RMSE, bias, R squared | Mean absolute error, root mean square error, mean error and coefficient of determination of continuous predictions. | `metrics.mae`, `rmse`, `bias`, `r_squared`, `regression_report` |
| NMS (non-maximum suppression) | Removal of detection boxes that overlap a higher-scoring box by more than an IoU threshold. | `iou_threshold` of the detectors |
| Overall accuracy | Fraction of correctly classified samples. | `metrics.accuracy` |
| Precision and recall | `TP / (TP + FP)` and `TP / (TP + FN)`; for maps also called user's and producer's accuracy. | `metrics.precision`, `recall` |
| PSNR (peak signal-to-noise ratio) | `10 log10(R^2 / MSE)` with the data range R. | `metrics.psnr` |
| Q index | Universal image quality index of Wang and Bovik [59]. | `metrics.q_index` |
| SAM (spectral angle mapper) | Mean angle between predicted and reference spectra, in degrees. | `metrics.sam`, `spectral_angle` |
| SSIM (structural similarity) | Similarity of luminance, contrast and structure, computed with a Gaussian window (sigma 1.5) [60]. | `metrics.ssim` |
| Stratified area estimation | Unbiased area and accuracy estimates with confidence intervals from a stratified reference sample, following Olofsson et al. [61]. | `metrics.stratified_area_estimate`, `accuracy_assessment`, `sample_allocation` |

## 9. Machine learning and the model zoo

| Term | Definition | In Unbihexium |
| --- | --- | --- |
| Backend | Inference engine: `torch` (PyTorch) or `onnx` (ONNX Runtime); `auto` uses ONNX Runtime for `.onnx` files and PyTorch otherwise. | `--backend`, `ai.Predictor` |
| Capability | Named function of the library or model family, registered with a domain. The registry holds 147 capabilities (130 model families and 17 library features). | `registry.capabilities`, `unbihexium info` |
| Catalogue | The file `src/unbihexium/zoo/catalog.yaml` (version 2.0.0), the single source of truth of the model zoo: families, tasks, bands, outputs, units, required labels and suitable data. | `zoo.list_models`, `zoo.get_model`, `zoo.catalog_version` |
| CenterNet | Anchor-free detector that predicts object centres as heatmap peaks with size and offset, output stride 4 [62]; the architecture of the detection families. | `ai.models.networks` |
| Checkpoint | File `model.pt` or a training output `best.pt` / `last.pt` holding the configuration, weights and normalisation statistics as plain data. Loaded with `torch.load(weights_only=True)`, so it cannot execute code. | `zoo.load_model`, `unbihexium predict <checkpoint>` |
| Chip | Training sample cut from a larger image, `--chip-size` pixels square. | `ai.data`, `unbihexium train --chip-size` |
| Digest (weights digest) | SHA-256 over the sorted entries (key, shape and float32 little-endian bytes) of a model's state dictionary. The published digests of all 520 models are in `src/unbihexium/zoo/digests.json` and are checked whenever a starter model is built. Distinct from the file checksums in `model.sha256`. | `zoo.verify_model`, `unbihexium zoo verify` |
| EDSR | Enhanced deep residual super-resolution network with sub-pixel convolution [63]; the architecture of the super-resolution family. | `ai.models.networks` |
| Family | Model definition independent of size, for example `ship_detector`. There are 130 families. | `zoo/catalog.yaml` |
| Fine-tuning | Training a model starting from existing weights on new data. | `unbihexium train` |
| Local model store | Directory `$UNBIHEXIUM_CACHE/models/<model_id>/` (default `~/.cache/unbihexium/models/`) with `model.pt`, `config.json` and `model.sha256` of each built model. | `zoo.get_cache_dir`, `unbihexium zoo build`, `where`, `clear` |
| Manifest | JSON description of one family and its four variants under `model_zoo/manifests/`, validated against `model_zoo/manifest.schema.json`. Generated from the catalogue by `python -m unbihexium.zoo.sync`. | `model_zoo/manifests/` |
| Model card | Markdown description of a family (overview, inputs, outputs, variants, usage, training, suitable data, limitations and responsible use) under `model_zoo/cards/`, indexed in `model_zoo/MODEL_CARDS.md`. | `model_zoo/cards/` |
| Model id | `<family>_<variant>`, for example `ship_detector_tiny`. Model ids carry no version number. | `zoo.parse_model_id` |
| Model zoo | The 520 models (130 families in 4 variants) defined by the catalogue, with task APIs, training, evaluation and export. | `unbihexium.zoo`, `unbihexium.ai` |
| Normalisation statistics | Per-band mean and standard deviation estimated from the training data and stored in the checkpoint, so that inference and ONNX exports scale inputs identically. | `ai.training` |
| ONNX | Open Neural Network Exchange, a file format for trained networks [64]. Exports are compared with PyTorch in ONNX Runtime unless `--no-verify` is given, and carry their configuration, so ONNX inference needs no PyTorch. | `unbihexium zoo export`, extra `onnx` |
| Pipeline | Registered sequence of processing steps with step records, seeding and provenance. Steps return a mapping. | `core.Pipeline`, `unbihexium pipeline list`, `run` |
| Provenance record and evidence | Record of a pipeline run (inputs, outputs, models, configuration, environment) and SHA-256 evidence of the files involved. | `core.ProvenanceRecord`, `core.Evidence` |
| requires_training | Catalogue flag; true for every learned model and false for the 28 spectral index models. Returned by `unbihexium zoo info` and by the REST service. | `ModelZooEntry`, `/predict` response |
| Starter model | Complete, trainable network of a family with deterministic starter weights derived from the model id and verified against the published digest. Starter models have not been trained on Earth observation data and their predictions are not meaningful until they are trained. | `zoo.load_model`, `unbihexium zoo build` |
| Synthetic dataset | Generated images and labels for every trainable task, used to check a training setup without data (`--synthetic N`). A model trained on it is not useful for real imagery. | `ai.data`, `unbihexium train --synthetic` |
| Task | Kind of model output: detection, segmentation, change detection, dense regression, scene regression, enhancement, super-resolution or spectral index. | `zoo.Task` |
| Task API | Class that runs a zoo model on a raster and returns a georeferenced result, for example `ShipDetector` or `WaterDetector`. | `unbihexium.ai` |
| Tiled inference | Prediction on overlapping tiles (default overlap 0.25) that are blended into one output, so that images of any size can be processed. | `ai.Predictor`, `--tile-size`, `--overlap` |
| U-Net | Encoder-decoder network with skip connections [65]; the architecture of the segmentation, change detection, dense regression and enhancement families. | `ai.models.networks` |
| Variant | Size of a model: `tiny`, `base`, `large` or `mega`, differing in base channels (16, 32, 48, 64), encoder levels (3, 4, 4, 5), blocks per level (1, 1, 2, 2) and tile size (256, 256, 512, 512 px). Task APIs default to `base`. | `zoo.Variant`, `zoo.VariantSpec` |

## 10. Software, releases and supply chain

| Term | Definition | In Unbihexium |
| --- | --- | --- |
| Artifact attestation | Signed statement by GitHub Actions that a file was built by a given workflow run, verifiable with `gh attestation verify`. Created for the distributions of each release. | `.github/workflows/release.yml` |
| Extra | Optional dependency group of the package, for example `unbihexium[torch,onnx,serving]`. | `pyproject.toml` |
| Hashed lock file | Requirements file that pins every package with SHA-256 hashes and is installed with `pip install --require-hashes`. | `requirements.txt`, `requirements-dev.txt`, `.github/requirements/` |
| in-toto | Framework and attestation format for supply-chain metadata; SLSA provenance is an in-toto statement [66]. | `unbihexium-<tag>.intoto.jsonl` |
| MPL-2.0 | Mozilla Public License 2.0, the file-level copyleft licence of the project [67]. Releases v1.0.0 and v1.0.1 were published under Apache-2.0. | `LICENSE.txt` |
| OpenSSF Scorecard | Automated assessment of the security practices of a repository [68]. | `.github/workflows/scorecard.yml` |
| PEP 440 | Python's version specification; release versions of the project are `MAJOR.MINOR.PATCH` [69]. | [VERSIONING.md](../VERSIONING.md) |
| REUSE | Specification for declaring the copyright and licence of every file [70]. | `REUSE.toml` |
| SBOM (software bill of materials) | Inventory of the components of a software artefact. An SPDX [71] SBOM is generated for every release and for every pushed container image, and attested with a signed attestation. | `.github/workflows/release.yml`, `.github/workflows/docker.yml` |
| Security Insights | Machine-readable security metadata of the project in the OpenSSF Security Insights format. | `security-insights.yml` |
| Semantic Versioning | Version scheme in which the major number changes for incompatible changes [72]. | [VERSIONING.md](../VERSIONING.md) |
| Sigstore | Keyless signing of artefacts with short-lived certificates bound to an identity and a public transparency log [73]. Release distributions are signed; the `.sigstore.json` bundle holds the signature, certificate and log entry. | `.github/workflows/release.yml`, [SECURITY.md](../SECURITY.md) |
| SLSA (Supply-chain Levels for Software Artifacts) | Framework of supply-chain requirements; its provenance records how and from which source an artefact was built [74]. Each release from the current workflow carries `unbihexium-<tag>.intoto.jsonl`. Releases v1.0.0 and v1.0.1 predate signing and provenance. | `.github/workflows/release.yml` |

## References

[1] Chavez, P. S. An improved dark-object subtraction technique for atmospheric scattering correction of multispectral data. Remote Sensing of Environment 24(3), 459-479. 1988. <https://doi.org/10.1016/0034-4257(88)90019-3>

[2] Open Geospatial Consortium. OGC Cloud Optimized GeoTIFF Standard, version 1.0. 2023. <https://docs.ogc.org/is/21-026/21-026.html>

[3] International Association of Oil and Gas Producers. EPSG Geodetic Parameter Dataset. 2026. <https://epsg.org/>

[4] Butler, H., Daly, M., Doyle, A., Gillies, S., Hagen, S. and Schaub, T. RFC 7946: The GeoJSON Format. IETF. 2016. <https://www.rfc-editor.org/rfc/rfc7946>

[5] Open Geospatial Consortium. GeoParquet specification. 2024. <https://github.com/opengeospatial/geoparquet>

[6] Open Geospatial Consortium. OGC GeoTIFF Standard, version 1.1. 2019. <https://docs.ogc.org/is/19-008r4/19-008r4.html>

[7] STAC contributors. SpatioTemporal Asset Catalog specification. 2025. <https://github.com/radiantearth/stac-spec>

[8] Zarr developers. Zarr specifications. 2024. <https://github.com/zarr-developers/zarr-specs>

[9] Rouse, J. W., Haas, R. H., Schell, J. A. and Deering, D. W. Monitoring vegetation systems in the Great Plains with ERTS. Third Earth Resources Technology Satellite-1 Symposium, NASA SP-351, 309-317. 1974. <https://ntrs.nasa.gov/citations/19740022614>

[10] Gitelson, A. A., Kaufman, Y. J. and Merzlyak, M. N. Use of a green channel in remote sensing of global vegetation from EOS-MODIS. Remote Sensing of Environment 58(3), 289-298. 1996. <https://doi.org/10.1016/S0034-4257(96)00072-7>

[11] Barnes, E. M. et al. Coincident detection of crop water stress, nitrogen status and canopy density using ground-based multispectral data. Proceedings of the 5th International Conference on Precision Agriculture. 2000.

[12] Huete, A., Didan, K., Miura, T., Rodriguez, E. P., Gao, X. and Ferreira, L. G. Overview of the radiometric and biophysical performance of the MODIS vegetation indices. Remote Sensing of Environment 83(1-2), 195-213. 2002. <https://doi.org/10.1016/S0034-4257(02)00096-2>

[13] Jiang, Z., Huete, A. R., Didan, K. and Miura, T. Development of a two-band enhanced vegetation index without a blue band. Remote Sensing of Environment 112(10), 3833-3845. 2008. <https://doi.org/10.1016/j.rse.2008.06.006>

[14] Huete, A. R. A soil-adjusted vegetation index (SAVI). Remote Sensing of Environment 25(3), 295-309. 1988. <https://doi.org/10.1016/0034-4257(88)90106-X>

[15] Rondeaux, G., Steven, M. and Baret, F. Optimization of soil-adjusted vegetation indices. Remote Sensing of Environment 55(2), 95-107. 1996. <https://doi.org/10.1016/0034-4257(95)00186-7>

[16] Qi, J., Chehbouni, A., Huete, A. R., Kerr, Y. H. and Sorooshian, S. A modified soil adjusted vegetation index. Remote Sensing of Environment 48(2), 119-126. 1994. <https://doi.org/10.1016/0034-4257(94)90134-1>

[17] Kaufman, Y. J. and Tanre, D. Atmospherically resistant vegetation index (ARVI) for EOS-MODIS. IEEE Transactions on Geoscience and Remote Sensing 30(2), 261-270. 1992. <https://doi.org/10.1109/36.134076>

[18] Gitelson, A. A., Stark, R., Grits, U., Rundquist, D., Kaufman, Y. and Derry, D. Vegetation and soil lines in visible spectral space. International Journal of Remote Sensing 23(13), 2537-2562. 2002. <https://doi.org/10.1080/01431160110107806>

[19] Camps-Valls, G. et al. A unified vegetation index for quantifying the terrestrial biosphere. Science Advances 7(9), eabc7447. 2021. <https://doi.org/10.1126/sciadv.abc7447>

[20] Gitelson, A. A., Gritz, Y. and Merzlyak, M. N. Relationships between leaf chlorophyll content and spectral reflectance. Journal of Plant Physiology 160(3), 271-282. 2003. <https://doi.org/10.1078/0176-1617-00887>

[21] McFeeters, S. K. The use of the normalized difference water index (NDWI) in the delineation of open water features. International Journal of Remote Sensing 17(7), 1425-1432. 1996. <https://doi.org/10.1080/01431169608948714>

[22] Xu, H. Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. International Journal of Remote Sensing 27(14), 3025-3033. 2006. <https://doi.org/10.1080/01431160600589179>

[23] Gao, B.-C. NDWI: a normalized difference water index for remote sensing of vegetation liquid water from space. Remote Sensing of Environment 58(3), 257-266. 1996. <https://doi.org/10.1016/S0034-4257(96)00067-3>

[24] Feyisa, G. L., Meilby, H., Fensholt, R. and Proud, S. R. Automated water extraction index: a new technique for surface water mapping using Landsat imagery. Remote Sensing of Environment 140, 23-35. 2014. <https://doi.org/10.1016/j.rse.2013.08.029>

[25] Zha, Y., Gao, J. and Ni, S. Use of normalized difference built-up index in automatically mapping urban areas from TM imagery. International Journal of Remote Sensing 24(3), 583-594. 2003. <https://doi.org/10.1080/01431160304987>

[26] Rikimaru, A., Roy, P. S. and Miyatake, S. Tropical forest cover density mapping. Tropical Ecology 43(1), 39-47. 2002.

[27] Hall, D. K., Riggs, G. A. and Salomonson, V. V. Development of methods for mapping global snow cover using moderate resolution imaging spectroradiometer data. Remote Sensing of Environment 54(2), 127-140. 1995. <https://doi.org/10.1016/0034-4257(95)00137-P>

[28] Key, C. H. and Benson, N. C. Landscape assessment: ground measure of severity, the Composite Burn Index, and remote sensing of severity, the Normalized Burn Ratio. USDA Forest Service General Technical Report RMRS-GTR-164-CD. 2006.

[29] Miller, J. D. and Thode, A. E. Quantifying burn severity in a heterogeneous landscape with a relative version of the delta Normalized Burn Ratio (dNBR). Remote Sensing of Environment 109(1), 66-80. 2007. <https://doi.org/10.1016/j.rse.2006.12.006>

[30] Hunt, E. R. and Rock, B. N. Detection of changes in leaf water content using near- and middle-infrared reflectances. Remote Sensing of Environment 30(1), 43-54. 1989. <https://doi.org/10.1016/0034-4257(89)90046-1>

[31] Kim, Y. and van Zyl, J. J. A time-series approach to estimate soil moisture using polarimetric radar data. IEEE Transactions on Geoscience and Remote Sensing 47(8), 2519-2527. 2009. <https://doi.org/10.1109/TGRS.2009.2014944>

[32] Miranda, N. and Meadows, P. J. Radiometric calibration of S-1 Level-1 products generated by the S-1 IPF. European Space Agency, ESA-EOPG-CSCOP-TN-0002, issue 1.0. 2015.

[33] Cloude, S. R. and Pottier, E. An entropy based classification scheme for land applications of polarimetric SAR. IEEE Transactions on Geoscience and Remote Sensing 35(1), 68-78. 1997. <https://doi.org/10.1109/36.551935>

[34] Ghiglia, D. C. and Pritt, M. D. Two-Dimensional Phase Unwrapping: Theory, Algorithms, and Software. Wiley. 1998.

[35] Freeman, A. and Durden, S. L. A three-component scattering model for polarimetric SAR data. IEEE Transactions on Geoscience and Remote Sensing 36(3), 963-973. 1998. <https://doi.org/10.1109/36.673687>

[36] Yamaguchi, Y., Moriyama, T., Ishido, M. and Yamada, H. Four-component scattering model for polarimetric SAR image decomposition. IEEE Transactions on Geoscience and Remote Sensing 43(8), 1699-1706. 2005. <https://doi.org/10.1109/TGRS.2005.852084>

[37] Lee, J.-S. Digital image enhancement and noise filtering by use of local statistics. IEEE Transactions on Pattern Analysis and Machine Intelligence 2(2), 165-168. 1980. <https://doi.org/10.1109/TPAMI.1980.4766994>

[38] Frost, V. S., Stiles, J. A., Shanmugan, K. S. and Holtzman, J. C. A model for radar images and its application to adaptive digital filtering of multiplicative noise. IEEE Transactions on Pattern Analysis and Machine Intelligence 4(2), 157-166. 1982. <https://doi.org/10.1109/TPAMI.1982.4767223>

[39] Kuan, D. T., Sawchuk, A. A., Strand, T. C. and Chavel, P. Adaptive noise smoothing filter for images with signal-dependent noise. IEEE Transactions on Pattern Analysis and Machine Intelligence 7(2), 165-177. 1985. <https://doi.org/10.1109/TPAMI.1985.4767641>

[40] Goldstein, R. M. and Werner, C. L. Radar interferogram filtering for geophysical applications. Geophysical Research Letters 25(21), 4035-4038. 1998. <https://doi.org/10.1029/1998GL900033>

[41] Zevenbergen, L. W. and Thorne, C. R. Quantitative analysis of land surface topography. Earth Surface Processes and Landforms 12(1), 47-56. 1987. <https://doi.org/10.1002/esp.3290120107>

[42] O'Callaghan, J. F. and Mark, D. M. The extraction of drainage networks from digital elevation data. Computer Vision, Graphics, and Image Processing 28(3), 323-344. 1984. <https://doi.org/10.1016/S0734-189X(84)80011-0>

[43] Barnes, R., Lehman, C. and Mulla, D. Priority-flood: an optimal depression-filling and watershed-labeling algorithm for digital elevation models. Computers and Geosciences 62, 117-127. 2014. <https://doi.org/10.1016/j.cageo.2013.04.024>

[44] Horn, B. K. P. Hill shading and the reflectance map. Proceedings of the IEEE 69(1), 14-47. 1981. <https://doi.org/10.1109/PROC.1981.11918>

[45] Beven, K. J. and Kirkby, M. J. A physically based, variable contributing area model of basin hydrology. Hydrological Sciences Bulletin 24(1), 43-69. 1979. <https://doi.org/10.1080/02626667909491834>

[46] Sappington, J. M., Longshore, K. M. and Thompson, D. B. Quantifying landscape ruggedness for animal habitat analysis: a case study using bighorn sheep in the Mojave Desert. Journal of Wildlife Management 71(5), 1419-1426. 2007. <https://doi.org/10.2193/2005-723>

[47] Saaty, T. L. A scaling method for priorities in hierarchical structures. Journal of Mathematical Psychology 15(3), 234-281. 1977. <https://doi.org/10.1016/0022-2496(77)90033-5>

[48] Geary, R. C. The contiguity ratio and statistical mapping. The Incorporated Statistician 5(3), 115-145. 1954. <https://doi.org/10.2307/2986645>

[49] Ord, J. K. and Getis, A. Local spatial autocorrelation statistics: distributional issues and an application. Geographical Analysis 27(4), 286-306. 1995. <https://doi.org/10.1111/j.1538-4632.1995.tb00912.x>

[50] Shepard, D. A two-dimensional interpolation function for irregularly-spaced data. Proceedings of the 23rd ACM National Conference, 517-524. 1968. <https://doi.org/10.1145/800186.810616>

[51] Matheron, G. Principles of geostatistics. Economic Geology 58(8), 1246-1266. 1963. <https://doi.org/10.2113/gsecongeo.58.8.1246>

[52] Anselin, L. Local indicators of spatial association: LISA. Geographical Analysis 27(2), 93-115. 1995. <https://doi.org/10.1111/j.1538-4632.1995.tb00338.x>

[53] Moran, P. A. P. Notes on continuous stochastic phenomena. Biometrika 37(1-2), 17-23. 1950. <https://doi.org/10.1093/biomet/37.1-2.17>

[54] Cressie, N. and Hawkins, D. M. Robust estimation of the variogram: I. Mathematical Geology 12(2), 115-125. 1980. <https://doi.org/10.1007/BF01035243>

[55] Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A. The PASCAL visual object classes (VOC) challenge. International Journal of Computer Vision 88(2), 303-338. 2010. <https://doi.org/10.1007/s11263-009-0275-4>

[56] Lin, T.-Y. et al. Microsoft COCO: common objects in context. arXiv:1405.0312. 2014. <https://arxiv.org/abs/1405.0312>

[57] Wald, L. Data Fusion: Definitions and Architectures. Fusion of Images of Different Spatial Resolutions. Les Presses de l'Ecole des Mines, Paris. 2002.

[58] Cohen, J. A coefficient of agreement for nominal scales. Educational and Psychological Measurement 20(1), 37-46. 1960. <https://doi.org/10.1177/001316446002000104>

[59] Wang, Z. and Bovik, A. C. A universal image quality index. IEEE Signal Processing Letters 9(3), 81-84. 2002. <https://doi.org/10.1109/97.995823>

[60] Wang, Z., Bovik, A. C., Sheikh, H. R. and Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600-612. 2004. <https://doi.org/10.1109/TIP.2003.819861>

[61] Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E. and Wulder, M. A. Good practices for estimating area and assessing accuracy of land change. Remote Sensing of Environment 148, 42-57. 2014. <https://doi.org/10.1016/j.rse.2014.02.015>

[62] Zhou, X., Wang, D. and Kraehenbuehl, P. Objects as points. arXiv:1904.07850. 2019. <https://arxiv.org/abs/1904.07850>

[63] Lim, B., Son, S., Kim, H., Nah, S. and Lee, K. M. Enhanced deep residual networks for single image super-resolution. CVPR Workshops. 2017. <https://arxiv.org/abs/1707.02921>

[64] ONNX project. Open Neural Network Exchange. 2026. <https://onnx.ai/>

[65] Ronneberger, O., Fischer, P. and Brox, T. U-Net: convolutional networks for biomedical image segmentation. MICCAI 2015, LNCS 9351, 234-241. 2015. <https://arxiv.org/abs/1505.04597>

[66] in-toto project. in-toto attestation framework. 2026. <https://github.com/in-toto/attestation>

[67] Mozilla Foundation. Mozilla Public License, version 2.0. 2012. <https://mozilla.org/MPL/2.0/>

[68] OpenSSF. OpenSSF Scorecard. 2026. <https://scorecard.dev/>

[69] Coghlan, N. and Stufft, D. PEP 440: Version Identification and Dependency Specification. Python Software Foundation. 2013. <https://peps.python.org/pep-0440/>

[70] Free Software Foundation Europe. REUSE Specification, version 3.3. 2024. <https://reuse.software/spec-3.3/>

[71] Linux Foundation. SPDX Specification, version 2.3. 2022. <https://spdx.github.io/spdx-spec/v2.3/>

[72] Preston-Werner, T. Semantic Versioning 2.0.0. 2013. <https://semver.org/spec/v2.0.0.html>

[73] Sigstore project. Sigstore documentation. 2026. <https://docs.sigstore.dev/>

[74] OpenSSF. Supply-chain Levels for Software Artifacts (SLSA), specification version 1.0. 2023. <https://slsa.dev/spec/v1.0/>

<!--
=============================================================================
End of file docs/glossary.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
