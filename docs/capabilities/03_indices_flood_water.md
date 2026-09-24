<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/capabilities/03_indices_flood_water.md
Title       : Capability domain 03: spectral indices, floods and water
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Capability domain 03: spectral indices, floods and water

| Field | Value |
| --- | --- |
| Document | UBX-DOC-CAP-03 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | The main branch of Unbihexium (declared version 1.0.1, model catalogue 2.0.0) |

## Abstract

This document describes capability domain 03, "spectral indices, floods and water". It covers the model families of the catalogue domains `indices` and `water`, the spectral index functions of `unbihexium.indices` and `unbihexium.core.index` (registry capability `spectral_indices`), the `unbihexium index` command, and the hydrological functions of `unbihexium.terrain` that support flood and watershed analysis. It is written for users who compute indices or map water, for contributors and for reviewers. Every formula is given as implemented, with its primary source; the fifteen model families are listed in tables generated from the catalogue; all examples were executed. The seven spectral index families (28 models) compute exact formulas and need no training; the eight water families are untrained starter models, and the domain describes their intended applications, not validated products.

## Contents

1. [Scope and status](#1-scope-and-status)
2. [Place in the registry and the catalogue](#2-place-in-the-registry-and-the-catalogue)
3. [Model families](#3-model-families)
4. [Spectral index functions](#4-spectral-index-functions)
5. [Hydrological terrain analysis](#5-hydrological-terrain-analysis)
6. [Examples](#6-examples)
7. [Limitations and responsible use](#7-limitations-and-responsible-use)
8. [Related documents](#8-related-documents)
9. [References](#references)

## 1. Scope and status

### 1.1 What the domain covers

- **Spectral indices.** Band arithmetic on surface reflectance: vegetation, water, moisture, built-up, snow and burn indices, the burn severity classes of dNBR, and two radar ratios. They are available as NumPy functions, as a registry of named indices with sensor band mapping, on the command line, and as seven model families that wrap a formula as a network so that it runs in the tiled inference pipeline and exports to ONNX.
- **Water and floods.** Eight learned model families for water surfaces, reservoirs, marine pollution, flood susceptibility and depth, runoff, water quality and bathymetry, and the deterministic hydrology of a DEM: depression filling, D8 flow routing, flow accumulation, watersheds, stream extraction and the topographic wetness index.

Flood mapping from radar backscatter (`sar_flood_detector`, task class `FloodMapper`) belongs to the SAR domain and is described in [domain 12](12_radar_sar.md).

### 1.2 Status

The index functions, the hydrology and the seven spectral index families are exact, deterministic implementations of published formulas; the index families have no trainable parameters (0 parameters in every variant) and their four variants give identical results. The eight water families are **untrained starter models** with deterministic initial weights; their predictions are meaningless until they are trained on reference data (Section 6.4 shows an example). No accuracy figures are published for any model.

### 1.3 Changes from the previous version

Version 1 of this document described "CNN architectures" and parameter counts for the index calculators, performance tables, a Temperature Condition Index, a Vegetation Health Index and a flood probability formula. The index families are exact formula modules without parameters; there is no code for TCI, VHI or a flood probability formula; no performance was measured. `flood_risk` and `flood_risk_assessor` are listed here, in the domain of their catalogue entry, and not repeated in the risk domain. These parts were removed.

## 2. Place in the registry and the catalogue

The enumeration `unbihexium.registry.CapabilityDomain` has the members `INDICES = "indices"` and `WATER = "water"`.

| Registry domain | Model capabilities (families) | Library capabilities | Total |
| --- | --- | --- | --- |
| `indices` | 7, maturity `stable`, tag `requires_training` = `false` | `spectral_indices` (`unbihexium.indices`), `stable` | 8 |
| `water` | 8, maturity `beta`, tag `requires_training` = `true` | none | 8 |

The family `water_surface_detector` runs in the registered pipeline `water_detection` (task class `WaterDetector`). The hydrological functions of Section 5 are part of the library capability `terrain_analysis`, which the registry files under the domain `analysis` ([domain 02](02_tourism_data_processing.md)).

## 3. Model families

### 3.1 Inventory

The tables were generated from the catalogue and the model registry of the installed package with the script of [index.md, Section 4](index.md#4-regenerating-the-family-tables), run with the arguments `indices water`. Input bands are listed in channel order; units are given in brackets.

| Family | Domain | Task | Input bands | Outputs | Parameters (tiny / base / large / mega) |
| --- | --- | --- | --- | --- | --- |
| `evi_calculator` | indices | spectral_index | blue, red, nir | evi | 0 / 0 / 0 / 0 |
| `msi_calculator` | indices | spectral_index | nir, swir1 | msi | 0 / 0 / 0 / 0 |
| `nbr_calculator` | indices | spectral_index | nir, swir2 | nbr | 0 / 0 / 0 / 0 |
| `ndvi_calculator` | indices | spectral_index | red, nir | ndvi | 0 / 0 / 0 / 0 |
| `ndwi_calculator` | indices | spectral_index | green, nir | ndwi | 0 / 0 / 0 / 0 |
| `savi_calculator` | indices | spectral_index | red, nir | savi | 0 / 0 / 0 / 0 |
| `vegetation_condition` | indices | spectral_index | ndvi, ndvi_min, ndvi_max | vci | 0 / 0 / 0 / 0 |
| `marine_pollution_detector` | water | segmentation | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | water, floating_debris, oil_sheen | 733,971 / 7,060,515 / 22,062,195 / 60,454,723 |
| `reservoir_monitor` | water | segmentation | blue, green, red, nir | background, water | 733,090 / 7,058,754 / 22,059,554 / 60,451,202 |
| `water_surface_detector` | water | segmentation | blue, green, red, nir | background, water | 733,090 / 7,058,754 / 22,059,554 / 60,451,202 |
| `flood_risk` | water | dense_regression | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, elevation, slope | flood_susceptibility [1] | 734,225 / 7,061,025 / 22,062,961 / 60,455,745 |
| `water_quality_assessor` | water | dense_regression | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | chlorophyll_a, turbidity [mg m-3, FNU] | 733,954 / 7,060,482 / 22,062,146 / 60,454,658 |
| `flood_risk_assessor` | water | dense_regression | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, elevation, slope | water_depth [m] | 734,225 / 7,061,025 / 22,062,961 / 60,455,745 |
| `watershed_manager` | water | dense_regression | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, elevation, slope | runoff_coefficient [1] | 734,225 / 7,061,025 / 22,062,961 / 60,455,745 |
| `offshore_survey` | water | dense_regression | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | depth [m] | 733,937 / 7,060,449 / 22,062,097 / 60,454,593 |

| Family | Name | Intended application | Reference data needed for training |
| --- | --- | --- | --- |
| `evi_calculator` | EVI Calculator | Enhanced Vegetation Index, 2.5 (NIR - RED) / (NIR + 6 RED - 7.5 BLUE + 1) (Huete et al., 2002). | None, the formula needs no training. |
| `msi_calculator` | MSI Calculator | Moisture Stress Index, SWIR1 / NIR (Rock et al., 1986). | None, the formula needs no training. |
| `nbr_calculator` | NBR Calculator | Normalized Burn Ratio, (NIR - SWIR2) / (NIR + SWIR2) (Key and Benson, 2006). | None, the formula needs no training. |
| `ndvi_calculator` | NDVI Calculator | Normalized Difference Vegetation Index, (NIR - RED) / (NIR + RED) (Rouse et al., 1974). | None, the formula needs no training. |
| `ndwi_calculator` | NDWI Calculator | Normalized Difference Water Index, (GREEN - NIR) / (GREEN + NIR) (McFeeters, 1996). | None, the formula needs no training. |
| `savi_calculator` | SAVI Calculator | Soil Adjusted Vegetation Index, 1.5 (NIR - RED) / (NIR + RED + 0.5) (Huete, 1988). | None, the formula needs no training. |
| `vegetation_condition` | Vegetation Condition Index | Vegetation Condition Index, (NDVI - NDVImin) / (NDVImax - NDVImin) (Kogan, 1995). | None, the formula needs no training. |
| `marine_pollution_detector` | Marine Pollution Detector | Segments floating debris and surface pollution at sea. | Pollution masks. |
| `reservoir_monitor` | Reservoir Monitor | Segments the water surface of reservoirs to track storage. | Water masks. |
| `water_surface_detector` | Water Surface Detector | Segments open water surfaces. | Water masks. |
| `flood_risk` | Flood Risk | Estimates flood susceptibility per pixel. | Flood susceptibility from historical flood extents or hydraulic models. |
| `water_quality_assessor` | Water Quality Assessor | Estimates chlorophyll-a concentration and turbidity in water bodies. | In situ water quality samples. |
| `flood_risk_assessor` | Flood Depth Estimator | Estimates flood water depth. | Flood depth rasters from hydraulic models or surveys. |
| `watershed_manager` | Runoff Estimator | Estimates a runoff coefficient per pixel for watershed management. | Runoff coefficients from hydrological models. |
| `offshore_survey` | Bathymetry Estimator | Estimates shallow water depth (satellite-derived bathymetry). | Echo sounding or LiDAR bathymetry. |

For the water families, "Intended application" describes what a model does once it has been trained.

### 3.2 Spectral index families

The index families are instances of `unbihexium.ai.models.spectral.SpectralIndex`, a PyTorch module without trainable parameters. The input channels are surface reflectance in the order of the catalogue bands (Section 3.1); the output is one channel with NaN where the denominator is zero. The formulas are

$$
\mathrm{NDVI} = \frac{\rho_{NIR} - \rho_{R}}{\rho_{NIR} + \rho_{R}}, \quad
\mathrm{NDWI} = \frac{\rho_{G} - \rho_{NIR}}{\rho_{G} + \rho_{NIR}}, \quad
\mathrm{EVI} = 2.5\,\frac{\rho_{NIR} - \rho_{R}}{\rho_{NIR} + 6\rho_{R} - 7.5\rho_{B} + 1}, \quad
\mathrm{SAVI} = 1.5\,\frac{\rho_{NIR} - \rho_{R}}{\rho_{NIR} + \rho_{R} + 0.5},
$$

$$
\mathrm{MSI} = \frac{\rho_{SWIR1}}{\rho_{NIR}}, \quad
\mathrm{NBR} = \frac{\rho_{NIR} - \rho_{SWIR2}}{\rho_{NIR} + \rho_{SWIR2}}, \quad
\mathrm{VCI} = \frac{\mathrm{NDVI} - \mathrm{NDVI}_{\min}}{\mathrm{NDVI}_{\max} - \mathrm{NDVI}_{\min}}
$$

after Rouse et al. [1], McFeeters [2], Huete et al. [3], Huete [4], Rock et al. [5], Key and Benson [6] and Kogan [7]. For VCI, the per-pixel minimum and maximum are the multi-year extremes of NDVI for the same period of the year, which the user computes from an NDVI time series. VCI is returned as a fraction; Kogan's original definition multiplies it by 100.

### 3.3 Water families

The segmentation families (`unet`) return class logits $(N, K, H, W)$; `SemanticSegmenter` and `WaterDetector` turn them into a class map (Section 4.1 of [domain 01](01_ai_products.md#41-task-interfaces)). The regression families (`unet_regression`) return one channel per target in the listed units; `flood_risk` and `watershed_manager` have the value range $[0, 1]$ and end in a sigmoid. Several families expect co-registered elevation and slope bands, which `unbihexium.terrain.slope` can derive from a DEM.

## 4. Spectral index functions

### 4.1 Interfaces

The library offers three equivalent ways to compute an index; the command line and the index registry share one implementation, the array functions another.

| Interface | Input | Coverage |
| --- | --- | --- |
| `unbihexium.indices`: one function per index (`ndvi(nir, red)`, `evi(nir, red, blue, g, c1, c2, l)`, ...) and `compute_index(name, **bands)` | NumPy arrays or scalars of reflectance, broadcast together | 25 indices by name (`INDEX_FUNCTIONS`), plus `dnbr`, `rdnbr`, `burn_severity`, `cross_pol_ratio`, `normalized_difference` and `safe_divide` |
| `unbihexium.core.index`: `IndexRegistry`, `compute_index(name, bands, sensor=None, nodata=None, **parameters)`, `compute_indices`, `dnbr`, `classify_burn_severity` | dictionary of arrays keyed by common band names (`BLUE`, `RED`, `NIR`, `SWIR1`, ...) or, with `sensor`, by product band names (`B04`, `SR_B4`) of `sentinel2_msi`, `landsat8_oli`, `landsat9_oli2` | 27 registered indices with formula, bands, parameters, value range and reference |
| `unbihexium index NAME -i INPUT -o OUTPUT` | multi-band raster; band numbers set with `--blue`, `--green`, `--red`, `--nir`, `--swir1`, `--swir2`, `--coastal`, `--rededge1` to `--rededge3`, `--nir08` | the 27 indices of `IndexRegistry`; defaults follow the 13-band Sentinel-2 order (B01 to B08, B8A, B09 to B12) |

All functions return float64 (the command line writes float32 GeoTIFF). A ratio is NaN where its denominator is zero or an input is not finite, so that no artificial values enter later statistics.

### 4.2 Formulas

$\rho$ denotes reflectance in $[0, 1]$; $B$, $G$, $R$, $RE$, $NIR$, $SWIR1$ (about 1.6 micrometres) and $SWIR2$ (about 2.2 micrometres) the bands. "Both" means that the index is available in `unbihexium.indices` and in `IndexRegistry`.

| Index | Formula as implemented | Available in | Source |
| --- | --- | --- | --- |
| NDVI | $(NIR - R)/(NIR + R)$ | both | [1] |
| GNDVI | $(NIR - G)/(NIR + G)$ | both | [8] |
| NDRE | $(NIR - RE)/(NIR + RE)$ (red edge 1) | both | [9], [10] |
| EVI | $G_f (NIR - R)/(NIR + C_1 R - C_2 B + L)$, $G_f = 2.5$, $C_1 = 6$, $C_2 = 7.5$, $L = 1$ | both | [3] |
| EVI2 | $2.5 (NIR - R)/(NIR + 2.4 R + 1)$ | both | [11] |
| SAVI | $(1 + L)(NIR - R)/(NIR + R + L)$, $L = 0.5$ | both | [4] |
| OSAVI | $(NIR - R)/(NIR + R + 0.16)$ | both | [12] |
| MSAVI (MSAVI2) | $\left(2 NIR + 1 - \sqrt{(2 NIR + 1)^2 - 8 (NIR - R)}\right)/2$ | both | [13] |
| ARVI | $(NIR - RB)/(NIR + RB)$, $RB = R - \gamma (B - R)$, $\gamma = 1$ | both | [14] |
| VARI | $(G - R)/(G + R - B)$ | both | [15] |
| kNDVI | $\tanh(\mathrm{NDVI}^2)$ (RBF kernel with $\sigma = (NIR + R)/2$) | `unbihexium.indices` | [16] |
| SR | $NIR / R$ | `IndexRegistry` | [17] |
| WDRVI | $(\alpha NIR - R)/(\alpha NIR + R)$, $\alpha = 0.1$ | `IndexRegistry` | [18] |
| CIgreen | $NIR / G - 1$ | both (`ci_green`) | [19] |
| CIre | $NIR / RE - 1$ | both (`ci_rededge`) | [19] |
| NDWI | $(G - NIR)/(G + NIR)$ | both | [2] |
| MNDWI | $(G - SWIR1)/(G + SWIR1)$ | both | [20] |
| NDMI | $(NIR - SWIR1)/(NIR + SWIR1)$ (the NDWI of Gao) | both | [21] |
| AWEInsh | $4 (G - SWIR1) - (0.25 NIR + 2.75 SWIR2)$ | both | [22] |
| AWEIsh | $B + 2.5 G - 1.5 (NIR + SWIR1) - 0.25 SWIR2$ | both | [22] |
| NDTI | $(R - G)/(R + G)$ | `IndexRegistry` | [23] |
| NDCI | $(RE - R)/(RE + R)$ | `IndexRegistry` | [24] |
| NDBI | $(SWIR1 - NIR)/(SWIR1 + NIR)$ | both | [25] |
| BSI | $((SWIR1 + R) - (NIR + B))/((SWIR1 + R) + (NIR + B))$ | both | [26] |
| NDSI | $(G - SWIR1)/(G + SWIR1)$ | both | [27] |
| NBR | $(NIR - SWIR2)/(NIR + SWIR2)$ | both | [6] |
| NBR2 | $(SWIR1 - SWIR2)/(SWIR1 + SWIR2)$ | both | [37] |
| MSI | $SWIR1 / NIR$ | both | [5], [28] |
| RVI (radar) | $8 \sigma^0_{HV} / (\sigma^0_{HH} + \sigma^0_{VV} + 2 \sigma^0_{HV})$, linear backscatter | `unbihexium.indices` | [29] |

The parameters of EVI, SAVI, ARVI and WDRVI can be changed: as keyword arguments of the array functions (`g`, `c1`, `c2`, `l`, `gamma`) or of `unbihexium.core.index.compute_index` (`G`, `C1`, `C2`, `L`, `gamma`, `alpha`). The function `cross_pol_ratio(sigma_cross, sigma_co)` returns the linear ratio, for example VH/VV.

### 4.3 Burn severity

The differenced and relative differenced burn ratios are

$$
\mathrm{dNBR} = \mathrm{NBR}_{pre} - \mathrm{NBR}_{post}, \qquad
\mathrm{RdNBR} = \frac{\mathrm{dNBR}}{\sqrt{\lvert \mathrm{NBR}_{pre} \rvert}}
$$

after Key and Benson [6] and Miller and Thode [30]. The code works with unscaled NBR values; Miller and Thode multiply NBR by 1000, so thresholds taken from their paper must be rescaled. `burn_severity(dnbr)` (and `classify_burn_severity` in `unbihexium.core.index`) returns the seven classes of Key and Benson with the lower limits $-0.25$, $-0.1$, $0.1$, $0.27$, $0.44$ and $0.66$: 0 enhanced regrowth (high), 1 enhanced regrowth (low), 2 unburned, 3 low severity, 4 moderate-low severity, 5 moderate-high severity, 6 high severity; $-1$ marks NaN.

## 5. Hydrological terrain analysis

`unbihexium.terrain.hydrology` analyses surface flow on a DEM whose first row is the northern edge; NaN cells are nodata.

| Function | Output | Method and source |
| --- | --- | --- |
| `fill_depressions(dem, epsilon=0.0)` | DEM without pits | priority-flood filling, optionally with a small gradient `epsilon` across flats (Barnes et al. [31]) |
| `flow_direction_d8(dem, resolution)` | uint8 D8 codes | steepest descent to one of eight neighbours, drops to diagonal neighbours divided by the diagonal distance (O'Callaghan and Mark [32]); codes 1 east, 2 south-east, 4 south, 8 south-west, 16 west, 32 north-west, 64 north, 128 north-east, 0 for cells without a lower neighbour inside the grid |
| `flow_accumulation(flow_dir, weights=None)` | number (or weight) of upstream cells, excluding the cell itself | accumulation along the D8 receivers [33] |
| `watershed(flow_dir, outlet)` | boolean mask | cells that drain to the outlet `(row, column)` |
| `extract_streams(accumulation, threshold)` | boolean mask | cells whose accumulation is at least `threshold` |
| `twi(dem, resolution, fill=True, min_slope=0.1)` | topographic wetness index | $\ln(a / \tan\beta)$ (Beven and Kirkby [34]) |

For the wetness index, the specific catchment area is $a = (A + 1)\,\Delta x\,\Delta y / \sqrt{\Delta x\,\Delta y}$ with the flow accumulation $A$ of the filled DEM, and $\beta$ is the Horn slope of the original DEM, floored at `min_slope` degrees so that flat cells stay finite.

## 6. Examples

The examples were executed on 24 September 2026 against the main branch with CPython 3.13 on a CPU, with `UNBIHEXIUM_CACHE` set to a temporary directory. They share one Python session and one working directory.

### 6.1 Index functions

```python
import numpy as np
from unbihexium.indices import BURN_SEVERITY_CLASSES, burn_severity, compute_index, dnbr, evi, nbr, ndvi

red = np.array([0.05, 0.10, 0.20, 0.0])
nir = np.array([0.40, 0.30, 0.25, 0.0])
blue = np.array([0.03, 0.05, 0.10, 0.0])
print(ndvi(nir, red).round(4))
print(evi(nir, red, blue).round(4))
print(compute_index("savi", nir=nir, red=red).round(4))

pre = nbr(nir=np.array([0.40, 0.35, 0.30]), swir=np.array([0.10, 0.12, 0.15]))
post = nbr(nir=np.array([0.38, 0.20, 0.10]), swir=np.array([0.11, 0.20, 0.30]))
d = dnbr(pre, post)
print(d.round(3), [BURN_SEVERITY_CLASSES[c] for c in burn_severity(d)])
```

```text
[0.7778 0.5    0.1111    nan]
[0.5932 0.3279 0.0735 0.    ]
[0.5526 0.3333 0.0789 0.    ]
[0.049 0.489 0.833] ['unburned', 'moderate-high severity', 'high severity']
```

The last pixel has zero reflectance in every band: NDVI is undefined (NaN), whereas the denominators of EVI and SAVI contain the constants 1 and 0.5 and give 0.

### 6.2 Index registry with sensor band names

```python
import numpy as np
from unbihexium.core.index import compute_index as registry_index

bands = {"B04": np.array([0.05, 0.10]), "B08": np.array([0.40, 0.30])}
print(registry_index("NDVI", bands, sensor="sentinel2").round(4))
print(registry_index("WDRVI", {"NIR": np.array([0.4]), "RED": np.array([0.05])}, alpha=0.2).round(4))
```

```text
[0.7778 0.5   ]
[0.2308]
```

### 6.3 The NDVI model equals the formula

The index families are exact. The `tiny` NDVI model gives the value of the formula up to float32 rounding:

```python
import numpy as np
from unbihexium.ai import predict
from unbihexium.indices import ndvi

stack = np.stack([np.full((8, 8), 0.05), np.full((8, 8), 0.40)]).astype("float32")  # red, nir
result = predict("ndvi_calculator_tiny", stack)
print(result.names, result.values.shape, float(result.values[0, 0, 0]), float(ndvi(0.40, 0.05)))
```

```text
['ndvi'] (1, 8, 8) 0.7777777314186096 0.7777777777777778
```

### 6.4 A water starter model

For contrast, the untrained water detector labels three quarters of a perfectly uniform image as water. It must be trained before use.

```python
import numpy as np
from unbihexium.ai import WaterDetector

uniform = np.full((4, 32, 32), 0.1, dtype="float32")  # blue, green, red, nir
result = WaterDetector("water_surface_detector_tiny").predict(uniform)
print({k: round(v, 3) for k, v in result.class_fractions().items()})
```

```text
{'background': 0.247, 'water': 0.753}
```

### 6.5 Hydrology

A valley that drains south with a pit in the middle:

```python
import numpy as np
from unbihexium.terrain import (
    extract_streams, fill_depressions, flow_accumulation, flow_direction_d8, twi, watershed,
)

y, x = np.mgrid[0:6, 0:5]
dem = 10.0 - y + 0.5 * np.abs(x - 2)
dem[2, 2] = 5.0  # pit
filled = fill_depressions(dem, epsilon=1e-3)
print(round(float(filled[2, 2]), 3))
flow = flow_direction_d8(filled)
acc = flow_accumulation(flow)
print(acc.astype(int))
print(extract_streams(acc, 5).astype(int)[:, 2], int(watershed(flow, (5, 2)).sum()))
print(twi(dem)[:, 2].round(2))
```

```text
7.001
[[ 0  0  0  0  0]
 [ 0  1  3  1  0]
 [ 0  1 12  1  0]
 [ 0  1 13  1  0]
 [ 0  1 18  1  0]
 [ 0  2 29  2  0]]
[0 0 1 1 1 1] 30
[0.   0.83 2.56 4.03 2.94 3.4 ]
```

The pit is raised to its spill elevation (7 m) plus the flat gradient, all 29 upstream cells drain through the outlet at the southern edge, and the wetness index is highest along the valley floor.

### 6.6 Command line

```python
import numpy as np
from unbihexium.core.raster import Raster

bands = np.full((13, 32, 32), 0.1, dtype="float32")  # Sentinel-2 order B01 to B12
bands[3] = 0.05  # B04, red
bands[7] = 0.40  # B08, near infrared
Raster.from_array(
    bands, crs="EPSG:32633", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6700000.0)
).to_file("s2.tif")
```

```bash
unbihexium index NDVI -i s2.tif -o ndvi.tif
unbihexium index ndwi -i s2.tif -o ndwi.tif
```

```text
Wrote: ndvi.tif (NDVI)
Wrote: ndwi.tif (NDWI)
```

The registered water pipeline runs `water_surface_detector` on a four-band raster (blue, green, red, near infrared) and writes its class map as a GeoTIFF. It prints a line of the form `Completed: <run id> -> water.tif` with a random run id.

```python
Raster.from_array(
    bands[[1, 2, 3, 7]], crs="EPSG:32633", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6700000.0)
).to_file("rgbn.tif")
```

```bash
unbihexium pipeline run water_detection -i rgbn.tif -o water.tif -p variant=tiny
```

## 7. Limitations and responsible use

### 7.1 Conventions

The key words MUST, SHOULD and MAY in this section are to be interpreted as described in RFC 2119 and RFC 8174 [35], [36] when, and only when, they appear in capitals.

### 7.2 Limitations

- Index formulas assume surface reflectance in $[0, 1]$. Digital numbers MUST be converted first (for example with `unbihexium.preprocessing.sentinel2_reflectance`, see [domain 04](04_environment_forestry_image_processing.md)); indices of top-of-atmosphere reflectance differ from those of surface reflectance.
- Thresholds on indices (water where NDWI > 0, the burn severity limits) are published starting points that SHOULD be calibrated locally.
- The eight water families are untrained. Their outputs MUST NOT be used for flood warnings, insurance, planning or any other decision before the model has been trained and validated on independent reference data. Read [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md).
- D8 routing sends all flow to one neighbour and cannot represent divergent flow; the wetness index depends on the DEM resolution and SHOULD be compared only between DEMs of the same resolution.

## 8. Related documents

- [Capability index](index.md), [domain 02](02_tourism_data_processing.md) (terrain derivatives), [domain 04](04_environment_forestry_image_processing.md) (radiometric preprocessing) and [domain 12](12_radar_sar.md) (SAR flood mapping).
- [Model catalogue](../model_zoo/model_catalog.md), [training](../model_zoo/training.md) and [inference](../model_zoo/inference.md).
- [Command line reference](../reference/cli.md) and [API reference](../reference/api.md).

## References

[1] Rouse, J. W., Haas, R. H., Schell, J. A., Deering, D. W. Monitoring vegetation systems in the Great Plains with ERTS. Third ERTS Symposium, NASA SP-351, 309-317. 1974. <https://ntrs.nasa.gov/citations/19740022614>

[2] McFeeters, S. K. The use of the normalized difference water index (NDWI) in the delineation of open water features. International Journal of Remote Sensing 17(7), 1425-1432. 1996. <https://doi.org/10.1080/01431169608948714>

[3] Huete, A., Didan, K., Miura, T., Rodriguez, E. P., Gao, X., Ferreira, L. G. Overview of the radiometric and biophysical performance of the MODIS vegetation indices. Remote Sensing of Environment 83(1-2), 195-213. 2002. <https://doi.org/10.1016/S0034-4257(02)00096-2>

[4] Huete, A. R. A soil-adjusted vegetation index (SAVI). Remote Sensing of Environment 25(3), 295-309. 1988. <https://doi.org/10.1016/0034-4257(88)90106-X>

[5] Rock, B. N., Vogelmann, J. E., Williams, D. L., Vogelmann, A. F., Hoshizaki, T. Remote detection of forest damage. BioScience 36(7), 439-445. 1986. <https://doi.org/10.2307/1310339>

[6] Key, C. H., Benson, N. C. Landscape assessment: ground measure of severity, the Composite Burn Index; and remote sensing of severity, the Normalized Burn Ratio. In FIREMON: Fire Effects Monitoring and Inventory System, USDA Forest Service General Technical Report RMRS-GTR-164-CD. 2006. <https://research.fs.usda.gov/treesearch/24066>

[7] Kogan, F. N. Application of vegetation index and brightness temperature for drought detection. Advances in Space Research 15(11), 91-100. 1995. <https://doi.org/10.1016/0273-1177(95)00079-T>

[8] Gitelson, A. A., Kaufman, Y. J., Merzlyak, M. N. Use of a green channel in remote sensing of global vegetation from EOS-MODIS. Remote Sensing of Environment 58(3), 289-298. 1996. <https://doi.org/10.1016/S0034-4257(96)00072-7>

[9] Gitelson, A. A., Merzlyak, M. N. Spectral reflectance changes associated with autumn senescence of Aesculus hippocastanum L. and Acer platanoides L. leaves. Journal of Plant Physiology 143(3), 286-292. 1994. <https://doi.org/10.1016/S0176-1617(11)81633-0>

[10] Barnes, E. M., et al. Coincident detection of crop water stress, nitrogen status and canopy density using ground-based multispectral data. Proceedings of the 5th International Conference on Precision Agriculture. 2000.

[11] Jiang, Z., Huete, A. R., Didan, K., Miura, T. Development of a two-band enhanced vegetation index without a blue band. Remote Sensing of Environment 112(10), 3833-3845. 2008. <https://doi.org/10.1016/j.rse.2008.06.006>

[12] Rondeaux, G., Steven, M., Baret, F. Optimization of soil-adjusted vegetation indices. Remote Sensing of Environment 55(2), 95-107. 1996. <https://doi.org/10.1016/0034-4257(95)00186-7>

[13] Qi, J., Chehbouni, A., Huete, A. R., Kerr, Y. H., Sorooshian, S. A modified soil adjusted vegetation index. Remote Sensing of Environment 48(2), 119-126. 1994. <https://doi.org/10.1016/0034-4257(94)90134-1>

[14] Kaufman, Y. J., Tanre, D. Atmospherically resistant vegetation index (ARVI) for EOS-MODIS. IEEE Transactions on Geoscience and Remote Sensing 30(2), 261-270. 1992. <https://doi.org/10.1109/36.134076>

[15] Gitelson, A. A., Kaufman, Y. J., Stark, R., Rundquist, D. Novel algorithms for remote estimation of vegetation fraction. Remote Sensing of Environment 80(1), 76-87. 2002. <https://doi.org/10.1016/S0034-4257(01)00289-9>

[16] Camps-Valls, G., et al. A unified vegetation index for quantifying the terrestrial biosphere. Science Advances 7(9), eabc7447. 2021. <https://doi.org/10.1126/sciadv.abc7447>

[17] Jordan, C. F. Derivation of leaf-area index from quality of light on the forest floor. Ecology 50(4), 663-666. 1969. <https://doi.org/10.2307/1936256>

[18] Gitelson, A. A. Wide dynamic range vegetation index for remote quantification of biophysical characteristics of vegetation. Journal of Plant Physiology 161(2), 165-173. 2004. <https://doi.org/10.1078/0176-1617-01176>

[19] Gitelson, A. A., Gritz, Y., Merzlyak, M. N. Relationships between leaf chlorophyll content and spectral reflectance and algorithms for non-destructive chlorophyll assessment in higher plant leaves. Journal of Plant Physiology 160(3), 271-282. 2003. <https://doi.org/10.1078/0176-1617-00887>

[20] Xu, H. Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. International Journal of Remote Sensing 27(14), 3025-3033. 2006. <https://doi.org/10.1080/01431160600589179>

[21] Gao, B.-C. NDWI: a normalized difference water index for remote sensing of vegetation liquid water from space. Remote Sensing of Environment 58(3), 257-266. 1996. <https://doi.org/10.1016/S0034-4257(96)00067-3>

[22] Feyisa, G. L., Meilby, H., Fensholt, R., Proud, S. R. Automated water extraction index: a new technique for surface water mapping using Landsat imagery. Remote Sensing of Environment 140, 23-35. 2014. <https://doi.org/10.1016/j.rse.2013.08.029>

[23] Lacaux, J. P., Tourre, Y. M., Vignolles, C., Ndione, J. A., Lafaye, M. Classification of ponds from high-spatial resolution remote sensing: application to Rift Valley Fever epidemics in Senegal. Remote Sensing of Environment 106(1), 66-74. 2007. <https://doi.org/10.1016/j.rse.2006.07.012>

[24] Mishra, S., Mishra, D. R. Normalized difference chlorophyll index: a novel model for remote estimation of chlorophyll-a concentration in turbid productive waters. Remote Sensing of Environment 117, 394-406. 2012. <https://doi.org/10.1016/j.rse.2011.10.016>

[25] Zha, Y., Gao, J., Ni, S. Use of normalized difference built-up index in automatically mapping urban areas from TM imagery. International Journal of Remote Sensing 24(3), 583-594. 2003. <https://doi.org/10.1080/01431160304987>

[26] Rikimaru, A., Roy, P. S., Miyatake, S. Tropical forest cover density mapping. Tropical Ecology 43(1), 39-47. 2002.

[27] Hall, D. K., Riggs, G. A., Salomonson, V. V. Development of methods for mapping global snow cover using moderate resolution imaging spectroradiometer data. Remote Sensing of Environment 54(2), 127-140. 1995. <https://doi.org/10.1016/0034-4257(95)00137-P>

[28] Hunt, E. R., Rock, B. N. Detection of changes in leaf water content using near- and middle-infrared reflectances. Remote Sensing of Environment 30(1), 43-54. 1989. <https://doi.org/10.1016/0034-4257(89)90046-1>

[29] Kim, Y., van Zyl, J. J. A time-series approach to estimate soil moisture using polarimetric radar data. IEEE Transactions on Geoscience and Remote Sensing 47(8), 2519-2527. 2009. <https://doi.org/10.1109/TGRS.2009.2014944>

[30] Miller, J. D., Thode, A. E. Quantifying burn severity in a heterogeneous landscape with a relative version of the delta Normalized Burn Ratio (dNBR). Remote Sensing of Environment 109(1), 66-80. 2007. <https://doi.org/10.1016/j.rse.2006.12.006>

[31] Barnes, R., Lehman, C., Mulla, D. Priority-flood: an optimal depression-filling and watershed-labeling algorithm for digital elevation models. Computers and Geosciences 62, 117-127. 2014. <https://doi.org/10.1016/j.cageo.2013.04.024>

[32] O'Callaghan, J. F., Mark, D. M. The extraction of drainage networks from digital elevation data. Computer Vision, Graphics, and Image Processing 28(3), 323-344. 1984. <https://doi.org/10.1016/S0734-189X(84)80011-0>

[33] Jenson, S. K., Domingue, J. O. Extracting topographic structure from digital elevation data for geographic information system analysis. Photogrammetric Engineering and Remote Sensing 54(11), 1593-1600. 1988.

[34] Beven, K. J., Kirkby, M. J. A physically based, variable contributing area model of basin hydrology. Hydrological Sciences Bulletin 24(1), 43-69. 1979. <https://doi.org/10.1080/02626667909491834>

[35] Bradner, S. Key words for use in RFCs to indicate requirement levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[36] Leiba, B. Ambiguity of uppercase vs lowercase in RFC 2119 key words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[37] U.S. Geological Survey. Landsat Collection 2 surface reflectance-derived spectral indices product guide (definition of NBR2). <https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices>

<!--
=============================================================================
End of file docs/capabilities/03_indices_flood_water.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
