<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/capabilities/10_satellite_imagery_features.md
Title       : Capability Domain 10: Satellite Imagery Features
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Capability Domain 10: Satellite Imagery Features

| Field | Value |
| --- | --- |
| Document | UBX-DOC-610 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch |

## Abstract

This document describes how Unbihexium handles the properties of optical satellite imagery that every analysis depends on: the band tables of the supported sensors, the conversion of stored digital numbers to reflectance, radiance and brightness temperature, the quality layers that mark clouds and invalid pixels, the fusion of panchromatic and multispectral bands (pansharpening), radiometric enhancement for display and multi-date normalisation, and colour composites. It also lists the six model families intended for panchromatic, multispectral, stereo and synthetic imagery. It is written for users who prepare Sentinel-2 or Landsat data for analysis, for contributors, and for reviewers who need to know which formulas are implemented and where they come from. All examples were executed against the current code. The model families are untrained starter models and are described as intended applications, not as validated products; classical stereo photogrammetry, orthorectification and physically based atmospheric correction are not implemented.

## Contents

1. [Scope and status](#1-scope-and-status)
2. [Components of the capability](#2-components-of-the-capability)
3. [Sensor band tables](#3-sensor-band-tables)
4. [Radiometric conversion and quality masks](#4-radiometric-conversion-and-quality-masks)
5. [Pansharpening](#5-pansharpening)
6. [Enhancement and colour composites](#6-enhancement-and-colour-composites)
7. [Model families](#7-model-families)
8. [Command line](#8-command-line)
9. [Limitations](#9-limitations)
10. [Related documents](#10-related-documents)
11. [References](#references)

## 1. Scope and status

### 1.1 What the capability covers

The capability registry (`unbihexium.registry.CapabilityRegistry`) has no domain named "satellite imagery features"; this document collects components from several registered capabilities:

- the library capability `image_preprocessing` (domain `imaging`, maturity `stable`), implemented by `unbihexium.preprocessing`: radiometry, masks, pansharpening, enhancement, resampling and transforms (resampling is described in [11_resolution_metadata_qa.md](11_resolution_metadata_qa.md));
- the library capability `visualization` (domain `imaging`, maturity `stable`), implemented by `unbihexium.visualization`, of which the colour composites are described here;
- the sensor tables of `unbihexium.core.sensor` and the `Scene` record of `unbihexium.core.scene`;
- six model families: `pansharpening`, `multispectral_processor`, `panchromatic_processor`, `stereo_processor` and `tri_stereo_processor` (catalogue domain `imaging`) and `synthetic_imagery` (catalogue domain `ai`).

The other families of the `imaging` domain are documented in [01_ai_products.md](01_ai_products.md) (for example `cloud_mask`, `super_resolution`, `coregistration`, `orthorectification`), [08_value_added_imagery.md](08_value_added_imagery.md) (elevation models) and [11_resolution_metadata_qa.md](11_resolution_metadata_qa.md) (tiling, mosaics, relief displacement and digitisation).

### 1.2 Status

The preprocessing and visualisation functions are deterministic implementations of published methods and are covered by the unit tests in `tests/`. The 24 models of the six families (four size variants each) are untrained starter models: each has a complete, trainable network with deterministically initialised weights whose SHA-256 digest is published in `src/unbihexium/zoo/digests.json`, but none has been trained on satellite imagery, so their outputs carry no information until the model is trained on data for the user's sensor and area (see [docs/model_zoo/training.md](../model_zoo/training.md)). Of the 520 models of the zoo, only the 28 models of the 7 spectral index families compute exact formulas without training, and none of them is listed here. No accuracy figures are published for any model; see section 2 of [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md).

### 1.3 Conventions

Multi-band arrays are band-first, `(C, H, W)`; single bands are `(H, W)`. Missing values are NaN or an explicit `nodata` value, which the functions convert to NaN. Reflectance is dimensionless (0 to 1), radiance is in W m-2 sr-1 um-1 and temperature in kelvin.

## 2. Components of the capability

| Component | Public names | Purpose |
| --- | --- | --- |
| Sensor tables (`unbihexium.core.sensor`) | `get_sensor`, `list_sensors`, `SensorModel`, `SpectralBand`, `SARMode` | Band names, common names, centre wavelengths, bandwidths and pixel sizes of Sentinel-2 MSI, Landsat 8 OLI/TIRS, Landsat 9 OLI-2/TIRS-2 and Sentinel-1 C-SAR |
| Radiometry (`unbihexium.preprocessing.radiometry`) | `sentinel2_reflectance`, `parse_landsat_mtl`, `landsat_radiance`, `landsat_toa_reflectance`, `landsat_toa_reflectance_from_mtl`, `landsat_brightness_temperature`, `landsat_c2l2_reflectance`, `landsat_c2l2_temperature`, `earth_sun_distance`, `radiance_to_reflectance`, `dark_object_subtraction` | Digital numbers to physical quantities and image-based haze removal |
| Masks (`unbihexium.preprocessing.masks`) | `SCL_CLASSES`, `scl_valid_mask`, `qa_bits`, `landsat_qa_mask`, `landsat_cloud_confidence`, `buffer_mask`, `apply_mask` | Validity masks from the Sentinel-2 scene classification and the Landsat QA_PIXEL band |
| Pansharpening (`unbihexium.preprocessing.pansharpen`) | `upsample_to_pan`, `brovey`, `ihs`, `gram_schmidt` | Component substitution fusion of panchromatic and multispectral bands |
| Enhancement (`unbihexium.preprocessing.enhancement`) | `linear_stretch`, `percentile_stretch`, `gamma_correction`, `histogram_equalize`, `histogram_match` | Radiometric enhancement and relative normalisation |
| Composites (`unbihexium.visualization`) | `rgb_composite`, `SENTINEL2_COMPOSITES`, `LANDSAT89_COMPOSITES`, `save_png`, `quicklook` | Named band combinations rendered to 8-bit images |
| Scenes (`unbihexium.core.scene`) | `Scene`, `SceneMetadata` | One acquisition as named band rasters, with alignment of bands of different pixel sizes and spectral indices by product band names |

## 3. Sensor band tables

`get_sensor(name)` accepts the keys `sentinel2_msi`, `landsat8_oli`, `landsat9_oli2` and `sentinel1_sar` and the aliases `sentinel2`, `s2`, `landsat8`, `l8`, `landsat9`, `l9`, `sentinel1` and `s1`. A `SensorModel` resolves product band names (for example `B04`, `B4`, or the Collection 2 Level-2 names `SR_B4` and `ST_B10`) to common names (`RED`), which lets `unbihexium.core.index.compute_index` and `Scene.compute_index` accept product band names. The band limits of Sentinel-2 are taken from the ESA product specification [1] and those of Landsat 8 and 9 from the USGS data format documentation [2].

```python
from unbihexium.core.sensor import get_sensor

s2 = get_sensor("sentinel2")
for resolution in (10.0, 20.0, 60.0):
    print(int(resolution), "m:", s2.bands_at_resolution(resolution))
red_edge = s2.band("B05")
print(red_edge.common_name, red_edge.center_nm, red_edge.lower_nm, red_edge.upper_nm)
l9 = get_sensor("landsat9")
print(l9.band("SR_B4").common_name, l9.band("B8").resolution_m, l9.band("B10").resolution_m)
```

Output:

```text
10 m: ['B02', 'B03', 'B04', 'B08']
20 m: ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
60 m: ['B01', 'B09', 'B10']
REDEDGE1 704.1 696.6 711.6
RED 15.0 100.0
```

The Landsat thermal bands are listed with the 100 m sampling of the TIRS instrument; the delivered products are resampled to 30 m.

## 4. Radiometric conversion and quality masks

### 4.1 Sentinel-2

`sentinel2_reflectance(dn, offset=None, quantification=10000, processing_baseline=None, nodata=0)` computes

$$
\rho = \frac{DN + o}{Q}
$$

with the quantification value $Q = 10000$ and the offset $o = -1000$ (`RADIO_ADD_OFFSET` for Level-1C, `BOA_ADD_OFFSET` for Level-2A) for products of processing baseline 04.00 or later, introduced on 25 January 2022, and $o = 0$ for older products [1]. When neither `offset` nor `processing_baseline` is given, the current convention ($o = -1000$) is assumed. DN 0 marks no data and becomes NaN.

### 4.2 Landsat 8 and 9

For Collection 2 Level-1 products, with the rescaling factors of the MTL metadata file [2]:

$$
L_\lambda = M_L Q_{cal} + A_L, \qquad \rho_\lambda = \frac{M_\rho Q_{cal} + A_\rho}{\sin\theta_{SE}}, \qquad T = \frac{K_2}{\ln(K_1 / L_\lambda + 1)}
$$

(`landsat_radiance`, `landsat_toa_reflectance`, `landsat_brightness_temperature`), where $\theta_{SE}$ is the sun elevation. `parse_landsat_mtl` reads an MTL text file into a flat dictionary and `landsat_toa_reflectance_from_mtl` looks up the factors of one band. Collection 2 Level-2 products use fixed factors: surface reflectance $2.75 \times 10^{-5} DN - 0.2$ and surface temperature $0.00341802\,DN + 149.0$ K (`landsat_c2l2_reflectance`, `landsat_c2l2_temperature`) [3].

### 4.3 Radiance to reflectance and haze removal

`radiance_to_reflectance(radiance, esun, sun_zenith, day_of_year)` computes the top-of-atmosphere reflectance [4]

$$
\rho = \frac{\pi L_\lambda d^2}{E_{SUN,\lambda} \cos\theta_s}
$$

with the Earth-Sun distance $d$ in astronomical units from `earth_sun_distance`, which evaluates the Fourier series of Spencer [5] for the eccentricity correction $E_0 = (r_0/r)^2$ and returns $d = 1/\sqrt{E_0}$. `dark_object_subtraction(reflectance, dark_percentile=0.01, dark_reflectance=0.01)` implements the DOS1 method of Chavez [6]: the darkest pixels of each band (the given percentile) are assumed to have a surface reflectance of 1 %, so the path reflectance $\rho_{dark} - 0.01$ is subtracted from every pixel; it returns the corrected bands and the subtracted values.

### 4.4 Quality masks

`scl_valid_mask(scl, invalid=(0, 1, 3, 8, 9, 10))` returns True for usable pixels of the Sentinel-2 Level-2A scene classification layer; the default excludes no data, saturated or defective pixels, cloud shadows, medium and high probability cloud and thin cirrus [7]. `landsat_qa_mask(qa, ...)` returns True for flagged (unusable) pixels of the Landsat Collection 2 QA_PIXEL band, with the bits 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow and 7 water selectable by keyword [3]; `landsat_cloud_confidence` extracts the two-bit confidence fields (0 none, 1 low, 2 medium, 3 high). `buffer_mask` grows a mask by a number of pixels and `apply_mask` sets masked pixels to NaN.

### 4.5 Example

```python
import numpy as np

from unbihexium.preprocessing import (
    dark_object_subtraction,
    earth_sun_distance,
    landsat_cloud_confidence,
    landsat_qa_mask,
    landsat_toa_reflectance_from_mtl,
    parse_landsat_mtl,
    scl_valid_mask,
    sentinel2_reflectance,
)

# Sentinel-2 L2A digital numbers of processing baseline 05.10 (offset -1000).
dn = np.array([[0, 1000, 1500], [2200, 3500, 11000]], dtype=np.uint16)
print(sentinel2_reflectance(dn, processing_baseline="05.10"))

# Scene classification layer: 4 vegetation, 5 not vegetated, 8 and 9 cloud, 3 shadow, 0 no data.
scl = np.array([[4, 8, 3], [5, 9, 0]])
print(scl_valid_mask(scl))

# Landsat 8/9 Level-1 TOA reflectance from an excerpt of an MTL file.
mtl = parse_landsat_mtl("""
GROUP = LEVEL1_RADIOMETRIC_RESCALING
    REFLECTANCE_MULT_BAND_4 = 2.0000E-05
    REFLECTANCE_ADD_BAND_4 = -0.100000
END_GROUP = LEVEL1_RADIOMETRIC_RESCALING
GROUP = IMAGE_ATTRIBUTES
    SUN_ELEVATION = 45.0
END_GROUP = IMAGE_ATTRIBUTES
""")
q = np.array([[0, 8000, 12000]], dtype=np.uint16)
print(np.round(landsat_toa_reflectance_from_mtl(q, mtl, band=4), 4))

# QA_PIXEL values: clear land, high-confidence cloud, cloud shadow, fill.
qa = np.array([[21824, 22280, 23888, 1]], dtype=np.uint16)
print(landsat_qa_mask(qa), landsat_cloud_confidence(qa))

# Earth-Sun distance near perihelion and aphelion, and DOS1 haze removal.
print(round(earth_sun_distance(3), 4), round(earth_sun_distance(185), 4))
hazy = np.random.default_rng(0).uniform(0.06, 0.4, size=(2, 50, 50))
corrected, path = dark_object_subtraction(hazy)
print(np.round(path, 3), round(float(np.nanmin(corrected)), 3))
```

Output:

```text
[[ nan 0.   0.05]
 [0.12 0.25 1.  ]]
[[ True False False]
 [ True False False]]
[[   nan 0.0849 0.198 ]]
[[False  True  True  True]] [[1 3 1 0]]
0.9829 1.0171
[0.05 0.05] 0.01
```

## 5. Pansharpening

### 5.1 Method

The three fusion functions are component substitution methods [8]. The multispectral bands $MS_k$ must first be resampled to the panchromatic grid, for example with `upsample_to_pan(ms, shape, order=3)` (spline interpolation with pixel-area alignment). An intensity $I = \sum_k w_k MS_k + b$ is computed and the spatial detail of the panchromatic band $P$ is injected into every band:

$$
F_k = MS_k + g_k (P' - I)
$$

where $P'$ is the panchromatic band matched to the mean and standard deviation of $I$ (disabled with `match=False` in `brovey` and `ihs`). The functions differ in the weights and gains:

| Function | Intensity | Injection gain $g_k$ | Source |
| --- | --- | --- | --- |
| `brovey` | weighted mean of the bands, $b = 0$ (equal weights unless `weights` is given) | $MS_k / I$, that is $F_k = MS_k P' / I$ | Gillespie, Kahle and Walker [9] |
| `ihs` | weighted mean of the bands, $b = 0$ | 1 (fast generalised IHS) | Tu et al. [10] |
| `gram_schmidt` | weights $w_k$ and offset $b$ from a least-squares regression of $P$ on the bands (or fixed `weights`) | $\mathrm{cov}(MS_k, I)/\mathrm{var}(I)$ | Laben and Brower [11], Aiazzi, Baronti and Selva [12] |

The Gram-Schmidt function is the adaptive variant (GSA) of Aiazzi et al., which reproduces Gram-Schmidt spectral sharpening with an intensity estimated from the data. Pixels with NaN in any input are NaN in the output.

### 5.2 Example

The example follows the reduced-resolution protocol of Wald [13]: a reference image at the pan resolution is degraded by a factor of 4, fused again, and compared with the reference with the spectral angle mapper (SAM), ERGAS and the Q index of `unbihexium.metrics` (formulas in [11_resolution_metadata_qa.md](11_resolution_metadata_qa.md)).

```python
import numpy as np
from scipy import ndimage

from unbihexium.metrics import ergas, q_index, sam
from unbihexium.preprocessing import aggregate, brovey, gram_schmidt, ihs, upsample_to_pan

# Reference 4-band image (blue, green, red, nir) at the pan resolution, 128 x 128.
rng = np.random.default_rng(5)
texture = ndimage.gaussian_filter(rng.normal(size=(128, 128)), 2.0)
field = ndimage.gaussian_filter(rng.normal(size=(128, 128)), 12.0)
reference = np.stack([
    0.05 + 0.02 * texture + 0.10 * field,
    0.08 + 0.03 * texture + 0.12 * field,
    0.07 + 0.04 * texture + 0.15 * field,
    0.30 + 0.10 * texture - 0.20 * field,
]).clip(0.001, 1.0)
pan = reference[:3].mean(axis=0)  # A pan band covering blue to red.

# Multispectral bands observed at 4 times coarser pixels, then brought back to the pan grid.
ms_low = np.stack([aggregate(band, 4, method="mean") for band in reference])
ms_up = upsample_to_pan(ms_low, pan.shape, order=3)

print(f"{'method':>12} {'SAM deg':>8} {'ERGAS':>7} {'Q':>6}")
for name, fused in (("upsampled", ms_up), ("brovey", brovey(pan, ms_up)),
                    ("ihs", ihs(pan, ms_up)), ("gram_schmidt", gram_schmidt(pan, ms_up))):
    print(f"{name:>12} {sam(reference, fused):8.3f} {ergas(reference, fused, ratio=4):7.3f} {q_index(reference, fused):6.3f}")
```

Output:

```text
      method  SAM deg   ERGAS      Q
   upsampled    0.104   0.483  0.867
      brovey    0.104   0.647  0.930
         ihs    0.526   1.021  0.885
gram_schmidt    0.124   0.237  0.979
```

The values describe one synthetic image and show how the functions are combined; they do not rank the methods on real data, for which see the comparison of Vivone et al. [8].

## 6. Enhancement and colour composites

The enhancement functions work band by band and ignore NaN [14]:

- `linear_stretch(image, low, high)` maps $[low, high]$ linearly to $[0, 1]$ (clipped by default); `percentile_stretch(image, low=2, high=98)` uses band percentiles as limits;
- `gamma_correction(image, gamma)` computes $x^{1/\gamma}$ for values in $[0, 1]$;
- `histogram_equalize(image)` maps values through the empirical cumulative distribution function;
- `histogram_match(source, reference)` maps every source value to the reference quantile at the same empirical probability, $F_{ref}^{-1}(F_{src}(x))$, with linear interpolation between quantiles. It serves as a relative radiometric normalisation of multi-date images.

`rgb_composite(stack, bands, band_names=None, stretch="percentile", low=2, high=98, gamma=1.0)` renders three bands to an 8-bit image. `bands` can be indices, band names or the name of a composite from `SENTINEL2_COMPOSITES` (`true_color`, `color_infrared`, `swir`, `agriculture`, `geology`, `bathymetric`) or `LANDSAT89_COMPOSITES` (the same names except `bathymetric`). `save_png(path, image, transform=None)` writes a PNG and, with an affine transform, a world file.

```python
import numpy as np

from unbihexium.preprocessing import histogram_match, percentile_stretch
from unbihexium.visualization import rgb_composite, save_png

rng = np.random.default_rng(8)
names = ["B02", "B03", "B04", "B08", "B11", "B12"]
stack = rng.gamma(4.0, 0.03, size=(6, 64, 64))  # Reflectance-like values.

# Relative radiometric normalisation: a second date with haze, lower gain and noise, matched to the first.
second = 0.8 * stack + 0.04 + rng.normal(0, 0.005, size=stack.shape)
matched = histogram_match(second, stack)
print(np.round([np.abs(second - stack).mean(), np.abs(matched - stack).mean()], 4))

# 2 to 98 % stretch to [0, 1], and an 8-bit false-colour composite by composite name.
stretched = percentile_stretch(stack, low=2, high=98)
print(round(float(np.nanmin(stretched)), 3), round(float(np.nanmax(stretched)), 3))
cir = rgb_composite(stack, bands="color_infrared", band_names=names)
print(cir.shape, cir.dtype)
print(save_png("cir.png", cir, transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6700000.0)))
```

Output:

```text
[0.0181 0.005 ]
0.0 1.0
(64, 64, 3) uint8
cir.png
```

The world file `cir.pgw` is written next to the PNG.

## 7. Model families

### 7.1 Families

The tables below were generated from the model catalogue with the script of section 7.2. The networks are those of `unbihexium.ai.models.networks`: a U-Net with a residual encoder produces every output of these families. Parameter counts are those of the built models.

| Family | Domain | Task | Network | Input bands | Outputs | Parameters (tiny / base / large / mega) |
| --- | --- | --- | --- | --- | --- | --- |
| `pansharpening` | imaging | enhancement | U-Net | ms_blue, ms_green, ms_red, ms_nir, pan | blue, green, red, nir | 733,268 / 7,059,108 / 22,060,084 / 60,451,908 |
| `multispectral_processor` | imaging | enhancement | U-Net | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | 734,090 / 7,060,746 / 22,062,538 / 60,455,178 |
| `panchromatic_processor` | imaging | enhancement | U-Net | pan | pan | 732,641 / 7,057,857 / 22,058,209 / 60,449,409 |
| `stereo_processor` | imaging | dense_regression | U-Net | pan_t1, pan_t2 | disparity (px) | 732,785 / 7,058,145 / 22,058,641 / 60,449,985 |
| `tri_stereo_processor` | imaging | dense_regression | U-Net | pan_forward, pan_nadir, pan_backward | surface_elevation (m) | 732,929 / 7,058,433 / 22,059,073 / 60,450,561 |
| `synthetic_imagery` | ai | enhancement | U-Net | VV, VH | red, green, blue | 732,819 / 7,058,211 / 22,058,739 / 60,450,115 |

| Family | Intended application (once trained) | Reference data needed for training | Suitable input data |
| --- | --- | --- | --- |
| `pansharpening` | Fuses upsampled multispectral bands with the panchromatic band into a sharpened multispectral image. | Sharpened references, for example by Wald's protocol. | Multispectral and panchromatic bands of the same sensor |
| `multispectral_processor` | Reduces noise and striping in multispectral imagery. | Clean reference images, for example temporal composites. | Sentinel-2 L2A |
| `panchromatic_processor` | Reduces noise in panchromatic imagery. | Clean panchromatic references. | Panchromatic satellite imagery |
| `stereo_processor` | Estimates disparity between the two images of an epipolar stereo pair. | Reference disparity maps. | Epipolar-rectified stereo pairs |
| `tri_stereo_processor` | Estimates surface elevation from a tri-stereo acquisition. | Reference DSMs. | Epipolar-rectified tri-stereo panchromatic imagery |
| `synthetic_imagery` | Synthesises an optical RGB image from SAR backscatter, for example to fill cloud gaps. | Co-located cloud-free optical images. | Sentinel-1 GRD with co-located Sentinel-2 L2A |

The recommended tile size is 256 pixels for the `tiny` and `base` variants and 512 pixels for `large` and `mega`. The stereo families expect epipolar-rectified images stacked on the channel axis; the library does not rectify stereo pairs (section 9). The classical fusion functions of section 5 are an alternative to the `pansharpening` family that needs no training, and they can produce training references for it.

### 7.2 Generating the tables

```python
from unbihexium.zoo import list_models
from unbihexium.zoo.catalog import get_spec

FAMILIES = ["pansharpening", "multispectral_processor", "panchromatic_processor",
            "stereo_processor", "tri_stereo_processor", "synthetic_imagery"]
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

### 7.3 Running a tiny model

`predict` builds a catalogue model in memory with its deterministic starter weights and checks them against the published digest; `unbihexium zoo build` stores a model in the directory named by `UNBIHEXIUM_CACHE` (default `~/.cache/unbihexium`). The input stacks the four upsampled multispectral bands and the panchromatic band in the catalogue order. The output of the untrained model is meaningless; the example only shows the input layout and the result type.

```python
import numpy as np
from rasterio.transform import from_origin

from unbihexium.ai import predict
from unbihexium.io import write_geotiff

rng = np.random.default_rng(1)
bands = rng.uniform(0.02, 0.4, size=(5, 64, 64)).astype("float32")  # ms_blue, ms_green, ms_red, ms_nir, pan
write_geotiff(bands, "ms_pan.tif", crs="EPSG:32635", transform=from_origin(500000, 6700000, 2, 2))
result = predict("pansharpening_tiny", "ms_pan.tif")
print(type(result).__name__, result.model_id, result.bands, result.raster.shape, result.raster.resolution)
```

Output:

```text
EnhancementResult pansharpening_tiny ['blue', 'green', 'red', 'nir'] (4, 64, 64) (2.0, 2.0)
```

## 8. Command line

The command line has no commands specific to this capability; the preprocessing functions are used from Python. The model families are handled by the generic commands of [docs/reference/cli.md](../reference/cli.md), for example (with `ms_pan.tif` from section 7.3):

```bash
unbihexium zoo list --domain imaging --task enhancement --variant tiny
unbihexium zoo info pansharpening_tiny
unbihexium predict pansharpening_tiny ms_pan.tif sharpened.tif
```

The last command prints `Wrote: sharpened.tif (pansharpening_tiny)`. Spectral indices of multi-band rasters are computed with `unbihexium index`, described in [03_indices_flood_water.md](03_indices_flood_water.md).

## 9. Limitations

Earlier versions of this document described several features that have no code behind them in the current release. The following are not implemented:

- **Stereo photogrammetry.** There is no epipolar rectification, dense matching, triangulation or rational polynomial coefficient (RPC) sensor model. The stereo families of section 7 are untrained networks that expect rectified inputs.
- **Orthorectification and co-registration.** No geometric correction of raw imagery is implemented; the corresponding families listed in [01_ai_products.md](01_ai_products.md) are untrained. Rasters can be reprojected onto a common grid with `Raster.reproject` and `Raster.match` (GDAL warper through rasterio).
- **Atmospheric correction.** Only the image-based DOS1 method is available. Physically based correction (for example with a radiative transfer model) is not implemented; Level-2 products of the data providers should be used instead.
- **Pansharpening methods.** Only component substitution (Brovey, IHS, GSA) is implemented; principal component substitution and multiresolution analysis methods (for example wavelet-based fusion) are not.
- **Sensors.** Band tables exist for Sentinel-2, Landsat 8 and 9 and Sentinel-1 only. Other sensors are used by passing band arrays and coefficients explicitly.
- **Models.** The six families are untrained (section 1.2).

## 10. Related documents

- [README.md](../../README.md): project overview and installation.
- [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md): limits of the starter models.
- [11_resolution_metadata_qa.md](11_resolution_metadata_qa.md): pixel size, metadata, resampling and quality metrics.
- [03_indices_flood_water.md](03_indices_flood_water.md): spectral indices.
- [docs/model_zoo/training.md](../model_zoo/training.md), [docs/model_zoo/inference.md](../model_zoo/inference.md) and [docs/model_zoo/model_catalog.md](../model_zoo/model_catalog.md): training, inference and the complete catalogue.
- [index.md](index.md): overview of all capability documents.

## References

[1] European Space Agency. Sentinel-2 Products Specification Document, S2-PDGS-TAS-DI-PSD. 2021 onwards.

[2] U.S. Geological Survey. Landsat 8-9 Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS) Collection 2 Level 1 Data Format Control Book, LSDS-1822. 2020 onwards.

[3] U.S. Geological Survey. Landsat 8-9 Collection 2 Level 2 Science Product Guide, LSDS-1619. 2020 onwards.

[4] Chander, G., Markham, B. L., Helder, D. L. Summary of current radiometric calibration coefficients for Landsat MSS, TM, ETM+, and EO-1 ALI sensors. Remote Sensing of Environment 113(5), 893-903. 2009. <https://doi.org/10.1016/j.rse.2009.01.007>

[5] Spencer, J. W. Fourier series representation of the position of the sun. Search 2(5), 172. 1971.

[6] Chavez, P. S. Image-based atmospheric corrections, revisited and improved. Photogrammetric Engineering and Remote Sensing 62(9), 1025-1036. 1996.

[7] Main-Knorn, M., Pflug, B., Louis, J., Debaecker, V., Mueller-Wilm, U., Gascon, F. Sen2Cor for Sentinel-2. Proceedings of SPIE 10427, Image and Signal Processing for Remote Sensing XXIII. 2017. <https://doi.org/10.1117/12.2278218>

[8] Vivone, G., Alparone, L., Chanussot, J., Dalla Mura, M., Garzelli, A., Licciardi, G. A., Restaino, R., Wald, L. A critical comparison among pansharpening algorithms. IEEE Transactions on Geoscience and Remote Sensing 53(5), 2565-2586. 2015. <https://doi.org/10.1109/TGRS.2014.2361734>

[9] Gillespie, A. R., Kahle, A. B., Walker, R. E. Color enhancement of highly correlated images. II. Channel ratio and "chromaticity" transformation techniques. Remote Sensing of Environment 22(3), 343-365. 1987. <https://doi.org/10.1016/0034-4257(87)90088-5>

[10] Tu, T.-M., Su, S.-C., Shyu, H.-C., Huang, P. S. A new look at IHS-like image fusion methods. Information Fusion 2(3), 177-186. 2001. <https://doi.org/10.1016/S1566-2535(01)00036-7>

[11] Laben, C. A., Brower, B. V. Process for enhancing the spatial resolution of multispectral imagery using pan-sharpening. US Patent 6,011,875. 2000.

[12] Aiazzi, B., Baronti, S., Selva, M. Improving component substitution pansharpening through multivariate regression of MS+Pan data. IEEE Transactions on Geoscience and Remote Sensing 45(10), 3230-3239. 2007. <https://doi.org/10.1109/TGRS.2007.901007>

[13] Wald, L., Ranchin, T., Mangolini, M. Fusion of satellite images of different spatial resolutions: assessing the quality of resulting images. Photogrammetric Engineering and Remote Sensing 63(6), 691-699. 1997.

[14] Gonzalez, R. C., Woods, R. E. Digital Image Processing, 4th edition. Pearson. 2018.

<!--
=============================================================================
End of file docs/capabilities/10_satellite_imagery_features.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
