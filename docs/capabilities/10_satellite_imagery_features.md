# Capability 10: Satellite Imagery Features

## Executive Summary

This document provides comprehensive documentation for the Satellite Imagery Features capability domain within the Unbihexium framework. This domain encompasses specialized processing models for satellite imagery including stereo processing, pansharpening, multi-spectral fusion, and synthetic imagery generation.

The domain comprises 6 base model architectures with 24 total variants, focusing on image enhancement and specialized satellite data processing.

---

## Domain Overview

### Scope and Objectives

1. **Stereo Processing**: Generate 3D information from stereo and tri-stereo satellite imagery
2. **Pansharpening**: Fuse panchromatic and multispectral imagery for enhanced resolution
3. **Multispectral Processing**: Process and enhance multispectral satellite data
4. **Synthetic Imagery**: Generate synthetic satellite imagery for training and simulation

### Domain Statistics

| Metric | Value |
|--------|-------|
| Base Model Architectures | 6 |
| Total Model Variants | 24 |
| Minimum Parameters (tiny) | 186,177 |
| Maximum Parameters (mega) | 2,956,803 |
| Primary Tasks | Terrain, Enhancement |
| Production Status | Fully Production Ready |

---

## Model Inventory

### Complete Model Listing

| Model ID | Task | Architecture | Output | Variants | Parameter Range |
|----------|------|--------------|--------|----------|-----------------|
| stereo_processor | Terrain | CNN | DSM | 4 | 186,177 - 2,956,545 |
| tri_stereo_processor | Terrain | CNN | DSM | 4 | 186,177 - 2,956,545 |
| pansharpening | Enhancement | CNN | RGB | 4 | 186,243 - 2,956,803 |
| multispectral_processor | Enhancement | CNN | MS | 4 | 186,243 - 2,956,803 |
| panchromatic_processor | Enhancement | CNN | Pan | 4 | 186,243 - 2,956,803 |
| synthetic_imagery | Enhancement | CNN | RGB | 4 | 186,243 - 2,956,803 |

---

## Performance Metrics

| Model | Metric | Tiny | Base | Large | Mega | Reference |
|-------|--------|------|------|-------|------|-----------|
| stereo_processor | RMSE | 2.0m | 1.5m | 1.0m | 0.6m | Ground Truth |
| pansharpening | PSNR | 30 dB | 33 dB | 36 dB | 39 dB | Reference |
| pansharpening | SSIM | 0.85 | 0.90 | 0.94 | 0.97 | Reference |
| multispectral_processor | SNR | 25 dB | 30 dB | 35 dB | 40 dB | Reference |

---

## Pansharpening Methodology

### Component Substitution Methods

#### Intensity-Hue-Saturation (IHS)

RGB to IHS transformation:

$$
\begin{bmatrix} I \\ v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 1/3 & 1/3 & 1/3 \\ -\sqrt{2}/6 & -\sqrt{2}/6 & 2\sqrt{2}/6 \\ 1/\sqrt{2} & -1/\sqrt{2} & 0 \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix}
$$

Substitution:

$$
I_{new} = P_{histogram-matched}
$$

Inverse transformation:

$$
\begin{bmatrix} R' \\ G' \\ B' \end{bmatrix} = \begin{bmatrix} 1 & -1/\sqrt{2} & 1/\sqrt{2} \\ 1 & -1/\sqrt{2} & -1/\sqrt{2} \\ 1 & \sqrt{2} & 0 \end{bmatrix} \begin{bmatrix} I_{new} \\ v_1 \\ v_2 \end{bmatrix}
$$

#### Principal Component Analysis (PCA)

1. Transform MS bands to principal components
2. Replace PC1 with histogram-matched Pan
3. Inverse transform to get sharpened MS

$$
\text{PCA}: X_{MS} \rightarrow PC_1, PC_2, ..., PC_n
$$

$$
\text{Substitution}: PC_1 \leftarrow P_{matched}
$$

$$
\text{Inverse}: PC'_1, PC_2, ..., PC_n \rightarrow X'_{MS}
$$

#### Gram-Schmidt (GS)

1. Simulate a low-resolution Pan from MS bands
2. Apply Gram-Schmidt orthogonalization
3. Substitute first component with high-res Pan
4. Reverse transform

### Multi-Resolution Analysis Methods

#### Generalized Intensity Modulation (GIA)

$$
MS'_i = MS_i \times \frac{P}{\tilde{P}}
$$

Where $\tilde{P}$ is the low-pass filtered Pan.

#### Additive Wavelet Transform

$$
MS'_i = MS_i + g_i \times (P - \tilde{P})
$$

Where $g_i$ is the injection gain for band $i$.

### Quality Assessment Metrics

Spectral Fidelity:

$$
\text{SAM}(\mathbf{x}, \mathbf{y}) = \arccos\left(\frac{\mathbf{x} \cdot \mathbf{y}}{||\mathbf{x}|| \cdot ||\mathbf{y}||}\right)
$$

Spatial Quality:

$$
\text{SCC} = \frac{\sum_i (H_i - \bar{H})(H'_i - \bar{H'})}{\sqrt{\sum_i (H_i - \bar{H})^2 \cdot \sum_i (H'_i - \bar{H'})^2}}
$$

Q-Index (Universal Quality Index):

$$
Q = \frac{4 \sigma_{xy} \bar{x} \bar{y}}{(\sigma_x^2 + \sigma_y^2)[(\bar{x})^2 + (\bar{y})^2]}
$$

ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse):

$$
\text{ERGAS} = 100 \times \frac{h}{l} \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\frac{\text{RMSE}_i}{\mu_i}\right)^2}
$$

Where $h/l$ is the ratio of high to low resolution.

---

## Stereo Processing

### Stereo Geometry

Base-to-Height ratio:

$$
B/H = \frac{B}{H} = \frac{\text{baseline}}{\text{altitude}}
$$

Optimal B/H for different applications:

| Application | Optimal B/H | Height Precision |
|-------------|-------------|------------------|
| Urban mapping | 0.5-0.7 | Best |
| Forestry | 0.3-0.5 | Good |
| Mountainous | 0.3-0.4 | Adequate |

### Triangulation

Point cloud generation from disparity:

$$
Z = \frac{f \times B}{d}
$$

Where:
- $Z$ = Depth
- $f$ = Focal length
- $B$ = Baseline
- $d$ = Disparity

### 3D Coordinate Computation

$$
X = \frac{(x - x_0) \times Z}{f}
$$

$$
Y = \frac{(y - y_0) \times Z}{f}
$$

---

## Satellite Sensor Specifications

### Optical Sensors

| Satellite | Pan (m) | MS (m) | Swath (km) | Revisit |
|-----------|---------|--------|------------|---------|
| WorldView-3 | 0.31 | 1.24 | 13.1 | 1 day |
| WorldView-2 | 0.46 | 1.84 | 16.4 | 1.1 days |
| Pleiades-NEO | 0.30 | 1.20 | 14 | Daily |
| SPOT-7 | 1.5 | 6.0 | 60 | 1 day |
| Sentinel-2 | - | 10 | 290 | 5 days |
| Landsat-9 | 15 | 30 | 185 | 16 days |

### Spectral Bands

| Sensor | Blue | Green | Red | NIR | SWIR1 | SWIR2 |
|--------|------|-------|-----|-----|-------|-------|
| WorldView-3 | 450-510 | 510-580 | 630-690 | 770-895 | 1195-1225 | 2145-2185 |
| Sentinel-2 | 458-523 | 543-578 | 650-680 | 785-900 | 1565-1655 | 2100-2280 |
| Landsat-9 | 452-512 | 533-590 | 636-673 | 851-879 | 1566-1651 | 2107-2294 |

---

## Usage Examples

### CLI Usage

```bash
# Pansharpening
unbihexium infer pansharpening_mega \
    --input-pan panchromatic.tif \
    --input-ms multispectral.tif \
    --output pansharpened.tif \
    --method gram-schmidt

# Stereo processing
unbihexium infer stereo_processor_mega \
    --input-left image_left.tif \
    --input-right image_right.tif \
    --output dsm.tif \
    --rpc left.rpc right.rpc

# Tri-stereo processing
unbihexium infer tri_stereo_processor_large \
    --input-nadir nadir.tif \
    --input-forward forward.tif \
    --input-backward backward.tif \
    --output dsm_tri.tif

# Multispectral enhancement
unbihexium infer multispectral_processor_large \
    --input sentinel2.tif \
    --output enhanced.tif \
    --denoise true \
    --atmospheric-correction true
```

### Python API Usage

```python
from unbihexium import Pipeline, Config
from unbihexium.zoo import get_model
import rasterio
import numpy as np

# Pansharpening
pan_model = get_model("pansharpening_mega")

config = Config(
    tile_size=512,
    overlap=64,
    batch_size=4,
    device="cuda:0"
)

pan_pipeline = Pipeline.from_config(
    capability="pansharpening",
    variant="mega",
    config=config
)

# Load inputs
with rasterio.open("panchromatic.tif") as src:
    pan = src.read(1)
    pan_profile = src.profile

with rasterio.open("multispectral.tif") as src:
    ms = src.read()
    ms_profile = src.profile

# Run pansharpening
sharpened = pan_pipeline.run(
    pan="panchromatic.tif",
    ms="multispectral.tif"
)

sharpened.save("pansharpened.tif")

# Calculate quality metrics
from unbihexium.metrics import calculate_sam, calculate_ergas, calculate_qindex

sam = calculate_sam(ms_reference, sharpened)
ergas = calculate_ergas(ms_reference, sharpened, ratio=4)
q = calculate_qindex(ms_reference, sharpened)

print(f"SAM: {sam:.4f} (lower is better)")
print(f"ERGAS: {ergas:.4f} (lower is better)")
print(f"Q-index: {q:.4f} (higher is better)")

# Stereo Processing
stereo_model = get_model("stereo_processor_mega")

stereo_config = Config(
    tile_size=512,
    overlap=128,
    batch_size=2,
    device="cuda:0",
    disparity_range=(-64, 64)
)

stereo_pipeline = Pipeline.from_config(
    capability="stereo_processing",
    variant="mega",
    config=stereo_config
)

dsm = stereo_pipeline.run(
    left="image_left.tif",
    right="image_right.tif",
    rpc_left="left.rpc",
    rpc_right="right.rpc"
)

dsm.save("dsm.tif")

# Statistics
print(f"DSM range: {dsm.min():.1f}m - {dsm.max():.1f}m")
print(f"Mean elevation: {dsm.mean():.1f}m")
print(f"Coverage: {dsm.valid_percentage:.1f}%")
```

---

## Technical Requirements

### Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| CPU | 4 cores | 8 cores | 16+ cores |
| RAM | 16 GB | 32 GB | 64 GB |
| GPU | None | RTX 3070 | A100 |
| Storage | 100 GB | 500 GB | 2 TB |

### Input Data Requirements

| Data Type | Format | Requirements |
|-----------|--------|--------------|
| Panchromatic | GeoTIFF | 16-bit, georeferenced |
| Multispectral | GeoTIFF | 16-bit, aligned to Pan |
| Stereo Pair | GeoTIFF | RPC or rigorous model |
| Tri-stereo | GeoTIFF | Three near-parallel views |

---

## Quality Assurance

### Pansharpening Quality Targets

| Metric | Target | Excellent |
|--------|--------|-----------|
| SAM | < 0.05 | < 0.02 |
| ERGAS | < 3.0 | < 1.5 |
| Q-index | > 0.9 | > 0.95 |
| SSIM | > 0.9 | > 0.95 |

### Stereo Quality Targets

| Metric | Target | Excellent |
|--------|--------|-----------|
| RMSE | < 2.0m | < 1.0m |
| LE90 | < 3.0m | < 1.5m |
| Completeness | > 90% | > 95% |

---

## References

1. Vivone, G. et al. (2015). A Critical Comparison Among Pansharpening Algorithms. IEEE TGRS.
2. Hirschmüller, H. (2008). Stereo Processing by Semiglobal Matching. IEEE TPAMI.
3. Toutin, T. (2004). Geometric Processing of Remote Sensing Images. International Journal of RS.
4. Alparone, L. et al. (2015). Remote Sensing Image Fusion. CRC Press.
5. Gleyzes, M.A. et al. (2012). Pleiades System Architecture. ISPRS Archives.
