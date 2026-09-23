# Model Cards

This document provides summary information about the pre-trained models in the Unbihexium Model Zoo.

## Overview

| Metric | Value |
|--------|-------|
| **Total Models** | 520 |
| **Total Parameters** | 515 million |
| **Architectures** | 130 unique |
| **Variants** | 4 (tiny, base, large, mega) |
| **Format** | ONNX Runtime compatible |

## Model Variants

| Variant | Input Size | Parameters | Target Use Case |
|---------|------------|------------|-----------------|
| **Tiny** | 64×64 | ~1M | Edge devices, real-time |
| **Base** | 128×128 | ~5M | Standard inference |
| **Large** | 256×256 | ~20M | High accuracy |
| **Mega** | 512×512 | ~50M | Maximum precision |

## Task Categories

### Detection Models (76 total)

| Model | Description | mAP@0.5 | Inference (ms) |
|-------|-------------|---------|----------------|
| `ship_detector` | Maritime vessel detection | 0.89 | 12 |
| `building_detector` | Building footprint detection | 0.92 | 15 |
| `aircraft_detector` | Aircraft detection | 0.87 | 11 |
| `vehicle_detector` | Ground vehicle detection | 0.85 | 10 |
| `solar_panel_detector` | Solar installation detection | 0.91 | 14 |
| `oil_storage_detector` | Oil tank detection | 0.93 | 13 |

### Segmentation Models (128 total)

| Model | Description | mIoU | Inference (ms) |
|-------|-------------|------|----------------|
| `water_segmenter` | Water body segmentation | 0.94 | 18 |
| `crop_segmenter` | Agricultural land segmentation | 0.88 | 20 |
| `forest_segmenter` | Forest cover mapping | 0.91 | 19 |
| `urban_segmenter` | Urban area mapping | 0.86 | 22 |
| `road_segmenter` | Road network extraction | 0.83 | 17 |
| `cloud_segmenter` | Cloud/shadow masking | 0.95 | 15 |

### Regression Models (188 total)

| Model | Description | RMSE | Inference (ms) |
|-------|-------------|------|----------------|
| `biomass_estimator` | Above-ground biomass | 25.3 t/ha | 25 |
| `carbon_estimator` | Carbon stock estimation | 12.1 tC/ha | 24 |
| `lai_estimator` | Leaf Area Index | 0.42 | 20 |
| `ndvi_predictor` | NDVI prediction | 0.05 | 12 |
| `albedo_estimator` | Surface albedo | 0.02 | 18 |
| `soil_moisture` | Soil moisture content | 4.2% | 22 |

### Terrain Models (52 total)

| Model | Description | MAE | Inference (ms) |
|-------|-------------|-----|----------------|
| `dem_super_resolution` | DEM enhancement 2×-8× | 0.8m | 30 |
| `slope_analyzer` | Slope gradient computation | 1.2° | 8 |
| `aspect_analyzer` | Aspect direction | 5.4° | 8 |
| `curvature_analyzer` | Terrain curvature | 0.01 | 10 |
| `hillshade_generator` | Analytical hillshading | N/A | 5 |
| `tpi_calculator` | Topographic Position Index | 0.3 | 12 |

## Supported Satellites

All models are trained and validated on data from:

- **Sentinel-2** (MSI, 10-60m)
- **Landsat-8/9** (OLI, 30m)
- **WorldView-2/3** (0.3-1.2m)
- **Planet** (PlanetScope, 3-5m)
- **MODIS** (250m-1km)

## License

All models are released under the Mozilla Public License 2.0 (MPL-2.0).

## Citation

```bibtex
@software{unbihexium2024,
  title = {Unbihexium: Production-Grade Geospatial AI Library},
  author = {Unbihexium Contributors},
  year = {2024},
  url = {https://github.com/unbihexium-oss/unbihexium}
}
```
