# SAR Backscatter Normaliser

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `sar_amplitude` |
| Task | dense_regression |
| Domain | sar |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates terrain-flattened gamma0 backscatter from sigma0 and elevation.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `VV`
2. `VH`
3. `elevation`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `gamma0_vv` | dB |
| 1 | `gamma0_vh` | dB |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `sar_amplitude_tiny` | 732,946 | 256 | `f02906369953e0de` |
| `sar_amplitude_base` | 7,058,466 | 256 | `386e373ae8e577f5` |
| `sar_amplitude_large` | 22,059,122 | 512 | `763ea7660be4670f` |
| `sar_amplitude_mega` | 60,450,626 | 512 | `1abd5f9a0a1a2e8b` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("sar_amplitude_base")  # verified starter weights
```

## Training

Required reference data: Terrain-flattened gamma0 from radiometric terrain correction.

```bash
unbihexium train sar_amplitude_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-1 GRD
- Copernicus DEM

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
