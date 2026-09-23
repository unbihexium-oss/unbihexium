# InSAR Phase Unwrapper

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `sar_phase_displacement` |
| Task | dense_regression |
| Domain | sar |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates unwrapped phase from a wrapped interferogram.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `cos_phase`
2. `sin_phase`
3. `coherence`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `unwrapped_phase` | rad |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `sar_phase_displacement_tiny` | 732,929 | 256 | `8bea37fb15961d21` |
| `sar_phase_displacement_base` | 7,058,433 | 256 | `60d15effbd75bbae` |
| `sar_phase_displacement_large` | 22,059,073 | 512 | `c1407634ccf6ab5d` |
| `sar_phase_displacement_mega` | 60,450,561 | 512 | `cd04e615dafba067` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("sar_phase_displacement_base")  # verified starter weights
```

## Training

Required reference data: Unwrapped phase from SNAPHU or similar processing.

```bash
unbihexium train sar_phase_displacement_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-1 SLC interferograms

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
