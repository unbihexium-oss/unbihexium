# Yield Predictor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `yield_predictor` |
| Task | scene_regression |
| Domain | agriculture |
| Architecture | encoder_regressor |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Predicts crop yield of a field chip.

## Inputs

10 channels, float32, shape (N, 10, H, W):

1. `B02`
2. `B03`
3. `B04`
4. `B05`
5. `B06`
6. `B07`
7. `B08`
8. `B8A`
9. `B11`
10. `B12`

## Outputs

Tensor (N, K) with one value per target and chip.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `yield` | t ha-1 |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `yield_predictor_tiny` | 495,489 | 256 | `1f974ae8bcfcab57` |
| `yield_predictor_base` | 3,747,073 | 256 | `d32078d9364a7d8c` |
| `yield_predictor_large` | 14,609,089 | 512 | `407c6b533823f694` |
| `yield_predictor_mega` | 37,766,401 | 512 | `c20e80ed603b9a78` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("yield_predictor_base")  # verified starter weights
```

## Training

Required reference data: Field-level yield records.

```bash
unbihexium train yield_predictor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
