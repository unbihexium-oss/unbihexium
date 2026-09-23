# Preparedness Scorer

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `preparedness_manager` |
| Task | scene_regression |
| Domain | risk |
| Architecture | encoder_regressor |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Scores the disaster preparedness of a community area.

## Inputs

10 channels, float32, shape (N, 10, H, W):

1. `B02`
2. `B03`
3. `B04`
4. `B08`
5. `B11`
6. `B12`
7. `elevation`
8. `slope`
9. `population`
10. `road_distance`

## Outputs

Tensor (N, K) with one value per target and chip.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `preparedness` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `preparedness_manager_tiny` | 495,489 | 256 | `cae9a826cf5977a5` |
| `preparedness_manager_base` | 3,747,073 | 256 | `23767f595c1e6465` |
| `preparedness_manager_large` | 14,609,089 | 512 | `5553a3b460003457` |
| `preparedness_manager_mega` | 37,766,401 | 512 | `673f0b9f2e3adcfe` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("preparedness_manager_base")  # verified starter weights
```

## Training

Required reference data: Preparedness scores from surveys.

```bash
unbihexium train preparedness_manager_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Copernicus DEM
- population and road distance rasters

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
