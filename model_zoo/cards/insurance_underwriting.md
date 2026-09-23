# Insurance Risk Scorer

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `insurance_underwriting` |
| Task | scene_regression |
| Domain | risk |
| Architecture | encoder_regressor |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Scores the natural hazard risk of an insured site.

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
| 0 | `hazard_score` | 1 |
| 1 | `exposure_score` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `insurance_underwriting_tiny` | 495,522 | 256 | `6aef3a494b566a43` |
| `insurance_underwriting_base` | 3,747,138 | 256 | `9e1b2fe0139603d3` |
| `insurance_underwriting_large` | 14,609,186 | 512 | `6c371454fbe0b8ed` |
| `insurance_underwriting_mega` | 37,766,530 | 512 | `4cb28f4705828e8a` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("insurance_underwriting_base")  # verified starter weights
```

## Training

Required reference data: Claims history or risk model scores per site.

```bash
unbihexium train insurance_underwriting_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Copernicus DEM
- population and road distance rasters

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
