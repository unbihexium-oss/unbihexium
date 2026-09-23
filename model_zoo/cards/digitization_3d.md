# Building Height Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `digitization_3d` |
| Task | dense_regression |
| Domain | urban |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates building heights from imagery and a surface model.

## Inputs

4 channels, float32, shape (N, 4, H, W):

1. `red`
2. `green`
3. `blue`
4. `surface_height`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `building_height` | m |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `digitization_3d_tiny` | 733,073 | 256 | `b094c68e59e834fc` |
| `digitization_3d_base` | 7,058,721 | 256 | `796043b78aeab35f` |
| `digitization_3d_large` | 22,059,505 | 512 | `2bf3b93bec2fe03d` |
| `digitization_3d_mega` | 60,451,137 | 512 | `2ba65dc138854885` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("digitization_3d_base")  # verified starter weights
```

## Training

Required reference data: Building heights from LiDAR or cadastral 3D models.

```bash
unbihexium train digitization_3d_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- RGB orthophotos with a DSM

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
