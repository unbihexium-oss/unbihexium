# Marine Pollution Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `marine_pollution_detector` |
| Task | segmentation |
| Domain | water |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Segments floating debris and surface pollution at sea.

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

Tensor (N, K, H, W) of class logits; apply softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `water` | - |
| 1 | `floating_debris` | - |
| 2 | `oil_sheen` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `marine_pollution_detector_tiny` | 733,971 | 256 | `1d23facf80938d81` |
| `marine_pollution_detector_base` | 7,060,515 | 256 | `7134db2efaa58833` |
| `marine_pollution_detector_large` | 22,062,195 | 512 | `a3df6b8cff64c398` |
| `marine_pollution_detector_mega` | 60,454,723 | 512 | `88c77d45e920e560` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("marine_pollution_detector_base")  # verified starter weights
```

## Training

Required reference data: Pollution masks.

```bash
unbihexium train marine_pollution_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
