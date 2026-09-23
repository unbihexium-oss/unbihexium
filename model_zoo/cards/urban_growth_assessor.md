# Urban Growth Assessor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `urban_growth_assessor` |
| Task | change_detection |
| Domain | urban |
| Architecture | unet_early_fusion |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects new built-up area between two dates.

## Inputs

20 channels, float32, shape (N, 20, H, W):

1. `B02_t1`
2. `B03_t1`
3. `B04_t1`
4. `B05_t1`
5. `B06_t1`
6. `B07_t1`
7. `B08_t1`
8. `B8A_t1`
9. `B11_t1`
10. `B12_t1`
11. `B02_t2`
12. `B03_t2`
13. `B04_t2`
14. `B05_t2`
15. `B06_t2`
16. `B07_t2`
17. `B08_t2`
18. `B8A_t2`
19. `B11_t2`
20. `B12_t2`

## Outputs

Tensor (N, K, H, W) of change class logits; softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `no_change` | - |
| 1 | `urban_expansion` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `urban_growth_assessor_tiny` | 735,394 | 256 | `ca454e80e6d41f59` |
| `urban_growth_assessor_base` | 7,063,362 | 256 | `853679a701edacb1` |
| `urban_growth_assessor_large` | 22,066,466 | 512 | `3c97701f34ecbb21` |
| `urban_growth_assessor_mega` | 60,460,418 | 512 | `211a474b7c468a65` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("urban_growth_assessor_base")  # verified starter weights
```

## Training

Required reference data: Urban expansion masks.

```bash
unbihexium train urban_growth_assessor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A image pairs
- Landsat 8/9 image pairs

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
