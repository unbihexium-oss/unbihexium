# Model Card: ubx-flood-risk-mlp-1.0.0

## Model Overview

| Property | Value |
| ---------- | ------- |
| Model ID | ubx-flood-risk-mlp-1.0.0 |
| Task | Risk Scoring |
| Architecture | MLP |
| Version | 1.0.0 |
| License | MPL-2.0 |
| Status | Production |

## Description

Multi-layer perceptron for flood risk scoring based on terrain and environmental features. Outputs a normalized risk score between 0 and 1.

## Architecture

```mermaid
graph LR
    I[10 Features] --> H1[Dense 32 + ReLU]
    H1 --> D1[Dropout 0.1]
    D1 --> H2[Dense 16 + ReLU]
    H2 --> D2[Dropout 0.1]
    D2 --> O[Risk Score]
```

## Input Features

| Index | Feature | Description |
| ------- | --------- | ------------- |
| 0 | elevation | Terrain elevation |
| 1 | slope | Terrain slope |
| 2 | distance_to_water | Distance to nearest water body |
| 3 | precipitation | Average precipitation |
| 4 | soil_type | Soil permeability index |
| 5 | land_cover | Land cover type index |
| 6 | drainage | Drainage capacity |
| 7 | population | Population density |
| 8 | infrastructure | Infrastructure value |
| 9 | historical_flood | Historical flood frequency |

## Performance Metrics

| Metric | Value |
| -------- | ------- |
| RMSE | 0.12 |
| R-squared | 0.85 |

### RMSE Formula

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

## Intended Use

- Flood hazard mapping
- Emergency planning
- Insurance risk assessment
- Urban planning support

## Limitations

- Trained on synthetic data
- Does not model temporal dynamics
- Assumes static feature values
- Not validated against real flood events

## Ethical Considerations

- Results should inform, not replace, expert judgment
- Avoid discrimination in insurance applications
- Transparent about model uncertainty

## Provenance

- Architecture: `src/unbihexium/ai/models/mlp.py`
- Training: `scripts/train_all_models.py`
- Seed: 42
