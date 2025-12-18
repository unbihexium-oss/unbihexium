# Model Zoo Assets

This directory contains tiny model files for smoke testing.

## Models

| Model | Task | Size |
|-------|------|------|
| ship_detector_tiny.pt | Detection | ~100KB |
| building_detector_tiny.pt | Detection | ~100KB |
| segmentation_tiny.pt | Segmentation | ~100KB |
| super_resolution_tiny.pt | Super Resolution | ~100KB |
| change_detector_tiny.pt | Change Detection | ~100KB |

## Usage

These models are automatically used by the test suite and CLI smoke tests.

```python
from unbihexium.zoo import download_model

path = download_model("ship_detector_tiny")
```

## Full Models

Full production models are available as GitHub Release assets.
