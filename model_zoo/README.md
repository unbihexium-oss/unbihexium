# Unbihexium Model Zoo

## Overview

The Unbihexium Model Zoo contains **520 production-ready models** for geospatial AI and Earth observation applications. All models are available in ONNX and PyTorch formats with SHA256 checksums for integrity verification.

**Important:** Models are stored using Git LFS. After cloning, run `git lfs pull` to download actual model files.

```bash
# Clone and download models
git clone https://github.com/unbihexium-oss/unbihexium.git
cd unbihexium
git lfs install
git lfs pull  # Downloads all models (~4 GB)

# OR download specific variant only
git lfs pull --include "model_zoo/assets/tiny/*"
```

---

## Quick Statistics

| Metric | Value |
| -------- | ------- |
| Total Models | 520 |
| Base Architectures | 130 |
| Variants | 4 (tiny, base, large, mega) |
| Total Parameters | 515,466,484 |
| Formats | ONNX, PyTorch |

---

## Directory Structure

```
model_zoo/
├── assets/
│   ├── tiny/           # 130 tiny models (32ch, 64px)
│   │   ├── ship_detector_tiny/
│   │   │   ├── model.onnx
│   │   │   ├── model.pt
│   │   │   ├── config.json
│   │   │   └── model.sha256
│   │   └── ...
│   ├── base/           # 130 base models (64ch, 128px)
│   ├── large/          # 130 large models (96ch, 256px)
│   └── mega/           # 130 mega models (128ch, 512px)
└── README.md
```

---

## Variant Specifications

| Variant | Resolution | Channels | Params Range | Use Case |
| --------- | ------------ | ---------- | -------------- | ---------- |
| tiny | 64x64 | 32 | 50K-259K | Edge, real-time |
| base | 128x128 | 64 | 191K-1M | Standard production |
| large | 256x256 | 96 | 425K-2.3M | High accuracy |
| mega | 512x512 | 128 | 752K-4.1M | Maximum quality |

---

## Usage

### CLI

```bash
# List models
unbihexium zoo list

# Download model
unbihexium zoo download ship_detector_mega --verify

# Model info
unbihexium zoo info building_detector_large
```

### Python

```python
from unbihexium.zoo import get_model, list_models

# List available models
models = list_models(variant="mega")

# Load model
model = get_model("ship_detector_mega")
predictions = model.predict(image)
```

---

## Model Categories

| Category | Models | Task |
| ---------- | -------- | ------ |
| Detection | 19 | Object localization |
| Segmentation | 32 | Pixel classification |
| Regression | 47 | Value prediction |
| Terrain | 13 | Elevation products |
| Enhancement | 11 | Image improvement |
| Index | 7 | Spectral indices |
| Super Resolution | 1 | Upscaling |

---

## Verification

All models include SHA256 checksums:

```bash
# Verify single model
sha256sum -c model.sha256

# Verify all models
python scripts/validate_models.py
```

---

## License

Models are licensed under MPL-2.0, same as the library.

See [model_catalog.md](../docs/model_zoo/model_catalog.md) for complete documentation.
