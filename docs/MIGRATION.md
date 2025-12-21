# Migration Guide

This document provides guidance for migrating between major versions of the Unbihexium library.

## Table of Contents

1. [Version Compatibility Matrix](#version-compatibility-matrix)
2. [Migrating to v1.0.0](#migrating-to-v100)
3. [API Changes](#api-changes)
4. [Deprecated Features](#deprecated-features)
5. [Breaking Changes](#breaking-changes)

---

## Version Compatibility Matrix

| Unbihexium Version | Python | ONNX Runtime | PyTorch | NumPy |
|-------------------|--------|--------------|---------|-------|
| 1.0.x | 3.10-3.12 | 1.15+ | 2.0+ | 1.24+ |
| 0.9.x | 3.9-3.11 | 1.14+ | 1.13+ | 1.23+ |
| 0.8.x | 3.8-3.10 | 1.12+ | 1.12+ | 1.22+ |

---

## Migrating to v1.0.0

### Prerequisites

Before migrating, ensure your environment meets the following requirements:

- Python 3.10 or higher
- ONNX Runtime 1.15 or higher
- Sufficient disk space for model assets (approximately 4 GB)

### Step-by-Step Migration

#### 1. Update Dependencies

```bash
pip install --upgrade unbihexium>=1.0.0
```

#### 2. Update Import Statements

The module structure has been reorganized for clarity:

```python
# Before (v0.x)
from unbihexium import detect_ships, segment_water

# After (v1.0)
from unbihexium.ai.detection import ShipDetector
from unbihexium.ai.segmentation import WaterDetector
```

#### 3. Update Model Loading

Model loading now uses the unified registry:

```python
# Before (v0.x)
model = load_model("ship_detector", variant="base")

# After (v1.0)
from unbihexium.zoo import ModelLoader
model = ModelLoader.load("ship_detector", variant="base")
```

#### 4. Update Raster Operations

The Raster class API has been refined:

```python
# Before (v0.x)
raster = Raster.open(path)
result = model.predict(raster.data)

# After (v1.0)
from unbihexium.core import Raster
raster = Raster.from_file(path)
result = model.predict(raster)
```

---

## API Changes

### Detection Module

| v0.x | v1.0 | Notes |
|------|------|-------|
| `detect(image)` | `predict(raster)` | Now accepts Raster objects |
| `threshold` parameter | `confidence_threshold` | Renamed for clarity |
| Returns `list[dict]` | Returns `DetectionResult` | Structured dataclass |

### Segmentation Module

| v0.x | v1.0 | Notes |
|------|------|-------|
| `segment(image)` | `predict(raster)` | Unified API |
| Returns `np.ndarray` | Returns `SegmentationResult` | Includes metadata |

### I/O Module

| v0.x | v1.0 | Notes |
|------|------|-------|
| `read_geotiff(path)` | `RasterIO.read(path)` | Class-based API |
| `write_geotiff(data, path)` | `RasterIO.write(raster, path)` | Accepts Raster objects |

---

## Deprecated Features

The following features are deprecated and will be removed in v2.0:

| Feature | Replacement | Removal Version |
|---------|-------------|-----------------|
| `unbihexium.legacy` module | Use main modules | v2.0 |
| `Model.run()` method | Use `Model.predict()` | v2.0 |
| String-based model loading | Use `ModelLoader` class | v2.0 |

---

## Breaking Changes

### v1.0.0

1. **Minimum Python Version**: Python 3.10 is now required
2. **Raster API**: `Raster.open()` renamed to `Raster.from_file()`
3. **Detection Output**: Now returns `DetectionResult` dataclass
4. **Segmentation Output**: Now returns `SegmentationResult` dataclass
5. **Model Registry**: Unified under `unbihexium.zoo` module

### Configuration Changes

The configuration system has been updated:

```python
# Before (v0.x)
import unbihexium
unbihexium.set_cache_dir("/path/to/cache")

# After (v1.0)
from unbihexium.config import settings
settings.cache_dir = "/path/to/cache"
```

---

## Troubleshooting

### Common Migration Issues

#### Issue: ImportError for deprecated modules

```
ImportError: cannot import name 'detect_ships' from 'unbihexium'
```

**Solution**: Update to the new class-based API:

```python
from unbihexium.ai.detection import ShipDetector
detector = ShipDetector()
result = detector.predict(raster)
```

#### Issue: Model loading fails

**Solution**: Clear the model cache and re-download:

```python
from unbihexium.zoo import ModelCache
ModelCache.clear()
```

---

## Support

For migration assistance, please:

1. Consult the API Reference documentation
2. Open an issue on GitHub with the migration label
3. Review the changelog for detailed version history
