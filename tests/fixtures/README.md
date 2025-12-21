# Test Fixtures

This directory contains sample data files for testing the Unbihexium library.

## Available Fixtures

| File | Description | Shape | Data Type |
|------|-------------|-------|-----------|
| `sample_data.npy` | Generic 3-band raster | (3, 64, 64) | float32 |
| `sentinel2_sample.npy` | Simulated Sentinel-2 10-band imagery | (10, 64, 64) | float32 |
| `dem_sample.npy` | Digital Elevation Model sample | (1, 128, 128) | float32 |
| `mask_sample.npy` | Segmentation mask sample | (64, 64) | uint8 |

## Usage

```python
import numpy as np
from pathlib import Path

fixtures_dir = Path(__file__).parent

# Load Sentinel-2 sample
sentinel2 = np.load(fixtures_dir / "sentinel2_sample.npy")

# Load DEM sample
dem = np.load(fixtures_dir / "dem_sample.npy")

# Load mask sample
mask = np.load(fixtures_dir / "mask_sample.npy")
```

## Band Information

### Sentinel-2 Sample

The Sentinel-2 sample simulates the following bands:

| Index | Band | Central Wavelength | Resolution |
|-------|------|-------------------|------------|
| 0 | B2 (Blue) | 490 nm | 10m |
| 1 | B3 (Green) | 560 nm | 10m |
| 2 | B4 (Red) | 665 nm | 10m |
| 3 | B5 (Red Edge 1) | 705 nm | 20m |
| 4 | B6 (Red Edge 2) | 740 nm | 20m |
| 5 | B7 (Red Edge 3) | 783 nm | 20m |
| 6 | B8 (NIR) | 842 nm | 10m |
| 7 | B8A (NIR Narrow) | 865 nm | 20m |
| 8 | B11 (SWIR 1) | 1610 nm | 20m |
| 9 | B12 (SWIR 2) | 2190 nm | 20m |

### DEM Sample

- Elevation range: 0-1000 meters
- Simulated terrain with random values

### Mask Sample

- 5 classes (0-4)
- Suitable for testing segmentation outputs
