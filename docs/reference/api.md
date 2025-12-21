# API Reference

This document provides the complete API reference for the Unbihexium library.

## Table of Contents

1. [Core Module](#core-module)
2. [AI Module](#ai-module)
3. [I/O Module](#io-module)
4. [Analysis Module](#analysis-module)
5. [Geostatistics Module](#geostatistics-module)
6. [SAR Module](#sar-module)
7. [Terrain Module](#terrain-module)
8. [Serving Module](#serving-module)
9. [Registry Module](#registry-module)
10. [Zoo Module](#zoo-module)

---

## Core Module

::: unbihexium.core

### Raster

The primary data structure for geospatial raster data.

```python
from unbihexium.core import Raster

# Load from file
raster = Raster.from_file("path/to/image.tif")

# Create from array
raster = Raster.from_array(data, crs="EPSG:4326", transform=transform)

# Properties
print(raster.shape)      # (bands, height, width)
print(raster.crs)        # Coordinate Reference System
print(raster.bounds)     # Bounding box
print(raster.resolution) # Pixel resolution
```

### Scene

Container for multi-temporal or multi-sensor imagery.

```python
from unbihexium.core import Scene

scene = Scene(source="path/to/data")
scene.load()
```

### Model

Base class for all inference models.

```python
from unbihexium.core import Model

model = Model.load("model_name", variant="base")
result = model.predict(raster)
```

---

## AI Module

::: unbihexium.ai

### Detection

```python
from unbihexium.ai.detection import (
    ShipDetector,
    BuildingDetector,
    AircraftDetector,
    VehicleDetector,
)

# Initialize detector
detector = ShipDetector(threshold=0.5)

# Run detection
result = detector.predict(raster)

# Access results
for detection in result.detections:
    print(f"Class: {detection.class_name}")
    print(f"Confidence: {detection.confidence}")
    print(f"BBox: {detection.bbox}")
```

### Segmentation

```python
from unbihexium.ai.segmentation import (
    WaterDetector,
    CropDetector,
    ChangeDetector,
    SemanticSegmenter,
)

# Initialize segmenter
segmenter = WaterDetector(threshold=0.5)

# Run segmentation
result = segmenter.predict(raster)

# Access mask
mask = result.mask  # numpy array
classes = result.classes  # list of class names
```

### Super Resolution

```python
from unbihexium.ai.super_resolution import SuperResolution

sr = SuperResolution(scale_factor=2, tile_size=256)
result = sr.enhance(raster)
enhanced_raster = result.raster
```

---

## I/O Module

::: unbihexium.io

### RasterIO

```python
from unbihexium.io import RasterIO

# Read
raster = RasterIO.read("input.tif")

# Write
RasterIO.write(raster, "output.tif", driver="GTiff")

# Supported formats
formats = RasterIO.supported_formats()
```

### STACIO

```python
from unbihexium.io import STACIO

client = STACIO(catalog_url="https://earth-search.aws.element84.com/v1")

# Search
items = client.search(
    bbox=[-122.5, 37.5, -122.0, 38.0],
    datetime="2023-01-01/2023-12-31",
    collections=["sentinel-2-l2a"]
)

# Load
raster = client.load_item(items[0])
```

### ZarrIO

```python
from unbihexium.io import ZarrIO

# Read Zarr store
store = ZarrIO.open("data.zarr")

# Write to Zarr
ZarrIO.write(raster, "output.zarr", chunks=(1, 256, 256))
```

---

## Analysis Module

::: unbihexium.analysis

### Zonal Statistics

```python
from unbihexium.analysis import zonal_statistics

result = zonal_statistics(raster, zones)

for zone in result:
    print(f"Zone {zone.zone_id}: mean={zone.mean}, std={zone.std}")
```

### Weighted Overlay

```python
from unbihexium.analysis import weighted_overlay

layers = [slope, aspect, elevation]
weights = [0.4, 0.3, 0.3]
suitability = weighted_overlay(layers, weights)
```

### Network Analysis

```python
from unbihexium.analysis import NetworkAnalyzer

analyzer = NetworkAnalyzer()
path = analyzer.shortest_path(start, end)
area = analyzer.service_area(origin, distance=1000)
```

---

## Geostatistics Module

::: unbihexium.geostat

### Kriging

```python
from unbihexium.geostat import OrdinaryKriging, UniversalKriging

# Ordinary Kriging
ok = OrdinaryKriging(variogram_model="spherical")
prediction, variance = ok.execute(points, values, grid)

# Universal Kriging
uk = UniversalKriging(drift_terms=["specified"])
prediction, variance = uk.execute(points, values, grid)
```

### Variogram

```python
from unbihexium.geostat import Variogram

variogram = Variogram(points, values)
variogram.fit(model="spherical")
print(f"Sill: {variogram.sill}")
print(f"Range: {variogram.range}")
print(f"Nugget: {variogram.nugget}")
```

### Spatial Autocorrelation

```python
from unbihexium.geostat import MoransI, GearysC

# Moran's I
moran = MoransI(data, weights)
print(f"I: {moran.I}, p-value: {moran.p_value}")

# Geary's C
geary = GearysC(data, weights)
print(f"C: {geary.C}, p-value: {geary.p_value}")
```

---

## SAR Module

::: unbihexium.sar

### Amplitude Processing

```python
from unbihexium.sar import (
    calibrate_amplitude,
    compute_sigma0,
    compute_gamma0,
    speckle_filter,
)

# Calibrate
calibrated = calibrate_amplitude(sar_data, metadata)

# Compute backscatter
sigma0 = compute_sigma0(calibrated, incidence_angle)
gamma0 = compute_gamma0(calibrated, incidence_angle)

# Filter speckle
filtered = speckle_filter(sigma0, method="lee", window_size=5)
```

### Polarimetry

```python
from unbihexium.sar import (
    pauli_decomposition,
    freeman_durden_decomposition,
    h_alpha_decomposition,
)

# Pauli decomposition
pauli = pauli_decomposition(coherency_matrix)

# Freeman-Durden decomposition
components = freeman_durden_decomposition(coherency_matrix)

# H-Alpha decomposition
h, alpha, anisotropy = h_alpha_decomposition(coherency_matrix)
```

### Interferometry

```python
from unbihexium.sar import (
    compute_interferogram,
    compute_coherence,
    phase_unwrapping,
)

# Compute interferogram
interferogram = compute_interferogram(master, slave)

# Compute coherence
coherence = compute_coherence(master, slave, window_size=5)

# Unwrap phase
unwrapped = phase_unwrapping(interferogram)
```

---

## Terrain Module

::: unbihexium.terrain

### Terrain Analysis

```python
from unbihexium.terrain import (
    slope,
    aspect,
    hillshade,
    curvature,
    tpi,
    tri,
)

# Compute derivatives
slope_map = slope(dem, resolution=10)
aspect_map = aspect(dem, resolution=10)

# Visualization
shaded = hillshade(dem, azimuth=315, altitude=45)

# Terrain indices
tpi_map = tpi(dem, radius=100)
tri_map = tri(dem)
```

---

## Serving Module

::: unbihexium.serving

### FastAPI Application

```python
from unbihexium.serving import create_app

app = create_app()

# Run with uvicorn
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | GET | List available models |
| `/capabilities` | GET | List capabilities |
| `/predict` | POST | Run inference |

---

## Registry Module

::: unbihexium.registry

### Model Registry

```python
from unbihexium.registry import ModelRegistry

# List models
models = ModelRegistry.list_all()

# Get model by name
model_info = ModelRegistry.get("ship_detector")

# Filter by capability
detection_models = ModelRegistry.by_capability("detection")
```

### Capability Registry

```python
from unbihexium.registry import CapabilityRegistry

# List capabilities
capabilities = CapabilityRegistry.list_all()

# Get capability details
cap = CapabilityRegistry.get("water_detection")
```

---

## Zoo Module

::: unbihexium.zoo

### Model Loader

```python
from unbihexium.zoo import ModelLoader

# Load model
model = ModelLoader.load("ship_detector", variant="base")

# Load with custom cache
model = ModelLoader.load(
    "building_detector",
    variant="large",
    cache_dir="/custom/cache"
)
```

### Model Cache

```python
from unbihexium.zoo import ModelCache

# List cached models
cached = ModelCache.list()

# Clear cache
ModelCache.clear()

# Get cache size
size_mb = ModelCache.size()
```

---

## Indices Module

::: unbihexium.indices

### Spectral Indices

```python
from unbihexium.indices import (
    ndvi,
    ndwi,
    evi,
    savi,
    nbr,
)

# Normalized Difference Vegetation Index
vegetation = ndvi(nir, red)

# Normalized Difference Water Index
water = ndwi(green, nir)

# Enhanced Vegetation Index
enhanced = evi(nir, red, blue)

# Soil Adjusted Vegetation Index
soil_adjusted = savi(nir, red, L=0.5)

# Normalized Burn Ratio
burn = nbr(nir, swir)
```
