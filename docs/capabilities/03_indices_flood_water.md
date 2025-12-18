# 03 - Indices, Flood and Water

## Purpose

Spectral index calculation and water/flood management capabilities.

## Audience

Remote sensing analysts, hydrologists, environmental scientists.

## Prerequisites

- Python 3.10+
- Multispectral imagery with appropriate bands
- DEM for watershed analysis

## Inputs/Outputs

| Input | Format | Output | Format |
|-------|--------|--------|--------|
| Multispectral bands | GeoTIFF | Index rasters | GeoTIFF |
| DEM | GeoTIFF | Watershed delineation | GeoJSON |
| Time series | Zarr | Trend analysis | JSON |

## Pipeline Architecture

```mermaid
flowchart TB
    subgraph Indices
        NDVI[NDVI]
        NBR[NBR]
        EVI[EVI]
        MSI[MSI]
        NDWI[NDWI]
        SAVI[SAVI]
    end

    subgraph Water Management
        WS[Watershed Management]
        FR[Flood Risk]
        WQ[Water Quality]
        WD[Water Detection]
        RM[Reservoir Monitoring]
        TS[Timeseries Analysis]
    end

    BANDS[Spectral Bands] --> Indices
    DEM[DEM] --> Water Management
    Indices --> WD
```

## Algorithms

### NDVI (Normalized Difference Vegetation Index)

$$NDVI = \frac{NIR - RED}{NIR + RED}$$

### NBR (Normalized Burn Ratio)

$$NBR = \frac{NIR - SWIR2}{NIR + SWIR2}$$

### EVI (Enhanced Vegetation Index)

$$EVI = G \times \frac{NIR - RED}{NIR + C_1 \times RED - C_2 \times BLUE + L}$$

Where: $G = 2.5$, $C_1 = 6$, $C_2 = 7.5$, $L = 1$

### NDWI (Normalized Difference Water Index)

$$NDWI = \frac{GREEN - NIR}{GREEN + NIR}$$

### SAVI (Soil-Adjusted Vegetation Index)

$$SAVI = \frac{(NIR - RED)}{(NIR + RED + L)} \times (1 + L)$$

### MSI (Moisture Stress Index)

$$MSI = \frac{SWIR1}{NIR}$$

## Metrics

| Index | Range | Water Threshold | Vegetation Threshold |
|-------|-------|-----------------|----------------------|
| NDVI | -1 to 1 | < 0.1 | > 0.3 |
| NDWI | -1 to 1 | > 0.3 | < 0 |
| NBR | -1 to 1 | N/A | N/A |

## Mandatory Mapping Table

| Bullet Item | capability_id | Module Path | Pipeline ID | CLI Example | Example Script | Test Path | Model ID(s) | Maturity |
|-------------|---------------|-------------|-------------|-------------|----------------|-----------|-------------|----------|
| NDVI | ndvi | `unbihexium.core.index.compute_index` | ndvi | `unbihexium index NDVI -i input.tif -o ndvi.tif` | `examples/indices.py` | `tests/unit/test_core.py` | ndvi_calculator_tiny, ndvi_calculator_base, ndvi_calculator_large | production |
| NBR | nbr | `unbihexium.core.index.compute_index` | nbr | `unbihexium index NBR -i input.tif -o nbr.tif` | `examples/indices.py` | `tests/unit/test_core.py` | nbr_calculator_tiny, nbr_calculator_base, nbr_calculator_large | production |
| EVI | evi | `unbihexium.core.index.compute_index` | evi | `unbihexium index EVI -i input.tif -o evi.tif` | `examples/indices.py` | `tests/unit/test_core.py` | evi_calculator_tiny, evi_calculator_base, evi_calculator_large | production |
| MSI | msi | `unbihexium.core.index.compute_index` | msi | `unbihexium index MSI -i input.tif -o msi.tif` | `examples/indices.py` | `tests/unit/test_core.py` | msi_calculator_tiny, msi_calculator_base, msi_calculator_large | production |
| NDWI | ndwi | `unbihexium.core.index.compute_index` | ndwi | `unbihexium index NDWI -i input.tif -o ndwi.tif` | `examples/indices.py` | `tests/unit/test_core.py` | ndwi_calculator_tiny, ndwi_calculator_base, ndwi_calculator_large | production |
| SAVI | savi | `unbihexium.core.index.compute_index` | savi | `unbihexium index SAVI -i input.tif -o savi.tif` | `examples/indices.py` | `tests/unit/test_core.py` | savi_calculator_tiny, savi_calculator_base, savi_calculator_large | production |
| River/watershed management | watershed | `unbihexium.analysis.watershed` | watershed | `unbihexium pipeline run watershed -i dem.tif -o basins.geojson` | `examples/watershed.py` | `tests/unit/test_analysis.py` | watershed_manager_tiny, watershed_manager_base, watershed_manager_large | production |
| Flood risk assessment | flood_risk | `unbihexium.analysis.suitability.FloodRisk` | flood_risk | `unbihexium pipeline run flood_risk -i dem.tif -i rainfall.tif -o risk.tif` | `examples/flood_risk.py` | `tests/unit/test_analysis.py` | flood_risk_assessor_tiny, flood_risk_assessor_base, flood_risk_assessor_large | production |
| Water quality assessment | water_quality | `unbihexium.analysis.water.WaterQuality` | water_qual | `unbihexium pipeline run water_qual -i multispectral.tif -o quality.tif` | `examples/water_quality.py` | `tests/unit/test_analysis.py` | water_quality_assessor_tiny, water_quality_assessor_base, water_quality_assessor_large | production |
| Water surface detection | water_surface | `unbihexium.ai.segmentation.WaterDetector` | water_detect | `unbihexium pipeline run water_detect -i input.tif -o water.tif` | `examples/water_detection.py` | `tests/unit/test_ai.py` | water_surface_detector_tiny, water_surface_detector_base, water_surface_detector_large | production |
| Reservoir/water bodies monitoring | reservoir_monitor | `unbihexium.analysis.water.ReservoirMonitor` | reservoir | `unbihexium pipeline run reservoir -i timeseries/ -o level.json` | `examples/reservoir.py` | `tests/unit/test_analysis.py` | reservoir_monitor_tiny, reservoir_monitor_base, reservoir_monitor_large | production |
| Timeseries mapping and analysis | timeseries | `unbihexium.analysis.timeseries` | ts_analysis | `unbihexium pipeline run ts_analysis -i stack.zarr -o trends.json` | `examples/timeseries.py` | `tests/unit/test_analysis.py` | timeseries_analyzer_tiny, timeseries_analyzer_base, timeseries_analyzer_large | production |

## Limitations

- Index accuracy depends on atmospheric correction
- Water quality estimation requires calibration data
- Flood risk models require validation with historical events

## Examples (CLI)

```bash
# Calculate NDVI
unbihexium index NDVI -i sentinel2.tif -o ndvi.tif --red B04 --nir B08

# Flood risk assessment
unbihexium pipeline run flood_risk -i dem.tif -i rainfall.tif -o flood_risk.tif

# Watershed delineation
unbihexium pipeline run watershed -i dem.tif -o watersheds.geojson
```

## API Entry Points

```python
from unbihexium.core.index import compute_index, IndexRegistry
from unbihexium.analysis.water import WaterQuality, ReservoirMonitor
from unbihexium.ai.segmentation import WaterDetector
```

## Tests

- Unit tests: `tests/unit/test_core.py`
- Index tests: `tests/unit/test_indices.py`

## Models

Index calculators and water detection models in all tiers.

## References

- [Documentation Index](../index.md)
- [Table of Contents](../toc.md)
- [Indices Tutorial](../tutorials/indices.md)
