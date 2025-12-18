# 02 - Tourism and Data Processing

## Purpose

Geospatial analytics for tourism planning, route optimization, and comprehensive data processing capabilities.

## Audience

Tourism planners, GIS analysts, data scientists, municipal planners.

## Prerequisites

- Python 3.10+
- Network data (roads, paths)
- Suitability criteria layers

## Inputs/Outputs

| Input | Format | Output | Format |
|-------|--------|--------|--------|
| Road network | GeoJSON, Shapefile | Optimized routes | GeoJSON |
| Criteria layers | GeoTIFF | Suitability maps | GeoTIFF |
| Point data | CSV, GeoJSON | Statistics | JSON |

## Pipeline Architecture

```mermaid
flowchart TB
    subgraph Tourism
        RP[Route Planning]
        SS[Site Selection]
        TM[Destination Monitoring]
        IM[Interactive Mapping]
        NA[Navigation/Accessibility]
    end

    subgraph Data Processing
        DM[Data Management]
        IA[Image Analysis]
        SA[Spatial Analysis]
        GA[Geostatistical Analysis]
        SUA[Suitability Analysis]
        NET[Network Analysis]
        BV[Business Valuation]
        RT[Raster Tiling]
        ZS[Zonal Statistics]
    end

    DATA[Input Data] --> Tourism
    DATA --> Data Processing
```

## Algorithms

### Dijkstra Shortest Path

$$d(v) = \min_{u \in N(v)} \{d(u) + w(u,v)\}$$

### Accessibility Score

$$A_i = \sum_{j=1}^{n} O_j \cdot f(c_{ij})$$

Where $O_j$ is opportunity at $j$ and $f(c_{ij})$ is impedance function.

### AHP Consistency Ratio

$$CR = \frac{CI}{RI}$$

Where $CI = \frac{\lambda_{max} - n}{n - 1}$

## Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| Travel time | Route duration | minutes |
| Accessibility index | Reachability score | 0-1 |
| Suitability score | Multi-criteria result | 0-1 |

## Mandatory Mapping Table

| Bullet Item | capability_id | Module Path | Pipeline ID | CLI Example | Example Script | Test Path | Model ID(s) | Maturity |
|-------------|---------------|-------------|-------------|-------------|----------------|-----------|-------------|----------|
| Route planning | route_planning | `unbihexium.analysis.network.route_planning` | route_plan | `unbihexium pipeline run route_plan -i network.geojson -o route.geojson` | `examples/route_planning.py` | `tests/unit/test_analysis.py` | route_planner_tiny, route_planner_base, route_planner_large | production |
| Suitable site finding | site_suitability | `unbihexium.analysis.suitability.SiteFinder` | site_find | `unbihexium pipeline run site_find -i layers.yaml -o sites.geojson` | `examples/site_finding.py` | `tests/unit/test_analysis.py` | site_suitability_tiny, site_suitability_base, site_suitability_large | production |
| Monitoring/mapping tourist destinations | tourist_monitoring | `unbihexium.analysis.monitoring` | tourist_mon | `unbihexium pipeline run tourist_mon -i dest.geojson -o report.json` | `examples/tourist_monitoring.py` | `tests/unit/test_analysis.py` | tourist_destination_monitor_tiny, tourist_destination_monitor_base, tourist_destination_monitor_large | production |
| Interactive mapping | interactive_mapping | `unbihexium.analysis.mapping` | int_map | `unbihexium pipeline run int_map -i data.geojson -o map.html` | `examples/interactive_map.py` | `tests/unit/test_analysis.py` | classical/no-weights | production |
| Navigation and accessibility | accessibility | `unbihexium.analysis.network.accessibility` | access | `unbihexium pipeline run access -i network.geojson -o access.tif` | `examples/accessibility.py` | `tests/unit/test_analysis.py` | accessibility_analyzer_tiny, accessibility_analyzer_base, accessibility_analyzer_large | production |
| Data management | data_management | `unbihexium.io` | data_mgmt | `unbihexium data manage -i input/ -o output/` | `examples/data_management.py` | `tests/unit/test_io.py` | classical/no-weights | production |
| Image analysis | image_analysis | `unbihexium.core.raster` | img_analysis | `unbihexium pipeline run img_analysis -i image.tif -o stats.json` | `examples/image_analysis.py` | `tests/unit/test_core.py` | classical/no-weights | production |
| Spatial analyses | spatial_analysis | `unbihexium.analysis.zonal` | spatial | `unbihexium pipeline run spatial -i data.geojson -o result.geojson` | `examples/spatial_analysis.py` | `tests/unit/test_analysis.py` | spatial_analyzer_tiny, spatial_analyzer_base, spatial_analyzer_large | production |
| Statistical and geostatistical analyses | geostat_analysis | `unbihexium.geostat` | geostat | `unbihexium pipeline run geostat -i points.csv -o kriging.tif` | `examples/geostat.py` | `tests/unit/test_geostat.py` | geostatistical_analyzer_tiny, geostatistical_analyzer_base, geostatistical_analyzer_large | production |
| Suitability analysis | suitability | `unbihexium.analysis.suitability.AHP` | suitability | `unbihexium pipeline run suitability -i criteria.yaml -o suitable.tif` | `examples/suitability.py` | `tests/unit/test_analysis.py` | site_suitability_tiny, site_suitability_base, site_suitability_large | production |
| Network analysis | network_analysis | `unbihexium.analysis.network` | network | `unbihexium pipeline run network -i roads.geojson -o analysis.json` | `examples/network.py` | `tests/unit/test_analysis.py` | network_analyzer_tiny, network_analyzer_base, network_analyzer_large | production |
| Business valuation analysis | business_valuation | `unbihexium.analysis.suitability.ValuationAnalyzer` | biz_val | `unbihexium pipeline run biz_val -i parcels.geojson -o valuation.json` | `examples/valuation.py` | `tests/unit/test_analysis.py` | business_valuation_tiny, business_valuation_base, business_valuation_large | production |
| Raster tiling | raster_tiling | `unbihexium.core.tile.TileGrid` | tile | `unbihexium pipeline run tile -i large.tif -o tiles/` | `examples/tiling.py` | `tests/unit/test_core.py` | raster_tiler_tiny, raster_tiler_base, raster_tiler_large | production |
| Zonal statistics | zonal_stats | `unbihexium.analysis.zonal.zonal_statistics` | zonal | `unbihexium pipeline run zonal -i raster.tif -i zones.geojson -o stats.json` | `examples/zonal_stats.py` | `tests/unit/test_analysis.py` | zonal_statistics_tiny, zonal_statistics_base, zonal_statistics_large | production |

## Limitations

- Network analysis requires properly topologized network data
- AHP requires expert-defined pairwise comparisons
- Large rasters may require chunked processing

## Examples (CLI)

```bash
# Route planning
unbihexium pipeline run route_plan -i roads.geojson --origin "10.0,45.0" --dest "11.0,46.0" -o route.geojson

# Suitability analysis
unbihexium pipeline run suitability -i criteria.yaml -o suitability.tif

# Zonal statistics
unbihexium pipeline run zonal -i ndvi.tif -i parcels.geojson -o stats.json
```

## API Entry Points

```python
from unbihexium.analysis.network import NetworkAnalyzer, route_planning
from unbihexium.analysis.suitability import AHP, SiteFinder
from unbihexium.analysis.zonal import zonal_statistics
from unbihexium.geostat import Variogram, OrdinaryKriging
```

## Tests

- Unit tests: `tests/unit/test_analysis.py`
- Geostat tests: `tests/unit/test_geostat.py`

## Models

Classical algorithms and ML-based models available in all tiers.

## References

- [Documentation Index](../index.md)
- [Table of Contents](../toc.md)
- [Geostatistics Tutorial](../tutorials/geostat.md)

---

## Quick Navigation

| Prev | Home | Next |
|------|------|------|

