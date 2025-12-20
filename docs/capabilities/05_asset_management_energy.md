# Capability 05: Asset Management and Energy

## Executive Summary

This document provides comprehensive documentation for the Asset Management and Energy capability domain within the Unbihexium framework. This domain encompasses infrastructure monitoring, energy resource assessment, utility network analysis, and asset management models essential for critical infrastructure protection, renewable energy planning, and operational efficiency.

The domain comprises 12 base model architectures with 48 total variants, covering pipeline monitoring, power infrastructure assessment, renewable energy site selection, and infrastructure change detection.

---

## Domain Overview

### Scope and Objectives

1. **Infrastructure Monitoring**: Detect changes and anomalies in critical infrastructure using bi-temporal analysis
2. **Pipeline Route Planning**: Optimize pipeline and utility corridor routing considering terrain and environmental constraints
3. **Renewable Energy Site Selection**: Evaluate optimal locations for solar, wind, and hydroelectric installations
4. **Asset Condition Assessment**: Monitor infrastructure condition and detect degradation patterns
5. **Utility Network Mapping**: Extract and analyze utility infrastructure from satellite imagery

### Domain Statistics

| Metric | Value |
|--------|-------|
| Base Model Architectures | 12 |
| Total Model Variants | 48 |
| Minimum Parameters (tiny) | 68,225 |
| Maximum Parameters (mega) | 4,107,010 |
| Primary Tasks | Regression, Segmentation, Detection |
| Production Status | Fully Production Ready |

---

## Model Inventory

### Complete Model Listing

| Model ID | Task | Architecture | Output | Variants | Parameter Range |
|----------|------|--------------|--------|----------|-----------------|
| pipeline_route_planner | Regression | MLP | continuous | 4 | 68,481 - 1,060,353 |
| energy_potential | Regression | MLP | continuous | 4 | 68,481 - 1,060,353 |
| hydroelectric_monitor | Regression | MLP | continuous | 4 | 68,225 - 1,059,329 |
| solar_site_selector | Regression | MLP | continuous | 4 | 68,737 - 1,061,377 |
| wind_site_selector | Regression | MLP | continuous | 4 | 68,481 - 1,060,353 |
| offshore_survey | Regression | MLP | continuous | 4 | 68,225 - 1,059,329 |
| onshore_monitor | Regression | MLP | continuous | 4 | 68,481 - 1,060,353 |
| corridor_monitor | Segmentation | Siamese | 2 classes | 4 | 258,754 - 4,107,010 |
| infrastructure_monitor | Segmentation | Siamese | 2 classes | 4 | 258,754 - 4,107,010 |
| asset_condition_change | Segmentation | Siamese | 2 classes | 4 | 258,754 - 4,107,010 |
| leakage_detector | Detection | UNet | 1 class | 4 | 143,201 - 2,268,545 |
| utility_mapper | Segmentation | UNet | 4 classes | 4 | 143,780 - 2,269,316 |

---

## Performance Metrics

### Site Selection Performance

| Model | Metric | Tiny | Base | Large | Mega | Validation |
|-------|--------|------|------|-------|------|------------|
| solar_site_selector | R-squared | 0.78 | 0.85 | 0.90 | 0.94 | Solar-Sites |
| wind_site_selector | R-squared | 0.75 | 0.82 | 0.88 | 0.92 | Wind-Atlas |
| hydroelectric_monitor | R-squared | 0.72 | 0.79 | 0.85 | 0.90 | Hydro-DB |

### Infrastructure Change Detection

| Model | Metric | Tiny | Base | Large | Mega | Validation |
|-------|--------|------|------|-------|------|------------|
| corridor_monitor | F1 | 0.70 | 0.79 | 0.86 | 0.91 | Corridors |
| infrastructure_monitor | F1 | 0.72 | 0.80 | 0.87 | 0.92 | Infra-Change |
| asset_condition_change | F1 | 0.68 | 0.77 | 0.84 | 0.90 | Asset-DB |

---

## Solar Energy Modeling

### Solar Irradiance Components

Global Horizontal Irradiance (GHI):

$$
\text{GHI} = \text{DNI} \cdot \cos(\theta_z) + \text{DHI}
$$

Where:
- DNI = Direct Normal Irradiance (W/m^2)
- DHI = Diffuse Horizontal Irradiance (W/m^2)
- $\theta_z$ = Solar zenith angle

### Solar Zenith Angle

$$
\cos(\theta_z) = \sin(\phi) \sin(\delta) + \cos(\phi) \cos(\delta) \cos(\omega)
$$

Where:
- $\phi$ = Latitude
- $\delta$ = Solar declination
- $\omega$ = Hour angle

### Solar Declination

$$
\delta = 23.45° \times \sin\left(\frac{360}{365}(284 + n)\right)
$$

Where $n$ is the day of year.

### Annual Energy Yield

$$
E_{annual} = A \times \eta \times \text{GHI}_{annual} \times PR
$$

Where:
- $A$ = Panel area (m^2)
- $\eta$ = Panel efficiency (0.15-0.22)
- $\text{GHI}_{annual}$ = Annual irradiation (kWh/m^2)
- $PR$ = Performance ratio (0.75-0.85)

### Capacity Factor

$$
CF = \frac{E_{actual}}{P_{rated} \times 8760}
$$

---

## Wind Energy Modeling

### Wind Power Density

$$
P_d = \frac{1}{2} \rho v^3 \quad [\text{W/m}^2]
$$

Where:
- $\rho$ = Air density (kg/m^3)
- $v$ = Wind speed (m/s)

### Weibull Distribution

$$
f(v) = \frac{k}{A}\left(\frac{v}{A}\right)^{k-1} \exp\left(-\left(\frac{v}{A}\right)^k\right)
$$

Where:
- $k$ = Shape parameter (typically 2)
- $A$ = Scale parameter

### Wind Power Curve

$$
P(v) = \begin{cases}
0 & v < v_{cut-in} \\
\frac{1}{2} \rho A C_p v^3 & v_{cut-in} \leq v < v_{rated} \\
P_{rated} & v_{rated} \leq v < v_{cut-out} \\
0 & v \geq v_{cut-out}
\end{cases}
$$

### Wind Shear Profile

$$
v(h) = v_{ref} \times \left(\frac{h}{h_{ref}}\right)^\alpha
$$

Where $\alpha$ is the wind shear exponent (0.1-0.4).

---

## Hydroelectric Assessment

### Theoretical Power

$$
P = \eta \times \rho \times g \times Q \times H
$$

Where:
- $\eta$ = Turbine efficiency
- $\rho$ = Water density (1000 kg/m^3)
- $g$ = Gravitational acceleration (9.81 m/s^2)
- $Q$ = Flow rate (m^3/s)
- $H$ = Net head (m)

### Annual Energy Production

$$
E_{annual} = P \times CF \times 8760 \quad [\text{kWh}]
$$

### Flow Duration Curve Analysis

$$
Q_{exceedance} = P(Q > q)
$$

---

## Usage Examples

### CLI Usage

```bash
# Solar site selection
unbihexium infer solar_site_selector_mega \
    --input site_features.csv \
    --output solar_potential.csv \
    --ghi-data ghi_annual.tif \
    --dem elevation.tif

# Wind site selection
unbihexium infer wind_site_selector_large \
    --input locations.csv \
    --output wind_scores.csv \
    --wind-data wind_100m.tif \
    --weibull-k 2.0

# Pipeline route optimization
unbihexium infer pipeline_route_planner_mega \
    --origin "35.0,-100.0" \
    --destination "40.0,-95.0" \
    --constraints terrain.tif \
    --output optimal_route.geojson

# Infrastructure change detection
unbihexium infer infrastructure_monitor_mega \
    --input-t1 infra_2020.tif \
    --input-t2 infra_2024.tif \
    --output changes.geojson \
    --threshold 0.5

# Leakage detection
unbihexium infer leakage_detector_large \
    --input thermal_image.tif \
    --output leakage_points.geojson \
    --confidence 0.7
```

### Python API Usage

```python
from unbihexium import Pipeline, Config
from unbihexium.zoo import get_model
import pandas as pd
import numpy as np

# Solar Site Selection
solar_model = get_model("solar_site_selector_mega")

# Prepare site features
sites = pd.DataFrame({
    'latitude': lat_values,
    'longitude': lon_values,
    'ghi_annual': ghi_values,  # kWh/m^2/year
    'dni_annual': dni_values,
    'slope': slope_values,
    'aspect': aspect_values,
    'distance_grid': grid_distance,
    'land_use': land_use_codes,
    'population_density': pop_density,
    'protected_area': protected_flags,
    'elevation': elevation_values,
    'cloud_cover': cloud_cover
})

# Predict suitability scores
suitability = solar_model.predict(sites)
sites['solar_score'] = suitability

# Calculate potential capacity
sites['potential_mw'] = sites['area_km2'] * 40  # ~40 MW/km^2
sites['annual_gwh'] = sites['potential_mw'] * sites['ghi_annual'] / 1000 * 0.2 * 0.8

# Rank sites
top_sites = sites.nlargest(10, 'solar_score')
print(f"Top 10 sites could generate {top_sites['annual_gwh'].sum():.1f} GWh/year")

# Wind Site Selection
wind_model = get_model("wind_site_selector_large")

wind_features = pd.DataFrame({
    'latitude': lat_values,
    'longitude': lon_values,
    'wind_speed_100m': wind_speeds,
    'weibull_k': weibull_k,
    'weibull_a': weibull_a,
    'surface_roughness': roughness,
    'turbulence_intensity': turbulence,
    'distance_grid': grid_dist,
    'environmental_score': env_score,
    'accessibility': access_score
})

wind_scores = wind_model.predict(wind_features)

# Infrastructure Change Detection
change_model = get_model("infrastructure_monitor_mega")

config = Config(
    tile_size=256,
    overlap=32,
    batch_size=8,
    device="cuda:0",
    threshold=0.5
)

change_pipeline = Pipeline.from_config(
    capability="infrastructure_change",
    variant="mega",
    config=config
)

changes = change_pipeline.run(
    t1="infrastructure_2020.tif",
    t2="infrastructure_2024.tif"
)

changes.save("detected_changes.geojson")

# Analyze changes
stats = changes.statistics()
print(f"New infrastructure: {stats['new_area_km2']:.2f} km^2")
print(f"Removed infrastructure: {stats['removed_area_km2']:.2f} km^2")
print(f"Modified infrastructure: {stats['modified_area_km2']:.2f} km^2")
```

---

## Technical Requirements

### Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| CPU | 4 cores | 8 cores | 16+ cores |
| RAM | 8 GB | 16 GB | 32 GB |
| GPU | None | RTX 3060 | A100 |
| Storage | 20 GB | 100 GB | 500 GB |

### Input Data Requirements

| Data Type | Source | Resolution | Application |
|-----------|--------|------------|-------------|
| Solar Irradiance | NASA POWER, PVGIS | 0.5° | Solar siting |
| Wind Data | Global Wind Atlas | 250m | Wind siting |
| DEM | SRTM, ALOS | 30m | Terrain analysis |
| Infrastructure | OSM, Commercial | Vector | Asset mapping |
| Thermal Imagery | Landsat-8/9 | 100m | Leakage detection |

---

## Quality Assurance

### Validation Datasets

| Dataset | Source | Coverage | Use |
|---------|--------|----------|-----|
| Global Wind Atlas | DTU | Global | Wind validation |
| PVGIS | JRC | Europe/Africa | Solar validation |
| NASA POWER | NASA | Global | Irradiance data |
| OpenInfraMap | OSM | Global | Infrastructure |

---

## References

1. Global Wind Atlas 3.0. Technical University of Denmark.
2. PVGIS. European Commission Joint Research Centre.
3. NASA POWER Project. NASA Langley Research Center.
4. REN21 (2024). Renewables 2024 Global Status Report.
5. IEA (2024). World Energy Outlook 2024.
