#!/usr/bin/env python3
"""Generate complete inventory.yaml for model zoo."""

from pathlib import Path

# Model domains mapping
domains = {
    # 1) AI PRODUCTS
    "super_resolution": ("ai", "super_resolution"),
    "synthetic_imagery": ("ai", "image_generation"),
    "ship_detector": ("ai", "detection"),
    "building_detector": ("ai", "detection"),
    "aircraft_detector": ("ai", "detection"),
    "change_detector": ("ai", "change_detection"),
    "vehicle_detector": ("ai", "detection"),
    "military_objects_detector": ("ai", "detection"),
    "multi_solution_segmentation": ("ai", "segmentation"),
    "greenhouse_detector": ("ai", "detection"),
    "crop_detector": ("ai", "detection"),
    "crop_boundary_delineation": ("ai", "segmentation"),
    "water_surface_detector": ("ai", "segmentation"),
    # 2) TOURISM / DATA PROCESSING
    "route_planner": ("tourism", "planning"),
    "site_suitability": ("analysis", "suitability"),
    "tourist_destination_monitor": ("tourism", "monitoring"),
    "accessibility_analyzer": ("analysis", "accessibility"),
    "spatial_analyzer": ("analysis", "spatial"),
    "geostatistical_analyzer": ("geostat", "analysis"),
    "network_analyzer": ("analysis", "network"),
    "business_valuation": ("analysis", "valuation"),
    "raster_tiler": ("processing", "tiling"),
    "zonal_statistics": ("analysis", "zonal"),
    # 3) VEGETATION & INDICES / FLOOD & WATER
    "ndvi_calculator": ("indices", "vegetation"),
    "nbr_calculator": ("indices", "burn"),
    "evi_calculator": ("indices", "vegetation"),
    "msi_calculator": ("indices", "moisture"),
    "ndwi_calculator": ("indices", "water"),
    "savi_calculator": ("indices", "vegetation"),
    "watershed_manager": ("water", "watershed"),
    "flood_risk_assessor": ("water", "flood"),
    "water_quality_assessor": ("water", "quality"),
    "reservoir_monitor": ("water", "monitoring"),
    "timeseries_analyzer": ("analysis", "timeseries"),
    # 4) ENVIRONMENT & FORESTRY / IMAGE PROCESSING
    "environmental_monitor": ("environment", "monitoring"),
    "protected_area_change_detector": ("environment", "change_detection"),
    "wildlife_habitat_analyzer": ("environment", "habitat"),
    "emergency_disaster_manager": ("risk", "disaster"),
    "forest_monitor": ("forestry", "monitoring"),
    "deforestation_detector": ("forestry", "change_detection"),
    "forest_density_estimator": ("forestry", "regression"),
    "tree_height_estimator": ("forestry", "regression"),
    "thematic_mapper": ("mapping", "thematic"),
    "desertification_monitor": ("environment", "monitoring"),
    "drought_monitor": ("environment", "monitoring"),
    "land_degradation_detector": ("environment", "detection"),
    "natural_resources_monitor": ("environment", "monitoring"),
    "marine_pollution_detector": ("environment", "detection"),
    "lulc_classifier": ("classification", "landcover"),
    "salinity_detector": ("environment", "detection"),
    "erosion_detector": ("environment", "detection"),
    "pansharpening": ("processing", "enhancement"),
    "coregistration": ("processing", "alignment"),
    "cloud_mask": ("processing", "masking"),
    "dtm_generator": ("processing", "terrain"),
    "dsm_generator": ("processing", "surface"),
    "orthorectification": ("processing", "geometric"),
    "mosaicking": ("processing", "mosaic"),
    # 5) GEOSPATIAL ASSET MANAGEMENT / ENERGY
    "object_detector": ("asset", "detection"),
    "damage_assessor": ("asset", "assessment"),
    "asset_condition_change": ("asset", "change_detection"),
    "economic_spatial_assessor": ("asset", "valuation"),
    "utility_mapper": ("asset", "mapping"),
    "spatial_relationship": ("analysis", "spatial"),
    "wind_site_selector": ("energy", "suitability"),
    "solar_site_selector": ("energy", "suitability"),
    "land_surface_temperature": ("energy", "thermal"),
    "energy_potential": ("energy", "assessment"),
    "hydroelectric_monitor": ("energy", "monitoring"),
    "pipeline_route_planner": ("energy", "planning"),
    "corridor_monitor": ("energy", "monitoring"),
    "encroachment_detector": ("energy", "detection"),
    "offshore_survey": ("energy", "survey"),
    "ground_displacement": ("sar", "displacement"),
    "onshore_monitor": ("energy", "monitoring"),
    "leakage_detector": ("energy", "detection"),
    # 6) URBAN / AGRICULTURE
    "urban_planner": ("urban", "planning"),
    "urban_growth_assessor": ("urban", "assessment"),
    "resource_allocation": ("urban", "optimization"),
    "infrastructure_monitor": ("urban", "monitoring"),
    "construction_monitor": ("urban", "monitoring"),
    "builtup_detector": ("urban", "detection"),
    "road_network_analyzer": ("urban", "network"),
    "transportation_mapper": ("urban", "mapping"),
    "model_3d": ("urban", "modeling"),
    "topography_mapper": ("mapping", "topography"),
    "crop_classifier": ("agriculture", "classification"),
    "plowed_land_detector": ("agriculture", "detection"),
    "yield_predictor": ("agriculture", "regression"),
    "crop_growth_monitor": ("agriculture", "monitoring"),
    "vegetation_condition": ("agriculture", "assessment"),
    "crop_health_assessor": ("agriculture", "assessment"),
    "grazing_potential": ("agriculture", "suitability"),
    "perennial_garden_suitability": ("agriculture", "suitability"),
    "pivot_inventory": ("agriculture", "inventory"),
    "beekeeping_suitability": ("agriculture", "suitability"),
    "livestock_estimator": ("agriculture", "regression"),
    "field_surveyor": ("agriculture", "survey"),
    # 7) RISK / DEFENSE-INTELLIGENCE
    "hazard_vulnerability": ("risk", "assessment"),
    "insurance_underwriting": ("risk", "assessment"),
    "disaster_management": ("risk", "planning"),
    "environmental_risk": ("risk", "assessment"),
    "flood_risk": ("risk", "assessment"),
    "seismic_risk": ("risk", "assessment"),
    "landslide_risk": ("risk", "assessment"),
    "wildfire_risk": ("risk", "assessment"),
    "fire_monitor": ("risk", "monitoring"),
    "security_monitor": ("defense", "monitoring"),
    "target_detector": ("defense", "detection"),
    "mobility_analyzer": ("defense", "analysis"),
    "viewshed_analyzer": ("defense", "analysis"),
    "border_monitor": ("defense", "monitoring"),
    "maritime_awareness": ("defense", "awareness"),
    "preparedness_manager": ("defense", "planning"),
    # 8) VALUE-ADDED IMAGERY
    "dem_generator": ("processing", "terrain"),
    "ortho_processor": ("processing", "geometric"),
    "mosaic_processor": ("processing", "mosaic"),
    # 9-10) SATELLITE IMAGERY
    "stereo_processor": ("processing", "stereo"),
    "tri_stereo_processor": ("processing", "stereo"),
    "digitization_2d": ("processing", "digitization"),
    "digitization_3d": ("processing", "digitization"),
    "panchromatic_processor": ("processing", "spectral"),
    "multispectral_processor": ("processing", "spectral"),
    # 11) SAR
    "sar_amplitude": ("sar", "amplitude"),
    "sar_phase_displacement": ("sar", "interferometry"),
    "sar_ship_detector": ("sar", "detection"),
    "sar_oil_spill_detector": ("sar", "detection"),
    "sar_flood_detector": ("sar", "detection"),
    "sar_subsidence_monitor": ("sar", "monitoring"),
    "sar_mapping_workflow": ("sar", "mapping"),
}

yaml_content = """# Model Zoo Inventory
# Complete catalog of ALL models covering 12 major capability domains
# Total: 390 models (130 per tier)

schema_version: "1.0"
total_models: 390

domains:
  ai: AI Products (detection, segmentation, super-resolution)
  tourism: Tourism analytics and route planning
  analysis: Spatial and statistical analysis
  indices: Vegetation and environmental indices
  water: Flood and water management
  environment: Environmental monitoring
  forestry: Forest and vegetation monitoring
  processing: Image processing primitives
  asset: Geospatial asset management
  energy: Energy infrastructure and site selection
  urban: Urban planning and monitoring
  agriculture: Agricultural analysis
  risk: Risk assessment and disaster management
  defense: Security and surveillance analytics (neutral)
  sar: SAR/Radar imaging and analysis
  mapping: Thematic and topographic mapping
  geostat: Geostatistical analysis
  classification: Land cover and scene classification

tiers:
  tiny:
    description: Smoke test models for CI/CD
    source: repo
    path: assets/tiny/
  base:
    description: Production-ready models
    source: repo
    path: assets/base/
  large:
    description: High-accuracy models
    source: repo
    path: assets/large/

models:
"""

for model_base, (domain, task) in sorted(domains.items()):
    yaml_content += f"""
  - id: {model_base}
    domain: {domain}
    task: {task}
    tiers: [tiny, base, large]
    license: Apache-2.0
    version: "1.0.0"
"""

with open("model_zoo/inventory.yaml", "w") as f:
    f.write(yaml_content)

print(f"Created inventory.yaml with {len(domains)} model definitions")
