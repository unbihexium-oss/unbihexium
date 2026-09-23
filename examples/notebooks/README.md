# Unbihexium Notebooks

Production-grade Jupyter notebooks demonstrating the Unbihexium Earth Observation, Geospatial AI, Remote Sensing, and SAR library.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unbihexium-oss/unbihexium/blob/main/examples/notebooks/)

## Overview

This directory contains **130 comprehensive notebooks** covering all model topics in the Unbihexium Model Zoo. Each notebook demonstrates:

- All 4 model variants (tiny, base, large, mega)
- Both ONNX and PyTorch inference
- Performance benchmarking
- Integration examples
- Best practices

## Notebook Index

### Detection Models

| # | Notebook | Description |
| --- | ---------- | ------------- |
| 002 | [aircraft_detector](./002_aircraft_detector.ipynb) | Detection and classification of aircraft from satellite and aerial imagery |
| 005 | [border_monitor](./005_border_monitor.ipynb) | Border surveillance and perimeter monitoring from satellite imagery |
| 006 | [building_detector](./006_building_detector.ipynb) | Automatic building footprint detection and extraction |
| 013 | [corridor_monitor](./013_corridor_monitor.ipynb) | Linear infrastructure corridor monitoring |
| 016 | [crop_detector](./016_crop_detector.ipynb) | Crop presence detection and mapping |
| 038 | [fire_monitor](./038_fire_monitor.ipynb) | Active fire detection and burn area mapping |
| 045 | [greenhouse_detector](./045_greenhouse_detector.ipynb) | Greenhouse and plastic cover detection |
| 049 | [infrastructure_monitor](./049_infrastructure_monitor.ipynb) | Critical infrastructure monitoring and assessment |
| 054 | [leakage_detector](./054_leakage_detector.ipynb) | Pipeline and infrastructure leakage detection |
| 057 | [marine_pollution_detector](./057_marine_pollution_detector.ipynb) | Marine pollution and oil spill detection |
| 058 | [maritime_awareness](./058_maritime_awareness.ipynb) | Maritime domain awareness and vessel monitoring |
| 059 | [military_objects_detector](./059_military_objects_detector.ipynb) | Military infrastructure and vehicle detection |
| 072 | [object_detector](./072_object_detector.ipynb) | General-purpose object detection |
| 073 | [offshore_survey](./073_offshore_survey.ipynb) | Offshore infrastructure and marine survey |
| 074 | [onshore_monitor](./074_onshore_monitor.ipynb) | Onshore oil and gas facility monitoring |
| 081 | [pivot_inventory](./081_pivot_inventory.ipynb) | Center pivot irrigation inventory and mapping |
| 082 | [plowed_land_detector](./082_plowed_land_detector.ipynb) | Plowed and tilled land detection |
| 094 | [sar_oil_spill_detector](./094_sar_oil_spill_detector.ipynb) | Oil spill detection from SAR imagery |
| 096 | [sar_ship_detector](./096_sar_ship_detector.ipynb) | Ship detection from SAR imagery |
| 099 | [security_monitor](./099_security_monitor.ipynb) | Security and surveillance monitoring |
| 101 | [ship_detector](./101_ship_detector.ipynb) | Maritime vessel detection and tracking |
| 109 | [target_detector](./109_target_detector.ipynb) | Specific target detection and identification |
| 121 | [vehicle_detector](./121_vehicle_detector.ipynb) | Vehicle detection and counting |

### Segmentation Models

| # | Notebook | Description |
| --- | ---------- | ------------- |
| 007 | [builtup_detector](./007_builtup_detector.ipynb) | Urban and built-up area detection and mapping |
| 010 | [cloud_mask](./010_cloud_mask.ipynb) | Cloud and cloud shadow detection and masking |
| 014 | [crop_boundary_delineation](./014_crop_boundary_delineation.ipynb) | Agricultural field boundary detection and delineation |
| 023 | [digitization_2d](./023_digitization_2d.ipynb) | Automatic 2D feature digitization and vectorization |
| 035 | [erosion_detector](./035_erosion_detector.ipynb) | Soil erosion detection and severity mapping |
| 065 | [multi_solution_segmentation](./065_multi_solution_segmentation.ipynb) | Multi-scale image segmentation |
| 088 | [road_network_analyzer](./088_road_network_analyzer.ipynb) | Road network extraction and analysis |
| 092 | [sar_flood_detector](./092_sar_flood_detector.ipynb) | Flood mapping from SAR imagery |
| 114 | [transportation_mapper](./114_transportation_mapper.ipynb) | Transportation infrastructure mapping |
| 119 | [utility_mapper](./119_utility_mapper.ipynb) | Utility infrastructure mapping |
| 124 | [water_surface_detector](./124_water_surface_detector.ipynb) | Water body detection and mapping |
| 125 | [watershed_manager](./125_watershed_manager.ipynb) | Watershed delineation and management |

### Change Detection Models

| # | Notebook | Description |
| --- | ---------- | ------------- |
| 003 | [asset_condition_change](./003_asset_condition_change.ipynb) | Monitoring infrastructure asset condition changes |
| 009 | [change_detector](./009_change_detector.ipynb) | General-purpose bi-temporal change detection |
| 011 | [construction_monitor](./011_construction_monitor.ipynb) | Construction site progress monitoring and analysis |
| 020 | [deforestation_detector](./020_deforestation_detector.ipynb) | Forest cover loss and deforestation detection |
| 022 | [desertification_monitor](./022_desertification_monitor.ipynb) | Land degradation and desertification monitoring |
| 031 | [encroachment_detector](./031_encroachment_detector.ipynb) | Illegal encroachment and land grab detection |
| 042 | [forest_monitor](./042_forest_monitor.ipynb) | Forest health and change monitoring |
| 048 | [hydroelectric_monitor](./048_hydroelectric_monitor.ipynb) | Hydroelectric infrastructure and reservoir monitoring |
| 051 | [land_degradation_detector](./051_land_degradation_detector.ipynb) | Land degradation detection and severity mapping |
| 084 | [protected_area_change_detector](./084_protected_area_change_detector.ipynb) | Change detection in protected areas |
| 113 | [tourist_destination_monitor](./113_tourist_destination_monitor.ipynb) | Tourism site monitoring and impact assessment |
| 117 | [urban_growth_assessor](./117_urban_growth_assessor.ipynb) | Urban expansion and growth assessment |

### Classification Models

| # | Notebook | Description |
| --- | ---------- | ------------- |
| 004 | [beekeeping_suitability](./004_beekeeping_suitability.ipynb) | Site suitability analysis for apiculture |
| 015 | [crop_classifier](./015_crop_classifier.ipynb) | Crop type classification from multispectral imagery |
| 018 | [crop_health_assessor](./018_crop_health_assessor.ipynb) | Crop health and stress assessment |
| 019 | [damage_assessor](./019_damage_assessor.ipynb) | Post-disaster damage assessment and mapping |
| 025 | [disaster_management](./025_disaster_management.ipynb) | Multi-hazard disaster response and management |
| 030 | [emergency_disaster_manager](./030_emergency_disaster_manager.ipynb) | Emergency response coordination |
| 033 | [environmental_monitor](./033_environmental_monitor.ipynb) | Comprehensive environmental condition monitoring |
| 040 | [flood_risk_assessor](./040_flood_risk_assessor.ipynb) | Comprehensive flood hazard and risk analysis |
| 056 | [lulc_classifier](./056_lulc_classifier.ipynb) | Land Use Land Cover classification |
| 067 | [natural_resources_monitor](./067_natural_resources_monitor.ipynb) | Natural resource monitoring and management |
| 079 | [perennial_garden_suitability](./079_perennial_garden_suitability.ipynb) | Perennial crop and garden site suitability |
| 083 | [preparedness_manager](./083_preparedness_manager.ipynb) | Disaster preparedness and planning support |
| 102 | [site_suitability](./102_site_suitability.ipynb) | General site suitability analysis |
| 105 | [spatial_relationship](./105_spatial_relationship.ipynb) | Spatial relationship and topology analysis |
| 110 | [thematic_mapper](./110_thematic_mapper.ipynb) | Thematic map generation and classification |
| 118 | [urban_planner](./118_urban_planner.ipynb) | Urban planning and development support |
| 127 | [wildlife_habitat_analyzer](./127_wildlife_habitat_analyzer.ipynb) | Wildlife habitat suitability analysis |

### Regression Models

| # | Notebook | Description |
| --- | ---------- | ------------- |
| 001 | [accessibility_analyzer](./001_accessibility_analyzer.ipynb) | Spatial accessibility and reachability analysis |
| 008 | [business_valuation](./008_business_valuation.ipynb) | Geospatial factors for business valuation |
| 012 | [coregistration](./012_coregistration.ipynb) | Image coregistration and alignment |
| 017 | [crop_growth_monitor](./017_crop_growth_monitor.ipynb) | Crop growth stage monitoring |
| 021 | [dem_generator](./021_dem_generator.ipynb) | Digital Elevation Model generation |
| 024 | [digitization_3d](./024_digitization_3d.ipynb) | 3D feature extraction and modeling |
| 026 | [drought_monitor](./026_drought_monitor.ipynb) | Drought condition monitoring |
| 027 | [dsm_generator](./027_dsm_generator.ipynb) | Digital Surface Model generation |
| 028 | [dtm_generator](./028_dtm_generator.ipynb) | Digital Terrain Model generation |
| 029 | [economic_spatial_assessor](./029_economic_spatial_assessor.ipynb) | Spatial economic activity assessment |
| 032 | [energy_potential](./032_energy_potential.ipynb) | Renewable energy site potential |
| 034 | [environmental_risk](./034_environmental_risk.ipynb) | Environmental risk assessment |
| 036 | [evi_calculator](./036_evi_calculator.ipynb) | Enhanced Vegetation Index calculation |
| 037 | [field_surveyor](./037_field_surveyor.ipynb) | Automated field survey |
| 039 | [flood_risk](./039_flood_risk.ipynb) | Flood risk assessment |
| 041 | [forest_density_estimator](./041_forest_density_estimator.ipynb) | Forest canopy density estimation |
| 043 | [geostatistical_analyzer](./043_geostatistical_analyzer.ipynb) | Spatial statistics and geostatistical analysis |
| 044 | [grazing_potential](./044_grazing_potential.ipynb) | Pasture and grazing land suitability |
| 046 | [ground_displacement](./046_ground_displacement.ipynb) | Ground surface displacement monitoring |
| 047 | [hazard_vulnerability](./047_hazard_vulnerability.ipynb) | Multi-hazard vulnerability assessment |
| 050 | [insurance_underwriting](./050_insurance_underwriting.ipynb) | Geospatial risk factors for insurance |
| 052 | [land_surface_temperature](./052_land_surface_temperature.ipynb) | Land surface temperature estimation |
| 053 | [landslide_risk](./053_landslide_risk.ipynb) | Landslide susceptibility mapping |
| 055 | [livestock_estimator](./055_livestock_estimator.ipynb) | Livestock population estimation |
| 060 | [mobility_analyzer](./060_mobility_analyzer.ipynb) | Transportation and mobility pattern analysis |
| 061 | [model_3d](./061_model_3d.ipynb) | 3D scene reconstruction |
| 062 | [mosaic_processor](./062_mosaic_processor.ipynb) | Image mosaicking |
| 063 | [mosaicking](./063_mosaicking.ipynb) | Large-scale image mosaic generation |
| 064 | [msi_calculator](./064_msi_calculator.ipynb) | Moisture Stress Index calculation |
| 066 | [multispectral_processor](./066_multispectral_processor.ipynb) | Multispectral image processing |
| 068 | [nbr_calculator](./068_nbr_calculator.ipynb) | Normalized Burn Ratio calculation |
| 069 | [ndvi_calculator](./069_ndvi_calculator.ipynb) | NDVI calculation |
| 070 | [ndwi_calculator](./070_ndwi_calculator.ipynb) | NDWI calculation |
| 071 | [network_analyzer](./071_network_analyzer.ipynb) | Transportation and utility network analysis |
| 075 | [ortho_processor](./075_ortho_processor.ipynb) | Orthorectification and geometric correction |
| 076 | [orthorectification](./076_orthorectification.ipynb) | Image orthorectification |
| 077 | [panchromatic_processor](./077_panchromatic_processor.ipynb) | Panchromatic image processing |
| 078 | [pansharpening](./078_pansharpening.ipynb) | Panchromatic and multispectral fusion |
| 080 | [pipeline_route_planner](./080_pipeline_route_planner.ipynb) | Optimal pipeline routing |
| 085 | [raster_tiler](./085_raster_tiler.ipynb) | Raster data tiling |
| 086 | [reservoir_monitor](./086_reservoir_monitor.ipynb) | Reservoir water level monitoring |
| 087 | [resource_allocation](./087_resource_allocation.ipynb) | Optimal resource allocation |
| 089 | [route_planner](./089_route_planner.ipynb) | Optimal route planning |
| 090 | [salinity_detector](./090_salinity_detector.ipynb) | Soil salinity detection |
| 091 | [sar_amplitude](./091_sar_amplitude.ipynb) | SAR amplitude image processing |
| 093 | [sar_mapping_workflow](./093_sar_mapping_workflow.ipynb) | End-to-end SAR processing |
| 095 | [sar_phase_displacement](./095_sar_phase_displacement.ipynb) | InSAR phase-based displacement |
| 097 | [sar_subsidence_monitor](./097_sar_subsidence_monitor.ipynb) | Ground subsidence monitoring |
| 098 | [savi_calculator](./098_savi_calculator.ipynb) | SAVI calculation |
| 100 | [seismic_risk](./100_seismic_risk.ipynb) | Seismic hazard assessment |
| 103 | [solar_site_selector](./103_solar_site_selector.ipynb) | Solar energy site selection |
| 104 | [spatial_analyzer](./104_spatial_analyzer.ipynb) | General spatial analysis |
| 106 | [stereo_processor](./106_stereo_processor.ipynb) | Stereo image processing |
| 107 | [super_resolution](./107_super_resolution.ipynb) | Image super-resolution |
| 108 | [synthetic_imagery](./108_synthetic_imagery.ipynb) | Synthetic image generation |
| 111 | [timeseries_analyzer](./111_timeseries_analyzer.ipynb) | Multi-temporal time series analysis |
| 112 | [topography_mapper](./112_topography_mapper.ipynb) | Topographic mapping |
| 115 | [tree_height_estimator](./115_tree_height_estimator.ipynb) | Tree height estimation |
| 116 | [tri_stereo_processor](./116_tri_stereo_processor.ipynb) | Three-view stereo processing |
| 120 | [vegetation_condition](./120_vegetation_condition.ipynb) | Vegetation health assessment |
| 122 | [viewshed_analyzer](./122_viewshed_analyzer.ipynb) | Viewshed and visibility analysis |
| 123 | [water_quality_assessor](./123_water_quality_assessor.ipynb) | Water quality parameter estimation |
| 126 | [wildfire_risk](./126_wildfire_risk.ipynb) | Wildfire risk modeling |
| 128 | [wind_site_selector](./128_wind_site_selector.ipynb) | Wind energy site selection |
| 129 | [yield_predictor](./129_yield_predictor.ipynb) | Crop yield prediction |
| 130 | [zonal_statistics](./130_zonal_statistics.ipynb) | Zonal statistics computation |

## Running Notebooks

### Local Environment

```bash
# Clone repository
git clone https://github.com/unbihexium-oss/unbihexium.git
cd unbihexium

# Install dependencies
pip install unbihexium onnxruntime torch jupyter

# Launch Jupyter
jupyter notebook examples/notebooks/
```

### Google Colab

Each notebook includes a "Open in Colab" badge. Click the badge to run the notebook in Google Colab with no local setup required.

## Model Variants

Each notebook demonstrates all four model variants:

| Variant | Resolution | Use Case |
| --------- | ------------ | ---------- |
| tiny | 32x32 | Edge devices, prototyping |
| base | 64x64 | Production deployments |
| large | 128x128 | High accuracy applications |
| mega | 256x256 | Research, maximum precision |

## License

MPL-2.0

## Copyright

2025 Unbihexium OSS Foundation
