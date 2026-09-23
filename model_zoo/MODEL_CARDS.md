# Model Cards

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

The model zoo has 130 model families in four size variants (520 models, 10,655,126,116 parameters in total). Every learned model is a starter model: a complete, trainable architecture with deterministic starter weights that has not been trained on Earth observation data. Train or fine-tune a model before using its predictions. The spectral index family implements published formulas and needs no training.

| Family | Task | Domain | Inputs | Outputs | Status |
| --- | --- | --- | --- | --- | --- |
| [Aircraft Detector](cards/aircraft_detector.md) | detection | ai | 3 | 1 | starter |
| [Border Area Monitor](cards/border_monitor.md) | detection | defense | 3 | 3 | starter |
| [Building Detector](cards/building_detector.md) | detection | urban | 3 | 1 | starter |
| [Built-up Area Detector](cards/builtup_detector.md) | detection | urban | 4 | 1 | starter |
| [Crop Parcel Detector](cards/crop_detector.md) | detection | agriculture | 4 | 2 | starter |
| [Building Damage Assessor](cards/damage_assessor.md) | detection | risk | 3 | 2 | starter |
| [Encroachment Detector](cards/encroachment_detector.md) | detection | assets | 3 | 3 | starter |
| [Active Fire Detector](cards/fire_monitor.md) | detection | environment | 3 | 1 | starter |
| [Greenhouse Detector](cards/greenhouse_detector.md) | detection | agriculture | 3 | 1 | starter |
| [Leakage Detector](cards/leakage_detector.md) | detection | assets | 5 | 1 | starter |
| [Maritime Awareness Detector](cards/maritime_awareness.md) | detection | defense | 3 | 2 | starter |
| [Military Objects Detector](cards/military_objects_detector.md) | detection | defense | 3 | 4 | starter |
| [Generic Object Detector](cards/object_detector.md) | detection | ai | 3 | 6 | starter |
| [Centre Pivot Inventory](cards/pivot_inventory.md) | detection | agriculture | 4 | 1 | starter |
| [SAR Ship Detector](cards/sar_ship_detector.md) | detection | sar | 2 | 1 | starter |
| [Security Monitor](cards/security_monitor.md) | detection | defense | 3 | 3 | starter |
| [Ship Detector](cards/ship_detector.md) | detection | ai | 3 | 1 | starter |
| [Target Detector](cards/target_detector.md) | detection | defense | 3 | 1 | starter |
| [Vehicle Detector](cards/vehicle_detector.md) | detection | ai | 3 | 3 | starter |
| [Cloud and Shadow Mask](cards/cloud_mask.md) | segmentation | imaging | 13 | 4 | starter |
| [Corridor Monitor](cards/corridor_monitor.md) | segmentation | assets | 3 | 3 | starter |
| [Crop Boundary Delineation](cards/crop_boundary_delineation.md) | segmentation | agriculture | 4 | 3 | starter |
| [Crop Type Classifier](cards/crop_classifier.md) | segmentation | agriculture | 10 | 7 | starter |
| [Desertification Monitor](cards/desertification_monitor.md) | segmentation | environment | 10 | 4 | starter |
| [2D Digitisation](cards/digitization_2d.md) | segmentation | imaging | 3 | 5 | starter |
| [Erosion Detector](cards/erosion_detector.md) | segmentation | environment | 5 | 3 | starter |
| [Forest Monitor](cards/forest_monitor.md) | segmentation | forestry | 10 | 4 | starter |
| [Infrastructure Monitor](cards/infrastructure_monitor.md) | segmentation | assets | 3 | 5 | starter |
| [Land Degradation Detector](cards/land_degradation_detector.md) | segmentation | environment | 10 | 2 | starter |
| [Land Use and Land Cover Classifier](cards/lulc_classifier.md) | segmentation | environment | 10 | 11 | starter |
| [Marine Pollution Detector](cards/marine_pollution_detector.md) | segmentation | water | 10 | 3 | starter |
| [General Semantic Segmentation](cards/multi_solution_segmentation.md) | segmentation | ai | 3 | 6 | starter |
| [Ploughed Land Detector](cards/plowed_land_detector.md) | segmentation | agriculture | 4 | 2 | starter |
| [Reservoir Monitor](cards/reservoir_monitor.md) | segmentation | water | 4 | 2 | starter |
| [Road Network Extractor](cards/road_network_analyzer.md) | segmentation | urban | 3 | 2 | starter |
| [Soil Salinity Detector](cards/salinity_detector.md) | segmentation | agriculture | 10 | 4 | starter |
| [SAR Flood Detector](cards/sar_flood_detector.md) | segmentation | sar | 2 | 2 | starter |
| [SAR Oil Spill Detector](cards/sar_oil_spill_detector.md) | segmentation | sar | 1 | 3 | starter |
| [Thematic Mapper](cards/thematic_mapper.md) | segmentation | imaging | 10 | 8 | starter |
| [Landform Mapper](cards/topography_mapper.md) | segmentation | imaging | 1 | 6 | starter |
| [Tourist Destination Monitor](cards/tourist_destination_monitor.md) | segmentation | tourism | 3 | 5 | starter |
| [Transportation Mapper](cards/transportation_mapper.md) | segmentation | urban | 3 | 5 | starter |
| [Urban Land Use Mapper](cards/urban_planner.md) | segmentation | urban | 3 | 5 | starter |
| [Utility Mapper](cards/utility_mapper.md) | segmentation | assets | 3 | 4 | starter |
| [Water Surface Detector](cards/water_surface_detector.md) | segmentation | water | 4 | 2 | starter |
| [Asset Condition Change](cards/asset_condition_change.md) | change_detection | assets | 6 | 3 | starter |
| [Change Detector](cards/change_detector.md) | change_detection | ai | 6 | 2 | starter |
| [Construction Monitor](cards/construction_monitor.md) | change_detection | urban | 6 | 3 | starter |
| [Deforestation Detector](cards/deforestation_detector.md) | change_detection | forestry | 20 | 2 | starter |
| [Protected Area Change Detector](cards/protected_area_change_detector.md) | change_detection | environment | 20 | 4 | starter |
| [Urban Growth Assessor](cards/urban_growth_assessor.md) | change_detection | urban | 20 | 2 | starter |
| [Accessibility Analyser](cards/accessibility_analyzer.md) | dense_regression | tourism | 10 | 1 | starter |
| [Beekeeping Suitability](cards/beekeeping_suitability.md) | dense_regression | agriculture | 12 | 1 | starter |
| [Crop Growth Monitor](cards/crop_growth_monitor.md) | dense_regression | agriculture | 10 | 1 | starter |
| [Crop Health Assessor](cards/crop_health_assessor.md) | dense_regression | agriculture | 10 | 1 | starter |
| [Drought Monitor](cards/drought_monitor.md) | dense_regression | environment | 10 | 1 | starter |
| [Energy Potential](cards/energy_potential.md) | dense_regression | energy | 12 | 1 | starter |
| [Environmental Risk](cards/environmental_risk.md) | dense_regression | risk | 10 | 1 | starter |
| [Flood Risk](cards/flood_risk.md) | dense_regression | water | 12 | 1 | starter |
| [Forest Density Estimator](cards/forest_density_estimator.md) | dense_regression | forestry | 10 | 1 | starter |
| [Grazing Potential](cards/grazing_potential.md) | dense_regression | agriculture | 10 | 1 | starter |
| [Hazard Vulnerability](cards/hazard_vulnerability.md) | dense_regression | risk | 10 | 1 | starter |
| [Landslide Susceptibility](cards/landslide_risk.md) | dense_regression | risk | 12 | 1 | starter |
| [Mobility Analyser](cards/mobility_analyzer.md) | dense_regression | urban | 10 | 1 | starter |
| [Perennial Garden Suitability](cards/perennial_garden_suitability.md) | dense_regression | agriculture | 12 | 1 | starter |
| [Pipeline Route Cost Surface](cards/pipeline_route_planner.md) | dense_regression | assets | 10 | 1 | starter |
| [Route Cost Surface](cards/route_planner.md) | dense_regression | tourism | 10 | 1 | starter |
| [Seismic Risk](cards/seismic_risk.md) | dense_regression | risk | 10 | 1 | starter |
| [Site Suitability](cards/site_suitability.md) | dense_regression | analysis | 10 | 1 | starter |
| [Solar Site Selector](cards/solar_site_selector.md) | dense_regression | energy | 12 | 1 | starter |
| [Spatial Density Estimator](cards/spatial_analyzer.md) | dense_regression | analysis | 4 | 1 | starter |
| [Visibility Estimator](cards/viewshed_analyzer.md) | dense_regression | tourism | 1 | 1 | starter |
| [Water Quality Assessor](cards/water_quality_assessor.md) | dense_regression | water | 10 | 2 | starter |
| [Wildfire Risk](cards/wildfire_risk.md) | dense_regression | risk | 12 | 1 | starter |
| [Wildlife Habitat Suitability](cards/wildlife_habitat_analyzer.md) | dense_regression | environment | 12 | 1 | starter |
| [Wind Site Selector](cards/wind_site_selector.md) | dense_regression | energy | 12 | 1 | starter |
| [Flood Depth Estimator](cards/flood_risk_assessor.md) | dense_regression | water | 12 | 1 | starter |
| [Geostatistical Surface Estimator](cards/geostatistical_analyzer.md) | dense_regression | analysis | 12 | 1 | starter |
| [Proximity Estimator](cards/spatial_relationship.md) | dense_regression | analysis | 4 | 1 | starter |
| [Natural Resources Monitor](cards/natural_resources_monitor.md) | dense_regression | environment | 12 | 1 | starter |
| [Environmental Condition Monitor](cards/environmental_monitor.md) | dense_regression | environment | 10 | 1 | starter |
| [Road Network Density Estimator](cards/network_analyzer.md) | dense_regression | urban | 4 | 1 | starter |
| [Runoff Estimator](cards/watershed_manager.md) | dense_regression | water | 12 | 1 | starter |
| [Hydroelectric Reservoir Level Estimator](cards/hydroelectric_monitor.md) | dense_regression | energy | 5 | 1 | starter |
| [Bathymetry Estimator](cards/offshore_survey.md) | dense_regression | water | 10 | 1 | starter |
| [Surface Temperature Anomaly](cards/onshore_monitor.md) | dense_regression | energy | 7 | 1 | starter |
| [Field Productivity Estimator](cards/field_surveyor.md) | dense_regression | agriculture | 10 | 1 | starter |
| [Yield Predictor](cards/yield_predictor.md) | scene_regression | agriculture | 10 | 1 | starter |
| [Economic Activity Estimator](cards/business_valuation.md) | scene_regression | analysis | 4 | 1 | starter |
| [Property Value Estimator](cards/economic_spatial_assessor.md) | scene_regression | analysis | 3 | 1 | starter |
| [Insurance Risk Scorer](cards/insurance_underwriting.md) | scene_regression | risk | 10 | 2 | starter |
| [Livestock Estimator](cards/livestock_estimator.md) | scene_regression | agriculture | 3 | 1 | starter |
| [Service Demand Estimator](cards/resource_allocation.md) | scene_regression | analysis | 4 | 2 | starter |
| [Disaster Impact Estimator](cards/disaster_management.md) | scene_regression | risk | 10 | 1 | starter |
| [Emergency Needs Estimator](cards/emergency_disaster_manager.md) | scene_regression | risk | 4 | 2 | starter |
| [Preparedness Scorer](cards/preparedness_manager.md) | scene_regression | risk | 10 | 1 | starter |
| [Phenology Estimator](cards/timeseries_analyzer.md) | scene_regression | agriculture | 6 | 3 | starter |
| [Zonal Cover Estimator](cards/zonal_statistics.md) | scene_regression | analysis | 10 | 3 | starter |
| [DEM from Stereo](cards/dem_generator.md) | dense_regression | imaging | 2 | 1 | starter |
| [Building Height Estimator](cards/digitization_3d.md) | dense_regression | urban | 4 | 1 | starter |
| [DSM from Stereo](cards/dsm_generator.md) | dense_regression | imaging | 2 | 1 | starter |
| [DTM from DSM](cards/dtm_generator.md) | dense_regression | imaging | 1 | 1 | starter |
| [Ground Displacement Estimator](cards/ground_displacement.md) | dense_regression | sar | 3 | 1 | starter |
| [Land Surface Temperature](cards/land_surface_temperature.md) | dense_regression | environment | 7 | 1 | starter |
| [Normalised Surface Model](cards/model_3d.md) | dense_regression | imaging | 4 | 1 | starter |
| [SAR Backscatter Normaliser](cards/sar_amplitude.md) | dense_regression | sar | 3 | 2 | starter |
| [InSAR Phase Unwrapper](cards/sar_phase_displacement.md) | dense_regression | sar | 3 | 1 | starter |
| [Subsidence Velocity Estimator](cards/sar_subsidence_monitor.md) | dense_regression | sar | 6 | 1 | starter |
| [Stereo Disparity Estimator](cards/stereo_processor.md) | dense_regression | imaging | 2 | 1 | starter |
| [Canopy Height Estimator](cards/tree_height_estimator.md) | dense_regression | forestry | 12 | 1 | starter |
| [DSM from Tri-Stereo](cards/tri_stereo_processor.md) | dense_regression | imaging | 3 | 1 | starter |
| [Co-registration Flow Estimator](cards/coregistration.md) | enhancement | imaging | 6 | 2 | starter |
| [Mosaic Colour Harmoniser](cards/mosaic_processor.md) | enhancement | imaging | 3 | 3 | starter |
| [Seamline Blender](cards/mosaicking.md) | enhancement | imaging | 6 | 3 | starter |
| [Multispectral Denoiser](cards/multispectral_processor.md) | enhancement | imaging | 10 | 10 | starter |
| [Orthorectification Flow Estimator](cards/ortho_processor.md) | enhancement | imaging | 4 | 2 | starter |
| [Learned Orthorectification](cards/orthorectification.md) | enhancement | imaging | 4 | 3 | starter |
| [Panchromatic Denoiser](cards/panchromatic_processor.md) | enhancement | imaging | 1 | 1 | starter |
| [Pansharpening](cards/pansharpening.md) | enhancement | imaging | 5 | 4 | starter |
| [Tile Radiometric Normaliser](cards/raster_tiler.md) | enhancement | imaging | 3 | 3 | starter |
| [SAR Despeckler](cards/sar_mapping_workflow.md) | enhancement | sar | 2 | 2 | starter |
| [SAR to Optical Translator](cards/synthetic_imagery.md) | enhancement | ai | 2 | 3 | starter |
| [Super-Resolution](cards/super_resolution.md) | super_resolution | imaging | 3 | 3 | starter |
| [EVI Calculator](cards/evi_calculator.md) | spectral_index | indices | 3 | 1 | reference formula |
| [MSI Calculator](cards/msi_calculator.md) | spectral_index | indices | 2 | 1 | reference formula |
| [NBR Calculator](cards/nbr_calculator.md) | spectral_index | indices | 2 | 1 | reference formula |
| [NDVI Calculator](cards/ndvi_calculator.md) | spectral_index | indices | 2 | 1 | reference formula |
| [NDWI Calculator](cards/ndwi_calculator.md) | spectral_index | indices | 2 | 1 | reference formula |
| [SAVI Calculator](cards/savi_calculator.md) | spectral_index | indices | 2 | 1 | reference formula |
| [Vegetation Condition Index](cards/vegetation_condition.md) | spectral_index | indices | 3 | 1 | reference formula |
