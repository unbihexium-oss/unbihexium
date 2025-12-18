#!/usr/bin/env python3
"""Model artifact audit script.

Ensures every model in the catalog has a real artifact file.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Audit model artifacts."""
    repo_root = Path(__file__).parent.parent

    # All model base names (without tier suffix)
    all_models = [
        # 1) AI PRODUCTS
        "super_resolution", "synthetic_imagery", "ship_detector", "building_detector",
        "aircraft_detector", "change_detector", "vehicle_detector", "military_objects_detector",
        "multi_solution_segmentation", "greenhouse_detector", "crop_detector",
        "crop_boundary_delineation", "water_surface_detector",
        # 2) TOURISM / DATA PROCESSING
        "route_planner", "site_suitability", "tourist_destination_monitor",
        "accessibility_analyzer", "spatial_analyzer", "geostatistical_analyzer",
        "network_analyzer", "business_valuation", "raster_tiler", "zonal_statistics",
        # 3) VEGETATION & INDICES / FLOOD & WATER
        "ndvi_calculator", "nbr_calculator", "evi_calculator", "msi_calculator",
        "ndwi_calculator", "savi_calculator", "watershed_manager", "flood_risk_assessor",
        "water_quality_assessor", "reservoir_monitor", "timeseries_analyzer",
        # 4) ENVIRONMENT & FORESTRY / IMAGE PROCESSING
        "environmental_monitor", "protected_area_change_detector", "wildlife_habitat_analyzer",
        "emergency_disaster_manager", "forest_monitor", "deforestation_detector",
        "forest_density_estimator", "tree_height_estimator", "thematic_mapper",
        "desertification_monitor", "drought_monitor", "land_degradation_detector",
        "natural_resources_monitor", "marine_pollution_detector", "lulc_classifier",
        "salinity_detector", "erosion_detector", "pansharpening", "coregistration",
        "cloud_mask", "dtm_generator", "dsm_generator", "orthorectification", "mosaicking",
        # 5) GEOSPATIAL ASSET MANAGEMENT / ENERGY
        "object_detector", "damage_assessor", "asset_condition_change",
        "economic_spatial_assessor", "utility_mapper", "spatial_relationship",
        "wind_site_selector", "solar_site_selector", "land_surface_temperature",
        "energy_potential", "hydroelectric_monitor", "pipeline_route_planner",
        "corridor_monitor", "encroachment_detector", "offshore_survey",
        "ground_displacement", "onshore_monitor", "leakage_detector",
        # 6) URBAN / AGRICULTURE
        "urban_planner", "urban_growth_assessor", "resource_allocation",
        "infrastructure_monitor", "construction_monitor", "builtup_detector",
        "road_network_analyzer", "transportation_mapper", "model_3d",
        "topography_mapper", "crop_classifier", "plowed_land_detector",
        "yield_predictor", "crop_growth_monitor", "vegetation_condition",
        "crop_health_assessor", "grazing_potential", "perennial_garden_suitability",
        "pivot_inventory", "beekeeping_suitability", "livestock_estimator", "field_surveyor",
        # 7) RISK / DEFENSE-INTELLIGENCE (Neutral)
        "hazard_vulnerability", "insurance_underwriting", "disaster_management",
        "environmental_risk", "flood_risk", "seismic_risk", "landslide_risk",
        "wildfire_risk", "fire_monitor", "security_monitor", "target_detector",
        "mobility_analyzer", "viewshed_analyzer", "border_monitor",
        "maritime_awareness", "preparedness_manager",
        # 8) VALUE-ADDED IMAGERY
        "dem_generator", "ortho_processor", "mosaic_processor",
        # 9-10) SATELLITE IMAGERY FEATURES
        "stereo_processor", "tri_stereo_processor", "digitization_2d",
        "digitization_3d", "panchromatic_processor", "multispectral_processor",
        # 11) SAR / RADAR IMAGING
        "sar_amplitude", "sar_phase_displacement", "sar_ship_detector",
        "sar_oil_spill_detector", "sar_flood_detector", "sar_subsidence_monitor",
        "sar_mapping_workflow",
    ]

    tiers = ["tiny", "base", "large"]
    issues = []
    total_found = 0

    for tier in tiers:
        tier_dir = repo_root / "model_zoo" / "assets" / tier
        if not tier_dir.exists():
            issues.append(f"Directory not found: {tier_dir}")
            continue

        for model_base in all_models:
            model_file = tier_dir / f"{model_base}_{tier}.pt"
            if not model_file.exists():
                issues.append(f"Missing: {tier}/{model_base}_{tier}.pt")
            elif model_file.stat().st_size == 0:
                issues.append(f"Empty: {tier}/{model_base}_{tier}.pt")
            else:
                total_found += 1

    expected_total = len(all_models) * len(tiers)

    if issues:
        print("Model Artifact Audit Issues:")
        for issue in issues[:20]:  # Show first 20
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues")
        return 1

    print(f"Model Artifact Audit: All {total_found}/{expected_total} models verified")
    print(f"  - Tiny: {len(all_models)} models")
    print(f"  - Base: {len(all_models)} models")
    print(f"  - Large: {len(all_models)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
