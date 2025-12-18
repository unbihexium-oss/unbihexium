"""Model zoo registry for managing model entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unbihexium.core.model import ModelConfig, ModelTask, ModelFramework


@dataclass
class ModelZooEntry:
    """Entry in the model zoo."""

    model_id: str
    config: ModelConfig
    sha256: str = ""
    download_url: str | None = None
    size_bytes: int = 0
    license: str = "Apache-2.0"
    source: str = "release"  # repo, release, lfs, external
    version: str = "1.0.0"
    description: str = ""
    domain: str = "general"
    maturity: str = "stable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.config.name,
            "task": self.config.task.value,
            "framework": self.config.framework.value,
            "sha256": self.sha256,
            "download_url": self.download_url,
            "size_bytes": self.size_bytes,
            "license": self.license,
            "source": self.source,
            "version": self.version,
            "domain": self.domain,
            "maturity": self.maturity,
        }


# Global model registry
_models: dict[str, ModelZooEntry] = {}


def register_model(entry: ModelZooEntry) -> None:
    """Register a model in the zoo."""
    _models[entry.model_id] = entry


def get_model(model_id: str) -> ModelZooEntry | None:
    """Get a model entry by ID."""
    return _models.get(model_id)


def list_models(task: str | None = None, domain: str | None = None) -> list[ModelZooEntry]:
    """List all models, optionally filtered by task or domain."""
    models = list(_models.values())
    if task:
        models = [m for m in models if m.config.task.value == task]
    if domain:
        models = [m for m in models if m.domain == domain]
    return models


# =============================================================================
# DETECTION MODELS
# =============================================================================

_detection_models = [
    # Ship Detection
    ModelZooEntry(
        model_id="ship_detector_tiny",
        config=ModelConfig(model_id="ship_detector_tiny", name="Ship Detector Tiny", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1),
        size_bytes=102400, source="repo", domain="maritime", description="Tiny ship detection model for smoke tests",
    ),
    ModelZooEntry(
        model_id="ship_detector_base",
        config=ModelConfig(model_id="ship_detector_base", name="Ship Detector Base", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1),
        size_bytes=52428800, source="release", domain="maritime", description="Base ship detection model",
    ),
    ModelZooEntry(
        model_id="ship_detector_large",
        config=ModelConfig(model_id="ship_detector_large", name="Ship Detector Large", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=5),
        size_bytes=209715200, source="release", domain="maritime", description="Large ship detection model",
    ),
    # Building Detection
    ModelZooEntry(
        model_id="building_detector_tiny",
        config=ModelConfig(model_id="building_detector_tiny", name="Building Detector Tiny", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1),
        size_bytes=102400, source="repo", domain="urban", description="Tiny building detection model",
    ),
    ModelZooEntry(
        model_id="building_detector_base",
        config=ModelConfig(model_id="building_detector_base", name="Building Detector Base", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1),
        size_bytes=52428800, source="release", domain="urban", description="Base building detection model",
    ),
    ModelZooEntry(
        model_id="building_detector_large",
        config=ModelConfig(model_id="building_detector_large", name="Building Detector Large", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=3),
        size_bytes=209715200, source="release", domain="urban", description="Large building detection model",
    ),
    # Aircraft Detection
    ModelZooEntry(
        model_id="aircraft_detector_tiny",
        config=ModelConfig(model_id="aircraft_detector_tiny", name="Aircraft Detector Tiny", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1),
        size_bytes=102400, source="repo", domain="aviation", description="Tiny aircraft detection model",
    ),
    ModelZooEntry(
        model_id="aircraft_detector_base",
        config=ModelConfig(model_id="aircraft_detector_base", name="Aircraft Detector Base", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1),
        size_bytes=52428800, source="release", domain="aviation", description="Base aircraft detection model",
    ),
    # Vehicle Detection
    ModelZooEntry(
        model_id="vehicle_detector_tiny",
        config=ModelConfig(model_id="vehicle_detector_tiny", name="Vehicle Detector Tiny", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1),
        size_bytes=102400, source="repo", domain="transport", description="Tiny vehicle detection model",
    ),
    ModelZooEntry(
        model_id="vehicle_detector_base",
        config=ModelConfig(model_id="vehicle_detector_base", name="Vehicle Detector Base", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=4),
        size_bytes=52428800, source="release", domain="transport", description="Base vehicle detection model",
    ),
    # Energy Detection
    ModelZooEntry(
        model_id="solar_panel_detector_tiny",
        config=ModelConfig(model_id="solar_panel_detector_tiny", name="Solar Panel Detector Tiny", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1),
        size_bytes=102400, source="repo", domain="energy", description="Tiny solar panel detection model",
    ),
    ModelZooEntry(
        model_id="solar_panel_detector_base",
        config=ModelConfig(model_id="solar_panel_detector_base", name="Solar Panel Detector Base", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=2),
        size_bytes=52428800, source="release", domain="energy", description="Base solar panel detection model",
    ),
    ModelZooEntry(
        model_id="wind_turbine_detector_tiny",
        config=ModelConfig(model_id="wind_turbine_detector_tiny", name="Wind Turbine Detector Tiny", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1),
        size_bytes=102400, source="repo", domain="energy", description="Tiny wind turbine detection model",
    ),
    ModelZooEntry(
        model_id="wind_turbine_detector_base",
        config=ModelConfig(model_id="wind_turbine_detector_base", name="Wind Turbine Detector Base", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1),
        size_bytes=52428800, source="release", domain="energy", description="Base wind turbine detection model",
    ),
    ModelZooEntry(
        model_id="oil_tank_detector_tiny",
        config=ModelConfig(model_id="oil_tank_detector_tiny", name="Oil Tank Detector Tiny", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1),
        size_bytes=102400, source="repo", domain="energy", description="Tiny oil tank detection model",
    ),
    ModelZooEntry(
        model_id="oil_tank_detector_base",
        config=ModelConfig(model_id="oil_tank_detector_base", name="Oil Tank Detector Base", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=2),
        size_bytes=52428800, source="release", domain="energy", description="Base oil tank detection model",
    ),
    ModelZooEntry(
        model_id="pool_detector_tiny",
        config=ModelConfig(model_id="pool_detector_tiny", name="Swimming Pool Detector Tiny", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1),
        size_bytes=102400, source="repo", domain="urban", description="Tiny swimming pool detection model",
    ),
]

# =============================================================================
# SEGMENTATION MODELS
# =============================================================================

_segmentation_models = [
    ModelZooEntry(
        model_id="segmentation_tiny",
        config=ModelConfig(model_id="segmentation_tiny", name="Semantic Segmentation Tiny", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=2),
        size_bytes=102400, source="repo", domain="general", description="Tiny semantic segmentation model",
    ),
    ModelZooEntry(
        model_id="segmentation_base",
        config=ModelConfig(model_id="segmentation_base", name="Semantic Segmentation Base", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=6),
        size_bytes=104857600, source="release", domain="general", description="Base semantic segmentation model",
    ),
    ModelZooEntry(
        model_id="land_cover_tiny",
        config=ModelConfig(model_id="land_cover_tiny", name="Land Cover Classifier Tiny", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=5),
        size_bytes=102400, source="repo", domain="environment", description="Tiny land cover classification model",
    ),
    ModelZooEntry(
        model_id="land_cover_base",
        config=ModelConfig(model_id="land_cover_base", name="Land Cover Classifier Base", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=8),
        size_bytes=104857600, source="release", domain="environment", description="Base land cover classification model",
    ),
    ModelZooEntry(
        model_id="water_segmentation_tiny",
        config=ModelConfig(model_id="water_segmentation_tiny", name="Water Segmentation Tiny", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=2),
        size_bytes=102400, source="repo", domain="water", description="Tiny water body segmentation model",
    ),
    ModelZooEntry(
        model_id="water_segmentation_base",
        config=ModelConfig(model_id="water_segmentation_base", name="Water Segmentation Base", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=5),
        size_bytes=104857600, source="release", domain="water", description="Base water body segmentation model",
    ),
    ModelZooEntry(
        model_id="flood_mapping_tiny",
        config=ModelConfig(model_id="flood_mapping_tiny", name="Flood Mapping Tiny", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=2),
        size_bytes=102400, source="repo", domain="water", description="Tiny flood extent mapping model",
    ),
    ModelZooEntry(
        model_id="flood_mapping_base",
        config=ModelConfig(model_id="flood_mapping_base", name="Flood Mapping Base", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=3),
        size_bytes=104857600, source="release", domain="water", description="Base flood extent mapping model",
    ),
    ModelZooEntry(
        model_id="crop_segmentation_tiny",
        config=ModelConfig(model_id="crop_segmentation_tiny", name="Crop Segmentation Tiny", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=2),
        size_bytes=102400, source="repo", domain="agriculture", description="Tiny crop type segmentation model",
    ),
    ModelZooEntry(
        model_id="crop_segmentation_base",
        config=ModelConfig(model_id="crop_segmentation_base", name="Crop Segmentation Base", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=7),
        size_bytes=104857600, source="release", domain="agriculture", description="Base crop type segmentation model",
    ),
    ModelZooEntry(
        model_id="forest_segmentation_tiny",
        config=ModelConfig(model_id="forest_segmentation_tiny", name="Forest Segmentation Tiny", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=2),
        size_bytes=102400, source="repo", domain="forestry", description="Tiny forest segmentation model",
    ),
    ModelZooEntry(
        model_id="forest_segmentation_base",
        config=ModelConfig(model_id="forest_segmentation_base", name="Forest Segmentation Base", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=4),
        size_bytes=104857600, source="release", domain="forestry", description="Base forest segmentation model",
    ),
    ModelZooEntry(
        model_id="road_segmentation_tiny",
        config=ModelConfig(model_id="road_segmentation_tiny", name="Road Segmentation Tiny", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=2),
        size_bytes=102400, source="repo", domain="transport", description="Tiny road segmentation model",
    ),
    ModelZooEntry(
        model_id="road_segmentation_base",
        config=ModelConfig(model_id="road_segmentation_base", name="Road Segmentation Base", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=5),
        size_bytes=104857600, source="release", domain="transport", description="Base road segmentation model",
    ),
    ModelZooEntry(
        model_id="building_footprint_tiny",
        config=ModelConfig(model_id="building_footprint_tiny", name="Building Footprint Tiny", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=2),
        size_bytes=102400, source="repo", domain="urban", description="Tiny building footprint segmentation model",
    ),
    ModelZooEntry(
        model_id="building_footprint_base",
        config=ModelConfig(model_id="building_footprint_base", name="Building Footprint Base", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=4),
        size_bytes=104857600, source="release", domain="urban", description="Base building footprint segmentation model",
    ),
    ModelZooEntry(
        model_id="greenhouse_detector_tiny",
        config=ModelConfig(model_id="greenhouse_detector_tiny", name="Greenhouse Detector Tiny", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=2),
        size_bytes=102400, source="repo", domain="agriculture", description="Tiny greenhouse detection model",
    ),
    ModelZooEntry(
        model_id="greenhouse_detector_base",
        config=ModelConfig(model_id="greenhouse_detector_base", name="Greenhouse Detector Base", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=3),
        size_bytes=104857600, source="release", domain="agriculture", description="Base greenhouse detection model",
    ),
    ModelZooEntry(
        model_id="cloud_segmentation_tiny",
        config=ModelConfig(model_id="cloud_segmentation_tiny", name="Cloud Segmentation Tiny", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=3),
        size_bytes=102400, source="repo", domain="imaging", description="Tiny cloud and shadow segmentation model",
    ),
    ModelZooEntry(
        model_id="cloud_segmentation_base",
        config=ModelConfig(model_id="cloud_segmentation_base", name="Cloud Segmentation Base", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=4),
        size_bytes=104857600, source="release", domain="imaging", description="Base cloud and shadow segmentation model",
    ),
]

# =============================================================================
# CHANGE DETECTION MODELS
# =============================================================================

_change_detection_models = [
    ModelZooEntry(
        model_id="change_detector_tiny",
        config=ModelConfig(model_id="change_detector_tiny", name="Change Detector Tiny", task=ModelTask.CHANGE_DETECTION, framework=ModelFramework.PYTORCH, input_channels=6, num_classes=2),
        size_bytes=102400, source="repo", domain="general", description="Tiny change detection model",
    ),
    ModelZooEntry(
        model_id="change_detector_base",
        config=ModelConfig(model_id="change_detector_base", name="Change Detector Base", task=ModelTask.CHANGE_DETECTION, framework=ModelFramework.PYTORCH, input_channels=6, num_classes=2),
        size_bytes=209715200, source="release", domain="general", description="Base change detection model",
    ),
    ModelZooEntry(
        model_id="urban_change_detector_tiny",
        config=ModelConfig(model_id="urban_change_detector_tiny", name="Urban Change Detector Tiny", task=ModelTask.CHANGE_DETECTION, framework=ModelFramework.PYTORCH, input_channels=6, num_classes=3),
        size_bytes=102400, source="repo", domain="urban", description="Tiny urban change detection model",
    ),
    ModelZooEntry(
        model_id="deforestation_detector_tiny",
        config=ModelConfig(model_id="deforestation_detector_tiny", name="Deforestation Detector Tiny", task=ModelTask.CHANGE_DETECTION, framework=ModelFramework.PYTORCH, input_channels=6, num_classes=3),
        size_bytes=102400, source="repo", domain="forestry", description="Tiny deforestation detection model",
    ),
    ModelZooEntry(
        model_id="deforestation_detector_base",
        config=ModelConfig(model_id="deforestation_detector_base", name="Deforestation Detector Base", task=ModelTask.CHANGE_DETECTION, framework=ModelFramework.PYTORCH, input_channels=6, num_classes=4),
        size_bytes=209715200, source="release", domain="forestry", description="Base deforestation detection model",
    ),
]

# =============================================================================
# SUPER RESOLUTION MODELS
# =============================================================================

_super_resolution_models = [
    ModelZooEntry(
        model_id="super_resolution_tiny",
        config=ModelConfig(model_id="super_resolution_tiny", name="Super Resolution Tiny", task=ModelTask.SUPER_RESOLUTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=3),
        size_bytes=102400, source="repo", domain="imaging", description="Tiny super resolution model (2x)",
    ),
    ModelZooEntry(
        model_id="super_resolution_2x",
        config=ModelConfig(model_id="super_resolution_2x", name="Super Resolution 2x", task=ModelTask.SUPER_RESOLUTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=3),
        size_bytes=52428800, source="release", domain="imaging", description="Production super resolution model (2x)",
    ),
    ModelZooEntry(
        model_id="super_resolution_4x",
        config=ModelConfig(model_id="super_resolution_4x", name="Super Resolution 4x", task=ModelTask.SUPER_RESOLUTION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=3),
        size_bytes=104857600, source="release", domain="imaging", description="Production super resolution model (4x)",
    ),
    ModelZooEntry(
        model_id="pan_sharpening_tiny",
        config=ModelConfig(model_id="pan_sharpening_tiny", name="Pan Sharpening Tiny", task=ModelTask.SUPER_RESOLUTION, framework=ModelFramework.PYTORCH, input_channels=4, num_classes=3),
        size_bytes=102400, source="repo", domain="imaging", description="Tiny pan-sharpening model",
    ),
    ModelZooEntry(
        model_id="pan_sharpening_base",
        config=ModelConfig(model_id="pan_sharpening_base", name="Pan Sharpening Base", task=ModelTask.SUPER_RESOLUTION, framework=ModelFramework.PYTORCH, input_channels=4, num_classes=3),
        size_bytes=104857600, source="release", domain="imaging", description="Base pan-sharpening model",
    ),
]

# =============================================================================
# SAR MODELS
# =============================================================================

_sar_models = [
    ModelZooEntry(
        model_id="sar_segmentation_tiny",
        config=ModelConfig(model_id="sar_segmentation_tiny", name="SAR Segmentation Tiny", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=2, num_classes=2),
        size_bytes=102400, source="repo", domain="sar", maturity="research", description="Tiny SAR segmentation model",
    ),
    ModelZooEntry(
        model_id="sar_ship_detector_tiny",
        config=ModelConfig(model_id="sar_ship_detector_tiny", name="SAR Ship Detector Tiny", task=ModelTask.DETECTION, framework=ModelFramework.PYTORCH, input_channels=2, num_classes=1),
        size_bytes=102400, source="repo", domain="sar", maturity="research", description="Tiny SAR ship detection model",
    ),
    ModelZooEntry(
        model_id="sar_oil_spill_detector_tiny",
        config=ModelConfig(model_id="sar_oil_spill_detector_tiny", name="SAR Oil Spill Detector Tiny", task=ModelTask.SEGMENTATION, framework=ModelFramework.PYTORCH, input_channels=2, num_classes=2),
        size_bytes=102400, source="repo", domain="sar", maturity="research", description="Tiny SAR oil spill detection model",
    ),
]

# =============================================================================
# CLASSIFICATION MODELS
# =============================================================================

_classification_models = [
    ModelZooEntry(
        model_id="scene_classifier_tiny",
        config=ModelConfig(model_id="scene_classifier_tiny", name="Scene Classifier Tiny", task=ModelTask.CLASSIFICATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=5),
        size_bytes=102400, source="repo", domain="general", description="Tiny scene classification model",
    ),
    ModelZooEntry(
        model_id="scene_classifier_base",
        config=ModelConfig(model_id="scene_classifier_base", name="Scene Classifier Base", task=ModelTask.CLASSIFICATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=15),
        size_bytes=52428800, source="release", domain="general", description="Base scene classification model",
    ),
    ModelZooEntry(
        model_id="damage_classifier_tiny",
        config=ModelConfig(model_id="damage_classifier_tiny", name="Damage Classifier Tiny", task=ModelTask.CLASSIFICATION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=4),
        size_bytes=102400, source="repo", domain="risk", description="Tiny building damage classification model",
    ),
]

# =============================================================================
# EMBEDDING MODELS
# =============================================================================

_embedding_models = [
    ModelZooEntry(
        model_id="geo_embedding_tiny",
        config=ModelConfig(model_id="geo_embedding_tiny", name="Geo Embedding Tiny", task=ModelTask.EMBEDDING, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=256),
        size_bytes=102400, source="repo", domain="general", description="Tiny geospatial embedding model",
    ),
    ModelZooEntry(
        model_id="geo_embedding_base",
        config=ModelConfig(model_id="geo_embedding_base", name="Geo Embedding Base", task=ModelTask.EMBEDDING, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=768),
        size_bytes=314572800, source="release", domain="general", description="Base geospatial embedding model",
    ),
]

# =============================================================================
# REGRESSION MODELS
# =============================================================================

_regression_models = [
    ModelZooEntry(
        model_id="yield_predictor_tiny",
        config=ModelConfig(model_id="yield_predictor_tiny", name="Yield Predictor Tiny", task=ModelTask.REGRESSION, framework=ModelFramework.PYTORCH, input_channels=12, num_classes=1),
        size_bytes=102400, source="repo", domain="agriculture", description="Tiny crop yield prediction model",
    ),
    ModelZooEntry(
        model_id="yield_predictor_base",
        config=ModelConfig(model_id="yield_predictor_base", name="Yield Predictor Base", task=ModelTask.REGRESSION, framework=ModelFramework.PYTORCH, input_channels=12, num_classes=1),
        size_bytes=104857600, source="release", domain="agriculture", description="Base crop yield prediction model",
    ),
    ModelZooEntry(
        model_id="biomass_estimator_tiny",
        config=ModelConfig(model_id="biomass_estimator_tiny", name="Biomass Estimator Tiny", task=ModelTask.REGRESSION, framework=ModelFramework.PYTORCH, input_channels=6, num_classes=1),
        size_bytes=102400, source="repo", domain="forestry", description="Tiny above-ground biomass estimation model",
    ),
    ModelZooEntry(
        model_id="height_estimator_tiny",
        config=ModelConfig(model_id="height_estimator_tiny", name="Height Estimator Tiny", task=ModelTask.REGRESSION, framework=ModelFramework.PYTORCH, input_channels=3, num_classes=1),
        size_bytes=102400, source="repo", domain="urban", description="Tiny building height estimation model",
    ),
]

# Register all models
for model in _detection_models:
    register_model(model)

for model in _segmentation_models:
    register_model(model)

for model in _change_detection_models:
    register_model(model)

for model in _super_resolution_models:
    register_model(model)

for model in _sar_models:
    register_model(model)

for model in _classification_models:
    register_model(model)

for model in _embedding_models:
    register_model(model)

for model in _regression_models:
    register_model(model)
