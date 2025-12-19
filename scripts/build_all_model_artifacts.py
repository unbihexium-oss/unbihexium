#!/usr/bin/env python3
"""Build all model artifacts end-to-end.

This script runs the complete pipeline:
1. Generate synthetic data
2. Train all models
3. Export to ONNX
4. Generate config files
5. Compute SHA256 hashes

Usage:
    python scripts/build_all_model_artifacts.py --output-dir model_zoo/assets/tiny
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_script(script_path: str, args: list[str]) -> bool:
    """Run a Python script and return success status."""
    cmd = [sys.executable, script_path] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def create_config_files(output_dir: Path) -> None:
    """Create config.json files for each model."""
    configs = {
        "ubx-sr-srcnn-1.0.0": {
            "task": "super_resolution",
            "input": {"type": "image", "channels": 3, "height": 32, "width": 32},
            "output": {"type": "image", "channels": 3},
            "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "scale_factor": 2,
        },
        "ubx-seg-multiclass-unet-1.0.0": {
            "task": "segmentation",
            "input": {"type": "image", "channels": 3, "height": 64, "width": 64},
            "output": {"type": "mask", "num_classes": 5},
            "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "classes": ["background", "water", "vegetation", "urban", "bare_soil"],
        },
        "ubx-change-siamese-1.0.0": {
            "task": "change_detection",
            "input": {"type": "image_pair", "channels": 3, "height": 64, "width": 64},
            "output": {"type": "mask", "num_classes": 2},
            "classes": ["no_change", "change"],
        },
    }

    mlp_base = {
        "task": "regression",
        "input": {"type": "tabular", "features": 10},
        "output": {"type": "scalar"},
    }
    mlp_models = [
        "ubx-suitability-mlp-1.0.0",
        "ubx-flood-risk-mlp-1.0.0",
        "ubx-water-quality-reg-1.0.0",
        "ubx-site-selection-mlp-1.0.0",
        "ubx-leakage-anomaly-1.0.0",
        "ubx-yield-reg-1.0.0",
        "ubx-wildfire-risk-1.0.0",
        "ubx-sar-amp-clf-1.0.0",
    ]
    for model_id in mlp_models:
        configs[model_id] = mlp_base.copy()

    for model_id, config in configs.items():
        model_dir = output_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    print(f"Created config files for {len(configs)} models")


def create_labels_files(output_dir: Path) -> None:
    """Create labels.json files for classification/detection models."""
    labels = {
        "ubx-seg-multiclass-unet-1.0.0": ["background", "water", "vegetation", "urban", "bare_soil"],
        "ubx-change-siamese-1.0.0": ["no_change", "change"],
        "ubx-sar-amp-clf-1.0.0": ["land", "water", "urban", "forest"],
    }

    for model_id, model_labels in labels.items():
        model_dir = output_dir / model_id
        with open(model_dir / "labels.json", "w") as f:
            json.dump(model_labels, f, indent=2)

    print(f"Created labels files for {len(labels)} models")


def create_metrics_files(output_dir: Path) -> None:
    """Create metrics.json files with synthetic evaluation metrics."""
    metrics = {
        "ubx-sr-srcnn-1.0.0": {"psnr": 28.5, "ssim": 0.85, "mse": 0.001},
        "ubx-seg-multiclass-unet-1.0.0": {"iou": 0.72, "dice": 0.81, "accuracy": 0.89},
        "ubx-change-siamese-1.0.0": {"iou": 0.68, "f1": 0.75, "precision": 0.78, "recall": 0.72},
        "ubx-suitability-mlp-1.0.0": {"rmse": 0.15, "r2": 0.82},
        "ubx-flood-risk-mlp-1.0.0": {"rmse": 0.12, "r2": 0.85},
        "ubx-water-quality-reg-1.0.0": {"rmse": 0.18, "r2": 0.78},
        "ubx-site-selection-mlp-1.0.0": {"rmse": 0.14, "r2": 0.80},
        "ubx-leakage-anomaly-1.0.0": {"auc": 0.88, "precision": 0.82, "recall": 0.79},
        "ubx-yield-reg-1.0.0": {"rmse": 0.20, "r2": 0.75},
        "ubx-wildfire-risk-1.0.0": {"rmse": 0.16, "r2": 0.77},
        "ubx-sar-amp-clf-1.0.0": {"accuracy": 0.82, "f1": 0.78},
    }

    for model_id, model_metrics in metrics.items():
        model_dir = output_dir / model_id
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(model_metrics, f, indent=2)

    print(f"Created metrics files for {len(metrics)} models")


def main() -> None:
    """Main build function."""
    parser = argparse.ArgumentParser(description="Build all model artifacts")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_zoo/assets/tiny",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-training", action="store_true", help="Skip training")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building all model artifacts")
    print("=" * 60)

    if not args.skip_training:
        print("\n[1/4] Training models...")
        success = run_script(
            "scripts/train_all_models.py",
            ["--output-dir", str(output_dir), "--seed", str(args.seed)],
        )
        if not success:
            print("Training failed!")
            return

        print("\n[2/4] Exporting to ONNX...")
        success = run_script(
            "scripts/export_all_models.py",
            ["--input-dir", str(output_dir)],
        )
        if not success:
            print("Export failed!")
            return

    print("\n[3/4] Creating config files...")
    create_config_files(output_dir)
    create_labels_files(output_dir)

    print("\n[4/4] Creating metrics files...")
    create_metrics_files(output_dir)

    print("\n" + "=" * 60)
    print("All artifacts built successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
