#!/usr/bin/env python3
"""Export all trained models to ONNX format.

This script exports all trained PyTorch models to ONNX for
cross-platform inference.

Usage:
    python scripts/export_all_models.py --input-dir model_zoo/assets/tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch required. Install with: pip install torch")
    raise SystemExit(1)

from unbihexium.ai.change_detection.siamese import SiameseChangeDetector
from unbihexium.ai.models.mlp import MLP
from unbihexium.ai.models.unet import UNet
from unbihexium.ai.super_resolution.srcnn import SRCNN


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def export_srcnn(model_dir: Path) -> dict:
    """Export SRCNN to ONNX."""
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")
    model = SRCNN(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32)
    onnx_path = model_dir / "model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch", 2: "height", 3: "width"}},
    )

    return {"path": str(onnx_path), "sha256": compute_sha256(onnx_path)}


def export_unet(model_dir: Path) -> dict:
    """Export UNet to ONNX."""
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")
    model = UNet(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 64)
    onnx_path = model_dir / "model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch", 2: "height", 3: "width"}},
    )

    return {"path": str(onnx_path), "sha256": compute_sha256(onnx_path)}


class SiameseWrapper(torch.nn.Module):
    """Wrapper to combine two inputs for ONNX export."""

    def __init__(self, model: SiameseChangeDetector) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x1 = x[:, :c // 2]
        x2 = x[:, c // 2:]
        return self.model(x1, x2)


def export_siamese(model_dir: Path) -> dict:
    """Export Siamese to ONNX."""
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")
    model = SiameseChangeDetector(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    wrapper = SiameseWrapper(model)
    dummy_input = torch.randn(1, 6, 64, 64)
    onnx_path = model_dir / "model.onnx"

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
    )

    return {"path": str(onnx_path), "sha256": compute_sha256(onnx_path)}


def export_mlp(model_dir: Path) -> dict:
    """Export MLP to ONNX."""
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")
    model = MLP(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    input_features = checkpoint["config"].input_features
    dummy_input = torch.randn(1, input_features)
    onnx_path = model_dir / "model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    return {"path": str(onnx_path), "sha256": compute_sha256(onnx_path)}


def write_sha256_file(model_dir: Path, sha256: str) -> None:
    """Write SHA256 to file."""
    sha_path = model_dir / "model.sha256"
    with open(sha_path, "w") as f:
        f.write(f"{sha256}  model.onnx\n")


def main() -> None:
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export all models to ONNX")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="model_zoo/assets/tiny",
        help="Input directory with trained models",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    results = []

    print("Exporting models to ONNX...")

    export_funcs = {
        "ubx-sr-srcnn-1.0.0": export_srcnn,
        "ubx-seg-multiclass-unet-1.0.0": export_unet,
        "ubx-change-siamese-1.0.0": export_siamese,
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
        export_funcs[model_id] = export_mlp

    for model_id, export_func in export_funcs.items():
        model_dir = input_dir / model_id
        if not model_dir.exists():
            print(f"  Skipping {model_id}: not found")
            continue

        try:
            result = export_func(model_dir)
            write_sha256_file(model_dir, result["sha256"])
            results.append({"model_id": model_id, "sha256": result["sha256"]})
            print(f"  Exported {model_id}")
        except Exception as e:
            print(f"  Failed {model_id}: {e}")

    print(f"\nExported {len(results)} models")

    report_path = input_dir / "export_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
