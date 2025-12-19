#!/usr/bin/env python3
"""Train all baseline models on synthetic data.

This script trains all models in the Unbihexium model zoo using
deterministic synthetic data.

Usage:
    python scripts/train_all_models.py --output-dir model_zoo/assets/tiny

Models are saved in PyTorch format (.pt) ready for ONNX export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch required. Install with: pip install torch")
    raise SystemExit(1)

from unbihexium.ai.change_detection.siamese import SiameseChangeDetector, SiameseConfig
from unbihexium.ai.models.mlp import MLP, MLPConfig
from unbihexium.ai.models.unet import UNet, UNetConfig
from unbihexium.ai.super_resolution.srcnn import SRCNN, SRCNNConfig
from unbihexium.ai.synthesis.augmenter import SyntheticGenerator


def train_srcnn(output_dir: Path, seed: int = 42, epochs: int = 5) -> dict:
    """Train SRCNN super-resolution model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = SRCNNConfig(input_channels=3, scale_factor=2)
    model = SRCNN(config)

    gen = SyntheticGenerator(seed=seed)
    images = [gen.generate_rgb_image(32, 32) for _ in range(50)]
    images = np.stack(images)
    images_tensor = torch.from_numpy(images)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(images_tensor)
        loss = criterion(output, torch.nn.functional.interpolate(images_tensor, scale_factor=2))
        loss.backward()
        optimizer.step()

    model_dir = output_dir / "ubx-sr-srcnn-1.0.0"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "state_dict": model.state_dict()}, model_dir / "model.pt")

    return {"model_id": "ubx-sr-srcnn-1.0.0", "loss": float(loss.item())}


def train_segmentation_unet(output_dir: Path, seed: int = 42, epochs: int = 5) -> dict:
    """Train UNet segmentation model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = UNetConfig(in_channels=3, num_classes=5, features=(16, 32, 64, 128))
    model = UNet(config)

    gen = SyntheticGenerator(seed=seed)
    images = np.stack([gen.generate_rgb_image(64, 64) for _ in range(50)])
    masks = np.stack([gen.generate_mask(64, 64, num_classes=5) for _ in range(50)])

    images_tensor = torch.from_numpy(images)
    masks_tensor = torch.from_numpy(masks).long()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(images_tensor)
        loss = criterion(output, masks_tensor)
        loss.backward()
        optimizer.step()

    model_dir = output_dir / "ubx-seg-multiclass-unet-1.0.0"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "state_dict": model.state_dict()}, model_dir / "model.pt")

    return {"model_id": "ubx-seg-multiclass-unet-1.0.0", "loss": float(loss.item())}


def train_change_siamese(output_dir: Path, seed: int = 42, epochs: int = 5) -> dict:
    """Train Siamese change detector."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = SiameseConfig(in_channels=3, base_features=16, num_classes=2)
    model = SiameseChangeDetector(config)

    gen = SyntheticGenerator(seed=seed)
    pairs = [gen.generate_change_pair(64, 64) for _ in range(50)]
    before = np.stack([p[0] for p in pairs])
    after = np.stack([p[1] for p in pairs])
    masks = np.stack([p[2] for p in pairs])

    before_tensor = torch.from_numpy(before)
    after_tensor = torch.from_numpy(after)
    masks_tensor = torch.from_numpy(masks).long()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(before_tensor, after_tensor)
        loss = criterion(output, masks_tensor)
        loss.backward()
        optimizer.step()

    model_dir = output_dir / "ubx-change-siamese-1.0.0"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "state_dict": model.state_dict()}, model_dir / "model.pt")

    return {"model_id": "ubx-change-siamese-1.0.0", "loss": float(loss.item())}


def train_mlp(
    model_id: str,
    output_dir: Path,
    input_features: int = 10,
    output_size: int = 1,
    seed: int = 42,
    epochs: int = 10,
) -> dict:
    """Train MLP model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = MLPConfig(input_features=input_features, hidden_sizes=(32, 16), output_size=output_size)
    model = MLP(config)

    gen = SyntheticGenerator(seed=seed)
    features = gen.generate_tabular_features(200, input_features)
    targets = features @ gen.rng.rand(input_features).astype(np.float32)
    if output_size > 1:
        targets = np.tile(targets[:, None], (1, output_size))

    features_tensor = torch.from_numpy(features)
    targets_tensor = torch.from_numpy(targets)
    if output_size == 1:
        targets_tensor = targets_tensor.unsqueeze(1)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(features_tensor)
        loss = criterion(output, targets_tensor)
        loss.backward()
        optimizer.step()

    model_dir = output_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "state_dict": model.state_dict()}, model_dir / "model.pt")

    return {"model_id": model_id, "loss": float(loss.item())}


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_zoo/assets/tiny",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = []

    print("Training all models...")

    # Vision models
    results.append(train_srcnn(output_dir, seed=args.seed, epochs=args.epochs))
    results.append(train_segmentation_unet(output_dir, seed=args.seed, epochs=args.epochs))
    results.append(train_change_siamese(output_dir, seed=args.seed, epochs=args.epochs))

    # MLP models
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

    for i, model_id in enumerate(mlp_models):
        results.append(train_mlp(model_id, output_dir, seed=args.seed + i, epochs=args.epochs))

    print(f"\nTrained {len(results)} models:")
    for r in results:
        print(f"  {r['model_id']}: loss={r['loss']:.4f}")

    report_path = output_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining report saved to: {report_path}")


if __name__ == "__main__":
    main()
