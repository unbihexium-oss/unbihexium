#!/usr/bin/env python3
"""Build synthetic datasets for model training.

This script generates deterministic synthetic datasets for training
all baseline models in the Unbihexium model zoo.

Usage:
    python scripts/build_synthetic_datasets.py --output-dir data/synthetic

All outputs are fully reproducible with the same seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from unbihexium.ai.synthesis.augmenter import SyntheticGenerator


def build_image_dataset(
    output_dir: Path,
    num_samples: int = 100,
    image_size: int = 64,
    seed: int = 42,
) -> None:
    """Build synthetic image dataset for segmentation/detection.

    Args:
        output_dir: Output directory.
        num_samples: Number of samples.
        image_size: Image size.
        seed: Random seed.
    """
    gen = SyntheticGenerator(seed=seed)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        image = gen.generate_rgb_image(image_size, image_size)
        mask = gen.generate_mask(image_size, image_size, num_classes=5)

        np.save(images_dir / f"image_{i:04d}.npy", image)
        np.save(masks_dir / f"mask_{i:04d}.npy", mask)

    metadata = {
        "num_samples": num_samples,
        "image_size": image_size,
        "seed": seed,
        "num_classes": 5,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Built image dataset: {num_samples} samples in {output_dir}")


def build_change_dataset(
    output_dir: Path,
    num_samples: int = 50,
    image_size: int = 64,
    seed: int = 42,
) -> None:
    """Build synthetic change detection dataset.

    Args:
        output_dir: Output directory.
        num_samples: Number of samples.
        image_size: Image size.
        seed: Random seed.
    """
    gen = SyntheticGenerator(seed=seed)
    before_dir = output_dir / "before"
    after_dir = output_dir / "after"
    masks_dir = output_dir / "masks"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        before, after, mask = gen.generate_change_pair(image_size, image_size)

        np.save(before_dir / f"before_{i:04d}.npy", before)
        np.save(after_dir / f"after_{i:04d}.npy", after)
        np.save(masks_dir / f"mask_{i:04d}.npy", mask)

    metadata = {
        "num_samples": num_samples,
        "image_size": image_size,
        "seed": seed,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Built change dataset: {num_samples} samples in {output_dir}")


def build_tabular_dataset(
    output_dir: Path,
    num_samples: int = 500,
    num_features: int = 10,
    seed: int = 42,
) -> None:
    """Build synthetic tabular dataset for MLP models.

    Args:
        output_dir: Output directory.
        num_samples: Number of samples.
        num_features: Number of features.
        seed: Random seed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gen = SyntheticGenerator(seed=seed)

    features = gen.generate_tabular_features(num_samples, num_features)
    weights = gen.rng.rand(num_features).astype(np.float32)
    targets = features @ weights + gen.rng.randn(num_samples).astype(np.float32) * 0.1

    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "targets.npy", targets)

    metadata = {
        "num_samples": num_samples,
        "num_features": num_features,
        "seed": seed,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Built tabular dataset: {num_samples} samples in {output_dir}")


def main() -> None:
    """Main function to build all synthetic datasets."""
    parser = argparse.ArgumentParser(description="Build synthetic datasets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    base_dir = Path(args.output_dir)

    print("Building synthetic datasets...")
    print(f"Seed: {args.seed}")

    build_image_dataset(base_dir / "segmentation", seed=args.seed)
    build_change_dataset(base_dir / "change", seed=args.seed + 1)
    build_tabular_dataset(base_dir / "tabular", seed=args.seed + 2)

    print("\nAll datasets built successfully.")


if __name__ == "__main__":
    main()
