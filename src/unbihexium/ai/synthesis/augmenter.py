"""Synthesis and augmentation utilities.

This module provides:
- Deterministic synthetic data generation
- Augmentation policies for training
- EO-specific augmentations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class AugmentConfig:
    """Configuration for augmentation.

    Attributes:
        flip_horizontal: Enable horizontal flip.
        flip_vertical: Enable vertical flip.
        rotate90: Enable 90-degree rotations.
        brightness_range: Brightness adjustment range.
        gamma_range: Gamma adjustment range.
    """

    flip_horizontal: bool = True
    flip_vertical: bool = True
    rotate90: bool = True
    brightness_range: tuple[float, float] = (0.9, 1.1)
    gamma_range: tuple[float, float] = (0.8, 1.2)


class SyntheticGenerator:
    """Generator for synthetic EO-like data.

    Produces deterministic synthetic imagery for training and testing.
    All outputs are reproducible with the same seed.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize generator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_rgb_image(
        self,
        height: int = 64,
        width: int = 64,
        add_objects: bool = True,
    ) -> NDArray[np.floating[Any]]:
        """Generate synthetic RGB image.

        Args:
            height: Image height.
            width: Image width.
            add_objects: Whether to add synthetic objects.

        Returns:
            RGB image array (3, H, W) in [0, 1].
        """
        base = self.rng.rand(3, height, width).astype(np.float32)
        base = base * 0.3 + 0.2

        if add_objects:
            num_objects = self.rng.randint(1, 5)
            for _ in range(num_objects):
                cx = self.rng.randint(10, width - 10)
                cy = self.rng.randint(10, height - 10)
                size = self.rng.randint(5, 15)

                y1 = max(0, cy - size // 2)
                y2 = min(height, cy + size // 2)
                x1 = max(0, cx - size // 2)
                x2 = min(width, cx + size // 2)

                color = self.rng.rand(3).astype(np.float32) * 0.5 + 0.5
                base[:, y1:y2, x1:x2] = color[:, None, None]

        return np.clip(base, 0, 1)

    def generate_mask(
        self,
        height: int = 64,
        width: int = 64,
        num_classes: int = 2,
    ) -> NDArray[np.integer[Any]]:
        """Generate synthetic segmentation mask.

        Args:
            height: Mask height.
            width: Mask width.
            num_classes: Number of classes.

        Returns:
            Mask array (H, W) with class indices.
        """
        mask = np.zeros((height, width), dtype=np.int32)

        num_regions = self.rng.randint(2, 6)
        for _ in range(num_regions):
            cx = self.rng.randint(0, width)
            cy = self.rng.randint(0, height)
            radius = self.rng.randint(10, 30)
            class_id = self.rng.randint(1, num_classes)

            y, x = np.ogrid[:height, :width]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mask[dist <= radius] = class_id

        return mask

    def generate_change_pair(
        self,
        height: int = 64,
        width: int = 64,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.integer[Any]]]:
        """Generate before/after pair with change mask.

        Args:
            height: Image height.
            width: Image width.

        Returns:
            Tuple of (before, after, change_mask).
        """
        before = self.generate_rgb_image(height, width, add_objects=True)
        after = before.copy()

        change_mask = np.zeros((height, width), dtype=np.int32)

        num_changes = self.rng.randint(1, 4)
        for _ in range(num_changes):
            cx = self.rng.randint(10, width - 10)
            cy = self.rng.randint(10, height - 10)
            size = self.rng.randint(8, 20)

            y1 = max(0, cy - size // 2)
            y2 = min(height, cy + size // 2)
            x1 = max(0, cx - size // 2)
            x2 = min(width, cx + size // 2)

            new_color = self.rng.rand(3).astype(np.float32) * 0.5 + 0.5
            after[:, y1:y2, x1:x2] = new_color[:, None, None]
            change_mask[y1:y2, x1:x2] = 1

        return before, after, change_mask

    def generate_tabular_features(
        self,
        num_samples: int = 100,
        num_features: int = 10,
    ) -> NDArray[np.floating[Any]]:
        """Generate synthetic tabular features.

        Args:
            num_samples: Number of samples.
            num_features: Number of features.

        Returns:
            Feature array (num_samples, num_features).
        """
        return self.rng.rand(num_samples, num_features).astype(np.float32)


class Augmenter:
    """Augmentation pipeline for EO imagery."""

    def __init__(self, config: AugmentConfig | None = None, seed: int = 42) -> None:
        """Initialize augmenter.

        Args:
            config: Augmentation configuration.
            seed: Random seed.
        """
        self.config = config or AugmentConfig()
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        image: NDArray[np.floating[Any]],
        mask: NDArray[np.integer[Any]] | None = None,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.integer[Any]] | None]:
        """Apply augmentation.

        Args:
            image: Image array (C, H, W).
            mask: Optional mask array (H, W).

        Returns:
            Tuple of (augmented_image, augmented_mask).
        """
        cfg = self.config

        if cfg.flip_horizontal and self.rng.rand() > 0.5:
            image = np.flip(image, axis=2).copy()
            if mask is not None:
                mask = np.flip(mask, axis=1).copy()

        if cfg.flip_vertical and self.rng.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=0).copy()

        if cfg.rotate90 and self.rng.rand() > 0.5:
            k = self.rng.randint(1, 4)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            if mask is not None:
                mask = np.rot90(mask, k).copy()

        brightness = self.rng.uniform(*cfg.brightness_range)
        image = np.clip(image * brightness, 0, 1)

        return image.astype(np.float32), mask
