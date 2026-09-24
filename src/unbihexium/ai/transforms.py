# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/transforms.py
# Title       : Training samples, normalisation and data augmentation
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# A Sample holds one image with the target of its task:
#
#   mask     (H, W) integer class labels          segmentation, change
#   values   (K, H, W) float targets              dense regression,
#            (K, s*H, s*W) high resolution image  enhancement, super-resolution
#   boxes    (N, 4) pixel boxes and labels (N,)   detection
#   vector   (K,) float targets                   scene regression
#
# The transforms keep image and target aligned: crops and pads cut the same
# window from both, and the eight symmetries of the square (rotations by 90
# degrees and mirror images) move boxes with the pixels. Photometric jitter
# changes only the image. Normalization standardises every band with
# statistics estimated from the training data; the statistics are stored in
# the checkpoint so that inference applies the same scaling.
#
# Labels outside the image after padding are IGNORE_INDEX (255) for class
# masks and NaN for continuous targets; the losses and metrics skip them.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Sample record and copies with replaced fields.
from dataclasses import dataclass, replace

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Class label that the losses and metrics ignore.
IGNORE_INDEX = 255


# One training or evaluation sample.
@dataclass
class Sample:
    # Image bands, shape (C, H, W), float32.
    image: NDArray[np.float32]
    # Class labels, shape (H, W), int64.
    mask: NDArray[np.int64] | None = None
    # Dense targets, shape (K, H, W) or (K, s*H, s*W).
    values: NDArray[np.float32] | None = None
    # Object boxes, shape (N, 4), pixel coordinates x1, y1, x2, y2.
    boxes: NDArray[np.float32] | None = None
    # Class id of every box, shape (N,).
    labels: NDArray[np.int64] | None = None
    # Scene-level targets, shape (K,).
    vector: NDArray[np.float32] | None = None
    # Name of the sample, for reports.
    name: str = ""

    # Image height in pixels.
    @property
    def height(self) -> int:
        # Second axis of the image.
        return int(self.image.shape[1])

    # Image width in pixels.
    @property
    def width(self) -> int:
        # Third axis of the image.
        return int(self.image.shape[2])

    # Ratio between the target grid and the image grid (super-resolution).
    @property
    def scale(self) -> int:
        # Only dense targets can have a different resolution.
        if self.values is None:
            # Same grid.
            return 1
        # Integer ratio of the widths.
        return max(1, int(self.values.shape[-1]) // max(self.width, 1))


# Per-band standardisation: (x - mean) / std.
class Normalization:
    # Create a normalisation from per-band statistics.
    def __init__(self, mean: list[float] | NDArray[Any], std: list[float] | NDArray[Any]) -> None:
        # Means as float32 column vectors for broadcasting.
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        # Standard deviations, bounded away from zero.
        self.std = np.maximum(np.asarray(std, dtype=np.float32).reshape(-1, 1, 1), np.float32(1e-6))

    # Number of bands.
    @property
    def channels(self) -> int:
        # Length of the mean vector.
        return int(self.mean.shape[0])

    # Normalisation that leaves the data unchanged.
    @classmethod
    def identity(cls, channels: int) -> Normalization:
        # Zero mean and unit standard deviation.
        return cls(np.zeros(channels), np.ones(channels))

    # Estimate the statistics from images, ignoring NaN.
    @classmethod
    def fit(cls, images: Any, max_pixels: int = 2_000_000, seed: int = 0) -> Normalization:
        # Random generator for pixel subsampling.
        rng = np.random.default_rng(seed)
        # Per-band sums, filled on the first image.
        total = squares = counts = None
        # Accumulate the images.
        for image in images:
            # Bands as rows of float64 values.
            x = np.asarray(image, dtype=np.float64).reshape(np.shape(image)[0], -1)
            # Subsample large images to bound the cost.
            if x.shape[1] > max_pixels:
                # Random pixel positions.
                x = x[:, rng.choice(x.shape[1], max_pixels, replace=False)]
            # Valid values.
            valid = np.isfinite(x)
            # Zero out invalid values for the sums.
            x = np.where(valid, x, 0.0)
            # Initialise the sums on the first image.
            if total is None:
                # Sum of values, of squares and number of values per band.
                total, squares, counts = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
            # Add the sums.
            total += x.sum(axis=1)
            # Add the squares.
            squares += (x**2).sum(axis=1)
            # Add the counts.
            counts += valid.sum(axis=1)
        # At least one image is needed.
        if total is None or squares is None or counts is None:
            # Explain the problem.
            raise ValueError("cannot estimate normalisation statistics without images")
        # Avoid division by zero for empty bands.
        n = np.maximum(counts, 1)
        # Means per band.
        mean = total / n
        # Standard deviations per band.
        std = np.sqrt(np.maximum(squares / n - mean**2, 0.0))
        # Bands without variation keep a unit scale.
        std[std < 1e-6] = 1.0
        # Return the normalisation.
        return cls(mean, std)

    # Apply the normalisation; NaN becomes zero, the mean of the data.
    def __call__(self, image: NDArray[Any]) -> NDArray[np.float32]:
        # The number of bands must match.
        if np.shape(image)[0] != self.channels:
            # Explain the mismatch.
            raise ValueError(f"expected {self.channels} bands, got {np.shape(image)[0]}")
        # Standardise.
        x = (np.asarray(image, dtype=np.float32) - self.mean) / self.std
        # Replace missing values.
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # Plain dictionary for checkpoints.
    def to_dict(self) -> dict[str, list[float]]:
        # Means and standard deviations as lists.
        return {"mean": self.mean.ravel().tolist(), "std": self.std.ravel().tolist()}

    # Rebuild from a dictionary.
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Normalization:
        # Means and standard deviations.
        return cls(data["mean"], data["std"])


# Shift boxes into a window and drop boxes that are mostly outside it.
def crop_boxes(
    boxes: NDArray[Any] | None,  # (N, 4) boxes in the original grid.
    labels: NDArray[Any] | None,  # (N,) class ids.
    top: int,  # First row of the window.
    left: int,  # First column of the window.
    height: int,  # Window height.
    width: int,  # Window width.
    min_visible: float = 0.4,  # Minimum fraction of the box area inside the window.
) -> tuple[NDArray[np.float32] | None, NDArray[np.int64] | None]:  # Boxes and labels.
    # Nothing to crop.
    if boxes is None or labels is None or len(boxes) == 0:
        # Return the inputs unchanged.
        return boxes, labels
    # Boxes as float32.
    boxes = np.asarray(boxes, dtype=np.float32)
    # Shift by the window origin.
    shifted = boxes - np.array([left, top, left, top], dtype=np.float32)
    # Clip to the window.
    clipped = np.clip(shifted, 0, [width, height, width, height]).astype(np.float32)
    # Area of the clipped boxes.
    area = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    # Area of the original boxes.
    full = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6)
    # Keep boxes that are visible enough.
    keep = area / full >= min_visible
    # Apply the selection.
    return clipped[keep], np.asarray(labels, dtype=np.int64)[keep]


# Cut the window (top, left, height, width) from a sample.
def crop(sample: Sample, top: int, left: int, height: int, width: int) -> Sample:
    # Row and column slices of the image grid.
    rows, cols = slice(top, top + height), slice(left, left + width)
    # Target grid factor for super-resolution.
    s = sample.scale
    # Boxes shifted into the window.
    boxes, labels = crop_boxes(sample.boxes, sample.labels, top, left, height, width)
    # Rows of the target grid.
    target_rows = slice(top * s, (top + height) * s)
    # Columns of the target grid.
    target_cols = slice(left * s, (left + width) * s)
    # Target window, if the sample has dense targets.
    values = None if sample.values is None else sample.values[:, target_rows, target_cols]
    # New sample with the cropped arrays.
    return replace(
        sample,  # Original sample.
        image=sample.image[:, rows, cols],  # Image window.
        mask=None if sample.mask is None else sample.mask[rows, cols],  # Label window.
        values=values,  # Target window.
        boxes=boxes,  # Shifted boxes.
        labels=labels,  # Their classes.
    )  # End of the copy.


# Pad a sample at the bottom and right to at least (height, width).
def pad(sample: Sample, height: int, width: int) -> Sample:
    # Missing rows and columns.
    dh, dw = max(height - sample.height, 0), max(width - sample.width, 0)
    # Nothing to do for large enough samples.
    if dh == 0 and dw == 0:
        # Return the sample unchanged.
        return sample
    # Target grid factor.
    s = sample.scale
    # Pad the image with zeros.
    image = np.pad(sample.image, ((0, 0), (0, dh), (0, dw)))
    # Pad the class labels with the ignored label.
    mask = (
        None  # Missing mask stays missing.
        if sample.mask is None  # Without a mask.
        else np.pad(sample.mask, ((0, dh), (0, dw)), constant_values=IGNORE_INDEX)  # Padded mask.
    )  # End of the mask.
    # Pad dense targets with NaN.
    values = None
    # Pad dense targets that exist.
    if sample.values is not None:
        # Padding widths on the target grid.
        widths = ((0, 0), (0, dh * s), (0, dw * s))
        # Padded values.
        values = np.pad(sample.values, widths, constant_values=np.nan)
    # Boxes do not move when padding at the bottom and right.
    return replace(sample, image=image, mask=mask, values=values)


# Cut a random window of the given size; pads samples that are too small.
def random_crop(sample: Sample, size: int, rng: np.random.Generator) -> Sample:
    # Make the sample at least as large as the window.
    sample = pad(sample, size, size)
    # Random top row.
    top = int(rng.integers(0, sample.height - size + 1))
    # Random left column.
    left = int(rng.integers(0, sample.width - size + 1))
    # Cut the window.
    return crop(sample, top, left, size, size)


# Rotate by k * 90 degrees counterclockwise, then optionally mirror left-right.
def dihedral(sample: Sample, k: int, flip: bool) -> Sample:
    # Rotate and flip an array whose last two axes are rows and columns.
    def move(a: NDArray[Any] | None) -> NDArray[Any] | None:
        # Missing arrays stay missing.
        if a is None:
            # Nothing to transform.
            return None
        # Rotate in the plane of the last two axes.
        a = np.rot90(a, k, axes=(-2, -1))
        # Mirror left-right.
        if flip:
            # Reverse the columns.
            a = a[..., ::-1]
        # Contiguous copy for PyTorch.
        return np.ascontiguousarray(a)

    # Boxes follow the pixels.
    boxes = sample.boxes
    # Transform the boxes.
    if boxes is not None and len(boxes):
        # Current width and height.
        w, h = float(sample.width), float(sample.height)
        # Copy as float32.
        b = boxes.astype(np.float32).copy()
        # Apply the quarter turns.
        for _ in range(k % 4):
            # Counterclockwise: (x, y) -> (y, w - x); width and height swap.
            b = np.stack([b[:, 1], w - b[:, 2], b[:, 3], w - b[:, 0]], axis=1)
            # The new width is the old height.
            w, h = h, w
        # Mirror left-right: (x, y) -> (w - x, y).
        if flip:
            # Reflect and reorder the x coordinates.
            b = np.stack([w - b[:, 2], b[:, 1], w - b[:, 0], b[:, 3]], axis=1)
        # Store the transformed boxes.
        boxes = b.astype(np.float32)
    # Transformed sample.
    return replace(
        sample,  # Original sample.
        image=move(sample.image),  # Image.
        mask=move(sample.mask),  # Class labels.
        values=move(sample.values),  # Dense targets.
        boxes=boxes,  # Boxes.
    )  # End of the copy.


# Random data augmentation for training.
class Augmenter:
    # Configure the augmentation.
    def __init__(
        self,  # The augmenter.
        geometric: bool = True,  # Random rotations by 90 degrees and mirrors.
        photometric: bool = False,  # Random brightness, contrast and noise.
        strength: float = 0.1,  # Magnitude of the photometric changes.
    ) -> None:  # The constructor returns nothing.
        # Whether to apply the symmetries of the square.
        self.geometric = geometric
        # Whether to change the radiometry.
        self.photometric = photometric
        # Magnitude of the radiometric changes.
        self.strength = strength

    # Augment one sample.
    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        # Random symmetry of the square.
        if self.geometric:
            # Number of quarter turns and mirror flag.
            sample = dihedral(sample, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))
        # Radiometric jitter of the image only.
        if self.photometric:
            # Number of bands.
            c = sample.image.shape[0]
            # Per-band gain around one.
            gain = 1.0 + rng.uniform(-self.strength, self.strength, (c, 1, 1))
            # Per-band offset around zero.
            offset = rng.uniform(-self.strength, self.strength, (c, 1, 1))
            # Additive Gaussian noise.
            noise = rng.normal(0.0, self.strength / 4, sample.image.shape)
            # Apply the changes.
            image = (sample.image * gain + offset + noise).astype(np.float32)
            # Replace the image.
            sample = replace(sample, image=image)
        # Return the augmented sample.
        return sample


# =============================================================================
# End of module src/unbihexium/ai/transforms.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
