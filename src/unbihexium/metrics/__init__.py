# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Evaluation metrics module.

This module provides functions for computing common evaluation
metrics for geospatial AI models.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def iou(pred: NDArray, target: NDArray, smooth: float = 1e-6) -> float:
    """Compute Intersection over Union.

    Args:
        pred: Prediction array
        target: Target array
        smooth: Smoothing factor

    Returns:
        IoU score
    """
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    return (intersection + smooth) / (union + smooth)


def dice(pred: NDArray, target: NDArray, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient.

    Args:
        pred: Prediction array
        target: Target array
        smooth: Smoothing factor

    Returns:
        Dice score
    """
    intersection = np.sum(pred * target)
    return (2 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)


def precision(pred: NDArray, target: NDArray) -> float:
    """Compute precision.

    Args:
        pred: Prediction array
        target: Target array

    Returns:
        Precision score
    """
    tp = np.sum(pred * target)
    fp = np.sum(pred * (1 - target))
    return tp / (tp + fp + 1e-8)


def recall(pred: NDArray, target: NDArray) -> float:
    """Compute recall.

    Args:
        pred: Prediction array
        target: Target array

    Returns:
        Recall score
    """
    tp = np.sum(pred * target)
    fn = np.sum((1 - pred) * target)
    return tp / (tp + fn + 1e-8)


def f1_score(pred: NDArray, target: NDArray) -> float:
    """Compute F1 score.

    Args:
        pred: Prediction array
        target: Target array

    Returns:
        F1 score
    """
    p = precision(pred, target)
    r = recall(pred, target)
    return 2 * p * r / (p + r + 1e-8)


def accuracy(pred: NDArray, target: NDArray) -> float:
    """Compute accuracy.

    Args:
        pred: Prediction array
        target: Target array

    Returns:
        Accuracy score
    """
    return np.mean(pred == target)


def mean_iou(pred: NDArray, target: NDArray, num_classes: int) -> float:
    """Compute mean IoU over classes.

    Args:
        pred: Prediction array
        target: Target array
        num_classes: Number of classes

    Returns:
        Mean IoU score
    """
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        if np.sum(target_c) > 0:
            ious.append(iou(pred_c, target_c))
    return np.mean(ious) if ious else 0.0


def rmse(pred: NDArray, target: NDArray) -> float:
    """Compute Root Mean Square Error.

    Args:
        pred: Prediction array
        target: Target array

    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((pred - target) ** 2))


def mae(pred: NDArray, target: NDArray) -> float:
    """Compute Mean Absolute Error.

    Args:
        pred: Prediction array
        target: Target array

    Returns:
        MAE value
    """
    return np.mean(np.abs(pred - target))


def r_squared(pred: NDArray, target: NDArray) -> float:
    """Compute R-squared (coefficient of determination).

    Args:
        pred: Prediction array
        target: Target array

    Returns:
        R-squared value
    """
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


def psnr(pred: NDArray, target: NDArray, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Prediction array
        target: Target array
        max_val: Maximum pixel value

    Returns:
        PSNR in dB
    """
    mse = np.mean((pred - target) ** 2)
    return 10 * np.log10(max_val**2 / (mse + 1e-8))


def ssim(pred: NDArray, target: NDArray, window_size: int = 11) -> float:
    """Compute Structural Similarity Index.

    Args:
        pred: Prediction array
        target: Target array
        window_size: Window size for local statistics

    Returns:
        SSIM value
    """
    from scipy.ndimage import uniform_filter

    c1 = 0.01**2
    c2 = 0.03**2

    mu_x = uniform_filter(pred, window_size)
    mu_y = uniform_filter(target, window_size)

    sigma_x = uniform_filter(pred**2, window_size) - mu_x**2
    sigma_y = uniform_filter(target**2, window_size) - mu_y**2
    sigma_xy = uniform_filter(pred * target, window_size) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )

    return np.mean(ssim_map)


__all__ = [
    "iou",
    "dice",
    "precision",
    "recall",
    "f1_score",
    "accuracy",
    "mean_iou",
    "rmse",
    "mae",
    "r_squared",
    "psnr",
    "ssim",
]
