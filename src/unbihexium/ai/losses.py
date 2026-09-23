# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/losses.py
# Title       : Training losses of the model zoo tasks
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch (unbihexium[torch])
# =============================================================================
#
# Abstract
# --------
# One loss per task, combined by TaskLoss according to the model
# configuration:
#
#   detection          penalty-reduced focal loss on the heat maps plus L1
#                      losses on box size (weight 0.1) and centre offset
#                      (weight 1), normalised by the number of objects
#                      (Zhou et al., 2019)
#   segmentation,      cross-entropy plus soft Dice loss, ignoring label 255
#   change detection   (Milletari et al., 2016)
#   dense regression   masked L1, MSE or Huber loss; NaN targets are skipped
#   scene regression   the same losses on target vectors
#   enhancement,       L1 loss, robust for image restoration
#   super-resolution   (Lim et al., 2017)
#
# Every loss returns a scalar tensor and a dictionary of detached components
# for logging.
#
# References
# ----------
#   Zhou, X., Wang, D., Kraehenbuehl, P. (2019). Objects as points.
#     arXiv:1904.07850.
#   Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollar, P. (2017). Focal
#     loss for dense object detection. ICCV.
#   Milletari, F., Navab, N., Ahmadi, S.-A. (2016). V-Net: fully
#     convolutional neural networks for volumetric medical image
#     segmentation. 3DV.
#   Lim, B., Son, S., Kim, H., Nah, S., Lee, K. M. (2017). Enhanced deep
#     residual networks for single image super-resolution. CVPR Workshops.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Tensor type and functions.
import torch

# Functional losses.
import torch.nn.functional as F

# Module base class.
from torch import nn

# Label ignored by the losses.
from unbihexium.ai.transforms import IGNORE_INDEX

# Task of a model.
from unbihexium.zoo.catalog import Task

# Configuration of a model.
from unbihexium.zoo.config import BuildConfig

# Weight of the box size loss in the detection loss.
SIZE_WEIGHT = 0.1

# Weight of the offset loss in the detection loss.
OFFSET_WEIGHT = 1.0


# Detached Python float of a scalar tensor, for logging.
def _item(value: torch.Tensor) -> float:
    # Detach from the graph before converting.
    return float(value.detach())


# Penalty-reduced pixel-wise focal loss of CenterNet.
def centernet_focal_loss(
    logits: torch.Tensor,  # (N, K, h, w) heat map logits.
    target: torch.Tensor,  # (N, K, h, w) Gaussian heat maps in [0, 1].
    alpha: float = 2.0,  # Focusing exponent.
    beta: float = 4.0,  # Penalty reduction exponent near peaks.
) -> torch.Tensor:  # Scalar loss.
    # Probabilities, clamped away from 0 and 1 for the logarithms.
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    # Peaks are the positive locations.
    pos = target.eq(1).float()
    # Every other location is negative.
    neg = 1.0 - pos
    # Loss at the peaks.
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    # Loss elsewhere, reduced near the peaks.
    neg_loss = torch.log(1 - p) * p**alpha * (1 - target) ** beta * neg
    # Number of objects, at least one.
    num = pos.sum().clamp(min=1.0)
    # Normalised negative log-likelihood.
    return -(pos_loss.sum() + neg_loss.sum()) / num


# Detection loss: focal heat map loss and L1 size and offset losses.
def detection_loss(
    output: torch.Tensor,  # (N, K + 4, h, w) raw detector output.
    targets: dict[str, torch.Tensor],  # heatmap, size, offset and weight tensors.
) -> tuple[torch.Tensor, dict[str, float]]:  # Loss and components.
    # Number of classes.
    k = output.shape[1] - 4
    # Heat map loss.
    heat = centernet_focal_loss(output[:, :k], targets["heatmap"])
    # Centre weights.
    weight = targets["weight"]
    # Number of objects, at least one.
    num = weight.sum().clamp(min=1.0)
    # L1 loss on the box sizes at the centres.
    size = (torch.abs(output[:, k : k + 2] - targets["size"]) * weight).sum() / num
    # L1 loss on the offsets at the centres.
    offset = (torch.abs(output[:, k + 2 : k + 4] - targets["offset"]) * weight).sum() / num
    # Weighted sum.
    total = heat + SIZE_WEIGHT * size + OFFSET_WEIGHT * offset
    # Loss and components.
    return total, {"heatmap": _item(heat), "size": _item(size), "offset": _item(offset)}


# Soft Dice loss over the classes present in the batch.
def dice_loss(
    logits: torch.Tensor,  # (N, K, H, W) class logits.
    mask: torch.Tensor,  # (N, H, W) class labels.
    ignore_index: int = IGNORE_INDEX,  # Label that is skipped.
) -> torch.Tensor:  # Scalar loss.
    # Number of classes.
    k = logits.shape[1]
    # Valid pixels.
    valid = (mask != ignore_index).unsqueeze(1).float()
    # Class probabilities.
    prob = torch.softmax(logits.float(), dim=1) * valid
    # One-hot reference, zero at ignored pixels.
    onehot = F.one_hot(mask.clamp(0, k - 1), k).permute(0, 3, 1, 2).float() * valid
    # Overlap per class.
    inter = (prob * onehot).sum(dim=(0, 2, 3))
    # Total mass per class.
    total = prob.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    # Dice coefficient per class, smoothed.
    dice = (2 * inter + 1.0) / (total + 1.0)
    # Classes that occur in the reference.
    present = onehot.sum(dim=(0, 2, 3)) > 0
    # Mean over present classes, or zero if none.
    return (1 - dice[present]).mean() if present.any() else logits.sum() * 0.0


# Segmentation loss: cross-entropy plus Dice.
def segmentation_loss(
    logits: torch.Tensor,  # (N, K, H, W) class logits.
    mask: torch.Tensor,  # (N, H, W) class labels.
    class_weights: torch.Tensor | None = None,  # Optional weight per class.
    dice_weight: float = 1.0,  # Weight of the Dice term.
) -> tuple[torch.Tensor, dict[str, float]]:  # Loss and components.
    # Batches that contain only ignored pixels contribute nothing.
    if bool((mask == IGNORE_INDEX).all()):
        # Zero that keeps the graph connected.
        zero = logits.sum() * 0.0
        # Loss and components.
        return zero, {"ce": 0.0, "dice": 0.0}
    # Cross-entropy with ignored pixels.
    ce = F.cross_entropy(logits.float(), mask, weight=class_weights, ignore_index=IGNORE_INDEX)
    # Dice loss.
    dice = dice_loss(logits, mask)
    # Weighted sum.
    return ce + dice_weight * dice, {"ce": _item(ce), "dice": _item(dice)}


# Regression loss that skips NaN targets.
def masked_regression_loss(
    pred: torch.Tensor,  # Predictions.
    target: torch.Tensor,  # Targets, NaN where missing.
    kind: str = "l1",  # l1, mse or huber.
) -> tuple[torch.Tensor, dict[str, float]]:  # Loss and components.
    # Valid targets.
    valid = torch.isfinite(target)
    # No valid target in the batch.
    if not bool(valid.any()):
        # Zero that keeps the graph connected.
        return pred.sum() * 0.0, {kind: 0.0}
    # Valid predictions.
    p = pred.float()[valid]
    # Valid targets.
    t = target.float()[valid]
    # Mean absolute error.
    if kind == "l1":
        # L1 loss.
        loss = F.l1_loss(p, t)
    # Mean squared error.
    elif kind == "mse":
        # L2 loss.
        loss = F.mse_loss(p, t)
    # Huber loss, quadratic near zero.
    elif kind == "huber":
        # Smooth L1 loss.
        loss = F.huber_loss(p, t)
    # Unknown loss names are an error.
    else:
        # Explain the choices.
        raise ValueError(f"unknown regression loss {kind!r}; use l1, mse or huber")
    # Loss and component.
    return loss, {kind: _item(loss)}


# Loss of a model zoo task.
class TaskLoss(nn.Module):
    # Create the loss for a model configuration.
    def __init__(
        self,  # The loss module.
        config: BuildConfig,  # Model configuration.
        regression: str = "l1",  # Loss for regression targets.
        class_weights: list[float] | None = None,  # Optional segmentation class weights.
    ) -> None:  # The constructor returns nothing.
        # Initialise the module base class.
        super().__init__()
        # Task of the model.
        self.task = config.task
        # Regression loss name.
        self.regression = regression
        # Class weights as a buffer, so that they move with the module.
        weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        # Register the buffer.
        self.register_buffer("class_weights", weights, persistent=False)

    # Compute the loss for a batch.
    def forward(
        self,  # The loss module.
        output: torch.Tensor,  # Network output.
        targets: dict[str, torch.Tensor],  # Batch targets.
    ) -> tuple[torch.Tensor, dict[str, float]]:  # Loss and components.
        # Detection.
        if self.task is Task.DETECTION:
            # CenterNet loss.
            return detection_loss(output, targets)
        # Segmentation and change detection.
        if self.task in (Task.SEGMENTATION, Task.CHANGE_DETECTION):
            # Cross-entropy and Dice.
            return segmentation_loss(output, targets["mask"], self.class_weights)
        # Scene regression.
        if self.task is Task.SCENE_REGRESSION:
            # Loss on target vectors.
            return masked_regression_loss(output, targets["vector"], self.regression)
        # Dense regression.
        if self.task is Task.DENSE_REGRESSION:
            # Loss on target maps.
            return masked_regression_loss(output, targets["values"], self.regression)
        # Enhancement and super-resolution use L1.
        return masked_regression_loss(output, targets["values"], "l1")


# =============================================================================
# End of module src/unbihexium/ai/losses.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
