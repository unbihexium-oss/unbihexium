# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/decode.py
# Title       : Decoding of raw network outputs
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Turns the raw outputs of the model zoo networks into predictions. The
# functions work on NumPy arrays, so they serve both the PyTorch and the ONNX
# Runtime backends:
#
#   sigmoid, softmax     activation functions
#   decode_centernet     CenterNet heat maps, sizes and offsets to boxes
#   box_iou              pairwise intersection over union
#   nms                  class-aware greedy non-maximum suppression
#   labels_from_logits   class map from segmentation logits
#
# Detector output layout
# ----------------------
# A detector with K classes returns (K + 4, H/4, W/4): K heat map logits,
# box width and height in output-stride pixels, and the sub-pixel offset of
# the box centre. A local maximum at cell (i, j) with score s becomes the
# box centred at ((j + dx) * 4, (i + dy) * 4) with size (w * 4, h * 4)
# (Zhou et al., 2019, Objects as points, arXiv:1904.07850).
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Local maximum filter for peak extraction.
from scipy.ndimage import maximum_filter

# Output stride of the detectors.
from unbihexium.zoo.config import DETECTION_STRIDE


# Numerically stable logistic function.
def sigmoid(x: NDArray[Any]) -> NDArray[np.float32]:
    # Work in float64 to avoid overflow warnings in exp.
    z = np.asarray(x, dtype=np.float64)
    # 1 / (1 + exp(-x)), written with tanh for stability.
    return (0.5 * (1.0 + np.tanh(0.5 * z))).astype(np.float32)


# Numerically stable softmax along one axis.
def softmax(x: NDArray[Any], axis: int = 0) -> NDArray[np.float32]:
    # Work in float64.
    z = np.asarray(x, dtype=np.float64)
    # Subtract the maximum so that exp does not overflow.
    z = np.exp(z - np.nanmax(z, axis=axis, keepdims=True))
    # Normalise to a probability distribution.
    return (z / np.sum(z, axis=axis, keepdims=True)).astype(np.float32)


# Class map from segmentation logits of shape (K, H, W).
def labels_from_logits(
    logits: NDArray[Any],  # Logits or probabilities.
    threshold: float | None = None,  # Minimum probability of the winning class.
    nodata: int = 255,  # Label of pixels below the threshold.
) -> NDArray[np.uint8]:  # Class index per pixel.
    # Most likely class per pixel.
    labels = np.argmax(logits, axis=0).astype(np.uint8)
    # Without a threshold every pixel keeps its class.
    if threshold is None:
        # Return the class map.
        return labels
    # Probability of the winning class.
    confidence = softmax(logits, axis=0).max(axis=0)
    # Mark uncertain pixels.
    labels[confidence < threshold] = nodata
    # Return the class map.
    return labels


# Pairwise intersection over union of two sets of boxes (x1, y1, x2, y2).
def box_iou(a: NDArray[Any], b: NDArray[Any]) -> NDArray[np.float64]:
    # Boxes as (N, 4) and (M, 4) float arrays.
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    # Second set.
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    # Areas of the first set.
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    # Areas of the second set.
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    # Top-left corners of the intersections, shape (N, M, 2).
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    # Bottom-right corners of the intersections.
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    # Width and height of the intersections, zero when disjoint.
    wh = np.clip(bottom_right - top_left, 0, None)
    # Intersection areas.
    inter = wh[..., 0] * wh[..., 1]
    # Union areas.
    union = area_a[:, None] + area_b[None, :] - inter
    # Ratio, zero for empty unions.
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


# Greedy non-maximum suppression; returns the indices of kept boxes.
def nms(
    boxes: NDArray[Any],  # (N, 4) boxes.
    scores: NDArray[Any],  # (N,) scores.
    iou_threshold: float = 0.5,  # Overlap above which the weaker box is dropped.
    classes: NDArray[Any] | None = None,  # (N,) class ids; suppression is per class.
) -> NDArray[np.int64]:  # Indices of the kept boxes, by decreasing score.
    # Boxes as float arrays.
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    # Nothing to suppress.
    if boxes.shape[0] == 0:
        # Empty index array.
        return np.zeros(0, dtype=np.int64)
    # Offset boxes per class so that different classes never overlap.
    if classes is not None:
        # Offset larger than any coordinate.
        offset = (boxes.max() + 1.0) * np.asarray(classes, dtype=np.float64)[:, None]
        # Shift every class into its own region.
        boxes = boxes + offset
    # Candidates sorted by decreasing score.
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    # Indices of kept boxes.
    keep: list[int] = []
    # Process candidates until none remain.
    while order.size:
        # Best remaining box.
        best = int(order[0])
        # Keep it.
        keep.append(best)
        # Overlap of the best box with the others.
        overlap = box_iou(boxes[best], boxes[order[1:]])[0]
        # Drop boxes that overlap too much.
        order = order[1:][overlap <= iou_threshold]
    # Return the kept indices.
    return np.asarray(keep, dtype=np.int64)


# Decode CenterNet outputs of one image into boxes, scores and class ids.
def decode_centernet(
    output: NDArray[Any],  # (K + 4, h, w) raw detector output.
    threshold: float = 0.3,  # Minimum heat map score.
    max_detections: int = 500,  # Maximum number of boxes.
    stride: int = DETECTION_STRIDE,  # Output stride of the detector.
    min_size: float = 1.0,  # Minimum box side in input pixels.
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:  # Boxes, scores, ids.
    # Number of classes.
    k = output.shape[0] - 4
    # Class probabilities from the heat map logits.
    heat = sigmoid(output[:k])
    # 3 x 3 local maxima per class act as non-maximum suppression.
    peaks = heat == maximum_filter(heat, size=(1, 3, 3), mode="constant", cval=0.0)
    # Candidate cells: local maxima above the threshold.
    cls, rows, cols = np.nonzero(peaks & (heat >= threshold))
    # Scores of the candidates.
    scores = heat[cls, rows, cols].astype(np.float64)
    # Keep the best candidates only.
    if scores.size > max_detections:
        # Indices of the top scores.
        top = np.argpartition(-scores, max_detections - 1)[:max_detections]
        # Select them.
        cls, rows, cols, scores = cls[top], rows[top], cols[top], scores[top]
    # Box width and height in output-stride pixels, never negative.
    w = np.clip(output[k, rows, cols], 0, None).astype(np.float64)
    # Box height.
    h = np.clip(output[k + 1, rows, cols], 0, None).astype(np.float64)
    # Box centre in input pixels, including the sub-pixel offset.
    cx = (cols + output[k + 2, rows, cols]) * stride
    # Vertical centre.
    cy = (rows + output[k + 3, rows, cols]) * stride
    # Sizes in input pixels.
    half_w, half_h = w * stride / 2, h * stride / 2
    # Corner coordinates.
    boxes = np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1)
    # Drop degenerate boxes.
    valid = (2 * half_w >= min_size) & (2 * half_h >= min_size)
    # Sort the remaining boxes by decreasing score.
    order = np.argsort(-scores[valid], kind="stable")
    # Return boxes, scores and class ids.
    return boxes[valid][order], scores[valid][order], cls[valid][order].astype(np.int64)


# =============================================================================
# End of module src/unbihexium/ai/decode.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
