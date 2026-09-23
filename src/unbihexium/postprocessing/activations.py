# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/postprocessing/activations.py
# Title       : Class maps and confidence masks from model scores
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Turns scores of shape (K, H, W) into maps and uncertainty layers:
#
#   sigmoid, softmax     activations (shared with unbihexium.ai.decode)
#   threshold            binary map from probabilities
#   argmax               class map from scores
#   confidence_mask      class map with low-confidence pixels as nodata
#   prediction_entropy   normalised Shannon entropy of class probabilities
#   margin               difference between the two largest probabilities
#
# Method
# ------
# The normalised entropy H = -sum_k p_k ln p_k / ln K is 0 for a certain and
# 1 for a uniform prediction (Shannon, 1948). The margin p_(1) - p_(2) is
# the classical breaking-ties measure of active learning (Scheffer et al.,
# 2001); both are used to mask unreliable pixels of map products.
#
# References
# ----------
#   Shannon, C. E. (1948). A mathematical theory of communication. Bell
#     System Technical Journal 27(3), 379-423.
#   Scheffer, T., Decomain, C., Wrobel, S. (2001). Active hidden Markov
#     models for information extraction. Advances in Intelligent Data
#     Analysis, LNCS 2189, 309-318.
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

# Numerically stable activations shared with the inference code.
from unbihexium.ai.decode import sigmoid, softmax


# Binary map of values above (or below) a threshold.
def threshold(
    predictions: NDArray[Any],  # Probabilities or scores.
    threshold: float = 0.5,  # Decision threshold.
    above: bool = True,  # True marks values above the threshold.
) -> NDArray[np.uint8]:  # 1 where the condition holds, else 0.
    # Values as an array.
    p = np.asarray(predictions)
    # Strictly greater or strictly less, as in the previous releases.
    hit = p > threshold if above else p < threshold
    # Return 0/1 as uint8.
    return hit.astype(np.uint8)


# Class map from scores along an axis.
def argmax(predictions: NDArray[Any], axis: int = 0) -> NDArray[Any]:
    # Index of the largest score.
    labels = np.argmax(predictions, axis=axis)
    # Number of classes along the axis.
    k = np.shape(predictions)[axis]
    # uint8 is enough for up to 256 classes.
    return labels.astype(np.uint8 if k <= 256 else np.int32)


# Validate an array of class probabilities of shape (K, ...).
def _probabilities(probabilities: NDArray[Any]) -> NDArray[np.float64]:
    # Probabilities as float.
    p = np.asarray(probabilities, dtype=np.float64)
    # A class axis with at least two classes is needed.
    if p.ndim < 2 or p.shape[0] < 2:
        # Explain the requirement.
        raise ValueError("expected probabilities of shape (K, ...) with K >= 2")
    # Return the array.
    return p


# Class map with pixels below a confidence or margin set to nodata.
def confidence_mask(
    probabilities: NDArray[Any],  # (K, H, W) class probabilities.
    min_confidence: float = 0.5,  # Minimum probability of the winning class.
    min_margin: float = 0.0,  # Minimum difference to the runner-up.
    nodata: int = 255,  # Label of rejected pixels.
) -> NDArray[np.uint8]:  # Class map with rejected pixels as nodata.
    # Validated probabilities.
    p = _probabilities(probabilities)
    # Labels must fit uint8 next to the nodata value.
    if p.shape[0] > 255 or not 0 <= nodata <= 255:
        # Explain the requirement.
        raise ValueError("at most 255 classes and a nodata value in [0, 255] are supported")
    # Winning class per pixel.
    labels = np.argmax(p, axis=0).astype(np.uint8)
    # Rejected pixels.
    rejected = (p.max(axis=0) < min_confidence) | (margin(p) < min_margin)
    # Pixels with NaN probabilities are rejected too.
    rejected |= ~np.isfinite(p).all(axis=0)
    # Mark rejected pixels.
    labels[rejected] = nodata
    # Return the map.
    return labels


# Normalised Shannon entropy of class probabilities, in [0, 1].
def prediction_entropy(probabilities: NDArray[Any]) -> NDArray[np.float64]:
    # Validated probabilities.
    p = _probabilities(probabilities)
    # p ln p with the limit 0 at p = 0.
    plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    # Divide by the entropy of the uniform distribution.
    return -plogp.sum(axis=0) / np.log(p.shape[0])


# Difference between the largest and second largest probability.
def margin(probabilities: NDArray[Any]) -> NDArray[np.float64]:
    # Validated probabilities.
    p = _probabilities(probabilities)
    # The two largest values along the class axis.
    top2 = np.partition(p, p.shape[0] - 2, axis=0)[-2:]
    # Largest minus second largest.
    return top2[1] - top2[0]


# =============================================================================
# End of module src/unbihexium/postprocessing/activations.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
