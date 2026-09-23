# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/metrics/change.py
# Title       : Accuracy of change detection and class transition analysis
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Measures for binary change maps and for two-date class maps:
#
#   change_detection_metrics   true/false positives and negatives, detection
#                              rate, false alarm rate, missed detection rate,
#                              precision, F1, IoU, overall accuracy and kappa
#   change_map                 pixels whose class differs between two dates
#   transition_matrix          from-to counts of classes between two dates
#   transition_summary         gross gain, gross loss, net change and swap
#                              of every class (Pontius et al., 2004)
#
# Changed pixels are the positive class. The false alarm rate is FP / (FP +
# TN), the share of unchanged reference pixels flagged as change; the missed
# detection rate is FN / (TP + FN).
#
# References
# ----------
#   Bruzzone, L., Prieto, D. F. (2000). Automatic analysis of the difference
#     image for unsupervised change detection. IEEE Transactions on
#     Geoscience and Remote Sensing 38(3), 1171-1182.
#   Pontius, R. G., Shusas, E., McEachern, M. (2004). Detecting important
#     categorical land changes while accounting for persistence. Agriculture,
#     Ecosystems and Environment 101(2-3), 251-268.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values and sequences.
from typing import Any, Sequence

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Error matrix and kappa.
from unbihexium.metrics.classification import cohen_kappa, confusion_matrix


# Safe division that returns NaN for a zero denominator.
def _div(num: float, den: float) -> float:
    # NaN when undefined.
    return num / den if den else float("nan")


# Accuracy of a binary change map against a reference change map.
def change_detection_metrics(
    reference: NDArray[Any],  # Reference change mask (nonzero is change).
    predicted: NDArray[Any],  # Predicted change mask.
    valid: NDArray[Any] | None = None,  # True where pixels are evaluated.
) -> dict[str, float]:  # Counts and rates.
    # Reference as boolean.
    r = np.asarray(reference) > 0
    # Prediction as boolean.
    p = np.asarray(predicted) > 0
    # Shapes must agree.
    if r.shape != p.shape:
        # Explain the requirement.
        raise ValueError(f"shape mismatch: {r.shape} and {p.shape}")
    # Evaluated pixels.
    v = np.ones(r.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    # Changed in both.
    tp = float(np.sum(r & p & v))
    # Flagged but unchanged.
    fp = float(np.sum(~r & p & v))
    # Missed change.
    fn = float(np.sum(r & ~p & v))
    # Unchanged in both.
    tn = float(np.sum(~r & ~p & v))
    # Evaluated pixels.
    total = tp + fp + fn + tn
    # Kappa of the two-class matrix, rows reference.
    matrix = np.array([[tn, fp], [fn, tp]])
    # Report.
    return {
        "tp": tp,  # True positives.
        "fp": fp,  # False positives (false alarms).
        "fn": fn,  # False negatives (missed change).
        "tn": tn,  # True negatives.
        "detection_rate": _div(tp, tp + fn),  # Recall of change.
        "false_alarm_rate": _div(fp, fp + tn),  # Unchanged flagged as change.
        "missed_detection_rate": _div(fn, tp + fn),  # Change not detected.
        "precision": _div(tp, tp + fp),  # Share of flagged change that is real.
        "f1": _div(2 * tp, 2 * tp + fp + fn),  # Harmonic mean.
        "iou": _div(tp, tp + fp + fn),  # Jaccard index of change.
        "overall_accuracy": _div(tp + tn, total),  # Share of correct pixels.
        "kappa": cohen_kappa(matrix) if total else float("nan"),  # Chance-corrected agreement.
    }  # End of the report.


# Pixels whose class differs between two dates.
def change_map(
    before: NDArray[Any],  # Class map of the first date.
    after: NDArray[Any],  # Class map of the second date.
    nodata: Any | None = None,  # Label of missing pixels on either date.
) -> NDArray[np.bool_]:  # True where the class changed.
    # Maps as arrays.
    b, a = np.asarray(before), np.asarray(after)
    # Shapes must agree.
    if b.shape != a.shape:
        # Explain the requirement.
        raise ValueError(f"shape mismatch: {b.shape} and {a.shape}")
    # Pixels with different classes.
    changed = b != a
    # Missing pixels are never change.
    if nodata is not None:
        # Remove them.
        changed &= (b != nodata) & (a != nodata)
    # Return the mask.
    return changed


# From-to counts of classes; rows are the first date, columns the second.
def transition_matrix(
    before: NDArray[Any],  # Class map of the first date.
    after: NDArray[Any],  # Class map of the second date.
    labels: Sequence[Any] | None = None,  # Class order.
    nodata: Any | None = None,  # Label of missing pixels.
) -> NDArray[Any]:  # (K, K) counts.
    # The error matrix code counts label pairs in the same way.
    return confusion_matrix(before, after, labels=labels, ignore=nodata)


# Gross gains, losses, net change and swap of every class.
def transition_summary(
    matrix: NDArray[Any],  # Transition counts, rows first date.
    classes: Sequence[Any] | None = None,  # Class names.
) -> dict[str, dict[str, float]]:  # Per-class components.
    # Counts as float.
    m = np.asarray(matrix, dtype=np.float64)
    # The matrix must be square.
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        # Explain the requirement.
        raise ValueError(f"expected a square matrix, got shape {m.shape}")
    # Class names.
    names = [str(c) for c in classes] if classes is not None else [str(i) for i in range(len(m))]
    # Persistence on the diagonal.
    persist = np.diag(m)
    # Gross loss: first-date area that changed to another class.
    loss = m.sum(axis=1) - persist
    # Gross gain: second-date area that came from another class.
    gain = m.sum(axis=0) - persist
    # Report.
    return {
        name: {  # Components of one class.
            "persistence": float(persist[i]),  # Unchanged.
            "gain": float(gain[i]),  # Gross gain.
            "loss": float(loss[i]),  # Gross loss.
            "net_change": float(gain[i] - loss[i]),  # Gain minus loss.
            "swap": float(2 * min(gain[i], loss[i])),  # Simultaneous gain and loss.
            "total_change": float(gain[i] + loss[i]),  # Net change plus swap.
        }  # End of the class record.
        for i, name in enumerate(names)  # Every class.
    }  # End of the report.


# =============================================================================
# End of module src/unbihexium/metrics/change.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
