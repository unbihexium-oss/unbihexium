# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/metrics/classification.py
# Title       : Thematic accuracy assessment of classified maps
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Error matrix based accuracy measures of land cover and other class maps:
#
#   confusion_matrix       counts (or weights) of reference/map label pairs
#   accuracy_assessment    overall, producer's and user's accuracy,
#                          omission and commission errors, F1, IoU, Cohen's
#                          kappa, quantity and allocation disagreement
#   AccuracyAssessment     result record with to_dict()
#   cohen_kappa            kappa of an error matrix
#   iou, dice, precision, recall, f1_score, accuracy, mean_iou
#                          measures of binary masks and label arrays
#
# Convention: every matrix of unbihexium.metrics has the REFERENCE classes
# in its rows and the MAP (predicted) classes in its columns, as in
# unbihexium.ai.evaluation and scikit-learn. Producer's accuracy is the
# diagonal over the row totals, user's accuracy the diagonal over the
# column totals.
#
# The measures accept an error matrix of counts or of estimated population
# proportions, for example the area-weighted matrix of
# unbihexium.metrics.area.estimated_error_matrix; the latter gives unbiased
# accuracies under stratified random sampling (Olofsson et al., 2014).
#
# Method
# ------
# With proportions p_ij (reference i, map j), p_i+ and p_+j the row and
# column totals: overall accuracy OA = sum_i p_ii; kappa
# (OA - p_e) / (1 - p_e) with p_e = sum_i p_i+ p_+i (Cohen, 1960); quantity
# disagreement Q = sum_g |p_g+ - p_+g| / 2 and allocation disagreement
# A = sum_g 2 min(p_g+ - p_gg, p_+g - p_gg) / 2, with Q + A = 1 - OA
# (Pontius and Millones, 2011).
#
# References
# ----------
#   Congalton, R. G. (1991). A review of assessing the accuracy of
#     classifications of remotely sensed data. Remote Sensing of Environment
#     37(1), 35-46.
#   Cohen, J. (1960). A coefficient of agreement for nominal scales.
#     Educational and Psychological Measurement 20(1), 37-46.
#   Pontius, R. G., Millones, M. (2011). Death to kappa: birth of quantity
#     disagreement and allocation disagreement for accuracy assessment.
#     International Journal of Remote Sensing 32(15), 4407-4429.
#   Olofsson, P., et al. (2014). Good practices for estimating area and
#     assessing accuracy of land change. Remote Sensing of Environment 148,
#     42-57.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Result records.
from dataclasses import dataclass

# Type of loosely structured values and sequences.
from typing import Any, Sequence

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray


# Safe element-wise division that returns NaN for zero denominators.
def _ratio(num: NDArray[Any], den: NDArray[Any]) -> NDArray[np.float64]:
    # Numerator as float.
    n = np.asarray(num, dtype=np.float64)
    # Denominator as float.
    d = np.asarray(den, dtype=np.float64)
    # NaN where the denominator is zero.
    return np.divide(n, d, out=np.full(np.broadcast(n, d).shape, np.nan), where=d != 0)


# Validate a square error matrix.
def _square(matrix: NDArray[Any]) -> NDArray[np.float64]:
    # Matrix as float.
    m = np.asarray(matrix, dtype=np.float64)
    # It must be square and non-empty.
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] == 0:
        # Explain the requirement.
        raise ValueError(f"expected a square error matrix, got shape {m.shape}")
    # Entries must be finite and non-negative.
    if not np.isfinite(m).all() or (m < 0).any():
        # Explain the requirement.
        raise ValueError("error matrix entries must be finite and non-negative")
    # Some entry must be positive.
    if m.sum() == 0:
        # Explain the requirement.
        raise ValueError("error matrix is empty")
    # Return the matrix.
    return m


# Error matrix of reference and map labels; rows are reference classes.
def confusion_matrix(
    reference: NDArray[Any],  # Reference labels of any shape.
    predicted: NDArray[Any],  # Map labels of the same shape.
    labels: Sequence[Any] | None = None,  # Class order; None uses the sorted labels found.
    ignore: Any | None = None,  # Reference or map label that is not evaluated.
    weights: NDArray[Any] | None = None,  # Optional weight of every pixel or sample.
) -> NDArray[Any]:  # (K, K) counts, int64 unless weights are given.
    # Flatten the reference.
    r = np.asarray(reference).reshape(-1)
    # Flatten the map.
    p = np.asarray(predicted).reshape(-1)
    # The arrays must pair up.
    if r.shape != p.shape:
        # Explain the requirement.
        raise ValueError("reference and predicted must have the same number of elements")
    # Pairs that are evaluated.
    keep = np.ones(r.shape, dtype=bool)
    # Drop the ignored label on either side.
    if ignore is not None:
        # Ignored pairs.
        keep &= (r != ignore) & (p != ignore)
    # Floating point labels may carry NaN.
    if r.dtype.kind == "f" or p.dtype.kind == "f":
        # NaN pairs are not evaluated.
        keep &= ~(np.isnan(r.astype(float)) | np.isnan(p.astype(float)))
    # Class order.
    classes = np.unique(np.concatenate([r[keep], p[keep]])) if labels is None else labels
    # Classes as an array.
    classes = np.asarray(classes)
    # At least one class is needed.
    if classes.size == 0:
        # Explain the problem.
        raise ValueError("no labels to evaluate")
    # Position of every class in the order.
    order = np.argsort(classes)
    # Sorted classes for the lookup.
    sorted_classes = classes[order]
    # Index of the reference labels in the sorted classes.
    ri = np.searchsorted(sorted_classes, r).clip(0, len(classes) - 1)
    # Index of the map labels.
    pi = np.searchsorted(sorted_classes, p).clip(0, len(classes) - 1)
    # Labels outside the class list are not counted.
    keep &= (sorted_classes[ri] == r) & (sorted_classes[pi] == p)
    # Number of classes.
    k = len(classes)
    # Pair index in the requested class order.
    pair = order[ri[keep]] * k + order[pi[keep]]
    # Weights of the kept pairs.
    w = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)[keep]
    # Count the pairs.
    counts = np.bincount(pair, weights=w, minlength=k * k)
    # Back to a matrix.
    return counts.reshape(k, k)


# Cohen's kappa of an error matrix.
def cohen_kappa(matrix: NDArray[Any]) -> float:
    # Proportions.
    m = _square(matrix)
    # Normalise to unit sum.
    p = m / m.sum()
    # Observed agreement.
    observed = float(np.trace(p))
    # Agreement expected by chance.
    expected = float((p.sum(axis=1) * p.sum(axis=0)).sum())
    # Undefined when chance agreement is perfect.
    if expected >= 1.0:
        # Not a number.
        return float("nan")
    # Kappa.
    return (observed - expected) / (1.0 - expected)


# Accuracy measures of an error matrix.
@dataclass(frozen=True)
class AccuracyAssessment:
    # Class names in matrix order.
    classes: list[str]
    # Proportion of correctly classified units.
    overall_accuracy: float
    # Producer's accuracy (recall) per class.
    producers_accuracy: NDArray[np.float64]
    # User's accuracy (precision) per class.
    users_accuracy: NDArray[np.float64]
    # F1 score per class.
    f1: NDArray[np.float64]
    # Intersection over union per class.
    iou: NDArray[np.float64]
    # Cohen's kappa.
    kappa: float
    # Quantity disagreement of Pontius and Millones (2011).
    quantity_disagreement: float
    # Allocation disagreement of Pontius and Millones (2011).
    allocation_disagreement: float
    # Total of the matrix (units or area).
    total: float

    # Omission error per class, one minus producer's accuracy.
    @property
    def omission_error(self) -> NDArray[np.float64]:
        # Complement of the producer's accuracy.
        return 1.0 - self.producers_accuracy

    # Commission error per class, one minus user's accuracy.
    @property
    def commission_error(self) -> NDArray[np.float64]:
        # Complement of the user's accuracy.
        return 1.0 - self.users_accuracy

    # Mean IoU over classes with a defined value.
    @property
    def mean_iou(self) -> float:
        # Defined values.
        v = self.iou[np.isfinite(self.iou)]
        # NaN when no class is defined.
        return float(v.mean()) if v.size else float("nan")

    # Plain dictionary for reports and JSON.
    def to_dict(self) -> dict[str, Any]:
        # Map per-class arrays to names.
        def per_class(values: NDArray[np.float64]) -> dict[str, float]:
            # Class name to value.
            return dict(zip(self.classes, (float(v) for v in values)))

        # Report.
        return {
            "overall_accuracy": self.overall_accuracy,  # Overall accuracy.
            "kappa": self.kappa,  # Cohen's kappa.
            "quantity_disagreement": self.quantity_disagreement,  # Quantity part.
            "allocation_disagreement": self.allocation_disagreement,  # Allocation part.
            "mean_iou": self.mean_iou,  # Mean IoU.
            "producers_accuracy": per_class(self.producers_accuracy),  # Recall.
            "users_accuracy": per_class(self.users_accuracy),  # Precision.
            "omission_error": per_class(self.omission_error),  # Missed reference.
            "commission_error": per_class(self.commission_error),  # Wrongly mapped.
            "f1": per_class(self.f1),  # F1 score.
            "iou": per_class(self.iou),  # Intersection over union.
            "total": self.total,  # Units or area.
        }  # End of the report.


# Accuracy measures of an error matrix (rows reference, columns map).
def accuracy_assessment(
    matrix: NDArray[Any],  # Counts or proportions.
    classes: Sequence[Any] | None = None,  # Class names in matrix order.
) -> AccuracyAssessment:  # Result record.
    # Validated matrix.
    m = _square(matrix)
    # Number of classes.
    k = m.shape[0]
    # Class names.
    names = [str(c) for c in classes] if classes is not None else [str(i) for i in range(k)]
    # One name per class.
    if len(names) != k:
        # Explain the requirement.
        raise ValueError(f"expected {k} class names, got {len(names)}")
    # Total of the matrix.
    total = float(m.sum())
    # Proportions.
    p = m / total
    # Diagonal.
    diag = np.diag(p)
    # Reference totals (rows).
    ref = p.sum(axis=1)
    # Map totals (columns).
    mapped = p.sum(axis=0)
    # Quantity disagreement.
    quantity = float(np.abs(ref - mapped).sum() / 2.0)
    # Allocation disagreement.
    allocation = float((2.0 * np.minimum(ref - diag, mapped - diag)).sum() / 2.0)
    # Build the record.
    return AccuracyAssessment(
        classes=names,  # Class names.
        overall_accuracy=float(diag.sum()),  # Overall accuracy.
        producers_accuracy=_ratio(diag, ref),  # Diagonal over reference totals.
        users_accuracy=_ratio(diag, mapped),  # Diagonal over map totals.
        f1=_ratio(2.0 * diag, ref + mapped),  # Harmonic mean of both accuracies.
        iou=_ratio(diag, ref + mapped - diag),  # Jaccard index.
        kappa=cohen_kappa(m),  # Cohen's kappa.
        quantity_disagreement=quantity,  # Quantity part.
        allocation_disagreement=allocation,  # Allocation part.
        total=total,  # Units or area.
    )  # End of the record.


# Binary arrays of a prediction and a target.
def _binary(pred: NDArray[Any], target: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    # Prediction as boolean.
    p = np.asarray(pred) > 0
    # Target as boolean.
    t = np.asarray(target) > 0
    # Shapes must agree.
    if p.shape != t.shape:
        # Explain the requirement.
        raise ValueError(f"shape mismatch: {p.shape} and {t.shape}")
    # Return both.
    return p, t


# Intersection over union of binary masks.
def iou(pred: NDArray[Any], target: NDArray[Any], smooth: float = 1e-6) -> float:
    # Binary masks.
    p, t = _binary(pred, target)
    # Pixels in both.
    inter = float(np.sum(p & t))
    # Pixels in either.
    union = float(np.sum(p | t))
    # Smoothed ratio; two empty masks agree perfectly.
    return (inter + smooth) / (union + smooth)


# Dice coefficient (F1) of binary masks.
def dice(pred: NDArray[Any], target: NDArray[Any], smooth: float = 1e-6) -> float:
    # Binary masks.
    p, t = _binary(pred, target)
    # Pixels in both.
    inter = float(np.sum(p & t))
    # Smoothed ratio.
    return (2.0 * inter + smooth) / (float(p.sum() + t.sum()) + smooth)


# Precision of a binary prediction.
def precision(pred: NDArray[Any], target: NDArray[Any]) -> float:
    # Binary masks.
    p, t = _binary(pred, target)
    # True positives.
    tp = float(np.sum(p & t))
    # Predicted positives.
    pp = float(np.sum(p))
    # NaN without predicted positives.
    return tp / pp if pp else float("nan")


# Recall of a binary prediction.
def recall(pred: NDArray[Any], target: NDArray[Any]) -> float:
    # Binary masks.
    p, t = _binary(pred, target)
    # True positives.
    tp = float(np.sum(p & t))
    # Actual positives.
    ap = float(np.sum(t))
    # NaN without actual positives.
    return tp / ap if ap else float("nan")


# F1 score of a binary prediction.
def f1_score(pred: NDArray[Any], target: NDArray[Any]) -> float:
    # Binary masks.
    p, t = _binary(pred, target)
    # True positives.
    tp = float(np.sum(p & t))
    # Predicted plus actual positives.
    den = float(p.sum() + t.sum())
    # NaN when both are empty.
    return 2.0 * tp / den if den else float("nan")


# Fraction of equal labels.
def accuracy(pred: NDArray[Any], target: NDArray[Any]) -> float:
    # Labels as arrays.
    p, t = np.asarray(pred), np.asarray(target)
    # Shapes must agree.
    if p.shape != t.shape:
        # Explain the requirement.
        raise ValueError(f"shape mismatch: {p.shape} and {t.shape}")
    # Mean of the agreement.
    return float(np.mean(p == t))


# Mean IoU over the classes present in the target.
def mean_iou(pred: NDArray[Any], target: NDArray[Any], num_classes: int) -> float:
    # Error matrix over the classes 0 .. num_classes - 1.
    m = confusion_matrix(target, pred, labels=list(range(num_classes)))
    # True positives.
    tp = np.diag(m).astype(np.float64)
    # Reference and map totals.
    ref, mapped = m.sum(axis=1), m.sum(axis=0)
    # IoU per class.
    per_class = _ratio(tp, ref + mapped - tp)
    # Classes present in the target.
    present = ref > 0
    # Mean over present classes, zero when none is present.
    return float(per_class[present].mean()) if present.any() else 0.0


# =============================================================================
# End of module src/unbihexium/metrics/classification.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
