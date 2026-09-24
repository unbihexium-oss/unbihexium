# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/analysis/suitability.py
# Title       : Multi-criteria suitability analysis: AHP and weighted overlay
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# GIS multi-criteria evaluation in three steps (Malczewski, 2004):
#
#   1. standardise the factor layers to a common [0, 1] scale:
#      rescale_linear, fuzzy_membership (linear or sine-squared sigmoidal),
#      reclassify
#   2. derive criterion weights from pairwise comparisons with the
#      Analytic Hierarchy Process (AHP, Saaty 1977 and 1980)
#   3. combine the layers by weighted linear combination and exclude cells
#      with Boolean constraints: WeightedOverlay, weighted_overlay
#
# AHP
# ---
# A pairwise comparison matrix A (a_ij: importance of criterion i over j on
# the 1 to 9 scale, a_ji = 1 / a_ij) is summarised by its principal
# eigenvector w (normalised to sum 1) and eigenvalue lambda_max, or by the
# row geometric means (Crawford and Williams, 1985). The consistency index
# CI = (lambda_max - n) / (n - 1) is compared with the random index RI of
# random reciprocal matrices of the same size: CR = CI / RI. Judgements
# with CR < 0.1 are usually accepted (Saaty, 1980).
#
# References
# ----------
# Saaty, T. L. (1977). A scaling method for priorities in hierarchical
#   structures. Journal of Mathematical Psychology, 15(3), 234-281.
# Saaty, T. L. (1980). The Analytic Hierarchy Process. McGraw-Hill, New
#   York.
# Crawford, G., Williams, C. (1985). A note on the analysis of subjective
#   judgment matrices. Journal of Mathematical Psychology, 29(4), 387-405.
# Eastman, J. R. (2009). IDRISI Taiga Guide to GIS and Image Processing.
#   Clark Labs, Clark University, Worcester MA.
# Malczewski, J. (2004). GIS-based land-use suitability analysis: a critical
#   overview. Progress in Planning, 62(1), 3-65.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Result record.
from dataclasses import dataclass, field

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Random index of Saaty (1980) by matrix size.
RANDOM_INDEX = {
    1: 0.0,  # One criterion.
    2: 0.0,  # Two criteria are always consistent.
    3: 0.58,  # Three criteria.
    4: 0.90,  # Four criteria.
    5: 1.12,  # Five criteria.
    6: 1.24,  # Six criteria.
    7: 1.32,  # Seven criteria.
    8: 1.41,  # Eight criteria.
    9: 1.45,  # Nine criteria.
    10: 1.49,  # Ten criteria.
    11: 1.51,  # Eleven criteria.
    12: 1.48,  # Twelve criteria.
    13: 1.56,  # Thirteen criteria.
    14: 1.57,  # Fourteen criteria.
    15: 1.59,  # Fifteen criteria.
}  # End of the random index.


# Result of a weighted overlay.
@dataclass
class SuitabilityResult:
    # Suitability surface.
    suitability: NDArray[np.floating[Any]]
    # Normalised weight of each layer by name.
    weights: dict[str, float] = field(default_factory=dict)
    # AHP consistency ratio, when the weights came from AHP.
    consistency_ratio: float | None = None
    # Georeferenced raster of the surface, when the inputs were rasters.
    raster: Any = None


# Analytic Hierarchy Process weights from a pairwise comparison matrix.
class AHP:
    # Create the model for named criteria.
    def __init__(
        self,  # The instance.
        criteria: list[str] | None = None,  # Criterion names in matrix order.
        method: str = "eigenvector",  # "eigenvector" or "geometric_mean".
    ) -> None:  # The constructor returns nothing.
        # The method must be known.
        if method not in ("eigenvector", "geometric_mean"):
            # Report the invalid method.
            raise ValueError(f"unknown method {method!r}; expected eigenvector or geometric_mean")
        # Criterion names.
        self.criteria = list(criteria) if criteria is not None else []
        # Weighting method.
        self.method = method
        # Comparison matrix.
        self._matrix: NDArray[np.float64] | None = None
        # Weights.
        self._weights: NDArray[np.float64] | None = None
        # Principal eigenvalue (or its estimate).
        self._lambda_max: float | None = None

    # Model from judgements {(criterion a, criterion b): importance of a over b}.
    @classmethod
    def from_judgements(
        cls,  # The class.
        criteria: list[str],  # Criterion names.
        judgements: dict[tuple[str, str], float],  # One judgement per pair.
        method: str = "eigenvector",  # Weighting method.
    ) -> AHP:  # The model with its weights.
        # Index of every criterion.
        index = {name: i for i, name in enumerate(criteria)}
        # Identity matrix to start.
        matrix = np.eye(len(criteria))
        # Fill every judgement and its reciprocal.
        for (a, b), value in judgements.items():
            # Both criteria must be known.
            if a not in index or b not in index:
                # Report the unknown name.
                raise ValueError(f"unknown criterion in judgement {(a, b)}")
            # Judgements must be positive.
            if value <= 0:
                # Report the invalid value.
                raise ValueError(f"judgement {(a, b)} must be positive, got {value}")
            # Importance of a over b.
            matrix[index[a], index[b]] = value
            # Reciprocal.
            matrix[index[b], index[a]] = 1.0 / value
        # Model with the matrix.
        return cls(criteria, method).set_comparison_matrix(matrix)

    # Set and validate the comparison matrix, then compute the weights.
    def set_comparison_matrix(self, matrix: NDArray[Any]) -> AHP:
        # Float64 matrix.
        a = np.asarray(matrix, dtype=np.float64)
        # Square matrix.
        if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
            # Report the wrong shape.
            raise ValueError(f"comparison matrix must be square, got shape {a.shape}")
        # Positive entries.
        if not np.all(np.isfinite(a)) or np.any(a <= 0):
            # Report the invalid entries.
            raise ValueError("comparison matrix entries must be positive and finite")
        # Reciprocal within 2 % to tolerate rounded fractions such as 0.33.
        if not np.allclose(a * a.T, 1.0, rtol=0.02):
            # Report the violation.
            raise ValueError("comparison matrix must be reciprocal: a_ji = 1 / a_ij")
        # Size must match the criteria.
        if self.criteria and len(self.criteria) != a.shape[0]:
            # Report the mismatch.
            raise ValueError(
                f"{len(self.criteria)} criteria but a {a.shape[0]} x {a.shape[0]} matrix"
            )
        # Default criterion names.
        if not self.criteria:
            # c1, c2, ...
            self.criteria = [f"c{i + 1}" for i in range(a.shape[0])]
        # Store the matrix.
        self._matrix = a
        # Compute the weights.
        self._compute()
        # Allow chaining.
        return self

    # Alias of set_comparison_matrix.
    def fit(self, comparison_matrix: NDArray[Any]) -> AHP:
        # Same as setting the matrix.
        return self.set_comparison_matrix(comparison_matrix)

    # Compute the weights and lambda_max.
    def _compute(self) -> None:
        # The matrix is set.
        a = self._matrix
        # Guard for type checkers.
        assert a is not None
        # Principal eigenvector.
        if self.method == "eigenvector":
            # Eigen-decomposition of the positive matrix.
            values, vectors = np.linalg.eig(a)
            # Perron root: the eigenvalue with the largest real part.
            k = int(np.argmax(np.real(values)))
            # Its eigenvector has entries of one sign.
            w = np.abs(np.real(vectors[:, k]))
            # Principal eigenvalue.
            self._lambda_max = float(values[k].real)
        # Row geometric means.
        else:
            # n-th root of the row products.
            w = np.exp(np.mean(np.log(a), axis=1))
            # Estimate of lambda_max from A w = lambda w.
            self._lambda_max = float(np.mean((a @ w) / w))
        # Normalise to sum one.
        self._weights = w / w.sum()

    # Criterion weights summing to one.
    def calculate_weights(self) -> NDArray[np.float64]:
        # The matrix must be set.
        if self._weights is None:
            # Report the missing matrix.
            raise RuntimeError("set the comparison matrix first")
        # Copy of the weights.
        return self._weights.copy()

    # Weights, or None before the matrix is set.
    @property
    def weights(self) -> NDArray[np.float64] | None:
        # Stored weights.
        return None if self._weights is None else self._weights.copy()

    # Weights by criterion name.
    def weights_dict(self) -> dict[str, float]:
        # Pair names and weights.
        return dict(zip(self.criteria, (float(v) for v in self.calculate_weights())))

    # Principal eigenvalue (or its estimate for the geometric mean method).
    def lambda_max(self) -> float:
        # The matrix must be set.
        if self._lambda_max is None:
            # Report the missing matrix.
            raise RuntimeError("set the comparison matrix first")
        # Stored value.
        return self._lambda_max

    # Consistency index (lambda_max - n) / (n - 1).
    def consistency_index(self) -> float:
        # Matrix size.
        n = len(self.criteria)
        # One or two criteria are consistent by construction.
        if n <= 2:
            # Zero index.
            return 0.0
        # Saaty's index, never negative.
        return max(0.0, (self.lambda_max() - n) / (n - 1))

    # Consistency ratio CI / RI.
    def consistency_ratio(self) -> float:
        # Matrix size.
        n = len(self.criteria)
        # Random index of the size.
        ri = RANDOM_INDEX.get(n)
        # Sizes beyond the table.
        if ri is None:
            # Report the unsupported size.
            raise ValueError(f"no random index for {n} criteria (supported: 1 to 15)")
        # Zero for one or two criteria.
        return 0.0 if ri == 0 else self.consistency_index() / ri

    # Whether the judgements pass the consistency threshold.
    def is_consistent(self, threshold: float = 0.1) -> bool:
        # Compare the ratio with the threshold.
        return self._weights is not None and self.consistency_ratio() < threshold


# Linear rescaling to [0, 1] between low and high.
def rescale_linear(
    values: NDArray[Any],  # Factor layer.
    low: float | None = None,  # Value mapped to 0; the minimum by default.
    high: float | None = None,  # Value mapped to 1; the maximum by default.
    increasing: bool = True,  # Whether larger values are more suitable.
) -> NDArray[np.float64]:  # Standardised layer, NaN kept.
    # Float64 layer.
    x = np.asarray(values, dtype=np.float64)
    # Lower control point.
    lo = float(np.nanmin(x)) if low is None else float(low)
    # Upper control point.
    hi = float(np.nanmax(x)) if high is None else float(high)
    # A constant range cannot be rescaled.
    if hi == lo:
        # Report the degenerate range.
        raise ValueError("low and high must differ")
    # Position between the control points, clipped.
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    # Reverse for decreasing factors.
    return t if increasing else 1.0 - t


# Fuzzy membership between control points a (0) and b (1) (Eastman, 2009).
def fuzzy_membership(
    values: NDArray[Any],  # Factor layer.
    a: float,  # Value with membership 0.
    b: float,  # Value with membership 1.
    shape: str = "sigmoidal",  # "linear" or "sigmoidal" (sine squared).
) -> NDArray[np.float64]:  # Membership in [0, 1]; decreasing when a > b.
    # Linear position between the control points (a > b gives a decreasing function).
    t = rescale_linear(values, a, b) if a < b else 1.0 - rescale_linear(values, b, a)
    # Linear membership.
    if shape == "linear":
        # Position itself.
        return t
    # Sigmoidal membership.
    if shape == "sigmoidal":
        # sin^2 of the position times pi / 2.
        return np.sin(t * np.pi / 2.0) ** 2
    # Unknown shape.
    raise ValueError(f"unknown shape {shape!r}; expected linear or sigmoidal")


# Map value ranges to scores: [breaks[i-1], breaks[i]) -> scores[i].
def reclassify(
    values: NDArray[Any],  # Layer.
    breaks: list[float],  # Increasing class limits (k).
    scores: list[float],  # Score of each class (k + 1).
) -> NDArray[np.float64]:  # Reclassified layer, NaN kept.
    # Float64 layer.
    x = np.asarray(values, dtype=np.float64)
    # One more score than breaks.
    if len(scores) != len(breaks) + 1:
        # Report the mismatch.
        raise ValueError("scores must have one entry more than breaks")
    # Breaks must increase.
    if np.any(np.diff(breaks) <= 0):
        # Report the order problem.
        raise ValueError("breaks must be strictly increasing")
    # Class of every value.
    classes = np.digitize(x, breaks)
    # Score of the class; NaN stays NaN.
    return np.where(np.isnan(x), np.nan, np.asarray(scores, dtype=np.float64)[classes])


# Min-max rescaling of a layer; a constant layer is fully suitable (1).
def _rescale_or_one(layer: NDArray[np.float64]) -> NDArray[np.float64]:
    # Constant layers cannot be rescaled.
    if not np.nanmax(layer) > np.nanmin(layer):
        # Ones, keeping NaN.
        return np.where(np.isnan(layer), np.nan, 1.0)
    # Linear rescaling to [0, 1].
    return rescale_linear(layer)


# Weighted linear combination of factor layers with Boolean constraints.
class WeightedOverlay:
    # Configure the overlay.
    def __init__(self, rescale: bool = False) -> None:
        # Rescale each layer to [0, 1] with its own range first.
        self.rescale = rescale

    # Combine the layers.
    def calculate(
        self,  # The instance.
        layers: list[NDArray[Any]],  # Factor layers of one shape.
        weights: list[float] | NDArray[Any],  # One non-negative weight per layer.
        constraints: list[NDArray[Any]] | None = None,  # Masks; False excludes a cell (score 0).
    ) -> NDArray[np.float64]:  # Suitability, NaN where any layer is NaN.
        # At least one layer.
        if not layers:
            # Report the missing layers.
            raise ValueError("at least one layer is required")
        # One weight per layer.
        w = np.asarray(weights, dtype=np.float64)
        # Check the count.
        if w.shape != (len(layers),):
            # Report the mismatch.
            raise ValueError(f"{len(layers)} layers but {w.size} weights")
        # Weights must be non-negative with a positive sum.
        if np.any(w < 0) or w.sum() <= 0:
            # Report the invalid weights.
            raise ValueError("weights must be non-negative with a positive sum")
        # Normalise the weights.
        w = w / w.sum()
        # Layers as float64 arrays.
        stack = [np.asarray(layer, dtype=np.float64) for layer in layers]
        # All layers share a shape.
        if any(s.shape != stack[0].shape for s in stack):
            # Report the mismatch.
            raise ValueError("all layers must have the same shape")
        # Optional per-layer rescaling.
        if self.rescale:
            # Min-max rescaling of each layer.
            stack = [_rescale_or_one(s) for s in stack]
        # Weighted sum.
        result = np.zeros(stack[0].shape)
        # Add every layer.
        for weight, layer in zip(w, stack):
            # Weighted contribution.
            result = result + weight * layer
        # Apply the constraints.
        for mask in constraints or []:
            # Boolean mask of the same shape.
            m = np.asarray(mask, dtype=bool)
            # Check the shape.
            if m.shape != result.shape:
                # Report the mismatch.
                raise ValueError("constraint masks must have the layer shape")
            # Excluded cells score zero; NaN stays NaN.
            result = np.where(m | np.isnan(result), result, 0.0)
        # Return the surface.
        return result


# Weighted overlay of arrays or rasters, returning weights and a raster.
def weighted_overlay(
    layers: list[Any],  # Arrays or Raster objects (first band used).
    weights: list[float] | NDArray[Any],  # One weight per layer.
    normalize: bool = True,  # Rescale each layer to [0, 1] first.
    constraints: list[NDArray[Any]] | None = None,  # Boolean masks.
    names: list[str] | None = None,  # Layer names for the weight table.
) -> SuitabilityResult:  # Surface, weights and raster.
    # Arrays of the layers; rasters contribute their first band.
    arrays = [_layer_array(layer) for layer in layers]
    # Combine.
    surface = WeightedOverlay(rescale=normalize).calculate(arrays, weights, constraints)
    # Normalised weights.
    w = np.asarray(weights, dtype=np.float64) / float(np.sum(weights))
    # Layer names.
    labels = names if names is not None else [f"layer_{i}" for i in range(len(layers))]
    # Georeferenced output when the first layer is a raster.
    raster = _as_raster(surface, layers[0])
    # Package the result.
    return SuitabilityResult(surface, dict(zip(labels, (float(v) for v in w))), None, raster)


# First band of a raster or an array as float64.
def _layer_array(layer: Any) -> NDArray[np.float64]:
    # Raster objects expose their pixels as .data.
    data = getattr(layer, "data", layer)
    # Float64 array.
    a = np.asarray(data, dtype=np.float64)
    # First band of (bands, rows, cols) data.
    return a[0] if a.ndim == 3 else a


# Raster of a surface with the georeferencing of a template raster, or None.
def _as_raster(surface: NDArray[np.float64], template: Any) -> Any:
    # Plain arrays have no georeferencing.
    metadata = getattr(template, "metadata", None)
    # Nothing to copy.
    if metadata is None:
        # No raster.
        return None
    # Raster class of the library.
    from unbihexium.core.raster import Raster

    # Surface with the template's CRS and transform.
    return Raster.from_array(surface, crs=metadata.crs, transform=metadata.transform)


# =============================================================================
# End of module src/unbihexium/analysis/suitability.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
