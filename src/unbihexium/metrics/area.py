# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/metrics/area.py
# Title       : Unbiased area estimation and accuracy under stratified sampling
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Good-practice area estimation for maps assessed with a stratified random
# sample whose strata are the map classes:
#
#   estimated_error_matrix   population proportions p_ij from sample counts
#                            and mapped areas
#   stratified_area_estimate unbiased class areas, overall, user's and
#                            producer's accuracies with standard errors and
#                            confidence intervals
#   AreaEstimate             result record with to_dict()
#   sample_allocation        stratum sample sizes for a target standard
#                            error of overall accuracy
#
# Pixel counting of a map is biased by its classification errors; the
# sample-based estimator below corrects the mapped areas with the reference
# labels of the sample (Olofsson et al., 2013, 2014).
#
# Method
# ------
# Following the convention of unbihexium.metrics, the input matrix n has
# reference classes in rows and map classes (strata) in columns; below,
# n_ij denotes the count with MAP class i and REFERENCE class j (the
# transpose), as in Olofsson et al. (2014). With W_i = A_m,i / A_tot the
# mapped area proportion and n_i. the sample size of stratum i:
#
#   p_ij = W_i n_ij / n_i.
#   p_.j = sum_i p_ij,  A_j = A_tot p_.j
#   S(p_.j) = sqrt(sum_i W_i^2 (n_ij/n_i.)(1 - n_ij/n_i.) / (n_i. - 1))
#   O = sum_j p_jj,  V(O) = sum_i W_i^2 U_i (1 - U_i) / (n_i. - 1)
#   U_i = n_ii / n_i.,  V(U_i) = U_i (1 - U_i) / (n_i. - 1)
#   P_j = p_jj / p_.j, with the variance of the producer's accuracy of
#   Olofsson et al. (2014) and N_i. the mapped areas
#
# Confidence intervals are +/- z S with z the standard normal quantile.
#
# References
# ----------
#   Olofsson, P., Foody, G. M., Stehman, S. V., Woodcock, C. E. (2013).
#     Making better use of accuracy data in land change studies: estimating
#     accuracy and area and quantifying uncertainty using stratified
#     estimation. Remote Sensing of Environment 129, 122-131.
#   Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E.,
#     Wulder, M. A. (2014). Good practices for estimating area and assessing
#     accuracy of land change. Remote Sensing of Environment 148, 42-57.
#   Cochran, W. G. (1977). Sampling Techniques, 3rd ed. Wiley.
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

# Quantiles of the standard normal distribution.
from scipy.stats import norm


# Validate sample counts and mapped areas; return counts with map classes in rows.
def _inputs(
    matrix: NDArray[Any],  # Counts, rows reference, columns map.
    mapped_area: Sequence[float] | NDArray[Any],  # Mapped area of every map class.
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # (map x reference) counts, areas.
    # Counts as float.
    m = np.asarray(matrix, dtype=np.float64)
    # The matrix must be square.
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        # Explain the requirement.
        raise ValueError(f"expected a square matrix of counts, got shape {m.shape}")
    # Counts must be non-negative.
    if (m < 0).any() or not np.isfinite(m).all():
        # Explain the requirement.
        raise ValueError("sample counts must be finite and non-negative")
    # Areas as float.
    a = np.asarray(mapped_area, dtype=np.float64).reshape(-1)
    # One area per map class.
    if a.size != m.shape[0] or (a < 0).any() or a.sum() <= 0:
        # Explain the requirement.
        raise ValueError(f"expected {m.shape[0]} non-negative mapped areas with a positive sum")
    # Map classes in rows, as in Olofsson et al. (2014).
    n = m.T
    # Sample size of every stratum.
    ni = n.sum(axis=1)
    # Every stratum with mapped area needs at least two samples for a variance.
    if ((a > 0) & (ni < 2)).any():
        # Explain the requirement.
        raise ValueError("every mapped class needs at least two samples")
    # Return the counts and areas.
    return n, a


# Estimated population error matrix of proportions (rows reference, columns map).
def estimated_error_matrix(
    matrix: NDArray[Any],  # Sample counts, rows reference, columns map.
    mapped_area: Sequence[float] | NDArray[Any],  # Mapped area of every map class.
) -> NDArray[np.float64]:  # Proportions p summing to one, rows reference.
    # Validated inputs with map classes in rows.
    n, a = _inputs(matrix, mapped_area)
    # Stratum weights.
    w = a / a.sum()
    # Stratum sample sizes, one where empty to avoid division by zero.
    ni = np.maximum(n.sum(axis=1), 1.0)
    # p_ij = W_i n_ij / n_i. with map class i in rows.
    p = w[:, None] * n / ni[:, None]
    # Back to rows reference, columns map.
    return p.T


# Unbiased area and accuracy estimates with uncertainty.
@dataclass(frozen=True)
class AreaEstimate:
    # Class names.
    classes: list[str]
    # Estimated area proportion of every class (reference).
    proportion: NDArray[np.float64]
    # Standard error of the proportions.
    proportion_se: NDArray[np.float64]
    # Estimated area of every class, in the units of the mapped areas.
    area: NDArray[np.float64]
    # Standard error of the areas.
    area_se: NDArray[np.float64]
    # Half-width of the confidence interval of the areas.
    area_ci: NDArray[np.float64]
    # Mapped (pixel counting) area of every class.
    mapped_area: NDArray[np.float64]
    # Overall accuracy.
    overall_accuracy: float
    # Standard error of the overall accuracy.
    overall_accuracy_se: float
    # User's accuracy of every class.
    users_accuracy: NDArray[np.float64]
    # Standard error of the user's accuracies.
    users_accuracy_se: NDArray[np.float64]
    # Producer's accuracy of every class.
    producers_accuracy: NDArray[np.float64]
    # Standard error of the producer's accuracies.
    producers_accuracy_se: NDArray[np.float64]
    # Standard normal quantile of the intervals.
    z: float

    # Plain dictionary for reports and JSON.
    def to_dict(self) -> dict[str, Any]:
        # Per-class records.
        classes = {
            name: {  # Record of one class.
                "proportion": float(self.proportion[i]),  # Area proportion.
                "proportion_se": float(self.proportion_se[i]),  # Its standard error.
                "area": float(self.area[i]),  # Estimated area.
                "area_se": float(self.area_se[i]),  # Its standard error.
                "area_ci": float(self.area_ci[i]),  # Interval half-width.
                "mapped_area": float(self.mapped_area[i]),  # Pixel counting area.
                "users_accuracy": float(self.users_accuracy[i]),  # User's accuracy.
                "users_accuracy_ci": float(self.z * self.users_accuracy_se[i]),  # Half-width.
                "producers_accuracy": float(self.producers_accuracy[i]),  # Producer's accuracy.
                "producers_accuracy_ci": float(self.z * self.producers_accuracy_se[i]),  # Width.
            }  # End of the class record.
            for i, name in enumerate(self.classes)  # Every class.
        }  # End of the class records.
        # Report.
        return {
            "overall_accuracy": self.overall_accuracy,  # Overall accuracy.
            "overall_accuracy_ci": self.z * self.overall_accuracy_se,  # Half-width.
            "classes": classes,  # Per-class estimates.
        }  # End of the report.


# Area and accuracy estimates of Olofsson et al. (2014).
def stratified_area_estimate(
    matrix: NDArray[Any],  # Sample counts, rows reference, columns map.
    mapped_area: Sequence[float] | NDArray[Any],  # Mapped area (or pixels) of every map class.
    confidence: float = 0.95,  # Confidence level of the intervals.
    classes: Sequence[Any] | None = None,  # Class names.
) -> AreaEstimate:  # Estimates with standard errors.
    # The confidence level must lie in (0, 1).
    if not 0.0 < confidence < 1.0:
        # Explain the requirement.
        raise ValueError("confidence must lie between 0 and 1")
    # Validated inputs with map classes in rows.
    n, a = _inputs(matrix, mapped_area)
    # Number of classes.
    k = n.shape[0]
    # Class names.
    names = [str(c) for c in classes] if classes is not None else [str(i) for i in range(k)]
    # One name per class.
    if len(names) != k:
        # Explain the requirement.
        raise ValueError(f"expected {k} class names, got {len(names)}")
    # Total area.
    total = a.sum()
    # Stratum weights W_i.
    w = a / total
    # Stratum sample sizes n_i.; empty strata have no weight.
    ni = n.sum(axis=1)
    # Safe sample sizes for divisions.
    ni_safe = np.maximum(ni, 1.0)
    # Denominator n_i. - 1 of the variances, safe for empty strata.
    dof = np.maximum(ni - 1.0, 1.0)
    # Within-stratum proportions n_ij / n_i. (map i, reference j).
    q = n / ni_safe[:, None]
    # Estimated population proportions p_ij.
    p = w[:, None] * q
    # Area proportions of the reference classes, p_.j.
    prop = p.sum(axis=0)
    # Standard errors of the proportions.
    prop_se = np.sqrt((w[:, None] ** 2 * q * (1.0 - q) / dof[:, None]).sum(axis=0))
    # User's accuracies U_i.
    ua = np.diag(q).copy()
    # Strata without samples have no user's accuracy.
    ua[ni == 0] = np.nan
    # Standard errors of the user's accuracies.
    ua_se = np.sqrt(ua * (1.0 - ua) / dof)
    # Overall accuracy.
    oa = float(np.trace(p))
    # Standard error of the overall accuracy.
    oa_se = float(np.sqrt(np.nansum(w**2 * ua * (1.0 - ua) / dof)))
    # Producer's accuracies P_j = p_jj / p_.j.
    pa = np.divide(np.diag(p), prop, out=np.full(k, np.nan), where=prop > 0)
    # Estimated reference totals N_.j = sum_i N_i. n_ij / n_i.
    n_ref = (a[:, None] * q).sum(axis=0)
    # Terms N_i.^2 (n_ij/n_i.)(1 - n_ij/n_i.)/(n_i. - 1) for every (i, j).
    off = a[:, None] ** 2 * q * (1.0 - q) / dof[:, None]
    # Sum over i != j.
    off_sum = off.sum(axis=0) - np.diag(off)
    # First term of the variance: N_j.^2 (1 - P_j)^2 U_j (1 - U_j) / (n_j. - 1).
    first = a**2 * (1.0 - pa) ** 2 * ua * (1.0 - ua) / dof
    # Variance of the producer's accuracies.
    pa_var = np.divide(first + pa**2 * off_sum, n_ref**2, out=np.full(k, np.nan), where=n_ref > 0)
    # Standard normal quantile of the two-sided interval.
    z = float(norm.ppf(0.5 + confidence / 2.0))
    # Build the record.
    return AreaEstimate(
        classes=names,  # Class names.
        proportion=prop,  # Area proportions.
        proportion_se=prop_se,  # Their standard errors.
        area=prop * total,  # Estimated areas.
        area_se=prop_se * total,  # Their standard errors.
        area_ci=z * prop_se * total,  # Interval half-widths.
        mapped_area=a,  # Pixel counting areas.
        overall_accuracy=oa,  # Overall accuracy.
        overall_accuracy_se=oa_se,  # Its standard error.
        users_accuracy=ua,  # User's accuracies.
        users_accuracy_se=ua_se,  # Their standard errors.
        producers_accuracy=pa,  # Producer's accuracies.
        producers_accuracy_se=np.sqrt(pa_var),  # Their standard errors.
        z=z,  # Normal quantile.
    )  # End of the record.


# Stratum sample sizes for a target standard error of overall accuracy.
def sample_allocation(
    mapped_area: Sequence[float] | NDArray[Any],  # Mapped area of every class.
    expected_users_accuracy: Sequence[float] | NDArray[Any],  # Anticipated U_i.
    target_se: float = 0.01,  # Target standard error of overall accuracy.
    rare_minimum: int = 50,  # Minimum sample size of every stratum.
) -> NDArray[np.int64]:  # Sample size of every stratum.
    # Areas as float.
    a = np.asarray(mapped_area, dtype=np.float64).reshape(-1)
    # Anticipated accuracies as float.
    u = np.asarray(expected_users_accuracy, dtype=np.float64).reshape(-1)
    # Inputs must pair up and lie in their ranges.
    if a.shape != u.shape or (a < 0).any() or a.sum() <= 0 or ((u < 0) | (u > 1)).any():
        # Explain the requirement.
        raise ValueError("mapped_area and expected_users_accuracy must match and be valid")
    # The target must be positive.
    if target_se <= 0:
        # Explain the requirement.
        raise ValueError("target_se must be positive")
    # Stratum weights.
    w = a / a.sum()
    # Standard deviations of the strata, sqrt(U (1 - U)).
    s = np.sqrt(u * (1.0 - u))
    # Total sample size n = (sum W_i S_i / S(O))^2 (Cochran, 1977; Olofsson et al., 2014).
    n_total = (np.sum(w * s) / target_se) ** 2
    # Proportional allocation, raised to the minimum for rare classes.
    alloc = np.maximum(np.ceil(n_total * w), rare_minimum)
    # Return integer sizes.
    return alloc.astype(np.int64)


# =============================================================================
# End of module src/unbihexium/metrics/area.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
