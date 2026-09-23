# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/metrics/regression.py
# Title       : Error statistics of continuous map products
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Validation statistics of estimated against reference values (biomass,
# canopy height, soil moisture, temperature, spectral indices, ...). Pairs
# with NaN on either side are ignored:
#
#   bias, mae, rmse      mean error, mean absolute error, root mean square
#                        error
#   ubrmse               unbiased RMSE, sqrt(RMSE^2 - bias^2)
#   r_squared            coefficient of determination 1 - SS_res / SS_tot
#   pearson_r            Pearson correlation coefficient
#   regression_report    all of the above plus the ordinary least squares
#                        line of estimate on reference and relative errors
#
# Errors are estimate minus reference. r_squared is computed against the
# 1:1 line (it equals the Nash-Sutcliffe efficiency) and can be negative;
# it is not the squared correlation.
#
# References
# ----------
#   Willmott, C. J. (1982). Some comments on the evaluation of model
#     performance. Bulletin of the American Meteorological Society 63(11),
#     1309-1313.
#   Nash, J. E., Sutcliffe, J. V. (1970). River flow forecasting through
#     conceptual models part I: a discussion of principles. Journal of
#     Hydrology 10(3), 282-290.
#   Entekhabi, D., Reichle, R. H., Koster, R. D., Crow, W. T. (2010).
#     Performance metrics for soil moisture retrievals and application
#     requirements. Journal of Hydrometeorology 11(3), 832-840.
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


# Finite pairs of estimates and references as flat float arrays.
def _pairs(pred: NDArray[Any], target: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    # Estimates as float.
    p = np.asarray(pred, dtype=np.float64)
    # References as float.
    t = np.asarray(target, dtype=np.float64)
    # Shapes must agree.
    if p.shape != t.shape:
        # Explain the requirement.
        raise ValueError(f"shape mismatch: {p.shape} and {t.shape}")
    # Pairs that are finite on both sides.
    valid = np.isfinite(p) & np.isfinite(t)
    # At least one pair is needed.
    if not valid.any():
        # Explain the requirement.
        raise ValueError("no finite pairs to evaluate")
    # Return the valid pairs.
    return p[valid], t[valid]


# Mean error, estimate minus reference.
def bias(pred: NDArray[Any], target: NDArray[Any]) -> float:
    # Valid pairs.
    p, t = _pairs(pred, target)
    # Mean difference.
    return float(np.mean(p - t))


# Mean absolute error.
def mae(pred: NDArray[Any], target: NDArray[Any]) -> float:
    # Valid pairs.
    p, t = _pairs(pred, target)
    # Mean absolute difference.
    return float(np.mean(np.abs(p - t)))


# Root mean square error.
def rmse(pred: NDArray[Any], target: NDArray[Any]) -> float:
    # Valid pairs.
    p, t = _pairs(pred, target)
    # Square root of the mean squared difference.
    return float(np.sqrt(np.mean((p - t) ** 2)))


# Unbiased root mean square error.
def ubrmse(pred: NDArray[Any], target: NDArray[Any]) -> float:
    # Valid pairs.
    p, t = _pairs(pred, target)
    # Differences.
    d = p - t
    # Standard deviation of the differences equals sqrt(RMSE^2 - bias^2).
    return float(np.sqrt(max(np.mean(d**2) - np.mean(d) ** 2, 0.0)))


# Coefficient of determination against the 1:1 line.
def r_squared(pred: NDArray[Any], target: NDArray[Any]) -> float:
    # Valid pairs.
    p, t = _pairs(pred, target)
    # Residual sum of squares.
    ss_res = float(np.sum((t - p) ** 2))
    # Total sum of squares of the references.
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    # Undefined for constant references.
    if ss_tot == 0:
        # Not a number.
        return float("nan")
    # 1 - SS_res / SS_tot.
    return 1.0 - ss_res / ss_tot


# Pearson correlation coefficient.
def pearson_r(pred: NDArray[Any], target: NDArray[Any]) -> float:
    # Valid pairs.
    p, t = _pairs(pred, target)
    # Centred estimates.
    dp = p - p.mean()
    # Centred references.
    dt = t - t.mean()
    # Product of the norms.
    den = float(np.sqrt(np.sum(dp**2) * np.sum(dt**2)))
    # Undefined when either side is constant.
    if den == 0:
        # Not a number.
        return float("nan")
    # Normalised covariance.
    return float(np.sum(dp * dt) / den)


# All error statistics of a set of pairs.
def regression_report(pred: NDArray[Any], target: NDArray[Any]) -> dict[str, float]:
    # Valid pairs.
    p, t = _pairs(pred, target)
    # Variance of the references.
    var_t = float(np.var(t))
    # Slope of the least squares line of estimate on reference.
    slope = float(np.mean((p - p.mean()) * (t - t.mean())) / var_t) if var_t > 0 else float("nan")
    # Intercept of the line.
    intercept = float(p.mean() - slope * t.mean()) if var_t > 0 else float("nan")
    # Mean of the references.
    mean_ref = float(t.mean())
    # Root mean square error.
    e = rmse(p, t)
    # Report.
    return {
        "n": int(p.size),  # Number of valid pairs.
        "bias": bias(p, t),  # Mean error.
        "mae": mae(p, t),  # Mean absolute error.
        "rmse": e,  # Root mean square error.
        "ubrmse": ubrmse(p, t),  # Unbiased RMSE.
        "relative_rmse": e / mean_ref if mean_ref != 0 else float("nan"),  # RMSE / mean.
        "r2": r_squared(p, t),  # Coefficient of determination.
        "r": pearson_r(p, t),  # Correlation.
        "slope": slope,  # Least squares slope.
        "intercept": intercept,  # Least squares intercept.
    }  # End of the report.


# =============================================================================
# End of module src/unbihexium/metrics/regression.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
