# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/geostat/variogram.py
# Title       : Empirical and model semivariograms
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Isotropic semivariograms of point data z(x):
#
#   empirical_variogram   binned estimate of gamma(h) with the Matheron or
#                         the robust Cressie-Hawkins estimator
#   variogram_function    the model families below
#   Variogram             fits a model to the empirical variogram by
#                         weighted least squares, or holds given parameters
#
# Empirical estimators for the N(h) pairs of a distance bin:
#
#   Matheron        gamma(h) = sum (z_i - z_j)^2 / (2 N(h))
#   Cressie-Hawkins gamma(h) = (mean |z_i - z_j|^(1/2))^4
#                              / (2 (0.457 + 0.494 / N(h)))
#
# Models
# ------
# With nugget c0, partial sill c (total sill c0 + c) and scale a, for h > 0
# (gamma(0) = 0 always):
#
#   spherical    c0 + c (1.5 h / a - 0.5 (h / a)^3) for h < a, else c0 + c
#   exponential  c0 + c (1 - exp(-h / a)); practical range 3 a
#   gaussian     c0 + c (1 - exp(-(h / a)^2)); practical range sqrt(3) a
#   matern       c0 + c (1 - 2^(1 - v) / Gamma(v) (h / a)^v K_v(h / a)),
#                smoothness v (v = 0.5 is the exponential model)
#   linear       c0 + c h / a (no sill)
#   power        c0 + c (h / a)^v with 0 < v < 2 (no sill)
#
# The fit minimises sum N(h) (gamma_hat(h) - gamma(h))^2 over the bins, the
# pair-count weighting commonly used in practice (Cressie, 1985, discusses
# the weights).
#
# References
# ----------
# Matheron, G. (1963). Principles of geostatistics. Economic Geology, 58(8),
#   1246-1266.
# Cressie, N., Hawkins, D. M. (1980). Robust estimation of the variogram: I.
#   Mathematical Geology, 12(2), 115-125.
# Cressie, N. (1985). Fitting variogram models by weighted least squares.
#   Mathematical Geology, 17(5), 563-586.
# Stein, M. L. (1999). Interpolation of Spatial Data: Some Theory for
#   Kriging. Springer, New York.
# Webster, R., Oliver, M. A. (2007). Geostatistics for Environmental
#   Scientists, 2nd edition. Wiley, Chichester.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Result record.
from dataclasses import dataclass

# Model names.
from enum import Enum

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Least-squares fitting.
from scipy.optimize import curve_fit

# Pairwise distances.
from scipy.spatial.distance import pdist

# Gamma function.
from scipy.special import gamma as gamma_function

# Modified Bessel function of the second kind.
from scipy.special import kv


# Variogram model families.
class VariogramModel(str, Enum):
    # Spherical model with a finite range.
    SPHERICAL = "spherical"
    # Exponential model.
    EXPONENTIAL = "exponential"
    # Gaussian model (very smooth fields).
    GAUSSIAN = "gaussian"
    # Matern model with smoothness parameter.
    MATERN = "matern"
    # Linear model without sill.
    LINEAR = "linear"
    # Power model without sill.
    POWER = "power"


# Result of fitting a variogram.
@dataclass
class VariogramResult:
    # Mean pair distance of each non-empty lag bin.
    lags: NDArray[np.floating[Any]]
    # Empirical semivariance of each bin.
    semivariance: NDArray[np.floating[Any]]
    # Model family.
    model: VariogramModel
    # Nugget c0.
    nugget: float
    # Partial sill c; the total sill is nugget + sill.
    sill: float
    # Scale parameter a.
    range_param: float
    # Model values at the lags.
    fitted_values: NDArray[np.floating[Any]] | None = None
    # Number of pairs in each bin.
    counts: NDArray[np.integer[Any]] | None = None
    # Smoothness of the Matern model or exponent of the power model.
    shape: float = 0.5

    # Total sill c0 + c.
    @property
    def total_sill(self) -> float:
        # Sum of nugget and partial sill.
        return self.nugget + self.sill


# Validated coordinates (n, d) and values (n,).
def check_points(
    coordinates: NDArray[Any],  # Point coordinates.
    values: NDArray[Any],  # Values at the points.
    minimum: int = 2,  # Minimum number of points.
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # Coordinates and values.
    # Coordinates as a float64 matrix.
    x = np.asarray(coordinates, dtype=np.float64)
    # One-dimensional coordinates become a column.
    if x.ndim == 1:
        # Column of positions.
        x = x[:, None]
    # Values as a float64 vector.
    z = np.asarray(values, dtype=np.float64).ravel()
    # One value per point.
    if x.ndim != 2 or x.shape[0] != z.size:
        # Report the mismatch.
        raise ValueError(f"coordinates {x.shape} and values {z.shape} do not match")
    # Every coordinate and value must be finite.
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z))):
        # Report the invalid data.
        raise ValueError("coordinates and values must be finite; remove NaN points first")
    # Enough points for the method.
    if z.size < minimum:
        # Report the too small sample.
        raise ValueError(f"at least {minimum} points are needed, got {z.size}")
    # Return the arrays.
    return x, z


# Semivariance of a model at distances h (gamma(0) = 0).
def variogram_function(
    model: VariogramModel | str,  # Model family.
    h: NDArray[Any] | float,  # Distances.
    nugget: float,  # Nugget c0.
    sill: float,  # Partial sill c.
    range_param: float,  # Scale a.
    shape: float = 0.5,  # Matern smoothness or power exponent.
) -> NDArray[np.float64]:  # Semivariance.
    # Model as an enum member.
    kind = VariogramModel(model)
    # Distances as float64.
    d = np.abs(np.asarray(h, dtype=np.float64))
    # The scale must be positive.
    if range_param <= 0:
        # Report the invalid scale.
        raise ValueError(f"range_param must be positive, got {range_param}")
    # Scaled distance.
    r = d / range_param
    # Spherical model.
    if kind is VariogramModel.SPHERICAL:
        # Structured part, constant beyond the range.
        part = np.where(r < 1.0, 1.5 * r - 0.5 * r**3, 1.0)
    # Exponential model.
    elif kind is VariogramModel.EXPONENTIAL:
        # 1 - exp(-h / a).
        part = 1.0 - np.exp(-r)
    # Gaussian model.
    elif kind is VariogramModel.GAUSSIAN:
        # 1 - exp(-(h / a)^2).
        part = 1.0 - np.exp(-(r**2))
    # Matern model.
    elif kind is VariogramModel.MATERN:
        # Smoothness must be positive.
        if shape <= 0:
            # Report the invalid smoothness.
            raise ValueError(f"Matern smoothness must be positive, got {shape}")
        # Correlation 2^(1 - v) / Gamma(v) r^v K_v(r) for r > 0.
        with np.errstate(invalid="ignore", over="ignore"):
            # Evaluate at positive distances only.
            rp = np.where(r > 0, r, 1.0)
            # Matern correlation; K_v underflows to 0 at large r.
            corr = 2.0 ** (1.0 - shape) / gamma_function(shape) * rp**shape * kv(shape, rp)
        # Structured part.
        part = 1.0 - np.nan_to_num(corr, nan=0.0)
    # Linear model.
    elif kind is VariogramModel.LINEAR:
        # Proportional to distance.
        part = r
    # Power model.
    else:
        # Exponent must lie in (0, 2).
        if not 0.0 < shape < 2.0:
            # Report the invalid exponent.
            raise ValueError(f"power exponent must lie in (0, 2), got {shape}")
        # r^v.
        part = r**shape
    # Zero at zero distance, nugget plus structure elsewhere.
    return np.where(d > 0, nugget + sill * part, 0.0)


# Binned empirical semivariogram.
def empirical_variogram(
    coordinates: NDArray[Any],  # Point coordinates (n, d).
    values: NDArray[Any],  # Values (n,).
    n_lags: int = 15,  # Number of equal-width distance bins.
    max_lag: float | None = None,  # Largest distance; half the largest pair distance by default.
    estimator: str = "matheron",  # "matheron" or "cressie".
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:  # Lags, gamma, counts.
    # Validated data.
    x, z = check_points(coordinates, values)
    # At least one bin.
    if n_lags < 1:
        # Report the invalid bin count.
        raise ValueError(f"n_lags must be >= 1, got {n_lags}")
    # Distances of all pairs.
    dist = pdist(x)
    # Absolute value differences of all pairs, in the same order.
    diff = pdist(z[:, None], metric="cityblock")
    # Default maximum lag.
    limit = float(max_lag) if max_lag is not None else float(dist.max()) / 2.0
    # The maximum lag must be positive.
    if limit <= 0:
        # Report the degenerate geometry.
        raise ValueError("max_lag must be positive (are all points at one location?)")
    # Bin edges.
    edges = np.linspace(0.0, limit, n_lags + 1)
    # Bin of every pair; pairs beyond the limit are dropped.
    keep = dist <= limit
    # Bin indices, the last edge belongs to the last bin.
    bins = np.minimum(np.searchsorted(edges, dist[keep], side="right") - 1, n_lags - 1)
    # Pairs per bin.
    counts = np.bincount(bins, minlength=n_lags)
    # Sum of distances per bin.
    sum_d = np.bincount(bins, weights=dist[keep], minlength=n_lags)
    # Matheron estimator.
    if estimator == "matheron":
        # Sum of squared differences per bin.
        s = np.bincount(bins, weights=diff[keep] ** 2, minlength=n_lags)
        # Half the mean squared difference.
        with np.errstate(divide="ignore", invalid="ignore"):
            # Divide by the pair count.
            gamma = s / (2.0 * counts)
    # Cressie-Hawkins estimator.
    elif estimator == "cressie":
        # Sum of square roots of absolute differences.
        s = np.bincount(bins, weights=np.sqrt(diff[keep]), minlength=n_lags)
        # Robust estimator.
        with np.errstate(divide="ignore", invalid="ignore"):
            # Fourth power of the mean root, bias corrected.
            gamma = (s / counts) ** 4 / (2.0 * (0.457 + 0.494 / counts))
    # Unknown estimator.
    else:
        # Report the valid names.
        raise ValueError(f"unknown estimator {estimator!r}; expected 'matheron' or 'cressie'")
    # Mean pair distance per bin.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Divide by the pair count.
        lags = sum_d / counts
    # Keep bins with pairs.
    filled = counts > 0
    # Return lags, semivariance and counts.
    return lags[filled], gamma[filled], counts[filled].astype(np.int64)


# Semivariogram estimation and model fitting.
class Variogram:
    # Configure the empirical bins and the model family.
    def __init__(
        self,  # The instance.
        n_lags: int = 15,  # Number of distance bins.
        max_lag: float | None = None,  # Largest distance used.
        model: VariogramModel | str = VariogramModel.SPHERICAL,  # Model family.
        estimator: str = "matheron",  # Empirical estimator.
        shape: float = 0.5,  # Matern smoothness or power exponent (not fitted).
    ) -> None:  # The constructor returns nothing.
        # Number of bins.
        self.n_lags = n_lags
        # Largest distance.
        self.max_lag = max_lag
        # Model family.
        self.model = VariogramModel(model)
        # Empirical estimator.
        self.estimator = estimator
        # Fixed shape parameter.
        self.shape = shape
        # Fitted or given parameters.
        self._fitted: VariogramResult | None = None

    # Variogram with known parameters, ready for prediction and kriging.
    @classmethod
    def from_parameters(
        cls,  # The class.
        model: VariogramModel | str,  # Model family.
        nugget: float,  # Nugget c0.
        sill: float,  # Partial sill c.
        range_param: float,  # Scale a.
        shape: float = 0.5,  # Matern smoothness or power exponent.
    ) -> Variogram:  # The variogram with its parameters.
        # Parameters must be admissible.
        if nugget < 0 or sill < 0 or range_param <= 0:
            # Report the invalid parameters.
            raise ValueError("nugget and sill must be >= 0 and range_param > 0")
        # New variogram of the family.
        variogram = cls(model=model, shape=shape)
        # Record the parameters without empirical data.
        variogram._fitted = VariogramResult(
            lags=np.zeros(0),  # No empirical lags.
            semivariance=np.zeros(0),  # No empirical values.
            model=variogram.model,  # Family.
            nugget=float(nugget),  # Nugget.
            sill=float(sill),  # Partial sill.
            range_param=float(range_param),  # Scale.
            shape=float(shape),  # Shape.
        )  # End of the parameters.
        # Return the variogram.
        return variogram

    # Whether parameters are available.
    @property
    def is_fitted(self) -> bool:
        # Fitted or given parameters exist.
        return self._fitted is not None

    # Fitted parameters and empirical variogram.
    @property
    def result(self) -> VariogramResult:
        # Parameters are needed.
        if self._fitted is None:
            # Report the missing fit.
            raise RuntimeError("variogram not fitted; call fit() or from_parameters()")
        # Return the record.
        return self._fitted

    # Estimate the empirical variogram and fit the model to it.
    def fit(
        self,  # The instance.
        coordinates: NDArray[Any],  # Point coordinates (n, d).
        values: NDArray[Any],  # Values (n,).
    ) -> VariogramResult:  # Empirical variogram and fitted parameters.
        # Empirical variogram.
        lags, gamma, counts = empirical_variogram(
            coordinates,  # Locations.
            values,  # Values.
            self.n_lags,  # Number of bins.
            self.max_lag,  # Largest distance.
            self.estimator,  # Estimator.
        )  # End of the empirical estimate.
        # Largest lag, used for starting values and bounds.
        top = float(lags.max()) if lags.size else 1.0
        # Largest semivariance.
        gmax = float(gamma.max()) if gamma.size and gamma.max() > 0 else 1.0
        # Models without sill keep their scale fixed at the largest lag.
        unbounded = self.model in (VariogramModel.LINEAR, VariogramModel.POWER)
        # Model with the scale fixed for unbounded families.
        family, shape = self.model, self.shape

        # Model evaluated with free nugget, sill and (for bounded models) scale.
        def curve(h: NDArray[np.float64], *params: float) -> NDArray[np.float64]:
            # Scale is fixed for unbounded models.
            scale = top if unbounded else params[2]
            # Evaluate the family.
            return variogram_function(family, h, params[0], params[1], scale, shape)

        # Starting values: small nugget, sill from the plateau, scale a third of the lags.
        p0 = [min(float(gamma[0]), gmax) * 0.5, gmax, top / 3.0]
        # Bounds of nugget, sill and scale.
        lower, upper = [0.0, 0.0, top * 1e-6], [gmax * 2.0, gmax * 10.0, top * 100.0]
        # Unbounded models have two free parameters.
        count = 2 if unbounded else 3
        # Weighted least squares with pair-count weights (sigma = 1 / sqrt(N)).
        try:
            # Fit the curve.
            params, _ = curve_fit(
                curve,  # Model.
                lags,  # Distances.
                gamma,  # Empirical values.
                p0=p0[:count],  # Starting values.
                sigma=1.0 / np.sqrt(counts),  # Pair-count weights.
                bounds=(lower[:count], upper[:count]),  # Admissible parameters.
                maxfev=20000,  # Iteration budget.
            )  # End of the fit.
        # Too few bins for the free parameters.
        except (RuntimeError, ValueError, TypeError):
            # Fall back to the starting values.
            params = np.asarray(p0[:count])
        # Nugget, partial sill and scale.
        nugget, sill = float(params[0]), float(params[1])
        # Scale of the model.
        scale = top if unbounded else float(params[2])
        # Record the fit.
        self._fitted = VariogramResult(
            lags=lags,  # Mean lag distances.
            semivariance=gamma,  # Empirical semivariance.
            model=self.model,  # Family.
            nugget=nugget,  # Nugget.
            sill=sill,  # Partial sill.
            range_param=scale,  # Scale.
            fitted_values=variogram_function(self.model, lags, nugget, sill, scale, shape),  # Fit.
            counts=counts,  # Pair counts.
            shape=shape,  # Shape.
        )  # End of the result.
        # Return the result.
        return self._fitted

    # Semivariance of the fitted model at distance h (float for scalar input).
    def predict(self, h: NDArray[Any] | float) -> Any:
        # Parameters.
        p = self.result
        # Evaluate the model.
        value = variogram_function(p.model, h, p.nugget, p.sill, p.range_param, p.shape)
        # Scalars stay scalars.
        return float(value) if np.ndim(h) == 0 else value

    # Semivariance used by kriging (alias of predict for arrays).
    def __call__(self, h: NDArray[Any]) -> NDArray[np.float64]:
        # Parameters.
        p = self.result
        # Evaluate the model as an array.
        return variogram_function(p.model, h, p.nugget, p.sill, p.range_param, p.shape)


# =============================================================================
# End of module src/unbihexium/geostat/variogram.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
