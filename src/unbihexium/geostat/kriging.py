# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/geostat/kriging.py
# Title       : Ordinary and universal kriging, inverse distance weighting
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Spatial interpolation of point data:
#
#   OrdinaryKriging     unknown constant mean
#   UniversalKriging    mean with a linear or quadratic trend in the
#                       coordinates (kriging with external drift is not
#                       included)
#   idw                 inverse distance weighting (Shepard, 1968)
#
# Kriging system
# --------------
# With semivariances Gamma_ij = gamma(|x_i - x_j|), the drift functions F
# (a column of ones for ordinary kriging; 1, x, y for a linear trend; plus
# x^2, x y, y^2 for a quadratic trend) and a target x0, the weights lambda
# and Lagrange multipliers mu solve
#
#   [ Gamma  F ] [ lambda ]   [ gamma_0 ]
#   [ F^T    0 ] [ mu     ] = [ f_0     ]
#
# The prediction is lambda^T z and the kriging variance is
# lambda^T gamma_0 + mu^T f_0 (Cressie, 1993, sections 3.2 and 3.4). The
# predictor is exact: at a data location it returns the datum with zero
# variance. For universal kriging, the variogram is fitted to the residuals
# of an ordinary least-squares trend, the usual practical approximation.
#
# By default all data enter every prediction; with `n_neighbors` only the
# nearest points are used (local kriging), which scales to large data.
#
# References
# ----------
# Matheron, G. (1963). Principles of geostatistics. Economic Geology, 58(8),
#   1246-1266.
# Shepard, D. (1968). A two-dimensional interpolation function for
#   irregularly-spaced data. Proc. 23rd ACM National Conference, 517-524.
# Cressie, N. (1993). Statistics for Spatial Data, revised edition. Wiley,
#   New York.
# Webster, R., Oliver, M. A. (2007). Geostatistics for Environmental
#   Scientists, 2nd edition. Wiley, Chichester.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Result record.
from dataclasses import dataclass

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Dense linear solver.
from scipy.linalg import LinAlgError, solve

# Nearest-neighbour search.
from scipy.spatial import cKDTree

# Distance matrices.
from scipy.spatial.distance import cdist

# Variogram models and input validation.
from unbihexium.geostat.variogram import Variogram, check_points


# Predictions of a kriging model.
@dataclass
class KrigingResult:
    # Predicted values at the targets.
    predictions: NDArray[np.floating[Any]]
    # Kriging variance at the targets.
    variance: NDArray[np.floating[Any]]
    # Target coordinates.
    coordinates: NDArray[np.floating[Any]]

    # Kriging standard deviation.
    @property
    def std(self) -> NDArray[np.float64]:
        # Square root of the variance, clipped against round-off.
        return np.sqrt(np.maximum(np.asarray(self.variance, dtype=np.float64), 0.0))


# Target coordinates as an (m, d) float64 matrix.
def _targets(coordinates: NDArray[Any], dims: int) -> NDArray[np.float64]:
    # Float64 matrix.
    t = np.atleast_2d(np.asarray(coordinates, dtype=np.float64))
    # One-dimensional data given as a flat vector.
    if dims == 1 and t.shape[0] == 1 and t.shape[1] != 1:
        # One target per entry.
        t = t.T
    # Dimensions must match the data.
    if t.shape[1] != dims:
        # Report the mismatch.
        raise ValueError(f"targets have {t.shape[1]} coordinates, data have {dims}")
    # Return the matrix.
    return t


# Ordinary kriging with an unknown constant mean.
class OrdinaryKriging:
    # Configure the variogram and the neighbourhood.
    def __init__(
        self,  # The instance.
        variogram: Variogram | None = None,  # Fitted, given or unfitted variogram.
        n_neighbors: int | None = None,  # Nearest points per prediction; None uses all.
    ) -> None:  # The constructor returns nothing.
        # Variogram; fitted on the data when it has no parameters.
        self.variogram = variogram or Variogram()
        # Neighbourhood size.
        self.n_neighbors = n_neighbors
        # Data coordinates.
        self._coordinates: NDArray[np.float64] | None = None
        # Data values.
        self._values: NDArray[np.float64] | None = None
        # Neighbour search tree.
        self._tree: cKDTree | None = None

    # Drift functions at the given coordinates; a constant for ordinary kriging.
    def _drift(self, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        # Column of ones.
        return np.ones((coordinates.shape[0], 1))

    # Values the variogram is fitted to; the data for ordinary kriging.
    def _variogram_values(
        self,  # The model.
        coordinates: NDArray[np.float64],  # Data coordinates.
        values: NDArray[np.float64],  # Data values.
    ) -> NDArray[np.float64]:  # Values for the variogram fit.
        # Raw values.
        return values

    # Store the data and fit the variogram if needed.
    def fit(
        self,  # The instance.
        coordinates: NDArray[Any],  # Data coordinates (n, d).
        values: NDArray[Any],  # Data values (n,).
    ) -> OrdinaryKriging:  # The fitted model.
        # Validated data.
        x, z = check_points(coordinates, values, minimum=3)
        # Duplicate locations make the system singular.
        if np.unique(x, axis=0).shape[0] != x.shape[0]:
            # Report the duplicates.
            raise ValueError("duplicate coordinates; average the values of coincident points")
        # Store the data.
        self._coordinates, self._values = x, z
        # Fit the variogram when it has no parameters yet.
        if not self.variogram.is_fitted:
            # Fit to the (possibly detrended) values.
            self.variogram.fit(x, self._variogram_values(x, z))
        # Search tree for local kriging.
        self._tree = cKDTree(x) if self.n_neighbors else None
        # Allow chaining.
        return self

    # Solve the kriging system for a set of data points and targets.
    def _solve(
        self,  # The instance.
        x: NDArray[np.float64],  # Data coordinates (n, d).
        z: NDArray[np.float64],  # Data values (n,).
        t: NDArray[np.float64],  # Targets (m, d).
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # Predictions and variances.
        # Number of data points.
        n = x.shape[0]
        # Drift at the data.
        f = self._drift(x)
        # Number of drift functions.
        p = f.shape[1]
        # Kriging matrix.
        a = np.zeros((n + p, n + p))
        # Semivariances between data points; zero on the diagonal.
        a[:n, :n] = self.variogram(cdist(x, x))
        # Drift columns.
        a[:n, n:] = f
        # Drift rows.
        a[n:, :n] = f.T
        # Right-hand sides, one column per target.
        b = np.vstack([self.variogram(cdist(x, t)), self._drift(t).T])
        # Solve for weights and multipliers.
        try:
            # Symmetric indefinite system.
            sol = solve(a, b, assume_a="sym")
        # Singular systems: too few points for the drift, or degenerate geometry.
        except LinAlgError as exc:
            # Report the problem.
            raise ValueError(f"kriging system is singular: {exc}") from exc
        # Predictions lambda^T z.
        pred = sol[:n].T @ z
        # Variances lambda^T gamma_0 + mu^T f_0.
        var = np.sum(sol * b, axis=0)
        # Clip round-off below zero.
        return pred, np.maximum(var, 0.0)

    # Predict values and variances at target coordinates.
    def predict(self, target_coordinates: NDArray[Any]) -> KrigingResult:
        # The model needs data.
        if self._coordinates is None or self._values is None:
            # Report the missing fit.
            raise RuntimeError("model not fitted; call fit() first")
        # Data.
        x, z = self._coordinates, self._values
        # Targets as a matrix.
        t = _targets(target_coordinates, x.shape[1])
        # Global kriging: one system for all targets.
        if not self.n_neighbors or self.n_neighbors >= x.shape[0]:
            # Solve once.
            pred, var = self._solve(x, z, t)
        # Local kriging: one small system per target.
        else:
            # Nearest data of every target.
            _, idx = self._tree.query(t, k=self.n_neighbors)  # type: ignore[union-attr]
            # Output arrays.
            pred, var = np.empty(t.shape[0]), np.empty(t.shape[0])
            # Solve per target.
            for i, near in enumerate(idx):
                # Local system.
                p_i, v_i = self._solve(x[near], z[near], t[i : i + 1])
                # Store the results.
                pred[i], var[i] = p_i[0], v_i[0]
        # Package the result.
        return KrigingResult(predictions=pred, variance=var, coordinates=t)

    # Predict on the grid spanned by x and y coordinate vectors.
    def predict_grid(
        self,  # The instance.
        x: NDArray[Any],  # Column coordinates (nx,).
        y: NDArray[Any],  # Row coordinates (ny,).
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # Predictions and variances (ny, nx).
        # Coordinates as float.
        xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        # All grid nodes.
        gx, gy = np.meshgrid(xs, ys)
        # Predict at the nodes.
        result = self.predict(np.column_stack([gx.ravel(), gy.ravel()]))
        # Grid shape (ny, nx).
        shape = (ys.size, xs.size)
        # Predictions and variances as float64.
        pred = np.asarray(result.predictions, dtype=np.float64)
        # Variances.
        var = np.asarray(result.variance, dtype=np.float64)
        # Back to the grid shape.
        return pred.reshape(shape), var.reshape(shape)

    # Cross-validation with the fitted variogram (leave-one-out by default).
    def cross_validate(
        self,  # The instance.
        k_folds: int | None = None,  # Number of folds; None for leave-one-out.
        seed: int | None = 0,  # Random seed of the fold assignment.
    ) -> dict[str, float]:  # rmse, mae, bias and mean standardised squared error.
        # The model needs data.
        if self._coordinates is None or self._values is None:
            # Report the missing fit.
            raise RuntimeError("model not fitted; call fit() first")
        # Data.
        x, z = self._coordinates, self._values
        # Number of points.
        n = z.size
        # Fold of every point.
        if k_folds is None:
            # Every point is its own fold.
            folds = np.arange(n)
        # Random folds of almost equal size.
        else:
            # Folds must split the data.
            if not 2 <= k_folds <= n:
                # Report the invalid fold count.
                raise ValueError(f"k_folds must lie in [2, {n}], got {k_folds}")
            # Random permutation of the fold labels.
            folds = np.random.default_rng(seed).permutation(np.arange(n) % k_folds)
        # Prediction errors and variances.
        errors, variances = np.empty(n), np.empty(n)
        # Predict every fold from the others.
        for fold in np.unique(folds):
            # Held-out points.
            test = folds == fold
            # Model on the remaining points with the same variogram.
            model = self.__class__.__new__(self.__class__)
            # Copy the configuration.
            model.__dict__.update(self.__dict__)
            # Store the training data directly (no refit).
            model._coordinates, model._values = x[~test], z[~test]
            # Tree for local kriging.
            model._tree = cKDTree(x[~test]) if self.n_neighbors else None
            # Predict the held-out points.
            result = model.predict(x[test])
            # Errors: prediction minus observation.
            errors[test] = result.predictions - z[test]
            # Kriging variances.
            variances[test] = result.variance
        # Standardised squared errors (expected mean 1 for a good model).
        with np.errstate(divide="ignore", invalid="ignore"):
            # Error squared over variance.
            msse = float(np.nanmean(errors**2 / variances))
        # Summary statistics.
        return {
            "rmse": float(np.sqrt(np.mean(errors**2))),  # Root mean squared error.
            "mae": float(np.mean(np.abs(errors))),  # Mean absolute error.
            "bias": float(np.mean(errors)),  # Mean error.
            "msse": msse,  # Mean standardised squared error.
        }  # End of the statistics.


# Universal kriging with a polynomial trend in the coordinates.
class UniversalKriging(OrdinaryKriging):
    # Configure the variogram, the trend and the neighbourhood.
    def __init__(
        self,  # The instance.
        variogram: Variogram | None = None,  # Variogram of the residuals.
        drift_terms: int = 1,  # 0 constant, 1 linear, 2 quadratic trend.
        n_neighbors: int | None = None,  # Nearest points per prediction.
    ) -> None:  # The constructor returns nothing.
        # Common configuration.
        super().__init__(variogram, n_neighbors)
        # The trend order must be 0, 1 or 2.
        if drift_terms not in (0, 1, 2):
            # Report the invalid order.
            raise ValueError(f"drift_terms must be 0, 1 or 2, got {drift_terms}")
        # Trend order.
        self.drift_terms = drift_terms
        # Centre and scale of the coordinates for a well-conditioned drift.
        self._centre: NDArray[np.float64] | None = None
        # Scale of the coordinates.
        self._scale: float = 1.0

    # Polynomial drift functions of the (centred and scaled) coordinates.
    def _build_drift_matrix(self, coordinates: NDArray[Any]) -> NDArray[np.float64]:
        # Coordinates as a matrix.
        c = np.atleast_2d(np.asarray(coordinates, dtype=np.float64))
        # Centre of the data; the targets' own centre before fitting.
        centre = self._centre if self._centre is not None else c.mean(axis=0)
        # Normalised coordinates.
        u = (c - centre) / self._scale
        # Constant term.
        cols = [np.ones(c.shape[0])]
        # Linear terms.
        if self.drift_terms >= 1:
            # One column per coordinate.
            cols += [u[:, k] for k in range(u.shape[1])]
        # Quadratic terms.
        if self.drift_terms >= 2:
            # Products of all coordinate pairs, squares included.
            cols += [u[:, i] * u[:, j] for i in range(u.shape[1]) for j in range(i, u.shape[1])]
        # Drift matrix.
        return np.column_stack(cols)

    # Drift used in the kriging system.
    def _drift(self, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        # Polynomial terms.
        return self._build_drift_matrix(coordinates)

    # Residuals of an ordinary least-squares trend, for the variogram fit.
    def _variogram_values(
        self,  # The model.
        coordinates: NDArray[np.float64],  # Data coordinates.
        values: NDArray[np.float64],  # Data values.
    ) -> NDArray[np.float64]:  # Values for the variogram fit.
        # Trend design matrix.
        f = self._build_drift_matrix(coordinates)
        # Least-squares trend coefficients.
        beta, *_ = np.linalg.lstsq(f, values, rcond=None)
        # Residuals.
        return values - f @ beta

    # Store the data, fix the coordinate normalisation and fit the residual variogram.
    def fit(self, coordinates: NDArray[Any], values: NDArray[Any]) -> UniversalKriging:
        # Validated data.
        x, _ = check_points(coordinates, values, minimum=3)
        # Centre of the data.
        self._centre = x.mean(axis=0)
        # Largest spread, never zero.
        self._scale = float(np.ptp(x, axis=0).max()) or 1.0
        # Common fitting steps.
        super().fit(coordinates, values)
        # Allow chaining.
        return self


# Inverse distance weighting (Shepard, 1968).
def idw(
    coordinates: NDArray[Any],  # Data coordinates (n, d).
    values: NDArray[Any],  # Data values (n,).
    targets: NDArray[Any],  # Target coordinates (m, d).
    power: float = 2.0,  # Distance exponent.
    n_neighbors: int | None = None,  # Nearest points per target; None uses all.
) -> NDArray[np.float64]:  # Interpolated values (m,).
    # Validated data.
    x, z = check_points(coordinates, values, minimum=1)
    # The exponent must be positive.
    if power <= 0:
        # Report the invalid exponent.
        raise ValueError(f"power must be positive, got {power}")
    # Targets as a matrix.
    t = _targets(targets, x.shape[1])
    # Neighbours of every target.
    if n_neighbors is not None and n_neighbors < z.size:
        # Nearest points and their distances.
        dist, idx = cKDTree(x).query(t, k=n_neighbors)
        # A single neighbour comes back as vectors; keep one column per neighbour.
        dist, idx = dist.reshape(t.shape[0], -1), idx.reshape(t.shape[0], -1)
    # All points.
    else:
        # Full distance matrix.
        dist = cdist(t, x)
        # Every data index for every target.
        idx = np.broadcast_to(np.arange(z.size), dist.shape)
    # Distance-based weights; coincident points get infinite weight.
    with np.errstate(divide="ignore"):
        # 1 / d^p.
        w = 1.0 / np.asarray(dist) ** power
    # Targets on a data point.
    exact = np.isinf(w)
    # Weighted mean of the neighbours.
    with np.errstate(invalid="ignore"):
        # Sum w z / sum w.
        pred = np.sum(w * z[idx], axis=1) / np.sum(w, axis=1)
    # Exact interpolation at data points.
    hit = exact.any(axis=1)
    # Take the datum (the mean of coincident data).
    pred[hit] = np.array([z[idx[i][exact[i]]].mean() for i in np.flatnonzero(hit)])
    # Return the predictions.
    return pred


# =============================================================================
# End of module src/unbihexium/geostat/kriging.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
