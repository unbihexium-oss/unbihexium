"""Kriging interpolation for geostatistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve

from unbihexium.geostat.variogram import Variogram, VariogramResult


@dataclass
class KrigingResult:
    """Result of kriging interpolation."""

    predictions: NDArray[np.floating[Any]]
    variance: NDArray[np.floating[Any]]
    coordinates: NDArray[np.floating[Any]]


class OrdinaryKriging:
    """Ordinary Kriging interpolation.

    The ordinary kriging estimator is:
    $$\\hat{Z}(x_0) = \\sum_{i=1}^{n} \\lambda_i Z(x_i)$$

    where weights solve the kriging system:
    $$\\sum_{j=1}^{n} \\lambda_j \\gamma(x_i, x_j) + \\mu = \\gamma(x_i, x_0)$$
    $$\\sum_{j=1}^{n} \\lambda_j = 1$$
    """

    def __init__(self, variogram: Variogram | None = None) -> None:
        self.variogram = variogram or Variogram()
        self._fitted: VariogramResult | None = None
        self._coordinates: NDArray[np.floating[Any]] | None = None
        self._values: NDArray[np.floating[Any]] | None = None

    def fit(
        self,
        coordinates: NDArray[np.floating[Any]],
        values: NDArray[np.floating[Any]],
    ) -> OrdinaryKriging:
        """Fit the kriging model."""
        self._coordinates = coordinates
        self._values = values
        self._fitted = self.variogram.fit(coordinates, values)
        return self

    def predict(
        self,
        target_coordinates: NDArray[np.floating[Any]],
    ) -> KrigingResult:
        """Predict values at target locations."""
        if self._fitted is None or self._coordinates is None or self._values is None:
            msg = "Model not fitted"
            raise RuntimeError(msg)

        n = len(self._coordinates)
        m = len(target_coordinates)

        predictions = np.zeros(m)
        variance = np.zeros(m)

        # Build kriging matrix (n+1 x n+1)
        K = np.ones((n + 1, n + 1))
        K[-1, -1] = 0

        for i in range(n):
            for j in range(n):
                d = np.sqrt(np.sum((self._coordinates[i] - self._coordinates[j]) ** 2))
                K[i, j] = self.variogram.predict(d)

        for k in range(m):
            # Build right-hand side
            b = np.ones(n + 1)
            for i in range(n):
                d = np.sqrt(np.sum((self._coordinates[i] - target_coordinates[k]) ** 2))
                b[i] = self.variogram.predict(d)

            # Solve kriging system
            try:
                weights = solve(K, b)
                predictions[k] = np.sum(weights[:-1] * self._values)
                variance[k] = np.sum(weights[:-1] * b[:-1]) + weights[-1]
            except np.linalg.LinAlgError:
                predictions[k] = np.mean(self._values)
                variance[k] = np.var(self._values)

        return KrigingResult(
            predictions=predictions,
            variance=variance,
            coordinates=target_coordinates,
        )

    def cross_validate(self, k_folds: int = 5) -> dict[str, float]:
        """Perform k-fold cross-validation."""
        if self._coordinates is None or self._values is None:
            msg = "Model not fitted"
            raise RuntimeError(msg)

        n = len(self._values)
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k_folds)

        errors = []
        for i in range(k_folds):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])

            train_coords = self._coordinates[train_idx]
            train_values = self._values[train_idx]
            test_coords = self._coordinates[test_idx]
            test_values = self._values[test_idx]

            # Fit on training data
            ok = OrdinaryKriging()
            ok.fit(train_coords, train_values)
            result = ok.predict(test_coords)

            errors.extend((result.predictions - test_values).tolist())

        errors = np.array(errors)
        return {
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "bias": float(np.mean(errors)),
        }


class UniversalKriging(OrdinaryKriging):
    """Universal Kriging with trend."""

    def __init__(
        self,
        variogram: Variogram | None = None,
        drift_terms: int = 1,
    ) -> None:
        super().__init__(variogram)
        self.drift_terms = drift_terms

    def _build_drift_matrix(
        self, coordinates: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Build drift function matrix."""
        n = len(coordinates)
        if self.drift_terms == 0:
            return np.ones((n, 1))
        elif self.drift_terms == 1:
            # Linear drift
            return np.column_stack([np.ones(n), coordinates[:, 0], coordinates[:, 1]])
        else:
            # Quadratic drift
            x, y = coordinates[:, 0], coordinates[:, 1]
            return np.column_stack([np.ones(n), x, y, x**2, x * y, y**2])
