"""Variogram analysis for geostatistics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class VariogramModel(str, Enum):
    """Variogram model types."""

    SPHERICAL = "spherical"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"
    LINEAR = "linear"
    POWER = "power"


@dataclass
class VariogramResult:
    """Result of variogram analysis."""

    lags: NDArray[np.floating[Any]]
    semivariance: NDArray[np.floating[Any]]
    model: VariogramModel
    nugget: float
    sill: float
    range_param: float
    fitted_values: NDArray[np.floating[Any]] | None = None


class Variogram:
    """Variogram analysis for spatial autocorrelation.

    The semivariogram gamma(h) is defined as:
    $$\\gamma(h) = \\frac{1}{2N(h)} \\sum_{i=1}^{N(h)} [z(x_i) - z(x_i + h)]^2$$

    where h is the lag distance and N(h) is the number of pairs at that lag.
    """

    def __init__(
        self,
        n_lags: int = 15,
        max_lag: float | None = None,
        model: VariogramModel = VariogramModel.SPHERICAL,
    ) -> None:
        self.n_lags = n_lags
        self.max_lag = max_lag
        self.model = model
        self._fitted: VariogramResult | None = None

    def fit(
        self,
        coordinates: NDArray[np.floating[Any]],
        values: NDArray[np.floating[Any]],
    ) -> VariogramResult:
        """Fit a variogram to the data.

        Args:
            coordinates: Array of shape (n, 2) with (x, y) coordinates.
            values: Array of shape (n,) with values at each location.

        Returns:
            VariogramResult with fitted parameters.
        """
        n = len(values)
        if n < 2:
            msg = "Need at least 2 points for variogram"
            raise ValueError(msg)

        # Calculate pairwise distances
        distances = self._pairwise_distances(coordinates)

        # Determine lag bins
        max_dist = self.max_lag or np.max(distances) / 2
        lags = np.linspace(0, max_dist, self.n_lags + 1)
        lag_centers = (lags[:-1] + lags[1:]) / 2

        # Calculate empirical semivariance
        semivariance = np.zeros(self.n_lags)
        counts = np.zeros(self.n_lags)

        for i in range(n):
            for j in range(i + 1, n):
                d = distances[i, j]
                bin_idx = np.searchsorted(lags[1:], d)
                if bin_idx < self.n_lags:
                    semivariance[bin_idx] += 0.5 * (values[i] - values[j]) ** 2
                    counts[bin_idx] += 1

        # Avoid division by zero
        counts = np.maximum(counts, 1)
        semivariance /= counts

        # Fit variogram model
        nugget, sill, range_param = self._fit_model(lag_centers, semivariance)

        # Calculate fitted values
        fitted = self._model_function(lag_centers, nugget, sill, range_param)

        self._fitted = VariogramResult(
            lags=lag_centers,
            semivariance=semivariance,
            model=self.model,
            nugget=nugget,
            sill=sill,
            range_param=range_param,
            fitted_values=fitted,
        )

        return self._fitted

    def _pairwise_distances(
        self, coordinates: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Calculate pairwise Euclidean distances."""
        n = len(coordinates)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((coordinates[i] - coordinates[j]) ** 2))
                distances[i, j] = d
                distances[j, i] = d
        return distances

    def _fit_model(
        self,
        lags: NDArray[np.floating[Any]],
        semivariance: NDArray[np.floating[Any]],
    ) -> tuple[float, float, float]:
        """Fit model parameters using least squares."""
        from scipy.optimize import minimize

        # Initial guesses
        nugget_init = semivariance[0]
        sill_init = np.max(semivariance)
        range_init = (
            lags[np.argmax(semivariance > 0.95 * sill_init)]
            if np.any(semivariance > 0.95 * sill_init)
            else lags[-1]
        )

        def objective(params: NDArray[np.floating[Any]]) -> float:
            nugget, sill, range_param = params
            predicted = self._model_function(lags, nugget, sill, range_param)
            return float(np.sum((semivariance - predicted) ** 2))

        result = minimize(
            objective,
            [nugget_init, sill_init, range_init],
            bounds=[(0, None), (0, None), (0, None)],
        )

        return tuple(result.x)

    def _model_function(
        self,
        h: NDArray[np.floating[Any]],
        nugget: float,
        sill: float,
        range_param: float,
    ) -> NDArray[np.floating[Any]]:
        """Calculate variogram model values."""
        if self.model == VariogramModel.SPHERICAL:
            # Spherical model
            hr = np.minimum(h / range_param, 1)
            return nugget + sill * (1.5 * hr - 0.5 * hr**3)

        elif self.model == VariogramModel.EXPONENTIAL:
            # Exponential model
            return nugget + sill * (1 - np.exp(-h / range_param))

        elif self.model == VariogramModel.GAUSSIAN:
            # Gaussian model
            return nugget + sill * (1 - np.exp(-((h / range_param) ** 2)))

        elif self.model == VariogramModel.LINEAR:
            # Linear model
            return nugget + sill * h / range_param

        else:
            return nugget + sill * (h / range_param) ** 1.5

    def predict(self, h: float) -> float:
        """Predict semivariance at a given lag distance."""
        if self._fitted is None:
            msg = "Variogram not fitted"
            raise RuntimeError(msg)

        return float(
            self._model_function(
                np.array([h]),
                self._fitted.nugget,
                self._fitted.sill,
                self._fitted.range_param,
            )[0]
        )
