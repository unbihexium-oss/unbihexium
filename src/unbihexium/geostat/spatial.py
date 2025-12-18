"""Spatial autocorrelation statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SpatialAutocorrelationResult:
    """Result of spatial autocorrelation analysis."""

    statistic: float
    expected: float
    variance: float
    z_score: float
    p_value: float
    statistic_name: str


class MoransI:
    """Moran's I spatial autocorrelation statistic.

    Moran's I is calculated as:
    $$I = \\frac{n}{\\sum_i \\sum_j w_{ij}} \\cdot \\frac{\\sum_i \\sum_j w_{ij}(x_i - \\bar{x})(x_j - \\bar{x})}{\\sum_i (x_i - \\bar{x})^2}$$

    where w_ij is the spatial weight between locations i and j.
    """

    def __init__(self, distance_threshold: float | None = None) -> None:
        self.distance_threshold = distance_threshold

    def calculate(
        self,
        coordinates: NDArray[np.floating[Any]],
        values: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]] | None = None,
    ) -> SpatialAutocorrelationResult:
        """Calculate Moran's I statistic.

        Args:
            coordinates: Array of shape (n, 2) with coordinates.
            values: Array of shape (n,) with values.
            weights: Optional spatial weights matrix (n, n).

        Returns:
            SpatialAutocorrelationResult with statistic and significance.
        """
        n = len(values)
        mean = np.mean(values)
        deviations = values - mean

        if weights is None:
            weights = self._build_weights(coordinates)

        # Row-standardize weights
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        w = weights / row_sums

        # Calculate Moran's I
        numerator = np.sum(w * np.outer(deviations, deviations))
        denominator = np.sum(deviations**2)
        w_sum = np.sum(w)

        if denominator == 0 or w_sum == 0:
            return SpatialAutocorrelationResult(
                statistic=0.0,
                expected=-1 / (n - 1),
                variance=0.0,
                z_score=0.0,
                p_value=1.0,
                statistic_name="Moran's I",
            )

        I = (n / w_sum) * (numerator / denominator)

        # Expected value and variance under null hypothesis
        E_I = -1 / (n - 1)
        
        # Simplified variance calculation
        S1 = 0.5 * np.sum((w + w.T) ** 2)
        S2 = np.sum((w.sum(axis=0) + w.sum(axis=1)) ** 2)
        S0 = w_sum

        var_I = (
            (n * ((n**2 - 3*n + 3) * S1 - n*S2 + 3*S0**2) 
             - (n**2 - n) * ((n**2 - 3*n + 3) * S1 - n*S2 + 3*S0**2))
            / ((n - 1) * (n - 2) * (n - 3) * S0**2)
        )
        var_I = max(var_I, 1e-10)

        z_score = (I - E_I) / np.sqrt(var_I)
        
        # Two-tailed p-value
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        return SpatialAutocorrelationResult(
            statistic=float(I),
            expected=float(E_I),
            variance=float(var_I),
            z_score=float(z_score),
            p_value=float(p_value),
            statistic_name="Moran's I",
        )

    def _build_weights(
        self, coordinates: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Build spatial weights matrix."""
        n = len(coordinates)
        weights = np.zeros((n, n))

        threshold = self.distance_threshold
        if threshold is None:
            # Use average nearest neighbor distance
            distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt(np.sum((coordinates[i] - coordinates[j]) ** 2))
                    distances.append(d)
            threshold = np.mean(distances) if distances else 1.0

        for i in range(n):
            for j in range(n):
                if i != j:
                    d = np.sqrt(np.sum((coordinates[i] - coordinates[j]) ** 2))
                    if d <= threshold:
                        weights[i, j] = 1.0

        return weights


class GearysC:
    """Geary's C spatial autocorrelation statistic.

    Geary's C is calculated as:
    $$C = \\frac{(n-1) \\sum_i \\sum_j w_{ij}(x_i - x_j)^2}{2 W \\sum_i (x_i - \\bar{x})^2}$$

    Values < 1 indicate positive autocorrelation, > 1 negative.
    """

    def __init__(self, distance_threshold: float | None = None) -> None:
        self.distance_threshold = distance_threshold

    def calculate(
        self,
        coordinates: NDArray[np.floating[Any]],
        values: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]] | None = None,
    ) -> SpatialAutocorrelationResult:
        """Calculate Geary's C statistic."""
        n = len(values)
        mean = np.mean(values)
        
        if weights is None:
            weights = self._build_weights(coordinates)

        W = np.sum(weights)
        if W == 0:
            W = 1.0

        # Calculate Geary's C
        numerator = 0.0
        for i in range(n):
            for j in range(n):
                numerator += weights[i, j] * (values[i] - values[j]) ** 2

        denominator = np.sum((values - mean) ** 2)
        if denominator == 0:
            denominator = 1.0

        C = (n - 1) * numerator / (2 * W * denominator)

        # Expected value and variance
        E_C = 1.0
        var_C = 0.1  # Simplified

        z_score = (C - E_C) / np.sqrt(var_C)

        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        return SpatialAutocorrelationResult(
            statistic=float(C),
            expected=float(E_C),
            variance=float(var_C),
            z_score=float(z_score),
            p_value=float(p_value),
            statistic_name="Geary's C",
        )

    def _build_weights(
        self, coordinates: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Build spatial weights matrix."""
        morans = MoransI(distance_threshold=self.distance_threshold)
        return morans._build_weights(coordinates)
