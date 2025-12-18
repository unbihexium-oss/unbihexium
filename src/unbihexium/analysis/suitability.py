"""Suitability analysis with AHP and weighted overlay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unbihexium.core.raster import Raster


@dataclass
class SuitabilityResult:
    """Result of suitability analysis."""

    raster: Raster
    weights: dict[str, float]
    consistency_ratio: float | None = None


class AHP:
    """Analytic Hierarchy Process for weight derivation.

    The AHP method derives weights from pairwise comparison matrices.
    The consistency ratio (CR) should be < 0.1 for acceptable consistency.

    $$CR = \\frac{CI}{RI}$$

    where CI is the consistency index and RI is the random index.
    """

    # Random Index values for n=1 to 10
    RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]

    def __init__(self) -> None:
        self._weights: NDArray[np.floating[Any]] | None = None
        self._cr: float | None = None

    def fit(self, comparison_matrix: NDArray[np.floating[Any]]) -> AHP:
        """Derive weights from pairwise comparison matrix.

        Args:
            comparison_matrix: Square matrix of pairwise comparisons.
                Values indicate relative importance (1-9 scale).

        Returns:
            Self for chaining.
        """
        n = comparison_matrix.shape[0]

        # Calculate priority vector (eigenvector method)
        eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
        max_idx = np.argmax(eigenvalues.real)
        weights = eigenvectors[:, max_idx].real
        weights = weights / weights.sum()

        self._weights = weights

        # Calculate consistency ratio
        lambda_max = eigenvalues[max_idx].real
        ci = (lambda_max - n) / (n - 1)
        ri = self.RI[n] if n < len(self.RI) else 1.49
        self._cr = ci / ri if ri > 0 else 0

        return self

    @property
    def weights(self) -> NDArray[np.floating[Any]] | None:
        return self._weights

    @property
    def consistency_ratio(self) -> float | None:
        return self._cr

    def is_consistent(self, threshold: float = 0.1) -> bool:
        """Check if the comparison matrix is consistent."""
        return self._cr is not None and self._cr < threshold


def weighted_overlay(
    layers: list[Raster],
    weights: list[float],
    normalize: bool = True,
) -> SuitabilityResult:
    """Perform weighted overlay analysis.

    $$S = \\sum_{i=1}^{n} w_i \\cdot r_i$$

    Args:
        layers: List of raster layers (must have same dimensions).
        weights: List of weights for each layer.
        normalize: Whether to normalize layers to 0-1 range.

    Returns:
        SuitabilityResult with combined suitability raster.
    """
    if len(layers) != len(weights):
        msg = "Number of layers must equal number of weights"
        raise ValueError(msg)

    if not layers:
        msg = "At least one layer required"
        raise ValueError(msg)

    # Load all layers
    for layer in layers:
        layer.load()

    # Normalize weights
    weights_arr = np.array(weights)
    weights_arr = weights_arr / weights_arr.sum()

    # Get output shape from first layer
    first = layers[0]
    if first.data is None:
        msg = "First layer has no data"
        raise ValueError(msg)

    shape = first.data.shape[-2:]
    result = np.zeros(shape, dtype=np.float32)

    for layer, weight in zip(layers, weights_arr):
        if layer.data is None:
            continue

        data = layer.data[0] if layer.data.ndim == 3 else layer.data

        if normalize:
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)

        result += weight * data

    result_raster = Raster.from_array(
        result[np.newaxis, ...],
        crs=first.metadata.crs if first.metadata else "EPSG:4326",
    )

    weight_dict = {f"layer_{i}": float(w) for i, w in enumerate(weights_arr)}

    return SuitabilityResult(
        raster=result_raster,
        weights=weight_dict,
    )
