# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/geostat/spatial.py
# Title       : Spatial weights and spatial autocorrelation statistics
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Spatial weights and autocorrelation statistics of values x_i at n
# locations, with deviations z_i = x_i - mean(x):
#
#   distance_band_weights, knn_weights, contiguity_weights, row_standardize
#   MoransI, morans_i     I = (n / S0) sum_ij w_ij z_i z_j / sum_i z_i^2
#   GearysC, gearys_c     C = (n - 1) sum_ij w_ij (x_i - x_j)^2
#                             / (2 S0 sum_i z_i^2)
#   grid_morans_i         Moran's I of a raster with rook or queen
#                         contiguity, computed with array shifts
#   getis_ord_gi_star     local hot and cold spots (Gi* z-scores)
#   local_morans_i        local indicators of spatial association (LISA)
#
# Inference
# ---------
# With S0 = sum_ij w_ij, S1 = 1/2 sum_ij (w_ij + w_ji)^2,
# S2 = sum_i (w_i. + w_.i)^2 and the sample kurtosis
# b2 = n sum z^4 / (sum z^2)^2, the moments under the null hypothesis of no
# autocorrelation are (Cliff and Ord, 1981):
#
#   E[I] = -1 / (n - 1)
#   Var_N[I] = (n^2 S1 - n S2 + 3 S0^2) / ((n^2 - 1) S0^2) - E[I]^2
#   Var_R[I] = (n ((n^2 - 3n + 3) S1 - n S2 + 3 S0^2)
#               - b2 ((n^2 - n) S1 - 2 n S2 + 6 S0^2))
#              / ((n - 1)(n - 2)(n - 3) S0^2) - E[I]^2
#   E[C] = 1
#   Var_N[C] = ((2 S1 + S2)(n - 1) - 4 S0^2) / (2 (n + 1) S0^2)
#   Var_R[C] = ((n - 1) S1 (n^2 - 3n + 3 - (n - 1) b2)
#               - (n - 1) S2 (n^2 + 3n - 6 - (n^2 - n + 2) b2) / 4
#               + S0^2 (n^2 - 3 - (n - 1)^2 b2)) / (n (n - 2)(n - 3) S0^2)
#
# N assumes normally distributed values, R the randomisation of the
# observed values over the locations; Var_R is the exact variance of the
# permutation distribution. p-values are two-sided normal approximations;
# with `permutations` > 0 a pseudo p-value (one-sided in the direction of
# the observed statistic, (k + 1) / (permutations + 1)) is added.
#
# References
# ----------
# Moran, P. A. P. (1950). Notes on continuous stochastic phenomena.
#   Biometrika, 37(1-2), 17-23.
# Geary, R. C. (1954). The contiguity ratio and statistical mapping. The
#   Incorporated Statistician, 5(3), 115-145.
# Cliff, A. D., Ord, J. K. (1981). Spatial Processes: Models and
#   Applications. Pion, London.
# Getis, A., Ord, J. K. (1992). The analysis of spatial association by use
#   of distance statistics. Geographical Analysis, 24(3), 189-206.
# Ord, J. K., Getis, A. (1995). Local spatial autocorrelation statistics:
#   distributional issues and an application. Geographical Analysis, 27(4),
#   286-306.
# Anselin, L. (1995). Local indicators of spatial association: LISA.
#   Geographical Analysis, 27(2), 93-115.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Result records.
from dataclasses import dataclass

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Nearest-neighbour search.
from scipy.spatial import cKDTree

# Distance matrices.
from scipy.spatial.distance import cdist

# Normal distribution for p-values.
from scipy.stats import norm


# Result of a global autocorrelation statistic.
@dataclass
class SpatialAutocorrelationResult:
    # Observed statistic.
    statistic: float
    # Expected value under the null hypothesis.
    expected: float
    # Variance under the null hypothesis.
    variance: float
    # Standardised statistic.
    z_score: float
    # Two-sided p-value of the normal approximation.
    p_value: float
    # Name of the statistic.
    statistic_name: str
    # Pseudo p-value of the permutation test, when requested.
    p_value_permutation: float | None = None


# Result of a local statistic.
@dataclass
class LocalStatisticResult:
    # Statistic per location.
    statistic: NDArray[np.floating[Any]]
    # z-score per location.
    z_score: NDArray[np.floating[Any]]
    # p-value per location.
    p_value: NDArray[np.floating[Any]]
    # Name of the statistic.
    statistic_name: str
    # Moran scatter plot quadrant: 1 HH, 2 LH, 3 LL, 4 HL (LISA only).
    quadrant: NDArray[np.integer[Any]] | None = None


# Values as a finite float64 vector.
def _values(values: NDArray[Any], minimum: int = 4) -> NDArray[np.float64]:
    # Float64 vector.
    x = np.asarray(values, dtype=np.float64).ravel()
    # Values must be finite.
    if not np.all(np.isfinite(x)):
        # Report the invalid data.
        raise ValueError("values must be finite; remove NaN locations first")
    # Enough locations for the variance formulas.
    if x.size < minimum:
        # Report the too small sample.
        raise ValueError(f"at least {minimum} locations are needed, got {x.size}")
    # Return the vector.
    return x


# Weights as a float64 (n, n) matrix without self-weights.
def _weights(weights: NDArray[Any], n: int) -> NDArray[np.float64]:
    # Float64 matrix.
    w = np.array(weights, dtype=np.float64)
    # One row and column per location.
    if w.shape != (n, n):
        # Report the mismatch.
        raise ValueError(f"weights must have shape ({n}, {n}), got {w.shape}")
    # Weights must be finite and non-negative.
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        # Report the invalid weights.
        raise ValueError("weights must be finite and non-negative")
    # Remove self-weights.
    np.fill_diagonal(w, 0.0)
    # Return the matrix.
    return w


# Divide each row by its sum; rows without neighbours stay zero.
def row_standardize(weights: NDArray[Any]) -> NDArray[np.float64]:
    # Float64 matrix.
    w = np.asarray(weights, dtype=np.float64)
    # Row sums.
    sums = w.sum(axis=1, keepdims=True)
    # Divide where the row has neighbours.
    return np.divide(w, sums, out=np.zeros_like(w), where=sums > 0)


# Coordinates as an (n, d) float64 matrix.
def _coordinates(coordinates: NDArray[Any]) -> NDArray[np.float64]:
    # Float64 matrix.
    c = np.asarray(coordinates, dtype=np.float64)
    # Columns of positions.
    if c.ndim == 1:
        # One coordinate per location.
        c = c[:, None]
    # Return the matrix.
    return c


# Smallest distance band that gives every location at least one neighbour.
def min_threshold_distance(coordinates: NDArray[Any]) -> float:
    # Coordinates.
    c = _coordinates(coordinates)
    # Distance to the nearest other location.
    dist, _ = cKDTree(c).query(c, k=2)
    # Largest nearest-neighbour distance.
    return float(dist[:, 1].max())


# Distance band weights: binary, or inverse distance to a power, within a threshold.
def distance_band_weights(
    coordinates: NDArray[Any],  # Locations (n, d).
    threshold: float | None = None,  # Band width; None gives every location a neighbour.
    binary: bool = True,  # 1 inside the band, or d^-power.
    power: float = 1.0,  # Exponent of the inverse distance weights.
) -> NDArray[np.float64]:  # Symmetric (n, n) weights.
    # Coordinates.
    c = _coordinates(coordinates)
    # Default band.
    band = min_threshold_distance(c) if threshold is None else float(threshold)
    # Pairwise distances.
    d = cdist(c, c)
    # Pairs inside the band, excluding self-pairs.
    inside = (d <= band) & (d > 0)
    # Binary weights.
    if binary:
        # One inside the band.
        return inside.astype(np.float64)
    # Inverse distance weights inside the band.
    return np.where(inside, np.power(np.where(inside, d, 1.0), -power), 0.0)


# k-nearest-neighbour binary weights (not symmetric in general).
def knn_weights(coordinates: NDArray[Any], k: int) -> NDArray[np.float64]:
    # Coordinates.
    c = _coordinates(coordinates)
    # Number of locations.
    n = c.shape[0]
    # k must leave at least one other location.
    if not 1 <= k < n:
        # Report the invalid k.
        raise ValueError(f"k must lie in [1, {n - 1}], got {k}")
    # Nearest k + 1 locations, the first being the location itself.
    _, idx = cKDTree(c).query(c, k=k + 1)
    # Weight matrix.
    w = np.zeros((n, n))
    # Set the neighbours of every row.
    for i, row in enumerate(idx):
        # Exclude the location itself.
        w[i, [j for j in row if j != i][:k]] = 1.0
    # Return the weights.
    return w


# Binary contiguity weights of the cells of a grid (row-major order).
def contiguity_weights(
    shape: tuple[int, int],  # Grid shape (rows, cols).
    contiguity: str = "rook",  # "rook" (4 neighbours) or "queen" (8 neighbours).
) -> NDArray[np.float64]:  # Dense (rows * cols, rows * cols) weights.
    # Grid size.
    rows, cols = shape
    # Neighbour offsets.
    offsets = _grid_offsets(contiguity)
    # Number of cells.
    n = rows * cols
    # Weight matrix.
    w = np.zeros((n, n))
    # Cell positions.
    r, c = np.divmod(np.arange(n), cols)
    # Add every offset.
    for dr, dc in offsets:
        # Neighbour positions.
        nr, nc = r + dr, c + dc
        # Neighbours inside the grid.
        ok = (nr >= 0) & (nr < rows) & (nc >= 0) & (nc < cols)
        # Set the weights.
        w[np.arange(n)[ok], (nr * cols + nc)[ok]] = 1.0
    # Return the weights.
    return w


# Neighbour offsets of rook or queen contiguity.
def _grid_offsets(contiguity: str) -> list[tuple[int, int]]:
    # Edge neighbours.
    rook = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    # Rook contiguity.
    if contiguity == "rook":
        # Four neighbours.
        return rook
    # Queen contiguity.
    if contiguity == "queen":
        # Eight neighbours.
        return [*rook, (1, 1), (1, -1), (-1, 1), (-1, -1)]
    # Unknown contiguity.
    raise ValueError(f"unknown contiguity {contiguity!r}; expected 'rook' or 'queen'")


# Weight sums S0, S1 and S2.
def weight_sums(weights: NDArray[Any]) -> tuple[float, float, float]:
    # Float64 matrix.
    w = np.asarray(weights, dtype=np.float64)
    # Sum of all weights.
    s0 = float(w.sum())
    # Half the sum of squared symmetric weights.
    s1 = float(0.5 * np.sum((w + w.T) ** 2))
    # Squared sums of row and column totals.
    s2 = float(np.sum((w.sum(axis=1) + w.sum(axis=0)) ** 2))
    # Return the sums.
    return s0, s1, s2


# Pseudo p-value of a permutation test, one-sided in the direction of the observation.
def _pseudo_p(observed: float, simulated: NDArray[np.float64]) -> float:
    # Number of permutations.
    count = simulated.size
    # Permutations at least as large as the observation.
    larger = int(np.sum(simulated >= observed))
    # Use the other tail when the observation lies below most permutations.
    larger = min(larger, count - larger)
    # (k + 1) / (permutations + 1).
    return (larger + 1.0) / (count + 1.0)


# Moran's I from deviations, weight sums and the cross-product sum.
def _moran_moments(
    z: NDArray[np.float64],  # Deviations from the mean.
    cross: float,  # sum_ij w_ij z_i z_j.
    s0: float,  # Sum of weights.
    s1: float,  # S1 weight sum.
    s2: float,  # S2 weight sum.
    assumption: str,  # "randomization" or "normality".
) -> SpatialAutocorrelationResult:  # Statistic, moments and p-values.
    # Number of locations.
    n = z.size
    # Sum of squared deviations.
    m2 = float(np.sum(z * z))
    # Expected value.
    expected = -1.0 / (n - 1)
    # Constant values or no weights carry no information.
    if m2 == 0 or s0 == 0:
        # Undefined statistic.
        return SpatialAutocorrelationResult(np.nan, expected, np.nan, np.nan, np.nan, "Moran's I")
    # Statistic.
    stat = n / s0 * cross / m2
    # Normality assumption.
    if assumption == "normality":
        # Cliff and Ord variance under normality.
        var = (n * n * s1 - n * s2 + 3.0 * s0 * s0) / ((n * n - 1.0) * s0 * s0) - expected**2
    # Randomisation assumption.
    elif assumption == "randomization":
        # Sample kurtosis.
        b2 = n * float(np.sum(z**4)) / m2**2
        # First term of the numerator.
        a = n * ((n * n - 3 * n + 3) * s1 - n * s2 + 3.0 * s0 * s0)
        # Kurtosis term of the numerator.
        b = b2 * ((n * n - n) * s1 - 2.0 * n * s2 + 6.0 * s0 * s0)
        # Cliff and Ord variance under randomisation.
        var = (a - b) / ((n - 1.0) * (n - 2.0) * (n - 3.0) * s0 * s0) - expected**2
    # Unknown assumption.
    else:
        # Report the valid names.
        raise ValueError(f"unknown assumption {assumption!r}; expected randomization or normality")
    # Standardised statistic.
    z_score = (stat - expected) / np.sqrt(var) if var > 0 else np.nan
    # Two-sided p-value.
    p = float(2.0 * norm.sf(abs(z_score))) if np.isfinite(z_score) else np.nan
    # Package the result.
    return SpatialAutocorrelationResult(stat, expected, var, float(z_score), p, "Moran's I")


# Global Moran's I of values with a weight matrix.
def morans_i(
    values: NDArray[Any],  # Values (n,).
    weights: NDArray[Any],  # Weights (n, n).
    row_standardized: bool = True,  # Row-standardise the weights first.
    assumption: str = "randomization",  # Null distribution of the variance.
    permutations: int = 0,  # Number of random permutations for a pseudo p-value.
    seed: int | None = None,  # Random seed of the permutations.
) -> SpatialAutocorrelationResult:  # Statistic, moments and p-values.
    # Values.
    x = _values(values)
    # Weights.
    w = _weights(weights, x.size)
    # Optional row standardisation.
    if row_standardized:
        # Rows sum to one.
        w = row_standardize(w)
    # Deviations.
    z = x - x.mean()
    # Weight sums.
    s0, s1, s2 = weight_sums(w)
    # Moments and statistic.
    result = _moran_moments(z, float(z @ w @ z), s0, s1, s2, assumption)
    # Permutation test.
    if permutations > 0 and np.isfinite(result.statistic):
        # Random generator.
        rng = np.random.default_rng(seed)
        # Permuted deviations, one row per permutation.
        zp = np.array([rng.permutation(z) for _ in range(permutations)])
        # Statistic of every permutation.
        sim = x.size / s0 * np.einsum("pi,ij,pj->p", zp, w, zp) / float(np.sum(z * z))
        # Pseudo p-value.
        result.p_value_permutation = _pseudo_p(result.statistic, sim)
    # Return the result.
    return result


# Global Geary's C of values with a weight matrix.
def gearys_c(
    values: NDArray[Any],  # Values (n,).
    weights: NDArray[Any],  # Weights (n, n).
    row_standardized: bool = False,  # Row-standardise the weights first.
    assumption: str = "randomization",  # Null distribution of the variance.
    permutations: int = 0,  # Number of random permutations for a pseudo p-value.
    seed: int | None = None,  # Random seed of the permutations.
) -> SpatialAutocorrelationResult:  # Statistic, moments and p-values.
    # Values.
    x = _values(values)
    # Weights.
    w = _weights(weights, x.size)
    # Optional row standardisation.
    if row_standardized:
        # Rows sum to one.
        w = row_standardize(w)
    # Number of locations.
    n = x.size
    # Deviations.
    z = x - x.mean()
    # Sum of squared deviations.
    m2 = float(np.sum(z * z))
    # Weight sums.
    s0, s1, s2 = weight_sums(w)
    # Constant values or no weights carry no information.
    if m2 == 0 or s0 == 0:
        # Undefined statistic.
        return SpatialAutocorrelationResult(np.nan, 1.0, np.nan, np.nan, np.nan, "Geary's C")

    # Statistic of a value vector.
    def statistic(v: NDArray[np.float64]) -> float:
        # Weighted squared differences of all pairs.
        diff = np.sum(w * (v[:, None] - v[None, :]) ** 2)
        # Normalised ratio.
        return float((n - 1) * diff / (2.0 * s0 * m2))

    # Observed statistic.
    stat = statistic(x)
    # Normality assumption.
    if assumption == "normality":
        # Cliff and Ord variance under normality.
        var = ((2.0 * s1 + s2) * (n - 1) - 4.0 * s0 * s0) / (2.0 * (n + 1) * s0 * s0)
    # Randomisation assumption.
    elif assumption == "randomization":
        # Sample kurtosis.
        b2 = n * float(np.sum(z**4)) / m2**2
        # S1 term.
        t1 = (n - 1) * s1 * (n * n - 3 * n + 3 - (n - 1) * b2)
        # S2 term.
        t2 = 0.25 * (n - 1) * s2 * (n * n + 3 * n - 6 - (n * n - n + 2) * b2)
        # S0 term.
        t3 = s0 * s0 * (n * n - 3 - (n - 1) ** 2 * b2)
        # Cliff and Ord variance under randomisation.
        var = (t1 - t2 + t3) / (n * (n - 2.0) * (n - 3.0) * s0 * s0)
    # Unknown assumption.
    else:
        # Report the valid names.
        raise ValueError(f"unknown assumption {assumption!r}; expected randomization or normality")
    # Standardised statistic.
    z_score = (stat - 1.0) / np.sqrt(var) if var > 0 else np.nan
    # Two-sided p-value.
    p = float(2.0 * norm.sf(abs(z_score))) if np.isfinite(z_score) else np.nan
    # Package the result.
    result = SpatialAutocorrelationResult(stat, 1.0, float(var), float(z_score), p, "Geary's C")
    # Permutation test.
    if permutations > 0:
        # Random generator.
        rng = np.random.default_rng(seed)
        # Statistic of every permutation.
        sim = np.array([statistic(rng.permutation(x)) for _ in range(permutations)])
        # Pseudo p-value.
        result.p_value_permutation = _pseudo_p(stat, sim)
    # Return the result.
    return result


# Moran's I of a raster with binary rook or queen contiguity; NaN cells are excluded.
def grid_morans_i(
    array: NDArray[Any],  # 2-D raster.
    contiguity: str = "queen",  # "rook" or "queen".
    assumption: str = "randomization",  # Null distribution of the variance.
) -> SpatialAutocorrelationResult:  # Statistic, moments and p-values.
    # Raster as float64.
    a = np.asarray(array, dtype=np.float64)
    # Only 2-D rasters are supported.
    if a.ndim != 2:
        # Report the wrong shape.
        raise ValueError(f"expected a 2-D raster, got shape {a.shape}")
    # Valid cells.
    valid = np.isfinite(a)
    # Values of the valid cells.
    x = _values(a[valid])
    # Deviations on the grid, zero at invalid cells.
    z = np.where(valid, a - x.mean(), 0.0)
    # Padded deviations and validity for the shifts.
    zp, vp = np.pad(z, 1), np.pad(valid, 1)
    # Neighbour counts and neighbour sums.
    counts, lag = np.zeros(a.shape), np.zeros(a.shape)
    # Grid size.
    rows, cols = a.shape
    # Visit every neighbour offset.
    for dr, dc in _grid_offsets(contiguity):
        # Shifted validity.
        counts += vp[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        # Shifted deviations.
        lag += zp[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
    # Only valid cells count.
    counts = np.where(valid, counts, 0.0)
    # Binary symmetric weights: S0 = sum k_i, S1 = 2 S0, S2 = 4 sum k_i^2.
    s0 = float(counts.sum())
    # Cross-product sum_ij w_ij z_i z_j.
    cross = float(np.sum(z * lag))
    # Moments and statistic.
    return _moran_moments(z[valid], cross, s0, 2.0 * s0, 4.0 * float(np.sum(counts**2)), assumption)


# Getis-Ord Gi* z-scores of every location (Ord and Getis, 1995).
def getis_ord_gi_star(
    values: NDArray[Any],  # Values (n,).
    weights: NDArray[Any],  # Weights (n, n); the diagonal is set to 1 (the star).
) -> LocalStatisticResult:  # Statistics per location.
    # Values.
    x = _values(values, minimum=3)
    # Number of locations.
    n = x.size
    # Weights without self-weights.
    w = _weights(weights, n)
    # The star statistic includes the location itself.
    np.fill_diagonal(w, 1.0)
    # Mean and standard deviation (population form).
    mean, s = x.mean(), float(np.sqrt(np.mean(x * x) - x.mean() ** 2))
    # Constant values have no hot spots.
    if s == 0:
        # Report the degenerate data.
        raise ValueError("values are constant; Gi* is undefined")
    # Sum of weights per location.
    wi = w.sum(axis=1)
    # Sum of squared weights per location.
    s1i = np.sum(w * w, axis=1)
    # Denominator s sqrt((n S1i - Wi^2) / (n - 1)).
    den = s * np.sqrt((n * s1i - wi * wi) / (n - 1.0))
    # Gi* is already a z-score.
    with np.errstate(divide="ignore", invalid="ignore"):
        # (sum_j w_ij x_j - mean W_i) / den.
        g = (w @ x - mean * wi) / den
    # Two-sided p-values.
    p = 2.0 * norm.sf(np.abs(g))
    # Package the result.
    return LocalStatisticResult(statistic=g, z_score=g, p_value=p, statistic_name="Gi*")


# Local Moran's I with conditional permutation inference (Anselin, 1995).
def local_morans_i(
    values: NDArray[Any],  # Values (n,).
    weights: NDArray[Any],  # Weights (n, n).
    row_standardized: bool = True,  # Row-standardise the weights first.
    permutations: int = 999,  # Conditional permutations per location; 0 skips inference.
    seed: int | None = None,  # Random seed.
) -> LocalStatisticResult:  # Statistics per location.
    # Values.
    x = _values(values, minimum=3)
    # Number of locations.
    n = x.size
    # Weights.
    w = _weights(weights, n)
    # Optional row standardisation.
    if row_standardized:
        # Rows sum to one.
        w = row_standardize(w)
    # Deviations.
    z = x - x.mean()
    # Second moment with divisor n.
    m2 = float(np.mean(z * z))
    # Constant values have no local association.
    if m2 == 0:
        # Report the degenerate data.
        raise ValueError("values are constant; local Moran's I is undefined")
    # Spatial lag of the deviations.
    lag = w @ z
    # Local statistics.
    stat = z / m2 * lag
    # Quadrants: 1 HH, 2 LH, 3 LL, 4 HL.
    quadrant = np.where(z > 0, np.where(lag > 0, 1, 4), np.where(lag > 0, 2, 3)).astype(np.int64)
    # Without permutations, no inference.
    zs, ps = np.full(n, np.nan), np.full(n, np.nan)
    # Conditional permutation of the other values.
    if permutations > 0:
        # Random generator.
        rng = np.random.default_rng(seed)
        # Visit every location.
        for i in range(n):
            # Neighbours and their weights.
            nbr = np.flatnonzero(w[i])
            # Locations without neighbours stay undefined.
            if nbr.size == 0:
                # Next location.
                continue
            # Deviations of the other locations.
            others = np.delete(z, i)
            # Random draws without replacement: first k of a random ordering per permutation.
            keys = rng.random((permutations, n - 1))
            # Indices of the drawn values.
            draws = np.argpartition(keys, nbr.size - 1, axis=1)[:, : nbr.size]
            # Simulated statistics.
            sim = z[i] / m2 * (others[draws] @ w[i, nbr])
            # Standardise with the permutation moments.
            sd = sim.std()
            # z-score of the observation.
            zs[i] = (stat[i] - sim.mean()) / sd if sd > 0 else np.nan
            # Pseudo p-value.
            ps[i] = _pseudo_p(stat[i], sim)
    # Package the result.
    return LocalStatisticResult(stat, zs, ps, "local Moran's I", quadrant)


# Moran's I with weights built from coordinates.
class MoransI:
    # Configure the weights and the inference.
    def __init__(
        self,  # The instance.
        distance_threshold: float | None = None,  # Distance band; None gives everyone a neighbour.
        row_standardized: bool = True,  # Row-standardise the weights.
        assumption: str = "randomization",  # Null distribution of the variance.
        permutations: int = 0,  # Permutations for a pseudo p-value.
        seed: int | None = None,  # Random seed of the permutations.
    ) -> None:  # The constructor returns nothing.
        # Distance band.
        self.distance_threshold = distance_threshold
        # Row standardisation.
        self.row_standardized = row_standardized
        # Variance assumption.
        self.assumption = assumption
        # Permutations.
        self.permutations = permutations
        # Random seed.
        self.seed = seed

    # Binary distance band weights of the coordinates.
    def _build_weights(self, coordinates: NDArray[Any]) -> NDArray[np.float64]:
        # Distance band weights.
        return distance_band_weights(coordinates, self.distance_threshold)

    # Compute the statistic.
    def calculate(
        self,  # The instance.
        coordinates: NDArray[Any] | None,  # Locations (n, d); unused when weights are given.
        values: NDArray[Any],  # Values (n,).
        weights: NDArray[Any] | None = None,  # Weights (n, n).
    ) -> SpatialAutocorrelationResult:  # Statistic, moments and p-values.
        # Weights from the coordinates unless given.
        if weights is None:
            # Coordinates are required then.
            if coordinates is None:
                # Report the missing input.
                raise ValueError("coordinates or weights must be given")
            # Distance band weights.
            weights = self._build_weights(coordinates)
        # Compute Moran's I.
        return morans_i(
            values,  # Values.
            weights,  # Weights.
            self.row_standardized,  # Row standardisation.
            self.assumption,  # Variance assumption.
            self.permutations,  # Permutations.
            self.seed,  # Seed.
        )  # End of the call.


# Geary's C with weights built from coordinates.
class GearysC(MoransI):
    # Configure the weights and the inference; binary weights by default.
    def __init__(
        self,  # The instance.
        distance_threshold: float | None = None,  # Distance band; None gives everyone a neighbour.
        row_standardized: bool = False,  # Row-standardise the weights.
        assumption: str = "randomization",  # Null distribution of the variance.
        permutations: int = 0,  # Permutations for a pseudo p-value.
        seed: int | None = None,  # Random seed of the permutations.
    ) -> None:  # The constructor returns nothing.
        # Common configuration.
        super().__init__(distance_threshold, row_standardized, assumption, permutations, seed)

    # Compute the statistic.
    def calculate(
        self,  # The instance.
        coordinates: NDArray[Any] | None,  # Locations (n, d); unused when weights are given.
        values: NDArray[Any],  # Values (n,).
        weights: NDArray[Any] | None = None,  # Weights (n, n).
    ) -> SpatialAutocorrelationResult:  # Statistic, moments and p-values.
        # Weights from the coordinates unless given.
        if weights is None:
            # Coordinates are required then.
            if coordinates is None:
                # Report the missing input.
                raise ValueError("coordinates or weights must be given")
            # Distance band weights.
            weights = self._build_weights(coordinates)
        # Compute Geary's C.
        return gearys_c(
            values,  # Values.
            weights,  # Weights.
            self.row_standardized,  # Row standardisation.
            self.assumption,  # Variance assumption.
            self.permutations,  # Permutations.
            self.seed,  # Seed.
        )  # End of the call.


# =============================================================================
# End of module src/unbihexium/geostat/spatial.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
