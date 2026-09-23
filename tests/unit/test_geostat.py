# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_geostat.py
# Title       : Tests of variograms, kriging and spatial autocorrelation
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest, NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Checks unbihexium.geostat against values derived by hand: the model
# variograms at chosen distances (the Matern model with smoothness 1/2 is
# the exponential model), the empirical variogram of a four-point line,
# recovery of a known variogram from a dense simulated field, the kriging
# system far from uncorrelated data (mean and variance sill (1 + 1 / n)),
# exactness at the data and symmetry, universal kriging of a pure trend, inverse distance weighting,
# Moran's I and Geary's C of a four-point line with the normality variance
# computed by hand, and the randomisation variances against the exact
# moments of all permutations.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Exhaustive permutations for the randomisation moments.
import itertools

# Arrays.
import numpy as np

# Test framework.
import pytest

# Kriging under test.
from unbihexium.geostat.kriging import OrdinaryKriging, UniversalKriging, idw

# Spatial statistics under test.
from unbihexium.geostat.spatial import (
    GearysC,  # Geary's C from coordinates.
    MoransI,  # Moran's I from coordinates.
    contiguity_weights,  # Grid weights.
    distance_band_weights,  # Distance band weights.
    gearys_c,  # Geary's C from weights.
    getis_ord_gi_star,  # Hot spots.
    grid_morans_i,  # Raster Moran's I.
    knn_weights,  # Nearest-neighbour weights.
    local_morans_i,  # LISA.
    morans_i,  # Moran's I from weights.
    weight_sums,  # S0, S1, S2.
)  # End of the spatial imports.

# Variograms under test.
from unbihexium.geostat.variogram import (
    Variogram,  # Estimation and fitting.
    VariogramModel,  # Model families.
    empirical_variogram,  # Binned estimate.
    variogram_function,  # Model evaluation.
)  # End of the variogram imports.


# Scattered points with a smooth field and a small noise.
@pytest.fixture
def field() -> tuple[np.ndarray, np.ndarray]:
    # Seeded generator.
    rng = np.random.default_rng(42)
    # 60 points in a 100 x 100 square.
    coords = rng.random((60, 2)) * 100
    # Sine wave plus noise.
    values = np.sin(coords[:, 0] / 10) + rng.random(60) * 0.1
    # Return both.
    return coords, values


# Adjacency weights of four points on a line: 1-2, 2-3, 3-4.
def line_weights() -> np.ndarray:
    # Empty 4 x 4 matrix.
    w = np.zeros((4, 4))
    # Connect consecutive points in both directions.
    for i in range(3):
        # Symmetric link.
        w[i, i + 1] = w[i + 1, i] = 1.0
    # Return the weights.
    return w


# Model values at chosen distances.
def test_variogram_models() -> None:
    # Distances 0, 1 and 2 (beyond the spherical range of 2 at h = 3).
    h = np.array([0.0, 1.0, 2.0, 3.0])
    # Spherical: 0.1 + 1 (1.5 r - 0.5 r^3) with r = h / 2.
    sph = variogram_function("spherical", h, 0.1, 1.0, 2.0)
    # Hand values: 0, 0.1 + 0.6875, 1.1, 1.1.
    assert np.allclose(sph, [0.0, 0.7875, 1.1, 1.1])
    # Exponential: 0.1 + (1 - exp(-h / 2)).
    exp = variogram_function("exponential", h, 0.1, 1.0, 2.0)
    # Hand values.
    assert np.allclose(exp[1:], 0.1 + 1 - np.exp(-h[1:] / 2))
    # Matern with smoothness 1/2 equals the exponential model.
    assert np.allclose(variogram_function("matern", h, 0.1, 1.0, 2.0, shape=0.5), exp)
    # Gaussian: 0.1 + (1 - exp(-(h / 2)^2)).
    assert variogram_function("gaussian", 2.0, 0.1, 1.0, 2.0) == pytest.approx(1.1 - np.exp(-1))
    # Linear model.
    assert variogram_function("linear", 4.0, 0.0, 1.0, 2.0) == pytest.approx(2.0)
    # Power model with exponent 1.5.
    assert variogram_function("power", 4.0, 0.0, 1.0, 1.0, shape=1.5) == pytest.approx(8.0)
    # Non-positive scales are rejected.
    with pytest.raises(ValueError):
        # Zero scale.
        variogram_function("spherical", h, 0.0, 1.0, 0.0)


# Empirical variogram of four equally spaced points.
def test_empirical_variogram_by_hand() -> None:
    # Points at x = 0, 1, 2, 3 with values 0, 1, 3, 6.
    coords = np.array([[0.0], [1.0], [2.0], [3.0]])
    # Values.
    values = np.array([0.0, 1.0, 3.0, 6.0])
    # Three bins [0, 7/6), [7/6, 7/3), [7/3, 3.5] so that each lag has its own bin.
    lags, gamma, counts = empirical_variogram(coords, values, n_lags=3, max_lag=3.5)
    # Pairs at lags 1, 2 and 3: three, two and one pair.
    assert counts.tolist() == [3, 2, 1]
    # Mean distances of the bins.
    assert np.allclose(lags, [1.0, 2.0, 3.0])
    # Lag 1: (1 + 4 + 9) / 6; lag 2: (9 + 25) / 4; lag 3: 36 / 2.
    assert np.allclose(gamma, [14 / 6, 34 / 4, 18.0])
    # Cressie-Hawkins of lag 3: |6|^2 / (2 (0.457 + 0.494)).
    robust = empirical_variogram(coords, values, 3, 3.5, estimator="cressie")[1]
    # Last bin.
    assert robust[-1] == pytest.approx(36.0 / (2 * (0.457 + 0.494)))


# Fitting recovers a variogram from a field with known structure.
def test_variogram_fit(field: tuple[np.ndarray, np.ndarray]) -> None:
    # Data.
    coords, values = field
    # Fit a spherical model.
    result = Variogram(n_lags=10, model=VariogramModel.SPHERICAL).fit(coords, values)
    # Parameters are admissible.
    assert result.nugget >= 0 and result.sill >= 0 and result.range_param > 0
    # Fitted values accompany the empirical ones.
    assert result.fitted_values is not None and result.fitted_values.shape == result.lags.shape
    # White noise with variance 4 has a pure nugget variogram near 4.
    rng = np.random.default_rng(0)
    # Many points for a precise estimate.
    points, noisy = rng.random((400, 2)), rng.normal(0, 2, 400)
    # Fit an exponential model.
    noise = Variogram(n_lags=5, model="exponential").fit(points, noisy)
    # The total sill is the variance.
    assert noise.total_sill == pytest.approx(4.0, rel=0.15)
    # Predictions are floats for scalars and zero at distance zero.
    variogram = Variogram.from_parameters("spherical", 0.5, 1.0, 10.0)
    # Scalar prediction.
    at_range = variogram.predict(10.0)
    # Nugget plus partial sill at the range.
    assert isinstance(at_range, float) and at_range == pytest.approx(1.5)
    # gamma(0) is zero whatever the nugget.
    assert variogram.predict(0.0) == 0.0


# Kriging far from uncorrelated data gives their mean with variance sill (1 + 1 / n).
def test_kriging_single_point_variance() -> None:
    # Known exponential variogram with sill 1 and scale 10.
    variogram = Variogram.from_parameters("exponential", 0.0, 1.0, 10.0)
    # Three data far from each other (nearly uncorrelated).
    coords = np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0]])
    # Data values.
    model = OrdinaryKriging(variogram).fit(coords, np.array([1.0, 2.0, 3.0]))
    # A target far from all data.
    far = model.predict(np.array([[5000.0, 5000.0]]))
    # Mean of independent data with variance sill (1 + 1 / n).
    assert far.predictions[0] == pytest.approx(2.0, abs=1e-6)
    # Kriging variance.
    assert far.variance[0] == pytest.approx(1.0 + 1.0 / 3.0, abs=1e-6)
    # Exactness at the data.
    at_data = model.predict(coords)
    # Data values with zero variance.
    assert np.allclose(at_data.predictions, [1.0, 2.0, 3.0]) and np.allclose(at_data.variance, 0.0)


# Kriging of a symmetric configuration and a fitted model.
def test_ordinary_kriging(field: tuple[np.ndarray, np.ndarray]) -> None:
    # Two near points and a far one, with a target half-way between the near ones.
    variogram = Variogram.from_parameters("spherical", 0.0, 1.0, 5.0)
    # Model on the three points.
    points = np.array([[0.0, 0.0], [2.0, 0.0], [50.0, 50.0]])
    # Fit with the given variogram.
    model = OrdinaryKriging(variogram).fit(points, [1.0, 3.0, 2.0])
    # By symmetry the mid-point weights of the two near points are equal.
    mid = model.predict(np.array([[1.0, 0.0]])).predictions[0]
    # Equal weights of the near points, and a far value equal to their mean, give 2.
    assert mid == pytest.approx(2.0, abs=1e-6)
    # Fitted model on the field.
    coords, values = field
    # Fit and predict.
    result = OrdinaryKriging().fit(coords, values).predict(np.array([[50.0, 50.0], [25.0, 75.0]]))
    # Two predictions with non-negative variances.
    assert result.predictions.shape == (2,) and np.all(result.variance >= 0)
    # Standard deviation property.
    assert np.allclose(result.std**2, result.variance)
    # Leave-one-out cross-validation of the field.
    scores = OrdinaryKriging().fit(coords, values).cross_validate()
    # Errors are small compared with the field amplitude of 1.
    assert scores["rmse"] < 0.3 and abs(scores["bias"]) < 0.1
    # Local kriging with 12 neighbours gives similar errors.
    local = OrdinaryKriging(n_neighbors=12).fit(coords, values).cross_validate(k_folds=5)
    # Similar accuracy.
    assert local["rmse"] < 0.3
    # Duplicate coordinates are rejected.
    with pytest.raises(ValueError):
        # Two identical points.
        OrdinaryKriging().fit(np.array([[0, 0], [0, 0], [1, 1.0]]), [1, 2, 3])


# Universal kriging reproduces a linear trend exactly.
def test_universal_kriging_trend() -> None:
    # Grid of data points.
    gx, gy = np.meshgrid(np.arange(0.0, 10.0, 2.0), np.arange(0.0, 10.0, 2.0))
    # Coordinates.
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    # Pure linear trend z = 2 + 3 x - y.
    values = 2.0 + 3.0 * coords[:, 0] - coords[:, 1]
    # Residual variogram given explicitly.
    variogram = Variogram.from_parameters("exponential", 0.0, 1.0, 3.0)
    # Linear drift model.
    model = UniversalKriging(variogram, drift_terms=1).fit(coords, values)
    # Targets inside and outside the data.
    targets = np.array([[3.3, 4.1], [12.0, -3.0]])
    # The trend is reproduced exactly, even when extrapolating.
    trend = 2.0 + 3.0 * targets[:, 0] - targets[:, 1]
    # Predictions equal the trend.
    assert np.allclose(model.predict(targets).predictions, trend)
    # Invalid drift orders are rejected.
    with pytest.raises(ValueError):
        # Cubic drift.
        UniversalKriging(drift_terms=3)


# Inverse distance weighting by hand.
def test_idw() -> None:
    # Two points at x = 0 and x = 2 with values 0 and 10.
    coords = np.array([[0.0, 0.0], [2.0, 0.0]])
    # Values.
    values = np.array([0.0, 10.0])
    # The mid-point gets the mean.
    assert idw(coords, values, [[1.0, 0.0]])[0] == pytest.approx(5.0)
    # At x = 0.5 the weights are 1 / 0.25 and 1 / 2.25: 10 * 4 / (4 + 4 / 9) = 9.
    assert idw(coords, values, [[0.5, 0.0]])[0] == pytest.approx(10.0 * (1 / 2.25) / (4 + 1 / 2.25))
    # Data points are reproduced.
    assert np.allclose(idw(coords, values, coords), values)
    # One neighbour gives nearest-neighbour interpolation.
    assert idw(coords, values, [[1.6, 0.0]], n_neighbors=1)[0] == pytest.approx(10.0)


# Moran's I and Geary's C of a four-point line computed by hand.
def test_morans_i_gearys_c_by_hand() -> None:
    # Values 1, 2, 3, 4 on a line: deviations -1.5, -0.5, 0.5, 1.5.
    values = np.array([1.0, 2.0, 3.0, 4.0])
    # Binary weights: S0 = 6, S1 = 12, S2 = 40.
    w = line_weights()
    # Weight sums.
    assert weight_sums(w) == (6.0, 12.0, 40.0)
    # Moran's I = (4 / 6) * 2.5 / 5 = 1/3.
    moran = morans_i(values, w, row_standardized=False, assumption="normality")
    # Statistic and expectation.
    assert moran.statistic == pytest.approx(1 / 3) and moran.expected == pytest.approx(-1 / 3)
    # Normality variance (16 * 12 - 4 * 40 + 3 * 36) / (15 * 36) - 1/9 = 4/27.
    assert moran.variance == pytest.approx(4 / 27)
    # Geary's C = 3 * 6 / (2 * 6 * 5) = 0.3.
    geary = gearys_c(values, w)
    # Statistic.
    assert geary.statistic == pytest.approx(0.3) and geary.statistic_name == "Geary's C"
    # A checkerboard is perfectly negatively autocorrelated with rook weights.
    board = (np.indices((6, 6)).sum(axis=0) % 2).astype(float)
    # Moran's I of -1.
    assert grid_morans_i(board, "rook").statistic == pytest.approx(-1.0)


# Randomisation variances equal the exact moments over all permutations.
def test_randomization_variances_exact() -> None:
    # Six values and random asymmetric weights.
    rng = np.random.default_rng(3)
    # Values.
    values = rng.random(6) * 10
    # Weights with some zeros.
    w = rng.random((6, 6)) * (rng.random((6, 6)) > 0.4)
    # Statistics of every permutation of the values.
    perms = [values[list(p)] for p in itertools.permutations(range(6))]
    # Moran's I of every permutation.
    moran = np.array([morans_i(v, w, row_standardized=False).statistic for v in perms])
    # Geary's C of every permutation.
    geary = np.array([gearys_c(v, w).statistic for v in perms])
    # Analytical moments.
    rm, rc = morans_i(values, w, row_standardized=False), gearys_c(values, w)
    # Moran: mean and variance of the permutation distribution.
    assert moran.mean() == pytest.approx(rm.expected) and moran.var() == pytest.approx(rm.variance)
    # Geary: mean and variance of the permutation distribution.
    assert geary.mean() == pytest.approx(rc.expected) and geary.var() == pytest.approx(rc.variance)


# Raster Moran's I equals the dense computation and detects clustering.
def test_grid_morans_i(field: tuple[np.ndarray, np.ndarray]) -> None:
    # Random raster with a nodata cell.
    a = np.random.default_rng(4).random((5, 6))
    # Nodata.
    a[1, 2] = np.nan
    # Valid cells.
    valid = np.isfinite(a).ravel()
    # Dense queen weights of the valid cells.
    w = contiguity_weights(a.shape, "queen")[np.ix_(valid, valid)]
    # Dense result.
    dense = morans_i(a.ravel()[valid], w, row_standardized=False)
    # Shift-based result.
    grid = grid_morans_i(a, "queen")
    # Same statistic and variance.
    assert grid.statistic == pytest.approx(dense.statistic)
    # Same variance.
    assert grid.variance == pytest.approx(dense.variance)
    # The smooth field is positively autocorrelated with distance band weights.
    coords, values = field
    # Coordinate-based statistics.
    result = MoransI(distance_threshold=20.0, permutations=99, seed=0).calculate(coords, values)
    # Significant positive autocorrelation.
    assert result.statistic > 0.3 and result.p_value < 0.01 and result.p_value_permutation <= 0.02
    # Geary's C below 1 for positive autocorrelation.
    assert GearysC(distance_threshold=20.0).calculate(coords, values).statistic < 1.0


# Weight builders.
def test_weights() -> None:
    # Four points on a line with unit spacing.
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    # Default band gives every point a neighbour: the line adjacency.
    assert np.array_equal(distance_band_weights(coords), line_weights())
    # Inverse distance weights within 2 units.
    inv = distance_band_weights(coords, 2.0, binary=False)
    # Weight of distance 2 is 1/2.
    assert inv[0, 2] == pytest.approx(0.5) and inv[0, 3] == 0.0
    # One nearest neighbour of the end points.
    k1 = knn_weights(coords, 1)
    # Each row has one neighbour.
    assert k1.sum(axis=1).tolist() == [1.0, 1.0, 1.0, 1.0] and k1[0, 1] == 1.0
    # Rook weights of a 2 x 2 grid: each cell has two neighbours.
    assert contiguity_weights((2, 2), "rook").sum(axis=1).tolist() == [2.0, 2.0, 2.0, 2.0]


# Local statistics: Gi* by hand and LISA quadrants.
def test_local_statistics() -> None:
    # Five values on a line, high at one end.
    values = np.array([1.0, 1.0, 1.0, 5.0, 6.0])
    # Line adjacency of five points.
    w = contiguity_weights((1, 5), "rook")
    # Gi* of the last point by hand: neighbours {3, 4} with the star.
    gi = getis_ord_gi_star(values, w)
    # Mean 2.8 and population standard deviation.
    mean, s = values.mean(), values.std()
    # (5 + 6 - 2 * 2.8) / (s sqrt((5 * 2 - 4) / 4)).
    expected = (11.0 - 2 * mean) / (s * np.sqrt((5 * 2 - 4) / 4))
    # Compare.
    assert gi.statistic[4] == pytest.approx(expected)
    # The high end is a hot spot, the low end a cold spot.
    assert gi.z_score[4] > 0 > gi.z_score[0]
    # LISA of a monotone sequence.
    lisa = local_morans_i(np.arange(10.0), contiguity_weights((1, 10), "rook"), seed=0)
    # Low values next to low values (3) and high next to high (1).
    assert lisa.quadrant.tolist() == [3] * 5 + [1] * 5
    # Positive local association at the ends.
    assert lisa.statistic[0] > 0 and lisa.statistic[-1] > 0


# =============================================================================
# End of module tests/unit/test_geostat.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
