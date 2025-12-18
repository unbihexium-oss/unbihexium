"""Tests for geostatistics module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


class TestVariogram:
    """Tests for variogram analysis."""

    def test_fit_variogram(
        self,
        sample_coordinates: NDArray[np.floating[Any]],
        sample_values: NDArray[np.floating[Any]],
    ) -> None:
        from unbihexium.geostat.variogram import Variogram, VariogramModel

        variogram = Variogram(n_lags=10, model=VariogramModel.SPHERICAL)
        result = variogram.fit(sample_coordinates, sample_values)

        assert result.lags is not None
        assert result.semivariance is not None
        assert result.nugget >= 0
        assert result.sill >= 0
        assert result.range_param >= 0

    def test_predict_after_fit(
        self,
        sample_coordinates: NDArray[np.floating[Any]],
        sample_values: NDArray[np.floating[Any]],
    ) -> None:
        from unbihexium.geostat.variogram import Variogram

        variogram = Variogram()
        variogram.fit(sample_coordinates, sample_values)

        sv = variogram.predict(10.0)
        assert isinstance(sv, float)
        assert sv >= 0


class TestKriging:
    """Tests for kriging interpolation."""

    def test_ordinary_kriging_predict(
        self,
        sample_coordinates: NDArray[np.floating[Any]],
        sample_values: NDArray[np.floating[Any]],
    ) -> None:
        from unbihexium.geostat.kriging import OrdinaryKriging

        ok = OrdinaryKriging()
        ok.fit(sample_coordinates, sample_values)

        targets = np.array([[50.0, 50.0], [25.0, 75.0]])
        result = ok.predict(targets)

        assert len(result.predictions) == 2
        assert len(result.variance) == 2


class TestSpatialAutocorrelation:
    """Tests for Moran's I and Geary's C."""

    def test_morans_i(
        self,
        sample_coordinates: NDArray[np.floating[Any]],
        sample_values: NDArray[np.floating[Any]],
    ) -> None:
        from unbihexium.geostat.spatial import MoransI

        moran = MoransI(distance_threshold=20.0)
        result = moran.calculate(sample_coordinates, sample_values)

        assert result.statistic_name == "Moran's I"
        assert -1 <= result.statistic <= 1

    def test_gearys_c(
        self,
        sample_coordinates: NDArray[np.floating[Any]],
        sample_values: NDArray[np.floating[Any]],
    ) -> None:
        from unbihexium.geostat.spatial import GearysC

        geary = GearysC(distance_threshold=20.0)
        result = geary.calculate(sample_coordinates, sample_values)

        assert result.statistic_name == "Geary's C"
        assert result.statistic >= 0
