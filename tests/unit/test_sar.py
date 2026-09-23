# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for SAR (Synthetic Aperture Radar) modules.

This module provides comprehensive tests for:
- SAR amplitude processing (calibration, sigma0, gamma0, speckle filtering)
- Interferometry (coherence, interferogram, phase unwrapping, displacement)
- Polarimetry (Pauli, Freeman-Durden, H-Alpha decompositions)

All tests use synthetic complex arrays to avoid large data dependencies.
Tests are marked research-grade where applicable.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from unbihexium.sar.amplitude import (
    amplitude_to_db,
    calibrate_amplitude,
    compute_gamma0,
    compute_sigma0,
    speckle_filter,
)
from unbihexium.sar.interferometry import (
    InterferometricResult,
    compute_coherence,
    compute_displacement,
    compute_interferogram,
)
from unbihexium.sar.polarimetry import (
    PolarimetricResult,
    freeman_durden_decomposition,
    h_alpha_decomposition,
    pauli_decomposition,
)

# Fixtures


@pytest.fixture
def seed() -> int:
    """Deterministic seed for reproducibility."""
    return 42


@pytest.fixture
def synthetic_amplitude(seed: int) -> NDArray[np.floating]:
    """Generate synthetic SAR amplitude data."""
    np.random.seed(seed)
    # Rayleigh-like amplitude distribution
    return np.abs(np.random.randn(64, 64) + 1j * np.random.randn(64, 64)).astype(np.float32)


@pytest.fixture
def synthetic_slc_pair(seed: int) -> tuple[NDArray, NDArray]:
    """Generate synthetic SLC pair for interferometry."""
    np.random.seed(seed)
    shape = (32, 32)

    # Master SLC with random phase
    phase1 = np.random.rand(*shape) * 2 * np.pi
    slc1 = np.exp(1j * phase1).astype(np.complex64)

    # Slave SLC with correlated phase (high coherence)
    phase2 = phase1 + np.random.rand(*shape) * 0.2
    slc2 = np.exp(1j * phase2).astype(np.complex64)

    return slc1, slc2


@pytest.fixture
def synthetic_quad_pol(seed: int) -> dict[str, NDArray]:
    """Generate synthetic quad-pol SAR data."""
    np.random.seed(seed)
    shape = (24, 24)

    return {
        "HH": (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex64),
        "HV": (0.3 * np.random.randn(*shape) + 1j * 0.3 * np.random.randn(*shape)).astype(
            np.complex64
        ),
        "VH": (0.3 * np.random.randn(*shape) + 1j * 0.3 * np.random.randn(*shape)).astype(
            np.complex64
        ),
        "VV": (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex64),
    }


# Amplitude Processing Tests


class TestAmplitudeToDb:
    """Tests for amplitude to dB conversion."""

    def test_amplitude_to_db_shape(self, synthetic_amplitude: NDArray[np.floating]) -> None:
        """Test that output shape matches input."""
        db = amplitude_to_db(synthetic_amplitude)
        assert db.shape == synthetic_amplitude.shape

    def test_amplitude_to_db_finite(self, synthetic_amplitude: NDArray[np.floating]) -> None:
        """Test that output contains only finite values."""
        db = amplitude_to_db(synthetic_amplitude)
        assert np.all(np.isfinite(db))

    def test_amplitude_to_db_floor(self) -> None:
        """Test dB floor for zero/small values."""
        amplitude = np.array([0.0, 1e-15, 0.001], dtype=np.float32)
        db = amplitude_to_db(amplitude, floor=-40.0)

        # Very small values should be floored
        assert db[0] >= -40.0
        assert db[1] >= -40.0

    def test_amplitude_to_db_reference(self) -> None:
        """Test dB conversion against known values."""
        # Amplitude of 1 should give 0 dB
        amplitude = np.array([1.0], dtype=np.float32)
        db = amplitude_to_db(amplitude)
        assert db[0] == pytest.approx(0.0, abs=0.1)

        # Amplitude of 10 should give 20 dB
        amplitude = np.array([10.0], dtype=np.float32)
        db = amplitude_to_db(amplitude)
        assert db[0] == pytest.approx(20.0, abs=0.1)


class TestCalibrateAmplitude:
    """Tests for SAR amplitude calibration."""

    def test_calibrate_amplitude_shape(self, synthetic_amplitude: NDArray[np.floating]) -> None:
        """Test that output shape matches input."""
        calibrated = calibrate_amplitude(synthetic_amplitude, calibration_factor=1.0)
        assert calibrated.shape == synthetic_amplitude.shape

    def test_calibrate_amplitude_factor(self) -> None:
        """Test calibration factor application."""
        amplitude = np.ones((10, 10), dtype=np.float32)
        calibrated = calibrate_amplitude(amplitude, calibration_factor=2.0)
        assert np.allclose(calibrated, 2.0)


class TestComputeSigma0:
    """Tests for sigma0 computation."""

    def test_sigma0_shape(self, synthetic_amplitude: NDArray[np.floating]) -> None:
        """Test that output shape matches input."""
        sigma0 = compute_sigma0(synthetic_amplitude, incidence_angle=30.0)
        assert sigma0.shape == synthetic_amplitude.shape

    def test_sigma0_positive(self, synthetic_amplitude: NDArray[np.floating]) -> None:
        """Test that sigma0 is non-negative."""
        sigma0 = compute_sigma0(synthetic_amplitude, incidence_angle=30.0)
        assert np.all(sigma0 >= 0)


class TestComputeGamma0:
    """Tests for gamma0 computation."""

    def test_gamma0_shape(self, synthetic_amplitude: NDArray[np.floating]) -> None:
        """Test that output shape matches input."""
        sigma0 = compute_sigma0(synthetic_amplitude, incidence_angle=30.0)
        gamma0 = compute_gamma0(sigma0, incidence_angle=30.0)
        assert gamma0.shape == synthetic_amplitude.shape

    def test_gamma0_positive(self, synthetic_amplitude: NDArray[np.floating]) -> None:
        """Test that gamma0 is non-negative."""
        sigma0 = compute_sigma0(synthetic_amplitude, incidence_angle=30.0)
        gamma0 = compute_gamma0(sigma0, incidence_angle=30.0)
        assert np.all(gamma0 >= 0)


class TestSpeckleFilter:
    """Tests for speckle filtering."""

    def test_speckle_filter_shape(self, synthetic_amplitude: NDArray[np.floating]) -> None:
        """Test that output shape matches input."""
        filtered = speckle_filter(synthetic_amplitude, window_size=5)
        assert filtered.shape == synthetic_amplitude.shape

    def test_speckle_filter_smoothing(self, synthetic_amplitude: NDArray[np.floating]) -> None:
        """Test that filtering reduces variance."""
        filtered = speckle_filter(synthetic_amplitude, window_size=5)
        # Filtered image should have lower std (smoother)
        assert np.std(filtered) <= np.std(synthetic_amplitude)

    def test_speckle_filter_finite(self, synthetic_amplitude: NDArray[np.floating]) -> None:
        """Test that output contains only finite values."""
        filtered = speckle_filter(synthetic_amplitude, window_size=3)
        assert np.all(np.isfinite(filtered))


# Interferometry Tests


class TestComputeCoherence:
    """Tests for interferometric coherence computation."""

    def test_coherence_shape(self, synthetic_slc_pair: tuple[NDArray, NDArray]) -> None:
        """Test that output shape matches input."""
        slc1, slc2 = synthetic_slc_pair
        coherence = compute_coherence(slc1, slc2, window_size=3)
        assert coherence.shape == slc1.shape

    def test_coherence_range(self, synthetic_slc_pair: tuple[NDArray, NDArray]) -> None:
        """Test that coherence is in [0, 1] range."""
        slc1, slc2 = synthetic_slc_pair
        coherence = compute_coherence(slc1, slc2, window_size=3)
        assert np.all(coherence >= 0)
        assert np.all(coherence <= 1)

    def test_coherence_perfect(self) -> None:
        """Test coherence for identical SLCs."""
        slc = np.exp(1j * np.random.rand(16, 16) * 2 * np.pi).astype(np.complex64)
        coherence = compute_coherence(slc, slc, window_size=3)
        # Self-coherence should be 1
        assert np.mean(coherence) > 0.9


class TestComputeInterferogram:
    """Tests for interferogram computation."""

    def test_interferogram_shape(self, synthetic_slc_pair: tuple[NDArray, NDArray]) -> None:
        """Test that output shape matches input."""
        slc1, slc2 = synthetic_slc_pair
        phase, ifg = compute_interferogram(slc1, slc2)
        assert phase.shape == slc1.shape
        assert ifg.shape == slc1.shape

    def test_interferogram_phase_range(self, synthetic_slc_pair: tuple[NDArray, NDArray]) -> None:
        """Test that phase is in [-pi, pi] range."""
        slc1, slc2 = synthetic_slc_pair
        phase, _ = compute_interferogram(slc1, slc2)
        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)


class TestComputeDisplacement:
    """Tests for displacement computation (research-grade)."""

    def test_displacement_shape(self) -> None:
        """Test that output shape matches input."""
        unwrapped = np.random.rand(32, 32).astype(np.float32) * 10
        displacement = compute_displacement(unwrapped, wavelength=0.056, incidence_angle=35.0)
        assert displacement.shape == unwrapped.shape

    def test_displacement_finite(self) -> None:
        """Test that output contains only finite values."""
        unwrapped = np.random.rand(32, 32).astype(np.float32) * 10
        displacement = compute_displacement(unwrapped, wavelength=0.056, incidence_angle=35.0)
        assert np.all(np.isfinite(displacement))


# Polarimetry Tests


class TestPauliDecomposition:
    """Tests for Pauli decomposition."""

    def test_pauli_returns_result(self, synthetic_quad_pol: dict[str, NDArray]) -> None:
        """Test that Pauli returns PolarimetricResult."""
        result = pauli_decomposition(
            synthetic_quad_pol["HH"],
            synthetic_quad_pol["HV"],
            synthetic_quad_pol["VV"],
        )
        assert isinstance(result, PolarimetricResult)

    def test_pauli_components(self, synthetic_quad_pol: dict[str, NDArray]) -> None:
        """Test that Pauli returns expected components."""
        result = pauli_decomposition(
            synthetic_quad_pol["HH"],
            synthetic_quad_pol["HV"],
            synthetic_quad_pol["VV"],
        )
        assert "surface" in result.components
        assert "volume" in result.components
        assert "dihedral" in result.components


class TestFreemanDurdenDecomposition:
    """Tests for Freeman-Durden decomposition."""

    def test_freeman_durden_returns_result(self, synthetic_quad_pol: dict[str, NDArray]) -> None:
        """Test that Freeman-Durden returns PolarimetricResult."""
        result = freeman_durden_decomposition(
            synthetic_quad_pol["HH"],
            synthetic_quad_pol["HV"],
            synthetic_quad_pol["VV"],
        )
        assert isinstance(result, PolarimetricResult)


class TestHAlphaDecomposition:
    """Tests for H-Alpha decomposition."""

    def test_h_alpha_returns_result(self, synthetic_quad_pol: dict[str, NDArray]) -> None:
        """Test that H-Alpha returns PolarimetricResult."""
        result = h_alpha_decomposition(
            synthetic_quad_pol["HH"],
            synthetic_quad_pol["HV"],
            synthetic_quad_pol["VH"],
            synthetic_quad_pol["VV"],
        )
        assert isinstance(result, PolarimetricResult)

    def test_h_alpha_components(self, synthetic_quad_pol: dict[str, NDArray]) -> None:
        """Test that H-Alpha returns expected components."""
        result = h_alpha_decomposition(
            synthetic_quad_pol["HH"],
            synthetic_quad_pol["HV"],
            synthetic_quad_pol["VH"],
            synthetic_quad_pol["VV"],
        )
        assert "entropy" in result.components
        assert "anisotropy" in result.components
        assert "alpha" in result.components


# Integration Tests


class TestSARIntegration:
    """Integration tests for SAR processing chain."""

    def test_amplitude_processing_chain(self, synthetic_amplitude: NDArray[np.floating]) -> None:
        """Test full amplitude processing chain."""
        # Calibrate
        calibrated = calibrate_amplitude(synthetic_amplitude, calibration_factor=1.0)

        # Compute sigma0
        sigma0 = compute_sigma0(calibrated, incidence_angle=30.0)

        # Filter speckle
        filtered = speckle_filter(sigma0, window_size=3)

        # Convert to dB
        db = amplitude_to_db(filtered)

        # All outputs should be finite
        assert np.all(np.isfinite(db))

    def test_interferometric_chain(self, synthetic_slc_pair: tuple[NDArray, NDArray]) -> None:
        """Test full interferometric processing chain."""
        slc1, slc2 = synthetic_slc_pair

        # Compute coherence
        coherence = compute_coherence(slc1, slc2, window_size=3)

        # Compute interferogram
        phase, ifg = compute_interferogram(slc1, slc2)

        # Compute displacement (using phase directly as proxy for unwrapped)
        displacement = compute_displacement(phase, wavelength=0.056, incidence_angle=35.0)

        # All outputs should be finite
        assert np.all(np.isfinite(coherence))
        assert np.all(np.isfinite(phase))
        assert np.all(np.isfinite(displacement))
