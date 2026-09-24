# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_sar.py
# Title       : Tests of SAR calibration, interferometry and polarimetry
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest, NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Checks unbihexium.sar against values derived by hand: beta0, sigma0 and
# gamma0 of a known amplitude and angle, decibel conversion, block
# multilooking, the Lee and Kuan weights of a 3 x 3 window, the
# equivalent number of looks of simulated speckle, interferometric phase and
# coherence of synthetic images, residues of a phase vortex, exact phase
# unwrapping of residue-free phase with all methods, phase to displacement,
# and the Pauli, Freeman-Durden, Yamaguchi and H / A / alpha decompositions
# of canonical scatterers (plane, dihedral, helix) and of a diagonal
# coherency matrix.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Arrays.
import numpy as np

# Test framework.
import pytest

# Amplitude processing under test.
from unbihexium.sar.amplitude import (
    SPECKLE_FILTERS,  # Filter names.
    amplitude_to_db,  # Amplitude to dB.
    calibrate_amplitude,  # Constant scaling.
    compute_beta0,  # Radar brightness.
    compute_gamma0,  # gamma0.
    compute_sigma0,  # sigma0.
    db_to_power,  # dB to intensity.
    equivalent_number_of_looks,  # ENL.
    kuan_filter,  # Kuan filter.
    lee_filter,  # Lee filter.
    multilook,  # Block averaging.
    power_to_db,  # Intensity to dB.
    radiometric_calibration,  # LUT calibration.
    speckle_filter,  # Filters by name.
)  # End of the amplitude imports.

# Interferometry under test.
from unbihexium.sar.interferometry import (
    InterferometricResult,  # Result record.
    compute_coherence,  # Coherence.
    compute_displacement,  # Phase to range change.
    compute_interferogram,  # Interferogram.
    goldstein_filter,  # Adaptive filter.
    height_of_ambiguity,  # Topographic sensitivity.
    los_to_vertical,  # Vertical projection.
    phase_residues,  # Residue charges.
    phase_unwrapping,  # Unwrapping.
    wrap_phase,  # Wrapping.
)  # End of the interferometry imports.

# Polarimetry under test.
from unbihexium.sar.polarimetry import (
    PolarimetricResult,  # Result record.
    coherency_matrix,  # T3.
    compute_polarimetric_decomposition,  # Dispatcher.
    freeman_durden_decomposition,  # Three components.
    h_a_alpha,  # Eigen-analysis.
    h_alpha_decomposition,  # Cloude-Pottier.
    h_alpha_zones,  # H / alpha zones.
    pauli_decomposition,  # Pauli powers.
    pauli_rgb,  # Colour composite.
    yamaguchi_decomposition,  # Four components.
)  # End of the polarimetry imports.


# Rayleigh-distributed amplitude of one-look speckle.
@pytest.fixture
def speckle_amplitude() -> np.ndarray:
    # Seeded generator.
    rng = np.random.default_rng(42)
    # Modulus of a circular complex Gaussian.
    return np.abs(rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64)))


# Constant-valued complex channels of a given scattering matrix.
def channels(hh: complex, hv: complex, vv: complex, shape=(9, 9)) -> dict[str, np.ndarray]:
    # Fill each channel with its value.
    values = {"hh": hh, "hv": hv, "vv": vv}
    # Constant arrays.
    return {name: np.full(shape, v, dtype=complex) for name, v in values.items()}


# Decibel conversion of known values.
def test_decibels() -> None:
    # Amplitude 1 is 0 dB and 10 is 20 dB; zero takes the floor.
    assert np.allclose(amplitude_to_db(np.array([1.0, 10.0, 0.0])), [0.0, 20.0, -40.0])
    # Intensity 100 is 20 dB.
    assert power_to_db(np.array([100.0]))[0] == pytest.approx(20.0)
    # Zero intensity has no logarithm.
    assert np.isnan(power_to_db(np.array([0.0]))[0])
    # dB conversion round-trips.
    assert np.allclose(db_to_power(power_to_db(np.array([0.5, 3.0]))), [0.5, 3.0])


# Calibration constants and look-up tables.
def test_calibration() -> None:
    # The modulus of 3 + 4j is 5, scaled by 2.
    assert calibrate_amplitude(np.array([3 + 4j]), 2.0)[0] == pytest.approx(10.0)
    # (|DN|^2 - noise) / A^2 = (100 - 20) / 4.
    assert radiometric_calibration(np.array([10.0]), 2.0, noise=20.0)[0] == pytest.approx(20.0)
    # Noise above the signal is clipped to zero.
    assert radiometric_calibration(np.array([1.0]), 1.0, noise=5.0)[0] == 0.0
    # Non-positive look-up tables are rejected.
    with pytest.raises(ValueError):
        # Zero table value.
        radiometric_calibration(np.array([1.0]), 0.0)


# beta0, sigma0 and gamma0 of a known amplitude and incidence angle.
def test_sigma0_gamma0_values() -> None:
    # Amplitude 2 gives beta0 = 4.
    beta0 = compute_beta0(np.array([2.0]))
    # beta0 = A^2.
    assert beta0[0] == pytest.approx(4.0)
    # sigma0 = beta0 sin(30 deg) = 2.
    sigma0 = compute_sigma0(np.array([2.0]), 30.0)
    # Check the value.
    assert sigma0[0] == pytest.approx(2.0)
    # gamma0 = sigma0 / cos(30 deg).
    assert compute_gamma0(sigma0, 30.0)[0] == pytest.approx(2.0 / np.cos(np.pi / 6))
    # Radians give the same result.
    assert compute_sigma0(np.array([2.0]), np.pi / 6, degrees=False)[0] == pytest.approx(2.0)
    # A calibration constant divides beta0.
    assert compute_sigma0(np.array([2.0]), 30.0, calibration_lut=4.0)[0] == pytest.approx(0.5)


# sigma0 and gamma0 of speckle are non-negative for angles given in degrees.
def test_sigma0_gamma0_positive(speckle_amplitude: np.ndarray) -> None:
    # sigma0 at 30 degrees.
    sigma0 = compute_sigma0(speckle_amplitude, incidence_angle=30.0)
    # Shape is kept.
    assert sigma0.shape == speckle_amplitude.shape
    # Backscatter is never negative.
    assert np.all(sigma0 >= 0)
    # gamma0 is never negative either.
    assert np.all(compute_gamma0(sigma0, incidence_angle=30.0) >= 0)
    # Angles outside (0, 90) degrees are rejected.
    with pytest.raises(ValueError):
        # 95 degrees.
        compute_sigma0(speckle_amplitude, 95.0)


# Block averaging of a known image.
def test_multilook() -> None:
    # 4 x 4 image with values 0 .. 15.
    image = np.arange(16.0).reshape(4, 4)
    # 2 x 2 looks.
    looked = multilook(image, (2, 2))
    # Means of the four blocks.
    assert np.allclose(looked, [[2.5, 4.5], [10.5, 12.5]])
    # NaN pixels are ignored.
    image[0, 0] = np.nan
    # First block mean without the NaN: (1 + 4 + 5) / 3.
    assert multilook(image, (2, 2))[0, 0] == pytest.approx(10.0 / 3.0)
    # Complex input is detected: |1 + 1j|^2 = 2.
    assert multilook(np.full((2, 2), 1 + 1j), (2, 1))[0, 0] == pytest.approx(2.0)


# One-look intensity speckle has ENL 1; 2 x 2 multilooking gives about 4.
def test_equivalent_number_of_looks() -> None:
    # Exponentially distributed intensity.
    intensity = np.random.default_rng(1).exponential(1.0, size=(200, 200))
    # ENL of one look.
    assert equivalent_number_of_looks(intensity) == pytest.approx(1.0, abs=0.1)
    # ENL of four looks.
    assert equivalent_number_of_looks(multilook(intensity, (2, 2))) == pytest.approx(4.0, abs=0.4)
    # A constant area has infinite ENL.
    assert equivalent_number_of_looks(np.ones(10)) == float("inf")


# Lee and Kuan weights of a 3 x 3 window computed by hand.
def test_lee_kuan_centre_value() -> None:
    # A bright pixel in a dark window.
    image = np.ones((3, 3))
    # Centre value 4.
    image[1, 1] = 4.0
    # Window mean 4/3, variance 8/9, Ci^2 = 0.5; four looks give Cu^2 = 0.25.
    mean = 4.0 / 3.0
    # Lee weight 1 - Cu^2 / Ci^2 = 0.5.
    assert lee_filter(image, 3, looks=4)[1, 1] == pytest.approx(mean + 0.5 * (4.0 - mean))
    # Kuan weight (1 - Cu^2 / Ci^2) / (1 + Cu^2) = 0.4.
    assert kuan_filter(image, 3, looks=4)[1, 1] == pytest.approx(mean + 0.4 * (4.0 - mean))


# Every filter keeps constant images, and the adaptive ones smooth speckle.
@pytest.mark.parametrize("name", SPECKLE_FILTERS)
def test_speckle_filters(name: str) -> None:
    # Constant image.
    flat = np.full((16, 16), 3.0)
    # A constant image is a fixed point.
    assert np.allclose(speckle_filter(flat, name, 5), 3.0)
    # One-look speckle of a constant scene with reflectivity 5.
    noisy = 5.0 * np.random.default_rng(7).exponential(1.0, size=(64, 64))
    # Filtered image.
    out = speckle_filter(noisy, name, 5)
    # Shape and finiteness.
    assert out.shape == noisy.shape and np.all(np.isfinite(out))
    # Speckle variance is reduced.
    assert out.std() < noisy.std()
    # The mean is preserved except by the median (median of the exponential is ln 2).
    if name != "median":
        # Within 10 %.
        assert out.mean() == pytest.approx(noisy.mean(), rel=0.1)


# The enhanced Lee filter warns about nothing, also at point targets.
def test_enhanced_lee_without_warnings() -> None:
    # Warning control.
    import warnings

    # One-look speckle with a strong point target and a NaN pixel.
    image = 5.0 * np.random.default_rng(3).exponential(1.0, size=(32, 32))
    # Point target, where the local variation exceeds Cmax.
    image[16, 16] = 1e4
    # Missing pixel.
    image[2, 2] = np.nan
    # Every warning becomes an error.
    with warnings.catch_warnings():
        # Turn warnings into errors.
        warnings.simplefilter("error")
        # Filter the image.
        out = speckle_filter(image, "enhanced_lee", 5)
    # The point target is kept and the missing pixel stays missing.
    assert out[16, 16] == image[16, 16] and np.isnan(out[2, 2])


# Filters keep NaN pixels and validate their parameters.
def test_speckle_filter_nodata_and_errors() -> None:
    # Constant image with one invalid pixel.
    image = np.full((8, 8), 2.0)
    # Nodata pixel.
    image[3, 3] = np.nan
    # Every filter keeps the invalid pixel and fills nothing else with NaN.
    for name in SPECKLE_FILTERS:
        # Filter output.
        out = speckle_filter(image, name, 3)
        # The invalid pixel stays NaN and the rest is finite.
        assert np.isnan(out[3, 3]) and np.isfinite(out[~np.isnan(image)]).all(), name
    # Even windows are rejected.
    with pytest.raises(ValueError):
        # Window of 4 pixels.
        speckle_filter(image, "lee", 4)
    # Unknown names are rejected.
    with pytest.raises(ValueError):
        # Not a filter.
        speckle_filter(image, "wiener")


# Interferometric phase of two images with known phases.
def test_interferogram_phase() -> None:
    # Random phases of the two images.
    rng = np.random.default_rng(0)
    # Phases of the reference and secondary images.
    a, b = rng.uniform(-np.pi, np.pi, (2, 16, 16))
    # Phase of s1 * conj(s2) is the wrapped difference.
    phase, ifg = compute_interferogram(np.exp(1j * a), np.exp(1j * b))
    # Compare on the unit circle.
    assert np.allclose(np.exp(1j * phase), np.exp(1j * (a - b)))
    # Unit amplitudes multiply to unit amplitude.
    assert np.allclose(np.abs(ifg), 1.0)
    # Result record holds the products.
    record = InterferometricResult(coherence=np.ones((2, 2)), phase=phase[:2, :2])
    # No unwrapped phase yet.
    assert record.unwrapped_phase is None


# Coherence of identical, phase-shifted and independent images.
def test_coherence() -> None:
    # Random complex image.
    rng = np.random.default_rng(3)
    # Circular Gaussian samples.
    s = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    # Identical images are fully coherent.
    assert np.allclose(compute_coherence(s, s, 3), 1.0)
    # A constant phase shift does not change coherence.
    assert np.allclose(compute_coherence(s, s * np.exp(0.7j), 3), 1.0)
    # Independent images have low coherence (bias about sqrt(pi / (4 N)) = 0.13 for N = 49).
    other = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    # Mean coherence.
    assert compute_coherence(s, other, 7).mean() < 0.3


# Wrapping and residue detection.
def test_wrap_and_residues() -> None:
    # 3 pi wraps to -pi, -pi / 2 stays.
    assert np.allclose(wrap_phase(np.array([3 * np.pi, -np.pi / 2])), [-np.pi, -np.pi / 2])
    # Grid coordinates.
    y, x = np.mgrid[0:20, 0:20].astype(float)
    # A smooth ramp has no residues.
    assert not phase_residues(wrap_phase(0.3 * x + 0.2 * y)).any()
    # A phase vortex around a point between pixels has exactly one residue.
    vortex = np.angle((x - 9.5) + 1j * (y - 9.5))
    # Charges of the loops.
    charges = phase_residues(vortex)
    # One loop of charge +1 or -1.
    assert np.abs(charges).sum() == 1 and abs(charges.sum()) == 1


# Every unwrapping method recovers residue-free phase up to a constant.
@pytest.mark.parametrize("method", ["least_squares", "quality_guided", "simple"])
def test_phase_unwrapping_exact(method: str) -> None:
    # Grid coordinates.
    y, x = np.mgrid[0:40, 0:50].astype(float)
    # True phase spans many cycles with gradients below pi.
    truth = 0.4 * x + 0.25 * y + 0.002 * (x - 25) ** 2
    # Unwrap the wrapped phase.
    result = phase_unwrapping(wrap_phase(truth), method=method)
    # Difference to the truth.
    diff = result - truth
    # Constant up to round-off.
    assert np.ptp(diff) < 1e-8


# Coherence masking: masked pixels are NaN, the rest is recovered.
@pytest.mark.parametrize("method", ["least_squares", "quality_guided"])
def test_phase_unwrapping_masked(method: str) -> None:
    # Grid coordinates.
    y, x = np.mgrid[0:30, 0:30].astype(float)
    # True phase.
    truth = 0.5 * x - 0.3 * y
    # Coherence with a low-quality block and a medium-quality band.
    coherence = np.ones_like(truth)
    # Masked block.
    coherence[10:15, 10:15] = 0.1
    # Down-weighted but valid band.
    coherence[20:, :] = 0.6
    # Unwrap with the mask.
    result = phase_unwrapping(wrap_phase(truth), method, coherence=coherence)
    # Masked pixels are NaN.
    assert np.isnan(result[10:15, 10:15]).all()
    # Valid pixels differ from the truth by a constant.
    assert np.nanmax(result - truth) - np.nanmin(result - truth) < 1e-6
    # Unknown methods are rejected.
    with pytest.raises(ValueError):
        # Not a method.
        phase_unwrapping(truth, "branch_cut")


# The Goldstein filter is the identity for alpha 0 and removes residues of noisy fringes.
def test_goldstein_filter() -> None:
    # Grid coordinates.
    y, x = np.mgrid[0:64, 0:64].astype(float)
    # Noisy fringes.
    noise = np.random.default_rng(5).normal(0.0, 0.8, x.shape)
    # Complex interferogram.
    ifg = np.exp(1j * (0.3 * x + 0.1 * y + noise))
    # Alpha zero leaves the input unchanged.
    assert np.allclose(goldstein_filter(ifg, alpha=0.0), ifg)
    # Residues before filtering.
    before = np.abs(phase_residues(np.angle(ifg))).sum()
    # Residues after strong filtering.
    after = np.abs(phase_residues(np.angle(goldstein_filter(ifg, alpha=0.8)))).sum()
    # Filtering removes most residues.
    assert before > 20 and after < before / 4


# Phase to range change and vertical motion.
def test_displacement() -> None:
    # Sentinel-1 C-band wavelength in metres.
    wavelength = 0.05546576
    # One full cycle is half a wavelength of range change.
    los = compute_displacement(np.array([2 * np.pi]), wavelength)
    # lambda / 2.
    assert los[0] == pytest.approx(wavelength / 2)
    # At 60 degrees incidence, vertical motion is -los / cos(60).
    assert los_to_vertical(los, 60.0)[0] == pytest.approx(-wavelength)
    # The same through compute_displacement.
    vertical = compute_displacement(np.array([2 * np.pi]), wavelength, 60.0, vertical=True)
    # Same value.
    assert vertical[0] == pytest.approx(-wavelength)
    # Vertical projection needs the angle.
    with pytest.raises(ValueError):
        # No angle.
        compute_displacement(np.array([1.0]), wavelength, vertical=True)


# Height of ambiguity of a typical Sentinel-1 pair.
def test_height_of_ambiguity() -> None:
    # lambda R sin(theta) / (2 B) with R = 850 km, theta = 35 deg, B = 100 m.
    expected = 0.05546576 * 850e3 * np.sin(np.radians(35.0)) / 200.0
    # Compare.
    assert height_of_ambiguity(0.05546576, 850e3, 35.0, 100.0) == pytest.approx(expected)


# Pauli powers of canonical scatterers.
def test_pauli_decomposition() -> None:
    # A plane (odd bounce): HH = VV = 1.
    plane = pauli_decomposition(**channels(1, 0, 1))
    # Result type and component names.
    assert isinstance(plane, PolarimetricResult)
    # Names of the Pauli components.
    assert set(plane.components) == {"surface", "dihedral", "volume"}
    # All power is surface power: |1 + 1|^2 / 2 = 2.
    assert np.allclose(plane.components["surface"], 2.0)
    # No dihedral power.
    assert np.allclose(plane.components["dihedral"], 0.0)
    # A dihedral: HH = -VV.
    dihedral = pauli_decomposition(**channels(1, 0, -1))
    # All power is dihedral power.
    assert np.allclose(dihedral.components["dihedral"], 2.0)
    # Cross-pol power 2 |HV|^2, with VH averaged in.
    both = pauli_decomposition(**channels(0, 1, 0), vh=np.full((9, 9), 3.0 + 0j))
    # Mean cross-pol amplitude 2 gives 2 * 4 = 8.
    assert np.allclose(both.components["volume"], 8.0)
    # The composite is scaled to [0, 1].
    rgb = pauli_rgb(**channels(1, 0.2, -1))
    # Three channels in range.
    assert rgb.shape == (9, 9, 3) and rgb.min() >= 0 and rgb.max() <= 1


# Freeman-Durden powers of canonical scatterers and power conservation.
def test_freeman_durden() -> None:
    # Surface: all power (span 2) is surface power.
    surface = freeman_durden_decomposition(**channels(1, 0, 1)).components
    # Surface power.
    assert np.allclose(surface["surface"], 2.0) and np.allclose(surface["dihedral"], 0.0)
    # Dihedral: all power is dihedral power.
    dihedral = freeman_durden_decomposition(**channels(1, 0, -1)).components
    # Dihedral power.
    assert np.allclose(dihedral["dihedral"], 2.0) and np.allclose(dihedral["surface"], 0.0)
    # Random scene.
    rng = np.random.default_rng(2)
    # Complex Gaussian channels with weaker cross-pol.
    gauss = rng.normal(size=(3, 24, 24)) + 1j * rng.normal(size=(3, 24, 24))
    # Channels with amplitudes 1, 1 and 0.3.
    hh, vv, hv = gauss[0], gauss[1], 0.3 * gauss[2]
    # Decomposition.
    parts = freeman_durden_decomposition(hh, hv, vv).components
    # Sum of the powers.
    total = parts["surface"] + parts["dihedral"] + parts["volume"]
    # Powers are non-negative.
    assert min(parts[k].min() for k in ("surface", "dihedral", "volume")) >= 0
    # The powers sum to the span wherever nothing was clipped (most pixels).
    assert np.median(np.abs(total - parts["span"])) < 1e-9


# Yamaguchi powers: a pure helix and a pure plane.
def test_yamaguchi() -> None:
    # Left helix: S = 0.5 [[1, j], [j, -1]], span 1.
    helix = yamaguchi_decomposition(**channels(0.5, 0.5j, -0.5)).components
    # All power is helix power.
    assert np.allclose(helix["helix"], 1.0) and np.allclose(helix["span"], 1.0)
    # No volume, surface or dihedral power.
    assert np.allclose(helix["volume"] + helix["surface"] + helix["dihedral"], 0.0)
    # A plane has only surface power.
    plane = yamaguchi_decomposition(**channels(1, 0, 1)).components
    # Surface power equals the span.
    assert np.allclose(plane["surface"], 2.0) and np.allclose(plane["helix"], 0.0)


# Entropy, anisotropy and alpha of a diagonal coherency matrix.
def test_h_a_alpha_of_diagonal_matrix() -> None:
    # Eigenvalues 3, 2, 1 with the Pauli basis as eigenvectors.
    t3 = np.diag([3.0, 2.0, 1.0]).astype(complex)
    # Eigen-analysis.
    out = h_a_alpha(t3)
    # Probabilities 1/2, 1/3, 1/6.
    p = np.array([3.0, 2.0, 1.0]) / 6.0
    # Entropy with logarithm base 3.
    assert out["entropy"] == pytest.approx(-np.sum(p * np.log(p)) / np.log(3))
    # Anisotropy (2 - 1) / (2 + 1).
    assert out["anisotropy"] == pytest.approx(1.0 / 3.0)
    # Alpha angles 0, 90, 90 degrees weighted by p: 90 (1/3 + 1/6) = 45.
    assert out["alpha"] == pytest.approx(45.0)
    # Eigenvalues in descending order.
    assert (out["lambda1"], out["lambda2"], out["lambda3"]) == pytest.approx((3.0, 2.0, 1.0))


# H / alpha of canonical scatterers from four channels, and the zones.
def test_h_alpha_decomposition() -> None:
    # A plane given with HV and VH channels.
    plane = h_alpha_decomposition(**channels(1, 0, 1), vh=np.zeros((9, 9), complex))
    # Deterministic single scatterer: zero entropy and alpha.
    assert np.allclose(plane.entropy, 0.0) and np.allclose(plane.alpha, 0.0, atol=1e-6)
    # Components carry the same images.
    assert {"entropy", "anisotropy", "alpha"} <= set(plane.components)
    # A dihedral has alpha 90 degrees.
    dihedral = h_alpha_decomposition(**channels(1, 0, -1))
    # Alpha of a dihedral.
    assert np.allclose(dihedral.alpha, 90.0)
    # Averaged random scene has high entropy.
    rng = np.random.default_rng(9)
    # Independent channels of equal power.
    hh, hv, vv = rng.normal(size=(3, 20, 20)) + 1j * rng.normal(size=(3, 20, 20))
    # Entropy with a 7 x 7 window.
    assert h_alpha_decomposition(hh, hv, vv, window_size=7).entropy.mean() > 0.8
    # Zones of the H / alpha plane.
    zones = h_alpha_zones(np.array([0.2, 0.2, 0.7, 0.95, np.nan]), np.array([10, 60, 45, 50, 1]))
    # Low surface, low multiple, medium vegetation, high vegetation, undefined.
    assert zones.tolist() == [9, 7, 5, 2, 0]


# Coherency matrix trace equals the span; the dispatcher selects by name.
def test_coherency_and_dispatcher() -> None:
    # Arbitrary scatterer.
    c = channels(1 + 1j, 0.5, -0.2j)
    # Trace of T3.
    trace = np.trace(coherency_matrix(**c), axis1=-2, axis2=-1).real
    # Span |HH|^2 + 2 |HV|^2 + |VV|^2 = 2 + 0.5 + 0.04.
    assert np.allclose(trace, 2.54)
    # Every name returns its decomposition.
    for name in ("pauli", "freeman_durden", "yamaguchi", "h_alpha"):
        # Decompose.
        result = compute_polarimetric_decomposition(c["hh"], c["hv"], c["vv"], name)
        # Type of the result.
        assert result.decomposition_type == name
    # Unknown names are rejected.
    with pytest.raises(ValueError):
        # Not a decomposition.
        compute_polarimetric_decomposition(c["hh"], c["hv"], c["vv"], "krogager")


# Full amplitude chain: calibration, sigma0, filtering and decibels.
def test_amplitude_chain(speckle_amplitude: np.ndarray) -> None:
    # Calibrate.
    calibrated = calibrate_amplitude(speckle_amplitude, 1.0)
    # sigma0 at 35 degrees.
    sigma0 = compute_sigma0(calibrated, 35.0)
    # Speckle filter.
    filtered = speckle_filter(sigma0, "refined_lee")
    # Decibels with a floor.
    db = power_to_db(filtered, floor=-50.0)
    # All values are finite.
    assert np.all(np.isfinite(db))


# =============================================================================
# End of module tests/unit/test_sar.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
