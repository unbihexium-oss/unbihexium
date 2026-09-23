# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_indices.py
# Title       : Tests of the spectral and radar index functions
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest and NumPy
# =============================================================================
#
# Abstract
# --------
# Evaluates every index of unbihexium.indices on reflectances chosen so that
# the result can be computed by hand, checks that undefined ratios give NaN
# instead of arbitrary numbers, the burn severity classes of Key and Benson
# (2006), and evaluation by name.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Arrays.
import numpy as np

# Test framework.
import pytest

# Index functions under test.
from unbihexium import indices

# Vegetation-like reflectances: blue, green, red, red edge, NIR, SWIR1, SWIR2.
BANDS = {
    "blue": 0.04,  # Blue.
    "green": 0.08,  # Green.
    "red": 0.05,  # Red.
    "red_edge": 0.2,  # Red edge.
    "nir": 0.45,  # Near infrared.
    "swir1": 0.25,  # Short-wave infrared 1.
    "swir2": 0.15,  # Short-wave infrared 2.
}  # End of the bands.


# Normalised differences with hand-computed values.
def test_normalized_differences() -> None:
    # Shorthand for the bands.
    b = BANDS
    # NDVI (0.45 - 0.05) / 0.5 = 0.8.
    assert indices.ndvi(b["nir"], b["red"]) == pytest.approx(0.8)
    # GNDVI (0.45 - 0.08) / 0.53.
    assert indices.gndvi(b["nir"], b["green"]) == pytest.approx(0.37 / 0.53)
    # NDRE (0.45 - 0.2) / 0.65.
    assert indices.ndre(b["nir"], b["red_edge"]) == pytest.approx(0.25 / 0.65)
    # NDWI of McFeeters (0.08 - 0.45) / 0.53.
    assert indices.ndwi(b["green"], b["nir"]) == pytest.approx(-0.37 / 0.53)
    # MNDWI and NDSI share the formula (green - SWIR1) / (green + SWIR1).
    assert indices.mndwi(b["green"], b["swir1"]) == pytest.approx(-0.17 / 0.33)
    # NDSI.
    assert indices.ndsi(b["green"], b["swir1"]) == pytest.approx(-0.17 / 0.33)
    # NDMI (0.45 - 0.25) / 0.7 and NDBI is its negative.
    assert indices.ndmi(b["nir"], b["swir1"]) == pytest.approx(0.2 / 0.7)
    # NDBI.
    assert indices.ndbi(b["swir1"], b["nir"]) == pytest.approx(-0.2 / 0.7)
    # NBR (0.45 - 0.15) / 0.6 = 0.5 and NBR2 (0.25 - 0.15) / 0.4 = 0.25.
    assert indices.nbr(b["nir"], b["swir2"]) == pytest.approx(0.5)
    # NBR2.
    assert indices.nbr2(b["swir1"], b["swir2"]) == pytest.approx(0.25)
    # BSI ((0.25 + 0.05) - (0.45 + 0.04)) / (0.3 + 0.49).
    assert indices.bsi(b["blue"], b["red"], b["nir"], b["swir1"]) == pytest.approx(-0.19 / 0.79)


# Soil- and atmosphere-adjusted vegetation indices.
def test_adjusted_vegetation_indices() -> None:
    # Shorthand for the bands.
    n, r, bl, g = BANDS["nir"], BANDS["red"], BANDS["blue"], BANDS["green"]
    # EVI 2.5 * 0.4 / (0.45 + 0.3 - 0.3 + 1).
    assert indices.evi(n, r, bl) == pytest.approx(2.5 * 0.4 / 1.45)
    # EVI2 2.5 * 0.4 / (0.45 + 0.12 + 1).
    assert indices.evi2(n, r) == pytest.approx(1.0 / 1.57)
    # SAVI 1.5 * 0.4 / (0.5 + 0.5).
    assert indices.savi(n, r) == pytest.approx(0.6)
    # OSAVI 0.4 / 0.66.
    assert indices.osavi(n, r) == pytest.approx(0.4 / 0.66)
    # MSAVI2 (1.9 - sqrt(1.9^2 - 3.2)) / 2.
    assert indices.msavi(n, r) == pytest.approx((1.9 - np.sqrt(1.9**2 - 3.2)) / 2)
    # ARVI with RB = 0.05 - (0.04 - 0.05) = 0.06: (0.45 - 0.06) / 0.51.
    assert indices.arvi(n, r, bl) == pytest.approx(0.39 / 0.51)
    # VARI (0.08 - 0.05) / (0.08 + 0.05 - 0.04).
    assert indices.vari(g, r, bl) == pytest.approx(0.03 / 0.09)
    # kNDVI = tanh(NDVI^2) = tanh(0.64).
    assert indices.kndvi(n, r) == pytest.approx(np.tanh(0.64))
    # Chlorophyll indices NIR / band - 1.
    assert indices.ci_green(n, g) == pytest.approx(0.45 / 0.08 - 1)
    # Red-edge chlorophyll index.
    assert indices.ci_rededge(n, BANDS["red_edge"]) == pytest.approx(1.25)
    # Moisture stress SWIR1 / NIR.
    assert indices.msi(BANDS["swir1"], n) == pytest.approx(0.25 / 0.45)


# Water extraction indices of Feyisa et al. (2014).
def test_awei() -> None:
    # Shorthand for the bands.
    b = BANDS
    # 4 (0.08 - 0.25) - (0.25 * 0.45 + 2.75 * 0.15).
    expected = 4 * (0.08 - 0.25) - (0.25 * 0.45 + 2.75 * 0.15)
    # No-shadow variant.
    assert indices.awei_nsh(b["green"], b["nir"], b["swir1"], b["swir2"]) == pytest.approx(expected)
    # 0.04 + 2.5 * 0.08 - 1.5 * (0.45 + 0.25) - 0.25 * 0.15.
    shadow = 0.04 + 0.2 - 1.05 - 0.0375
    # Shadow variant.
    value = indices.awei_sh(b["blue"], b["green"], b["nir"], b["swir1"], b["swir2"])
    # Compare.
    assert value == pytest.approx(shadow)


# Undefined ratios are NaN, and arrays broadcast.
def test_nan_handling_and_arrays() -> None:
    # Zero reflectance in both bands.
    assert np.isnan(indices.ndvi(np.array([0.0]), np.array([0.0]))[0])
    # NaN input gives NaN output.
    assert np.isnan(indices.ndvi(np.array([np.nan]), np.array([0.1]))[0])
    # Arrays broadcast against scalars.
    out = indices.ndvi(np.array([[0.3, 0.5], [0.1, 0.2]]), 0.1)
    # Shape and values.
    assert out.shape == (2, 2) and out[0, 0] == pytest.approx(0.5)
    # Division helper.
    assert indices.safe_divide(np.array([1.0, 1.0]), np.array([2.0, 0.0])).tolist()[0] == 0.5


# Burn severity from pre- and post-fire NBR.
def test_burn_indices() -> None:
    # Pre-fire NBR 0.64 and post-fire -0.1: dNBR 0.74.
    d = indices.dnbr(0.64, -0.1)
    # Difference.
    assert d == pytest.approx(0.74)
    # RdNBR = 0.74 / sqrt(0.64).
    assert indices.rdnbr(0.64, -0.1) == pytest.approx(0.74 / 0.8)
    # Classes of Key and Benson (2006) for typical dNBR values.
    classes = indices.burn_severity(np.array([-0.3, -0.2, 0.0, 0.2, 0.3, 0.5, 0.74, np.nan]))
    # Regrowth high, regrowth low, unburned, low, moderate-low, moderate-high, high, undefined.
    assert classes.tolist() == [0, 1, 2, 3, 4, 5, 6, -1]
    # Seven class names.
    assert len(indices.BURN_SEVERITY_CLASSES) == 7


# Radar indices of linear backscatter.
def test_radar_indices() -> None:
    # RVI 8 * 0.02 / (0.1 + 0.08 + 0.04).
    assert indices.rvi(0.1, 0.02, 0.08) == pytest.approx(0.16 / 0.22)
    # Pure random volume (HH = VV = 3 HV) gives RVI 1.
    assert indices.rvi(3.0, 1.0, 3.0) == pytest.approx(1.0)
    # Cross-polarisation ratio.
    assert indices.cross_pol_ratio(0.01, 0.1) == pytest.approx(0.1)


# Evaluation by name.
def test_compute_index() -> None:
    # NDVI by name.
    assert indices.compute_index("NDVI", nir=0.45, red=0.05) == pytest.approx(0.8)
    # Every registered name maps to a function.
    assert all(callable(f) for f in indices.INDEX_FUNCTIONS.values())
    # Unknown names are rejected.
    with pytest.raises(ValueError):
        # Not an index.
        indices.compute_index("xyz", nir=0.4)
    # Missing bands are reported as value errors.
    with pytest.raises(ValueError):
        # NDVI without the red band.
        indices.compute_index("ndvi", nir=0.4)


# =============================================================================
# End of module tests/unit/test_indices.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
