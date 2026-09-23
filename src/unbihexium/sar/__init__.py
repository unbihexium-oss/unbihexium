# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/sar/__init__.py
# Title       : Synthetic aperture radar processing
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Array-level processing of synthetic aperture radar (SAR) images:
#
#   amplitude       radiometric calibration (beta0, sigma0, gamma0), decibel
#                   conversion, multilooking and speckle filters
#   interferometry  interferograms, coherence, Goldstein filtering, phase
#                   unwrapping and phase-to-displacement conversion
#   polarimetry     coherency and covariance matrices, Pauli,
#                   Freeman-Durden, Yamaguchi and H / A / alpha
#                   decompositions
#
# Every function takes and returns NumPy arrays; see the modules for the
# formulas, conventions and references.
# =============================================================================

# Amplitude processing.
from unbihexium.sar.amplitude import (
    amplitude_to_db,  # 20 log10 of amplitudes.
    calibrate_amplitude,  # Constant scaling of amplitudes.
    compute_beta0,  # Radar brightness.
    compute_gamma0,  # gamma0 from sigma0.
    compute_sigma0,  # sigma0 from amplitude.
    db_to_power,  # Decibels to intensity.
    enhanced_lee_filter,  # Enhanced Lee filter.
    equivalent_number_of_looks,  # ENL of a homogeneous area.
    frost_filter,  # Frost filter.
    gamma_map_filter,  # Gamma MAP filter.
    kuan_filter,  # Kuan filter.
    lee_filter,  # Lee filter.
    multilook,  # Block averaging.
    power_to_db,  # Intensity to decibels.
    radiometric_calibration,  # Look-up table calibration.
    refined_lee_filter,  # Refined Lee filter.
    speckle_filter,  # Filter by name.
)  # End of the amplitude imports.

# Interferometry.
from unbihexium.sar.interferometry import (
    InterferometricResult,  # Result record.
    compute_coherence,  # Sample coherence.
    compute_displacement,  # Phase to range change.
    compute_interferogram,  # Interferogram formation.
    goldstein_filter,  # Adaptive interferogram filter.
    height_of_ambiguity,  # Topographic sensitivity.
    los_to_vertical,  # Range change to vertical motion.
    phase_residues,  # Residue charges.
    phase_unwrapping,  # Phase unwrapping.
    wrap_phase,  # Wrap to [-pi, pi).
)  # End of the interferometry imports.

# Polarimetry.
from unbihexium.sar.polarimetry import (
    PolarimetricResult,  # Result record.
    coherency_matrix,  # Pauli coherency T3.
    compute_polarimetric_decomposition,  # Decomposition by name.
    covariance_matrix,  # Lexicographic covariance C3.
    freeman_durden_decomposition,  # Three components.
    h_a_alpha,  # Eigen-analysis of T3.
    h_alpha_decomposition,  # Cloude-Pottier decomposition.
    h_alpha_zones,  # H / alpha plane zones.
    pauli_decomposition,  # Pauli powers.
    pauli_rgb,  # Pauli colour composite.
    yamaguchi_decomposition,  # Four components.
)  # End of the polarimetry imports.

# Public names of the package.
__all__ = [
    "InterferometricResult",  # Interferometry result record.
    "PolarimetricResult",  # Polarimetry result record.
    "amplitude_to_db",  # Amplitude to decibels.
    "calibrate_amplitude",  # Constant scaling of amplitudes.
    "coherency_matrix",  # Pauli coherency T3.
    "compute_beta0",  # Radar brightness.
    "compute_coherence",  # Sample coherence.
    "compute_displacement",  # Phase to range change.
    "compute_gamma0",  # gamma0 from sigma0.
    "compute_interferogram",  # Interferogram formation.
    "compute_polarimetric_decomposition",  # Decomposition by name.
    "compute_sigma0",  # sigma0 from amplitude.
    "covariance_matrix",  # Lexicographic covariance C3.
    "db_to_power",  # Decibels to intensity.
    "enhanced_lee_filter",  # Enhanced Lee filter.
    "equivalent_number_of_looks",  # ENL of a homogeneous area.
    "freeman_durden_decomposition",  # Three components.
    "frost_filter",  # Frost filter.
    "gamma_map_filter",  # Gamma MAP filter.
    "goldstein_filter",  # Adaptive interferogram filter.
    "h_a_alpha",  # Eigen-analysis of T3.
    "h_alpha_decomposition",  # Cloude-Pottier decomposition.
    "h_alpha_zones",  # H / alpha plane zones.
    "height_of_ambiguity",  # Topographic sensitivity.
    "kuan_filter",  # Kuan filter.
    "lee_filter",  # Lee filter.
    "los_to_vertical",  # Range change to vertical motion.
    "multilook",  # Block averaging.
    "pauli_decomposition",  # Pauli powers.
    "pauli_rgb",  # Pauli colour composite.
    "phase_residues",  # Residue charges.
    "phase_unwrapping",  # Phase unwrapping.
    "power_to_db",  # Intensity to decibels.
    "radiometric_calibration",  # Look-up table calibration.
    "refined_lee_filter",  # Refined Lee filter.
    "speckle_filter",  # Filter by name.
    "wrap_phase",  # Wrap to [-pi, pi).
    "yamaguchi_decomposition",  # Four components.
]  # End of the public names.

# =============================================================================
# End of module src/unbihexium/sar/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
