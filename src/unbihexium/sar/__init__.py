"""SAR (Synthetic Aperture Radar) processing module."""

from unbihexium.sar.amplitude import (
    calibrate_amplitude,
    compute_sigma0,
    compute_gamma0,
    speckle_filter,
)
from unbihexium.sar.polarimetry import (
    compute_polarimetric_decomposition,
    pauli_decomposition,
    freeman_durden_decomposition,
    h_alpha_decomposition,
)
from unbihexium.sar.interferometry import (
    compute_coherence,
    compute_interferogram,
    phase_unwrapping,
)

__all__ = [
    # Amplitude
    "calibrate_amplitude",
    "compute_sigma0",
    "compute_gamma0",
    "speckle_filter",
    # Polarimetry
    "compute_polarimetric_decomposition",
    "pauli_decomposition",
    "freeman_durden_decomposition",
    "h_alpha_decomposition",
    # Interferometry
    "compute_coherence",
    "compute_interferogram",
    "phase_unwrapping",
]
