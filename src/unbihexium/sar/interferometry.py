"""SAR interferometry (InSAR) processing functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class InterferometricResult:
    """Result of interferometric processing."""

    coherence: NDArray[np.floating[Any]]
    phase: NDArray[np.floating[Any]]
    unwrapped_phase: NDArray[np.floating[Any]] | None = None


def compute_coherence(
    master: NDArray[np.complexfloating[Any, Any]],
    slave: NDArray[np.complexfloating[Any, Any]],
    window_size: int = 5,
) -> NDArray[np.floating[Any]]:
    """Compute interferometric coherence.

    Coherence formula:
    $$\\gamma = \\frac{|E[s_1 \\cdot s_2^*]|}{\\sqrt{E[|s_1|^2] \\cdot E[|s_2|^2]}}$$

    Args:
        master: Master SLC image.
        slave: Slave SLC image.
        window_size: Estimation window size.

    Returns:
        Coherence magnitude (0-1).
    """
    from scipy import ndimage

    if window_size % 2 == 0:
        window_size += 1

    # Cross-correlation
    cross = master * np.conj(slave)
    cross_mean = ndimage.uniform_filter(np.real(cross), size=window_size) + \
                 1j * ndimage.uniform_filter(np.imag(cross), size=window_size)

    # Power terms
    power_master = ndimage.uniform_filter(np.abs(master) ** 2, size=window_size)
    power_slave = ndimage.uniform_filter(np.abs(slave) ** 2, size=window_size)

    # Coherence
    denominator = np.sqrt(power_master * power_slave)
    denominator = np.maximum(denominator, 1e-10)

    coherence = np.abs(cross_mean) / denominator
    coherence = np.clip(coherence, 0, 1)

    return coherence.astype(np.float32)


def compute_interferogram(
    master: NDArray[np.complexfloating[Any, Any]],
    slave: NDArray[np.complexfloating[Any, Any]],
    multilook: tuple[int, int] = (1, 1),
) -> tuple[NDArray[np.floating[Any]], NDArray[np.complexfloating[Any, Any]]]:
    """Compute interferogram from SLC pair.

    Interferogram:
    $$I = s_1 \\cdot s_2^* = |s_1||s_2|e^{i(\\phi_1 - \\phi_2)}$$

    Args:
        master: Master SLC image.
        slave: Slave SLC image.
        multilook: Multilook factors (azimuth, range).

    Returns:
        Tuple of (phase, complex interferogram).
    """
    # Form interferogram
    ifg = master * np.conj(slave)

    # Multilooking
    if multilook != (1, 1):
        from scipy import ndimage
        ifg = ndimage.uniform_filter(
            np.real(ifg), size=multilook
        ) + 1j * ndimage.uniform_filter(
            np.imag(ifg), size=multilook
        )

    # Extract phase
    phase = np.angle(ifg)

    return phase.astype(np.float32), ifg


def phase_unwrapping(
    phase: NDArray[np.floating[Any]],
    method: str = "goldstein",
    coherence: NDArray[np.floating[Any]] | None = None,
    coherence_threshold: float = 0.3,
) -> NDArray[np.floating[Any]]:
    """Perform phase unwrapping.

    Unwrapping recovers the absolute phase from wrapped (-pi, pi) phase:
    $$\\phi_{unwrapped} = \\phi_{wrapped} + 2\\pi n$$

    Args:
        phase: Wrapped phase (-pi to pi).
        method: Unwrapping method (goldstein, snaphu_mcf, simple).
        coherence: Coherence for quality masking.
        coherence_threshold: Threshold for low-coherence masking.

    Returns:
        Unwrapped phase.
    """
    if method == "simple":
        # Simple 1D unwrapping per row
        unwrapped = np.zeros_like(phase)
        for i in range(phase.shape[0]):
            unwrapped[i] = np.unwrap(phase[i])
        return unwrapped.astype(np.float32)

    elif method == "goldstein":
        # Simplified Goldstein branch-cut method
        # In practice, use snaphu or other professional unwrapper

        # Detect residues (phase inconsistencies)
        dy = np.diff(phase, axis=0)
        dx = np.diff(phase, axis=1)

        # Wrap differences
        dy = np.angle(np.exp(1j * dy))
        dx = np.angle(np.exp(1j * dx))

        # Simple flood-fill unwrapping from center
        unwrapped = phase.copy()
        h, w = phase.shape
        stack = [(h // 2, w // 2)]
        visited = np.zeros_like(phase, dtype=bool)

        while stack:
            y, x = stack.pop()
            if y < 0 or y >= h or x < 0 or x >= w:
                continue
            if visited[y, x]:
                continue

            visited[y, x] = True

            # Check coherence
            if coherence is not None and coherence[y, x] < coherence_threshold:
                continue

            # Propagate to neighbors
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    diff = phase[ny, nx] - phase[y, x]
                    diff = np.angle(np.exp(1j * diff))
                    unwrapped[ny, nx] = unwrapped[y, x] + diff
                    stack.append((ny, nx))

        return unwrapped.astype(np.float32)

    else:
        raise ValueError(f"Unknown unwrapping method: {method}")


def compute_displacement(
    unwrapped_phase: NDArray[np.floating[Any]],
    wavelength: float,
    incidence_angle: float,
) -> NDArray[np.floating[Any]]:
    """Convert unwrapped phase to line-of-sight displacement.

    Displacement formula:
    $$d = \\frac{\\lambda \\cdot \\phi}{4\\pi}$$

    Args:
        unwrapped_phase: Unwrapped phase in radians.
        wavelength: Radar wavelength in meters.
        incidence_angle: Incidence angle in radians.

    Returns:
        Displacement in meters (negative = subsidence).
    """
    # LOS displacement
    displacement = (wavelength * unwrapped_phase) / (4 * np.pi)

    return displacement.astype(np.float32)
