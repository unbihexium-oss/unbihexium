"""SAR polarimetric decomposition functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class PolarimetricResult:
    """Result of polarimetric decomposition."""

    decomposition_type: str
    components: dict[str, NDArray[np.floating[Any]]]
    entropy: NDArray[np.floating[Any]] | None = None
    anisotropy: NDArray[np.floating[Any]] | None = None
    alpha: NDArray[np.floating[Any]] | None = None


def compute_polarimetric_decomposition(
    hh: NDArray[np.complexfloating[Any, Any]],
    hv: NDArray[np.complexfloating[Any, Any]],
    vv: NDArray[np.complexfloating[Any, Any]],
    decomposition: str = "pauli",
) -> PolarimetricResult:
    """Compute polarimetric decomposition.

    Args:
        hh: HH polarization channel.
        hv: HV polarization channel.
        vv: VV polarization channel.
        decomposition: Type of decomposition (pauli, freeman_durden, h_alpha).

    Returns:
        PolarimetricResult with decomposition components.
    """
    if decomposition == "pauli":
        return pauli_decomposition(hh, hv, vv)
    elif decomposition == "freeman_durden":
        return freeman_durden_decomposition(hh, hv, vv)
    elif decomposition == "h_alpha":
        return h_alpha_decomposition(hh, hv, vv)
    else:
        raise ValueError(f"Unknown decomposition: {decomposition}")


def pauli_decomposition(
    hh: NDArray[np.complexfloating[Any, Any]],
    hv: NDArray[np.complexfloating[Any, Any]],
    vv: NDArray[np.complexfloating[Any, Any]],
) -> PolarimetricResult:
    """Compute Pauli decomposition.

    Pauli basis:
    $$|a|^2 = |S_{HH} + S_{VV}|^2 / 2$$ (surface/single bounce)
    $$|b|^2 = |S_{HH} - S_{VV}|^2 / 2$$ (double bounce)
    $$|c|^2 = 2|S_{HV}|^2$$ (volume/diffuse)

    Args:
        hh: HH channel.
        hv: HV channel.
        vv: VV channel.

    Returns:
        PolarimetricResult with surface, double, volume components.
    """
    # Pauli decomposition
    surface = np.abs(hh + vv) ** 2 / 2  # Single bounce
    double = np.abs(hh - vv) ** 2 / 2   # Double bounce
    volume = 2 * np.abs(hv) ** 2         # Volume scattering

    return PolarimetricResult(
        decomposition_type="pauli",
        components={
            "surface": surface.astype(np.float32),
            "double_bounce": double.astype(np.float32),
            "volume": volume.astype(np.float32),
        },
    )


def freeman_durden_decomposition(
    hh: NDArray[np.complexfloating[Any, Any]],
    hv: NDArray[np.complexfloating[Any, Any]],
    vv: NDArray[np.complexfloating[Any, Any]],
) -> PolarimetricResult:
    """Compute Freeman-Durden three-component decomposition.

    Decomposes backscatter into surface, double-bounce, and volume components.

    Args:
        hh: HH channel.
        hv: HV channel.
        vv: VV channel.

    Returns:
        PolarimetricResult with Ps, Pd, Pv components.
    """
    # Covariance matrix elements
    c11 = np.abs(hh) ** 2
    c22 = np.abs(hv) ** 2
    c33 = np.abs(vv) ** 2
    c13 = hh * np.conj(vv)

    # Volume scattering (simplified model)
    fv = 8 * c22 / 3
    pv = fv

    # Remaining power after volume removal
    c11_r = c11 - fv / 2
    c33_r = c33 - fv / 2
    c13_r = c13 - fv / 6

    # Surface and double-bounce
    re_c13 = np.real(c13_r)

    # Determine dominant mechanism
    mask = re_c13 >= 0

    # Surface dominant
    ps = np.where(mask, c33_r + np.abs(c13_r) ** 2 / (c11_r + 1e-10), 0)
    pd = np.where(mask, c11_r - np.abs(c13_r) ** 2 / (c33_r + 1e-10), 0)

    # Double-bounce dominant
    ps = np.where(~mask, c11_r - np.abs(c13_r) ** 2 / (c33_r + 1e-10), ps)
    pd = np.where(~mask, c33_r + np.abs(c13_r) ** 2 / (c11_r + 1e-10), pd)

    return PolarimetricResult(
        decomposition_type="freeman_durden",
        components={
            "surface": np.maximum(ps, 0).astype(np.float32),
            "double_bounce": np.maximum(pd, 0).astype(np.float32),
            "volume": np.maximum(pv, 0).astype(np.float32),
        },
    )


def h_alpha_decomposition(
    hh: NDArray[np.complexfloating[Any, Any]],
    hv: NDArray[np.complexfloating[Any, Any]],
    vv: NDArray[np.complexfloating[Any, Any]],
) -> PolarimetricResult:
    """Compute H-Alpha decomposition (Cloude-Pottier).

    Entropy H measures randomness of scattering.
    Alpha angle classifies scattering mechanism.

    Args:
        hh: HH channel.
        hv: HV channel.
        vv: VV channel.

    Returns:
        PolarimetricResult with entropy, anisotropy, alpha.
    """
    # Build coherency matrix T3
    k1 = (hh + vv) / np.sqrt(2)
    k2 = (hh - vv) / np.sqrt(2)
    k3 = np.sqrt(2) * hv

    # Coherency matrix elements
    t11 = np.abs(k1) ** 2
    t22 = np.abs(k2) ** 2
    t33 = np.abs(k3) ** 2

    # Simplified eigenvalue estimation
    trace = t11 + t22 + t33
    trace = np.maximum(trace, 1e-10)

    p1 = t11 / trace
    p2 = t22 / trace
    p3 = t33 / trace

    # Entropy
    p1 = np.maximum(p1, 1e-10)
    p2 = np.maximum(p2, 1e-10)
    p3 = np.maximum(p3, 1e-10)

    entropy = -(p1 * np.log(p1) + p2 * np.log(p2) + p3 * np.log(p3)) / np.log(3)

    # Anisotropy
    anisotropy = (p2 - p3) / (p2 + p3 + 1e-10)

    # Alpha angle (simplified)
    alpha = np.arccos(np.sqrt(p1)) * 180 / np.pi

    return PolarimetricResult(
        decomposition_type="h_alpha",
        components={
            "p1": p1.astype(np.float32),
            "p2": p2.astype(np.float32),
            "p3": p3.astype(np.float32),
        },
        entropy=entropy.astype(np.float32),
        anisotropy=anisotropy.astype(np.float32),
        alpha=alpha.astype(np.float32),
    )
