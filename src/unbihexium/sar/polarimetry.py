# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/sar/polarimetry.py
# Title       : Polarimetric SAR matrices and target decompositions
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Decompositions of fully polarimetric (quad-pol) SAR data given as the
# complex scattering amplitudes S_HH, S_HV and S_VV (and optionally S_VH):
#
#   coherency_matrix               window-averaged Pauli coherency T3
#   covariance_matrix              window-averaged lexicographic covariance C3
#   pauli_decomposition            |HH + VV|^2 / 2, |HH - VV|^2 / 2, 2 |HV|^2
#   pauli_rgb                      Pauli colour composite scaled to [0, 1]
#   freeman_durden_decomposition   surface, dihedral and volume powers
#   yamaguchi_decomposition        adds helix scattering and adapts the
#                                  volume model to the HH / VV ratio
#   h_a_alpha                      entropy, anisotropy and mean alpha angle
#                                  of a coherency matrix
#   h_alpha_decomposition          the same from scattering amplitudes
#   h_alpha_zones                  the nine zones of the H / alpha plane
#   compute_polarimetric_decomposition   dispatcher by name
#
# Conventions
# -----------
# Reciprocity S_HV = S_VH is assumed; when S_VH is given, the cross-polar
# channel is their mean. The Pauli target vector is
# k = (S_HH + S_VV, S_HH - S_VV, 2 S_HV) / sqrt(2) and T3 = <k k^H>; the
# lexicographic vector is (S_HH, sqrt(2) S_HV, S_VV). Ensemble averages <.>
# are moving window means of size `window_size`. The span
# |HH|^2 + 2 |HV|^2 + |VV|^2 equals the trace of T3 and of C3, and the
# Freeman-Durden and Yamaguchi powers sum to the span wherever no power had
# to be clipped at zero.
#
# References
# ----------
# Cloude, S. R., Pottier, E. (1996). A review of target decomposition
#   theorems in radar polarimetry. IEEE Transactions on Geoscience and
#   Remote Sensing, 34(2), 498-518.
# Cloude, S. R., Pottier, E. (1997). An entropy based classification scheme
#   for land applications of polarimetric SAR. IEEE Transactions on
#   Geoscience and Remote Sensing, 35(1), 68-78.
# Freeman, A., Durden, S. L. (1998). A three-component scattering model for
#   polarimetric SAR data. IEEE Transactions on Geoscience and Remote
#   Sensing, 36(3), 963-973.
# Yamaguchi, Y., Moriyama, T., Ishido, M., Yamada, H. (2005). Four-component
#   scattering model for polarimetric SAR image decomposition. IEEE
#   Transactions on Geoscience and Remote Sensing, 43(8), 1699-1706.
# Lee, J.-S., Pottier, E. (2009). Polarimetric Radar Imaging: From Basics to
#   Applications. CRC Press, Boca Raton.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Result record.
from dataclasses import dataclass, field

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Moving window filters.
from scipy import ndimage

# Names accepted by compute_polarimetric_decomposition.
DECOMPOSITIONS = ("pauli", "freeman_durden", "yamaguchi", "h_alpha")


# Result of a polarimetric decomposition.
@dataclass
class PolarimetricResult:
    # Name of the decomposition.
    decomposition_type: str
    # Output images by name (powers, angles, eigenvalues).
    components: dict[str, NDArray[Any]] = field(default_factory=dict)
    # Entropy H in [0, 1] (H / A / alpha only).
    entropy: NDArray[np.floating[Any]] | None = None
    # Anisotropy A in [0, 1] (H / A / alpha only).
    anisotropy: NDArray[np.floating[Any]] | None = None
    # Mean alpha angle in degrees (H / A / alpha only).
    alpha: NDArray[np.floating[Any]] | None = None


# Complex channels as complex128 arrays of one shape, with the reciprocal cross-pol.
def _channels(
    hh: NDArray[Any],  # HH channel.
    hv: NDArray[Any],  # HV channel.
    vv: NDArray[Any],  # VV channel.
    vh: NDArray[Any] | None,  # Optional VH channel.
) -> tuple[NDArray[np.complex128], ...]:  # HH, HV and VV channels.
    # Convert the channels.
    shh, shv, svv = (np.asarray(c, dtype=np.complex128) for c in (hh, hv, vv))
    # All channels must share one shape.
    if not shh.shape == shv.shape == svv.shape:
        # Report the mismatch.
        raise ValueError(f"channel shapes differ: {shh.shape}, {shv.shape}, {svv.shape}")
    # Average the cross-polar channels under reciprocity.
    if vh is not None:
        # VH channel.
        svh = np.asarray(vh, dtype=np.complex128)
        # Same shape as the others.
        if svh.shape != shh.shape:
            # Report the mismatch.
            raise ValueError(f"VH shape {svh.shape} differs from {shh.shape}")
        # Reciprocal cross-polar term.
        shv = 0.5 * (shv + svh)
    # Return the channels.
    return shh, shv, svv


# Moving average of a real or complex image over the last two axes.
def _window_mean(data: NDArray[Any], window_size: int) -> NDArray[Any]:
    # No averaging for a one-pixel window.
    if window_size == 1:
        # Return the input.
        return data
    # Filter only over the image axes.
    size = (1,) * (data.ndim - 2) + (window_size, window_size)
    # Average real parts.
    real = ndimage.uniform_filter(np.real(data), size=size, mode="reflect")
    # Real data needs no imaginary part.
    if not np.iscomplexobj(data):
        # Return the real mean.
        return real
    # Average imaginary parts.
    return real + 1j * ndimage.uniform_filter(np.imag(data), size=size, mode="reflect")


# Check the averaging window.
def _check_window(window_size: int) -> int:
    # The window must be odd and positive.
    if window_size < 1 or window_size % 2 == 0:
        # Report the invalid window.
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
    # Return the size.
    return int(window_size)


# Outer products <k k^H> of a target vector stack, shape (..., 3, 3).
def _outer_mean(k: NDArray[np.complex128], window_size: int) -> NDArray[np.complex128]:
    # Outer product per pixel with the matrix axes first: (3, 3, ...).
    outer = k[:, None] * np.conj(k[None, :])
    # Average every element over the window.
    averaged = _window_mean(outer, _check_window(window_size))
    # Move the matrix axes last.
    return np.moveaxis(averaged, (0, 1), (-2, -1))


# Pauli coherency matrix T3, shape (..., 3, 3).
def coherency_matrix(
    hh: NDArray[Any],  # HH channel.
    hv: NDArray[Any],  # HV channel.
    vv: NDArray[Any],  # VV channel.
    vh: NDArray[Any] | None = None,  # Optional VH channel.
    window_size: int = 5,  # Odd averaging window.
) -> NDArray[np.complex128]:  # T3 per pixel.
    # Channels.
    shh, shv, svv = _channels(hh, hv, vv, vh)
    # Pauli target vector.
    k = np.stack([shh + svv, shh - svv, 2.0 * shv]) / np.sqrt(2.0)
    # Averaged outer product.
    return _outer_mean(k, window_size)


# Lexicographic covariance matrix C3, shape (..., 3, 3).
def covariance_matrix(
    hh: NDArray[Any],  # HH channel.
    hv: NDArray[Any],  # HV channel.
    vv: NDArray[Any],  # VV channel.
    vh: NDArray[Any] | None = None,  # Optional VH channel.
    window_size: int = 5,  # Odd averaging window.
) -> NDArray[np.complex128]:  # C3 per pixel.
    # Channels.
    shh, shv, svv = _channels(hh, hv, vv, vh)
    # Lexicographic target vector.
    k = np.stack([shh, np.sqrt(2.0) * shv, svv])
    # Averaged outer product.
    return _outer_mean(k, window_size)


# Pauli decomposition powers of each pixel.
def pauli_decomposition(
    hh: NDArray[Any],  # HH channel.
    hv: NDArray[Any],  # HV channel.
    vv: NDArray[Any],  # VV channel.
    vh: NDArray[Any] | None = None,  # Optional VH channel.
) -> PolarimetricResult:  # Components surface, dihedral and volume.
    # Channels.
    shh, shv, svv = _channels(hh, hv, vv, vh)
    # Odd-bounce (surface) power.
    surface = np.abs(shh + svv) ** 2 / 2.0
    # Even-bounce (dihedral) power.
    dihedral = np.abs(shh - svv) ** 2 / 2.0
    # Cross-polar (volume) power.
    volume = 2.0 * np.abs(shv) ** 2
    # Package the result.
    return PolarimetricResult(
        decomposition_type="pauli",  # Name.
        components={"surface": surface, "dihedral": dihedral, "volume": volume},  # Powers.
    )  # End of the result.


# Pauli RGB composite: red |HH - VV|, green 2 |HV|, blue |HH + VV|, scaled to [0, 1].
def pauli_rgb(
    hh: NDArray[Any],  # HH channel.
    hv: NDArray[Any],  # HV channel.
    vv: NDArray[Any],  # VV channel.
    vh: NDArray[Any] | None = None,  # Optional VH channel.
    percentile: float = 98.0,  # Percentile of each channel mapped to 1.
) -> NDArray[np.float64]:  # Image of shape (..., 3).
    # Pauli powers.
    powers = pauli_decomposition(hh, hv, vv, vh).components
    # Amplitudes in display order.
    names = ("dihedral", "volume", "surface")
    # Stack the amplitudes as the last axis.
    rgb = np.stack([np.sqrt(powers[name]) for name in names], axis=-1)
    # Scale each channel by its percentile.
    for band in range(3):
        # Percentile of the finite values.
        top = np.nanpercentile(rgb[..., band], percentile)
        # Avoid division by zero for empty channels.
        if top > 0:
            # Scale the channel.
            rgb[..., band] = rgb[..., band] / top
    # Limit to [0, 1].
    return np.clip(rgb, 0.0, 1.0)


# Solve the surface and dihedral powers of the Freeman-Durden model.
def _surface_dihedral(
    a: NDArray[np.float64],  # <|HH|^2> left after volume (and helix) removal.
    b: NDArray[np.float64],  # <|VV|^2> left after removal.
    c: NDArray[np.complex128],  # <HH VV*> left after removal.
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # Surface and dihedral powers.
    # Determinant of the remaining 2 x 2 matrix.
    det = a * b - np.abs(c) ** 2
    # Surface scattering dominates where Re(C) >= 0 (alpha fixed to -1).
    surface_dominant = np.real(c) >= 0
    # Denominators of the two cases.
    den_s = a + b + 2.0 * np.real(c)
    # Double-bounce case denominator.
    den_d = a + b - 2.0 * np.real(c)
    # Solve without dividing by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Surface-dominant case: fd from the determinant.
        fd_s = np.where(den_s > 0, det / np.where(den_s > 0, den_s, 1.0), 0.0)
        # Surface amplitude of that case.
        fs_s = b - fd_s
        # beta = (C + fd) / fs.
        beta = np.where(fs_s != 0, (c + fd_s) / np.where(fs_s != 0, fs_s, 1.0), 0.0)
        # Double-bounce-dominant case: fs from the determinant (beta fixed to 1).
        fs_d = np.where(den_d > 0, det / np.where(den_d > 0, den_d, 1.0), 0.0)
        # Dihedral amplitude of that case.
        fd_d = b - fs_d
        # alpha = (C - fs) / fd.
        alpha = np.where(fd_d != 0, (c - fs_d) / np.where(fd_d != 0, fd_d, 1.0), 0.0)
    # Surface power fs (1 + |beta|^2) or fs (1 + 1).
    ps = np.where(surface_dominant, fs_s * (1.0 + np.abs(beta) ** 2), 2.0 * fs_d)
    # Dihedral power fd (1 + 1) or fd (1 + |alpha|^2).
    pd = np.where(surface_dominant, 2.0 * fd_s, fd_d * (1.0 + np.abs(alpha) ** 2))
    # Physical powers are never negative.
    return np.maximum(ps, 0.0), np.maximum(pd, 0.0)


# Freeman-Durden (1998) three-component decomposition.
def freeman_durden_decomposition(
    hh: NDArray[Any],  # HH channel.
    hv: NDArray[Any],  # HV channel.
    vv: NDArray[Any],  # VV channel.
    vh: NDArray[Any] | None = None,  # Optional VH channel.
    window_size: int = 5,  # Odd averaging window.
) -> PolarimetricResult:  # Components surface, dihedral, volume and span.
    # Covariance matrix.
    c3 = covariance_matrix(hh, hv, vv, vh, window_size)
    # Co-polar powers and correlation.
    c11, c33, c13 = np.real(c3[..., 0, 0]), np.real(c3[..., 2, 2]), c3[..., 0, 2]
    # Cross-polar power <|HV|^2>.
    hv2 = np.real(c3[..., 1, 1]) / 2.0
    # Total power.
    span = c11 + 2.0 * hv2 + c33
    # Volume coefficient fv = 3 <|HV|^2>.
    fv = 3.0 * hv2
    # Co-polar powers left after volume removal.
    a, b = c11 - fv, c33 - fv
    # The volume model overflows where it exceeds a co-polar power.
    overflow = (a <= 0) | (b <= 0)
    # Surface and dihedral powers of the remainder.
    ps, pd = _surface_dihedral(np.maximum(a, 0.0), np.maximum(b, 0.0), c13 - fv / 3.0)
    # No surface or dihedral power where the volume overflowed.
    ps, pd = np.where(overflow, 0.0, ps), np.where(overflow, 0.0, pd)
    # Volume power 8 fv / 3, or the whole span on overflow.
    pv = np.where(overflow, span, 8.0 * fv / 3.0)
    # Package the result.
    return PolarimetricResult(
        decomposition_type="freeman_durden",  # Name.
        components={"surface": ps, "dihedral": pd, "volume": pv, "span": span},  # Powers.
    )  # End of the result.


# Yamaguchi et al. (2005) four-component decomposition.
def yamaguchi_decomposition(
    hh: NDArray[Any],  # HH channel.
    hv: NDArray[Any],  # HV channel.
    vv: NDArray[Any],  # VV channel.
    vh: NDArray[Any] | None = None,  # Optional VH channel.
    window_size: int = 5,  # Odd averaging window.
) -> PolarimetricResult:  # Components surface, dihedral, volume, helix and span.
    # Channels.
    shh, shv, svv = _channels(hh, hv, vv, vh)
    # Covariance matrix.
    c3 = covariance_matrix(shh, shv, svv, None, window_size)
    # Co-polar powers and correlation.
    c11, c33, c13 = np.real(c3[..., 0, 0]), np.real(c3[..., 2, 2]), c3[..., 0, 2]
    # Cross-polar power <|HV|^2>.
    hv2 = np.real(c3[..., 1, 1]) / 2.0
    # Total power.
    span = c11 + 2.0 * hv2 + c33
    # Helix power 2 |Im <HV* (HH - VV)>|.
    pc = 2.0 * np.abs(np.imag(_window_mean(np.conj(shv) * (shh - svv), window_size)))
    # Co-polar ratio 10 log10(<|VV|^2> / <|HH|^2>) in dB.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Ratio; empty windows give NaN and use the symmetric model.
        ratio = 10.0 * np.log10(c33 / c11)
    # Asymmetric volume models for strongly unequal co-polar powers.
    asymmetric = np.abs(np.nan_to_num(ratio)) > 2.0
    # Volume power: 8 <|HV|^2> - 2 Pc (symmetric) or 15/2 <|HV|^2> - 15/8 Pc.
    pv = np.where(asymmetric, 7.5 * hv2 - 1.875 * pc, 8.0 * hv2 - 2.0 * pc)
    # Negative volume power means the helix term was overestimated; drop it.
    pc = np.where(pv < 0, 0.0, pc)
    # Recompute the volume power without helix where needed.
    pv = np.where(pv < 0, np.where(asymmetric, 7.5 * hv2, 8.0 * hv2), pv)
    # Fractions of the volume power in HH, VV and HH VV* per model.
    f_hh = np.where(asymmetric, np.where(ratio < 0, 8.0, 3.0) / 15.0, 3.0 / 8.0)
    # VV fraction.
    f_vv = np.where(asymmetric, np.where(ratio < 0, 3.0, 8.0) / 15.0, 3.0 / 8.0)
    # HH VV* fraction.
    f_hv = np.where(asymmetric, 2.0 / 15.0, 1.0 / 8.0)
    # Remainder after volume and helix removal; the helix adds -Pc/4 to HH VV*.
    a = c11 - f_hh * pv - pc / 4.0
    # Remaining VV power.
    b = c33 - f_vv * pv - pc / 4.0
    # Remaining correlation.
    c = c13 - f_hv * pv + pc / 4.0
    # Volume and helix exceed the co-polar power: all remaining power is volume.
    overflow = (a <= 0) | (b <= 0)
    # Surface and dihedral powers.
    ps, pd = _surface_dihedral(np.maximum(a, 0.0), np.maximum(b, 0.0), c)
    # No surface or dihedral power where the model overflowed.
    ps, pd = np.where(overflow, 0.0, ps), np.where(overflow, 0.0, pd)
    # Volume takes the rest of the span in that case.
    pv = np.where(overflow, np.maximum(span - pc, 0.0), pv)
    # Package the result.
    return PolarimetricResult(
        decomposition_type="yamaguchi",  # Name.
        components={  # Powers by mechanism.
            "surface": ps,  # Odd-bounce power.
            "dihedral": pd,  # Even-bounce power.
            "volume": pv,  # Volume power.
            "helix": pc,  # Helix power.
            "span": span,  # Total power.
        },  # End of the components.
    )  # End of the result.


# Entropy, anisotropy, mean alpha and eigenvalues of coherency matrices (..., 3, 3).
def h_a_alpha(
    t3: NDArray[Any],  # Hermitian coherency matrices.
) -> dict[str, NDArray[np.float64]]:  # entropy, anisotropy, alpha, lambda1..3.
    # Hermitian stack.
    t = np.asarray(t3, dtype=np.complex128)
    # The last two axes must form 3 x 3 matrices.
    if t.shape[-2:] != (3, 3):
        # Report the wrong shape.
        raise ValueError(f"expected coherency matrices of shape (..., 3, 3), got {t.shape}")
    # Pixels with non-finite entries.
    bad = ~np.all(np.isfinite(t), axis=(-2, -1))
    # Replace them by the identity for the solver.
    t = np.where(bad[..., None, None], np.eye(3), t)
    # Eigenvalues in ascending order with unit eigenvectors in columns.
    values, vectors = np.linalg.eigh(t)
    # Descending order; round-off negatives set to zero.
    lam = np.maximum(values[..., ::-1], 0.0)
    # Eigenvectors in the same order.
    vec = vectors[..., ::-1]
    # Total power.
    total = lam.sum(axis=-1, keepdims=True)
    # Pseudo-probabilities.
    p = np.divide(lam, total, out=np.zeros_like(lam), where=total > 0)
    # Entropy with 0 log 0 = 0, logarithm base 3.
    with np.errstate(divide="ignore", invalid="ignore"):
        # -sum p log3 p.
        entropy = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0), axis=-1)
    # Base-3 normalisation.
    entropy = entropy / np.log(3.0)
    # Anisotropy (l2 - l3) / (l2 + l3), zero when both vanish.
    den = lam[..., 1] + lam[..., 2]
    # Divide where defined.
    anisotropy = np.divide(lam[..., 1] - lam[..., 2], den, out=np.zeros_like(den), where=den > 0)
    # Alpha angles of the eigenvectors from their first (odd-bounce) element.
    alphas = np.degrees(np.arccos(np.clip(np.abs(vec[..., 0, :]), 0.0, 1.0)))
    # Mean alpha weighted by the probabilities.
    alpha = np.sum(p * alphas, axis=-1)
    # Outputs per pixel.
    out = {
        "entropy": np.clip(entropy, 0.0, 1.0),  # Entropy in [0, 1].
        "anisotropy": anisotropy,  # Anisotropy in [0, 1].
        "alpha": alpha,  # Mean alpha in degrees.
        "lambda1": lam[..., 0],  # Largest eigenvalue.
        "lambda2": lam[..., 1],  # Middle eigenvalue.
        "lambda3": lam[..., 2],  # Smallest eigenvalue.
    }  # End of the outputs.
    # Restore NaN at invalid pixels.
    return {name: np.where(bad, np.nan, value) for name, value in out.items()}


# Cloude-Pottier H / A / alpha decomposition of scattering amplitudes.
def h_alpha_decomposition(
    hh: NDArray[Any],  # HH channel.
    hv: NDArray[Any],  # HV channel.
    vv: NDArray[Any],  # VV channel.
    vh: NDArray[Any] | None = None,  # Optional VH channel.
    window_size: int = 5,  # Odd averaging window; one pixel gives H = 0 everywhere.
) -> PolarimetricResult:  # Components entropy, anisotropy, alpha, lambda1..3.
    # Coherency matrices.
    t3 = coherency_matrix(hh, hv, vv, vh, window_size)
    # Eigen-analysis.
    parts = h_a_alpha(t3)
    # Package the result.
    return PolarimetricResult(
        decomposition_type="h_alpha",  # Name.
        components=parts,  # All outputs.
        entropy=parts["entropy"],  # Entropy.
        anisotropy=parts["anisotropy"],  # Anisotropy.
        alpha=parts["alpha"],  # Mean alpha.
    )  # End of the result.


# Zones 1 to 9 of the H / alpha plane (Lee and Pottier, 2009), 0 for NaN.
def h_alpha_zones(
    entropy: NDArray[Any],  # Entropy in [0, 1].
    alpha: NDArray[Any],  # Mean alpha in degrees.
) -> NDArray[np.int64]:  # Zone numbers.
    # Inputs as float arrays.
    h = np.asarray(entropy, dtype=np.float64)
    # Alpha in degrees.
    a = np.asarray(alpha, dtype=np.float64)
    # High entropy (H > 0.9): multiple (1), vegetation (2), non-feasible surface (3).
    high = np.where(a > 55.0, 1, np.where(a > 40.0, 2, 3))
    # Medium entropy (0.5 < H <= 0.9): multiple (4), vegetation (5), surface (6).
    medium = np.where(a > 50.0, 4, np.where(a > 40.0, 5, 6))
    # Low entropy (H <= 0.5): multiple (7), dipole (8), surface (9).
    low = np.where(a > 47.5, 7, np.where(a > 42.5, 8, 9))
    # Select by entropy.
    zones = np.where(h > 0.9, high, np.where(h > 0.5, medium, low))
    # Undefined pixels are zone 0.
    return np.where(np.isfinite(h) & np.isfinite(a), zones, 0).astype(np.int64)


# Decomposition selected by name.
def compute_polarimetric_decomposition(
    hh: NDArray[Any],  # HH channel.
    hv: NDArray[Any],  # HV channel.
    vv: NDArray[Any],  # VV channel.
    decomposition: str = "pauli",  # One of DECOMPOSITIONS.
    vh: NDArray[Any] | None = None,  # Optional VH channel.
    window_size: int = 5,  # Averaging window of the covariance-based decompositions.
) -> PolarimetricResult:  # Decomposition result.
    # Normalise the name.
    name = decomposition.lower().replace("-", "_")
    # Pauli basis.
    if name == "pauli":
        # Per-pixel powers.
        return pauli_decomposition(hh, hv, vv, vh)
    # Freeman-Durden.
    if name == "freeman_durden":
        # Three components.
        return freeman_durden_decomposition(hh, hv, vv, vh, window_size)
    # Yamaguchi.
    if name == "yamaguchi":
        # Four components.
        return yamaguchi_decomposition(hh, hv, vv, vh, window_size)
    # Cloude-Pottier.
    if name == "h_alpha":
        # Entropy, anisotropy and alpha.
        return h_alpha_decomposition(hh, hv, vv, vh, window_size)
    # Unknown name.
    raise ValueError(f"unknown decomposition {decomposition!r}; expected one of {DECOMPOSITIONS}")


# =============================================================================
# End of module src/unbihexium/sar/polarimetry.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
