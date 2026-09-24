# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/sar/interferometry.py
# Title       : Interferogram formation, coherence and phase unwrapping
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Interferometric SAR (InSAR) processing of co-registered single look
# complex (SLC) images s1 (reference) and s2 (secondary):
#
#   compute_interferogram   s1 * conj(s2), optionally averaged over a
#                           moving window, and its wrapped phase
#   compute_coherence       sample coherence magnitude
#   wrap_phase              phase wrapped to [-pi, pi)
#   phase_residues          charges of the 2 x 2 phase loops
#   goldstein_filter        adaptive power spectrum filter
#   phase_unwrapping        Itoh, weighted least squares (DCT preconditioned
#                           conjugate gradient) and quality-guided unwrapping
#   compute_displacement    unwrapped phase to line-of-sight range change
#   los_to_vertical         range change to vertical motion
#   height_of_ambiguity     topographic height per 2 pi of phase
#
# Sign convention
# ---------------
# With the phase of a return phi = -4 pi R / lambda, the interferometric
# phase is phi1 - phi2 = 4 pi (R2 - R1) / lambda. compute_displacement
# returns the range change R2 - R1 = lambda phi / (4 pi): positive values
# are motion away from the sensor (for example subsidence), negative values
# motion towards it. Processors that use the opposite phase convention
# should negate the phase first.
#
# Least-squares unwrapping
# ------------------------
# The unwrapped phase minimises sum w (D phi - wrap(D psi))^2 over the
# horizontal and vertical differences D of the grid (Ghiglia and Romero,
# 1994). Its normal equations are a weighted Poisson equation with Neumann
# boundary conditions. The unweighted equation is solved exactly with the
# discrete cosine transform; with weights, it serves as the preconditioner
# of a conjugate gradient solver. Without residues and with |D psi| < pi,
# the solution equals the true phase up to a constant.
#
# References
# ----------
# Itoh, K. (1982). Analysis of the phase unwrapping algorithm. Applied
#   Optics, 21(14), 2470.
# Ghiglia, D. C., Romero, L. A. (1994). Robust two-dimensional weighted and
#   unweighted phase unwrapping that uses fast transforms and iterative
#   methods. Journal of the Optical Society of America A, 11(1), 107-117.
# Ghiglia, D. C., Pritt, M. D. (1998). Two-Dimensional Phase Unwrapping:
#   Theory, Algorithms, and Software. Wiley, New York.
# Goldstein, R. M., Zebker, H. A., Werner, C. L. (1988). Satellite radar
#   interferometry: two-dimensional phase unwrapping. Radio Science, 23(4),
#   713-720.
# Goldstein, R. M., Werner, C. L. (1998). Radar interferogram filtering for
#   geophysical applications. Geophysical Research Letters, 25(21),
#   4035-4038.
# Herraez, M. A., Burton, D. R., Lalor, M. J., Gdeisat, M. A. (2002). Fast
#   two-dimensional phase-unwrapping algorithm based on sorting by
#   reliability following a noncontinuous path. Applied Optics, 41(35),
#   7437-7444.
# Hanssen, R. F. (2001). Radar Interferometry: Data Interpretation and
#   Error Analysis. Kluwer Academic Publishers, Dordrecht.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Priority queue of the quality-guided unwrapper.
import heapq

# Result record.
from dataclasses import dataclass

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Moving window filters.
from scipy import ndimage

# Discrete cosine transforms for the Poisson solver.
from scipy.fft import dctn, idctn

# Conjugate gradient solver and matrix-free operators.
from scipy.sparse.linalg import LinearOperator, cg

# Unwrapping methods accepted by phase_unwrapping.
UNWRAP_METHODS = ("least_squares", "quality_guided", "simple")


# Products of interferometric processing.
@dataclass
class InterferometricResult:
    # Coherence magnitude in [0, 1].
    coherence: NDArray[np.floating[Any]]
    # Wrapped interferometric phase in radians.
    phase: NDArray[np.floating[Any]]
    # Unwrapped phase in radians, when computed.
    unwrapped_phase: NDArray[np.floating[Any]] | None = None


# Wrap phase values to [-pi, pi).
def wrap_phase(phase: NDArray[Any]) -> NDArray[np.float64]:
    # Shift, reduce modulo 2 pi and shift back.
    return (np.asarray(phase, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


# Check that two SLC images have the same 2-D shape.
def _check_pair(master: NDArray[Any], slave: NDArray[Any]) -> None:
    # Shapes must agree.
    if np.shape(master) != np.shape(slave):
        # Report the mismatch.
        raise ValueError(f"image shapes differ: {np.shape(master)} and {np.shape(slave)}")
    # Only single images are supported.
    if np.ndim(master) != 2:
        # Report the wrong dimensionality.
        raise ValueError(f"expected 2-D images, got shape {np.shape(master)}")


# Moving average of a complex image over a (rows, cols) window.
def _complex_mean(data: NDArray[Any], size: tuple[int, int]) -> NDArray[np.complex128]:
    # Average the real part.
    real = ndimage.uniform_filter(np.real(data).astype(np.float64), size=size, mode="reflect")
    # Average the imaginary part.
    imag = ndimage.uniform_filter(np.imag(data).astype(np.float64), size=size, mode="reflect")
    # Recombine.
    return real + 1j * imag


# Interferogram s1 * conj(s2) and its wrapped phase.
def compute_interferogram(
    master: NDArray[Any],  # Reference SLC.
    slave: NDArray[Any],  # Secondary SLC, co-registered to the reference.
    multilook: tuple[int, int] = (1, 1),  # Moving window (rows, cols) for complex averaging.
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:  # Phase and complex interferogram.
    # Check the inputs.
    _check_pair(master, slave)
    # Pixel-wise interferogram.
    ifg = np.asarray(master, dtype=np.complex128) * np.conj(np.asarray(slave, dtype=np.complex128))
    # Window sizes.
    rows, cols = (int(v) for v in multilook)
    # Both sizes must be positive.
    if rows < 1 or cols < 1:
        # Report the invalid window.
        raise ValueError(f"multilook sizes must be positive, got {multilook}")
    # Complex averaging keeps the grid and reduces phase noise.
    if (rows, cols) != (1, 1):
        # Moving average of the interferogram.
        ifg = _complex_mean(ifg, (rows, cols))
    # Wrapped phase in [-pi, pi].
    return np.angle(ifg), ifg


# Sample coherence |<s1 s2*>| / sqrt(<|s1|^2> <|s2|^2>) over a square window.
def compute_coherence(
    master: NDArray[Any],  # Reference SLC.
    slave: NDArray[Any],  # Secondary SLC.
    window_size: int = 5,  # Odd estimation window size.
) -> NDArray[np.float64]:  # Coherence magnitude in [0, 1].
    # Check the inputs.
    _check_pair(master, slave)
    # The window must be odd and positive.
    if window_size < 1 or window_size % 2 == 0:
        # Report the invalid window.
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
    # Complex inputs.
    s1 = np.asarray(master, dtype=np.complex128)
    # Secondary image.
    s2 = np.asarray(slave, dtype=np.complex128)
    # Mean cross product.
    cross = _complex_mean(s1 * np.conj(s2), (window_size, window_size))
    # Mean power of the reference.
    p1 = ndimage.uniform_filter(np.abs(s1) ** 2, size=window_size, mode="reflect")
    # Mean power of the secondary.
    p2 = ndimage.uniform_filter(np.abs(s2) ** 2, size=window_size, mode="reflect")
    # Normalisation.
    denom = np.sqrt(p1 * p2)
    # Zero power windows have zero coherence.
    coherence = np.divide(np.abs(cross), denom, out=np.zeros_like(denom), where=denom > 0)
    # Guard against round-off above one; NaN input stays NaN.
    return np.where(np.isnan(denom), np.nan, np.clip(coherence, 0.0, 1.0))


# Residue charges (-1, 0, +1) of the 2 x 2 loops of a wrapped phase image.
def phase_residues(phase: NDArray[Any]) -> NDArray[np.int64]:
    # Wrapped phase.
    psi = np.asarray(phase, dtype=np.float64)
    # Wrapped differences around each loop, walking clockwise.
    loop = (
        wrap_phase(psi[:-1, 1:] - psi[:-1, :-1])  # Top edge, left to right.
        + wrap_phase(psi[1:, 1:] - psi[:-1, 1:])  # Right edge, downwards.
        + wrap_phase(psi[1:, :-1] - psi[1:, 1:])  # Bottom edge, right to left.
        + wrap_phase(psi[:-1, :-1] - psi[1:, :-1])  # Left edge, upwards.
    )  # End of the loop sum.
    # Loops sum to a multiple of 2 pi; NaN loops count as zero.
    return np.rint(np.nan_to_num(loop, nan=0.0) / (2.0 * np.pi)).astype(np.int64)


# Goldstein and Werner (1998) adaptive filter of a complex interferogram.
def goldstein_filter(
    interferogram: NDArray[Any],  # Complex interferogram.
    alpha: float = 0.5,  # Filter exponent in [0, 1]; 0 leaves the input unchanged.
    patch_size: int = 32,  # Size of the FFT patches (even).
    smoothing: int = 3,  # Size of the moving average applied to the spectrum magnitude.
) -> NDArray[np.complex128]:  # Filtered interferogram.
    # The exponent is limited to [0, 1].
    if not 0.0 <= alpha <= 1.0:
        # Report the invalid exponent.
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    # Patches must be even and at least 4 pixels wide.
    if patch_size < 4 or patch_size % 2:
        # Report the invalid patch size.
        raise ValueError(f"patch_size must be an even integer >= 4, got {patch_size}")
    # Complex input with NaN replaced by zero.
    ifg = np.nan_to_num(np.asarray(interferogram, dtype=np.complex128))
    # Patches overlap by half their size.
    step = patch_size // 2
    # Image size.
    rows, cols = ifg.shape
    # Pad so that the patches cover the whole image.
    pad_r = (-(rows - patch_size) % step) if rows > patch_size else patch_size - rows
    # Column padding.
    pad_c = (-(cols - patch_size) % step) if cols > patch_size else patch_size - cols
    # Padded interferogram.
    mode = "reflect" if min(rows, cols) > 1 else "edge"
    # Pad at the bottom and the right.
    padded = np.pad(ifg, ((0, pad_r), (0, pad_c)), mode=mode)
    # Weighted sum of filtered patches.
    out = np.zeros(padded.shape, dtype=np.complex128)
    # Sum of weights.
    weight = np.zeros(padded.shape, dtype=np.float64)
    # Triangular taper that blends overlapping patches.
    taper1d = 1.0 - np.abs(np.arange(patch_size) + 0.5 - patch_size / 2) / (patch_size / 2)
    # Two-dimensional taper.
    taper = np.outer(taper1d, taper1d)
    # Visit every patch.
    for r in range(0, padded.shape[0] - patch_size + 1, step):
        # Visit every column position.
        for c in range(0, padded.shape[1] - patch_size + 1, step):
            # Spectrum of the patch.
            spectrum = np.fft.fft2(padded[r : r + patch_size, c : c + patch_size])
            # Smoothed spectrum magnitude.
            mag = ndimage.uniform_filter(np.abs(spectrum), size=smoothing, mode="wrap")
            # Normalised response raised to alpha; flat patches pass unchanged.
            peak = mag.max()
            # Filter response.
            response = (mag / peak) ** alpha if peak > 0 else np.ones_like(mag)
            # Filtered patch.
            patch = np.fft.ifft2(spectrum * response)
            # Accumulate with the taper.
            out[r : r + patch_size, c : c + patch_size] += taper * patch
            # Accumulate the weights.
            weight[r : r + patch_size, c : c + patch_size] += taper
    # Normalise by the weights.
    result = out / np.where(weight > 0, weight, 1.0)
    # Crop to the input size.
    return result[:rows, :cols]


# Neumann Poisson solver: returns phi with L phi = rho (L = discrete Laplacian), mean zero.
def _poisson_dct(rho: NDArray[np.float64]) -> NDArray[np.float64]:
    # Grid size.
    m, n = rho.shape
    # Transform of the right-hand side.
    spectrum = dctn(rho, type=2, norm="ortho")
    # Eigenvalues of the Neumann Laplacian.
    eig = (
        2.0 * np.cos(np.pi * np.arange(m) / m)[:, None]  # Row modes.
        + 2.0 * np.cos(np.pi * np.arange(n) / n)[None, :]  # Column modes.
        - 4.0  # Diagonal term.
    )  # End of the eigenvalues.
    # The constant mode is undetermined; set it to zero.
    eig[0, 0] = 1.0
    # Divide in the transform domain.
    spectrum = spectrum / eig
    # Zero mean solution.
    spectrum[0, 0] = 0.0
    # Back to the grid.
    return idctn(spectrum, type=2, norm="ortho")


# Divergence D^T applied to horizontal and vertical edge fields.
def _divergence(gx: NDArray[np.float64], gy: NDArray[np.float64]) -> NDArray[np.float64]:
    # Row edges padded with zero columns at both ends.
    px = np.pad(gx, ((0, 0), (1, 1)))
    # Column edges padded with zero rows at both ends.
    py = np.pad(gy, ((1, 1), (0, 0)))
    # Backward differences of the edge fields.
    return (px[:, 1:] - px[:, :-1]) + (py[1:, :] - py[:-1, :])


# Weighted least-squares unwrapping (Ghiglia and Romero, 1994).
def _unwrap_least_squares(
    psi: NDArray[np.float64],  # Wrapped phase, NaN for invalid pixels.
    weights: NDArray[np.float64],  # Pixel weights in [0, 1].
    tol: float = 1e-10,  # Relative tolerance of the conjugate gradient.
    max_iter: int = 500,  # Maximum iterations of the conjugate gradient.
) -> NDArray[np.float64]:  # Unwrapped phase.
    # Invalid pixels get zero weight and a zero phase.
    w = np.where(np.isfinite(psi), weights, 0.0)
    # Phase with zeros at invalid pixels.
    p = np.nan_to_num(psi)
    # Wrapped horizontal differences.
    dx = wrap_phase(np.diff(p, axis=1))
    # Wrapped vertical differences.
    dy = wrap_phase(np.diff(p, axis=0))
    # Edge weights: the smaller squared weight of the two end pixels.
    wx = np.minimum(w[:, 1:], w[:, :-1]) ** 2
    # Vertical edge weights.
    wy = np.minimum(w[1:, :], w[:-1, :]) ** 2
    # Right-hand side of the normal equations (a weighted Laplacian of the phase).
    rho = _divergence(wx * dx, wy * dy)
    # Unweighted problem: one DCT solve is exact.
    if np.all(wx == 1.0) and np.all(wy == 1.0):
        # Direct solution.
        return _poisson_dct(rho)
    # Grid shape.
    shape = psi.shape

    # Weighted Laplacian applied to a flattened phase.
    def apply(v: NDArray[np.float64]) -> NDArray[np.float64]:
        # Phase on the grid.
        phi = v.reshape(shape)
        # Weighted Laplacian, negated so that the operator is positive semi-definite.
        return -_divergence(wx * np.diff(phi, axis=1), wy * np.diff(phi, axis=0)).ravel()

    # DCT Poisson solve as preconditioner (inverse of the negated Laplacian).
    def precondition(v: NDArray[np.float64]) -> NDArray[np.float64]:
        # Solve L z = -v.
        return _poisson_dct(-v.reshape(shape)).ravel()

    # Size of the system.
    size = psi.size
    # Matrix-free operator.
    operator = LinearOperator((size, size), matvec=apply, dtype=np.float64)
    # Matrix-free preconditioner.
    preconditioner = LinearOperator((size, size), matvec=precondition, dtype=np.float64)
    # Preconditioned conjugate gradient.
    solution, _ = cg(operator, -rho.ravel(), rtol=tol, maxiter=max_iter, M=preconditioner)
    # Solution on the grid.
    return solution.reshape(shape)


# Row-then-column Itoh unwrapping (valid only for noise-free phase without residues).
def _unwrap_simple(psi: NDArray[np.float64]) -> NDArray[np.float64]:
    # Unwrap the first column downwards.
    first = np.unwrap(psi[:, 0])
    # Unwrap every row from its first pixel.
    rows = np.unwrap(psi, axis=1)
    # Shift each row so that it starts at the unwrapped first column.
    return rows + (first - rows[:, 0])[:, None]


# Reliability of each pixel from wrapped second differences (Herraez et al., 2002).
def _reliability(psi: NDArray[np.float64]) -> NDArray[np.float64]:
    # Padded phase so that every pixel has eight neighbours.
    p = np.pad(np.nan_to_num(psi), 1, mode="edge")
    # Centre pixels.
    c = p[1:-1, 1:-1]
    # Horizontal second difference.
    h = wrap_phase(p[1:-1, :-2] - c) - wrap_phase(c - p[1:-1, 2:])
    # Vertical second difference.
    v = wrap_phase(p[:-2, 1:-1] - c) - wrap_phase(c - p[2:, 1:-1])
    # Diagonal second difference.
    d1 = wrap_phase(p[:-2, :-2] - c) - wrap_phase(c - p[2:, 2:])
    # Anti-diagonal second difference.
    d2 = wrap_phase(p[:-2, 2:] - c) - wrap_phase(c - p[2:, :-2])
    # Reliability is the inverse of the second-difference magnitude.
    return 1.0 / (np.sqrt(h * h + v * v + d1 * d1 + d2 * d2) + 1e-12)


# Quality-guided flood fill: unwrap the best pixels first (Ghiglia and Pritt, 1998).
def _unwrap_quality(psi: NDArray[np.float64], quality: NDArray[np.float64]) -> NDArray[np.float64]:
    # Image size.
    rows, cols = psi.shape
    # Unwrapped phase; NaN until a pixel is reached.
    out = np.full(psi.shape, np.nan)
    # Pixels that can be unwrapped.
    valid = np.isfinite(psi) & np.isfinite(quality)
    # Visit order of the seeds: best quality first.
    seeds = np.argsort(-np.where(valid, quality, -np.inf), axis=None)
    # Flood fill every connected valid region.
    for seed in seeds:
        # Seed position.
        r0, c0 = divmod(int(seed), cols)
        # Skip invalid or already unwrapped seeds.
        if not valid[r0, c0] or np.isfinite(out[r0, c0]):
            # Next seed.
            continue
        # The seed keeps its wrapped value.
        out[r0, c0] = psi[r0, c0]
        # Frontier ordered by decreasing quality: (-quality, row, col, parent row, parent col).
        heap: list[tuple[float, int, int, int, int]] = []
        # Neighbours of the seed.
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            # Neighbour position.
            r, c = r0 + dr, c0 + dc
            # Add valid neighbours inside the image.
            if 0 <= r < rows and 0 <= c < cols and valid[r, c]:
                # Push with its parent.
                heapq.heappush(heap, (-quality[r, c], r, c, r0, c0))
        # Grow the region.
        while heap:
            # Best frontier pixel.
            _, r, c, pr, pc = heapq.heappop(heap)
            # Pixels may be pushed several times.
            if np.isfinite(out[r, c]):
                # Already unwrapped.
                continue
            # Unwrap relative to the parent.
            out[r, c] = out[pr, pc] + wrap_phase(psi[r, c] - psi[pr, pc])
            # Push the neighbours.
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                # Neighbour position.
                nr, nc = r + dr, c + dc
                # Add valid, not yet unwrapped neighbours.
                if 0 <= nr < rows and 0 <= nc < cols and valid[nr, nc] and np.isnan(out[nr, nc]):
                    # Push with its parent.
                    heapq.heappush(heap, (-quality[nr, nc], nr, nc, r, c))
    # Return the unwrapped phase.
    return out


# Unwrap a wrapped phase image.
def phase_unwrapping(
    phase: NDArray[Any],  # Wrapped phase in radians.
    method: str = "least_squares",  # One of UNWRAP_METHODS.
    coherence: NDArray[Any] | None = None,  # Coherence used as weight or quality.
    coherence_threshold: float = 0.3,  # Pixels with lower coherence are masked (NaN).
) -> NDArray[np.float64]:  # Unwrapped phase, NaN at masked or invalid pixels.
    # Wrapped phase as float64 in [-pi, pi).
    psi = np.asarray(phase, dtype=np.float64)
    # Only images are supported.
    if psi.ndim != 2:
        # Report the wrong shape.
        raise ValueError(f"expected a 2-D phase image, got shape {psi.shape}")
    # Wrap the input in case it exceeds [-pi, pi).
    psi = np.where(np.isfinite(psi), wrap_phase(psi), np.nan)
    # Pixels excluded from unwrapping.
    mask = ~np.isfinite(psi)
    # Coherence as float64, if given.
    coh = None if coherence is None else np.asarray(coherence, dtype=np.float64)
    # Coherence as weights and quality.
    if coh is not None:
        # Shapes must agree.
        if coh.shape != psi.shape:
            # Report the mismatch.
            raise ValueError(f"coherence shape {coh.shape} differs from phase shape {psi.shape}")
        # Mask low or missing coherence.
        mask |= ~(coh >= coherence_threshold)
    # Masked pixels are removed from the input.
    psi = np.where(mask, np.nan, psi)
    # Normalise the method name.
    name = method.lower()
    # Itoh integration.
    if name == "simple":
        # Row and column integration of the wrapped differences.
        result = _unwrap_simple(np.nan_to_num(psi))
    # Weighted least squares.
    elif name == "least_squares":
        # Weights: coherence where given, one otherwise.
        weights = np.clip(coh, 0.0, 1.0) if coh is not None else np.ones_like(psi)
        # Least-squares solution.
        result = _unwrap_least_squares(psi, np.where(mask, 0.0, weights))
        # Remove the arbitrary constant: align with the wrapped phase on average.
        offset = np.angle(np.nanmean(np.exp(1j * (psi - result))))
        # Shifted solution.
        result = result + offset
    # Quality-guided flood fill.
    elif name == "quality_guided":
        # Coherence or phase reliability as quality.
        quality = coh if coh is not None else _reliability(psi)
        # Flood fill.
        result = _unwrap_quality(psi, np.where(mask, np.nan, quality))
    # Unknown method.
    else:
        # Report the valid names.
        raise ValueError(f"unknown unwrapping method {method!r}; expected one of {UNWRAP_METHODS}")
    # Masked pixels are NaN.
    return np.where(mask, np.nan, result)


# Line-of-sight range change R2 - R1 = lambda * phi / (4 pi).
def compute_displacement(
    unwrapped_phase: NDArray[Any],  # Unwrapped phase in radians.
    wavelength: float,  # Radar wavelength in metres (Sentinel-1 C band: 0.05546576).
    incidence_angle: NDArray[Any] | float | None = None,  # Incidence angle in degrees.
    vertical: bool = False,  # Project to vertical motion (needs the incidence angle).
) -> NDArray[np.float64]:  # Range change or vertical motion in metres.
    # The wavelength must be positive.
    if wavelength <= 0:
        # Report the invalid wavelength.
        raise ValueError(f"wavelength must be positive, got {wavelength}")
    # Range change; positive away from the sensor.
    los = wavelength * np.asarray(unwrapped_phase, dtype=np.float64) / (4.0 * np.pi)
    # Range change is the default output.
    if not vertical:
        # Return the line-of-sight change.
        return los
    # Vertical projection needs the incidence angle.
    if incidence_angle is None:
        # Report the missing angle.
        raise ValueError("vertical=True requires the incidence angle")
    # Vertical motion.
    return los_to_vertical(los, incidence_angle)


# Vertical motion (positive up) from range change, assuming purely vertical motion.
def los_to_vertical(
    range_change: NDArray[Any],  # Range change R2 - R1 in metres, positive away from sensor.
    incidence_angle: NDArray[Any] | float,  # Incidence angle in degrees.
) -> NDArray[np.float64]:  # Vertical motion in metres, positive upwards.
    # Incidence angle in radians.
    theta = np.deg2rad(np.asarray(incidence_angle, dtype=np.float64))
    # The angle must lie in [0, 90) degrees.
    if np.any((theta < 0) | (theta >= np.pi / 2)):
        # Report the invalid angle.
        raise ValueError("incidence angle must lie in [0, 90) degrees")
    # Uplift shortens the range.
    return -np.asarray(range_change, dtype=np.float64) / np.cos(theta)


# Height of ambiguity lambda R sin(theta) / (2 B_perp) of a repeat-pass pair.
def height_of_ambiguity(
    wavelength: float,  # Radar wavelength in metres.
    slant_range: float,  # Slant range in metres.
    incidence_angle: float,  # Incidence angle in degrees.
    perpendicular_baseline: float,  # Perpendicular baseline in metres.
) -> float:  # Height difference in metres that produces one 2 pi fringe.
    # A zero baseline has no topographic sensitivity.
    if perpendicular_baseline == 0:
        # Report the degenerate geometry.
        raise ValueError("perpendicular_baseline must be non-zero")
    # Incidence angle in radians.
    theta = np.deg2rad(incidence_angle)
    # Hanssen (2001), repeat-pass (two-way) geometry.
    return float(wavelength * slant_range * np.sin(theta) / (2.0 * perpendicular_baseline))


# =============================================================================
# End of module src/unbihexium/sar/interferometry.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
