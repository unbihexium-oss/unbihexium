# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/preprocessing/masks.py
# Title       : Cloud and quality masks from Sentinel-2 SCL and Landsat QA_PIXEL
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Validity masks from the quality layers delivered with Level-2 products:
#
#   SCL_CLASSES           Sentinel-2 L2A scene classification legend
#   scl_valid_mask        clear pixels from the SCL band
#   qa_bits               extract a bit field from an integer QA band
#   landsat_qa_mask       cloud, shadow, snow, cirrus and fill flags of the
#                         Landsat Collection 2 QA_PIXEL band
#   landsat_cloud_confidence
#                         two-bit confidence fields of QA_PIXEL
#   buffer_mask           grow a mask by a number of pixels
#   apply_mask            set masked pixels of an image to NaN
#
# All masks are boolean. scl_valid_mask returns True for usable pixels;
# landsat_qa_mask returns True for flagged (unusable) pixels, the same way
# the flags are stored in the QA band.
#
# Landsat 8/9 Collection 2 QA_PIXEL bits: 0 fill, 1 dilated cloud,
# 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow, 6 clear, 7 water,
# 8-9 cloud confidence, 10-11 cloud shadow confidence, 12-13 snow/ice
# confidence, 14-15 cirrus confidence (0 none, 1 low, 2 medium, 3 high).
#
# References
# ----------
#   ESA. Sentinel-2 Level-2A algorithm theoretical basis document,
#     S2PAD-ATBD-0001 (scene classification).
#   Main-Knorn, M., et al. (2017). Sen2Cor for Sentinel-2. Proceedings of
#     SPIE 10427, Image and Signal Processing for Remote Sensing XXIII.
#   USGS. Landsat 8-9 Collection 2 Level 2 science product guide (LSDS-1619),
#     pixel quality assessment band.
#   Foga, S., et al. (2017). Cloud detection algorithm comparison and
#     validation for operational Landsat data products. Remote Sensing of
#     Environment 194, 379-390.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values.
from typing import Any, Iterable

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Binary dilation for mask buffers.
from scipy.ndimage import binary_dilation

# Sentinel-2 L2A scene classification (SCL) values and their meaning.
SCL_CLASSES = {
    0: "no_data",  # No data.
    1: "saturated_or_defective",  # Saturated or defective pixel.
    2: "dark_area_pixels",  # Dark features and topographic shadows.
    3: "cloud_shadows",  # Cloud shadows.
    4: "vegetation",  # Vegetation.
    5: "not_vegetated",  # Bare soils and other non-vegetated surfaces.
    6: "water",  # Water.
    7: "unclassified",  # Unclassified.
    8: "cloud_medium_probability",  # Cloud, medium probability.
    9: "cloud_high_probability",  # Cloud, high probability.
    10: "thin_cirrus",  # Thin cirrus.
    11: "snow",  # Snow and ice.
}  # End of the SCL legend.

# SCL classes treated as invalid by default.
SCL_DEFAULT_INVALID = (0, 1, 3, 8, 9, 10)

# Bit positions of the Landsat Collection 2 QA_PIXEL flags.
LANDSAT_QA_BITS = {
    "fill": 0,  # Fill (no data).
    "dilated_cloud": 1,  # Cloud dilated by the processing system.
    "cirrus": 2,  # Cirrus (Landsat 8/9 only).
    "cloud": 3,  # Cloud.
    "cloud_shadow": 4,  # Cloud shadow.
    "snow": 5,  # Snow.
    "clear": 6,  # Clear of cloud and dilated cloud.
    "water": 7,  # Water.
}  # End of the single-bit flags.

# Start bits of the two-bit confidence fields of QA_PIXEL.
LANDSAT_QA_CONFIDENCE = {
    "cloud": 8,  # Cloud confidence.
    "cloud_shadow": 10,  # Cloud shadow confidence.
    "snow": 12,  # Snow and ice confidence.
    "cirrus": 14,  # Cirrus confidence.
}  # End of the confidence fields.


# Mask of usable pixels from a Sentinel-2 SCL band.
def scl_valid_mask(
    scl: NDArray[Any],  # Scene classification values.
    invalid: Iterable[int] = SCL_DEFAULT_INVALID,  # Classes that are not usable.
) -> NDArray[np.bool_]:  # True where the pixel is usable.
    # SCL values as integers.
    values = np.asarray(scl)
    # The band must hold integer classes.
    if not np.issubdtype(values.dtype, np.integer):
        # Explain the requirement.
        raise ValueError("scl must be an integer array")
    # Classes to exclude.
    excluded = np.fromiter(invalid, dtype=np.int64)
    # Unknown classes indicate a wrong band.
    if np.any((excluded < 0) | (excluded > 11)):
        # Explain the valid range.
        raise ValueError("SCL classes lie between 0 and 11")
    # Pixels whose class is not excluded.
    return ~np.isin(values, excluded)


# Extract a bit field of a given length from an integer QA band.
def qa_bits(qa: NDArray[Any], start: int, length: int = 1) -> NDArray[np.int64]:
    # QA values as integers.
    values = np.asarray(qa)
    # The band must hold integers.
    if not np.issubdtype(values.dtype, np.integer):
        # Explain the requirement.
        raise ValueError("qa must be an integer array")
    # Bit ranges must be valid.
    if start < 0 or length < 1 or start + length > 64:
        # Explain the valid range.
        raise ValueError("invalid bit range")
    # Shift the field to the right and keep its bits.
    return (values.astype(np.int64) >> start) & ((1 << length) - 1)


# Flag mask of the Landsat Collection 2 QA_PIXEL band.
def landsat_qa_mask(
    qa: NDArray[Any],  # QA_PIXEL values.
    fill: bool = True,  # Flag fill pixels.
    dilated_cloud: bool = True,  # Flag dilated cloud.
    cirrus: bool = True,  # Flag cirrus.
    cloud: bool = True,  # Flag cloud.
    cloud_shadow: bool = True,  # Flag cloud shadow.
    snow: bool = False,  # Flag snow.
    water: bool = False,  # Flag water.
) -> NDArray[np.bool_]:  # True where any selected flag is set.
    # Selected flags.
    selected = {
        "fill": fill,  # Fill.
        "dilated_cloud": dilated_cloud,  # Dilated cloud.
        "cirrus": cirrus,  # Cirrus.
        "cloud": cloud,  # Cloud.
        "cloud_shadow": cloud_shadow,  # Cloud shadow.
        "snow": snow,  # Snow.
        "water": water,  # Water.
    }  # End of the selection.
    # Bit mask with every selected flag.
    bits = sum(1 << LANDSAT_QA_BITS[name] for name, on in selected.items() if on)
    # QA values as integers, validated by qa_bits.
    values = qa_bits(qa, 0, 16)
    # Pixels with any selected bit set.
    return (values & bits) != 0


# Two-bit confidence of a QA_PIXEL field (0 none, 1 low, 2 medium, 3 high).
def landsat_cloud_confidence(qa: NDArray[Any], field: str = "cloud") -> NDArray[np.int64]:
    # The field must be known.
    if field not in LANDSAT_QA_CONFIDENCE:
        # Explain the accepted names.
        raise ValueError(f"field must be one of {sorted(LANDSAT_QA_CONFIDENCE)}")
    # Extract the two bits.
    return qa_bits(qa, LANDSAT_QA_CONFIDENCE[field], 2)


# Grow a boolean mask by a number of pixels with a round structuring element.
def buffer_mask(mask: NDArray[Any], pixels: int) -> NDArray[np.bool_]:
    # Mask as boolean.
    m = np.asarray(mask, dtype=bool)
    # The buffer must not be negative.
    if pixels < 0:
        # Explain the requirement.
        raise ValueError("pixels must not be negative")
    # A zero buffer changes nothing.
    if pixels == 0:
        # Return a copy.
        return m.copy()
    # Offsets of the disk.
    yy, xx = np.mgrid[-pixels : pixels + 1, -pixels : pixels + 1]
    # Disk of the given radius.
    disk = yy**2 + xx**2 <= pixels**2
    # Dilate the mask.
    return binary_dilation(m, structure=disk)


# Set pixels of an image where the mask is True to NaN.
def apply_mask(image: NDArray[Any], mask: NDArray[Any]) -> NDArray[np.float64]:
    # Float copy of the image.
    out = np.array(image, dtype=np.float64, copy=True)
    # Mask as boolean.
    m = np.asarray(mask, dtype=bool)
    # The mask must match the spatial shape.
    if m.shape != out.shape[-2:]:
        # Explain the requirement.
        raise ValueError(f"mask shape {m.shape} does not match image shape {out.shape[-2:]}")
    # Mask every band.
    out[..., m] = np.nan
    # Return the masked image.
    return out


# =============================================================================
# End of module src/unbihexium/preprocessing/masks.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
