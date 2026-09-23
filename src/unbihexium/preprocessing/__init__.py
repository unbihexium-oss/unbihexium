# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/preprocessing/__init__.py
# Title       : Radiometric and geometric preparation of imagery
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy, SciPy and scikit-image
# =============================================================================
#
# Abstract
# --------
# Preparation of optical imagery before analysis or model inference:
#
#   radiometry    DN to radiance and reflectance (Sentinel-2, Landsat 8/9),
#                 brightness temperature, Earth-Sun distance, DOS1
#   masks         Sentinel-2 SCL and Landsat QA_PIXEL cloud masks, buffers
#   enhancement   linear and percentile stretches, gamma, histogram
#                 equalisation and matching
#   pansharpen    Brovey, IHS and Gram-Schmidt adaptive fusion
#   resample      interpolation and block aggregation with missing values
#   transforms    Normalize, Resize, Pad, Compose and layout conversion
#
# Arrays are (H, W) or band-first (C, H, W); missing values are NaN or an
# explicit nodata value.
#
# Usage
# -----
#   from unbihexium.preprocessing import sentinel2_reflectance, scl_valid_mask
#   rho = sentinel2_reflectance(dn, processing_baseline="05.10")
#   rho[:, ~scl_valid_mask(scl)] = float("nan")
# =============================================================================

# Contrast stretches and histogram operations.
from unbihexium.preprocessing.enhancement import (
    gamma_correction,  # Power-law adjustment.
    histogram_equalize,  # Equalisation through the CDF.
    histogram_match,  # Matching to a reference distribution.
    linear_stretch,  # Stretch between fixed bounds.
    percentile_bounds,  # Percentile values per band.
    percentile_stretch,  # Stretch between percentiles.
)  # End of the enhancement imports.

# Quality masks.
from unbihexium.preprocessing.masks import (
    LANDSAT_QA_BITS,  # QA_PIXEL flag bits.
    LANDSAT_QA_CONFIDENCE,  # QA_PIXEL confidence fields.
    SCL_CLASSES,  # Sentinel-2 SCL legend.
    SCL_DEFAULT_INVALID,  # SCL classes masked by default.
    apply_mask,  # Set masked pixels to NaN.
    buffer_mask,  # Grow a mask.
    landsat_cloud_confidence,  # Two-bit confidence of QA_PIXEL.
    landsat_qa_mask,  # QA_PIXEL flag mask.
    qa_bits,  # Bit field extraction.
    scl_valid_mask,  # Usable pixels of the SCL band.
)  # End of the mask imports.

# Pansharpening.
from unbihexium.preprocessing.pansharpen import (
    brovey,  # Ratio transform.
    gram_schmidt,  # Gram-Schmidt adaptive.
    ihs,  # Fast IHS.
    upsample_to_pan,  # Bands onto the pan grid.
)  # End of the pansharpening imports.

# Radiometric calibration.
from unbihexium.preprocessing.radiometry import (
    LANDSAT_C2L2_SR_OFFSET,  # Level-2 reflectance offset.
    LANDSAT_C2L2_SR_SCALE,  # Level-2 reflectance scale.
    LANDSAT_C2L2_ST_OFFSET,  # Level-2 temperature offset.
    LANDSAT_C2L2_ST_SCALE,  # Level-2 temperature scale.
    S2_OFFSET_PB04,  # Sentinel-2 offset from baseline 04.00.
    S2_QUANTIFICATION,  # Sentinel-2 quantification value.
    dark_object_subtraction,  # DOS1 haze removal.
    earth_sun_distance,  # Earth-Sun distance.
    landsat_brightness_temperature,  # Thermal brightness temperature.
    landsat_c2l2_reflectance,  # Level-2 surface reflectance.
    landsat_c2l2_temperature,  # Level-2 surface temperature.
    landsat_radiance,  # Level-1 radiance.
    landsat_toa_reflectance,  # Level-1 TOA reflectance.
    landsat_toa_reflectance_from_mtl,  # TOA reflectance from MTL metadata.
    parse_landsat_mtl,  # MTL parser.
    radiance_to_reflectance,  # Radiance to reflectance with ESUN.
    sentinel2_reflectance,  # Sentinel-2 DN to reflectance.
)  # End of the radiometry imports.

# Resampling.
from unbihexium.preprocessing.resample import (
    AGGREGATIONS,  # Aggregation methods.
    RESAMPLING_ORDERS,  # Interpolation methods.
    aggregate,  # Block aggregation.
    resample,  # Interpolation.
    scaled_transform,  # Geotransform of a resampled grid.
)  # End of the resampling imports.

# Array transforms.
from unbihexium.preprocessing.transforms import (
    Compose,  # Chain of transforms.
    Normalize,  # Standardisation or min-max scaling.
    Pad,  # Padding.
    Resize,  # Resizing.
    from_tensor,  # (C, H, W) to (H, W, C).
    minmax_normalize,  # Per-band min-max scaling.
    standardize,  # Per-band z-scores.
    to_tensor,  # (H, W, C) to (C, H, W).
)  # End of the transform imports.

# Public names of the package.
__all__ = [
    "AGGREGATIONS",  # Aggregation methods.
    "LANDSAT_C2L2_SR_OFFSET",  # Level-2 reflectance offset.
    "LANDSAT_C2L2_SR_SCALE",  # Level-2 reflectance scale.
    "LANDSAT_C2L2_ST_OFFSET",  # Level-2 temperature offset.
    "LANDSAT_C2L2_ST_SCALE",  # Level-2 temperature scale.
    "LANDSAT_QA_BITS",  # QA_PIXEL flag bits.
    "LANDSAT_QA_CONFIDENCE",  # QA_PIXEL confidence fields.
    "RESAMPLING_ORDERS",  # Interpolation methods.
    "S2_OFFSET_PB04",  # Sentinel-2 offset from baseline 04.00.
    "S2_QUANTIFICATION",  # Sentinel-2 quantification value.
    "SCL_CLASSES",  # Sentinel-2 SCL legend.
    "SCL_DEFAULT_INVALID",  # SCL classes masked by default.
    "Compose",  # Chain of transforms.
    "Normalize",  # Standardisation or min-max scaling.
    "Pad",  # Padding.
    "Resize",  # Resizing.
    "aggregate",  # Block aggregation.
    "apply_mask",  # Set masked pixels to NaN.
    "brovey",  # Ratio transform.
    "buffer_mask",  # Grow a mask.
    "dark_object_subtraction",  # DOS1 haze removal.
    "earth_sun_distance",  # Earth-Sun distance.
    "from_tensor",  # (C, H, W) to (H, W, C).
    "gamma_correction",  # Power-law adjustment.
    "gram_schmidt",  # Gram-Schmidt adaptive.
    "histogram_equalize",  # Equalisation through the CDF.
    "histogram_match",  # Matching to a reference distribution.
    "ihs",  # Fast IHS.
    "landsat_brightness_temperature",  # Thermal brightness temperature.
    "landsat_c2l2_reflectance",  # Level-2 surface reflectance.
    "landsat_c2l2_temperature",  # Level-2 surface temperature.
    "landsat_cloud_confidence",  # Two-bit confidence of QA_PIXEL.
    "landsat_qa_mask",  # QA_PIXEL flag mask.
    "landsat_radiance",  # Level-1 radiance.
    "landsat_toa_reflectance",  # Level-1 TOA reflectance.
    "landsat_toa_reflectance_from_mtl",  # TOA reflectance from MTL metadata.
    "linear_stretch",  # Stretch between fixed bounds.
    "minmax_normalize",  # Per-band min-max scaling.
    "parse_landsat_mtl",  # MTL parser.
    "percentile_bounds",  # Percentile values per band.
    "percentile_stretch",  # Stretch between percentiles.
    "qa_bits",  # Bit field extraction.
    "radiance_to_reflectance",  # Radiance to reflectance with ESUN.
    "resample",  # Interpolation.
    "scaled_transform",  # Geotransform of a resampled grid.
    "scl_valid_mask",  # Usable pixels of the SCL band.
    "sentinel2_reflectance",  # Sentinel-2 DN to reflectance.
    "standardize",  # Per-band z-scores.
    "to_tensor",  # (H, W, C) to (C, H, W).
    "upsample_to_pan",  # Bands onto the pan grid.
]  # End of the public names.

# =============================================================================
# End of module src/unbihexium/preprocessing/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
