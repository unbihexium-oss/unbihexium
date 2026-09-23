# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/models/spectral.py
# Title       : Spectral index formulas as network modules
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch (unbihexium[torch])
# =============================================================================
#
# Abstract
# --------
# Exact implementations of published spectral indices as PyTorch modules
# without trainable parameters. Wrapping the formulas as modules lets them
# run in the same tiled inference pipeline as the learned models and export
# to ONNX. The input channels must be surface reflectance in the order of
# the bands listed for the index in the model zoo catalogue.
#
# Formulas (input channel order in brackets)
# ------------------------------------------
#   ndvi [red, nir]         (NIR - RED) / (NIR + RED)            Rouse et al. 1974
#   ndwi [green, nir]       (GREEN - NIR) / (GREEN + NIR)        McFeeters 1996
#   evi  [blue, red, nir]   2.5 (NIR - RED) /
#                           (NIR + 6 RED - 7.5 BLUE + 1)         Huete et al. 2002
#   savi [red, nir]         1.5 (NIR - RED) / (NIR + RED + 0.5)  Huete 1988
#   msi  [nir, swir1]       SWIR1 / NIR                          Rock et al. 1986
#   nbr  [nir, swir2]       (NIR - SWIR2) / (NIR + SWIR2)        Key and Benson 2006
#   vci  [ndvi, min, max]   (NDVI - MIN) / (MAX - MIN)           Kogan 1995
#
# Pixels where the denominator is zero are set to NaN, which marks them as
# undefined in every downstream statistic that ignores NaN.
#
# References
# ----------
#   Rouse, J. W., Haas, R. H., Schell, J. A., Deering, D. W. (1974).
#     Monitoring vegetation systems in the Great Plains with ERTS. Third ERTS
#     Symposium, NASA SP-351, 309-317.
#   McFeeters, S. K. (1996). The use of the normalized difference water index
#     (NDWI) in the delineation of open water features. International Journal
#     of Remote Sensing, 17(7), 1425-1432.
#   Huete, A., Didan, K., Miura, T., Rodriguez, E. P., Gao, X., Ferreira, L. G.
#     (2002). Overview of the radiometric and biophysical performance of the
#     MODIS vegetation indices. Remote Sensing of Environment, 83, 195-213.
#   Huete, A. R. (1988). A soil-adjusted vegetation index (SAVI). Remote
#     Sensing of Environment, 25(3), 295-309.
#   Rock, B. N., Vogelmann, J. E., Williams, D. L., Vogelmann, A. F.,
#     Hoshizaki, T. (1986). Remote detection of forest damage. BioScience,
#     36(7), 439-445.
#   Key, C. H., Benson, N. C. (2006). Landscape assessment: ground measure of
#     severity, the Composite Burn Index; and remote sensing of severity, the
#     Normalized Burn Ratio. USDA Forest Service, RMRS-GTR-164-CD.
#   Kogan, F. N. (1995). Application of vegetation index and brightness
#     temperature for drought detection. Advances in Space Research, 15(11),
#     91-100.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Tensor type and core functions.
import torch

# Module base class.
from torch import nn

# Number of input channels expected by each formula.
FORMULA_CHANNELS: dict[str, int] = {
    "ndvi": 2,  # red, nir
    "ndwi": 2,  # green, nir
    "evi": 3,  # blue, red, nir
    "savi": 2,  # red, nir
    "msi": 2,  # nir, swir1
    "nbr": 2,  # nir, swir2
    "vci": 3,  # ndvi, ndvi_min, ndvi_max
}  # End of the formula table.


# Divide two tensors, returning NaN where the denominator is zero.
def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    # Mask of pixels with a defined ratio.
    valid = denominator != 0
    # Replace zero denominators by one so that the division never fails.
    safe = torch.where(valid, denominator, torch.ones_like(denominator))
    # Keep the ratio where defined and NaN elsewhere.
    return torch.where(valid, numerator / safe, torch.full_like(numerator, float("nan")))


# Spectral index as a parameter-free network module.
class SpectralIndex(nn.Module):
    # Create the module for a formula identifier.
    def __init__(self, formula: str) -> None:
        # Initialise the module base class.
        super().__init__()
        # Unknown formulas are rejected with the list of supported ones.
        if formula not in FORMULA_CHANNELS:
            # Report the supported formulas.
            raise ValueError(f"unknown formula {formula!r}; supported: {sorted(FORMULA_CHANNELS)}")
        # Formula identifier.
        self.formula = formula
        # Number of input channels the formula expects.
        self.in_channels = FORMULA_CHANNELS[formula]

    # Compute the index; input (N, C, H, W), output (N, 1, H, W).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The channel count must match the formula.
        if x.shape[1] != self.in_channels:
            # Report the expected channel count.
            message = f"{self.formula} expects {self.in_channels} channels, got {x.shape[1]}"
            # Raise the error.
            raise ValueError(message)
        # Split the channels into single-channel tensors.
        c = [x[:, i : i + 1] for i in range(self.in_channels)]
        # Normalized Difference Vegetation Index.
        if self.formula == "ndvi":
            # c = [red, nir].
            return safe_divide(c[1] - c[0], c[1] + c[0])
        # Normalized Difference Water Index (McFeeters).
        if self.formula == "ndwi":
            # c = [green, nir].
            return safe_divide(c[0] - c[1], c[0] + c[1])
        # Enhanced Vegetation Index.
        if self.formula == "evi":
            # c = [blue, red, nir].
            return safe_divide(2.5 * (c[2] - c[1]), c[2] + 6.0 * c[1] - 7.5 * c[0] + 1.0)
        # Soil Adjusted Vegetation Index with L = 0.5.
        if self.formula == "savi":
            # c = [red, nir].
            return safe_divide(1.5 * (c[1] - c[0]), c[1] + c[0] + 0.5)
        # Moisture Stress Index.
        if self.formula == "msi":
            # c = [nir, swir1].
            return safe_divide(c[1], c[0])
        # Normalized Burn Ratio.
        if self.formula == "nbr":
            # c = [nir, swir2].
            return safe_divide(c[0] - c[1], c[0] + c[1])
        # Vegetation Condition Index; the remaining formula.
        return safe_divide(c[0] - c[1], c[2] - c[1])

    # Human-readable description used by print(model).
    def extra_repr(self) -> str:
        # Show the formula.
        return f"formula={self.formula}"


# =============================================================================
# End of module src/unbihexium/ai/models/spectral.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
