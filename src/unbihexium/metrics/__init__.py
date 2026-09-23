# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/metrics/__init__.py
# Title       : Accuracy assessment toolbox for map products
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Accuracy and quality measures of Earth observation map products:
#
#   classification   error matrix, overall, producer's and user's accuracy,
#                    kappa, F1, IoU, quantity and allocation disagreement
#   area             unbiased area estimates and accuracies with confidence
#                    intervals under stratified sampling (Olofsson et al.,
#                    2014), sample allocation
#   regression       bias, MAE, RMSE, unbiased RMSE, R^2 and correlation
#   image_quality    PSNR, SSIM, SAM, ERGAS and the Q index
#   change           change detection rates and class transition analysis
#
# Streaming metrics used during model training live in
# unbihexium.ai.evaluation; PSNR and SSIM are shared with it.
# Every error matrix has reference classes in rows and map classes in
# columns.
#
# Usage
# -----
#   from unbihexium.metrics import confusion_matrix, stratified_area_estimate
#   counts = confusion_matrix(reference_labels, map_labels, labels=[1, 2, 3])
#   estimate = stratified_area_estimate(counts, mapped_area_ha)
#   print(estimate.area, estimate.area_ci)
# =============================================================================

# Area estimation.
from unbihexium.metrics.area import (
    AreaEstimate,  # Result record.
    estimated_error_matrix,  # Population proportions.
    sample_allocation,  # Stratum sample sizes.
    stratified_area_estimate,  # Olofsson et al. (2014) estimator.
)  # End of the area imports.

# Change detection.
from unbihexium.metrics.change import (
    change_detection_metrics,  # Binary change accuracy.
    change_map,  # Changed pixels.
    transition_matrix,  # From-to counts.
    transition_summary,  # Gain, loss, net change and swap.
)  # End of the change imports.

# Thematic accuracy.
from unbihexium.metrics.classification import (
    AccuracyAssessment,  # Result record.
    accuracy,  # Fraction of equal labels.
    accuracy_assessment,  # Measures of an error matrix.
    cohen_kappa,  # Kappa.
    confusion_matrix,  # Error matrix.
    dice,  # Dice coefficient.
    f1_score,  # Binary F1.
    iou,  # Binary IoU.
    mean_iou,  # Mean IoU over classes.
    precision,  # Binary precision.
    recall,  # Binary recall.
)  # End of the classification imports.

# Image quality.
from unbihexium.metrics.image_quality import (
    calculate_ergas,  # Alias of ergas.
    calculate_qindex,  # Alias of q_index.
    calculate_sam,  # Alias of sam.
    ergas,  # Relative global error.
    psnr,  # Peak signal-to-noise ratio.
    q_index,  # Universal image quality index.
    sam,  # Mean spectral angle.
    spectral_angle,  # Per-pixel spectral angle.
    ssim,  # Structural similarity.
)  # End of the image quality imports.

# Continuous errors.
from unbihexium.metrics.regression import (
    bias,  # Mean error.
    mae,  # Mean absolute error.
    pearson_r,  # Correlation.
    r_squared,  # Coefficient of determination.
    regression_report,  # All statistics.
    rmse,  # Root mean square error.
    ubrmse,  # Unbiased RMSE.
)  # End of the regression imports.

# Public names of the package.
__all__ = [
    "AccuracyAssessment",  # Result record.
    "AreaEstimate",  # Result record.
    "accuracy",  # Fraction of equal labels.
    "accuracy_assessment",  # Measures of an error matrix.
    "bias",  # Mean error.
    "calculate_ergas",  # Alias of ergas.
    "calculate_qindex",  # Alias of q_index.
    "calculate_sam",  # Alias of sam.
    "change_detection_metrics",  # Binary change accuracy.
    "change_map",  # Changed pixels.
    "cohen_kappa",  # Kappa.
    "confusion_matrix",  # Error matrix.
    "dice",  # Dice coefficient.
    "ergas",  # Relative global error.
    "estimated_error_matrix",  # Population proportions.
    "f1_score",  # Binary F1.
    "iou",  # Binary IoU.
    "mae",  # Mean absolute error.
    "mean_iou",  # Mean IoU over classes.
    "pearson_r",  # Correlation.
    "precision",  # Binary precision.
    "psnr",  # Peak signal-to-noise ratio.
    "q_index",  # Universal image quality index.
    "r_squared",  # Coefficient of determination.
    "recall",  # Binary recall.
    "regression_report",  # All statistics.
    "rmse",  # Root mean square error.
    "sam",  # Mean spectral angle.
    "sample_allocation",  # Stratum sample sizes.
    "spectral_angle",  # Per-pixel spectral angle.
    "ssim",  # Structural similarity.
    "stratified_area_estimate",  # Olofsson et al. (2014) estimator.
    "transition_matrix",  # From-to counts.
    "transition_summary",  # Gain, loss, net change and swap.
    "ubrmse",  # Unbiased RMSE.
]  # End of the public names.

# =============================================================================
# End of module src/unbihexium/metrics/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
