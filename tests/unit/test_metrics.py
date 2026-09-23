# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_metrics.py
# Title       : Tests of the accuracy assessment toolbox
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest
# =============================================================================
#
# Abstract
# --------
# Checks the metrics package against hand-computed values and the worked
# example of Olofsson et al. (2014, section 5): 10 million 30 m pixels
# mapped as deforestation, forest gain, stable forest and stable
# non-forest, assessed with a stratified sample of 640 units. The published
# estimates are 21,158 +/- 6,158 ha of deforestation, 11,686 +/- 3,756 ha of
# gain, 285,770 +/- 15,510 ha of stable forest and 581,386 +/- 16,282 ha of
# stable non-forest, with user's accuracy of deforestation 0.88 +/- 0.07,
# producer's accuracy 0.75 +/- 0.21 and overall accuracy 0.95 +/- 0.02.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Arrays.
import numpy as np

# Test framework.
import pytest

# Functions under test.
from unbihexium.metrics import (
    accuracy,  # Fraction of equal labels.
    accuracy_assessment,  # Error matrix measures.
    bias,  # Mean error.
    calculate_ergas,  # Documentation alias.
    calculate_qindex,  # Documentation alias.
    calculate_sam,  # Documentation alias.
    change_detection_metrics,  # Change accuracy.
    change_map,  # Changed pixels.
    cohen_kappa,  # Kappa.
    confusion_matrix,  # Error matrix.
    dice,  # Dice.
    ergas,  # ERGAS.
    estimated_error_matrix,  # Population proportions.
    f1_score,  # Binary F1.
    iou,  # Binary IoU.
    mae,  # Mean absolute error.
    mean_iou,  # Mean IoU.
    pearson_r,  # Correlation.
    precision,  # Precision.
    psnr,  # PSNR.
    q_index,  # Q index.
    r_squared,  # R squared.
    recall,  # Recall.
    regression_report,  # All regression statistics.
    rmse,  # RMSE.
    sam,  # Spectral angle.
    sample_allocation,  # Sample sizes.
    ssim,  # SSIM.
    stratified_area_estimate,  # Area estimation.
    transition_matrix,  # From-to counts.
    transition_summary,  # Gain and loss.
    ubrmse,  # Unbiased RMSE.
)  # End of the imports under test.

# Sample counts of Olofsson et al. (2014), map classes in rows.
OLOFSSON_COUNTS = np.array(
    [  # Reference classes in the same order as the rows.
        [66, 0, 5, 4],  # Mapped deforestation.
        [0, 55, 8, 12],  # Mapped forest gain.
        [1, 0, 153, 11],  # Mapped stable forest.
        [2, 1, 9, 313],  # Mapped stable non-forest.
    ]
)  # End of the sample counts.

# Mapped pixels of every class.
OLOFSSON_PIXELS = np.array([200_000, 150_000, 3_200_000, 6_450_000])


# Error matrix convention: rows reference, columns map.
def test_confusion_matrix_convention() -> None:
    # Reference and map labels with an ignored value.
    reference = np.array([1, 1, 2, 2, 3, 3, 0])
    # Map labels.
    predicted = np.array([1, 2, 2, 2, 3, 1, 3])
    # Class order 3, 2, 1 and ignore 0.
    m = confusion_matrix(reference, predicted, labels=[3, 2, 1], ignore=0)
    # Hand-counted matrix in the requested order.
    np.testing.assert_array_equal(m, [[1, 0, 1], [0, 2, 0], [0, 1, 1]])
    # Weights replace counts.
    w = confusion_matrix(np.array([0, 1]), np.array([0, 0]), weights=np.array([2.5, 0.5]))
    # Weighted matrix.
    np.testing.assert_allclose(w, [[2.5, 0.0], [0.5, 0.0]])


# Accuracy measures of a small matrix.
def test_accuracy_assessment_values() -> None:
    # Reference in rows: 50 of class A, 50 of class B.
    m = np.array([[45, 5], [15, 35]])
    # Assessment.
    a = accuracy_assessment(m, classes=["A", "B"])
    # (45 + 35) / 100.
    assert a.overall_accuracy == pytest.approx(0.8)
    # Producer's accuracy: 45/50 and 35/50.
    np.testing.assert_allclose(a.producers_accuracy, [0.9, 0.7])
    # User's accuracy: 45/60 and 35/40.
    np.testing.assert_allclose(a.users_accuracy, [0.75, 0.875])
    # Chance agreement 0.5 * 0.6 + 0.5 * 0.4 = 0.5, kappa (0.8 - 0.5) / 0.5.
    assert a.kappa == pytest.approx(0.6)
    # Quantity disagreement |0.5 - 0.6| + |0.5 - 0.4|, halved.
    assert a.quantity_disagreement == pytest.approx(0.1)
    # Quantity plus allocation equals total disagreement.
    assert a.quantity_disagreement + a.allocation_disagreement == pytest.approx(0.2)
    # IoU of A: 45 / (50 + 60 - 45).
    assert a.iou[0] == pytest.approx(45 / 65)
    # Report keyed by class names.
    assert a.to_dict()["commission_error"]["A"] == pytest.approx(0.25)
    # Kappa of perfect agreement.
    assert cohen_kappa(np.diag([3, 7])) == pytest.approx(1.0)


# Olofsson et al. (2014) worked example reproduces the published estimates.
def test_olofsson_area_estimates() -> None:
    # Pixel area of 30 m pixels in hectares.
    area_ha = OLOFSSON_PIXELS * 0.09
    # Counts in the package convention: rows reference.
    estimate = stratified_area_estimate(OLOFSSON_COUNTS.T, area_ha)
    # Published areas in hectares.
    np.testing.assert_allclose(estimate.area, [21_158, 11_686, 285_770, 581_386], atol=1.0)
    # Published 95 % interval half-widths.
    np.testing.assert_allclose(estimate.area_ci, [6_158, 3_756, 15_510, 16_282], atol=1.0)
    # User's accuracy of deforestation 66 / 75 with interval 0.07.
    assert estimate.users_accuracy[0] == pytest.approx(0.88)
    # Its interval.
    assert estimate.z * estimate.users_accuracy_se[0] == pytest.approx(0.07, abs=0.005)
    # Producer's accuracy of deforestation with interval 0.21.
    assert estimate.producers_accuracy[0] == pytest.approx(0.75, abs=0.005)
    # Its interval.
    assert estimate.z * estimate.producers_accuracy_se[0] == pytest.approx(0.21, abs=0.005)
    # Overall accuracy 0.95 with interval 0.02.
    assert estimate.overall_accuracy == pytest.approx(0.95, abs=0.005)
    # Its interval.
    assert estimate.z * estimate.overall_accuracy_se == pytest.approx(0.02, abs=0.005)
    # The estimated areas add up to the mapped total.
    assert estimate.area.sum() == pytest.approx(area_ha.sum())
    # The area-weighted matrix gives the same overall accuracy.
    weighted = accuracy_assessment(estimated_error_matrix(OLOFSSON_COUNTS.T, area_ha))
    # Same value.
    assert weighted.overall_accuracy == pytest.approx(estimate.overall_accuracy)
    # Dictionary output.
    assert estimate.to_dict()["classes"]["0"]["area"] == pytest.approx(estimate.area[0])


# Sample allocation for a target standard error.
def test_sample_allocation() -> None:
    # Two strata of equal area with anticipated user's accuracies 0.9 and 0.9.
    n = sample_allocation([1.0, 1.0], [0.9, 0.9], target_se=0.03, rare_minimum=0)
    # n = (0.3 / 0.03)^2 = 100, split evenly.
    np.testing.assert_array_equal(n, [50, 50])
    # The minimum raises rare strata.
    assert sample_allocation([99.0, 1.0], [0.9, 0.9], 0.03, rare_minimum=50)[1] == 50


# Binary mask measures.
def test_binary_measures() -> None:
    # Prediction.
    pred = np.array([1, 1, 0, 0, 1])
    # Target.
    target = np.array([1, 0, 0, 1, 1])
    # TP 2, FP 1, FN 1: precision and recall 2/3.
    assert precision(pred, target) == pytest.approx(2 / 3)
    # Recall.
    assert recall(pred, target) == pytest.approx(2 / 3)
    # F1 = 2 * 2 / (3 + 3).
    assert f1_score(pred, target) == pytest.approx(2 / 3)
    # IoU = 2 / 4.
    assert iou(pred, target, smooth=0.0) == pytest.approx(0.5)
    # Dice equals F1.
    assert dice(pred, target, smooth=0.0) == pytest.approx(2 / 3)
    # Three of five labels agree.
    assert accuracy(pred, target) == pytest.approx(0.6)
    # Mean IoU over classes 0 and 1: 1/3 and 2/4.
    assert mean_iou(pred, target, 2) == pytest.approx((1 / 3 + 0.5) / 2)


# Regression statistics with NaN.
def test_regression_statistics() -> None:
    # Estimates with a missing value.
    est = np.array([2.0, 4.0, np.nan, 6.0])
    # References.
    ref = np.array([1.0, 3.0, 9.0, 8.0])
    # Errors 1, 1, -2.
    assert bias(est, ref) == pytest.approx(0.0)
    # Mean absolute error 4 / 3.
    assert mae(est, ref) == pytest.approx(4 / 3)
    # RMSE sqrt(6 / 3).
    assert rmse(est, ref) == pytest.approx(np.sqrt(2.0))
    # Zero bias makes ubRMSE equal to RMSE.
    assert ubrmse(est, ref) == pytest.approx(np.sqrt(2.0))
    # SS_res 6; SS_tot of [1, 3, 8] around 4 is 9 + 1 + 16 = 26.
    assert r_squared(est, ref) == pytest.approx(1 - 6 / 26)
    # A constant shift has correlation 1 and slope 1.
    report = regression_report(np.array([2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0]))
    # Correlation, slope, intercept, bias and ubRMSE.
    assert report["r"] == pytest.approx(1.0) and report["slope"] == pytest.approx(1.0)
    # Intercept and bias are the shift; ubRMSE removes it.
    assert report["intercept"] == pytest.approx(1.0) and report["ubrmse"] == pytest.approx(0.0)
    # Pearson correlation of anti-correlated values.
    assert pearson_r(np.array([1.0, 2.0]), np.array([2.0, 1.0])) == pytest.approx(-1.0)


# Image quality measures.
def test_image_quality() -> None:
    # Reference with two constant bands.
    ref = np.stack([np.full((4, 4), 2.0), np.full((4, 4), 4.0)])
    # Estimate scaled by 1.1.
    est = ref * 1.1
    # Scaling does not change the spectral angle.
    assert sam(ref, est) == pytest.approx(0.0, abs=1e-6)
    # Orthogonal spectra are 90 degrees apart.
    assert calculate_sam(np.array([[[1.0]], [[0.0]]]), np.array([[[0.0]], [[1.0]]])) == 90.0
    # Relative RMSE 0.1 in both bands; ERGAS = 100 / 4 * 0.1.
    assert ergas(ref, est, ratio=4) == pytest.approx(2.5)
    # Alias.
    assert calculate_ergas(ref, est, ratio=2) == pytest.approx(5.0)
    # Random image.
    x = np.random.default_rng(1).uniform(size=(2, 16, 16))
    # Q of identical images is 1.
    assert q_index(x, x) == pytest.approx(1.0)
    # A shift c keeps correlation and contrast; global Q is 2 m (m + c) / (m^2 + (m + c)^2).
    band = x[0]
    # Mean of the band.
    m = band.mean()
    # Expected luminance term.
    expected = 2 * m * (m + 1.0) / (m**2 + (m + 1.0) ** 2)
    # Global Q.
    assert calculate_qindex(band, band + 1.0, block_size=None) == pytest.approx(expected)
    # PSNR of a uniform error of 0.1 with range 1: 20 dB.
    assert psnr(np.full((4, 4), 0.1), np.zeros((4, 4))) == pytest.approx(20.0)
    # SSIM of identical images is 1.
    assert ssim(x, x) == pytest.approx(1.0)


# Change detection and transitions.
def test_change_metrics() -> None:
    # Reference change.
    ref = np.array([1, 1, 0, 0, 0, 0])
    # Predicted change.
    pred = np.array([1, 0, 1, 0, 0, 0])
    # Counts and rates.
    m = change_detection_metrics(ref, pred)
    # TP 1, FP 1, FN 1, TN 3.
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 3)
    # Detection rate 1/2 and false alarm rate 1/4.
    assert m["detection_rate"] == 0.5 and m["false_alarm_rate"] == 0.25
    # Kappa: p_o = 4/6, p_e = (4*4 + 2*2) / 36, kappa = (2/3 - 5/9) / (4/9).
    assert m["kappa"] == pytest.approx(0.25)
    # Two class maps.
    before = np.array([[1, 1], [2, 0]])
    # Second date.
    after = np.array([[1, 2], [2, 3]])
    # One real change; the nodata pixel does not count.
    changed = change_map(before, after, nodata=0)
    # Only the pixel from 1 to 2 changed.
    np.testing.assert_array_equal(changed, [[False, True], [False, False]])
    # From-to counts of classes 1 and 2.
    t = transition_matrix(before, after, labels=[1, 2], nodata=0)
    # One persistent 1, one 1 to 2, one persistent 2.
    np.testing.assert_array_equal(t, [[1, 1], [0, 1]])
    # Class 2 gains one pixel and loses none.
    summary = transition_summary(t, classes=["one", "two"])
    # Gain, loss and net change.
    assert summary["two"]["gain"] == 1 and summary["two"]["net_change"] == 1
    # Swap is zero without simultaneous gain and loss.
    assert summary["one"]["swap"] == 0 and summary["one"]["loss"] == 1


# =============================================================================
# End of module tests/unit/test_metrics.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
