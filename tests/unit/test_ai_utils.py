# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_ai_utils.py
# Title       : Tests of decoding, metrics, transforms and result records
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest; no PyTorch needed
# =============================================================================
#
# Abstract
# --------
# Checks the NumPy parts of the AI package against hand-computed values:
# box IoU and non-maximum suppression, CenterNet decoding of an encoded
# target, average precision, confusion matrix statistics, regression errors,
# PSNR and SSIM, normalisation, crops, pads and the symmetries of the square
# (boxes must follow the pixels), and georeferencing of the result records.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Arrays.
import numpy as np

# Test framework.
import pytest

# CenterNet target encoding.
from unbihexium.ai.data import encode_centernet, gaussian_radius

# Decoding functions under test.
from unbihexium.ai.decode import (
    box_iou,  # Pairwise IoU.
    decode_centernet,  # CenterNet decoding.
    labels_from_logits,  # Class maps.
    nms,  # Non-maximum suppression.
    sigmoid,  # Logistic function.
    softmax,  # Softmax.
)

# Metric accumulators under test.
from unbihexium.ai.evaluation import (
    ConfusionMatrix,  # Segmentation metrics.
    DetectionAccumulator,  # Detection metrics.
    ImageQuality,  # PSNR and SSIM.
    RegressionAccumulator,  # Regression errors.
    average_precision,  # AP of a curve.
    psnr,  # Peak signal-to-noise ratio.
    ssim,  # Structural similarity.
)  # End of the metric imports.

# Result records under test.
from unbihexium.ai.results import (
    Detection,  # One box.
    DetectionResult,  # Boxes of an image.
    RegressionResult,  # Values.
    SegmentationResult,  # Class map.
    pixel_to_map,  # Georeferencing.
    scaled_transform,  # Finer grids.
)  # End of the result imports.

# Transforms under test.
from unbihexium.ai.transforms import (
    IGNORE_INDEX,  # Ignored label.
    Augmenter,  # Random augmentation.
    Normalization,  # Band standardisation.
    Sample,  # Sample record.
    crop,  # Window cut.
    dihedral,  # Rotations and mirrors.
    pad,  # Padding.
    random_crop,  # Random window.
)  # End of the transform imports.


# Activation functions are stable and normalised.
def test_activations() -> None:
    # Extreme logits do not overflow.
    assert sigmoid(np.array([-1000.0, 0.0, 1000.0])).tolist() == [0.0, 0.5, 1.0]
    # Softmax sums to one along the axis.
    assert np.allclose(softmax(np.random.default_rng(0).normal(size=(4, 3, 3))).sum(axis=0), 1)
    # Low-confidence pixels get the no-data label.
    labels = labels_from_logits(np.array([[[0.0]], [[0.1]]]), threshold=0.9)
    # Both classes are close to 0.5.
    assert labels[0, 0] == 255


# IoU of identical, disjoint and half-overlapping boxes.
def test_box_iou() -> None:
    # Reference box.
    a = np.array([[0, 0, 10, 10]])
    # Identical, disjoint and half-overlapping boxes.
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]])
    # Expected IoU values.
    assert np.allclose(box_iou(a, b), [[1.0, 0.0, 50 / 150]])


# Non-maximum suppression keeps the best box per cluster and per class.
def test_nms() -> None:
    # Two overlapping boxes and one far away.
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    # Scores of the boxes.
    scores = np.array([0.9, 0.8, 0.7])
    # The weaker overlapping box is dropped.
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]
    # Boxes of different classes never suppress each other.
    assert nms(boxes, scores, 0.5, classes=np.array([0, 1, 0])).tolist() == [0, 1, 2]


# Encoding a box and decoding the perfect output returns the box.
def test_centernet_round_trip() -> None:
    # Two boxes of different classes.
    boxes = np.array([[10, 12, 30, 40], [60, 64, 76, 72]], dtype=np.float32)
    # Their classes.
    labels = np.array([0, 1])
    # Targets on a 128 x 128 image with stride 4.
    t = encode_centernet(boxes, labels, 2, 128, 128, 4)
    # Heat map logits that reproduce the target peaks exactly.
    p = np.clip(t["heatmap"], 1e-6, 1 - 1e-6)
    # Inverse of the logistic function.
    heat = np.log(p / (1 - p))
    # Raw output: logits, size and offset.
    output = np.concatenate([heat, t["size"], t["offset"]])
    # Decode.
    decoded, scores, classes = decode_centernet(output, threshold=0.9)
    # Two boxes, one per class.
    assert sorted(classes.tolist()) == [0, 1]
    # Boxes are recovered exactly.
    assert np.allclose(decoded[np.argsort(classes)], boxes, atol=1e-4)
    # Peaks have score close to one.
    assert scores.min() > 0.99
    # The Gaussian radius grows with the box size.
    assert gaussian_radius(40, 40) > gaussian_radius(10, 10) > 0


# Average precision of simple curves.
def test_average_precision() -> None:
    # Perfect ranking.
    assert average_precision(np.array([0.5, 1.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)
    # Half of the objects found with perfect precision.
    assert average_precision(np.array([0.5]), np.array([1.0])) == pytest.approx(0.5)


# The detection accumulator matches boxes greedily by score.
def test_detection_accumulator() -> None:
    # Two classes.
    acc = DetectionAccumulator(2, ["ship", "boat"])
    # One correct ship, one false ship and one missed boat.
    acc.update(
        np.array([[0, 0, 10, 10], [50, 50, 60, 60]]),  # Predicted boxes.
        np.array([0.9, 0.3]),  # Scores.
        np.array([0, 0]),  # Classes.
        np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),  # Reference boxes.
        np.array([0, 1]),  # Reference classes.
    )  # End of the update.
    # Report.
    report = acc.compute()
    # The ship is found first, so its AP is one.
    assert report["ap50_per_class"]["ship"] == pytest.approx(1.0)
    # The boat is missed.
    assert report["ap50_per_class"]["boat"] == pytest.approx(0.0)
    # Mean over both classes.
    assert report["map50"] == pytest.approx(0.5)


# Confusion matrix statistics against hand-computed values.
def test_confusion_matrix() -> None:
    # Two classes and an ignored label.
    cm = ConfusionMatrix(2, ["land", "water"])
    # References with one ignored pixel.
    ref = np.array([[0, 0, 1, 1, IGNORE_INDEX]])
    # Predictions with one error.
    pred = np.array([[0, 1, 1, 1, 0]])
    # Accumulate.
    cm.update(pred, ref)
    # Report.
    report = cm.compute()
    # Three of four evaluated pixels are right.
    assert report["accuracy"] == pytest.approx(0.75)
    # IoU of land: 1 / 2; IoU of water: 2 / 3.
    assert report["iou_per_class"] == pytest.approx({"land": 0.5, "water": 2 / 3})
    # Four pixels were evaluated.
    assert report["pixels"] == 4


# Regression errors ignore NaN references.
def test_regression_accumulator() -> None:
    # One output.
    acc = RegressionAccumulator(["height"])
    # Predictions (N=1, K=1, 1, 3).
    pred = np.array([[[[1.0, 2.0, 3.0]]]])
    # References with one missing value.
    ref = np.array([[[[1.0, 4.0, np.nan]]]])
    # Accumulate.
    acc.update(pred, ref)
    # Report.
    report = acc.compute()
    # Errors 0 and -2.
    assert report["mae"] == pytest.approx(1.0)
    # RMSE of (0, 2).
    assert report["rmse"] == pytest.approx(np.sqrt(2.0))
    # Mean error.
    assert report["bias"] == pytest.approx(-1.0)
    # Two valid values.
    assert report["per_output"]["height"]["count"] == 2


# PSNR and SSIM of identical and noisy images.
def test_image_quality() -> None:
    # Smooth test image.
    img = np.linspace(0, 1, 32 * 32).reshape(1, 32, 32)
    # Identical images.
    assert psnr(img, img) == float("inf")
    # SSIM of identical images is one.
    assert ssim(img, img) == pytest.approx(1.0)
    # Noisy copy.
    noisy = img + np.random.default_rng(0).normal(0, 0.05, img.shape)
    # PSNR of Gaussian noise with sigma 0.05 is about 26 dB.
    assert 24 < psnr(noisy, img) < 28
    # Streaming accumulator.
    quality = ImageQuality()
    # Add the pair.
    quality.update(noisy, img)
    # SSIM below one.
    assert quality.compute()["ssim"] < 1.0


# Normalisation standardises every band and ignores NaN.
def test_normalization() -> None:
    # Two bands with different scales.
    rng = np.random.default_rng(0)
    # Image with a missing pixel.
    image = np.stack([rng.normal(5, 2, (50, 50)), rng.normal(-1, 0.1, (50, 50))]).astype(np.float32)
    # Missing value.
    image[0, 0, 0] = np.nan
    # Estimate the statistics.
    norm = Normalization.fit([image])
    # Apply them.
    out = norm(image)
    # Zero mean and unit variance per band.
    assert np.allclose(np.nanmean(out.reshape(2, -1), axis=1), 0, atol=1e-3)
    # Missing values become zero.
    assert out[0, 0, 0] == 0
    # Round trip through a dictionary.
    assert np.allclose(Normalization.from_dict(norm.to_dict()).mean, norm.mean)
    # Band count mismatches are rejected.
    with pytest.raises(ValueError):
        # Three bands instead of two.
        norm(np.zeros((3, 4, 4)))


# Crops and pads keep image and targets aligned.
def test_crop_and_pad() -> None:
    # Image whose pixel values encode their column.
    image = np.tile(np.arange(10, dtype=np.float32), (1, 10, 1))
    # Sample with a mask and a box.
    sample = Sample(
        image=image,  # Image.
        mask=np.tile(np.arange(10), (10, 1)),  # Mask equal to the column.
        boxes=np.array([[2, 2, 6, 6]], dtype=np.float32),  # Box.
        labels=np.array([0]),  # Class.
    )  # End of the sample.
    # Cut the window starting at column 2.
    window = crop(sample, 1, 2, 5, 5)
    # Image and mask start at column 2.
    assert window.image[0, 0, 0] == 2 and window.mask[0, 0] == 2
    # The box moves with the window.
    assert window.boxes.tolist() == [[0, 1, 4, 5]]
    # Padding adds ignored labels.
    padded = pad(window, 8, 8)
    # New size.
    assert padded.image.shape == (1, 8, 8) and padded.mask[7, 7] == IGNORE_INDEX
    # A random crop has the requested size.
    assert random_crop(sample, 4, np.random.default_rng(0)).image.shape == (1, 4, 4)


# Rotations and mirrors move boxes with the pixels.
@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("flip", [False, True])
def test_dihedral_boxes(k: int, flip: bool) -> None:
    # Non-square image with one bright rectangle.
    image = np.zeros((1, 20, 30), dtype=np.float32)
    # Rectangle at rows 3..8, columns 5..17.
    image[0, 3:8, 5:17] = 1.0
    # Sample with the rectangle as a box.
    sample = Sample(image=image, boxes=np.array([[5, 3, 17, 8]], np.float32), labels=np.array([0]))
    # Transform.
    out = dihedral(sample, k, flip)
    # Bright pixels after the transform.
    rows, cols = np.nonzero(out.image[0])
    # Bounding box of the bright pixels.
    expected = [cols.min(), rows.min(), cols.max() + 1, rows.max() + 1]
    # The transformed box matches the pixels.
    assert out.boxes[0].tolist() == expected


# Photometric augmentation changes only the image.
def test_augmenter() -> None:
    # Sample with a mask.
    sample = Sample(image=np.ones((2, 8, 8), np.float32), mask=np.zeros((8, 8), np.int64))
    # Photometric only.
    out = Augmenter(geometric=False, photometric=True)(sample, np.random.default_rng(0))
    # The image changed.
    assert not np.allclose(out.image, sample.image)
    # The mask did not.
    assert np.array_equal(out.mask, sample.mask)


# Result records convert between pixel and map coordinates.
def test_results_georeferencing() -> None:
    # 10 m UTM grid.
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 7000000.0)
    # Pixel (2, 3) maps to the corner of that cell.
    assert pixel_to_map(transform, 2, 3) == (500020.0, 6999970.0)
    # A four times finer grid.
    assert scaled_transform(transform, 4)[0] == 2.5
    # Detection result with one box.
    ship = Detection((0, 0, 1, 1), 0.9, 0, "ship", (0, 0, 10, 10))
    # Result in UTM coordinates.
    result = DetectionResult([ship], crs="EPSG:32635")
    # GeoJSON uses the map box.
    feature = result.to_geojson()["features"][0]
    # Corner of the polygon.
    assert feature["geometry"]["coordinates"][0][2] == [10, 10]
    # Counts per class.
    assert result.counts_by_class() == {"ship": 1}
    # Segmentation areas use the pixel size.
    seg = SegmentationResult(np.array([[0, 1], [1, 1]]), ["land", "water"], transform=transform)
    # Three water pixels of 100 square metres.
    assert seg.class_areas()["water"] == pytest.approx(300.0)
    # Scene regression results summarise to a dictionary.
    reg = RegressionResult(np.array([4.2], np.float32), ["yield"], ["t ha-1"])
    # Values by name.
    assert reg.to_dict()["values"] == {"yield": pytest.approx(4.2)}


# =============================================================================
# End of module tests/unit/test_ai_utils.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
