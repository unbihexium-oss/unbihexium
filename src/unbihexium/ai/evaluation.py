# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/evaluation.py
# Title       : Accuracy metrics for detection, segmentation and regression
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Streaming accumulators for the accuracy measures reported by training,
# the evaluate command and the model cards:
#
#   DetectionAccumulator   average precision per class and mean average
#                          precision at IoU 0.5 and averaged over IoU 0.5 to
#                          0.95 (COCO style), with precision and recall
#   ConfusionMatrix        overall accuracy, per-class IoU, F1, precision and
#                          recall, mean IoU and Cohen's kappa
#   RegressionAccumulator  MAE, RMSE, bias and R^2 per output, ignoring NaN
#   ImageQuality           PSNR and SSIM for enhancement and super-resolution
#
# Every accumulator is updated batch by batch and summarised with compute(),
# so that large validation sets never need to fit in memory.
#
# References
# ----------
#   Everingham, M., et al. (2010). The PASCAL visual object classes (VOC)
#     challenge. International Journal of Computer Vision 88, 303-338.
#   Lin, T.-Y., et al. (2014). Microsoft COCO: common objects in context.
#     ECCV.
#   Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. (2004). Image
#     quality assessment: from error visibility to structural similarity.
#     IEEE Transactions on Image Processing 13(4), 600-612.
#   Cohen, J. (1960). A coefficient of agreement for nominal scales.
#     Educational and Psychological Measurement 20(1), 37-46.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Gaussian filter for SSIM.
from scipy.ndimage import gaussian_filter

# Pairwise IoU of boxes and decoding of raw outputs.
from unbihexium.ai.decode import box_iou, decode_centernet

# Task of a model.
from unbihexium.zoo.catalog import Task

# Configuration of a model.
from unbihexium.zoo.config import BuildConfig

# IoU thresholds of the COCO average precision.
COCO_IOU_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2).tolist())


# Average precision from a precision-recall curve, all-point interpolation.
def average_precision(recall: NDArray[Any], precision: NDArray[Any]) -> float:
    # Add sentinel values at both ends.
    r = np.concatenate([[0.0], recall, [1.0]])
    # Precision sentinels.
    p = np.concatenate([[0.0], precision, [0.0]])
    # Make precision monotonically decreasing from the right.
    p = np.maximum.accumulate(p[::-1])[::-1]
    # Points where the recall changes.
    steps = np.nonzero(r[1:] != r[:-1])[0]
    # Area under the interpolated curve.
    return float(np.sum((r[steps + 1] - r[steps]) * p[steps + 1]))


# Accumulates detections and ground truth boxes to compute average precision.
class DetectionAccumulator:
    # Create an empty accumulator.
    def __init__(self, num_classes: int, class_names: list[str] | None = None) -> None:
        # Number of classes.
        self.num_classes = num_classes
        # Names used in the report.
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        # Per image: predicted boxes, scores and classes.
        self._predictions: list[tuple[NDArray[Any], NDArray[Any], NDArray[Any]]] = []
        # Per image: ground truth boxes and classes.
        self._targets: list[tuple[NDArray[Any], NDArray[Any]]] = []

    # Add the predictions and ground truth of one image.
    def update(
        self,  # The accumulator.
        boxes: NDArray[Any],  # (N, 4) predicted boxes.
        scores: NDArray[Any],  # (N,) predicted scores.
        classes: NDArray[Any],  # (N,) predicted class ids.
        gt_boxes: NDArray[Any],  # (M, 4) reference boxes.
        gt_classes: NDArray[Any],  # (M,) reference class ids.
    ) -> None:  # Nothing is returned.
        # Store the predictions as arrays.
        self._predictions.append(
            (  # Prediction tuple.
                np.asarray(boxes, dtype=np.float64).reshape(-1, 4),  # Boxes.
                np.asarray(scores, dtype=np.float64).reshape(-1),  # Scores.
                np.asarray(classes, dtype=np.int64).reshape(-1),  # Classes.
            )  # End of the tuple.
        )  # End of the append.
        # Store the ground truth as arrays.
        self._targets.append(
            (  # Reference tuple.
                np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4),  # Boxes.
                np.asarray(gt_classes, dtype=np.int64).reshape(-1),  # Classes.
            )  # End of the tuple.
        )  # End of the append.

    # Average precision, precision and recall of one class at one IoU threshold.
    def _class_ap(self, cls: int, iou_threshold: float) -> tuple[float, float, float, int]:
        # Records of (score, is true positive) over all images.
        records: list[tuple[float, bool]] = []
        # Number of reference boxes of the class.
        num_gt = 0
        # Match predictions to references image by image.
        for (boxes, scores, classes), (gt_boxes, gt_classes) in zip(
            self._predictions,  # Predictions per image.
            self._targets,  # References per image.
        ):  # End of the pairs.
            # Predictions of the class, by decreasing score.
            idx = np.nonzero(classes == cls)[0]
            # Sort the predictions.
            idx = idx[np.argsort(-scores[idx], kind="stable")]
            # References of the class.
            gt = gt_boxes[gt_classes == cls]
            # Count the references.
            num_gt += len(gt)
            # Whether each reference is already matched.
            matched = np.zeros(len(gt), dtype=bool)
            # Overlaps between predictions and references.
            ious = box_iou(boxes[idx], gt) if len(gt) else np.zeros((len(idx), 0))
            # Greedy matching in score order.
            for row, i in enumerate(idx):
                # Best reference for this prediction.
                j = int(np.argmax(ious[row])) if ious.shape[1] else -1
                # True positive if it overlaps enough and is still free.
                hit = j >= 0 and ious[row, j] >= iou_threshold and not matched[j]
                # Mark the reference as used.
                if hit:
                    # Each reference can be matched once.
                    matched[j] = True
                # Record the prediction.
                records.append((float(scores[i]), bool(hit)))
        # Without references the class is undefined.
        if num_gt == 0:
            # NaN marks classes absent from the evaluation data.
            return float("nan"), float("nan"), float("nan"), 0
        # Without predictions precision is zero.
        if not records:
            # AP, precision and recall are zero.
            return 0.0, 0.0, 0.0, num_gt
        # Sort all predictions by score.
        records.sort(key=lambda r: -r[0])
        # Cumulative true positives.
        tp = np.cumsum([r[1] for r in records], dtype=np.float64)
        # Cumulative false positives.
        fp = np.cumsum([not r[1] for r in records], dtype=np.float64)
        # Recall along the ranking.
        recall = tp / num_gt
        # Precision along the ranking.
        precision = tp / np.maximum(tp + fp, 1e-12)
        # AP and final precision and recall.
        return average_precision(recall, precision), float(precision[-1]), float(recall[-1]), num_gt

    # Summarise the accumulated images.
    def compute(self) -> dict[str, Any]:
        # AP per class at IoU 0.5.
        ap50: dict[str, float] = {}
        # Precision and recall per class at IoU 0.5.
        precision: dict[str, float] = {}
        # Recall per class.
        recall: dict[str, float] = {}
        # COCO AP per class.
        ap_coco: dict[str, float] = {}
        # Evaluate every class.
        for cls, name in enumerate(self.class_names):
            # AP, precision and recall at IoU 0.5.
            ap, p, r, _ = self._class_ap(cls, 0.5)
            # Store the values.
            ap50[name], precision[name], recall[name] = ap, p, r
            # AP averaged over the COCO thresholds.
            ap_coco[name] = float(np.mean([self._class_ap(cls, t)[0] for t in COCO_IOU_THRESHOLDS]))

        # Mean over classes with references.
        def mean(values: dict[str, float]) -> float:
            # Finite values only.
            finite = [v for v in values.values() if np.isfinite(v)]
            # Mean, or NaN without any class.
            return float(np.mean(finite)) if finite else float("nan")

        # Report.
        return {
            "map50": mean(ap50),  # Mean AP at IoU 0.5.
            "map50_95": mean(ap_coco),  # Mean AP over IoU 0.5 to 0.95.
            "precision": mean(precision),  # Mean final precision.
            "recall": mean(recall),  # Mean final recall.
            "ap50_per_class": ap50,  # AP at IoU 0.5 per class.
            "images": len(self._targets),  # Number of evaluated images.
        }  # End of the report.


# Confusion matrix for semantic segmentation and change detection.
class ConfusionMatrix:
    # Create an empty matrix.
    def __init__(
        self,  # The matrix.
        num_classes: int,  # Number of classes.
        class_names: list[str] | None = None,  # Names used in the report.
        ignore_index: int = 255,  # Reference label that is not evaluated.
    ) -> None:  # The constructor returns nothing.
        # Number of classes.
        self.num_classes = num_classes
        # Names used in the report.
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        # Label that is skipped.
        self.ignore_index = ignore_index
        # Counts: rows are references, columns are predictions.
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Add a batch of predicted and reference labels of any shape.
    def update(self, predicted: NDArray[Any], reference: NDArray[Any]) -> None:
        # Flatten the predictions.
        p = np.asarray(predicted, dtype=np.int64).reshape(-1)
        # Flatten the references.
        r = np.asarray(reference, dtype=np.int64).reshape(-1)
        # Keep evaluated pixels with valid labels.
        valid = (r != self.ignore_index) & (r >= 0) & (r < self.num_classes)
        # Predictions must be valid class indices too.
        valid &= (p >= 0) & (p < self.num_classes)
        # Count the (reference, prediction) pairs.
        self.matrix += np.bincount(
            r[valid] * self.num_classes + p[valid],  # Pair index.
            minlength=self.num_classes**2,  # One bin per pair.
        ).reshape(self.num_classes, self.num_classes)  # Back to a matrix.

    # Summarise the matrix.
    def compute(self) -> dict[str, Any]:
        # Matrix as floats.
        m = self.matrix.astype(np.float64)
        # True positives per class.
        tp = np.diag(m)
        # Reference totals per class.
        ref = m.sum(axis=1)
        # Prediction totals per class.
        pred = m.sum(axis=0)
        # Number of evaluated pixels.
        total = m.sum()

        # Safe division that returns NaN for empty denominators.
        def ratio(num: NDArray[Any], den: NDArray[Any]) -> NDArray[np.float64]:
            # NaN where the denominator is zero.
            return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

        # Intersection over union per class.
        iou = ratio(tp, ref + pred - tp)
        # Precision per class.
        precision = ratio(tp, pred)
        # Recall per class.
        recall = ratio(tp, ref)
        # F1 per class.
        f1 = ratio(2 * tp, ref + pred)
        # Overall accuracy.
        accuracy = float(tp.sum() / total) if total else float("nan")
        # Agreement expected by chance.
        expected = float((ref * pred).sum() / total**2) if total else float("nan")
        # Cohen's kappa.
        kappa = (accuracy - expected) / (1 - expected) if total and expected < 1 else float("nan")
        # Classes present in the reference or the prediction.
        present = (ref + pred) > 0
        # Report.
        return {
            "accuracy": accuracy,  # Overall accuracy.
            "miou": float(np.nanmean(iou[present])) if present.any() else float("nan"),  # Mean IoU.
            "mf1": float(np.nanmean(f1[present])) if present.any() else float("nan"),  # Mean F1.
            "kappa": float(kappa),  # Cohen's kappa.
            "iou_per_class": dict(zip(self.class_names, iou.tolist())),  # IoU per class.
            "f1_per_class": dict(zip(self.class_names, f1.tolist())),  # F1 per class.
            "precision_per_class": dict(zip(self.class_names, precision.tolist())),  # Precision.
            "recall_per_class": dict(zip(self.class_names, recall.tolist())),  # Recall.
            "pixels": int(total),  # Evaluated pixels.
        }  # End of the report.


# Streaming regression errors per output, ignoring NaN references.
class RegressionAccumulator:
    # Create an empty accumulator.
    def __init__(self, names: list[str]) -> None:
        # Output names.
        self.names = list(names)
        # Number of outputs.
        k = len(self.names)
        # Number of valid values per output.
        self.n = np.zeros(k)
        # Sum of errors.
        self.sum_err = np.zeros(k)
        # Sum of absolute errors.
        self.sum_abs = np.zeros(k)
        # Sum of squared errors.
        self.sum_sq = np.zeros(k)
        # Sum of references.
        self.sum_ref = np.zeros(k)
        # Sum of squared references.
        self.sum_ref_sq = np.zeros(k)

    # Add predictions and references of shape (N, K) or (N, K, H, W).
    def update(self, predicted: NDArray[Any], reference: NDArray[Any]) -> None:
        # Predictions as float64.
        p = np.asarray(predicted, dtype=np.float64)
        # References as float64.
        r = np.asarray(reference, dtype=np.float64)
        # A batch axis and an output axis are required.
        if p.ndim < 2 or p.shape[1] != len(self.names) or p.shape != r.shape:
            # Explain the expected layout.
            raise ValueError(f"expected matching arrays of shape (N, {len(self.names)}, ...)")
        # Move the output axis first and flatten the rest.
        p = np.moveaxis(p, 1, 0).reshape(len(self.names), -1)
        # Same layout for the references.
        r = np.moveaxis(r, 1, 0).reshape(len(self.names), -1)
        # Valid pairs.
        valid = np.isfinite(p) & np.isfinite(r)
        # Errors, zero where invalid.
        err = np.where(valid, p - r, 0.0)
        # References, zero where invalid.
        ref = np.where(valid, r, 0.0)
        # Accumulate the counts.
        self.n += valid.sum(axis=1)
        # Accumulate the errors.
        self.sum_err += err.sum(axis=1)
        # Accumulate the absolute errors.
        self.sum_abs += np.abs(err).sum(axis=1)
        # Accumulate the squared errors.
        self.sum_sq += (err**2).sum(axis=1)
        # Accumulate the references.
        self.sum_ref += ref.sum(axis=1)
        # Accumulate the squared references.
        self.sum_ref_sq += (ref**2).sum(axis=1)

    # Summarise the errors.
    def compute(self) -> dict[str, Any]:
        # Avoid division by zero.
        n = np.maximum(self.n, 1)
        # Mean absolute error per output.
        mae = self.sum_abs / n
        # Root mean squared error per output.
        rmse = np.sqrt(self.sum_sq / n)
        # Mean error per output.
        bias = self.sum_err / n
        # Total sum of squares of the references.
        ss_tot = self.sum_ref_sq - self.sum_ref**2 / n
        # Coefficient of determination.
        r2 = np.where(ss_tot > 0, 1 - self.sum_sq / np.where(ss_tot > 0, ss_tot, 1), np.nan)
        # Outputs without valid values are undefined.
        empty = self.n == 0
        # Mark them as NaN.
        mae[empty] = rmse[empty] = bias[empty] = r2[empty] = np.nan
        # Report.
        return {
            "mae": float(np.nanmean(mae)) if not empty.all() else float("nan"),  # Mean MAE.
            "rmse": float(np.nanmean(rmse)) if not empty.all() else float("nan"),  # Mean RMSE.
            "bias": float(np.nanmean(bias)) if not empty.all() else float("nan"),  # Mean bias.
            "r2": float(np.nanmean(r2)) if np.isfinite(r2).any() else float("nan"),  # Mean R^2.
            "per_output": {  # Report per output.
                name: {  # Entry of one output.
                    "mae": float(mae[i]),  # Mean absolute error.
                    "rmse": float(rmse[i]),  # Root mean squared error.
                    "bias": float(bias[i]),  # Mean error.
                    "r2": float(r2[i]),  # Coefficient of determination.
                    "count": int(self.n[i]),  # Valid values.
                }  # End of the output entry.
                for i, name in enumerate(self.names)  # Every output.
            },  # End of the per-output report.
        }  # End of the report.


# Peak signal-to-noise ratio in decibels.
def psnr(predicted: NDArray[Any], reference: NDArray[Any], data_range: float = 1.0) -> float:
    # Mean squared error over finite values.
    diff = np.asarray(predicted, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    # Ignore NaN.
    mse = float(np.nanmean(diff**2))
    # Identical images have infinite PSNR.
    if mse == 0:
        # Positive infinity.
        return float("inf")
    # 10 log10(range^2 / MSE).
    return float(10 * np.log10(data_range**2 / mse))


# Mean structural similarity of two images of shape (C, H, W) or (H, W).
def ssim(
    predicted: NDArray[Any],  # Predicted image.
    reference: NDArray[Any],  # Reference image.
    data_range: float = 1.0,  # Value range of the images.
    sigma: float = 1.5,  # Standard deviation of the Gaussian window.
) -> float:  # Mean SSIM over pixels and bands.
    # Images as float64 with a band axis.
    x = np.atleast_3d(np.asarray(predicted, dtype=np.float64).T).T
    # Reference in the same layout.
    y = np.atleast_3d(np.asarray(reference, dtype=np.float64).T).T
    # Stabilising constants of Wang et al. (2004).
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    # SSIM of every band.
    values = []
    # Iterate over the bands.
    for xb, yb in zip(x, y):
        # Replace NaN so that the filters stay finite.
        xb, yb = np.nan_to_num(xb), np.nan_to_num(yb)
        # Local means.
        mx, my = gaussian_filter(xb, sigma), gaussian_filter(yb, sigma)
        # Local variances.
        vx = gaussian_filter(xb * xb, sigma) - mx * mx
        # Variance of the reference.
        vy = gaussian_filter(yb * yb, sigma) - my * my
        # Local covariance.
        cxy = gaussian_filter(xb * yb, sigma) - mx * my
        # SSIM map.
        s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
        # Mean of the map.
        values.append(float(s.mean()))
    # Mean over bands.
    return float(np.mean(values))


# Streaming PSNR and SSIM over a set of images.
class ImageQuality:
    # Create an empty accumulator.
    def __init__(self, data_range: float = 1.0) -> None:
        # Value range of the images.
        self.data_range = data_range
        # PSNR of every image.
        self.psnr: list[float] = []
        # SSIM of every image.
        self.ssim: list[float] = []
        # Regression errors over all pixels.
        self.errors = RegressionAccumulator(["image"])

    # Add a batch of images (N, C, H, W) or one image (C, H, W).
    def update(self, predicted: NDArray[Any], reference: NDArray[Any]) -> None:
        # Add a batch axis to single images.
        p = np.asarray(predicted)[None] if np.ndim(predicted) == 3 else np.asarray(predicted)
        # Same for the references.
        r = np.asarray(reference)[None] if np.ndim(reference) == 3 else np.asarray(reference)
        # Evaluate every image.
        for pi, ri in zip(p, r):
            # PSNR of the image.
            self.psnr.append(psnr(pi, ri, self.data_range))
            # SSIM of the image.
            self.ssim.append(ssim(pi, ri, self.data_range))
            # Pixel errors.
            self.errors.update(pi.reshape(1, 1, -1), ri.reshape(1, 1, -1))

    # Summarise the images.
    def compute(self) -> dict[str, Any]:
        # Finite PSNR values; identical images give infinity.
        finite = [v for v in self.psnr if np.isfinite(v)]
        # Pixel error summary.
        errors = self.errors.compute()
        # Report.
        return {
            "psnr": float(np.mean(finite)) if finite else float("inf"),  # Mean PSNR.
            "ssim": float(np.mean(self.ssim)) if self.ssim else float("nan"),  # Mean SSIM.
            "mae": errors["mae"],  # Mean absolute error.
            "rmse": errors["rmse"],  # Root mean squared error.
            "images": len(self.ssim),  # Evaluated images.
        }  # End of the report.


# Metric that selects the best checkpoint, and whether larger is better.
MONITOR = {
    Task.DETECTION: ("map50", True),  # Mean AP at IoU 0.5.
    Task.SEGMENTATION: ("miou", True),  # Mean IoU.
    Task.CHANGE_DETECTION: ("miou", True),  # Mean IoU.
    Task.DENSE_REGRESSION: ("rmse", False),  # Root mean squared error.
    Task.SCENE_REGRESSION: ("rmse", False),  # Root mean squared error.
    Task.ENHANCEMENT: ("psnr", True),  # Peak signal-to-noise ratio.
    Task.SUPER_RESOLUTION: ("psnr", True),  # Peak signal-to-noise ratio.
    Task.SPECTRAL_INDEX: ("rmse", False),  # Error against reference indices.
}  # End of the monitored metrics.


# Evaluates raw network outputs against batch targets for any task.
class TaskEvaluator:
    # Create the accumulator of the task.
    def __init__(
        self,  # The evaluator.
        config: BuildConfig,  # Model configuration.
        threshold: float = 0.3,  # Detection score threshold.
        data_range: float = 1.0,  # Value range for PSNR and SSIM.
    ) -> None:  # The constructor returns nothing.
        # Model configuration.
        self.config = config
        # Detection score threshold.
        self.threshold = threshold
        # Task of the model.
        task = config.task
        # Output names.
        names = list(config.outputs)
        # Displacement fields (dx, dy) are evaluated like regression targets.
        self.displacement = {"dx", "dy"} <= set(names)
        # Detection accumulator.
        if task is Task.DETECTION:
            # Average precision.
            self.acc: Any = DetectionAccumulator(len(names), names)
        # Classification accumulator.
        elif task in (Task.SEGMENTATION, Task.CHANGE_DETECTION):
            # Confusion matrix.
            self.acc = ConfusionMatrix(len(names), names)
        # Image quality accumulator; displacement fields are regression targets.
        elif task in (Task.ENHANCEMENT, Task.SUPER_RESOLUTION) and not self.displacement:
            # PSNR and SSIM.
            self.acc = ImageQuality(data_range)
        # Regression accumulator.
        else:
            # Errors per output.
            self.acc = RegressionAccumulator(names)

    # Name of the monitored metric and whether larger values are better.
    @property
    def monitor(self) -> tuple[str, bool]:
        # Displacement fields are judged by their error.
        if self.displacement:
            # Smaller RMSE is better.
            return ("rmse", False)
        # Lookup by task.
        return MONITOR[self.config.task]

    # Add a batch of raw outputs (N, ...) and NumPy targets.
    def update(self, outputs: NDArray[Any], targets: dict[str, Any]) -> None:
        # Task of the model.
        task = self.config.task
        # Detection: decode every image and compare with its boxes.
        if task is Task.DETECTION:
            # Every image of the batch.
            for i, o in enumerate(outputs):
                # Predicted boxes of the image.
                boxes, scores, classes = decode_centernet(o, self.threshold)
                # Compare with the reference boxes.
                self.acc.update(boxes, scores, classes, targets["boxes"][i], targets["labels"][i])
        # Classification: most likely class per pixel.
        elif task in (Task.SEGMENTATION, Task.CHANGE_DETECTION):
            # Confusion of predicted and reference labels.
            self.acc.update(np.argmax(outputs, axis=1), targets["mask"])
        # Scene regression: target vectors.
        elif task is Task.SCENE_REGRESSION:
            # Errors per output.
            self.acc.update(outputs, targets["vector"])
        # Dense tasks: target maps.
        else:
            # Errors or image quality.
            self.acc.update(outputs, targets["values"])

    # Summarise the accumulated batches.
    def compute(self) -> dict[str, Any]:
        # Report of the accumulator.
        return self.acc.compute()


# =============================================================================
# End of module src/unbihexium/ai/evaluation.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
