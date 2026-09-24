# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/inference.py
# Title       : Tiled inference with PyTorch or ONNX Runtime
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and either PyTorch
#               (unbihexium[torch]) or ONNX Runtime (unbihexium[onnx])
# =============================================================================
#
# Abstract
# --------
# Predictor runs a model zoo model on images of any size:
#
#   1. The image is normalised with the statistics stored by training
#      (config.extra["normalization"]); missing values become zero.
#   2. Images larger than the tile size are cut into overlapping tiles. Tiles
#      at the image border are padded by reflection.
#   3. Dense outputs (class probabilities, regression values, enhanced or
#      upscaled bands) are blended with weights that fall towards the tile
#      edges, which removes seams. Detections are decoded per tile, boxes
#      whose centre lies in the overlap margin of an inner tile edge are
#      dropped, and the remaining boxes pass class-aware non-maximum
#      suppression.
#   4. Pixels without valid input are NaN in the output.
#
# Models can be given as a ZooModel, a checkpoint (.pt), an ONNX file
# (.onnx) or a model id. ONNX files carry their configuration as metadata, so
# ONNX Runtime inference does not need PyTorch.
#
# Usage
# -----
#   from unbihexium.ai.inference import Predictor
#   predictor = Predictor("runs/ship/best.pt", tile_size=512)
#   boxes, scores, classes = predictor.detect(image, threshold=0.3)
#   probabilities = Predictor("lulc_classifier_base").dense(image)
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Parse the ONNX configuration metadata.
import json

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Decoding of network outputs.
from unbihexium.ai.decode import decode_centernet, nms, softmax

# Per-band standardisation.
from unbihexium.ai.transforms import Normalization

# Task of a model.
from unbihexium.zoo.catalog import Task

# Configuration of a model.
from unbihexium.zoo.config import BuildConfig

# Metadata key of the configuration in ONNX files.
ONNX_METADATA_KEY = "unbihexium_config"

# Smallest blending weight at tile edges.
MIN_BLEND_WEIGHT = 1e-3


# Runs a PyTorch ZooModel on NumPy batches.
class TorchBackend:
    # Wrap a model.
    def __init__(self, model: Any, device: str = "cpu") -> None:
        # PyTorch is imported lazily.
        import torch

        # Torch module used for inference.
        self._torch = torch
        # Device of the computation.
        self.device = torch.device(device)
        # Model in evaluation mode on the device.
        self.model = model.to(self.device).eval()
        # Configuration of the model.
        self.config: BuildConfig = model.config

    # Run a batch (N, C, H, W) and return the output as float32.
    def run(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        # Inference mode disables autograd bookkeeping.
        with self._torch.inference_mode():
            # Tensor on the device.
            x = self._torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
            # Forward pass, back to NumPy.
            return self.model(x).float().cpu().numpy()


# Runs an exported ONNX model with ONNX Runtime.
class OnnxBackend:
    # Open an ONNX file.
    def __init__(self, path: str | Path, providers: list[str] | None = None) -> None:
        # ONNX Runtime is imported lazily.
        import onnxruntime as ort

        # Inference session; the CPU provider is always available.
        self.session = ort.InferenceSession(
            str(path),  # Model file.
            providers=providers or ["CPUExecutionProvider"],  # Execution providers.
        )  # End of the session.
        # Custom metadata of the model.
        meta = self.session.get_modelmeta().custom_metadata_map
        # Files not exported by Unbihexium lack the configuration.
        if ONNX_METADATA_KEY not in meta:
            # Explain the problem.
            raise ValueError(f"{path} has no Unbihexium configuration metadata")
        # Configuration of the model.
        self.config = BuildConfig.from_dict(json.loads(meta[ONNX_METADATA_KEY]))
        # Name of the input tensor.
        self.input_name = self.session.get_inputs()[0].name

    # Run a batch (N, C, H, W) and return the output as float32.
    def run(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        # Single output of the graph.
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(batch, np.float32)})
        # First output as float32.
        return np.asarray(out[0], dtype=np.float32)


# Open a backend for a model given in any supported form.
def open_backend(
    source: Any,  # ZooModel, checkpoint path, ONNX path or model id.
    variant: str | None = None,  # Variant for model ids without a suffix.
    device: str = "cpu",  # Torch device.
    backend: str = "auto",  # auto, torch or onnx.
) -> TorchBackend | OnnxBackend:  # Backend with a config attribute.
    # Model objects are used directly.
    if hasattr(source, "config") and hasattr(source, "forward"):
        # PyTorch model.
        return TorchBackend(source, device)
    # Existing files decide by their extension.
    path = Path(str(source))
    # ONNX files.
    if path.suffix.lower() == ".onnx" and path.is_file():
        # ONNX Runtime backend.
        return OnnxBackend(path)
    # Checkpoint files; a variant only applies to catalogue names.
    checkpoint = path.suffix.lower() == ".pt" and path.is_file()
    # Checkpoints and model ids need PyTorch unless ONNX was requested.
    if backend == "onnx":
        # Checkpoint files are exported once, next to the model store.
        if checkpoint:
            # ONNX Runtime backend of the export.
            return OnnxBackend(_export_checkpoint(path))
        # Imported lazily: the store builds the ONNX file with PyTorch once.
        from unbihexium.zoo.store import ONNX_NAME, ensure_model

        # Cached ONNX export of the model.
        directory = ensure_model(f"{source}_{variant}" if variant else str(source), onnx=True)
        # ONNX Runtime backend.
        return OnnxBackend(directory / ONNX_NAME)
    # Imported lazily: loading requires PyTorch.
    from unbihexium.zoo.store import load_model

    # Checkpoint file or catalogue model.
    return TorchBackend(load_model(source, None if checkpoint else variant), device)


# ONNX export of a checkpoint file, cached by the SHA-256 of the checkpoint.
def _export_checkpoint(path: Path) -> Path:
    # PyTorch-dependent modules imported lazily.
    from unbihexium.zoo.checkpoint import load_checkpoint  # Reads checkpoint files.
    from unbihexium.zoo.export import export_onnx  # Writes ONNX files.
    from unbihexium.zoo.store import get_cache_dir  # Root of the model store.
    from unbihexium.zoo.verify import compute_sha256  # Content digest.

    # One export per checkpoint content, beside the model store.
    target = get_cache_dir().parent / "exports" / f"{compute_sha256(path)}.onnx"
    # Export once; the digest in the name ties the file to the checkpoint.
    if not target.is_file():
        # Create the directory.
        target.parent.mkdir(parents=True, exist_ok=True)
        # Temporary name, so that a failed export is never reused.
        partial = target.with_name(target.stem + ".partial.onnx")
        # Export and compare with PyTorch.
        export_onnx(load_checkpoint(path), partial)
        # Publish the finished file atomically.
        partial.replace(target)
    # Path of the export.
    return target


# Start positions of tiles of length `tile` with step `step` covering `size`.
def tile_starts(size: int, tile: int, step: int) -> list[int]:
    # A single tile covers small images.
    if size <= tile:
        # Start at zero.
        return [0]
    # Regular positions.
    starts = list(range(0, size - tile, step))
    # Last tile flush with the border.
    starts.append(size - tile)
    # Return the positions.
    return starts


# Two-dimensional blending weights that fall towards the edges.
def blend_weights(height: int, width: int) -> NDArray[np.float32]:
    # Distance to the nearest edge along rows, normalised to (0, 1].
    wy = np.minimum(np.arange(height) + 1, np.arange(height)[::-1] + 1) / ((height + 1) / 2)
    # Same along columns.
    wx = np.minimum(np.arange(width) + 1, np.arange(width)[::-1] + 1) / ((width + 1) / 2)
    # Outer product with a small floor.
    return np.maximum(np.outer(wy, wx), MIN_BLEND_WEIGHT).astype(np.float32)


# Tiled inference for a model zoo model.
class Predictor:
    # Open a model for inference.
    def __init__(
        self,  # The predictor.
        source: Any,  # ZooModel, checkpoint path, ONNX path or model id.
        variant: str | None = None,  # Variant for model ids.
        device: str = "cpu",  # Torch device.
        backend: str = "auto",  # auto, torch or onnx.
        tile_size: int | None = None,  # Tile size; default from the configuration.
        overlap: float = 0.25,  # Overlap between tiles as a fraction of the tile size.
        batch_size: int = 4,  # Tiles per forward pass.
        normalization: Normalization | None = None,  # Overrides the stored statistics.
    ) -> None:  # The constructor returns nothing.
        # Backend running the model.
        self.backend = open_backend(source, variant, device, backend)
        # Configuration of the model.
        self.config: BuildConfig = self.backend.config
        # Sizes that avoid padding inside the network.
        m = self.config.size_multiple
        # Tile size rounded up to that multiple.
        tile = tile_size or self.config.tile_size
        # Round the tile size.
        self.tile_size = int(np.ceil(tile / m) * m)
        # Overlap fraction, limited to [0, 0.9].
        self.overlap = float(min(max(overlap, 0.0), 0.9))
        # Tiles per forward pass.
        self.batch_size = max(1, int(batch_size))
        # Stored statistics, if training recorded them.
        stored = self.config.normalization
        # Explicit statistics win over stored ones.
        self.normalization = normalization or (Normalization.from_dict(stored) if stored else None)

    # Model id of the loaded model.
    @property
    def model_id(self) -> str:
        # Taken from the configuration.
        return self.config.model_id

    # Validate and normalise an image; returns the input and the invalid mask.
    def prepare(self, image: NDArray[Any]) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        # Float32 array with a band axis.
        x = np.asarray(image, dtype=np.float32)
        # Single-band images get a band axis.
        if x.ndim == 2:
            # Add the axis.
            x = x[None]
        # The band count must match the model.
        if x.ndim != 3 or x.shape[0] != self.config.in_channels:
            # Explain the expected input.
            raise ValueError(
                f"{self.model_id} expects {self.config.in_channels} bands "
                f"({', '.join(self.config.channel_names)}), got shape {x.shape}"
            )  # End of the error.
        # Pixels where every band is missing.
        invalid = ~np.isfinite(x).any(axis=0)
        # Spectral index formulas work on the raw values and propagate NaN.
        if self.config.task is Task.SPECTRAL_INDEX:
            # Return the raw input.
            return x, invalid
        # Apply the training normalisation or replace missing values only.
        x = self.normalization(x) if self.normalization else np.nan_to_num(x, nan=0.0)
        # Return the input and the invalid mask.
        return x.astype(np.float32), invalid

    # Tile windows (top, left) for an image of the given size.
    def windows(self, height: int, width: int) -> list[tuple[int, int]]:
        # Step between tiles, a multiple of the network size multiple.
        m = self.config.size_multiple
        # Step without the overlap.
        step = max(m, int(self.tile_size * (1 - self.overlap)) // m * m)
        # Row starts.
        rows = tile_starts(height, self.tile_size, step)
        # Column starts.
        cols = tile_starts(width, self.tile_size, step)
        # All combinations.
        return [(r, c) for r in rows for c in cols]

    # Cut and pad tiles, run them in batches and yield (window, output) pairs.
    def _run_tiles(self, x: NDArray[np.float32]) -> Any:
        # Image size.
        _, h, w = x.shape
        # Tile size, or the padded image size for small images.
        m = self.config.size_multiple
        # Effective tile height.
        th = self.tile_size if h > self.tile_size else int(np.ceil(h / m) * m)
        # Effective tile width.
        tw = self.tile_size if w > self.tile_size else int(np.ceil(w / m) * m)
        # Pad the image so that every window is complete.
        mode = "reflect" if min(h, w) > 1 else "edge"
        # Padded image.
        padded = np.pad(x, ((0, 0), (0, max(th - h, 0)), (0, max(tw - w, 0))), mode=mode)
        # Windows over the original extent.
        windows = self.windows(h, w) if (h > th or w > tw) else [(0, 0)]
        # Process the windows in batches.
        for i in range(0, len(windows), self.batch_size):
            # Windows of this batch.
            chunk = windows[i : i + self.batch_size]
            # Stack the tiles.
            batch = np.stack([padded[:, r : r + th, c : c + tw] for r, c in chunk])
            # Run the model.
            out = self.backend.run(batch)
            # Yield every tile with its output.
            for window, o in zip(chunk, out):
                # Window, tile size and output.
                yield window, (th, tw), o

    # Dense prediction: probabilities, values or bands on the output grid.
    def dense(self, image: NDArray[Any]) -> NDArray[np.float32]:
        # Detection and scene models have no dense output.
        if self.config.task in (Task.DETECTION, Task.SCENE_REGRESSION):
            # Explain the correct method.
            raise ValueError(f"{self.model_id} is a {self.config.task.value} model")
        # Normalised input and invalid pixels.
        x, invalid = self.prepare(image)
        # Image size.
        _, h, w = x.shape
        # Output grid factor.
        s = self.config.scale if self.config.task is Task.SUPER_RESOLUTION else 1
        # Classification outputs are blended as probabilities.
        classify = self.config.task in (Task.SEGMENTATION, Task.CHANGE_DETECTION)
        # Accumulated outputs on the padded output grid.
        acc: NDArray[np.float64] | None = None
        # Accumulated weights.
        weight: NDArray[np.float64] | None = None
        # Blend the tiles.
        for (r, c), (th, tw), o in self._run_tiles(x):
            # Allocate the accumulators on the first tile.
            if acc is None or weight is None:
                # Windows end inside the image, or at the padded tile size.
                ph, pw = max(h, th) * s, max(w, tw) * s
                # Output accumulator.
                acc = np.zeros((o.shape[0], ph, pw), dtype=np.float64)
                # Weight accumulator.
                weight = np.zeros((ph, pw), dtype=np.float64)
                # Blending weights of a tile.
                wmap = blend_weights(th * s, tw * s)
            # Probabilities instead of logits.
            o = softmax(o, axis=0) if classify else o
            # Output window.
            rows = slice(r * s, (r + th) * s)
            # Output columns.
            cols = slice(c * s, (c + tw) * s)
            # Add the weighted tile; NaN values do not contribute.
            valid = np.isfinite(o)
            # Weighted values.
            acc[:, rows, cols] += np.where(valid, o * wmap, 0.0)
            # Weights of valid values; NaN in any band removes the pixel.
            weight[rows, cols] += wmap * valid.all(axis=0)
        # Mypy: at least one tile ran.
        assert acc is not None and weight is not None
        # Weighted mean, NaN where nothing contributed.
        with np.errstate(invalid="ignore", divide="ignore"):
            # Divide by the weights.
            out = (acc / weight)[:, : h * s, : w * s].astype(np.float32)
        # Invalid input pixels on the output grid.
        mask = np.repeat(np.repeat(invalid, s, axis=0), s, axis=1)
        # Mark them as NaN.
        out[:, mask] = np.nan
        # Return the dense output.
        return out

    # Object detection: boxes (N, 4) in pixels, scores (N,) and class ids (N,).
    def detect(
        self,  # The predictor.
        image: NDArray[Any],  # Image (C, H, W).
        threshold: float = 0.3,  # Minimum score.
        max_detections: int = 1000,  # Maximum boxes per image.
        iou_threshold: float = 0.5,  # Overlap for non-maximum suppression.
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:  # Detections.
        # Only detection models produce boxes.
        if self.config.task is not Task.DETECTION:
            # Explain the correct method.
            raise ValueError(f"{self.model_id} is not a detection model")
        # Normalised input and invalid pixels.
        x, invalid = self.prepare(image)
        # Image size.
        _, h, w = x.shape
        # Margin of inner tile edges in which detections are dropped.
        margin = self.tile_size * self.overlap / 2
        # Collected boxes, scores and classes.
        all_boxes, all_scores, all_classes = [], [], []
        # Decode every tile.
        for (r, c), (th, tw), o in self._run_tiles(x):
            # Boxes of the tile.
            boxes, scores, classes = decode_centernet(o, threshold, max_detections)
            # Centres of the boxes in tile pixels.
            cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
            # Keep centres away from inner edges; image borders keep everything.
            keep = (
                ((cx >= margin) | (c == 0))  # Left margin.
                & ((cx < tw - margin) | (c + tw >= w))  # Right margin.
                & ((cy >= margin) | (r == 0))  # Top margin.
                & ((cy < th - margin) | (r + th >= h))  # Bottom margin.
            )  # End of the selection.
            # Shift into image coordinates.
            all_boxes.append(boxes[keep] + np.array([c, r, c, r], dtype=np.float64))
            # Scores of the kept boxes.
            all_scores.append(scores[keep])
            # Classes of the kept boxes.
            all_classes.append(classes[keep])
        # Concatenate the tiles.
        boxes = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4))
        # Scores.
        scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
        # Classes.
        classes = np.concatenate(all_classes) if all_classes else np.zeros(0, dtype=np.int64)
        # Clip to the image.
        boxes = np.clip(boxes, 0, [w, h, w, h])
        # Drop boxes centred on invalid pixels.
        if len(boxes) and invalid.any():
            # Centre pixels of the boxes.
            ci = np.clip(((boxes[:, 1] + boxes[:, 3]) / 2).astype(int), 0, h - 1)
            # Centre columns.
            cj = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2).astype(int), 0, w - 1)
            # Valid centres.
            ok = ~invalid[ci, cj]
            # Apply the selection.
            boxes, scores, classes = boxes[ok], scores[ok], classes[ok]
        # Suppress duplicates from overlapping tiles.
        keep_idx = nms(boxes, scores, iou_threshold, classes)[:max_detections]
        # Return the kept detections.
        return boxes[keep_idx], scores[keep_idx], classes[keep_idx]

    # Scene-level prediction: one value per output for the whole image.
    def scene(self, image: NDArray[Any]) -> NDArray[np.float32]:
        # Only scene models produce vectors.
        if self.config.task is not Task.SCENE_REGRESSION:
            # Explain the correct method.
            raise ValueError(f"{self.model_id} is not a scene regression model")
        # Normalised input.
        x, _ = self.prepare(image)
        # Pad to the size multiple of the network.
        m = self.config.size_multiple
        # Padded height.
        ph = int(np.ceil(x.shape[1] / m) * m)
        # Padded width.
        pw = int(np.ceil(x.shape[2] / m) * m)
        # Reflect padding keeps the statistics of the image.
        mode = "reflect" if min(x.shape[1:]) > 1 else "edge"
        # Padded image.
        x = np.pad(x, ((0, 0), (0, ph - x.shape[1]), (0, pw - x.shape[2])), mode=mode)
        # Global pooling handles any size in a single pass.
        return self.backend.run(x[None])[0]


# =============================================================================
# End of module src/unbihexium/ai/inference.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
