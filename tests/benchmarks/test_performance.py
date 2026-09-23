# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/benchmarks/test_performance.py
# Title       : Throughput and memory measurements of the model zoo
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest and PyTorch
# =============================================================================
#
# Abstract
# --------
# Measures inference throughput of tiny detection and segmentation models,
# the scaling of throughput with the batch size, and the memory footprint
# of model weights and of tiled inference. The measurements are printed
# (run pytest with -s to see them) and checked against loose bounds that
# every supported CPU meets, so that the tests catch gross regressions
# (for example accidental quadratic tiling) without being flaky.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Wall-clock timing.
import time

# Peak memory of NumPy allocations.
import tracemalloc

# Arrays.
import numpy as np

# Test framework.
import pytest

# PyTorch is optional for the library; skip these measurements without it.
torch = pytest.importorskip("torch")

# Tiled inference.
from unbihexium.ai.inference import Predictor  # noqa: E402 - imported after the skip check

# Model construction.
from unbihexium.ai.models import build_model  # noqa: E402 - imported after the skip check


# Images per second of a model on random input, after one warm-up pass.
def throughput(model_id: str, batch: int, size: int = 128, repeats: int = 3) -> float:
    # Tiny model in evaluation mode.
    model = build_model(model_id)
    # Random input batch.
    x = torch.rand(batch, model.config.in_channels, size, size)
    # Inference without gradients.
    with torch.inference_mode():
        # Warm-up pass, excluded from the timing.
        model(x)
        # Start of the timed passes.
        start = time.perf_counter()
        # Timed passes.
        for _ in range(repeats):
            # Forward pass.
            model(x)
        # Elapsed time.
        elapsed = time.perf_counter() - start
    # Images per second.
    return batch * repeats / elapsed


# Throughput measurements.
class TestInferenceBenchmarks:
    # Detection throughput of the tiny ship detector.
    def test_detection_throughput(self) -> None:
        # Images of 128 x 128 pixels per second.
        rate = throughput("ship_detector_tiny", batch=4)
        # Report the measurement.
        print(f"ship_detector_tiny: {rate:.1f} images/s at 128 px")
        # Loose bound met by every supported CPU.
        assert rate > 1.0

    # Segmentation throughput of the tiny land cover model.
    def test_segmentation_throughput(self) -> None:
        # Images of 128 x 128 pixels per second.
        rate = throughput("lulc_classifier_tiny", batch=4)
        # Report the measurement.
        print(f"lulc_classifier_tiny: {rate:.1f} images/s at 128 px")
        # Loose bound met by every supported CPU.
        assert rate > 1.0

    # Larger batches are not slower per image than single images.
    def test_batch_size_scaling(self) -> None:
        # Throughput with batch size 1.
        single = throughput("water_surface_detector_tiny", batch=1)
        # Throughput with batch size 8.
        batched = throughput("water_surface_detector_tiny", batch=8)
        # Report the measurements.
        print(f"batch 1: {single:.1f} images/s, batch 8: {batched:.1f} images/s")
        # Batching may not help on every CPU, but it must not collapse throughput.
        assert batched > 0.5 * single


# Memory measurements.
class TestMemoryBenchmarks:
    # Weight memory equals four bytes per parameter.
    def test_model_memory_footprint(self) -> None:
        # Base segmentation model.
        model = build_model("lulc_classifier_base")
        # Bytes of all parameters and buffers.
        tensors = list(model.parameters()) + list(model.buffers())
        # Total size in bytes.
        size = sum(t.numel() * t.element_size() for t in tensors)
        # Report the footprint.
        print(f"lulc_classifier_base: {size / 2**20:.1f} MiB of weights")
        # float32 weights: four bytes per parameter.
        assert size >= 4 * model.num_parameters()

    # Tiled inference on a large image keeps NumPy memory bounded.
    def test_inference_memory_usage(self) -> None:
        # Tiny regression model with 256 pixel tiles.
        predictor = Predictor(build_model("tree_height_estimator_tiny"), tile_size=256)
        # Image of 12 bands and 768 x 768 pixels (27 MiB as float32).
        image = np.random.default_rng(0).random((12, 768, 768), dtype=np.float32)
        # Track NumPy allocations.
        tracemalloc.start()
        # Run the tiled prediction.
        values = predictor.dense(image)
        # Peak traced memory in bytes.
        _, peak = tracemalloc.get_traced_memory()
        # Stop tracking.
        tracemalloc.stop()
        # Report the peak.
        print(f"tiled inference peak: {peak / 2**20:.1f} MiB")
        # The output covers the whole image.
        assert values.shape == (1, 768, 768)
        # Peak memory stays within a small multiple of the input size.
        assert peak < 10 * image.nbytes


# =============================================================================
# End of module tests/benchmarks/test_performance.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
