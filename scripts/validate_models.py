#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : scripts/validate_models.py
# Title       : End-to-end validation of the model zoo models
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch; --onnx also requires
#               onnx and onnxruntime
# =============================================================================
#
# Abstract
# --------
# Validates every model of the model zoo end to end. For each selected model
# the script:
#
#   1. builds the model with its deterministic starter weights;
#   2. compares the weights digest with the published digest;
#   3. runs a forward pass on a random input and checks the output shape
#      and that the output is finite;
#   4. with --onnx, exports the model to ONNX and compares ONNX Runtime with
#      PyTorch on a second input size.
#
# Usage
# -----
#   python scripts/validate_models.py                    # all 520 models
#   python scripts/validate_models.py --variant tiny     # 130 tiny models
#   python scripts/validate_models.py --variant tiny --onnx
#   python scripts/validate_models.py --family ship_detector
#
# Exit status
# -----------
#   0  every selected model passed
#   1  at least one model failed
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Command line parsing.
import argparse

# Set the exit status of the script.
import sys

# Temporary directory for ONNX exports.
import tempfile

# Measure the time per model.
import time

# Represent file paths.
from pathlib import Path

# Tensors and inference mode.
import torch

# Model construction.
from unbihexium.ai.models import DETECTION_STRIDE, build_model

# Catalogue and registry.
from unbihexium.zoo import Task, Variant, get_model, list_specs

# Spatial size of the validation input; divisible by 2**depth of every variant.
SIZE = 64


# Expected output shape for a batch of one input of size SIZE.
def expected_shape(spec) -> tuple[int, ...]:
    # Number of outputs.
    k = spec.out_channels
    # Detectors predict at stride 4 with four extra channels.
    if spec.task is Task.DETECTION:
        # Heatmaps, size and offset.
        return (1, k + 4, SIZE // DETECTION_STRIDE, SIZE // DETECTION_STRIDE)
    # Scene regressors predict one vector.
    if spec.task is Task.SCENE_REGRESSION:
        # One value per target.
        return (1, k)
    # Super-resolution scales the output.
    if spec.task is Task.SUPER_RESOLUTION:
        # Upscaled output.
        return (1, k, SIZE * spec.scale, SIZE * spec.scale)
    # Dense tasks keep the input size.
    return (1, k, SIZE, SIZE)


# Validate one model and return an error message or None.
def validate(spec, variant: Variant, onnx_dir: Path | None) -> str | None:
    # Build the starter model.
    model = build_model(spec.family, variant)
    # Published registry entry.
    entry = get_model(model.model_id)
    # The digest must match the published one.
    if entry is None or model.digest() != entry.weights_digest:
        # Report the mismatch.
        return "weights digest differs from the published digest"
    # Random input with values in [0, 1).
    x = torch.rand(1, spec.in_channels, SIZE, SIZE)
    # Forward pass without gradients.
    with torch.no_grad():
        # Output of the model.
        y = model(x)
    # The output shape must match the task.
    if tuple(y.shape) != expected_shape(spec):
        # Report the wrong shape.
        return f"output shape {tuple(y.shape)} != expected {expected_shape(spec)}"
    # The output must be finite.
    if not torch.isfinite(y).all():
        # Report non-finite values.
        return "output contains NaN or infinite values"
    # Optional ONNX export and comparison.
    if onnx_dir is not None:
        # Export module; imported here because it needs onnx.
        from unbihexium.zoo.export import export_onnx  # Export and verify.

        # Export and verify against ONNX Runtime.
        export_onnx(model, onnx_dir / f"{model.model_id}.onnx")
    # The model passed.
    return None


# Validate the selected models and return the exit status.
def main(argv: list[str]) -> int:
    # Argument parser.
    parser = argparse.ArgumentParser(description="Validate the model zoo models")
    # Restrict to one variant.
    parser.add_argument("--variant", choices=[v.value for v in Variant], help="only this variant")
    # Restrict to one family.
    parser.add_argument("--family", help="only this model family")
    # Also export and compare ONNX.
    parser.add_argument("--onnx", action="store_true", help="export to ONNX and compare")
    # Parse the arguments.
    args = parser.parse_args(argv)
    # Selected variants.
    variants = [Variant(args.variant)] if args.variant else list(Variant)
    # Selected families.
    specs = [s for s in list_specs() if args.family in (None, s.family)]
    # Unknown families select nothing.
    if not specs:
        # Report the unknown family.
        print(f"unknown family {args.family!r}")
        # Fail.
        return 1
    # Failures as (model id, message).
    failures: list[tuple[str, str]] = []
    # Number of validated models.
    checked = 0
    # Directory for ONNX files, removed at the end.
    with tempfile.TemporaryDirectory() as tmp:
        # ONNX directory when requested.
        onnx_dir = Path(tmp) if args.onnx else None
        # Validate every selected model.
        for spec in specs:
            # Every selected variant.
            for variant in variants:
                # Start time of the model.
                start = time.perf_counter()
                # Validate and capture unexpected exceptions as failures.
                try:
                    # Error message or None.
                    problem = validate(spec, variant, onnx_dir)
                # Any exception is a failure of this model.
                except Exception as exc:  # Report every failure.
                    # Exception type and message.
                    problem = f"{type(exc).__name__}: {exc}"
                # Count the model.
                checked += 1
                # Model id of the validated model.
                model_id = spec.model_id(variant)
                # Record failures.
                if problem:
                    # Keep the failure.
                    failures.append((model_id, problem))
                # Progress line with the result and duration.
                status = "FAIL" if problem else "ok"
                # Print the progress line.
                print(f"{status:4} {model_id} ({time.perf_counter() - start:.1f} s)", flush=True)
    # Summary.
    print(f"\n{checked} models validated, {len(failures)} failed.")
    # Details of the failures.
    for model_id, problem in failures:
        # One line per failure.
        print(f"  {model_id}: {problem}")
    # Success only without failures.
    return 1 if failures else 0


# Run the validation when the file is executed as a script.
if __name__ == "__main__":
    # Pass the command line arguments without the script name.
    sys.exit(main(sys.argv[1:]))

# =============================================================================
# End of module scripts/validate_models.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
