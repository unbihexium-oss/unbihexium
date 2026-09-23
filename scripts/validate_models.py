#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : scripts/validate_models.py
# Title       : Model zoo loading validation
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch (unbihexium[torch])
# =============================================================================
#
# Abstract
# --------
# Validates that every model of the model zoo (130 capabilities in four size
# variants, 520 models in total) can be loaded. For each model directory under
# model_zoo/assets/<variant>/ the script checks that:
#
#   - config.json, model.pt, model.onnx and model.sha256 exist;
#   - config.json is valid JSON and contains the "params" and "task" keys;
#   - model.pt loads with PyTorch and contains a "model_state_dict" entry.
#
# The script prints a per-variant and an overall summary and lists up to ten
# errors. The weight files must be downloaded from Git LFS first
# (`git lfs pull`); LFS pointer files fail the PyTorch loading check.
#
# Security note
# -------------
# torch.load is called with weights_only=False, which unpickles arbitrary
# Python objects. Run the script only on model files from this repository.
#
# Usage
# -----
#   python scripts/validate_models.py
#
# The script must run from the repository root.
#
# Exit status
# -----------
#   0  every model passed
#   1  at least one model failed
# =============================================================================

# Parse the config.json files.
import json

# Set the exit status of the script.
import sys

# Walk the model zoo directory tree.
from pathlib import Path

# Load the PyTorch checkpoints.
import torch


# Validate every model and return True when all of them passed.
def validate_models():
    # Directory that holds one subdirectory per size variant.
    root = Path("model_zoo/assets")
    # Size variants of the model zoo, from smallest to largest.
    variants = ["tiny", "base", "large", "mega"]

    # Number of model directories checked in all variants.
    total_checked = 0
    # Number of model directories that passed every check.
    total_passed = 0
    # Error messages of the models that failed.
    errors = []

    # Top rule of the report banner.
    print("=" * 60)
    # Report title.
    print("MODEL VALIDATION")
    # Bottom rule of the report banner.
    print("=" * 60)

    # Validate the models of each size variant.
    for v in variants:
        # Directory of this variant.
        vpath = root / v
        # A variant may be missing in a partial checkout.
        if not vpath.exists():
            # Report the skipped variant.
            print(f"\n[SKIP] Variant {v} not found")
            # Continue with the next variant.
            continue

        # Every entry of the variant directory is one model.
        models = list(vpath.iterdir())
        # Number of models of this variant that passed.
        v_passed = 0

        # Heading with the variant name and its number of models.
        print(f"\n{v.upper()} ({len(models)} models):")

        # Validate each model directory.
        for m in models:
            # Model configuration.
            cfg_path = m / "config.json"
            # PyTorch checkpoint.
            pt_path = m / "model.pt"
            # ONNX export.
            onnx_path = m / "model.onnx"
            # SHA-256 digests of the weight files.
            sha_path = m / "model.sha256"

            # Count the model as checked.
            total_checked += 1

            # Check all files exist
            if not all(p.exists() for p in [cfg_path, pt_path, onnx_path, sha_path]):
                # Record the failure.
                errors.append(f"{m.name}: Missing files")
                # Continue with the next model.
                continue

            # Check config is valid JSON
            try:
                # Parse the configuration file.
                cfg = json.load(open(cfg_path))
                # The configuration must describe the parameters and the task.
                if "params" not in cfg or "task" not in cfg:
                    # Record the failure.
                    errors.append(f"{m.name}: Invalid config")
                    # Continue with the next model.
                    continue
            # Any error while reading or parsing counts as a parse error.
            except:
                # Record the failure.
                errors.append(f"{m.name}: Config parse error")
                # Continue with the next model.
                continue

            # Check PT file loads
            try:
                # Load the checkpoint on the CPU so that no GPU is required.
                data = torch.load(pt_path, weights_only=False, map_location="cpu")
                # The checkpoint must contain the model weights.
                if "model_state_dict" not in data:
                    # Record the failure.
                    errors.append(f"{m.name}: Invalid PT structure")
                    # Continue with the next model.
                    continue
            # Record the loading error message.
            except Exception as e:
                # Include the PyTorch error in the message.
                errors.append(f"{m.name}: PT load error: {e}")
                # Continue with the next model.
                continue

            # Count the model as passed for this variant.
            v_passed += 1
            # Count the model as passed overall.
            total_passed += 1

        # Summary line of the variant.
        print(f"  Passed: {v_passed}/{len(models)}")

    # Top rule of the summary banner.
    print("\n" + "=" * 60)
    # Summary title.
    print("SUMMARY")
    # Bottom rule of the summary banner.
    print("=" * 60)
    # Number of models checked.
    print(f"Total Models Checked: {total_checked}")
    # Number of models that passed.
    print(f"Total Passed: {total_passed}")
    # Number of models that failed.
    print(f"Total Failed: {total_checked - total_passed}")

    # List the errors when there are any.
    if errors:
        # Heading with the number of errors.
        print(f"\nErrors ({len(errors)}):")
        # Show at most the first ten errors.
        for e in errors[:10]:
            # One error per line.
            print(f"  - {e}")
        # Say how many errors were not shown.
        if len(errors) > 10:
            # Number of hidden errors.
            print(f"  ... and {len(errors) - 10} more")

    # Success only when every checked model passed.
    return total_passed == total_checked


# Run the validation when the file is executed as a script.
if __name__ == "__main__":
    # Validate all models.
    success = validate_models()
    # Exit with 0 on success and 1 on failure.
    sys.exit(0 if success else 1)

# =============================================================================
# End of module scripts/validate_models.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
