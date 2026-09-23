# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_models.py
# Title       : Tests of the model zoo network architectures
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest and PyTorch
# =============================================================================
#
# Abstract
# --------
# Builds every model family in its tiny variant and checks the output shape
# for its task, the determinism and published digests of the starter
# weights, customised inputs and outputs, value ranges, the exact spectral
# index formulas, and that every architecture can learn (one optimisation
# step reduces the loss on a fixed batch).
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Numerical reference values.
import numpy as np

# Test framework.
import pytest

# PyTorch is optional for the library; skip these tests without it.
torch = pytest.importorskip("torch")

# Model construction under test.
from unbihexium.ai.models import (  # noqa: E402 - imported after the skip check
    DETECTION_STRIDE,  # Output stride of the detector.
    SpectralIndex,  # Formula modules.
    build_model,  # Model factory.
    weights_digest,  # Digest function.
)  # End of the model imports.

# Catalogue and registry.
from unbihexium.zoo import Task, get_model, list_specs  # noqa: E402 - after the skip check


# Expected output shape of a model for an input of size (1, C, size, size).
def expected_shape(spec, size: int) -> tuple[int, ...]:
    # Number of outputs.
    k = spec.out_channels
    # Detectors predict at stride 4 with 4 extra channels.
    if spec.task is Task.DETECTION:
        # Heatmaps, size and offset.
        return (1, k + 4, size // DETECTION_STRIDE, size // DETECTION_STRIDE)
    # Scene regressors predict one vector.
    if spec.task is Task.SCENE_REGRESSION:
        # One value per target.
        return (1, k)
    # Super-resolution scales the spatial size.
    if spec.task is Task.SUPER_RESOLUTION:
        # Upscaled output.
        return (1, k, size * spec.scale, size * spec.scale)
    # Dense tasks keep the spatial size.
    return (1, k, size, size)


# Every family builds and produces the output shape of its task.
@pytest.mark.parametrize("spec", list_specs(), ids=lambda s: s.family)
def test_forward_shape(spec) -> None:
    # Tiny variant for speed.
    model = build_model(spec.family, "tiny")
    # Input size divisible by 2**depth of the tiny variant.
    size = 32
    # Random input with the family's channel count.
    x = torch.rand(1, spec.in_channels, size, size)
    # Inference without gradients.
    with torch.no_grad():
        # Forward pass.
        y = model(x)
    # Shape of the task.
    assert tuple(y.shape) == expected_shape(spec, size)
    # Outputs are finite for positive reflectance inputs.
    assert torch.isfinite(y).all()


# Non-square inputs whose size is not a multiple of 2**depth still work.
def test_odd_input_size() -> None:
    # U-Net model.
    model = build_model("lulc_classifier", "tiny")
    # Awkward input size.
    x = torch.rand(1, model.config.in_channels, 37, 50)
    # Forward pass.
    with torch.no_grad():
        # Output keeps the input size.
        assert model(x).shape[-2:] == (37, 50)


# Starter weights are deterministic and match the published digests.
@pytest.mark.parametrize(
    "model_id", ["ship_detector_tiny", "lulc_classifier_base", "yield_predictor_tiny"]
)
def test_digest_matches_published(model_id: str) -> None:
    # Build twice.
    a = build_model(model_id)
    # Second build.
    b = build_model(model_id)
    # Identical weights.
    assert a.digest() == b.digest()
    # The digest equals the published one.
    assert a.digest() == get_model(model_id).weights_digest
    # Different models have different weights.
    assert a.digest() != build_model("vehicle_detector_tiny").digest()


# Customised inputs and outputs change the architecture accordingly.
def test_customised_model() -> None:
    # Four input bands and two classes instead of the catalogue layout.
    model = build_model(
        "lulc_classifier",  # Family.
        "tiny",  # Variant.
        channel_names=["b", "g", "r", "n"],  # Four bands instead of ten.
        outputs=["water", "land"],  # Two classes instead of eleven.
    )  # End of the build call.
    # The configuration records the customisation.
    assert model.config.customised and model.config.in_channels == 4
    # Output channels follow the new classes.
    with torch.no_grad():
        # Forward pass with four bands.
        assert model(torch.rand(2, 4, 32, 32)).shape == (2, 2, 32, 32)


# Bounded regression outputs stay inside their range.
def test_value_range() -> None:
    # Canopy height is bounded to [0, 60] m.
    model = build_model("tree_height_estimator", "tiny")
    # Large inputs push the network towards its limits.
    x = torch.randn(1, model.config.in_channels, 32, 32) * 100
    # Forward pass.
    with torch.no_grad():
        # Output.
        y = model(x)
    # Inside the interval.
    assert float(y.min()) >= 0.0 and float(y.max()) <= 60.0


# Spectral index modules reproduce the published formulas exactly.
@pytest.mark.parametrize(
    ("formula", "bands", "expected"),  # Parameter names.
    [  # Cases: formula, band values, expected index value.
        ("ndvi", [0.1, 0.5], (0.5 - 0.1) / (0.5 + 0.1)),  # red, nir
        ("ndwi", [0.3, 0.1], (0.3 - 0.1) / (0.3 + 0.1)),  # green, nir
        ("evi", [0.05, 0.1, 0.5], 2.5 * 0.4 / (0.5 + 0.6 - 0.375 + 1.0)),  # blue, red, nir
        ("savi", [0.1, 0.5], 1.5 * 0.4 / (0.5 + 0.1 + 0.5)),  # red, nir
        ("msi", [0.5, 0.25], 0.25 / 0.5),  # nir, swir1
        ("nbr", [0.5, 0.2], (0.5 - 0.2) / (0.5 + 0.2)),  # nir, swir2
        ("vci", [0.4, 0.2, 0.8], (0.4 - 0.2) / (0.8 - 0.2)),  # ndvi, min, max
    ],  # End of the cases.
)  # End of the parametrisation.
def test_spectral_formulas(formula: str, bands: list[float], expected: float) -> None:
    # Module for the formula.
    module = SpectralIndex(formula)
    # One pixel with the given band values.
    x = torch.tensor(bands, dtype=torch.float32).view(1, -1, 1, 1)
    # Computed value.
    value = float(module(x))
    # Matches the formula to float32 precision.
    assert value == pytest.approx(expected, rel=1e-6)


# Zero denominators yield NaN instead of infinities.
def test_spectral_nan() -> None:
    # NDVI of a black pixel is undefined.
    value = SpectralIndex("ndvi")(torch.zeros(1, 2, 1, 1))
    # NaN marks the undefined pixel.
    assert torch.isnan(value).all()


# Every task architecture learns: one optimisation step lowers the loss.
@pytest.mark.parametrize(
    "family",  # Parameter name.
    [  # One family per architecture.
        "ship_detector",
        "lulc_classifier",
        "change_detector",
        "tree_height_estimator",
        "yield_predictor",
        "pansharpening",
        "super_resolution",
    ],  # One family per architecture.
)  # End of the parametrisation.
def test_architecture_learns(family: str) -> None:
    # Fixed seed for a reproducible batch.
    torch.manual_seed(0)
    # Tiny model in training mode.
    model = build_model(family, "tiny").train()
    # Fixed input batch.
    x = torch.rand(2, model.config.in_channels, 32, 32)
    # Fixed random target with the output shape.
    with torch.no_grad():
        # Target shaped like the output.
        target = torch.rand_like(model(x))
    # Plain SGD with a small learning rate.
    optimiser = torch.optim.SGD(model.parameters(), lr=1e-2)
    # Loss before the step.
    loss_before = torch.nn.functional.mse_loss(model(x), target)
    # Gradient step.
    loss_before.backward()
    # Update the weights.
    optimiser.step()
    # Loss after the step.
    with torch.no_grad():
        # Same loss on the same batch.
        loss_after = torch.nn.functional.mse_loss(model(x), target)
    # The loss decreased.
    assert float(loss_after) < float(loss_before.detach())


# The digest depends only on the weights, not on the module object.
def test_digest_of_state_dict() -> None:
    # Any model.
    model = build_model("ndvi_calculator", "tiny")
    # Digest of the module and of its state dict agree.
    assert weights_digest(model) == weights_digest(model.state_dict())
    # Parameter-free modules have the digest of an empty state.
    assert np.isclose(model.num_parameters(), 0)


# =============================================================================
# End of module tests/unit/test_models.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
