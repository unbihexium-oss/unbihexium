# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/zoo/export.py
# Title       : ONNX export and verification of model zoo models
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch, onnx and onnxruntime
# =============================================================================
#
# Abstract
# --------
# Exports a ZooModel to ONNX (Open Neural Network Exchange) with a dynamic
# batch size, height and width, stores the model configuration in the ONNX
# metadata, and verifies the export by comparing ONNX Runtime outputs with
# the PyTorch outputs on a random input. ONNX models run without PyTorch,
# for example with unbihexium[onnx] in production services.
#
# The exported graph has one input named "input" of shape (N, C, H, W) and
# one output named "output" with the layout documented in
# unbihexium.ai.models.networks. Height and width should be multiples of
# 2**depth of the variant (8 for tiny, 16 for base and large, 32 for mega)
# for the best results; other sizes work but lose a few border pixels of
# context.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON encoding of the configuration in the ONNX metadata.
import json

# Suppress exporter warnings that are irrelevant for users.
import warnings

# Represent file paths.
from pathlib import Path

# Comparison of outputs.
import numpy as np

# Tracing and tensors.
import torch

# Model type.
from unbihexium.ai.models.factory import ZooModel

# ONNX operator set used for exports; supported by ONNX Runtime 1.19 and later.
OPSET = 18

# Metadata key holding the model configuration.
METADATA_KEY = "unbihexium_config"


# Raised when an exported model does not reproduce the PyTorch outputs.
class ExportError(RuntimeError):
    # No behaviour beyond RuntimeError; the class exists for precise handling.
    pass


# Spatial size used for tracing and verification of a model.
def example_size(model: ZooModel) -> int:
    # 2**depth pixels divide cleanly through every encoder level; use a
    # comfortable multiple of 64 that suits every variant.
    return 128


# Export a model to ONNX and return the path.
def export_onnx(
    model: ZooModel,  # Model to export.
    path: str | Path,  # Destination ONNX file.
    verify: bool = True,  # Compare ONNX Runtime with PyTorch.
    tolerance: float = 1e-3,  # Relative tolerance of the comparison.
) -> Path:  # Path of the written file.
    # Normalise the path.
    path = Path(path)
    # Create the parent directory if needed.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Export in evaluation mode on the CPU.
    model = model.eval().cpu()
    # Example input with the model's channel count.
    size = example_size(model)
    # Random example with a fixed seed so that exports are reproducible.
    generator = torch.Generator().manual_seed(0)
    # Values in [0, 1), like normalised reflectance.
    example = torch.rand(1, model.config.in_channels, size, size, generator=generator)
    # Batch, height and width are dynamic.
    dynamic_axes = {"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch"}}
    # Dense outputs also have dynamic spatial axes.
    if model.config.task.is_dense or model.output_stride > 1:
        # Output height and width follow the input.
        dynamic_axes["output"].update({2: "out_height", 3: "out_width"})
    # The exporter emits warnings about internal details; silence them.
    with warnings.catch_warnings():
        # Ignore all exporter warnings in this block.
        warnings.simplefilter("ignore")
        # Trace the model and write the ONNX file.
        torch.onnx.export(
            model,  # Model to export.
            (example,),  # Example input tuple.
            str(path),  # Output file.
            input_names=["input"],  # Name of the input.
            output_names=["output"],  # Name of the output.
            dynamic_axes=dynamic_axes,  # Dynamic dimensions.
            opset_version=OPSET,  # Operator set.
            dynamo=False,  # TorchScript-based exporter: stable graphs, no onnxscript needed.
        )  # End of the export call.
    # Store the model configuration in the ONNX metadata.
    _write_metadata(path, model)
    # Compare ONNX Runtime with PyTorch when requested.
    if verify:
        # Raises ExportError on a mismatch.
        verify_onnx(model, path, tolerance)
    # Return the written path.
    return path


# Add the model configuration to the metadata of an ONNX file.
def _write_metadata(path: Path, model: ZooModel) -> None:
    # onnx is imported lazily: it is only needed for export.
    import onnx

    # Load the exported graph.
    proto = onnx.load(str(path))
    # Metadata entry with the JSON configuration.
    entry = proto.metadata_props.add()
    # Metadata key.
    entry.key = METADATA_KEY
    # JSON value.
    entry.value = json.dumps(model.config.to_dict(), sort_keys=True)
    # Also record the weights digest.
    digest_entry = proto.metadata_props.add()
    # Metadata key of the digest.
    digest_entry.key = "unbihexium_weights_digest"
    # Digest of the exported weights.
    digest_entry.value = model.digest()
    # Write the graph back.
    onnx.save(proto, str(path))


# Compare ONNX Runtime and PyTorch outputs on a random input.
def verify_onnx(model: ZooModel, path: str | Path, tolerance: float = 1e-3) -> float:
    # onnxruntime is imported lazily: it is only needed for verification.
    import onnxruntime as ort

    # Use a size different from the tracing size to test dynamic axes.
    size = example_size(model) // 2 + 32
    # Deterministic random input.
    generator = torch.Generator().manual_seed(1)
    # Example input.
    example = torch.rand(2, model.config.in_channels, size, size, generator=generator)
    # PyTorch reference output.
    with torch.no_grad():
        # Run the model.
        expected = model.eval()(example).numpy()
    # ONNX Runtime session on the CPU.
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # ONNX Runtime output.
    actual = session.run(None, {"input": example.numpy()})[0]
    # Shapes must agree exactly.
    if actual.shape != expected.shape:
        # Report the shape mismatch.
        raise ExportError(f"{path}: ONNX output shape {actual.shape} != PyTorch {expected.shape}")
    # Largest absolute difference, ignoring NaN in both outputs.
    diff = float(np.nanmax(np.abs(actual - expected))) if np.isfinite(expected).any() else 0.0
    # Scale the tolerance by the magnitude of the outputs.
    scale = max(1.0, float(np.nanmax(np.abs(expected))) if np.isfinite(expected).any() else 1.0)
    # Fail when the difference is too large.
    if diff > tolerance * scale:
        # Report the numerical mismatch.
        raise ExportError(f"{path}: ONNX differs from PyTorch by {diff:.3g}")
    # Return the difference for logs.
    return diff


# Read the model configuration stored in an ONNX file.
def read_onnx_config(path: str | Path) -> dict[str, object]:
    # onnx is imported lazily.
    import onnx

    # Load only the model metadata and graph.
    proto = onnx.load(str(path), load_external_data=False)
    # Find the configuration entry.
    for entry in proto.metadata_props:
        # Match the metadata key.
        if entry.key == METADATA_KEY:
            # Decode the JSON configuration.
            return json.loads(entry.value)
    # Files not exported by Unbihexium have no configuration.
    raise ExportError(f"{path} has no Unbihexium configuration metadata")


# =============================================================================
# End of module src/unbihexium/zoo/export.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
