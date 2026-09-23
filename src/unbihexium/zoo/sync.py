# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/zoo/sync.py
# Title       : Generate the model zoo digests, manifests and model cards
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch (unbihexium[torch])
# =============================================================================
#
# Abstract
# --------
# Keeps every derived file of the model zoo consistent with catalog.yaml:
#
#   src/unbihexium/zoo/digests.json      weights digest and parameter count of
#                                        all 520 models (packaged)
#   model_zoo/inventory.yaml             one entry per model family
#   model_zoo/capability_to_models.yaml  family -> model ids
#   model_zoo/manifests/<family>.json    manifest per family (see schema)
#   model_zoo/cards/<family>.md          model card per family
#   model_zoo/checksums.txt              "<digest>  <model id>" per model
#   model_zoo/MODEL_CARDS.md             index of the model cards
#
# The digests are computed by building every model with its deterministic
# starter weights, which takes a few minutes on a CPU. In check mode nothing
# is written; the command fails when a file is missing or out of date, which
# lets CI detect a catalogue change that was not followed by a sync.
#
# Usage
# -----
#   python -m unbihexium.zoo.sync --root .          # regenerate everything
#   python -m unbihexium.zoo.sync --root . --check  # verify only
#   python -m unbihexium.zoo.sync --root . --skip-digests  # reuse digests
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Command line parsing.
import argparse

# JSON output.
import json

# Parse the lines of the generated YAML.
import re

# Exit status.
import sys

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# YAML output.
import yaml

# Catalogue types and lookups.
from unbihexium.zoo.catalog import (
    MODEL_LICENSE,  # Licence of the models.
    ModelSpec,  # Catalogue entry of a family.
    Task,  # Task enumeration.
    Variant,  # Size variant enumeration.
    all_model_ids,  # Every model id of the zoo.
    catalog_version,  # Version of the catalogue.
    get_variant,  # Hyperparameters of a variant.
    list_specs,  # All model families.
)  # End of the catalogue imports.

# Path of the digest table inside the package.
DIGESTS_PATH = Path(__file__).with_name("digests.json")

# HTML comment placed at the top of generated Markdown files.
GENERATED_NOTE = (
    "<!-- Generated from src/unbihexium/zoo/catalog.yaml "  # Source of the content.
    "by `python -m unbihexium.zoo.sync`. -->"  # Generator.
)  # End of the note.

# Description of the digest algorithm, recorded in digests.json.
DIGEST_ALGORITHM = "sha256 over sorted state dict entries: key, shape and float32 LE bytes"

# Description of the initialisation, recorded in digests.json.
INIT_METHOD = "numpy.random.RandomState(seed), seed = first 4 bytes of sha256('unbihexium:' + id)"

# Architecture name of each task, used in manifests and cards.
ARCHITECTURES = {
    Task.DETECTION: "centernet",  # Anchor-free detector.
    Task.SEGMENTATION: "unet",  # Encoder-decoder.
    Task.CHANGE_DETECTION: "unet_early_fusion",  # U-Net on stacked dates.
    Task.DENSE_REGRESSION: "unet_regression",  # U-Net with regression output.
    Task.SCENE_REGRESSION: "encoder_regressor",  # Pooled encoder.
    Task.ENHANCEMENT: "unet_image_to_image",  # U-Net image translation.
    Task.SUPER_RESOLUTION: "residual_subpixel",  # EDSR-style network.
    Task.SPECTRAL_INDEX: "spectral_formula",  # Exact formula.
}  # End of the architecture table.

# One-sentence explanation of the output layout of each task.
OUTPUT_LAYOUT = {
    Task.DETECTION: (  # Detector outputs.
        "Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in "
        "output-stride pixels, and the x and y centre offsets."
    ),  # End of the detection layout.
    Task.SEGMENTATION: "Tensor (N, K, H, W) of class logits; apply softmax over K.",  # Classes.
    Task.CHANGE_DETECTION: "Tensor (N, K, H, W) of change class logits; softmax over K.",  # Change.
    Task.DENSE_REGRESSION: "Tensor (N, K, H, W) of target values in the listed units.",  # Values.
    Task.SCENE_REGRESSION: "Tensor (N, K) with one value per target and chip.",  # Vector.
    Task.ENHANCEMENT: "Tensor (N, K, H, W) of output bands or displacement components.",  # Images.
    Task.SUPER_RESOLUTION: "Tensor (N, K, sH, sW) at s times the input resolution.",  # Upscaled.
    Task.SPECTRAL_INDEX: "Tensor (N, 1, H, W) with the index value; NaN where undefined.",  # Index.
}  # End of the output layout table.

# Rule line of the header and footer blocks of generated text files.
RULE = "# " + "=" * 77

# Explanation of the keys of the generated YAML files, used as comments.
KEY_COMMENTS = {
    "version": "Catalogue version the file was generated from.",  # Version.
    "families": "Number of model families.",  # Family count.
    "models": "Number of models: four size variants per family.",  # Model count.
    "entries": "One summary per model family, in catalogue order.",  # Entry list.
    "family": "Family identifier; the model ids append the variant.",  # Family id.
    "name": "Human-readable name.",  # Name.
    "task": "Task, which fixes the input and output layout.",  # Task.
    "domain": "Capability domain of the family.",  # Domain.
    "architecture": "Network architecture of the task.",  # Architecture.
    "in_channels": "Number of input bands the model expects.",  # Input bands.
    "outputs": "Output classes, targets or bands, in channel order.",  # Outputs.
    "license": "Licence of the model weights.",  # Licence.
    "status": "starter: untrained weights; reference: exact formula.",  # Status.
    "mappings": "One mapping per model family, in catalogue order.",  # Mapping list.
    "primary_model_id": "Default model of the family (base variant).",  # Default.
    "model_ids": "Models of the family: tiny, base, large and mega.",  # Model ids.
}  # End of the key comments.

# Comments of list items, keyed by the key of the list.
ITEM_COMMENTS = {
    "outputs": "Output channel {n}.",  # Output channel.
    "model_ids": "Variant {n} of the family.",  # Variant.
}  # End of the item comments.

# Indentation, list dash and key of a YAML line written by yaml.safe_dump.
YAML_LINE = re.compile(r"^(?P<indent>\s*)(?P<dash>- )?(?:(?P<key>[\w.-]+):(?: |$))?")


# Header block of a generated text file.
def generated_header(path: str, title: str, fmt: str, abstract: list[str]) -> str:
    # Lines of the block.
    lines = [
        "# This Source Code Form is subject to the terms of the Mozilla Public",  # Notice.
        "# License, v. 2.0. If a copy of the MPL was not distributed with this",  # Notice.
        "# file, You can obtain one at https://mozilla.org/MPL/2.0/.",  # Notice.
        "#",  # Separator.
        RULE,  # Opening rule.
        "# Project     : Unbihexium",  # Project.
        f"# File        : {path}",  # Repository path.
        f"# Title       : {title}",  # Title.
        "# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>",  # Author.
        "# Affiliation : University of Helsinki",  # Affiliation.
        "# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors",  # Copyright.
        "# Licence     : Mozilla Public License 2.0, see LICENSE.txt",  # Licence.
        f"# Format      : {fmt}",  # Format.
        RULE,  # Closing rule of the fields.
        "#",  # Separator.
        "# Abstract",  # Section title.
        "# --------",  # Underline.
        *[f"# {line}".rstrip() for line in abstract],  # Abstract text.
        RULE,  # Closing rule.
    ]  # End of the lines.
    # One line each, and a blank line before the data.
    return "\n".join(lines) + "\n\n"


# Footer block of a generated text file.
def generated_footer(path: str) -> str:
    # Lines of the block.
    lines = [
        "",  # Blank line after the data.
        RULE,  # Opening rule.
        f"# End of file {path}",  # Closing line.
        "# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).",  # Project.
        "# Cite the project as described in CITATION.cff.",  # Citation.
        RULE,  # Closing rule.
    ]  # End of the lines.
    # One line each.
    return "\n".join(lines) + "\n"


# Add an explanatory comment to every line of YAML written by yaml.safe_dump.
def annotate_yaml(text: str) -> str:
    # Keys of the enclosing mappings with their columns.
    parents: list[tuple[int, str]] = []
    # Item counters of the open lists, by key.
    counters: dict[str, int] = {}
    # Annotated lines.
    out = []
    # Walk over the lines.
    for line in text.splitlines():
        # Parse the line; the pattern matches every line.
        match = YAML_LINE.match(line)
        # Column of the key or item.
        column = len(match.group("indent")) + (2 if match.group("dash") else 0)
        # Close the mappings that this line leaves.
        while parents and parents[-1][0] >= column:
            # Leave the mapping.
            parents.pop()
        # Key of the line, if any.
        key = match.group("key")
        # Key of the enclosing mapping or list.
        parent = parents[-1][1] if parents else ""
        # A plain list item.
        if match.group("dash") and not key:
            # Number the items of the list.
            counters[parent] = counters.get(parent, 0) + 1
            # Comment of the item.
            comment = ITEM_COMMENTS.get(parent, "List item {n}.").format(n=counters[parent])
        # A known key.
        elif key in KEY_COMMENTS:
            # Explanation of the key.
            comment = KEY_COMMENTS[key]
        # A family key of the capability map.
        elif parent == "mappings":
            # Name the family.
            comment = f"Mapping of the {key} family."
        # Anything else.
        else:
            # Generic explanation.
            comment = "Generated value."
        # A key may open a nested mapping or list.
        if key:
            # Remember it.
            parents.append((column, key))
            # Its items are numbered from one.
            counters[key] = 0
        # Line with its comment.
        out.append(f"{line}  # {comment}")
    # Joined lines with a final newline.
    return "\n".join(out) + "\n"


# Compute the digest table by building every model.
def compute_digests(verbose: bool = True) -> dict[str, dict[str, Any]]:
    # PyTorch-dependent imports are local so that --check can run without them.
    from unbihexium.ai.models.factory import build_model  # Builds models.

    # Digest and parameter count per model id.
    table: dict[str, dict[str, Any]] = {}
    # All families and variants.
    for spec in list_specs():
        # Every size variant of the family.
        for variant in Variant:
            # Build the starter model.
            model = build_model(spec.family, variant)
            # Record digest and parameter count.
            table[model.model_id] = {
                "weights_digest": model.digest(),  # SHA-256 of the weights.
                "num_parameters": model.num_parameters(),  # Trainable parameters.
            }  # End of the record.
            # Report progress.
            if verbose:
                # One line per model.
                print(f"{model.model_id}: {model.num_parameters():,} parameters", flush=True)
            # Free the memory of large variants before building the next one.
            del model
    # Return the table.
    return table


# Load the current digest table, or an empty one.
def load_digests() -> dict[str, dict[str, Any]]:
    # A missing file yields an empty table.
    if not DIGESTS_PATH.is_file():
        # No digests yet.
        return {}
    # Parse the JSON file.
    return json.loads(DIGESTS_PATH.read_text(encoding="utf-8")).get("models", {})


# Render the digest table as JSON text.
def render_digests(table: dict[str, dict[str, Any]]) -> str:
    # Document with metadata and models sorted by id.
    document = {
        "catalog_version": catalog_version(),  # Catalogue the digests belong to.
        "algorithm": DIGEST_ALGORITHM,  # How the digests are computed.
        "initialisation": INIT_METHOD,  # How the weights are initialised.
        "models": {k: table[k] for k in sorted(table)},  # Digests per model.
    }  # End of the document.
    # Pretty-printed JSON with a final newline.
    return json.dumps(document, indent=2, sort_keys=False) + "\n"


# Render the inventory of model families as YAML text.
def render_inventory(specs: list[ModelSpec]) -> str:
    # Header block.
    header = generated_header(
        "model_zoo/inventory.yaml",  # Repository path.
        "Inventory of the model zoo families",  # Title.
        "YAML, generated by python -m unbihexium.zoo.sync",  # Format.
        [  # Abstract lines.
            "One summary per model family of the Unbihexium model zoo: task,",  # Abstract.
            "architecture, input bands, outputs, licence and status. Every family",  # Abstract.
            "has four size variants (tiny, base, large and mega). Starter models",  # Abstract.
            "have untrained weights and must be trained before use; reference",  # Abstract.
            "models compute an exact spectral index formula.",  # Abstract.
            "",  # Blank line.
            "Generated from src/unbihexium/zoo/catalog.yaml by",  # Abstract.
            "`python -m unbihexium.zoo.sync`. Do not edit by hand.",  # Abstract.
        ],  # End of the abstract.
    )  # End of the header.
    # One summary entry per family.
    data = {
        "version": catalog_version(),  # Catalogue version.
        "families": len(specs),  # Number of families.
        "models": len(specs) * len(Variant),  # Number of models.
        "entries": [  # Family summaries.
            {  # Summary of one family.
                "family": s.family,  # Family id.
                "name": s.name,  # Human-readable name.
                "task": s.task.value,  # Task.
                "domain": s.domain,  # Capability domain.
                "architecture": ARCHITECTURES[s.task],  # Architecture.
                "in_channels": s.in_channels,  # Input channels.
                "outputs": list(s.outputs),  # Outputs.
                "license": MODEL_LICENSE,  # Licence.
                "status": "reference" if not s.task.is_trainable else "starter",  # Status.
            }  # End of the summary.
            for s in specs  # One per family.
        ],  # End of the entry list.
    }  # End of the data.
    # YAML without reordering keys, with a comment on every line.
    body = annotate_yaml(yaml.safe_dump(data, sort_keys=False, allow_unicode=False, width=100))
    # Header, data and footer.
    return header + body + generated_footer("model_zoo/inventory.yaml")


# Render the mapping of families to model ids as YAML text.
def render_capability_map(specs: list[ModelSpec]) -> str:
    # Header block.
    header = generated_header(
        "model_zoo/capability_to_models.yaml",  # Repository path.
        "Mapping of the model zoo capabilities to model ids",  # Title.
        "YAML, generated by python -m unbihexium.zoo.sync",  # Format.
        [  # Abstract lines.
            "Maps every model family (capability) to its four model ids and names",  # Abstract.
            "the default model, the base variant. Tools that select a model for a",  # Abstract.
            "capability read this file.",  # Abstract.
            "",  # Blank line.
            "Generated from src/unbihexium/zoo/catalog.yaml by",  # Abstract.
            "`python -m unbihexium.zoo.sync`. Do not edit by hand.",  # Abstract.
        ],  # End of the abstract.
    )  # End of the header.
    # Family -> description and model ids.
    data = {
        "version": catalog_version(),  # Catalogue version.
        "mappings": {  # One mapping per family.
            s.family: {  # Mapping of one family.
                "name": s.name,  # Human-readable name.
                "task": s.task.value,  # Task.
                "primary_model_id": s.model_id(Variant.BASE),  # Default model.
                "model_ids": [s.model_id(v) for v in Variant],  # All variants.
            }  # End of the mapping.
            for s in specs  # One per family.
        },  # End of the mappings.
    }  # End of the data.
    # YAML without reordering keys, with a comment on every line.
    body = annotate_yaml(yaml.safe_dump(data, sort_keys=False, allow_unicode=False, width=100))
    # Header, data and footer.
    return header + body + generated_footer("model_zoo/capability_to_models.yaml")


# Digest record of one model, or an empty record.
def _record(
    digests: dict[str, dict[str, Any]],  # Digest table.
    spec: ModelSpec,  # Model family.
    variant: Variant,  # Size variant.
) -> dict[str, Any]:  # Record of the model.
    # Look the model id up in the digest table.
    return digests.get(spec.model_id(variant), {})


# Render the manifest of one family as JSON text.
def render_manifest(spec: ModelSpec, digests: dict[str, dict[str, Any]]) -> str:
    # Manifest document.
    document = {
        "$schema": "../manifest.schema.json",  # Schema of the manifest.
        "family": spec.family,  # Family id.
        "name": spec.name,  # Human-readable name.
        "version": catalog_version(),  # Catalogue version.
        "task": spec.task.value,  # Task.
        "domain": spec.domain,  # Capability domain.
        "architecture": ARCHITECTURES[spec.task],  # Architecture.
        "license": MODEL_LICENSE,  # Licence.
        "status": "reference" if not spec.task.is_trainable else "starter",  # Status.
        "trained": False,  # No model of the zoo is trained on EO data.
        "description": spec.description,  # Purpose.
        "inputs": {  # Input description.
            "bands": list(spec.bands),  # Bands of one acquisition.
            "dates": spec.dates,  # Acquisitions.
            "channels": list(spec.channel_names),  # All input channels.
        },  # End of the inputs.
        "outputs": {  # Output description.
            "names": list(spec.outputs),  # Classes, targets or bands.
            "units": list(spec.units),  # Target units.
            "range": list(spec.value_range) if spec.value_range else None,  # Bounds.
            "scale": spec.scale,  # Upscaling factor.
            "formula": spec.formula,  # Index formula.
            "layout": OUTPUT_LAYOUT[spec.task],  # Tensor layout.
        },  # End of the outputs.
        "training_labels": spec.labels,  # Reference data for training.
        "data_sources": list(spec.sources),  # Suitable input data.
        "variants": {  # One record per size variant.
            v.value: {  # Record of one variant.
                "model_id": spec.model_id(v),  # Model id.
                "tile_size": get_variant(v).tile_size,  # Recommended tile size.
                "base_channels": get_variant(v).base_channels,  # Encoder width.
                "depth": get_variant(v).depth,  # Encoder depth.
                "num_parameters": _record(digests, spec, v).get("num_parameters", 0),  # Size.
                "weights_digest": _record(digests, spec, v).get("weights_digest", ""),  # Digest.
            }  # End of the variant record.
            for v in Variant  # All variants.
        },  # End of the variants.
    }  # End of the document.
    # Pretty-printed JSON with a final newline.
    return json.dumps(document, indent=2) + "\n"


# Render the model card of one family as Markdown text.
def render_card(spec: ModelSpec, digests: dict[str, dict[str, Any]]) -> str:
    # Whether the family needs training before use.
    trainable = spec.task.is_trainable
    # Status sentence shown at the top of the card.
    status = (
        "Starter model: complete architecture with deterministic starter weights, "
        "**not trained**. Train or fine-tune it on labelled data before use."
        if trainable  # Learned models.
        else "Reference implementation of a published formula; no training needed."  # Formulas.
    )  # End of the status sentence.
    # Card lines.
    lines = [
        f"# {spec.name}",  # Title.
        "",  # Blank line.
        GENERATED_NOTE,  # Generator note.
        "",  # Blank line.
        f"> {status}",  # Status block quote.
        "",  # Blank line.
        "## Overview",  # Section heading.
        "",  # Blank line.
        "| Property | Value |",  # Table header.
        "| --- | --- |",  # Table separator.
        f"| Family | `{spec.family}` |",  # Family id.
        f"| Task | {spec.task.value} |",  # Task.
        f"| Domain | {spec.domain} |",  # Domain.
        f"| Architecture | {ARCHITECTURES[spec.task]} |",  # Architecture.
        f"| Licence | {MODEL_LICENSE} |",  # Licence.
        f"| Trained on Earth observation data | {'No' if trainable else 'Not applicable'} |",
        "",  # Blank line.
        spec.description,  # Description.
        "",  # Blank line.
        "## Inputs",  # Section heading.
        "",  # Blank line.
        f"{spec.in_channels} channels, float32, shape (N, {spec.in_channels}, H, W):",  # Summary.
        "",  # Blank line.
        *[f"{i + 1}. `{name}`" for i, name in enumerate(spec.channel_names)],  # Channel list.
        "",  # Blank line.
        "## Outputs",  # Section heading.
        "",  # Blank line.
        OUTPUT_LAYOUT[spec.task],  # Tensor layout.
        "",  # Blank line.
        "| Index | Name | Unit |",  # Table header.
        "| --- | --- | --- |",  # Table separator.
        *[  # One row per output.
            f"| {i} | `{name}` | {spec.units[i] if spec.units else '-'} |"
            for i, name in enumerate(spec.outputs)  # Outputs with their index.
        ],  # End of the rows.
        "",  # Blank line.
    ]  # End of the first part.
    # Value range of regression outputs.
    if spec.value_range:
        # One sentence about the range.
        lines += [f"Outputs are bounded to {list(spec.value_range)}.", ""]
    # Variants table.
    lines += [
        "## Variants",  # Section heading.
        "",  # Blank line.
        "| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |",  # Header.
        "| --- | --- | --- | --- |",  # Separator.
    ]  # End of the table header.
    # One row per variant.
    for v in Variant:
        # Record of the variant.
        record = digests.get(spec.model_id(v), {})
        # Table row.
        lines.append(
            f"| `{spec.model_id(v)}` | {record.get('num_parameters', 0):,} | "
            f"{get_variant(v).tile_size} | `{record.get('weights_digest', '')[:16]}` |"
        )  # End of the row.
    # Usage section.
    lines += [
        "",  # Blank line.
        "## Usage",  # Section heading.
        "",  # Blank line.
        "```python",  # Code block start.
        "from unbihexium.zoo import load_model",  # Import.
        "",  # Blank line.
        f'model = load_model("{spec.model_id(Variant.BASE)}")  # verified starter weights',
        "```",  # Code block end.
        "",  # Blank line.
    ]  # End of the usage section.
    # Training section for learned models.
    if trainable:
        # Instructions and data requirements.
        lines += [
            "## Training",  # Section heading.
            "",  # Blank line.
            f"Required reference data: {spec.labels}",  # Labels.
            "",  # Blank line.
            "```bash",  # Code block start.
            f"unbihexium train {spec.model_id(Variant.BASE)} --data path/to/dataset --epochs 50",
            "```",  # Code block end.
            "",  # Blank line.
            "See docs/model_zoo/training.md for the dataset layout.",  # Pointer.
            "",  # Blank line.
        ]  # End of the training section.
    # Data sources.
    lines += ["## Suitable data", ""] + [f"- {s}" for s in spec.sources] + [""]
    # Limitations and responsible use.
    lines += [
        "## Limitations and responsible use",  # Section heading.
        "",  # Blank line.
        (  # Limitation paragraph.
            "The starter weights produce meaningless predictions until the model is trained. "
            "After training, validate the model on independent reference data for your area, "
            "sensor and season, and report its accuracy with the results."
            if trainable  # Learned models.
            else "The index is only meaningful for reflectance of the listed bands."  # Index.
        ),  # End of the paragraph.
        "",  # Blank line.
        "Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.",
    ]  # End of the section.
    # Join with newlines and end with a newline.
    return "\n".join(lines) + "\n"


# Render the index of all model cards as Markdown text.
def render_card_index(specs: list[ModelSpec], digests: dict[str, dict[str, Any]]) -> str:
    # Total number of parameters over all 520 models.
    total = sum(record.get("num_parameters", 0) for record in digests.values())
    # Header and introduction.
    lines = [
        "# Model Cards",  # Title.
        "",  # Blank line.
        GENERATED_NOTE,  # Generator note.
        "",  # Blank line.
        (  # Introduction paragraph.
            f"The model zoo has {len(specs)} model families in four size variants "
            f"({len(specs) * len(Variant)} models, {total:,} parameters in total). "
            "Every learned model is a starter model: a complete, trainable architecture with "
            "deterministic starter weights that has not been trained on Earth observation data. "
            "Train or fine-tune a model before using its predictions. The spectral index "
            "family implements published formulas and needs no training."
        ),  # End of the paragraph.
        "",  # Blank line.
        "| Family | Task | Domain | Inputs | Outputs | Status |",  # Table header.
        "| --- | --- | --- | --- | --- | --- |",  # Table separator.
    ]  # End of the header.
    # One row per family, linking to its card.
    for s in specs:
        # Status text.
        status = "starter" if s.task.is_trainable else "reference formula"
        # Table row.
        lines.append(
            f"| [{s.name}](cards/{s.family}.md) | {s.task.value} | {s.domain} | "
            f"{s.in_channels} | {s.out_channels} | {status} |"
        )  # End of the row.
    # Join with newlines and end with a newline.
    return "\n".join(lines) + "\n"


# Render the checksum list of all models.
def render_checksums(digests: dict[str, dict[str, Any]]) -> str:
    # Header block.
    header = generated_header(
        "model_zoo/checksums.txt",  # Repository path.
        "Weights digests of the model zoo",  # Title.
        "Text, one '<digest>  <model id>  # comment' line per model",  # Format.
        [  # Abstract lines.
            "SHA-256 weights digest of every model of the zoo, sorted by model id.",  # Abstract.
            "The digest covers the sorted state dict entries (key, shape and",  # Abstract.
            "float32 little-endian bytes), so it does not depend on the file format",  # Abstract.
            "of a checkpoint. `unbihexium zoo verify` checks a cached model against",  # Abstract.
            "the same digests, which are packaged in src/unbihexium/zoo/digests.json.",  # Abstract.
            "The comment of each line gives the number of parameters of the model.",  # Abstract.
            "",  # Blank line.
            "Generated by `python -m unbihexium.zoo.sync`. Do not edit by hand.",  # Abstract.
        ],  # End of the abstract.
    )  # End of the header.
    # One line per model, sorted by id, with the parameter count as comment.
    lines = [
        f"{d['weights_digest']}  {k}  # {d['num_parameters']:,} parameters.\n"  # Entry.
        for k, d in sorted(digests.items())  # Every model.
    ]  # End of the lines.
    # Header, entries and footer.
    return header + "".join(lines) + generated_footer("model_zoo/checksums.txt")


# All generated files: path relative to the repository root -> content.
def generated_files(root: Path, digests: dict[str, dict[str, Any]]) -> dict[Path, str]:
    # Model families in catalogue order.
    specs = list_specs()
    # Model zoo directory of the repository.
    zoo = root / "model_zoo"
    # Files that do not depend on the family.
    files = {
        root / "src" / "unbihexium" / "zoo" / "digests.json": render_digests(digests),  # Digests.
        zoo / "inventory.yaml": render_inventory(specs),  # Inventory.
        zoo / "capability_to_models.yaml": render_capability_map(specs),  # Mapping.
        zoo / "checksums.txt": render_checksums(digests),  # Checksums.
        zoo / "MODEL_CARDS.md": render_card_index(specs, digests),  # Card index.
    }  # End of the fixed files.
    # One manifest and one card per family.
    for spec in specs:
        # Manifest.
        files[zoo / "manifests" / f"{spec.family}.json"] = render_manifest(spec, digests)
        # Model card.
        files[zoo / "cards" / f"{spec.family}.md"] = render_card(spec, digests)
    # Return the mapping.
    return files


# Whether a file is missing or has content different from text.
def _differs(path: Path, text: str) -> bool:
    # Missing files and different content both need a write.
    return not path.is_file() or path.read_text(encoding="utf-8") != text


# Command line entry point.
def main(argv: list[str] | None = None) -> int:
    # Argument parser.
    parser = argparse.ArgumentParser(description="Synchronise the model zoo with the catalogue")
    # Repository root.
    parser.add_argument("--root", default=".", help="repository root")
    # Check mode.
    parser.add_argument("--check", action="store_true", help="verify only, write nothing")
    # Reuse the existing digests instead of rebuilding all models.
    parser.add_argument("--skip-digests", action="store_true", help="reuse digests.json")
    # Parse the arguments.
    args = parser.parse_args(argv)
    # Repository root.
    root = Path(args.root).resolve()
    # Digests: existing ones for checks and skips, recomputed otherwise.
    digests = load_digests() if (args.check or args.skip_digests) else compute_digests()
    # Every catalogue model needs a digest.
    missing = [mid for mid in all_model_ids() if mid not in digests]
    # Report missing digests.
    if missing:
        # Explain how to fix it.
        print(f"{len(missing)} models have no digest; run without --skip-digests/--check")
        # Fail.
        return 1
    # Files that should exist with their content.
    files = generated_files(root, digests)
    # Files that differ from their expected content.
    stale = [p for p, text in files.items() if _differs(p, text)]
    # Check mode reports and never writes.
    if args.check:
        # One line per stale file.
        for path in stale:
            # Relative path for readability.
            print(f"::error file={path.relative_to(root)}::out of date; run the model zoo sync")
        # Summary.
        print(f"{len(files)} model zoo files checked, {len(stale)} out of date.")
        # Fail when any file is stale.
        return 1 if stale else 0
    # Write the stale files.
    for path in stale:
        # Create the parent directory.
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write the content.
        path.write_text(files[path], encoding="utf-8")
    # Summary.
    print(f"{len(files)} model zoo files, {len(stale)} written.")
    # Success.
    return 0


# Run the command when the module is executed.
if __name__ == "__main__":
    # Exit with the status returned by main().
    sys.exit(main())

# =============================================================================
# End of module src/unbihexium/zoo/sync.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
