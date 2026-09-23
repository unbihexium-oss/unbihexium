# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : .github/scripts/check_model_zoo.py
# Title       : Model zoo consistency and reproducibility check
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires unbihexium and jsonschema;
#               --rebuild also requires PyTorch
# =============================================================================
#
# Abstract
# --------
# Checks the model zoo of the repository:
#
#   1. every generated file (digests.json, inventory, capability map,
#      manifests, cards, checksums) is in sync with catalog.yaml
#      (`python -m unbihexium.zoo.sync --check`);
#   2. every manifest validates against model_zoo/manifest.schema.json;
#   3. there is exactly one manifest and one card per model family, and no
#      stale files from removed families;
#   4. with --rebuild, the models of the selected variants are rebuilt and
#      their weights digests compared with the published ones, which proves
#      that the starter weights are reproducible on the CI platform.
#
# Usage
# -----
#   python .github/scripts/check_model_zoo.py
#   python .github/scripts/check_model_zoo.py --rebuild tiny
#   python .github/scripts/check_model_zoo.py --rebuild all
#
# Exit status
# -----------
#   0  the model zoo is consistent (and reproducible, with --rebuild)
#   1  at least one problem was found
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Command line parsing.
import argparse

# Parse manifests.
import json

# Set the exit status of the script.
import sys

# Represent file paths.
from pathlib import Path

# JSON Schema validation of the manifests.
import jsonschema

# Catalogue of the installed package.
from unbihexium.zoo.catalog import Variant, list_specs

# Generator of the model zoo files.
from unbihexium.zoo.sync import load_digests  # Published digests.
from unbihexium.zoo.sync import main as sync_main  # File synchronisation.

# Model zoo directory of the repository.
ROOT = Path("model_zoo")


# Print a problem as a GitHub Actions error annotation.
def error(path: Path | str, message: str) -> None:
    # The ::error:: prefix makes the message an annotation in the job log.
    print(f"::error file={path}::{message}")


# Validate the manifests and look for missing or stale files.
def check_files() -> int:
    # Number of problems.
    failures = 0
    # Schema of the manifests.
    schema = json.loads((ROOT / "manifest.schema.json").read_text(encoding="utf-8"))
    # Validator for the schema's draft.
    validator = jsonschema.validators.validator_for(schema)(schema)
    # Families of the catalogue.
    families = {spec.family for spec in list_specs()}
    # Families that have a manifest.
    manifests = {p.stem for p in (ROOT / "manifests").glob("*.json")}
    # Families that have a card.
    cards = {p.stem for p in (ROOT / "cards").glob("*.md")}
    # Every family needs a manifest and a card.
    for family in sorted(families - manifests):
        # Report the missing manifest.
        error(ROOT / "manifests", f"no manifest for {family}")
        # Count the problem.
        failures += 1
    # Every family needs a card.
    for family in sorted(families - cards):
        # Report the missing card.
        error(ROOT / "cards", f"no model card for {family}")
        # Count the problem.
        failures += 1
    # Files of families that no longer exist.
    for stale in sorted((manifests | cards) - families):
        # Report the stale file.
        error(ROOT, f"file for unknown family {stale}; remove it")
        # Count the problem.
        failures += 1
    # Validate every manifest against the schema.
    for path in sorted((ROOT / "manifests").glob("*.json")):
        # Parse the manifest.
        manifest = json.loads(path.read_text(encoding="utf-8"))
        # Report every schema violation.
        for problem in validator.iter_errors(manifest):
            # Location and message of the violation.
            error(path, f"{'/'.join(map(str, problem.path)) or '<root>'}: {problem.message}")
            # Count the problem.
            failures += 1
    # Return the number of problems.
    return failures


# Rebuild models and compare their digests with the published ones.
def check_rebuild(selection: str) -> int:
    # PyTorch is needed only for this check.
    from unbihexium.ai.models import build_model  # Builds models.

    # Published digests.
    digests = load_digests()
    # Variants to rebuild.
    variants = list(Variant) if selection == "all" else [Variant(selection)]
    # Number of problems and of rebuilt models.
    failures = checked = 0
    # Every family in the selected variants.
    for spec in list_specs():
        # Every selected variant.
        for variant in variants:
            # Build the starter model.
            model = build_model(spec.family, variant)
            # Count the model.
            checked += 1
            # Compare with the published digest.
            if model.digest() != digests.get(model.model_id, {}).get("weights_digest"):
                # Report the mismatch.
                error(
                    "src/unbihexium/zoo/digests.json",  # File with the published digests.
                    f"{model.model_id}: rebuilt digest differs",  # Message.
                )  # End of the report.
                # Count the problem.
                failures += 1
            # Free the memory before the next model.
            del model
    # Summary.
    print(f"Rebuilt {checked} models: {checked - failures} reproduce their published digest.")
    # Return the number of problems.
    return failures


# Run the checks and return the exit status.
def main(argv: list[str]) -> int:
    # Argument parser.
    parser = argparse.ArgumentParser(description="Check the model zoo")
    # Optional rebuild of one variant or all.
    parser.add_argument(
        "--rebuild",  # Option name.
        choices=["tiny", "base", "large", "mega", "all"],  # Variants or all.
        help="rebuild models and compare their digests",  # Help text.
    )  # End of the option.
    # Parse the arguments.
    args = parser.parse_args(argv)
    # Generated files must be in sync with the catalogue.
    failures = sync_main(["--root", ".", "--check"])
    # Manifests, cards and stale files.
    failures += check_files()
    # Reproducibility of the starter weights.
    if args.rebuild:
        # Rebuild and compare digests.
        failures += check_rebuild(args.rebuild)
    # Summary.
    print("Model zoo check passed." if not failures else f"{failures} model zoo problem(s) found.")
    # Non-zero exit status fails the CI job.
    return 1 if failures else 0


# Run the check when the file is executed as a script.
if __name__ == "__main__":
    # Pass the command line arguments without the script name.
    sys.exit(main(sys.argv[1:]))

# =============================================================================
# End of module .github/scripts/check_model_zoo.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
