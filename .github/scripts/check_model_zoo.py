# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : .github/scripts/check_model_zoo.py
# Title       : Model zoo structure and integrity check
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
# Checks the structure and integrity of the model zoo. For every model variant
# under model_zoo/assets/<tier>/<name>/ the script verifies that:
#
#   - the required files exist (config.json, metrics.json, model.onnx,
#     model.pt and model.sha256);
#   - config.json and metrics.json are valid JSON;
#   - the SHA-256 recorded in model.sha256 matches each weight file. For Git
#     LFS pointer files the oid in the pointer is compared; for downloaded
#     files the content is hashed;
#   - a model card exists at model_zoo/cards/<name>.md and declares the
#     MPL-2.0 licence;
#   - a manifest exists in model_zoo/manifests/ for the model family.
#
# Usage
# -----
#   python .github/scripts/check_model_zoo.py
#
# The script must run from the repository root.
#
# Exit status
# -----------
#   0  every variant passed every check
#   1  at least one problem was found
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Compute SHA-256 digests of weight files.
import hashlib

# Validate the JSON configuration and metrics files.
import json

# Recognise Git LFS pointer files.
import re

# Set the exit status of the script.
import sys

# Walk the model zoo directory tree.
from pathlib import Path

# Root directory of the model zoo, relative to the repository root.
ROOT = Path("model_zoo")

# Files that every model variant directory must contain.
REQUIRED = ("config.json", "metrics.json", "model.onnx", "model.pt", "model.sha256")

# Weight files whose digests are recorded in model.sha256.
WEIGHTS = ("model.onnx", "model.pt")

# Size tiers, one directory each under model_zoo/assets/.
TIERS = ("tiny", "base", "large", "mega")

# A Git LFS pointer file: the spec version line followed by the SHA-256 oid of
# the real content. re.S lets ".*?" cross the line between them.
POINTER = re.compile(rb"^version https://git-lfs.github.com/spec/v1\n.*?oid sha256:([0-9a-f]{64})", re.S)


# Return the SHA-256 of a weight file and whether the file is an LFS pointer.
def file_sha256(path: Path) -> tuple[str, bool]:
    # A pointer file is small, so its first 512 bytes contain the whole pointer.
    head = path.read_bytes()[:512]
    # Try to read the file as a Git LFS pointer.
    match = POINTER.match(head)
    # For a pointer, the oid is the digest of the real content.
    if match:
        # Return the oid as text and flag the file as a pointer.
        return match.group(1).decode(), True
    # Otherwise hash the downloaded content.
    digest = hashlib.sha256()
    # Read the file in binary mode.
    with path.open("rb") as fh:
        # Read 1 MiB chunks until read() returns an empty bytes object.
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            # Feed each chunk to the digest.
            digest.update(chunk)
    # Return the hexadecimal digest and flag the file as real content.
    return digest.hexdigest(), False


# Print a problem as a GitHub Actions error annotation on the given path.
def error(path: Path, message: str) -> None:
    # The ::error:: prefix makes the message an annotation in the job log.
    print(f"::error file={path}::{message}")


# Run all checks and return the exit status.
def main() -> int:
    # Counters for problems, variants, LFS pointers and hashed weight files.
    failures = variants = pointers = hashed = 0
    # Model families that have a manifest, named after the manifest file.
    families = {p.stem for p in (ROOT / "manifests").glob("*.json")}

    # Visit every size tier.
    for tier in TIERS:
        # Visit the variant directories of the tier in a stable order.
        for variant in sorted((ROOT / "assets" / tier).iterdir()):
            # Ignore stray files next to the variant directories.
            if not variant.is_dir():
                # Continue with the next entry.
                continue
            # Count the variant.
            variants += 1
            # Collect the required files that are missing.
            missing = [name for name in REQUIRED if not (variant / name).is_file()]
            # Without all files the remaining checks cannot run.
            if missing:
                # Report the missing files.
                error(variant, f"Missing files: {', '.join(missing)}")
                # Count the problem.
                failures += 1
                # Continue with the next variant.
                continue

            # Both metadata files must be valid JSON.
            for name in ("config.json", "metrics.json"):
                # Parse the file to detect syntax errors.
                try:
                    # The parsed content itself is not needed.
                    json.loads((variant / name).read_text(encoding="utf-8"))
                # Report invalid JSON with the parser message.
                except json.JSONDecodeError as exc:
                    # Include the position reported by the parser.
                    error(variant / name, f"Invalid JSON: {exc}")
                    # Count the problem.
                    failures += 1

            # Map each weight file name to the digest recorded in model.sha256.
            recorded = {}
            # model.sha256 uses the sha256sum format: "<digest>  <file name>".
            for line in (variant / "model.sha256").read_text(encoding="utf-8").splitlines():
                # Split the line at whitespace.
                parts = line.split()
                # Ignore lines that are not "<digest> <name>".
                if len(parts) == 2:
                    # Store the digest under the file name.
                    recorded[parts[1]] = parts[0]
            # Compare the recorded digest of every weight file.
            for weight in WEIGHTS:
                # Compute the actual digest and whether the file is a pointer.
                actual, is_pointer = file_sha256(variant / weight)
                # Count pointer files (True adds 1, False adds 0).
                pointers += is_pointer
                # Count files whose content was hashed.
                hashed += not is_pointer
                # A missing or different recorded digest is a problem.
                if recorded.get(weight) != actual:
                    # Report both digests so that the mismatch is visible.
                    error(variant / weight, f"SHA256 {actual} does not match model.sha256 ({recorded.get(weight)})")
                    # Count the problem.
                    failures += 1

            # Each variant has a model card named after the variant.
            card = ROOT / "cards" / f"{variant.name}.md"
            # The card must exist.
            if not card.is_file():
                # Report the missing card.
                error(card, "Model card missing")
                # Count the problem.
                failures += 1
            # The card must state the licence of the model.
            elif "MPL-2.0" not in card.read_text(encoding="utf-8"):
                # Report the missing licence statement.
                error(card, "Model card does not declare the MPL-2.0 licence")
                # Count the problem.
                failures += 1

            # The family name is the variant name without its tier suffix.
            family = variant.name.rsplit("_", 1)[0]
            # Every family needs a manifest.
            if family not in families:
                # Report the missing manifest.
                error(variant, f"No manifest model_zoo/manifests/{family}.json")
                # Count the problem.
                failures += 1

    # Summarise what was checked.
    print(f"Checked {variants} variants: {pointers} LFS pointers and {hashed} downloaded files verified.")
    # Fail when any problem was found.
    if failures:
        # Print the number of problems.
        print(f"{failures} model zoo problem(s) found.")
        # Non-zero exit status fails the CI job.
        return 1
    # Confirm that every check passed.
    print("Model zoo check passed.")
    # Zero exit status marks success.
    return 0


# Run the check when the file is executed as a script.
if __name__ == "__main__":
    # Exit with the status returned by main().
    sys.exit(main())

# =============================================================================
# End of module .github/scripts/check_model_zoo.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
