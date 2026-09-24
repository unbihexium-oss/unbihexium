# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : .github/scripts/check_release_version.py
# Title       : Release tag and version check
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.11 to 3.14 (tomllib), standard library only
# =============================================================================
#
# Abstract
# --------
# Checks that a release tag names the version recorded in the metadata files
# that define the package and its citation: pyproject.toml (the version of
# the built distributions), src/unbihexium/_version.py, CITATION.cff (the
# version and the release URL) and codemeta.json (version and
# softwareVersion). The release workflow runs it before the build, so a tag
# pushed without a version bump fails before anything is published.
#
# Usage
# -----
#   python .github/scripts/check_release_version.py v1.2.3
#
# The script must run from the repository root, because it reads the
# metadata files from the working directory.
#
# Exit status
# -----------
#   0  every recorded version matches the tag
#   1  at least one version differs from the tag
#   2  wrong usage, or a tag that does not start with "v"
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Read codemeta.json.
import json

# Find the version in the Python and CFF files.
import re

# Read the arguments and set the exit status of the script.
import sys

# Represent and read file paths.
from pathlib import Path

# Parse pyproject.toml.
import tomllib


# Return the versions recorded in the metadata files, by place.
def recorded_versions(root: Path) -> dict[str, str | None]:
    # Static version of the distributions.
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    # Version module of the package.
    module = re.search(
        r'^__version__ = "([^"]+)"',  # Assignment of the version string.
        (root / "src/unbihexium/_version.py").read_text(encoding="utf-8"),  # Module text.
        re.MULTILINE,  # Anchor at line starts.
    )  # End of the search.
    # Citation file.
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    # Version of the citation.
    cff_version = re.search(r"^version: (\S+)", cff, re.MULTILINE)
    # Release page named by the citation.
    cff_url = re.search(r"/releases/tag/v([^\"\s]+)", cff)
    # CodeMeta description.
    codemeta = json.loads((root / "codemeta.json").read_text(encoding="utf-8"))
    # Every place with its version; None where the version was not found.
    return {
        "pyproject.toml": str(project.get("version")),  # [project] version.
        "src/unbihexium/_version.py": module[1] if module else None,  # __version__.
        "CITATION.cff": cff_version[1] if cff_version else None,  # version.
        "CITATION.cff release URL": cff_url[1] if cff_url else None,  # releases/tag/v<version>.
        "codemeta.json version": codemeta.get("version"),  # version.
        "codemeta.json softwareVersion": codemeta.get("softwareVersion"),  # softwareVersion.
    }  # End of the versions.


# Compare the tag with every recorded version and return the exit status.
def main(argv: list[str]) -> int:
    # The tag is the only argument.
    if len(argv) != 2 or not argv[1].startswith("v"):
        # Explain the usage.
        print("usage: check_release_version.py v<version>", file=sys.stderr)
        # Exit status for wrong usage.
        return 2
    # Tag and the version it names.
    tag, expected = argv[1], argv[1][1:]
    # Number of mismatches.
    failures = 0
    # Check every place.
    for place, version in recorded_versions(Path()).items():
        # A missing or different version is a mismatch.
        if version != expected:
            # Report it as an annotation of the workflow run.
            print(f"::error::{place} records version {version}, but the tag is {tag}")
            # Count it.
            failures += 1
    # Summarise and fail when any version differs.
    if failures:
        # Explain how to fix it.
        print(f"{failures} place(s) do not match {tag}; bump the version before tagging.")
        # Non-zero exit status fails the release job.
        return 1
    # Confirm the match.
    print(f"All recorded versions match {tag}.")
    # Zero exit status marks success.
    return 0


# Run the check when the file is executed as a script.
if __name__ == "__main__":
    # Exit with the status returned by main().
    sys.exit(main(sys.argv))

# =============================================================================
# End of module .github/scripts/check_release_version.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
