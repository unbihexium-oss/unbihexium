# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : .github/scripts/check_notebooks.py
# Title       : Example notebook validation
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires nbformat
# =============================================================================
#
# Abstract
# --------
# Validates the example notebooks. Every notebook must be valid nbformat 4 and
# must be committed without outputs or execution counts, so that no data,
# credentials or local paths leak into the repository and so that diffs stay
# reviewable.
#
# Usage
# -----
#   python .github/scripts/check_notebooks.py [directory]
#
# The directory defaults to examples/notebooks.
#
# Exit status
# -----------
#   0  every notebook is valid and has no outputs
#   1  at least one problem was found
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Access the command line arguments and set the exit status.
import sys

# Find the notebook files.
from pathlib import Path

# Read and validate Jupyter notebooks.
import nbformat


# Check every notebook in `root` and return the exit status.
def main(root: str = "examples/notebooks") -> int:
    # Number of problems found so far.
    failures = 0
    # Notebooks in a stable, alphabetical order.
    notebooks = sorted(Path(root).glob("*.ipynb"))
    # Check each notebook in turn.
    for path in notebooks:
        # Reading and schema validation both raise on invalid notebooks.
        try:
            # Parse the notebook and convert it to nbformat version 4.
            nb = nbformat.read(path, as_version=4)
            # Validate the notebook against the nbformat JSON schema.
            nbformat.validate(nb)
        # Report any read or validation problem and move on.
        except Exception as exc:  # noqa: BLE001 - report every validation problem
            # Report the problem as a GitHub Actions annotation.
            print(f"::error file={path}::Invalid notebook: {exc}")
            # Count the problem.
            failures += 1
            # The cells of an invalid notebook are not inspected.
            continue
        # Inspect every cell with its position in the notebook.
        for index, cell in enumerate(nb.cells):
            # Only code cells have outputs and execution counts.
            if cell.cell_type != "code":
                # Continue with the next cell.
                continue
            # Committed notebooks must have neither outputs nor counts.
            if cell.get("outputs") or cell.get("execution_count") is not None:
                # Report the cell index so that the author can find it.
                print(f"::error file={path}::Cell {index} contains outputs or an execution count; clear outputs before committing")
                # Count the problem.
                failures += 1
    # Summarise and fail when any problem was found.
    if failures:
        # Print the number of problems and of notebooks.
        print(f"{failures} problem(s) found in {len(notebooks)} notebooks.")
        # Non-zero exit status fails the CI job.
        return 1
    # Confirm that every notebook passed.
    print(f"All {len(notebooks)} notebooks are valid and have no outputs.")
    # Zero exit status marks success.
    return 0


# Run the check when the file is executed as a script.
if __name__ == "__main__":
    # An optional first argument replaces the default notebook directory.
    sys.exit(main(*sys.argv[1:]))

# =============================================================================
# End of module .github/scripts/check_notebooks.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
