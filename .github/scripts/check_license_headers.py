# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : .github/scripts/check_license_headers.py
# Title       : MPL-2.0 source file notice check
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
# Checks that every tracked Python and shell source file carries the MPL-2.0
# Exhibit A notice, and that LICENSE.txt contains the Mozilla Public License
# 2.0 text. The notice must appear at the top of the file: within the first
# five lines, so that an optional shebang and encoding line may precede it.
#
# Usage
# -----
#   python .github/scripts/check_license_headers.py
#
# The script must run from the repository root, because it lists files with
# `git ls-files` and reads LICENSE.txt from the working directory.
#
# Exit status
# -----------
#   0  LICENSE.txt and every source file are correct
#   1  at least one problem was found
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Run `git ls-files` to list the tracked files.
import subprocess

# Set the exit status of the script.
import sys

# Represent and read file paths.
from pathlib import Path

# The MPL-2.0 Exhibit A notice exactly as it must appear in source files.
NOTICE = (
    # First line of the notice.
    "# This Source Code Form is subject to the terms of the Mozilla Public\n"
    # Second line of the notice.
    "# License, v. 2.0. If a copy of the MPL was not distributed with this\n"
    # Third line of the notice.
    "# file, You can obtain one at https://mozilla.org/MPL/2.0/.\n"
)  # End of the notice text.

# File name patterns of the source files that must carry the notice.
PATTERNS = ["*.py", "*.bash", "*.sh"]

# Number of lines at the top of a file in which the notice must appear.
HEAD_LINES = 5


# Return the tracked files that match the given git pathspec patterns.
def tracked(patterns: list[str]) -> list[Path]:
    # -z separates the file names with NUL bytes, which is safe for any name.
    out = subprocess.run(["git", "ls-files", "-z", "--", *patterns], check=True, capture_output=True).stdout
    # Split the output at NUL bytes and drop the empty trailing entry.
    return [Path(p) for p in out.decode().split("\0") if p]


# Run all checks and return the exit status.
def main() -> int:
    # Number of problems found so far.
    failures = 0

    # Read the licence text at the repository root.
    licence = Path("LICENSE.txt").read_text(encoding="utf-8")
    # The official MPL-2.0 text starts with this title line.
    if not licence.startswith("Mozilla Public License Version 2.0"):
        # Report the problem as a GitHub Actions annotation on LICENSE.txt.
        print("::error file=LICENSE.txt::LICENSE.txt does not contain the Mozilla Public License 2.0 text")
        # Count the problem.
        failures += 1

    # List every tracked Python and shell file.
    files = tracked(PATTERNS)
    # Check each file in turn.
    for path in files:
        # Join the first HEAD_LINES lines, keeping their line endings.
        head = "".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[:HEAD_LINES])
        # The notice must appear as one contiguous block within those lines.
        if NOTICE not in head:
            # Report the missing notice as an annotation on line 1.
            print(f"::error file={path},line=1::Missing MPL-2.0 notice at the top of the file")
            # Count the problem.
            failures += 1

    # Summarise and fail when any problem was found.
    if failures:
        # Print the number of problems and of checked files.
        print(f"{failures} licence header problem(s) found in {len(files)} source files.")
        # Non-zero exit status fails the CI job.
        return 1
    # Confirm that every file passed.
    print(f"All {len(files)} source files carry the MPL-2.0 notice.")
    # Zero exit status marks success.
    return 0


# Run the check when the file is executed as a script.
if __name__ == "__main__":
    # Exit with the status returned by main().
    sys.exit(main())

# =============================================================================
# End of module .github/scripts/check_license_headers.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
