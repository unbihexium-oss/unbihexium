# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : .github/scripts/check_dependency_licenses.py
# Title       : Dependency licence policy check
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
# Reports the licences of the installed dependencies and fails when a licence
# is incompatible with the project licence policy. The input is the JSON
# written by `pip-licenses --format=json --with-urls`. When the script runs in
# GitHub Actions it also writes a Markdown table to the job summary.
#
# Policy
# ------
# Strong copyleft and source-available licences (GPL, AGPL, SSPL and the
# Commons Clause) are rejected, because they would impose obligations on the
# whole distribution that go beyond the file-level copyleft of the MPL-2.0.
# LGPL and MPL licences are accepted. A dual licence is accepted when at least
# one of its alternatives is accepted.
#
# Usage
# -----
#   pip-licenses --from=mixed --format=json --with-urls --output-file licenses.json
#   python .github/scripts/check_dependency_licenses.py licenses.json
#
# Exit status
# -----------
#   0  every dependency licence is allowed
#   1  at least one dependency licence is denied
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Parse the JSON report written by pip-licenses.
import json

# Read the GITHUB_STEP_SUMMARY environment variable.
import os

# Match licence names against the deny list.
import re

# Access the command line arguments and set the exit status.
import sys

# Licence names that are not allowed. The negative lookbehind keeps LGPL
# (which contains "GPL") on the allowed side.
DENY = re.compile(
    # GPL but not LGPL, the full GPL name, AGPL, SSPL and the Commons Clause.
    r"(?<![L])GPL|GNU General Public License|Affero|SSPL|Server Side Public|Commons Clause",
    # Licence names are compared without regard to case.
    re.IGNORECASE,
)  # End of the deny list pattern.

# Separators between the alternatives of a dual licence ("MIT OR GPL-2.0",
# or the "; " separated classifier lists that pip-licenses produces).
ALTERNATIVES = re.compile(r"\s+OR\s+|;\s*")

# Packages that are not third-party dependencies: the project itself.
IGNORE = {"unbihexium"}


# Decide whether a licence string is denied by the policy.
def is_denied(licence: str) -> bool:
    # A licence is denied only when every alternative of a (possibly dual)
    # licence is denied; parentheses and spaces around each alternative are
    # removed and empty alternatives are dropped.
    alternatives = [a.strip(" ()") for a in ALTERNATIVES.split(licence) if a.strip(" ()")]
    # An empty licence string is not denied here; it shows up in the report.
    return bool(alternatives) and all(DENY.search(a) for a in alternatives)


# Check the pip-licenses report at `path` and return the exit status.
def main(path: str) -> int:
    # Open the JSON report as UTF-8 text.
    with open(path, encoding="utf-8") as fh:
        # The report is a list of objects with Name, Version and License keys.
        packages = json.load(fh)

    # Table rows for the job summary, and the list of denied packages.
    rows, denied = [], []
    # Visit the packages in case-insensitive alphabetical order.
    for pkg in sorted(packages, key=lambda p: p["Name"].lower()):
        # Skip the project itself.
        if pkg["Name"].lower() in IGNORE:
            # Continue with the next package.
            continue
        # Packages without licence metadata are reported as UNKNOWN.
        licence = pkg.get("License", "UNKNOWN")
        # Add a Markdown table row for this package.
        rows.append(f"| {pkg['Name']} | {pkg.get('Version', '')} | {licence} |")
        # Record the package when its licence is not allowed.
        if is_denied(licence):
            # Keep name, version and licence for the error message.
            denied.append(f"{pkg['Name']} {pkg.get('Version', '')}: {licence}")

    # GitHub Actions sets this variable to the path of the job summary file.
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    # Write the summary only when running in GitHub Actions.
    if summary:
        # Append, because other steps may already have written to the summary.
        with open(summary, "a", encoding="utf-8") as fh:
            # Section heading and table header.
            fh.write("## Dependency licences\n\n| Package | Version | Licence |\n| --- | --- | --- |\n")
            # One row per package, followed by a final newline.
            fh.write("\n".join(rows) + "\n")

    # Report how many packages were checked.
    print(f"Checked {len(rows)} installed packages.")
    # Emit one GitHub Actions error annotation per denied package.
    for item in denied:
        # The ::error:: prefix makes the message an annotation in the job log.
        print(f"::error::Dependency licence not allowed: {item}")
    # Fail when at least one licence is denied.
    return 1 if denied else 0


# Run the check when the file is executed as a script.
if __name__ == "__main__":
    # The first command line argument is the path of the pip-licenses report.
    sys.exit(main(sys.argv[1]))

# =============================================================================
# End of module .github/scripts/check_dependency_licenses.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
