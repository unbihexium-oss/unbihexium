# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Report dependency licences and fail on licences incompatible with the project policy.

Reads the JSON produced by `pip-licenses --format=json --with-urls` and writes a
Markdown table to the GitHub job summary when available. Strong copyleft and
source-available licences (GPL, AGPL, SSPL, Commons Clause) are rejected because
they would impose obligations on the whole distribution that go beyond the
file-level copyleft of the MPL-2.0. LGPL and MPL are accepted.
"""

from __future__ import annotations

import json
import os
import re
import sys

DENY = re.compile(
    r"(?<![L])GPL|GNU General Public License|Affero|SSPL|Server Side Public|Commons Clause",
    re.IGNORECASE,
)
ALTERNATIVES = re.compile(r"\s+OR\s+|;\s*")
IGNORE = {"unbihexium"}


def is_denied(licence: str) -> bool:
    """Return True if every alternative of a (possibly dual) licence is denied."""
    alternatives = [a.strip(" ()") for a in ALTERNATIVES.split(licence) if a.strip(" ()")]
    return bool(alternatives) and all(DENY.search(a) for a in alternatives)


def main(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        packages = json.load(fh)

    rows, denied = [], []
    for pkg in sorted(packages, key=lambda p: p["Name"].lower()):
        if pkg["Name"].lower() in IGNORE:
            continue
        licence = pkg.get("License", "UNKNOWN")
        rows.append(f"| {pkg['Name']} | {pkg.get('Version', '')} | {licence} |")
        if is_denied(licence):
            denied.append(f"{pkg['Name']} {pkg.get('Version', '')}: {licence}")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## Dependency licences\n\n| Package | Version | Licence |\n| --- | --- | --- |\n")
            fh.write("\n".join(rows) + "\n")

    print(f"Checked {len(rows)} installed packages.")
    for item in denied:
        print(f"::error::Dependency licence not allowed: {item}")
    return 1 if denied else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
