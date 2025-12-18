#!/usr/bin/env python3
"""Documentation linting script.

Enforces:
- At least 1 Mermaid diagram per major page
- At least 1 LaTeX formula per major page
- At least 1 table per major page
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def check_doc(path: Path) -> list[str]:
    """Check a documentation file for required elements."""
    issues = []
    content = path.read_text(encoding="utf-8")

    # Check for Mermaid
    if "```mermaid" not in content:
        issues.append(f"{path.name}: missing Mermaid diagram")

    # Check for LaTeX formula
    if not re.search(r"\$[^$]+\$|\$\$[^$]+\$\$", content):
        issues.append(f"{path.name}: missing LaTeX formula")

    # Check for table
    if not re.search(r"^\|.+\|$", content, re.MULTILINE):
        issues.append(f"{path.name}: missing table")

    return issues


def main() -> int:
    """Run docs linting."""
    repo_root = Path(__file__).parent.parent
    docs_dir = repo_root / "docs"

    if not docs_dir.exists():
        print("docs/ directory not found")
        return 1

    # Major pages that must have all elements
    major_pages = [
        "index.md",
        "getting_started/quickstart.md",
        "getting_started/installation.md",
    ]

    all_issues = []

    for page in major_pages:
        page_path = docs_dir / page
        if page_path.exists():
            issues = check_doc(page_path)
            all_issues.extend(issues)

    if all_issues:
        print("Docs Lint Issues:")
        for issue in all_issues:
            print(f"  - {issue}")
        if "--check" in sys.argv:
            return 1

    print(f"Docs lint: checked {len(major_pages)} pages, {len(all_issues)} issues")
    return 0


if __name__ == "__main__":
    sys.exit(main())
