#!/usr/bin/env python3
"""Link checker for documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def check_links(docs_dir: Path) -> list[str]:
    """Check all internal links in documentation."""
    issues = []

    for md_file in docs_dir.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")

        # Find markdown links
        links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)

        for text, href in links:
            # Skip external links
            if href.startswith(("http://", "https://", "#")):
                continue

            # Resolve relative path
            if href.startswith("/"):
                target = docs_dir / href[1:]
            else:
                target = md_file.parent / href

            # Remove anchor
            target_str = str(target).split("#")[0]
            target = Path(target_str)

            if not target.exists():
                issues.append(f"{md_file.name}: broken link to {href}")

    return issues


def main() -> int:
    """Run link checker."""
    repo_root = Path(__file__).parent.parent
    docs_dir = repo_root / "docs"

    if not docs_dir.exists():
        print("docs/ directory not found")
        return 1

    issues = check_links(docs_dir)

    if issues:
        print("Link Check Issues:")
        for issue in issues:
            print(f"  - {issue}")
        if "--check" in sys.argv:
            return 1

    print(f"Link check: {len(issues)} broken links found")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
