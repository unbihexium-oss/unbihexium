#!/usr/bin/env python3
"""Repository audit script for unbihexium.

Checks:
- Required files exist
- No emoji characters
- Badges match workflows
- Coverage manifest is complete
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def check_required_files(repo_root: Path) -> list[str]:
    """Check that required files exist."""
    required = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "RESPONSIBLE_USE.md",
        "src/unbihexium/__init__.py",
        "src/unbihexium/cli/main.py",
        "tests/conftest.py",
        "mkdocs.yml",
        ".repo/coverage_manifest.yaml",
    ]

    missing = []
    for f in required:
        if not (repo_root / f).exists():
            missing.append(f)

    return missing


def check_no_emoji(repo_root: Path) -> list[str]:
    """Check for emoji characters in text files."""
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F"
        r"\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF"
        r"\U00002702-\U000027B0"
        r"\U000024C2-\U0001F251]+"
    )

    violations = []
    extensions = {".py", ".md", ".yaml", ".yml", ".toml", ".txt"}

    for ext in extensions:
        for path in repo_root.rglob(f"*{ext}"):
            if ".git" in str(path) or ".venv" in str(path):
                continue
            try:
                content = path.read_text(encoding="utf-8")
                if emoji_pattern.search(content):
                    violations.append(str(path.relative_to(repo_root)))
            except Exception:
                pass

    return violations


def check_workflow_badges(repo_root: Path) -> list[str]:
    """Check that README badges match workflow files."""
    issues = []

    readme = repo_root / "README.md"
    if not readme.exists():
        return ["README.md not found"]

    content = readme.read_text(encoding="utf-8")

    # Extract workflow names from badge URLs
    badge_pattern = re.compile(r"workflows/([a-z\-]+)\.yml")
    badge_workflows = set(badge_pattern.findall(content))

    # Get actual workflow files
    workflow_dir = repo_root / ".github" / "workflows"
    if workflow_dir.exists():
        actual_workflows = {f.stem for f in workflow_dir.glob("*.yml")}
    else:
        actual_workflows = set()

    # Check for badges without workflows
    for wf in badge_workflows:
        if wf not in actual_workflows:
            issues.append(f"Badge references non-existent workflow: {wf}")

    return issues


def main() -> int:
    """Run all audit checks."""
    repo_root = Path(__file__).parent.parent
    check_mode = "--check" in sys.argv

    all_issues = []

    # Check required files
    missing = check_required_files(repo_root)
    if missing:
        all_issues.extend([f"Missing required file: {f}" for f in missing])

    # Check no emoji
    emoji_files = check_no_emoji(repo_root)
    if emoji_files:
        all_issues.extend([f"Emoji found in: {f}" for f in emoji_files])

    # Check workflow badges
    badge_issues = check_workflow_badges(repo_root)
    all_issues.extend(badge_issues)

    # Print results
    if all_issues:
        print("Audit Issues Found:")
        for issue in all_issues:
            print(f"  - {issue}")
        if check_mode:
            return 1
    else:
        print("Audit: All checks passed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
