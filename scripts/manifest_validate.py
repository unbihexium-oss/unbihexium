#!/usr/bin/env python3
"""Manifest validation script for coverage manifest."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def validate_manifest(manifest_path: Path) -> list[str]:
    """Validate the coverage manifest."""
    issues = []

    if not manifest_path.exists():
        return [f"Manifest not found: {manifest_path}"]

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    if not manifest:
        return ["Manifest is empty"]

    capabilities = manifest.get("capabilities", [])
    if not capabilities:
        issues.append("No capabilities defined")

    required_fields = ["id", "name", "domain", "maturity", "entry_points"]

    for cap in capabilities:
        cap_id = cap.get("id", "unknown")

        for field in required_fields:
            if field not in cap:
                issues.append(f"Capability {cap_id}: missing {field}")

        # Check entry points exist
        for ep in cap.get("entry_points", []):
            parts = ep.split(".")
            if len(parts) < 3:
                issues.append(f"Capability {cap_id}: invalid entry point {ep}")

        # Check test path exists
        test_path = cap.get("test_path")
        if test_path:
            repo_root = manifest_path.parent.parent
            if not (repo_root / test_path).exists():
                issues.append(f"Capability {cap_id}: test not found {test_path}")

    models = manifest.get("models", [])
    for model in models:
        model_id = model.get("id", "unknown")
        if not model.get("path"):
            issues.append(f"Model {model_id}: missing path")

    return issues


def main() -> int:
    """Run manifest validation."""
    repo_root = Path(__file__).parent.parent
    manifest_path = repo_root / ".repo" / "coverage_manifest.yaml"

    issues = validate_manifest(manifest_path)

    if issues:
        print("Manifest Validation Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("Manifest validation: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
