#!/usr/bin/env python3
"""Secrets audit script for CI/CD workflows.

Parses .github/workflows/*.yml and extracts referenced secrets.
Produces artifacts/secrets_report.json and artifacts/secrets_report.md.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def extract_secrets_from_yaml(content: str) -> list[str]:
    """Extract secret references from YAML content."""
    pattern = r'\$\{\{\s*secrets\.(\w+)\s*\}\}'
    return list(set(re.findall(pattern, content)))


def analyze_workflows(workflows_dir: Path) -> dict:
    """Analyze all workflow files for secrets usage."""
    results = {
        "workflows": {},
        "all_secrets": set(),
        "secret_details": {}
    }

    if not workflows_dir.exists():
        return results

    for workflow_file in workflows_dir.glob("*.yml"):
        content = workflow_file.read_text(encoding="utf-8")
        secrets = extract_secrets_from_yaml(content)

        results["workflows"][workflow_file.name] = {
            "secrets": secrets,
            "has_conditional": "secrets." in content and "!= ''" in content
        }

        for secret in secrets:
            results["all_secrets"].add(secret)
            if secret not in results["secret_details"]:
                results["secret_details"][secret] = {
                    "used_by": [],
                    "required": False,
                    "description": get_secret_description(secret)
                }
            results["secret_details"][secret]["used_by"].append(workflow_file.name)

    results["all_secrets"] = sorted(results["all_secrets"])
    return results


def get_secret_description(secret_name: str) -> dict:
    """Get known secret descriptions and requirements."""
    secrets_info = {
        "GITHUB_TOKEN": {
            "required": False,
            "where_to_obtain": "Automatically provided by GitHub Actions",
            "permissions": "Varies by workflow",
            "storage": "automatic",
            "notes": "Default token, no setup needed"
        },
        "CODECOV_TOKEN": {
            "required": False,
            "where_to_obtain": "https://codecov.io/gh/unbihexium-oss/unbihexium/settings",
            "permissions": "Upload coverage reports",
            "storage": "repo",
            "notes": "Optional with OIDC; required for private repos"
        },
        "SCORECARD_TOKEN": {
            "required": False,
            "where_to_obtain": "GitHub PAT with public_repo scope",
            "permissions": "Read public repo data",
            "storage": "repo",
            "notes": "Uses GITHUB_TOKEN by default"
        },
        "PYPI_API_TOKEN": {
            "required": False,
            "where_to_obtain": "https://pypi.org/manage/account/token/",
            "permissions": "Upload packages",
            "storage": "repo",
            "notes": "Prefer Trusted Publishing (OIDC) instead"
        },
        "SONAR_TOKEN": {
            "required": False,
            "where_to_obtain": "https://sonarcloud.io/account/security",
            "permissions": "Analyze code",
            "storage": "repo",
            "notes": "Only if using SonarCloud"
        },
        "SNYK_TOKEN": {
            "required": False,
            "where_to_obtain": "https://app.snyk.io/account",
            "permissions": "Security scanning",
            "storage": "repo",
            "notes": "Only if using Snyk"
        }
    }

    return secrets_info.get(secret_name, {
        "required": False,
        "where_to_obtain": "Unknown",
        "permissions": "Unknown",
        "storage": "repo",
        "notes": "Custom secret"
    })


def generate_markdown_report(results: dict) -> str:
    """Generate markdown report."""
    lines = [
        "# Secrets Audit Report",
        "",
        "## Purpose",
        "",
        "This report documents all secrets used in CI/CD workflows.",
        "",
        "## Audience",
        "",
        "Repository maintainers from trusted organizations.",
        "",
        "## Secrets Summary",
        "",
        "```mermaid",
        "graph LR",
        "    subgraph Secrets",
    ]

    for secret in results["all_secrets"]:
        lines.append(f"        {secret}[{secret}]")

    lines.extend([
        "    end",
        "    subgraph Workflows",
    ])

    for wf in results["workflows"]:
        wf_id = wf.replace(".yml", "").replace("-", "_")
        lines.append(f"        {wf_id}[{wf}]")

    lines.extend([
        "    end",
        "```",
        "",
        "## Secrets Reference Table",
        "",
        "| Secret Name | Used By | Required | Where to Obtain | Permissions | Storage | Notes |",
        "|-------------|---------|----------|-----------------|-------------|---------|-------|"
    ])

    for secret in sorted(results["all_secrets"]):
        info = results["secret_details"].get(secret, {})
        desc = info.get("description", {})
        used_by = ", ".join(info.get("used_by", []))
        required = "Yes" if desc.get("required", False) else "No"
        where = desc.get("where_to_obtain", "Unknown")
        perms = desc.get("permissions", "Unknown")
        storage = desc.get("storage", "repo")
        notes = desc.get("notes", "")

        lines.append(f"| {secret} | {used_by} | {required} | {where} | {perms} | {storage} | {notes} |")

    lines.extend([
        "",
        "## Workflow Resilience",
        "",
        "All workflows should be resilient to missing secrets:",
        "",
        "$$resilience = \\frac{guarded\\_steps}{total\\_secret\\_steps}$$",
        "",
        "| Workflow | Secrets Used | Has Conditionals |",
        "|----------|--------------|------------------|"
    ])

    for wf, data in results["workflows"].items():
        secrets = ", ".join(data["secrets"]) if data["secrets"] else "None"
        conditional = "Yes" if data["has_conditional"] else "No"
        lines.append(f"| {wf} | {secrets} | {conditional} |")

    lines.extend([
        "",
        "## Recommendations",
        "",
        "1. Use OIDC-based authentication where possible (Codecov, PyPI)",
        "2. Guard all secret-dependent steps with conditionals",
        "3. Never fail CI for missing optional secrets",
        "",
        "## References",
        "",
        "- [Documentation Index](../index.md)",
        "- [Table of Contents](../toc.md)"
    ])

    return "\n".join(lines)


def main() -> int:
    """Run secrets audit."""
    repo_root = Path(__file__).parent.parent
    workflows_dir = repo_root / ".github" / "workflows"
    artifacts_dir = repo_root / "artifacts"

    artifacts_dir.mkdir(exist_ok=True)

    results = analyze_workflows(workflows_dir)

    # Write JSON report
    json_report = {
        "total_secrets": len(results["all_secrets"]),
        "total_workflows": len(results["workflows"]),
        "secrets": results["all_secrets"],
        "workflows": results["workflows"],
        "details": {k: {**v, "description": v.get("description", {})} for k, v in results["secret_details"].items()}
    }

    with open(artifacts_dir / "secrets_report.json", "w") as f:
        json.dump(json_report, f, indent=2, default=str)

    # Write Markdown report
    md_report = generate_markdown_report(results)
    with open(artifacts_dir / "secrets_report.md", "w") as f:
        f.write(md_report)

    # Also write to docs
    docs_security = repo_root / "docs" / "security"
    docs_security.mkdir(exist_ok=True)
    with open(docs_security / "secrets_and_tokens.md", "w") as f:
        f.write(md_report)

    print(f"Secrets Audit Complete:")
    print(f"  - Total secrets found: {len(results['all_secrets'])}")
    print(f"  - Workflows analyzed: {len(results['workflows'])}")
    print(f"  - Reports written to artifacts/ and docs/security/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
