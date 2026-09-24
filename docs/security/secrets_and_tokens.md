<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/security/secrets_and_tokens.md
Title       : Secrets and Tokens
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Secrets and Tokens

| Field | Value |
| --- | --- |
| Document | UBX-DOC-804 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch, including the GitHub Actions workflows |

## Abstract

This document lists every credential that the Unbihexium repository and its automation use: the repository secrets, the automatic `GITHUB_TOKEN` with the permissions each workflow grants it, and the short-lived OpenID Connect tokens used for Sigstore signatures, artifact attestations and the OpenSSF Scorecard. It also describes how leaked credentials are detected, what contributors and users must never commit, and how operators should handle the one secret the software itself reads at run time, the API key of the REST service. It is intended for maintainers, contributors, auditors and operators. The information was taken from the workflow files under [.github/workflows/](../../.github/workflows/) at the date of review.

## Contents

- [1. Introduction](#1-introduction)
- [2. Inventory of Credentials](#2-inventory-of-credentials)
- [3. GITHUB_TOKEN Permissions per Workflow](#3-github_token-permissions-per-workflow)
- [4. OpenID Connect Tokens](#4-openid-connect-tokens)
- [5. Protection of Secrets in Workflows](#5-protection-of-secrets-in-workflows)
- [6. Secret Scanning](#6-secret-scanning)
- [7. What Must Never Be Committed](#7-what-must-never-be-committed)
- [8. Run-Time Secrets of the Software](#8-run-time-secrets-of-the-software)
- [9. Responding to a Leaked Credential](#9-responding-to-a-leaked-credential)
- [References](#references)

## 1. Introduction

### 1.1 Purpose

A credential stored in a repository or exposed by a workflow can be used to publish a malicious release, push a container image or tamper with the repository. The purpose of this document is to keep the set of credentials small, visible and auditable.

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in all capitals.

### 1.3 Principles

- Long-lived secrets are used only where no short-lived alternative is configured.
- Every workflow sets a read-only default for `GITHUB_TOKEN` and widens permissions per job, only where needed [3].
- Signatures and attestations use keyless signing with short-lived OIDC identities; the project holds no signing key.

## 2. Inventory of Credentials

| Credential | Kind | Stored in | Used by | Purpose |
| --- | --- | --- | --- | --- |
| PyPI upload token | Short-lived API token, minted by PyPI for one run through trusted publishing | Not stored | [release.yml](../../.github/workflows/release.yml), step "Publish to PyPI" | Upload the sdist and the wheel to <https://pypi.org/project/unbihexium/> |
| `CODECOV_TOKEN` | Codecov upload token | Repository secret | [coverage.yml](../../.github/workflows/coverage.yml), step "Upload coverage to Codecov" | Upload the coverage report; it grants no write access to the repository |
| `GITHUB_TOKEN` | Automatic, per job, expires when the job ends | Provided by GitHub Actions | Every workflow (Section 3) | Read the repository; where granted, write releases, packages, labels, comments or code scanning results |
| OIDC ID token | Short-lived JSON Web Token, requested at run time | Not stored | [release.yml](../../.github/workflows/release.yml), [docker.yml](../../.github/workflows/docker.yml), [scorecard.yml](../../.github/workflows/scorecard.yml) | PyPI trusted publishing, Sigstore signing of the distributions and the container image, artifact attestations, Scorecard publication (Section 4) |

No other secret is referenced by any workflow. There are no cloud provider credentials, no registry passwords other than `GITHUB_TOKEN` for the GitHub Container Registry, no PyPI credentials, no signing keys and no deployment keys.

The release job runs in the GitHub environment `pypi` and publishes with PyPI trusted publishing [8]: PyPI checks the OIDC token of the job against the trusted publisher registered for the project (repository `unbihexium-oss/unbihexium`, workflow `release.yml`, environment `pypi`) and returns an upload token that expires after the upload. The lead maintainer, who administers the project on PyPI ([GOVERNANCE.md](../../GOVERNANCE.md)), registers the trusted publisher once, then deletes the former `PYPI_API_TOKEN` repository secret and revokes that token on PyPI. Required reviewers MAY be added to the `pypi` environment in the repository settings, so that every upload waits for an approval.

## 3. GITHUB_TOKEN Permissions per Workflow

Every workflow declares a top-level `permissions` block. The table lists that default and the permissions of each job that overrides it; a job without its own block inherits the default. A permission that is not listed is `none`.

| Workflow | Default | Job-level permissions |
| --- | --- | --- |
| [ci.yml](../../.github/workflows/ci.yml) | `contents: read` | none (inherit) |
| [codeql.yml](../../.github/workflows/codeql.yml) | `contents: read` | `analyze`: `contents: read`, `actions: read`, `security-events: write` |
| [container-scan.yml](../../.github/workflows/container-scan.yml) | `contents: read` | `grype`: `contents: read`, `security-events: write` |
| [coverage.yml](../../.github/workflows/coverage.yml) | `contents: read` | none (inherit) |
| [docker.yml](../../.github/workflows/docker.yml) | `contents: read` | `build-and-push`: `contents: read`, `packages: write`, `id-token: write`, `attestations: write` |
| [fuzz.yml](../../.github/workflows/fuzz.yml) | `contents: read` | none (inherit) |
| [integration.yml](../../.github/workflows/integration.yml) | `contents: read` | none (inherit) |
| [labeler.yml](../../.github/workflows/labeler.yml) | `contents: read` | `label`: `contents: read`, `pull-requests: write` |
| [labels.yml](../../.github/workflows/labels.yml) | `contents: read` | `sync`: `contents: read`, `issues: write` |
| [license-compliance.yml](../../.github/workflows/license-compliance.yml) | `contents: read` | none (inherit) |
| [links.yml](../../.github/workflows/links.yml) | `contents: read` | `lychee`: `contents: read`, `issues: write` |
| [markdown.yml](../../.github/workflows/markdown.yml) | `contents: read` | none (inherit) |
| [model-zoo.yml](../../.github/workflows/model-zoo.yml) | `contents: read` | none (inherit) |
| [package.yml](../../.github/workflows/package.yml) | `contents: read` | none (inherit) |
| [pr-title.yml](../../.github/workflows/pr-title.yml) | `contents: read` | `conventional-title`: `pull-requests: read` |
| [release.yml](../../.github/workflows/release.yml) | `contents: read` | `release`: `contents: write`, `id-token: write`, `attestations: write` |
| [repo-config.yml](../../.github/workflows/repo-config.yml) | `contents: read` | none (inherit) |
| [scorecard.yml](../../.github/workflows/scorecard.yml) | `read-all` | `analysis`: `security-events: write`, `id-token: write`, `contents: read`, `actions: read` |
| [secret-scan.yml](../../.github/workflows/secret-scan.yml) | `contents: read` | none (inherit) |
| [security.yml](../../.github/workflows/security.yml) | `contents: read` | none (inherit) |
| [stale.yml](../../.github/workflows/stale.yml) | `contents: read` | `stale`: `issues: write`, `pull-requests: write` |
| [text-policy.yml](../../.github/workflows/text-policy.yml) | `contents: read` | none (inherit) |
| [torch-lock.yml](../../.github/workflows/torch-lock.yml) | `contents: read` | none (inherit) |
| [welcome.yml](../../.github/workflows/welcome.yml) | `contents: read` | `welcome`: `issues: write`, `pull-requests: write` |
| [workflow-lint.yml](../../.github/workflows/workflow-lint.yml) | `contents: read` | none (inherit) |

Only two jobs can change what users download: `release` in release.yml (`contents: write` for the GitHub release and its assets) and `build-and-push` in docker.yml (`packages: write` for the container image). The docker job logs in to `ghcr.io` with `GITHUB_TOKEN` and does so only for pushes and tags, never for pull requests.

## 4. OpenID Connect Tokens

A job with `id-token: write` can request an OIDC ID token from GitHub that states which repository, workflow, ref and commit the job runs for [4]. The token lives for minutes and is not stored. Three jobs use it:

- **release** in release.yml. `actions/attest-build-provenance` exchanges the token for a short-lived signing certificate from the Sigstore certificate authority (Fulcio) and signs SLSA build provenance for every distribution; the attestation is stored by GitHub (`attestations: write`). `sigstore/gh-action-sigstore-python` does the same to sign each distribution and writes a `.sigstore.json` bundle; the signature is recorded in the public Rekor transparency log [5]. The certificate identity is the workflow path at the tag, `https://github.com/unbihexium-oss/unbihexium/.github/workflows/release.yml@refs/tags/<tag>`, which is what verifiers check (see [supply_chain_security.md](supply_chain_security.md)). `actions/attest-sbom` attests the SPDX SBOM of the release in the same way, and `pypa/gh-action-pypi-publish` exchanges the token for the PyPI upload token (Section 2).
- **build-and-push** in docker.yml. For pushed images only, `actions/attest-build-provenance` and `actions/attest-sbom` attest the image digest and store the attestations in the registry, and `cosign sign` signs the digest in keyless mode; the certificate identity is `https://github.com/unbihexium-oss/unbihexium/.github/workflows/docker.yml@<ref>`.
- **analysis** in scorecard.yml. The OpenSSF Scorecard action uses the token to prove to the public Scorecard API that the results come from this repository's workflow on the default branch [6].

Because no private key exists, there is nothing to rotate or leak for signing. The trust in a signature rests on the workflow file and on who can push tags.

## 5. Protection of Secrets in Workflows

- **Forks and Dependabot.** GitHub does not pass repository secrets to workflows triggered by pull requests from forks or by Dependabot, and gives them a read-only `GITHUB_TOKEN` [3]. The Codecov upload in coverage.yml is therefore skipped or fails for such pull requests; the step is configured with `fail_ci_if_error: false`, so the job still passes.
- **`pull_request_target`.** labeler.yml, pr-title.yml and welcome.yml run in the context of the base branch so that they can label or comment on pull requests from forks. None of them checks out or executes code from the pull request; they read only file lists, titles and their own configuration. New workflows MUST NOT combine `pull_request_target` with a checkout of the pull request head.
- **Credentials on disk.** Most checkouts set `persist-credentials: false`, so the token is not left in `.git/config`. The checkouts in ci.yml, codeql.yml, coverage.yml, docker.yml, integration.yml, labels.yml, release.yml and security.yml keep the default; in all of these except release.yml the token is read-only.
- **Pinned actions.** Every third-party action is pinned by full commit SHA, so a moved tag cannot inject code into a job that holds a secret (see [supply_chain_security.md](supply_chain_security.md)).
- **Static checks.** CodeQL analyses the workflows (`actions` language) for script injection and excessive permissions, and actionlint checks every workflow change.
- **Secrets in logs.** GitHub masks the values of repository secrets in job logs. Workflows MUST NOT print secrets, write them to artifacts or pass them to steps that do not need them.

## 6. Secret Scanning

### 6.1 TruffleHog in CI

[.github/workflows/secret-scan.yml](../../.github/workflows/secret-scan.yml) runs TruffleHog 3.97.6 [7] on every push to `main`, on every pull request to `main` and on demand. It fetches the full history and scans the commits new relative to the base. It is configured with `--results=verified`: only credentials that the issuing service confirms as live fail the job. This avoids false positives, but it also means that a secret for a service TruffleHog cannot verify, or a secret that has already been revoked, does not fail the check.

### 6.2 Local Hooks

The `detect-private-key` hook in [.pre-commit-config.yaml](../../.pre-commit-config.yaml) rejects commits that contain PEM private key headers. Contributors SHOULD install the hooks as described in [CONTRIBUTING.md](../../CONTRIBUTING.md).

### 6.3 GitHub Secret Scanning

GitHub's own secret scanning and push protection are repository settings, not files in the repository, and cannot be audited from the source tree. The controls described in this section do not depend on them.

## 7. What Must Never Be Committed

Contributors and users MUST NOT commit any of the following to the repository, to an issue, to a pull request or to a discussion:

- API tokens and passwords of any service, including PyPI, Codecov, GitHub, cloud storage and commercial imagery providers;
- the API key of a deployed REST service, and `.env` files with real values;
- private keys and certificates (`*.pem`, `*.key`, `*.p12`, `*.pfx`), `.netrc` and `.pypirc` files;
- STAC API authorisation headers or signed URLs with embedded credentials;
- configuration files written by `Config.to_yaml()` from a configuration that contains `serving.api_key`, because the key is stored in clear text (`Config.to_dict()` does not redact it);
- imagery or data whose licence forbids redistribution, and personal data (see [PRIVACY.md](../../PRIVACY.md)).

[.gitignore](../../.gitignore) ignores `.env`, `.env.*` (except `.env.example`), `secrets/`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `.netrc` and `.pypirc` as a safety net; it does not replace care. [.env.example](../../.env.example) is committed on purpose and MUST contain only placeholder values.

## 8. Run-Time Secrets of the Software

The library itself reads one secret: the optional API key of the REST service, set with the environment variable `UNBIHEXIUM_SERVING__API_KEY` (or `serving.api_key` in a YAML file named by `UNBIHEXIUM_CONFIG`). When it is set, clients MUST send it in the `X-API-Key` header on every route except `/health`; the service compares it in constant time with `hmac.compare_digest`. Operators:

- SHOULD provide the key through the environment of the container, for example from an uncommitted `.env` file or a Kubernetes Secret (see [docs/operations/docker.md](../operations/docker.md)), and MUST NOT write it into committed manifests;
- SHOULD use a long random value, for example `python -c "import secrets; print(secrets.token_urlsafe(32))"`, and rotate it when staff or clients change;
- MUST terminate TLS in front of the service when the key crosses an untrusted network, because the service speaks plain HTTP.

`unbihexium.io.stac.STACClient` accepts arbitrary request headers (for example an `Authorization` header) supplied by the caller; the library does not store them. Callers SHOULD read such values from the environment or a secret store rather than from source code.

## 9. Responding to a Leaked Credential

If a credential of the project is exposed, the maintainer revokes it at its issuer first (PyPI, Codecov), then replaces the repository secret, reviews the audit logs of the affected service for misuse, and removes the value from the history only after revocation, since rewriting history does not invalidate a copied secret. Anyone who finds an exposed credential of the project MUST report it privately as described in [SECURITY.md](../../SECURITY.md) and MUST NOT use it. Users who commit their own credentials by mistake SHOULD revoke them immediately; deleting the commit is not sufficient.

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] GitHub. Automatic token authentication. 2026. <https://docs.github.com/en/actions/security-for-github-actions/security-guides/automatic-token-authentication>

[4] GitHub. About security hardening with OpenID Connect. 2026. <https://docs.github.com/en/actions/security-for-github-actions/security-hardening-your-deployments/about-security-hardening-with-openid-connect>

[5] Sigstore. Sigstore documentation. 2026. <https://docs.sigstore.dev/>

[6] OpenSSF. OpenSSF Scorecard. 2026. <https://scorecard.dev/>

[7] Truffle Security. TruffleHog. 2026. <https://github.com/trufflesecurity/trufflehog>

[8] Python Packaging Authority. Publishing to PyPI with a Trusted Publisher. 2026. <https://docs.pypi.org/trusted-publishers/>

<!--
=============================================================================
End of file docs/security/secrets_and_tokens.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
