# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# File        : Makefile
# Title       : Development tasks for Unbihexium
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Format      : Makefile, read by GNU make 3.82 or later
# =============================================================================
#
# Abstract
# --------
# Development tasks for Unbihexium. Run `make` or `make help` to list every
# target with its description; the description is the `## ` text after the
# target and the `##@ ` lines start the groups of the help output.
#
# Requirements
#   - Python 3.10 to 3.14 (the interpreter on PATH, or set PYTHON=...)
#   - uv (https://docs.astral.sh/uv/) for `make lock` and `make lock-check`
#   - Docker for the docker-* targets
#   - Node.js (npx) for `make md-lint`
#
# Variables can be overridden on the command line, for example:
#   make test PYTHON=python3.13
#   make docker-build IMAGE=unbihexium:dev
#
# Targets mirror the checks in .github/workflows/ so that contributors can
# reproduce CI results locally before opening a pull request.
#
# Comments stand on their own line starting in column 0, also between recipe
# lines: GNU make ignores such lines, whereas a tab-indented `#` would be
# passed to the shell and echoed.
# =============================================================================

# Plain `make` shows the help instead of running a task.
.DEFAULT_GOAL := help
# Run recipes with Bash, which provides pipefail.
SHELL := /bin/bash
# Stop a recipe at the first failing command, unset variable or failing pipe.
.SHELLFLAGS := -eu -o pipefail -c

# Python interpreter used by every target.
PYTHON ?= python
# pip of that interpreter.
PIP ?= $(PYTHON) -m pip
# uv executable used to compile the lock files.
UV ?= uv
# Tag of the container image built and run by the docker-* targets.
IMAGE ?= unbihexium:local
# Package sources measured by the coverage report.
SRC := src/unbihexium
# Host port on which `make docker-api` publishes the REST API.
PORT ?= 8000

# Oldest supported Python version, so that the lock files work on all versions.
LOCK_PYTHON := 3.10
# Resolve for every platform and interpreter from LOCK_PYTHON upwards.
LOCK_FLAGS := --universal --python-version $(LOCK_PYTHON)

# Directory of the hashed lock files used by the GitHub Actions workflows.
CI_REQ := .github/requirements

# Every lock file that `make lock` writes and `make lock-check` compares.
LOCK_FILES := requirements.txt requirements-dev.txt $(CI_REQ)/requirements-ci-test.txt \
	$(CI_REQ)/requirements-ci-tools.txt $(CI_REQ)/requirements-ci-fuzz.txt

# Targets that do not create a file of the same name.
.PHONY: help install install-dev lock lock-check test test-fast test-cov \
        lint format format-check type-check security licence text-policy \
        yaml-lint md-lint notebooks model-zoo check pre-commit build check-dist docker-build docker-run docker-api \
        validate verify clean distclean

##@ Help

help: ## Show this help
# Print the "##@" group headings and every "target: ## text" line.
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@ / {printf "\n%s\n", substr($$0, 5); next} \
		/^[a-zA-Z0-9_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Installation

install: ## Install the package with the locked runtime dependencies
# Install the exact runtime versions from the lock file.
	$(PIP) install -r requirements.txt
# Install the package itself without resolving dependencies again.
	$(PIP) install --no-deps .

install-dev: ## Install the locked development environment and pre-commit hooks
# Install the exact development versions from the lock file.
	$(PIP) install -r requirements-dev.txt
# Install the package in editable mode without resolving dependencies again.
	$(PIP) install --no-deps -e .
# Register the Git hooks of .pre-commit-config.yaml.
	pre-commit install

lock: ## Regenerate requirements.txt, requirements-dev.txt and the hashed CI lock files
# Compile the runtime lock with the onnx and serving extras and hashes into a temporary file.
	$(UV) pip compile pyproject.toml $(LOCK_FLAGS) --generate-hashes --extra onnx --extra serving \
		--custom-compile-command "uv pip compile pyproject.toml $(LOCK_FLAGS) --generate-hashes --extra onnx --extra serving -o requirements.txt" \
		-o .requirements.lock.tmp
# Compile the development lock with every extra into a temporary file.
	$(UV) pip compile pyproject.toml $(LOCK_FLAGS) --extra all \
		--custom-compile-command "uv pip compile pyproject.toml $(LOCK_FLAGS) --extra all -o requirements-dev.txt" \
		-o .requirements-dev.lock.tmp
# Compile the hashed lock of the CI test environment (the dependencies of PyTorch, not
# PyTorch itself) into a temporary file.
	$(UV) pip compile pyproject.toml $(CI_REQ)/requirements-ci-test.in $(LOCK_FLAGS) --generate-hashes --extra test --extra onnx --extra serving --extra zarr --extra parquet --extra stac \
		--custom-compile-command "make lock" \
		-o .requirements-ci-test.lock.tmp
# Compile the hashed lock of the CI tools into a temporary file.
	$(UV) pip compile $(CI_REQ)/requirements-ci-tools.in $(LOCK_FLAGS) --generate-hashes \
		--custom-compile-command "make lock" \
		-o .requirements-ci-tools.lock.tmp
# Compile the hashed lock of the fuzzing job into a temporary file.
	$(UV) pip compile $(CI_REQ)/requirements-ci-fuzz.in $(LOCK_FLAGS) --generate-hashes \
		--custom-compile-command "make lock" \
		-o .requirements-ci-fuzz.lock.tmp
# Replace the uv output, keeping the header and the footer of each lock file.
	@$(PYTHON) scripts/merge_lock.py \
		requirements.txt .requirements.lock.tmp \
		requirements-dev.txt .requirements-dev.lock.tmp \
		$(CI_REQ)/requirements-ci-test.txt .requirements-ci-test.lock.tmp \
		$(CI_REQ)/requirements-ci-tools.txt .requirements-ci-tools.lock.tmp \
		$(CI_REQ)/requirements-ci-fuzz.txt .requirements-ci-fuzz.lock.tmp
# Remove the temporary files.
	@rm -f .requirements.lock.tmp .requirements-dev.lock.tmp .requirements-ci-*.lock.tmp
# Remind the maintainer to review the new pins.
	@echo "Lock files regenerated. Review the diff before committing."

lock-check: ## Fail if a lock file is out of date with its input
# Keep copies of the committed lock files.
	@for f in $(LOCK_FILES); do cp "$$f" "$$f.orig"; done
# Regenerate the lock files in place.
	@$(MAKE) --no-print-directory lock > /dev/null
# Compare with the copies, restore the committed files and report the result.
	@status=0; \
	for f in $(LOCK_FILES); do cmp -s "$$f" "$$f.orig" || { echo "Outdated: $$f"; status=1; }; mv "$$f.orig" "$$f"; done; \
	if [ $$status -ne 0 ]; then echo "Lock files are outdated: run make lock"; exit 1; fi; \
	echo "Lock files are up to date."

##@ Tests

test: ## Run the complete test suite
# Run every test with the options from pyproject.toml.
	$(PYTHON) -m pytest tests/

test-fast: ## Run unit tests in parallel, skipping slow and GPU tests
# Distribute the unit tests over all CPU cores with pytest-xdist.
	$(PYTHON) -m pytest tests/unit -n auto -m "not slow and not gpu"

test-cov: ## Run the test suite with branch coverage (terminal and XML report)
# Report missing lines in the terminal and write coverage.xml for Codecov.
	$(PYTHON) -m pytest tests/ --cov=$(SRC) --cov-report=term-missing --cov-report=xml

##@ Code quality

lint: ## Lint the code with ruff
# Apply the ruff rules configured in pyproject.toml.
	ruff check src/ tests/

format: ## Format the code with ruff and apply safe lint fixes
# Reformat the sources.
	ruff format src/ tests/
# Apply the automatic fixes that ruff considers safe.
	ruff check --fix src/ tests/

format-check: ## Check formatting without changing files
# Fail if any file would be reformatted.
	ruff format --check src/ tests/

type-check: ## Type check the package with pyright
# Check the package with the pyright settings from pyproject.toml.
	pyright src/

security: ## Run bandit and pip-audit
# Scan the package for insecure code patterns.
	bandit -c pyproject.toml -r src/
# Check the runtime lock file against the vulnerability databases.
	pip-audit -r requirements.txt --no-deps --disable-pip --progress-spinner off
# Check the development lock file against the vulnerability databases.
	pip-audit -r requirements-dev.txt --no-deps --disable-pip --progress-spinner off

licence: ## Check REUSE compliance and the MPL-2.0 notices
# reuse runs in the isolated pre-commit environment, which also copies
# LICENSE.txt to the git-ignored LICENSES/MPL-2.0.txt that REUSE expects.
	pre-commit run reuse-lint-file --all-files
# Check the MPL-2.0 notice at the top of every source file.
	$(PYTHON) .github/scripts/check_license_headers.py

text-policy: ## Check for emojis, em dashes and non-English letters
# Scan every tracked text file for forbidden characters.
	$(PYTHON) .github/scripts/check_text_policy.py

yaml-lint: ## Lint all YAML files with yamllint
# Run the yamllint hook with the rules of .yamllint.yml.
	pre-commit run yamllint --all-files

md-lint: ## Lint all Markdown files with markdownlint
# Run a pinned markdownlint-cli with the rules of .markdownlint.yaml.
	npx --yes markdownlint-cli@0.49.1 --config .markdownlint.yaml "**/*.md" --ignore node_modules

notebooks: ## Validate the example notebooks (format, no outputs)
# Check the notebook format and that no outputs are committed.
	$(PYTHON) .github/scripts/check_notebooks.py examples/notebooks

model-zoo: ## Check model zoo structure, checksums, cards and manifests
# Check the model zoo against its catalogue and checksums.
	$(PYTHON) .github/scripts/check_model_zoo.py

check: lint format-check type-check test licence text-policy yaml-lint notebooks model-zoo ## Run all local checks that CI runs

pre-commit: ## Run every pre-commit hook on all files
# Run all hooks, not only those for changed files.
	pre-commit run --all-files

##@ Packaging and containers

build: ## Build the sdist and wheel into dist/
# Build both distributions in an isolated environment.
	$(PYTHON) -m build

check-dist: build ## Build and validate the distributions with twine
# Validate the metadata and the rendered README, treating warnings as errors.
	$(PYTHON) -m twine check --strict dist/*

docker-build: ## Build the container image
# Build the image, passing the package version, the commit and the build time
# as OCI metadata.
	docker build \
		--build-arg VERSION="$$($(PYTHON) -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')" \
		--build-arg VCS_REF="$$(git rev-parse HEAD)" \
		--build-arg BUILD_DATE="$$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
		-t $(IMAGE) .

docker-run: ## Run the command line interface in the container
# Show the command line help from a throwaway container.
	docker run --rm $(IMAGE) unbihexium --help

docker-api: ## Run the REST API in the container on http://localhost:$(PORT)
# Serve the API on port 8000 in the container, published on PORT of the host.
	docker run --rm -p $(PORT):8000 $(IMAGE) \
		uvicorn unbihexium.serving.app:app --host 0.0.0.0 --port 8000

##@ Model zoo and verification

validate: ## Load and run every model (requires Git LFS weights)
# Build, run and compare every model of the zoo.
	$(PYTHON) scripts/validate_models.py

verify: ## Print the installed version and the number of registered models
# Print the version of the installed package.
	$(PYTHON) -c "import unbihexium; print(f'Version: {unbihexium.__version__}')"
# Print the number of models in the registry.
	$(PYTHON) -c "from unbihexium.zoo import list_models; print(f'Models: {len(list_models())}')"

##@ Cleaning

clean: ## Remove build artefacts and caches
# Remove build output, tool caches and coverage reports.
	rm -rf build/ dist/ site/ *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov/
# Remove every bytecode cache directory.
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
# Remove stray bytecode files outside the cache directories.
	find . -type f -name "*.py[co]" -delete

distclean: clean ## Also remove virtual environments and tox environments
# Remove the local virtual environments and the tox and nox environments.
	rm -rf .venv/ venv/ .tox/ .nox/

# =============================================================================
# End of file Makefile
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
