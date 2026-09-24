# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_config.py
# Title       : Tests of the layered configuration
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest and PyYAML
# =============================================================================
#
# Abstract
# --------
# Defaults validate; YAML files and UNBIHEXIUM_* environment variables are
# applied in order with type conversion; unknown keys and invalid values
# are reported together; update() accepts plain and qualified keys.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Represent file paths.
from pathlib import Path

# Test framework.
import pytest

# Configuration under test.
from unbihexium.config import (
    Config,  # Configuration.
    get_default_config,  # Defaults.
    load_config,  # Layered loading.
)  # End of the configuration imports.


# Defaults are valid and serialisable.
def test_defaults() -> None:
    # Default configuration.
    config = get_default_config()
    # Valid.
    assert config.problems() == []
    # Section values.
    assert config.model.variant == "base" and config.model.device == "cpu"
    # 10 MiB body limit of the service.
    assert config.serving.max_request_bytes == 10 * 1024 * 1024
    # Dictionary round trip.
    assert Config.from_dict(config.to_dict()) == config


# File, environment and overrides are applied in order.
def test_layers(tmp_path: Path) -> None:
    # YAML file with two sections.
    path = tmp_path / "config.yaml"
    # File content.
    path.write_text("model:\n  batch_size: 2\n  backend: onnx\n", encoding="utf-8")
    # Environment overriding the batch size and setting a list and None.
    env = {
        "UNBIHEXIUM_MODEL__BATCH_SIZE": "16",  # Integer from text.
        "UNBIHEXIUM_SERVING__CORS_ORIGINS": "https://a.org, https://b.org",  # List.
        "UNBIHEXIUM_SERVING__API_KEY": "none",  # Optional value.
        "UNBIHEXIUM_LOG_LEVEL": "debug",  # Top-level value.
        "UNBIHEXIUM_CACHE": "/tmp/cache",  # Unrelated variable, ignored.
    }  # End of the environment.
    # Layers: file, environment, overrides.
    config = load_config(path, environ=env, overrides={"model": {"variant": "tiny"}})
    # Environment wins over the file.
    assert config.model.batch_size == 16
    # File value without an environment variable.
    assert config.model.backend == "onnx"
    # Override applied last.
    assert config.model.variant == "tiny"
    # Converted values.
    assert config.serving.cors_origins == ["https://a.org", "https://b.org"]
    # None from text.
    assert config.serving.api_key is None
    # Level in upper case.
    assert config.log_level == "DEBUG"
    # The file path can come from UNBIHEXIUM_CONFIG.
    assert load_config(environ={"UNBIHEXIUM_CONFIG": str(path)}).model.batch_size == 2
    # Written YAML reads back equal.
    config.to_yaml(tmp_path / "out.yaml")
    # Round trip.
    assert Config.from_yaml(tmp_path / "out.yaml") == config


# Invalid settings are reported together; unknown keys are errors.
def test_validation(tmp_path: Path) -> None:
    # Two problems at once.
    bad = Config.from_dict({"model": {"variant": "huge", "batch_size": 0}})
    # Both are listed.
    with pytest.raises(ValueError, match=r"model\.variant.*model\.batch_size"):
        # Validate.
        bad.validate()
    # Port range.
    closed = Config.from_dict({"serving": {"port": 0}})
    # Reported.
    assert any("serving.port" in p for p in closed.problems())
    # Misspelt keys.
    with pytest.raises(ValueError, match=r"unknown setting model\.batchsize"):
        # Unknown key.
        Config.from_dict({"model": {"batchsize": 4}})
    # Unknown sections.
    with pytest.raises(ValueError, match="unknown configuration section"):
        # Unknown section.
        Config.from_dict({"modle": {}})
    # Fractions are not integers.
    with pytest.raises(ValueError, match="integer"):
        # 2.5 tiles.
        load_config(env=False, overrides={"model": {"batch_size": "2.5"}})
    # CUDA device indices are accepted, other names are not.
    assert Config.from_dict({"model": {"device": "cuda:1"}}).problems() == []
    # Unknown device.
    assert Config.from_dict({"model": {"device": "tpu"}}).problems()
    # YAML files must hold a mapping.
    (tmp_path / "list.yaml").write_text("- 1\n", encoding="utf-8")
    # Rejected.
    with pytest.raises(ValueError, match="mapping"):
        # Load the list.
        Config.from_yaml(tmp_path / "list.yaml")


# update() accepts plain and qualified keys and validates.
def test_update() -> None:
    # Defaults.
    config = Config()
    # Plain key of the model section and a qualified serving key.
    config.update(batch_size=4, **{"serving.port": 9000, "serving__rate_limit_per_minute": 30})
    # Values applied in place.
    assert config.model.batch_size == 4 and config.serving.port == 9000
    # Double underscore form.
    assert config.serving.rate_limit_per_minute == 30
    # Invalid values are rejected.
    with pytest.raises(ValueError, match=r"serving\.port"):
        # Port out of range.
        config.update(port=70000)
    # The failed update left the object unchanged.
    assert config.serving.port == 9000


# Settings that nothing read were removed; setting them fails with their name.
def test_removed_settings(tmp_path: Path) -> None:
    # Removed section in a file.
    path = tmp_path / "old.yaml"
    # Section of earlier releases.
    path.write_text("processing:\n  tile_size: 256\n", encoding="utf-8")
    # The message names the key.
    with pytest.raises(ValueError, match=r"processing\.tile_size was removed"):
        # Load the file.
        load_config(path, env=False)
    # Removed key in the environment.
    with pytest.raises(ValueError, match=r"model\.num_workers was removed"):
        # Load with the variable.
        load_config(environ={"UNBIHEXIUM_MODEL__NUM_WORKERS": "2"})
    # Removed key given to update().
    with pytest.raises(ValueError, match=r"model\.num_workers was removed"):
        # Plain key.
        Config().update(num_workers=2)


# The API key is redacted in dictionaries and files unless requested.
def test_secret_redaction(tmp_path: Path) -> None:
    # Configuration with a key.
    config = Config.from_dict({"serving": {"api_key": "s3cret"}})
    # The object keeps the key.
    assert config.serving.api_key == "s3cret"
    # Dictionaries hide it by default.
    assert config.to_dict()["serving"]["api_key"] == "***"
    # Explicit opt-in.
    assert config.to_dict(include_secrets=True)["serving"]["api_key"] == "s3cret"
    # Written files hide it by default.
    text = config.to_yaml(tmp_path / "c.yaml").read_text(encoding="utf-8")
    # Not in the file.
    assert "s3cret" not in text
    # A file written with the key reads back equal.
    config.to_yaml(tmp_path / "full.yaml", include_secrets=True)
    # Round trip.
    assert Config.from_yaml(tmp_path / "full.yaml") == config


# =============================================================================
# End of module tests/unit/test_config.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
