# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/config/settings.py
# Title       : Validated configuration from defaults, YAML and environment
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyYAML
# =============================================================================
#
# Abstract
# --------
# Config groups the settings of the library in three sections:
#
#   model        model zoo variant, device, backend, batch size, workers
#   processing   tile size and overlap, output format, compression, seed
#   serving      REST service: bind address, request limits, API key, CORS,
#                rate limit and model cache size
#
# plus the top-level log_level. Settings are layered, later layers winning:
#
#   1. defaults of the dataclasses
#   2. a YAML file (load_config(path) or the UNBIHEXIUM_CONFIG variable)
#   3. environment variables UNBIHEXIUM_<SECTION>__<KEY>, for example
#      UNBIHEXIUM_MODEL__BATCH_SIZE=16 or UNBIHEXIUM_SERVING__PORT=9000,
#      and UNBIHEXIUM_LOG_LEVEL for the top-level key
#   4. explicit overrides passed as a dictionary
#
# Values from text sources (the environment) are converted to the type of
# the field: integers, floats, booleans (true/false, yes/no, on/off, 1/0),
# comma-separated lists and "none" or "null" for optional fields. Unknown
# sections or keys are errors, which catches misspelt settings early.
# validate() reports every problem at once in a single ValueError.
#
# References
# ----------
# Wiggins, A. (2017). The Twelve-Factor App, III. Config: store config in
# the environment. https://12factor.net/config
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Dataclass machinery.
import dataclasses

# Environment variables.
import os

# Dataclasses with defaults.
from dataclasses import dataclass, field

# Cache of the process-wide settings.
from functools import lru_cache

# Represent file paths.
from pathlib import Path

# Types of loosely structured values and annotations.
from typing import Any, Mapping, get_args, get_origin, get_type_hints

# YAML files.
import yaml

# Atomic file writes.
from unbihexium.utils.files import atomic_write_text

# Prefix of the environment variables.
ENV_PREFIX = "UNBIHEXIUM_"

# Environment variable with the path of a YAML configuration file.
CONFIG_ENV = "UNBIHEXIUM_CONFIG"

# Size variants of the model zoo.
VARIANTS = ("tiny", "base", "large", "mega")

# Inference backends of the model zoo.
BACKENDS = ("auto", "torch", "onnx")

# Output raster formats.
OUTPUT_FORMATS = ("GTiff", "COG", "Zarr")

# GDAL GeoTIFF compression methods that are lossless.
COMPRESSIONS = ("NONE", "LZW", "DEFLATE", "ZSTD", "LZMA", "PACKBITS")

# Logging levels.
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# Text values accepted as true.
TRUE_WORDS = ("1", "true", "yes", "on")

# Text values accepted as false.
FALSE_WORDS = ("0", "false", "no", "off")


# Model selection and execution.
@dataclass
class ModelConfig:
    # Size variant of catalogue models.
    variant: str = "base"
    # Torch device: cpu, cuda, cuda:<n> or mps.
    device: str = "cpu"
    # Inference backend.
    backend: str = "auto"
    # Tiles per forward pass.
    batch_size: int = 8
    # Data loader workers for training.
    num_workers: int = 4


# Raster processing.
@dataclass
class ProcessingConfig:
    # Tile side in pixels.
    tile_size: int = 512
    # Overlap of neighbouring tiles in pixels.
    overlap: int = 64
    # Raster output format.
    output_format: str = "GTiff"
    # GeoTIFF compression.
    compression: str = "DEFLATE"
    # No-data value of written rasters, None for none.
    nodata: float | None = None
    # Seed of random operations, None for non-reproducible runs.
    seed: int | None = None


# REST service.
@dataclass
class ServingConfig:
    # Bind address; 127.0.0.1 unless the service is behind a proxy.
    host: str = "127.0.0.1"
    # TCP port.
    port: int = 8000
    # Largest accepted request body in bytes (10 MiB).
    max_request_bytes: int = 10 * 1024 * 1024
    # Largest accepted image in pixels (height times width).
    max_pixels: int = 2048 * 2048
    # Largest accepted number of values (bands times pixels).
    max_values: int = 16 * 1024 * 1024
    # API key required in the X-API-Key header, None for an open service.
    api_key: str | None = None
    # Origins allowed by CORS.
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    # Requests per minute and client, 0 for no limit.
    rate_limit_per_minute: int = 0
    # Number of opened models kept in memory.
    model_cache_size: int = 4


# Sections of the configuration and their classes.
SECTIONS: dict[str, type] = {
    "model": ModelConfig,  # Model selection.
    "processing": ProcessingConfig,  # Raster processing.
    "serving": ServingConfig,  # REST service.
}  # End of the sections.


# Whether an annotation accepts None.
def _optional(hint: Any) -> bool:
    # Optional annotations are unions with NoneType.
    return type(None) in get_args(hint)


# Type of a field without the None of optional annotations.
def _base(hint: Any) -> Any:
    # Plain annotations are their own base.
    if not _optional(hint):
        # Return the annotation.
        return hint
    # First member that is not NoneType.
    return next(a for a in get_args(hint) if a is not type(None))


# Convert a value to the type of a field; text values are parsed.
def _coerce(name: str, hint: Any, value: Any) -> Any:
    # Base type of the field.
    base = _base(hint)
    # None and its text forms for optional fields.
    if value is None or (isinstance(value, str) and value.strip().lower() in ("none", "null")):
        # Only optional fields accept None.
        if _optional(hint):
            # Keep None.
            return None
        # Explain the problem.
        raise ValueError(f"{name} must not be empty")
    # Lists: text is split at commas.
    if get_origin(base) is list:
        # Split text values.
        items = value.split(",") if isinstance(value, str) else list(value)
        # Strip the items and drop empty ones.
        return [str(v).strip() for v in items if str(v).strip()]
    # Booleans from words.
    if base is bool:
        # Already a boolean.
        if isinstance(value, bool):
            # Keep it.
            return value
        # Normalised text.
        word = str(value).strip().lower()
        # True words.
        if word in TRUE_WORDS:
            # True.
            return True
        # False words.
        if word in FALSE_WORDS:
            # False.
            return False
        # Explain the accepted words.
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    # Integers; floats with a fraction are rejected.
    if base is int:
        # Booleans are not integers here.
        if isinstance(value, bool):
            # Explain the problem.
            raise ValueError(f"{name} must be an integer, got {value!r}")
        # Parse text and numbers.
        try:
            # Through float to accept "1e3".
            number = float(value)
        # Unparseable text.
        except (TypeError, ValueError) as exc:
            # Explain the problem.
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
        # Fractions are errors.
        if not number.is_integer():
            # Explain the problem.
            raise ValueError(f"{name} must be an integer, got {value!r}")
        # Integer value.
        return int(number)
    # Floating point numbers.
    if base is float:
        # Parse text and numbers.
        try:
            # Float value.
            return float(value)
        # Unparseable text.
        except (TypeError, ValueError) as exc:
            # Explain the problem.
            raise ValueError(f"{name} must be a number, got {value!r}") from exc
    # Text.
    return str(value)


# Build a section object from a mapping, converting every value.
def _section(section: str, values: Mapping[str, Any], base: Any = None) -> Any:
    # Class of the section.
    cls = SECTIONS[section]
    # Resolved field annotations.
    hints = get_type_hints(cls)
    # Start from the given object or the defaults.
    current = dataclasses.asdict(base) if base is not None else dataclasses.asdict(cls())
    # Apply every value.
    for key, value in values.items():
        # Unknown keys are errors.
        if key not in hints:
            # List the accepted keys.
            raise ValueError(f"unknown setting {section}.{key}; known: {', '.join(hints)}")
        # Converted value.
        current[key] = _coerce(f"{section}.{key}", hints[key], value)
    # New section object.
    return cls(**current)


# Complete configuration of the library.
@dataclass
class Config:
    # Model selection and execution.
    model: ModelConfig = field(default_factory=ModelConfig)
    # Raster processing.
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    # REST service.
    serving: ServingConfig = field(default_factory=ServingConfig)
    # Level of the library logger.
    log_level: str = "WARNING"

    # List of the problems of the configuration.
    def problems(self) -> list[str]:
        # Collected problems.
        found: list[str] = []
        # Short names of the sections.
        m, p, s = self.model, self.processing, self.serving
        # Known variant.
        if m.variant not in VARIANTS:
            # Report it.
            found.append(f"model.variant must be one of {', '.join(VARIANTS)}")
        # Device syntax.
        if not (m.device in ("cpu", "cuda", "mps") or _is_cuda_index(m.device)):
            # Report it.
            found.append("model.device must be cpu, cuda, cuda:<n> or mps")
        # Known backend.
        if m.backend not in BACKENDS:
            # Report it.
            found.append(f"model.backend must be one of {', '.join(BACKENDS)}")
        # Positive batch size.
        if m.batch_size < 1:
            # Report it.
            found.append("model.batch_size must be at least 1")
        # Non-negative worker count.
        if m.num_workers < 0:
            # Report it.
            found.append("model.num_workers must not be negative")
        # Positive tile size.
        if p.tile_size < 1:
            # Report it.
            found.append("processing.tile_size must be at least 1")
        # Overlap below the tile size.
        if not 0 <= p.overlap < max(p.tile_size, 1):
            # Report it.
            found.append("processing.overlap must be in [0, tile_size)")
        # Known output format.
        if p.output_format not in OUTPUT_FORMATS:
            # Report it.
            found.append(f"processing.output_format must be one of {', '.join(OUTPUT_FORMATS)}")
        # Known lossless compression, case-insensitive.
        if p.compression.upper() not in COMPRESSIONS:
            # Report it.
            found.append(f"processing.compression must be one of {', '.join(COMPRESSIONS)}")
        # Seed range of NumPy.
        if p.seed is not None and not 0 <= p.seed < 2**32:
            # Report it.
            found.append("processing.seed must be in [0, 2**32)")
        # Valid TCP port.
        if not 1 <= s.port <= 65535:
            # Report it.
            found.append("serving.port must be in [1, 65535]")
        # Positive limits.
        for key in ("max_request_bytes", "max_pixels", "max_values", "model_cache_size"):
            # Every limit must be positive.
            if getattr(s, key) < 1:
                # Report it.
                found.append(f"serving.{key} must be at least 1")
        # Non-negative rate limit.
        if s.rate_limit_per_minute < 0:
            # Report it.
            found.append("serving.rate_limit_per_minute must not be negative")
        # Empty API keys would accept an empty header.
        if s.api_key is not None and not s.api_key.strip():
            # Report it.
            found.append("serving.api_key must not be empty; use None for an open service")
        # Known log level.
        if self.log_level.upper() not in LOG_LEVELS:
            # Report it.
            found.append(f"log_level must be one of {', '.join(LOG_LEVELS)}")
        # Return the problems.
        return found

    # Raise a ValueError listing every problem.
    def validate(self) -> Config:
        # Problems of the configuration.
        found = self.problems()
        # Report them together.
        if found:
            # One message with every problem.
            raise ValueError("invalid configuration: " + "; ".join(found))
        # Allow chaining.
        return self

    # Plain dictionary of the configuration.
    def to_dict(self) -> dict[str, Any]:
        # Recursive conversion of the dataclasses.
        return dataclasses.asdict(self)

    # Configuration from a nested dictionary; missing values keep their defaults.
    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Config:
        # Defaults.
        return cls().merged(data or {})

    # Copy with values of a nested dictionary applied.
    def merged(self, data: Mapping[str, Any]) -> Config:
        # Sections of the copy.
        sections = {name: getattr(self, name) for name in SECTIONS}
        # Top-level level.
        log_level = self.log_level
        # Apply every entry.
        for key, value in data.items():
            # Sections need a mapping.
            if key in SECTIONS:
                # Reject scalars in place of sections.
                if not isinstance(value, Mapping):
                    # Explain the problem.
                    raise ValueError(f"section {key} must be a mapping")
                # Updated section.
                sections[key] = _section(key, value, sections[key])
            # The top-level key.
            elif key == "log_level":
                # Text value.
                log_level = str(value).strip().upper()
            # Anything else is unknown.
            else:
                # Explain the problem.
                raise ValueError(f"unknown configuration section {key!r}")
        # New configuration.
        return Config(log_level=log_level, **sections)

    # Load and validate a YAML file.
    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        # Parse the file safely (no arbitrary objects).
        with open(path, encoding="utf-8") as handle:
            # Document of the file; empty files give None.
            data = yaml.safe_load(handle) or {}
        # The document must be a mapping.
        if not isinstance(data, Mapping):
            # Explain the problem.
            raise ValueError(f"{path}: the configuration must be a mapping")
        # Build and validate.
        return cls.from_dict(data).validate()

    # Write the configuration as YAML.
    def to_yaml(self, path: str | Path) -> Path:
        # Block-style YAML with the field order of the dataclasses.
        text = yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        # Write atomically.
        return atomic_write_text(path, text)

    # Configuration with values of UNBIHEXIUM_* environment variables applied.
    def with_env(self, environ: Mapping[str, str] | None = None) -> Config:
        # Environment to read.
        env = os.environ if environ is None else environ
        # Nested values found.
        data: dict[str, Any] = {}
        # Visit every variable.
        for name, value in env.items():
            # Only prefixed variables.
            if not name.startswith(ENV_PREFIX):
                # Skip it.
                continue
            # Name without the prefix, in lower case.
            key = name[len(ENV_PREFIX) :].lower()
            # Section variables use a double underscore.
            section, sep, item = key.partition("__")
            # Section settings.
            if sep and section in SECTIONS:
                # Record the value.
                data.setdefault(section, {})[item] = value
            # The top-level log level.
            elif key == "log_level":
                # Record it.
                data["log_level"] = value
        # Apply the values.
        return self.merged(data)

    # Update values by key; keys are "key", "section.key" or "section__key".
    def update(self, **kwargs: Any) -> Config:
        # Nested values.
        data: dict[str, Any] = {}
        # Visit every key.
        for key, value in kwargs.items():
            # Explicit section.
            section, sep, item = key.replace("__", ".").partition(".")
            # Qualified keys.
            if sep:
                # Record the value.
                data.setdefault(section, {})[item] = value
                # Next key.
                continue
            # The top-level level.
            if key == "log_level":
                # Record it.
                data["log_level"] = value
                # Next key.
                continue
            # Sections that define the key; the first one wins.
            owner = next((n for n, c in SECTIONS.items() if key in get_type_hints(c)), None)
            # Unknown keys are errors.
            if owner is None:
                # Explain the problem.
                raise ValueError(f"unknown setting {key!r}")
            # Record the value.
            data.setdefault(owner, {})[key] = value
        # Updated copy.
        updated = self.merged(data).validate()
        # Change this object in place, as earlier releases did.
        self.model, self.processing = updated.model, updated.processing
        # Remaining fields.
        self.serving, self.log_level = updated.serving, updated.log_level
        # Allow chaining.
        return self


# Whether a device name is cuda:<n>.
def _is_cuda_index(device: str) -> bool:
    # Split the prefix and the index.
    prefix, sep, index = device.partition(":")
    # A cuda prefix with a decimal index.
    return prefix == "cuda" and bool(sep) and index.isdigit()


# Default configuration.
def get_default_config() -> Config:
    # Defaults of the dataclasses.
    return Config()


# Layered configuration: defaults, YAML file, environment, overrides.
def load_config(
    path: str | Path | None = None,  # YAML file; default UNBIHEXIUM_CONFIG if set.
    env: bool = True,  # Apply UNBIHEXIUM_* environment variables.
    overrides: Mapping[str, Any] | None = None,  # Nested values applied last.
    environ: Mapping[str, str] | None = None,  # Environment; default os.environ.
) -> Config:  # Validated configuration.
    # Environment to read.
    variables = os.environ if environ is None else environ
    # File from the argument or the environment.
    source = path if path is not None else (variables.get(CONFIG_ENV) if env else None)
    # Defaults or the file.
    config = Config.from_dict(_read_yaml(source)) if source else Config()
    # Environment variables.
    if env:
        # Apply them.
        config = config.with_env(variables)
    # Explicit overrides.
    if overrides:
        # Apply them.
        config = config.merged(overrides)
    # Validate the result.
    return config.validate()


# Read a YAML mapping.
def _read_yaml(path: str | Path) -> dict[str, Any]:
    # Parse the file safely.
    with open(path, encoding="utf-8") as handle:
        # Document; empty files give None.
        data = yaml.safe_load(handle) or {}
    # The document must be a mapping.
    if not isinstance(data, dict):
        # Explain the problem.
        raise ValueError(f"{path}: the configuration must be a mapping")
    # Return the mapping.
    return data


# Process-wide settings, loaded once from the file and the environment.
@lru_cache(maxsize=1)
def get_settings() -> Config:
    # Layered configuration.
    return load_config()


# Forget the cached settings, for example after changing the environment.
def reset_settings() -> None:
    # Clear the cache.
    get_settings.cache_clear()


# =============================================================================
# End of module src/unbihexium/config/settings.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
