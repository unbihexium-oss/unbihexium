# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Configuration management module.

This module provides configuration classes and utilities
for managing library settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Model configuration."""

    variant: str = "large"
    device: str = "cuda:0"
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class ProcessingConfig:
    """Processing configuration."""

    tile_size: int = 512
    overlap: int = 64
    output_format: str = "GTiff"
    compression: str = "DEFLATE"


@dataclass
class Config:
    """Main configuration class."""

    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**data.get("model", {})),
            processing=ProcessingConfig(**data.get("processing", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "model": self.model.__dict__,
            "processing": self.processing.__dict__,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def update(self, **kwargs: Any) -> Config:
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.processing, key):
                setattr(self.processing, key, value)
        return self


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


__all__ = [
    "ModelConfig",
    "ProcessingConfig",
    "Config",
    "get_default_config",
]
