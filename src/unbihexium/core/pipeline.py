# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/core/pipeline.py
# Title       : Processing pipelines with run records and provenance
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# A pipeline is an ordered list of named steps. Each step is a callable that
# takes a dictionary of values and returns a new dictionary; the output of
# one step is the input of the next. Running a pipeline produces a
# PipelineRun that records:
#
#   status          pending, running, completed, failed or cancelled
#   steps           name, status, start and end time and duration per step
#   inputs/outputs  the values as strings (paths stay readable)
#   results         the output values themselves (not serialised)
#   seed            the seed of the random number generators
#   provenance      a ProvenanceRecord with SHA-256 evidence of every input
#                   and output that is a file
#
# Reproducibility: when the configuration has a seed (the `seed` field or a
# `seed` parameter), the Python, NumPy and, when loaded, PyTorch generators
# are seeded before the first step, so repeated runs with the same inputs
# give the same outputs.
#
# Usage
# -----
#   pipeline = Pipeline(PipelineConfig("ndvi", "NDVI", seed=0))
#   pipeline.add_step(read).add_step(compute).add_step(write)
#   run = pipeline.run({"input": "scene.tif"})
#   run.to_json("runs/ndvi.json")
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON serialisation.
import json

# Python random number generator.
import random

# Loaded modules, to seed PyTorch only when it is in use.
import sys

# Monotonic clock for durations.
import time

# Unique run identifiers.
import uuid

# Step callables and mappings.
from collections.abc import Callable, Mapping

# Record containers.
from dataclasses import dataclass, field

# Timestamps.
from datetime import datetime, timezone

# Run states.
from enum import Enum

# File paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays and the NumPy random number generator.
import numpy as np

# Provenance records.
from unbihexium.core.evidence import (
    Evidence,  # One artefact and its digest.
    EvidenceType,  # Role of an artefact.
    ProvenanceRecord,  # Provenance of a run.
    capture_environment,  # Software environment.
)  # End of the evidence imports.

# A pipeline step: values in, values out.
Step = Callable[[dict[str, Any]], Mapping[str, Any]]


# States of a run or a step.
class PipelineStatus(str, Enum):
    # Created but not started.
    PENDING = "pending"
    # Executing.
    RUNNING = "running"
    # Finished without error.
    COMPLETED = "completed"
    # Stopped by an error.
    FAILED = "failed"
    # Stopped by the caller.
    CANCELLED = "cancelled"


# Current time in UTC.
def _now() -> datetime:
    # Timezone-aware timestamp.
    return datetime.now(timezone.utc)


# Seed the Python, NumPy and (when loaded) PyTorch random number generators.
def seed_everything(seed: int) -> None:
    # Seeds must be non-negative integers that fit NumPy's legacy generator.
    if not 0 <= int(seed) < 2**32:
        # Explain the problem.
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")
    # Python generator.
    random.seed(int(seed))
    # NumPy legacy global generator, used by np.random.* functions.
    np.random.seed(int(seed))
    # PyTorch, only when the caller already imported it.
    torch = sys.modules.get("torch")
    # Seed the CPU and GPU generators.
    if torch is not None:
        # Seeds every device.
        torch.manual_seed(int(seed))


# Configuration of a pipeline.
@dataclass
class PipelineConfig:
    # Identifier of the pipeline.
    pipeline_id: str
    # Human-readable name.
    name: str
    # Description.
    description: str = ""
    # Version of the pipeline definition.
    version: str = "1.0.0"
    # Planned step names, for documentation and validation.
    steps: list[str] = field(default_factory=list)
    # Parameters of the steps.
    parameters: dict[str, Any] = field(default_factory=dict)
    # Seed of the random number generators, or None to leave them alone.
    seed: int | None = None
    # Whether runs must be reproducible (requires a seed to take effect).
    deterministic: bool = True

    # Validate the fields.
    def __post_init__(self) -> None:
        # The identifier names the pipeline in registries and records.
        if not self.pipeline_id:
            # Explain the problem.
            raise ValueError("pipeline_id must not be empty")
        # Negative seeds are rejected by NumPy.
        if self.seed is not None and self.seed < 0:
            # Explain the problem.
            raise ValueError(f"seed must be non-negative, got {self.seed}")

    # Seed in effect: the seed field, else a "seed" parameter, else None.
    @property
    def effective_seed(self) -> int | None:
        # The explicit field wins.
        if self.seed is not None:
            # Field value.
            return int(self.seed)
        # Parameter value, when present.
        value = self.parameters.get("seed")
        # Convert to an integer.
        return None if value is None else int(value)

    # Plain dictionary for JSON output.
    def to_dict(self) -> dict[str, Any]:
        # One entry per field.
        return {
            "pipeline_id": self.pipeline_id,  # Identifier.
            "name": self.name,  # Name.
            "description": self.description,  # Description.
            "version": self.version,  # Version.
            "steps": list(self.steps),  # Planned steps.
            "parameters": self.parameters,  # Parameters.
            "seed": self.seed,  # Seed.
            "deterministic": self.deterministic,  # Reproducibility flag.
        }  # End of the dictionary.

    # Configuration from a dictionary written by to_dict.
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineConfig:
        # Build the configuration.
        return cls(
            pipeline_id=data["pipeline_id"],  # Identifier.
            name=data.get("name", data["pipeline_id"]),  # Name.
            description=data.get("description", ""),  # Description.
            version=data.get("version", "1.0.0"),  # Version.
            steps=list(data.get("steps", [])),  # Planned steps.
            parameters=dict(data.get("parameters", {})),  # Parameters.
            seed=data.get("seed"),  # Seed.
            deterministic=bool(data.get("deterministic", True)),  # Reproducibility.
        )  # End of the configuration.


# Record of one step of a run.
@dataclass
class StepRecord:
    # Name of the step.
    name: str
    # State of the step.
    status: PipelineStatus = PipelineStatus.PENDING
    # Start time in ISO 8601.
    started_at: str | None = None
    # End time in ISO 8601.
    finished_at: str | None = None
    # Wall-clock duration from the monotonic clock.
    duration_seconds: float | None = None
    # Error message of a failed step.
    error: str | None = None

    # Plain dictionary for JSON output.
    def to_dict(self) -> dict[str, Any]:
        # One entry per field.
        return {
            "name": self.name,  # Name.
            "status": PipelineStatus(self.status).value,  # State.
            "started_at": self.started_at,  # Start time.
            "finished_at": self.finished_at,  # End time.
            "duration_seconds": self.duration_seconds,  # Duration.
            "error": self.error,  # Error message.
        }  # End of the dictionary.


# Record of one execution of a pipeline.
@dataclass
class PipelineRun:
    # Identifier of the run.
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Identifier of the pipeline.
    pipeline_id: str = ""
    # State of the run; plain strings equal to the enum values are accepted.
    status: PipelineStatus | str = PipelineStatus.PENDING
    # Start time.
    start_time: datetime | None = None
    # End time.
    end_time: datetime | None = None
    # Configuration at the time of the run.
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    # Input values as strings.
    inputs: dict[str, str] = field(default_factory=dict)
    # Output values as strings.
    outputs: dict[str, str] = field(default_factory=dict)
    # Timestamped log messages.
    logs: list[str] = field(default_factory=list)
    # Numeric metrics such as step durations.
    metrics: dict[str, float] = field(default_factory=dict)
    # Error message of a failed run.
    error: str | None = None
    # Seed used for the run.
    seed: int | None = None
    # Records of the executed steps.
    steps: list[StepRecord] = field(default_factory=list)
    # Provenance of the run, built when the run ends.
    provenance: ProvenanceRecord | None = field(default=None, repr=False)
    # Output values themselves; not serialised.
    results: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # Move a pending run to running.
    def start(self) -> None:
        # Only pending runs can start.
        if PipelineStatus(self.status) is not PipelineStatus.PENDING:
            # Explain the problem.
            raise RuntimeError(f"run {self.run_id} is {PipelineStatus(self.status).value}")
        # New state.
        self.status = PipelineStatus.RUNNING
        # Start time.
        self.start_time = _now()

    # Mark the run as completed.
    def complete(self) -> None:
        # New state.
        self.status = PipelineStatus.COMPLETED
        # End time.
        self.end_time = _now()

    # Mark the run as failed with an error message.
    def fail(self, error: str) -> None:
        # New state.
        self.status = PipelineStatus.FAILED
        # End time.
        self.end_time = _now()
        # Error message.
        self.error = error

    # Mark the run as cancelled.
    def cancel(self, reason: str = "") -> None:
        # New state.
        self.status = PipelineStatus.CANCELLED
        # End time.
        self.end_time = _now()
        # Reason, kept in the error field.
        self.error = reason or None

    # Append a timestamped log message.
    def log(self, message: str) -> None:
        # ISO 8601 timestamp and message.
        self.logs.append(f"[{_now().isoformat()}] {message}")

    # Duration of the run in seconds, once it ended.
    @property
    def duration_seconds(self) -> float | None:
        # Both times are needed.
        if self.start_time is None or self.end_time is None:
            # Unknown duration.
            return None
        # Difference of the timestamps.
        return (self.end_time - self.start_time).total_seconds()

    # Plain dictionary for JSON output.
    def to_dict(self) -> dict[str, Any]:
        # ISO strings of the times.
        start = self.start_time.isoformat() if self.start_time else None
        # End time.
        end = self.end_time.isoformat() if self.end_time else None
        # One entry per field.
        return {
            "run_id": self.run_id,  # Identifier.
            "pipeline_id": self.pipeline_id,  # Pipeline.
            "status": PipelineStatus(self.status).value,  # State.
            "start_time": start,  # Start time.
            "end_time": end,  # End time.
            "duration_seconds": self.duration_seconds,  # Duration.
            "config_snapshot": self.config_snapshot,  # Configuration.
            "inputs": self.inputs,  # Inputs.
            "outputs": self.outputs,  # Outputs.
            "logs": list(self.logs),  # Log messages.
            "metrics": self.metrics,  # Metrics.
            "error": self.error,  # Error message.
            "seed": self.seed,  # Seed.
            "steps": [s.to_dict() for s in self.steps],  # Step records.
            "provenance": self.provenance.to_dict() if self.provenance else None,  # Provenance.
        }  # End of the dictionary.

    # JSON text of the run, optionally written to a file.
    def to_json(self, path: str | Path | None = None) -> str:
        # Serialise.
        text = json.dumps(self.to_dict(), indent=2, default=str)
        # Write the file when a path is given.
        if path is not None:
            # Output file.
            target = Path(path)
            # Create the parent directory.
            target.parent.mkdir(parents=True, exist_ok=True)
            # Write the text with a final newline.
            target.write_text(text + "\n", encoding="utf-8")
        # Return the text.
        return text


# String form of a value for the run record.
def _as_text(value: Any) -> str:
    # Paths and strings keep their text; other values use str().
    return str(value)


# Evidence of values that are existing files, plain strings otherwise.
def _artefacts(values: Mapping[str, Any], kind: EvidenceType) -> list[Evidence | str]:
    # Collected artefacts.
    items: list[Evidence | str] = []
    # Visit every value.
    for key, value in values.items():
        # Only strings and paths can name files.
        if isinstance(value, (str, Path)) and str(value) and Path(value).is_file():
            # Evidence with the SHA-256 digest of the file.
            items.append(Evidence.from_file(value, kind, description=key))
        # Other values are recorded by their text.
        else:
            # Key and value.
            items.append(f"{key}={_as_text(value)}")
    # Return the artefacts.
    return items


# Ordered list of named steps.
class Pipeline:
    # Create an empty pipeline.
    def __init__(self, config: PipelineConfig) -> None:
        # Configuration.
        self.config = config
        # Steps as (name, callable) pairs.
        self._steps: list[tuple[str, Step]] = []
        # Most recent run.
        self._current_run: PipelineRun | None = None
        # Every run of this pipeline object.
        self.runs: list[PipelineRun] = []

    # Pipeline from a configuration (kept for callers of earlier releases).
    @classmethod
    def from_config(cls, config: PipelineConfig) -> Pipeline:
        # Empty pipeline.
        return cls(config)

    # Names of the steps in order.
    @property
    def step_names(self) -> list[str]:
        # First element of each pair.
        return [name for name, _ in self._steps]

    # Number of steps.
    def __len__(self) -> int:
        # Length of the step list.
        return len(self._steps)

    # Append a step; returns the pipeline so that calls can be chained.
    def add_step(self, step: Step, name: str | None = None) -> Pipeline:
        # Steps must be callable.
        if not callable(step):
            # Explain the problem.
            raise TypeError(f"step must be callable, got {type(step).__name__}")
        # Name: explicit, else the function name, else the position.
        label = name or getattr(step, "__name__", "") or f"step_{len(self._steps)}"
        # Names identify steps in the run record.
        if label in self.step_names:
            # Explain the problem.
            raise ValueError(f"a step named {label!r} already exists")
        # Append the pair.
        self._steps.append((label, step))
        # Allow chaining.
        return self

    # Decorator form of add_step.
    def step(self, name: str | None = None) -> Callable[[Step], Step]:
        # The decorator registers the function and returns it unchanged.
        def register(function: Step) -> Step:
            # Append the step.
            self.add_step(function, name)
            # Return the function.
            return function

        # Return the decorator.
        return register

    # Snapshot of the configuration for the run record.
    def _snapshot_config(self) -> dict[str, Any]:
        # Configuration dictionary with the step names actually used.
        return {**self.config.to_dict(), "executed_steps": self.step_names}

    # Create a pending run without executing it.
    def create_run(self, inputs: Mapping[str, Any] | None = None) -> PipelineRun:
        # Values of the run.
        values = dict(inputs or {})
        # Pending run.
        run = PipelineRun(
            pipeline_id=self.config.pipeline_id,  # Pipeline.
            config_snapshot=self._snapshot_config(),  # Configuration.
            inputs={k: _as_text(v) for k, v in values.items()},  # Inputs as text.
            seed=self.config.effective_seed,  # Seed.
        )  # End of the run.
        # Remember the run.
        self._current_run = run
        # Keep the history.
        self.runs.append(run)
        # Return the run.
        return run

    # Execute every step on the inputs and return the run record.
    def run(
        self,  # The pipeline.
        inputs: Mapping[str, Any] | None = None,  # Values given to the first step.
        record_provenance: bool = True,  # Whether to build a provenance record.
    ) -> PipelineRun:  # The finished run.
        # Values passed between the steps.
        values: dict[str, Any] = dict(inputs or {})
        # Pending run.
        run = self.create_run(values)
        # Seed the generators before the first step.
        if run.seed is not None:
            # Python, NumPy and PyTorch.
            seed_everything(run.seed)
        # Running state.
        run.start()
        # Log the start.
        run.log(f"starting pipeline {self.config.name} ({len(self._steps)} steps)")
        # Execute and record every step.
        try:
            # Steps in order.
            for name, function in self._steps:
                # Step record.
                record = StepRecord(name=name, status=PipelineStatus.RUNNING)
                # Keep the record in the run.
                run.steps.append(record)
                # Start time.
                record.started_at = _now().isoformat()
                # Monotonic start.
                tick = time.perf_counter()
                # Log the step.
                run.log(f"executing step {name}")
                # Execute the step.
                try:
                    # Step output.
                    output = function(values)
                    # Steps must return a mapping of values.
                    if not isinstance(output, Mapping):
                        # Explain the problem.
                        raise TypeError(f"step {name} returned {type(output).__name__}, not a dict")
                # Record the failure of the step and re-raise.
                except Exception as error:
                    # Failed state.
                    record.status = PipelineStatus.FAILED
                    # Error message.
                    record.error = f"{type(error).__name__}: {error}"
                    # Re-raise for the outer handler.
                    raise
                # Record the end of the step, whatever the outcome.
                finally:
                    # Duration.
                    record.duration_seconds = time.perf_counter() - tick
                    # End time.
                    record.finished_at = _now().isoformat()
                    # Duration metric.
                    run.metrics[f"step.{name}.seconds"] = record.duration_seconds
                # Completed state.
                record.status = PipelineStatus.COMPLETED
                # Values for the next step.
                values = dict(output)
            # Output values.
            run.results = values
            # Output values as text.
            run.outputs = {k: _as_text(v) for k, v in values.items()}
            # Completed state.
            run.complete()
            # Log the end.
            run.log("pipeline completed")
        # Record the failure of the run and re-raise.
        except Exception as error:
            # Failed state.
            run.fail(f"{type(error).__name__}: {error}")
            # Log the failure.
            run.log(f"pipeline failed: {error}")
            # Provenance of the partial run.
            if record_provenance:
                # Inputs only; there are no outputs.
                run.provenance = self._provenance(run, dict(inputs or {}), {})
            # Re-raise for the caller.
            raise
        # Provenance of the completed run.
        if record_provenance:
            # Inputs and outputs with evidence.
            run.provenance = self._provenance(run, dict(inputs or {}), values)
        # Total duration metric.
        run.metrics["run.seconds"] = run.duration_seconds or 0.0
        # Return the run.
        return run

    # Build the provenance record of a run.
    def _provenance(
        self,  # The pipeline.
        run: PipelineRun,  # The run.
        inputs: Mapping[str, Any],  # Input values.
        outputs: Mapping[str, Any],  # Output values.
    ) -> ProvenanceRecord:  # The record.
        # Model identifiers named in the parameters.
        model = self.config.parameters.get("model_id") or self.config.parameters.get("model")
        # Build the record.
        return ProvenanceRecord(
            run_id=run.run_id,  # Run.
            pipeline_id=run.pipeline_id,  # Pipeline.
            inputs=_artefacts(inputs, EvidenceType.INPUT),  # Input evidence.
            outputs=_artefacts(outputs, EvidenceType.OUTPUT),  # Output evidence.
            model_ids=[str(model)] if isinstance(model, str) else [],  # Models.
            config=run.config_snapshot,  # Configuration.
            environment=capture_environment(),  # Software environment.
        )  # End of the record.

    # Short description.
    def __repr__(self) -> str:
        # Identifier and step names.
        return f"Pipeline(id={self.config.pipeline_id!r}, steps={self.step_names})"


# =============================================================================
# End of module src/unbihexium/core/pipeline.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
