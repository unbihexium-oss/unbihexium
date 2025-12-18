"""Pipeline abstraction for geospatial processing workflows."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineConfig:
    """Configuration for a pipeline."""

    pipeline_id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    steps: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    deterministic: bool = True


@dataclass
class PipelineRun:
    """Record of a pipeline execution."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_id: str = ""
    status: PipelineStatus = PipelineStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def start(self) -> None:
        self.status = PipelineStatus.RUNNING
        self.start_time = datetime.now()

    def complete(self) -> None:
        self.status = PipelineStatus.COMPLETED
        self.end_time = datetime.now()

    def fail(self, error: str) -> None:
        self.status = PipelineStatus.FAILED
        self.end_time = datetime.now()
        self.error = error

    def log(self, message: str) -> None:
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")

    @property
    def duration_seconds(self) -> float | None:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "config_snapshot": self.config_snapshot,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "error": self.error,
        }


class Pipeline:
    """Geospatial processing pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._steps: list[Callable[..., Any]] = []
        self._current_run: PipelineRun | None = None

    def add_step(self, step: Callable[..., Any]) -> Pipeline:
        """Add a processing step to the pipeline."""
        self._steps.append(step)
        return self

    def run(self, inputs: dict[str, Any]) -> PipelineRun:
        """Execute the pipeline."""
        run = PipelineRun(
            pipeline_id=self.config.pipeline_id,
            config_snapshot=self._snapshot_config(),
            inputs={k: str(v) for k, v in inputs.items()},
        )
        self._current_run = run
        run.start()
        run.log(f"Starting pipeline: {self.config.name}")

        try:
            result = inputs
            for i, step in enumerate(self._steps):
                step_name = step.__name__ if hasattr(step, "__name__") else f"step_{i}"
                run.log(f"Executing step: {step_name}")
                result = step(result)
            run.outputs = {k: str(v) for k, v in result.items()} if isinstance(result, dict) else {}
            run.complete()
            run.log("Pipeline completed successfully")
        except Exception as e:
            run.fail(str(e))
            run.log(f"Pipeline failed: {e}")
            raise

        return run

    def _snapshot_config(self) -> dict[str, Any]:
        """Create a snapshot of the current configuration."""
        return {
            "pipeline_id": self.config.pipeline_id,
            "name": self.config.name,
            "version": self.config.version,
            "parameters": self.config.parameters,
            "seed": self.config.seed,
        }

    @classmethod
    def from_config(cls, config: PipelineConfig) -> Pipeline:
        return cls(config)
