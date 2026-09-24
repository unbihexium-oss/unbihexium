# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_pipeline.py
# Title       : Tests of pipelines, run records, evidence and provenance
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest and NumPy
# =============================================================================
#
# Abstract
# --------
# Checks pipeline configuration, run records and their state transitions,
# step execution, failure handling, seeding for reproducibility, the SHA-256
# evidence of files and arrays (against the FIPS 180-4 test vector of
# "abc"), provenance records with tamper detection, and the registration of
# the task pipelines of the AI package.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Reference digests.
import hashlib

# JSON documents.
import json

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Test framework.
import pytest

# Evidence and provenance.
from unbihexium.core.evidence import Evidence, EvidenceType, ProvenanceRecord, sha256_array

# Pipelines.
from unbihexium.core.pipeline import (
    Pipeline,  # Ordered steps.
    PipelineConfig,  # Configuration.
    PipelineRun,  # Run record.
    PipelineStatus,  # Run states.
    seed_everything,  # Generator seeding.
)  # End of the pipeline imports.

# Registry of the named pipelines.
from unbihexium.registry.pipelines import PipelineRegistry

# SHA-256 of "abc" (FIPS 180-4, appendix B.1 of the 2002 edition).
SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# Configuration shared by the tests.
@pytest.fixture
def sample_config() -> PipelineConfig:
    # Three planned steps and one parameter.
    return PipelineConfig(
        pipeline_id="test_pipeline",  # Identifier.
        name="Test Pipeline",  # Name.
        description="A test pipeline for unit testing",  # Description.
        steps=["load", "process", "save"],  # Planned steps.
        parameters={"threshold": 0.5},  # Parameters.
    )  # End of the configuration.


# Step that adds one to "value".
def add_one(values: dict[str, Any]) -> dict[str, Any]:
    # New value.
    return {"value": values.get("value", 0) + 1}


# Step that doubles "value".
def double(values: dict[str, Any]) -> dict[str, Any]:
    # New value.
    return {"value": values["value"] * 2}


# Minimal and full configurations keep their fields.
def test_config_fields(sample_config: PipelineConfig) -> None:
    # Minimal configuration.
    minimal = PipelineConfig(pipeline_id="minimal", name="Minimal Pipeline")
    # Defaults.
    assert (minimal.version, minimal.seed, minimal.steps) == ("1.0.0", None, [])
    # Full configuration.
    assert sample_config.steps == ["load", "process", "save"]
    # Nested parameters stay as given.
    nested = PipelineConfig("nested", "Nested", parameters={"model": {"id": "m", "threshold": 0.5}})
    # Nested value.
    assert nested.parameters["model"]["id"] == "m"
    # Round trip through a dictionary.
    assert PipelineConfig.from_dict(sample_config.to_dict()) == sample_config


# Invalid configurations are rejected.
def test_config_validation() -> None:
    # Empty identifier.
    with pytest.raises(ValueError, match="pipeline_id"):
        # Invalid configuration.
        PipelineConfig(pipeline_id="", name="x")
    # Negative seed.
    with pytest.raises(ValueError, match="seed"):
        # Invalid configuration.
        PipelineConfig(pipeline_id="x", name="x", seed=-1)


# The seed field wins over a seed parameter.
def test_config_effective_seed() -> None:
    # Seed as a parameter.
    assert PipelineConfig("a", "a", parameters={"seed": 42}).effective_seed == 42
    # Field and parameter.
    assert PipelineConfig("b", "b", parameters={"seed": 42}, seed=7).effective_seed == 7
    # No seed.
    assert PipelineConfig("c", "c").effective_seed is None


# Run records accept plain strings for the status.
def test_run_record_fields() -> None:
    # Pending run with an input.
    run = PipelineRun(
        run_id="run_001",  # Identifier.
        pipeline_id="p",  # Pipeline.
        status="pending",  # State as text.
        inputs={"image": "a.tif"},  # Inputs.
    )  # End of the run.
    # Strings compare equal to the enum values.
    assert run.status == "pending" == PipelineStatus.PENDING
    # Input kept.
    assert run.inputs == {"image": "a.tif"}
    # Manual transitions.
    run.status = "completed"
    # New state.
    assert run.status == PipelineStatus.COMPLETED


# Runs move from pending to running to completed once.
def test_run_transitions() -> None:
    # Pending run.
    run = PipelineRun(pipeline_id="p")
    # Unique identifier.
    assert run.run_id != PipelineRun().run_id
    # No duration before the end.
    assert run.duration_seconds is None
    # Start.
    run.start()
    # Running.
    assert run.status is PipelineStatus.RUNNING
    # A running run cannot start again.
    with pytest.raises(RuntimeError, match="running"):
        # Second start.
        run.start()
    # Complete.
    run.complete()
    # Duration from the timestamps.
    assert run.status is PipelineStatus.COMPLETED and (run.duration_seconds or 0) >= 0
    # JSON record.
    record = json.loads(run.to_json())
    # Status as text.
    assert record["status"] == "completed" and record["pipeline_id"] == "p"


# Pending runs carry the configuration and the inputs as text.
def test_create_run(sample_config: PipelineConfig) -> None:
    # Pipeline without steps.
    pipeline = Pipeline(config=sample_config)
    # Pending run.
    run = pipeline.create_run(inputs={"image": Path("/data/image.tif")})
    # Pipeline, state and inputs.
    assert run.pipeline_id == "test_pipeline" and run.status == "pending"
    # Paths become text.
    assert run.inputs == {"image": "/data/image.tif"}
    # Configuration snapshot.
    assert run.config_snapshot["parameters"] == {"threshold": 0.5}
    # The run is kept in the history.
    assert pipeline.runs == [run]


# Steps are chained; outputs are recorded as text.
def test_run_chains_steps() -> None:
    # Pipeline of two steps.
    pipeline = Pipeline(PipelineConfig("chain", "Chain")).add_step(add_one).add_step(double)
    # (10 + 1) * 2.
    run = pipeline.run({"value": 10})
    # Completed with the result.
    assert run.status is PipelineStatus.COMPLETED and run.outputs == {"value": "22"}
    # Result values.
    assert run.results == {"value": 22}
    # Step records in order.
    assert [(s.name, s.status) for s in run.steps] == [
        ("add_one", PipelineStatus.COMPLETED),  # First step.
        ("double", PipelineStatus.COMPLETED),  # Second step.
    ]  # End of the steps.
    # Durations as metrics.
    assert {"step.add_one.seconds", "step.double.seconds", "run.seconds"} <= set(run.metrics)
    # Step names.
    assert pipeline.step_names == ["add_one", "double"] and len(pipeline) == 2


# Steps are registered by decorator; names must be unique and steps callable.
def test_step_registration() -> None:
    # Empty pipeline.
    pipeline = Pipeline(PipelineConfig("deco", "Decorated"))

    # Registered under an explicit name.
    @pipeline.step("increment")
    def step(values: dict[str, Any]) -> dict[str, Any]:
        # Add one.
        return add_one(values)

    # Registered name.
    assert pipeline.step_names == ["increment"]
    # Duplicate names are rejected.
    with pytest.raises(ValueError, match="already exists"):
        # Same name.
        pipeline.add_step(add_one, name="increment")
    # Non-callable steps are rejected.
    with pytest.raises(TypeError, match="callable"):
        # A number.
        pipeline.add_step(3)  # type: ignore[arg-type]


# Failing steps fail the run and re-raise the error.
def test_run_failure() -> None:
    # Step that fails.
    def broken(values: dict[str, Any]) -> dict[str, Any]:
        # Always fails.
        raise ZeroDivisionError("division by zero")

    # Pipeline with a good and a failing step.
    pipeline = Pipeline(PipelineConfig("fail", "Fail")).add_step(add_one).add_step(broken)
    # The error reaches the caller.
    with pytest.raises(ZeroDivisionError):
        # Run.
        pipeline.run({"value": 1})
    # The run record is kept.
    run = pipeline.runs[-1]
    # Failed with the error.
    assert run.status is PipelineStatus.FAILED and "ZeroDivisionError" in (run.error or "")
    # The failing step is marked.
    assert [s.status for s in run.steps] == [PipelineStatus.COMPLETED, PipelineStatus.FAILED]
    # Steps that return no mapping fail too.
    bad = Pipeline(PipelineConfig("bad", "Bad")).add_step(lambda values: 42, name="number")  # type: ignore[arg-type,return-value]
    # Type error.
    with pytest.raises(TypeError, match="not a dict"):
        # Run.
        bad.run()


# A seed makes random steps reproducible.
def test_seed_reproducibility() -> None:
    # Step that draws random numbers from the global generator.
    def draw(values: dict[str, Any]) -> dict[str, Any]:
        # Five uniform numbers.
        return {"sample": np.random.rand(5)}

    # Seeded pipeline.
    seeded = Pipeline(PipelineConfig("seed", "Seed", parameters={"seed": 42})).add_step(draw)
    # Two runs.
    first, second = seeded.run().results["sample"], seeded.run().results["sample"]
    # Identical samples.
    assert np.array_equal(first, second)
    # Same as seeding NumPy by hand.
    np.random.seed(42)
    # Reference sample.
    assert np.array_equal(first, np.random.rand(5))
    # A different seed gives a different sample.
    other = Pipeline(PipelineConfig("seed2", "Seed", seed=43)).add_step(draw).run()
    # Different values.
    assert not np.array_equal(first, other.results["sample"])
    # Seeds outside the generator range are rejected.
    with pytest.raises(ValueError, match="seed"):
        # Too large.
        seed_everything(2**32)


# Runs record SHA-256 evidence of input and output files.
def test_run_provenance(tmp_path: Path) -> None:
    # Input file with the content "abc".
    source = tmp_path / "input.txt"
    # Write it.
    source.write_bytes(b"abc")

    # Step that writes an upper-case copy.
    def upper(values: dict[str, Any]) -> dict[str, Any]:
        # Output file.
        target = tmp_path / "output.txt"
        # Write the copy.
        target.write_bytes(Path(values["input"]).read_bytes().upper())
        # Output path.
        return {"output": str(target)}

    # Pipeline with the model named in its parameters.
    pipeline = Pipeline(PipelineConfig("copy", "Copy", parameters={"model_id": "m1"}))
    # Add the step and run it.
    run = pipeline.add_step(upper).run({"input": source})
    # Provenance of the run.
    record = run.provenance
    # Input evidence with the FIPS 180-4 digest of "abc".
    assert record is not None and isinstance(record.inputs[0], Evidence)
    # Digest of the input.
    assert record.inputs[0].checksum == SHA256_ABC
    # Output evidence with the digest of "ABC".
    assert record.outputs[0].checksum == hashlib.sha256(b"ABC").hexdigest()  # type: ignore[union-attr]
    # Model identifier from the parameters.
    assert record.model_ids == ["m1"] and record.run_id == run.run_id
    # Software environment.
    assert "numpy" in record.environment and "python" in record.environment
    # The outputs verify.
    assert record.verify_outputs() == []


# Input files are hashed when the run starts, before a step can change them.
def test_run_provenance_hashes_inputs_first(tmp_path: Path) -> None:
    # Input file with the content "abc".
    source = tmp_path / "input.txt"
    # Write it.
    source.write_bytes(b"abc")

    # Step that overwrites its input.
    def overwrite(values: dict[str, Any]) -> dict[str, Any]:
        # Replace the content.
        Path(values["input"]).write_bytes(b"changed")
        # No outputs.
        return {}

    # Pipeline with the step.
    pipeline = Pipeline(PipelineConfig("overwrite", "Overwrite"))
    # Run it.
    record = pipeline.add_step(overwrite).run({"input": source}).provenance
    # The record describes the file that was read, not the changed file.
    assert record is not None and record.inputs[0].checksum == SHA256_ABC  # type: ignore[union-attr]
    # Each input appears once.
    assert len(record.inputs) == 1


# Evidence records identify artefacts by digest.
def test_evidence_fields() -> None:
    # Record without a digest.
    evidence = Evidence(source="test_source", timestamp="2025-01-01T00:00:00Z", description="Test")
    # Fields.
    assert evidence.source == "test_source" and evidence.description == "Test"
    # Identifier derived from the type and the source.
    assert evidence.evidence_id.startswith("input-")
    # Invalid digests are rejected.
    with pytest.raises(ValueError, match="SHA-256"):
        # Too short.
        Evidence(source="x", checksum="abc")
    # Round trip through a dictionary.
    assert Evidence.from_dict(evidence.to_dict()) == evidence


# File digests match the FIPS 180-4 test vector and are deterministic.
def test_evidence_checksum(tmp_path: Path) -> None:
    # File with the content "abc".
    path = tmp_path / "abc.txt"
    # Write it.
    path.write_bytes(b"abc")
    # Known digest.
    assert Evidence.compute_checksum(path) == SHA256_ABC
    # Evidence of the file.
    evidence = Evidence.from_file(path, EvidenceType.OUTPUT)
    # Digest, size and type.
    assert evidence.checksum == SHA256_ABC and evidence.size_bytes == 3
    # Role of the file.
    assert evidence.evidence_type is EvidenceType.OUTPUT
    # The unchanged file verifies.
    assert evidence.verify()
    # A changed file does not.
    path.write_bytes(b"abd")
    # Verification fails.
    assert not evidence.verify()
    # Missing files are reported.
    with pytest.raises(FileNotFoundError):
        # No such file.
        Evidence.from_file(tmp_path / "missing.txt")


# Array digests cover the dtype and the shape.
def test_evidence_array() -> None:
    # Array of zeros.
    zeros = np.zeros(4, dtype=np.float32)
    # Same bytes, different shape.
    assert sha256_array(zeros) != sha256_array(zeros.reshape(2, 2))
    # Same values, different dtype.
    assert sha256_array(zeros) != sha256_array(zeros.astype(np.float64))
    # Same array, same digest.
    assert Evidence.from_array(zeros, "zeros").checksum == sha256_array(zeros.copy())


# Provenance records serialise, verify their digest and detect tampering.
def test_provenance_record(tmp_path: Path) -> None:
    # Record with plain paths.
    record = ProvenanceRecord(
        run_id="prov_001",  # Run.
        pipeline_id="test_pipeline",  # Pipeline.
        inputs=["/input/file.tif"],  # Inputs.
        outputs=["/output/result.tif"],  # Outputs.
        model_ids=["model_001"],  # Models.
        config={"threshold": 0.5},  # Configuration.
    )  # End of the record.
    # JSON text contains the identifiers.
    text = record.to_json(tmp_path / "prov.json")
    # Identifiers present.
    assert "prov_001" in text and "model_001" in text
    # Round trip from the file.
    back = ProvenanceRecord.from_json(tmp_path / "prov.json")
    # Same content and digest.
    assert back == record and back.digest() == record.digest()
    # Round trip from the text.
    assert ProvenanceRecord.from_json(text) == record
    # A modified document fails the digest check.
    document = json.loads(text)
    # Change the configuration.
    document["config"]["threshold"] = 0.9
    # Digest mismatch.
    with pytest.raises(ValueError, match="digest"):
        # Load the modified document.
        ProvenanceRecord.from_dict(document)
    # Records without inputs serialise too.
    empty = ProvenanceRecord(run_id="empty_test", pipeline_id="test")
    # Empty lists.
    assert json.loads(empty.to_json())["inputs"] == []


# Models are added by identifier or by weight evidence.
def test_provenance_models(tmp_path: Path) -> None:
    # Weight file.
    weights = tmp_path / "weights.bin"
    # Write it.
    weights.write_bytes(b"abc")
    # Empty record.
    record = ProvenanceRecord(run_id="r", pipeline_id="p")
    # By identifier.
    record.add_model("ship_detector_tiny")
    # By evidence.
    record.add_model(Evidence.from_file(weights, EvidenceType.MODEL))
    # Identifiers of both.
    assert record.model_ids == ["ship_detector_tiny", "model-" + SHA256_ABC[:16]]
    # Output with a digest that no longer matches.
    output = tmp_path / "out.txt"
    # Write and record.
    output.write_bytes(b"x")
    # Evidence of the output.
    record.add_output(Evidence.from_file(output, EvidenceType.OUTPUT))
    # Modify the output.
    output.write_bytes(b"y")
    # Reported by verify_outputs.
    assert record.verify_outputs() == [str(output)]


# The AI package registers its task pipelines.
def test_registry() -> None:
    # Importing the package registers the pipelines.
    import unbihexium.ai

    # Registered identifiers.
    ids = [entry.pipeline_id for entry in PipelineRegistry.list_all()]
    # Ship detection is one of them.
    assert "ship_detection" in ids
    # Unknown identifiers give None.
    assert PipelineRegistry.get("no_such_pipeline") is None
    # The factory builds a pipeline with one step.
    pipeline = PipelineRegistry.create("ship_detection")
    # One step named after the task.
    assert isinstance(pipeline, Pipeline) and pipeline.step_names == ["run_task"]


# =============================================================================
# End of module tests/unit/test_pipeline.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
