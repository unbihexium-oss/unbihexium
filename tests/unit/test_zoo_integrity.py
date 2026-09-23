# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for model integrity and verification."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from unbihexium.zoo.verify import (
    VerificationError,
    compute_sha256,
    read_sha256_file,
    verify_file,
)


class TestComputeSha256:
    """Tests for SHA256 computation."""

    def test_compute_sha256_deterministic(self) -> None:
        """Test that SHA256 is deterministic."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test content")
            path = Path(f.name)

        try:
            hash1 = compute_sha256(path)
            hash2 = compute_sha256(path)
            assert hash1 == hash2
        finally:
            path.unlink()

    def test_compute_sha256_different_content(self) -> None:
        """Test that different content produces different hash."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"content A")
            path1 = Path(f.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"content B")
            path2 = Path(f.name)

        try:
            hash1 = compute_sha256(path1)
            hash2 = compute_sha256(path2)
            assert hash1 != hash2
        finally:
            path1.unlink()
            path2.unlink()

    def test_compute_sha256_format(self) -> None:
        """Test that SHA256 is hexadecimal and correct length."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test")
            path = Path(f.name)

        try:
            hash_value = compute_sha256(path)
            assert len(hash_value) == 64
            assert hash_value.isalnum()
        finally:
            path.unlink()


class TestReadSha256File:
    """Tests for reading SHA256 files."""

    def test_read_sha256_file(self) -> None:
        """Test reading a .sha256 file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sha256", mode="w") as f:
            f.write("abc123def456  model.onnx\n")
            f.write("789xyz  config.json\n")
            path = Path(f.name)

        try:
            hashes = read_sha256_file(path)
            assert hashes["model.onnx"] == "abc123def456"
            assert hashes["config.json"] == "789xyz"
        finally:
            path.unlink()


class TestVerifyFile:
    """Tests for file verification."""

    def test_verify_file_success(self) -> None:
        """Test successful file verification."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test content")
            path = Path(f.name)

        try:
            expected = compute_sha256(path)
            assert verify_file(path, expected) is True
        finally:
            path.unlink()

    def test_verify_file_failure(self) -> None:
        """Test failed file verification."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test content")
            path = Path(f.name)

        try:
            wrong_hash = "a" * 64
            with pytest.raises(VerificationError):
                verify_file(path, wrong_hash)
        finally:
            path.unlink()

    def test_verify_file_not_found(self) -> None:
        """Test verification of non-existent file."""
        path = Path("/nonexistent/file.txt")
        with pytest.raises(VerificationError):
            verify_file(path, "a" * 64)


class TestIntegrityFailsClosed:
    """Tests to ensure integrity failures are closed (fail-safe)."""

    def test_tampered_file_fails(self) -> None:
        """Test that tampering with a file causes verification to fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            model_file = model_dir / "model.onnx"
            model_file.write_bytes(b"original content")

            original_hash = compute_sha256(model_file)

            sha_file = model_dir / "model.sha256"
            sha_file.write_text(f"{original_hash}  model.onnx\n")

            model_file.write_bytes(b"tampered content")

            hashes = read_sha256_file(sha_file)
            with pytest.raises(VerificationError):
                verify_file(model_file, hashes["model.onnx"])
