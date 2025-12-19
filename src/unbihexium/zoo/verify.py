"""Model verification utilities.

Provides SHA256 verification for model artifacts to ensure integrity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class VerificationError(Exception):
    """Raised when model verification fails."""


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        filepath: Path to file.

    Returns:
        Hexadecimal SHA256 hash string.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def read_sha256_file(sha_path: Path) -> dict[str, str]:
    """Read SHA256 hashes from a .sha256 file.

    Format: <hash>  <filename>

    Args:
        sha_path: Path to .sha256 file.

    Returns:
        Dictionary mapping filename to expected hash.
    """
    hashes = {}
    with open(sha_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                hash_value = parts[0]
                filename = parts[-1]
                hashes[filename] = hash_value
    return hashes


def verify_file(
    filepath: Path,
    expected_sha256: str,
) -> bool:
    """Verify a file against expected SHA256 hash.

    Args:
        filepath: Path to file to verify.
        expected_sha256: Expected SHA256 hash.

    Returns:
        True if verification passes.

    Raises:
        VerificationError: If verification fails.
    """
    if not filepath.exists():
        raise VerificationError(f"File not found: {filepath}")

    actual = compute_sha256(filepath)
    if actual != expected_sha256.lower():
        raise VerificationError(
            f"SHA256 mismatch for {filepath.name}: "
            f"expected {expected_sha256[:16]}..., got {actual[:16]}..."
        )

    return True


def verify_model(
    model_dir: Path,
    files_to_verify: list[str] | None = None,
) -> dict[str, bool]:
    """Verify all artifacts in a model directory.

    Args:
        model_dir: Path to model directory.
        files_to_verify: Specific files to verify. If None, verifies all .onnx files.

    Returns:
        Dictionary mapping filename to verification result.

    Raises:
        VerificationError: If any verification fails.
    """
    results = {}

    sha_path = model_dir / "model.sha256"
    if not sha_path.exists():
        raise VerificationError(f"SHA256 file not found: {sha_path}")

    expected_hashes = read_sha256_file(sha_path)

    if files_to_verify is None:
        files_to_verify = list(expected_hashes.keys())

    for filename in files_to_verify:
        filepath = model_dir / filename

        if filename not in expected_hashes:
            raise VerificationError(f"No expected hash for {filename}")

        verify_file(filepath, expected_hashes[filename])
        results[filename] = True

    return results
