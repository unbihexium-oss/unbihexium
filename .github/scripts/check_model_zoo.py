# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Check the structure and integrity of the model zoo.

For every model variant under model_zoo/assets/<tier>/<name>/ the script checks:

- the required files exist (config.json, metrics.json, model.onnx, model.pt, model.sha256);
- config.json and metrics.json are valid JSON;
- the SHA256 recorded in model.sha256 matches each weight file. For Git LFS
  pointer files the pointer oid is compared; for downloaded files the content is hashed;
- a model card exists in model_zoo/cards/<name>.md and declares the MPL-2.0 licence;
- a manifest exists in model_zoo/manifests/ for the model family.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path("model_zoo")
REQUIRED = ("config.json", "metrics.json", "model.onnx", "model.pt", "model.sha256")
WEIGHTS = ("model.onnx", "model.pt")
TIERS = ("tiny", "base", "large", "mega")
POINTER = re.compile(rb"^version https://git-lfs.github.com/spec/v1\n.*?oid sha256:([0-9a-f]{64})", re.S)


def file_sha256(path: Path) -> tuple[str, bool]:
    """Return the SHA256 of a weight file and whether it is an LFS pointer."""
    head = path.read_bytes()[:512]
    match = POINTER.match(head)
    if match:
        return match.group(1).decode(), True
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest(), False


def error(path: Path, message: str) -> None:
    print(f"::error file={path}::{message}")


def main() -> int:
    failures = variants = pointers = hashed = 0
    families = {p.stem for p in (ROOT / "manifests").glob("*.json")}

    for tier in TIERS:
        for variant in sorted((ROOT / "assets" / tier).iterdir()):
            if not variant.is_dir():
                continue
            variants += 1
            missing = [name for name in REQUIRED if not (variant / name).is_file()]
            if missing:
                error(variant, f"Missing files: {', '.join(missing)}")
                failures += 1
                continue

            for name in ("config.json", "metrics.json"):
                try:
                    json.loads((variant / name).read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    error(variant / name, f"Invalid JSON: {exc}")
                    failures += 1

            recorded = {}
            for line in (variant / "model.sha256").read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2:
                    recorded[parts[1]] = parts[0]
            for weight in WEIGHTS:
                actual, is_pointer = file_sha256(variant / weight)
                pointers += is_pointer
                hashed += not is_pointer
                if recorded.get(weight) != actual:
                    error(variant / weight, f"SHA256 {actual} does not match model.sha256 ({recorded.get(weight)})")
                    failures += 1

            card = ROOT / "cards" / f"{variant.name}.md"
            if not card.is_file():
                error(card, "Model card missing")
                failures += 1
            elif "MPL-2.0" not in card.read_text(encoding="utf-8"):
                error(card, "Model card does not declare the MPL-2.0 licence")
                failures += 1

            family = variant.name.rsplit("_", 1)[0]
            if family not in families:
                error(variant, f"No manifest model_zoo/manifests/{family}.json")
                failures += 1

    print(f"Checked {variants} variants: {pointers} LFS pointers and {hashed} downloaded files verified.")
    if failures:
        print(f"{failures} model zoo problem(s) found.")
        return 1
    print("Model zoo check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
