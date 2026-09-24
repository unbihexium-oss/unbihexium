<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/model_zoo/download_and_verify.md
Title       : Building and Verifying Models
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Building and Verifying Models

| Field | Value |
| --- | --- |
| Document | UBX-DOC-702 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | The main branch of Unbihexium, model zoo catalogue version 2.0.0 (not part of release 1.0.1) |

## Abstract

This document is the practical guide to obtaining models of the Unbihexium model zoo on a local machine and checking their integrity. It is written for users who build models before training or inference, for operators who prepare model stores for other machines, and for auditors who want to confirm that a model is the published starter model. It covers the `unbihexium zoo` commands `list`, `info`, `build`, `verify`, `where`, `export` and `clear`, the equivalent Python functions, what each verification step checks and what it does not, the relation between the store and the checksums under `model_zoo/`, the repository-level checks, and the errors a user may meet. Although the file name mentions downloading, catalogue models are never downloaded: they are built locally from the catalogue (see [distribution.md](distribution.md)). The learned models obtained this way are untrained starter models; only the 28 spectral index models compute exact formulas without training.

## Contents

1. [Introduction](#1-introduction)
2. [Prerequisites and the model store](#2-prerequisites-and-the-model-store)
3. [Command line](#3-command-line)
4. [Python API](#4-python-api)
5. [What is verified](#5-what-is-verified)
6. [Repository checks](#6-repository-checks)
7. [Troubleshooting](#7-troubleshooting)
8. [References](#references)

## 1. Introduction

### 1.1 Status of the models

A model built from the catalogue carries deterministic starter weights. Every learned model (123 families, 492 models) is untrained, and its predictions are meaningless until it is trained on labelled data ([training.md](training.md)). The 7 spectral index families (28 models) compute published formulas exactly and are usable as built. Verification proves that a model is the published starter model; it does not say anything about its accuracy.

### 1.2 Conventions

The key words MUST, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in capitals.

The command examples were run on the main branch on 2026-09-24 in a directory where `UNBIHEXIUM_CACHE=ubx-cache`, so that the store paths in the output are relative. Tables printed by `zoo list` are shortened.

## 2. Prerequisites and the model store

| Operation | Requirement |
| --- | --- |
| `zoo list`, `zoo info`, `zoo where`, `zoo clear` | Core installation; the catalogue ships with the package |
| `zoo build`, `zoo verify`, `load_model`, `ensure_model` | Extra `torch`: `pip install "unbihexium[torch]"` |
| `zoo build --onnx`, `zoo export` | Extra `torch` (which installs `onnx`) and extra `onnx` for the ONNX Runtime comparison |

The model zoo is only on the main branch; install from source as described in [README.md](../../README.md) until a release contains it.

Models are stored under `$UNBIHEXIUM_CACHE/models/<model id>/`, or `~/.cache/unbihexium/models/<model id>/` when the variable is not set. The option `--cache-dir DIR` of `zoo build`, `zoo verify`, `zoo where` and `zoo clear` and the `cache_dir` argument of the Python functions select another root for one call. `zoo verify`, `zoo where` and `zoo clear` have no `--cache-dir` option and always use the default root, so a store in another location MUST be selected with `UNBIHEXIUM_CACHE` for those commands.

## 3. Command line

### 3.1 Finding a model

`zoo list` prints the models of the catalogue with their task, domain and parameter count; `--task`, `--domain` and `--variant` filter the list and `--json` prints every catalogue field.

```bash
unbihexium zoo list --task detection --variant tiny
```

```text
                          Model zoo (19 models)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Model ID                       ┃ Task      ┃ Domain      ┃ Parameters ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ aircraft_detector_tiny         │ detection │ ai          │    730,581 │
│ border_monitor_tiny            │ detection │ defense     │    730,647 │
│ building_detector_tiny         │ detection │ urban       │    730,581 │
│ builtup_detector_tiny          │ detection │ urban       │    730,725 │
```

`zoo info MODEL_ID` prints one entry as JSON, including the input channels in order, the outputs, the published weights digest and `"requires_training": true` for every learned model:

```bash
unbihexium zoo info ship_detector_tiny
```

```text
{
  "model_id": "ship_detector_tiny",
  "variant": "tiny",
  "tile_size": 256,
  "weights_digest": "e278457bcbb75b891f2b0730889fab1a33a7f346fa695e33cc5545b5952e2899",
  "num_parameters": 730581,
  "source": "build",
  "download_url": null,
  "version": "2.0.0",
  "requires_training": true,
  "family": "ship_detector",
  ...
}
```

### 3.2 Building a model

`zoo build MODEL_ID` builds the model with its starter weights, compares the weights digest with the published one, and writes the checkpoint, the configuration and the checksum file to the store. `--onnx` also exports the model to ONNX and compares ONNX Runtime with PyTorch. `--force` rebuilds a model that is already in the store.

```bash
unbihexium zoo build ship_detector_tiny --onnx
unbihexium zoo build ndvi_calculator_base
```

```text
Cached: ubx-cache/models/ship_detector_tiny
Cached: ubx-cache/models/ndvi_calculator_base
```

The directory then contains:

| File | Content |
| --- | --- |
| `model.pt` | Checkpoint: format marker, configuration, state dictionary, weights digest and (for trained models) training metadata; read with `torch.load(weights_only=True)` |
| `model.onnx` | ONNX export (opset 18), only with `--onnx`; carries the configuration and the weights digest as metadata |
| `config.json` | The catalogue fields of the model, its variant and tile size, the published weights digest and the SHA-256 of `model.pt` (`checkpoint_sha256`) |
| `model.sha256` | SHA-256 of the files above in the format of `sha256sum` |

A family name without a variant, such as `ship_detector`, is accepted and denotes the `base` variant. `unbihexium zoo download` is a hidden alias of `zoo build` without the `--onnx` option, kept for compatibility with earlier scripts.

### 3.3 Verifying a cached model

```bash
unbihexium zoo verify ship_detector_tiny
```

```text
Verified: ship_detector_tiny
```

The command checks every file listed in `model.sha256`, loads the checkpoint, checks the digest recorded inside it, and compares the weights with the published digest (Section 5.3). It exits with status 1 and the message `ship_detector_tiny is not cached or does not verify` when the model is missing, damaged or different, including a modified `model.onnx` or `config.json`. The same file checksums can be checked without Python:

```bash
cd ubx-cache/models/ship_detector_tiny && sha256sum -c model.sha256
```

```text
config.json: OK
model.onnx: OK
model.pt: OK
```

### 3.4 Locating a cached model

```bash
unbihexium zoo where ship_detector_tiny
```

```text
ubx-cache/models/ship_detector_tiny/model.pt
```

For a model that is not cached the command exits with status 1 and suggests `unbihexium zoo build`.

### 3.5 Exporting to ONNX

`zoo export MODEL OUTPUT` exports a catalogue model or a checkpoint file to ONNX and, unless `--no-verify` is given, runs the export in ONNX Runtime and compares the result with PyTorch:

```bash
unbihexium zoo export ship_detector_tiny ship_detector_tiny.onnx
unbihexium zoo export runs/water_surface_detector_tiny/best.pt water.onnx
```

```text
Exported: ship_detector_tiny.onnx
```

The export uses ONNX opset 18 [4] with a dynamic batch axis and dynamic height and width, and stores the model configuration (including the normalisation statistics of trained models) and the weights digest in the ONNX metadata, so that [inference](inference.md) with ONNX Runtime needs no PyTorch.

### 3.6 Removing models

```bash
unbihexium zoo clear ship_detector_tiny
unbihexium zoo clear --yes
```

```text
Removed 1 model(s)
Removed 1 model(s)
```

Without a model identifier the command removes every cached model and asks for confirmation unless `--yes` is given. Removing a model is always safe: it can be rebuilt.

## 4. Python API

The functions below are exported by `unbihexium.zoo`. All of them accept a `cache_dir` argument except `load_model`, which builds in memory.

| Function | Purpose |
| --- | --- |
| `load_model(name, variant=None, verify=True)` | Build a catalogue model in memory and compare its digest with the published one, or load a `.pt` checkpoint |
| `ensure_model(model_id, cache_dir=None, onnx=False, force=False)` | Build the model into the store and return its directory |
| `download_model(model_id, cache_dir=None, force=False)` | Same as `ensure_model` without ONNX; returns the checkpoint path (the name is historical) |
| `verify_model(model_id, cache_dir=None)` | `True` if every file matches `model.sha256`, the cached checkpoint loads, matches its recorded digest and matches the published digest; `False` otherwise, also for corrupt files |
| `is_model_cached`, `get_cached_model_path`, `model_dir`, `list_cached`, `get_cache_dir` | Inspect the store |
| `clear_cache(model_id=None, cache_dir=None)` | Remove one or all cached models; returns the number removed |
| `verify_directory(directory)`, `verify_file(path, expected)`, `compute_sha256(path)` | File checksums in `sha256sum` format |
| `get_model(model_id)`, `list_models(task=None, domain=None, variant=None)` | Catalogue entries with the published digests |

```python
from unbihexium.zoo import (
    clear_cache,
    ensure_model,
    get_model,
    list_cached,
    load_model,
    verify_directory,
    verify_model,
)

directory = ensure_model("ship_detector_tiny", onnx=True)
print(sorted(p.name for p in directory.iterdir()))
print(verify_model("ship_detector_tiny"), verify_directory(directory))

model = load_model("ship_detector_tiny")
print(model.summary())
print(model.digest() == get_model("ship_detector_tiny").weights_digest)
print(list_cached(), clear_cache())
```

```text
['config.json', 'model.onnx', 'model.pt', 'model.sha256']
True {'config.json': True, 'model.onnx': True, 'model.pt': True}
ship_detector_tiny: detection, 3 input channels, 1 outputs, 730,581 parameters
True
['ship_detector_tiny'] 1
```

## 5. What is verified

### 5.1 Overview

| Check | Performed by | Detects |
| --- | --- | --- |
| Weights digest against `digests.json` | `load_model`, `ensure_model`, `zoo build`, `verify_model`, `zoo verify` | A model that is not the published starter model (changed code, catalogue or numerical environment, altered weights) |
| Digest recorded in the checkpoint | Every checkpoint load (`load_checkpoint`), `verify_model` | A checkpoint whose weights no longer match the digest written with them |
| File checksums in `model.sha256` | `verify_model`, `zoo verify`, `ensure_model` (which rebuilds a store entry that does not match), `verify_directory`, `sha256sum -c` | A changed file in the store, including `model.onnx` and `config.json` |
| ONNX Runtime against PyTorch | `export_onnx` (`zoo build --onnx`, `zoo export`) | An export whose output shape differs from PyTorch, or whose largest absolute difference exceeds 0.001 times the larger of 1 and the largest absolute PyTorch output |

### 5.2 The weights digest

The weights digest is a SHA-256 hash [3] over the sorted state dictionary (key, shape and little-endian float32 values of every tensor) and is defined in [distribution.md](distribution.md#34-weights-digest). It does not depend on the file format, so the same digest identifies a model in memory, in a checkpoint and after a rebuild on another machine.

### 5.3 How `verify_model` decides

`verify_model` returns `True` only if all of the following hold:

1. `model.sha256` exists, lists `model.pt` and `config.json` (and `model.onnx` when it was exported), and every listed file exists and matches its SHA-256 digest;
2. `model.pt` is an Unbihexium checkpoint that `torch.load(weights_only=True)` can read, and the digest of its weights equals the digest recorded in it;
3. for a catalogue model that was not customised, the digest equals the published digest in `digests.json`, or, for a model registered by the user with a `weights_digest`, that digest.

Any other outcome, including an unreadable or corrupt file, returns `False` instead of raising. A file changed in the store therefore fails in step 1, and a checkpoint that was re-saved together with a matching `model.sha256` and a new recorded digest fails in step 3. The following example alters one tensor and re-saves the checkpoint with a new digest; `verify_model` rejects it:

```python
import torch

from unbihexium.zoo import ensure_model, verify_model
from unbihexium.zoo.checkpoint import load_checkpoint, save_checkpoint

path = ensure_model("ship_detector_tiny") / "model.pt"
model = load_checkpoint(path)
with torch.no_grad():
    next(model.parameters()).add_(1.0)
save_checkpoint(model, path)
print(verify_model("ship_detector_tiny"))
```

```text
False
```

The digests show that a model is the published one; they are not a signature. They protect against accidental damage and against substitution only as far as the published `digests.json` itself is trusted, which is the case when it comes from a verified checkout of the repository or from a signed release distribution.

### 5.4 Checksums in `model_zoo/`

[model_zoo/checksums.txt](../../model_zoo/checksums.txt) lists the weights digest of every model, one line `<digest>  <model id>  # <parameters> parameters.` per model, sorted by model identifier. The digests are the same as in `src/unbihexium/zoo/digests.json`; they are weights digests, not file hashes, so the file cannot be used with `sha256sum -c`. To compare a model with it:

```bash
grep ' ship_detector_tiny ' model_zoo/checksums.txt
python -c "from unbihexium.zoo import load_model; print(load_model('ship_detector_tiny').digest())"
```

```text
e278457bcbb75b891f2b0730889fab1a33a7f346fa695e33cc5545b5952e2899  ship_detector_tiny  # 730,581 parameters.
e278457bcbb75b891f2b0730889fab1a33a7f346fa695e33cc5545b5952e2899
```

Each model card under [model_zoo/cards/](../../model_zoo/cards/) shows the first 16 hexadecimal digits of the digest of each variant, and each manifest under [model_zoo/manifests/](../../model_zoo/manifests/) the full digest.

## 6. Repository checks

In a checkout of the repository with the development dependencies installed, the following commands check the whole zoo. The run times were measured on 2026-09-24 in a container with 4 vCPUs, CPython 3.13, PyTorch 2.14 on the CPU:

| Command | Check | Time |
| --- | --- | --- |
| `python -m unbihexium.zoo.sync --root . --check` | Every generated file under `model_zoo/` and `digests.json` matches the catalogue | 17 s |
| `python .github/scripts/check_model_zoo.py` (also `make model-zoo`) | The same, plus schema validation of the manifests and one manifest and card per family | 18 s |
| `python .github/scripts/check_model_zoo.py --rebuild tiny` | The same, plus rebuilding the 130 tiny models and comparing their digests | 40 s |
| `python .github/scripts/check_model_zoo.py --rebuild all` | The same for all 520 models | not measured; considerably longer |
| `python scripts/validate_models.py --variant tiny --onnx` | Build, digest, forward pass with finite output of the expected shape, ONNX export compared with ONNX Runtime | not measured |

`scripts/validate_models.py` also accepts `--family NAME` to validate one family. The workflow `.github/workflows/model-zoo.yml` runs the first two checks and a rebuild in continuous integration (see [distribution.md](distribution.md#5-reproducibility-checks)).

## 7. Troubleshooting

| Symptom | Cause and remedy |
| --- | --- |
| `zoo build` fails with an error that ends in `install PyTorch with pip install 'unbihexium[torch]'` | Building needs PyTorch; install the extra `torch`. ONNX files can be run without it. |
| `Error: unknown model ...; see unbihexium zoo list` | The identifier is not in the catalogue; check the spelling and the variant suffix. |
| `VerificationError: <id>: weights digest ... != published ...` | The rebuilt weights differ from the published ones. Check that the installed package matches the checkout that produced `digests.json`, and report a reproducible mismatch on a supported platform as an issue. Do not use the model as the published starter model. |
| `CheckpointError: ... weights do not match the recorded digest` | The checkpoint was altered or damaged; delete it and rebuild with `zoo build --force`. |
| `CheckpointError: ... was written by a newer Unbihexium` | The checkpoint format is newer than the installed library; upgrade. |
| `zoo verify` reports "not cached" for a model built with `--cache-dir` | Pass the same `--cache-dir` to `zoo verify` (and to `zoo where` and `zoo clear`), or set `UNBIHEXIUM_CACHE`. |
| `ExportError: ... ONNX differs from PyTorch` | ONNX Runtime disagrees with PyTorch beyond the tolerance; report it with the model identifier and the versions of `torch`, `onnx` and `onnxruntime`. |

Security problems, such as a way to make a checkpoint load execute code, MUST be reported privately as described in [SECURITY.md](../../SECURITY.md), not in a public issue.

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] National Institute of Standards and Technology. Secure Hash Standard (SHS). FIPS PUB 180-4. 2015. <https://doi.org/10.6028/NIST.FIPS.180-4>

[4] ONNX project. ONNX operator schemas and versioning. 2026. <https://onnx.ai/onnx/intro/concepts.html>

<!--
=============================================================================
End of file docs/model_zoo/download_and_verify.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
