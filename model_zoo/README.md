# Unbihexium Model Zoo

## Overview

The model zoo offers 130 model families for Earth observation, each in four size variants (tiny, base, large and mega): 520 models in total. The families cover object detection, semantic segmentation, change detection, per-pixel and scene-level regression, image enhancement, super-resolution and spectral indices.

**Status of the models.** Every learned model is a *starter model*: a complete, trainable network for its task, with the input and output layout documented in its model card and deterministic starter weights. The starter models have **not** been trained on Earth observation data, so their predictions are meaningless until you train or fine-tune them on labelled data for your area, sensor and season. The seven spectral index models (NDVI, NDWI, EVI, SAVI, MSI, NBR and VCI) implement the published formulas exactly and need no training.

## Obtaining a model

No large downloads and no Git LFS are needed. The starter weights of every model are generated locally from its model id with a platform-independent random stream, and verified against the published SHA-256 digest:

```python
from unbihexium.zoo import load_model

model = load_model("ship_detector_base")   # build in memory, verify the digest
print(model.summary())
```

```python
from unbihexium.zoo import ensure_model

directory = ensure_model("ship_detector_base", onnx=True)
# $UNBIHEXIUM_CACHE/models/ship_detector_base/
#   model.pt       checkpoint, loadable with torch.load(weights_only=True)
#   model.onnx     ONNX export, verified against PyTorch
#   config.json    inputs, outputs and metadata
#   model.sha256   file checksums
```

Building models requires `pip install "unbihexium[torch]"`. ONNX files run with `pip install "unbihexium[onnx]"` only.

## How reproducibility works

| Step | Method |
| --- | --- |
| Seed | First four bytes of SHA-256(`"unbihexium:" + model id`) |
| Random stream | `numpy.random.RandomState`, frozen across NumPy releases (NEP 19) |
| Scheme | He normal for convolutions and linear layers, identity for group normalisation, task-specific output initialisation |
| Digest | SHA-256 over the sorted state dict: key, shape and little-endian float32 bytes |

The published digests are in [`src/unbihexium/zoo/digests.json`](../src/unbihexium/zoo/digests.json) and [`checksums.txt`](checksums.txt). The Model Zoo workflow rebuilds the tiny variants on every pull request and all 520 models every week to prove that the weights are reproducible.

## Directory structure

```text
model_zoo/
  README.md                   this file
  MODEL_CARDS.md              index of the model cards
  cards/<family>.md           model card per family
  manifests/<family>.json     machine-readable manifest per family
  manifest.schema.json        JSON Schema of the manifests
  inventory.yaml              summary of all families
  capability_to_models.yaml   family to model id mapping
  checksums.txt               weights digest of every model
```

Every file in this directory except this README and the schema is generated from [`src/unbihexium/zoo/catalog.yaml`](../src/unbihexium/zoo/catalog.yaml):

```bash
python -m unbihexium.zoo.sync --root .          # regenerate after editing the catalogue
python -m unbihexium.zoo.sync --root . --check  # verify, as CI does
```

## Validation

```bash
python .github/scripts/check_model_zoo.py              # consistency and schema checks
python .github/scripts/check_model_zoo.py --rebuild all  # rebuild and compare digests
python scripts/validate_models.py --variant tiny --onnx  # forward pass and ONNX export
```

## Licence

All models are licensed under the [Mozilla Public License 2.0](../LICENSE.txt), like the rest of the repository. They contain no third-party weights and no third-party training data.

## Responsible use

Validate every trained model on independent reference data before use and report its accuracy with its results. See [RESPONSIBLE_USE.md](../RESPONSIBLE_USE.md), in particular for detection models used for security, defence or border monitoring.
