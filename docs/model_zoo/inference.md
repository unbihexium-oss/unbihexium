# Running Model Zoo Models

This guide explains how to run a model on imagery of any size, with PyTorch or ONNX Runtime, and what the outputs of every task look like.

Models can be given in four forms:

| Form | Example | Backend |
| --- | --- | --- |
| Catalogue family or model id | `ship_detector`, `ship_detector_large` | PyTorch, starter weights (untrained) |
| Trained checkpoint | `runs/ship_detector_base/best.pt` | PyTorch |
| ONNX export | `ship_detector.onnx` | ONNX Runtime, no PyTorch needed |
| Model object | `build_model("ship_detector", "tiny")` | PyTorch |

Starter weights are untrained, so catalogue models only produce meaningful results after [training](training.md). The seven spectral index models (NDVI, NDWI, EVI, SAVI, MSI, NBR and VCI) are exact formulas and can be used directly.

## Command line

```bash
unbihexium predict MODEL INPUT OUTPUT [options]
```

| Task | Output |
| --- | --- |
| Detection | GeoJSON FeatureCollection of boxes with `class_name` and `confidence` |
| Segmentation, change detection | single-band GeoTIFF of class indices, 255 for no data |
| Dense regression, spectral index | float32 GeoTIFF with one band per output, NaN for no data |
| Enhancement | GeoTIFF of the output bands on the input grid |
| Super-resolution | GeoTIFF on a grid `scale` times finer, covering the same area |
| Scene regression | JSON document with one value per output |

Options: `--threshold` (detection score or segmentation probability), `--tile-size`, `--overlap` (fraction, default 0.25), `--backend auto|torch|onnx`, `--device`, `--variant` and, for change detection, `--second` with the image of the second date.

## Python

```python
from unbihexium.ai import ChangeDetector, LandCoverClassifier, TreeHeightEstimator, predict, write_result

land_cover = LandCoverClassifier(weights="runs/lulc/best.pt").predict("sentinel2.tif")
print(land_cover.class_fractions(), land_cover.class_areas())

change = ChangeDetector(weights="runs/change/best.pt").predict_pair("2024.tif", "2026.tif")
write_result(change, "change.tif")

height = TreeHeightEstimator(weights="runs/height/best.pt").predict("stack.tif")
print(height.summary())

result = predict("runs/any_model/best.pt", "scene.tif")  # picks the API from the task
```

Inputs may be file paths, `Raster` objects or NumPy arrays of shape `(bands, rows, cols)`. Results keep the CRS and transform of georeferenced inputs.

## Tiling

Images larger than the tile size (256 pixels for tiny and base, 512 for large and mega) are processed in overlapping tiles:

- Tiles at the image border are padded by reflection.
- Dense outputs are blended with weights that fall towards the tile edges, so there are no seams. Segmentation tiles are blended as probabilities.
- Detections whose centre lies in the overlap margin of an inner tile edge are dropped, and class-aware non-maximum suppression removes the remaining duplicates.
- Pixels where every input band is NaN or equal to the raster no-data value are NaN (regression) or 255 (classes) in the output.

The normalisation statistics recorded during training are applied automatically.

## ONNX Runtime

```bash
unbihexium zoo export runs/ship_detector_base/best.pt ship.onnx
```

The export uses opset 18 with dynamic batch and spatial axes, stores the model configuration and normalisation as metadata, and is compared with PyTorch before it is written. `Predictor("ship.onnx")` and `unbihexium predict ship.onnx ...` only need `pip install "unbihexium[onnx]"`.

## Low-level access

```python
from unbihexium.ai.inference import Predictor

predictor = Predictor("runs/ship_detector_base/best.pt", tile_size=512, overlap=0.25)
boxes, scores, classes = predictor.detect(image, threshold=0.3)
probabilities = Predictor("runs/lulc/best.pt").dense(image)
values = Predictor("runs/yield/best.pt").scene(image)
```
