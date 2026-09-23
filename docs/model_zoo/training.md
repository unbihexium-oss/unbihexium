# Training Model Zoo Models

Every learned model in the zoo is a starter model: a complete network for its task with deterministic, untrained weights. This guide shows how to turn a starter model into a useful model with your own labelled data, how to check a training setup on synthetic data first, and how to evaluate and use the result.

Training needs PyTorch:

```bash
pip install "unbihexium[torch]"
```

## Quick check on synthetic data

Before you prepare real data, check that training runs on your machine. The `--synthetic` option generates learnable toy data for the task of the model:

```bash
unbihexium train ship_detector_tiny --synthetic 64 --epochs 5 --chip-size 64
```

The command prints the metrics of every epoch and writes `runs/ship_detector_tiny/best.pt`, `last.pt` and `history.json`. Synthetic data only proves that the pipeline works; a model trained on it is useless for real imagery.

## Dataset layout

```text
dataset/
  dataset.yaml             optional settings
  train/images/<id>.tif    image bands (.tif, .tiff, .npy or .npz)
  train/labels/<id>.tif    target with the same file stem
  val/images/...           validation split, same layout (recommended)
  val/labels/...
  test/images/...          optional test split
  test/labels/...
```

Images are read as `(bands, rows, cols)`. Large GeoTIFFs are fine: training reads random windows of `--chip-size` pixels, validation reads a regular grid of windows. The band order must match the model inputs listed in its model card, for example `red, green, blue` for `ship_detector` or the ten Sentinel-2 bands for `lulc_classifier`. Use `band_indices` in `dataset.yaml` to select and reorder the bands of your files.

### Targets by task

| Task | Label file | Content |
| --- | --- | --- |
| Detection | `labels/<id>.json` or `.geojson` | `{"boxes": [[x1, y1, x2, y2], ...], "labels": [...]}` in pixel coordinates, or a GeoJSON FeatureCollection in map coordinates with a `class` property |
| Segmentation | `labels/<id>.tif` (or `.npy`, `.png`) | single band of class indices; 255 is ignored |
| Change detection | `labels/<id>.tif` | single band: 0 for no change, 1 and up for the change classes |
| Dense regression | `labels/<id>.tif` | one band per output; NaN marks missing reference values |
| Scene regression | `labels/<id>.json` or `<split>/targets.csv` | `{"values": {"yield": 5.2}}`, or a CSV with the columns `id` and one column per output |
| Enhancement | `labels/<id>.tif` | one band per output on the image grid |
| Super-resolution | `labels/<id>.tif` | the high-resolution bands on a grid `scale` times finer |

Change detection images stack the bands of the first date followed by the bands of the second date, so a model with `red_t1, green_t1, blue_t1, red_t2, green_t2, blue_t2` inputs expects six bands.

GeoJSON coordinates are converted to pixels with the transform of the GeoTIFF. They are assumed to be in the CRS of the image unless the collection declares `"crs": "EPSG:<code>"`, in which case they are reprojected.

Detection labels may be class names or indices. Scene regression chips should have the size of `--chip-size`; larger images are cropped.

### dataset.yaml

```yaml
# Bands to read from the image files, zero-based, in model order.
band_indices: [3, 2, 1]
# Class names; replaces the classes of the catalogue model.
classes: [cargo, tanker, fishing]
# Input channel names; replaces the inputs of the catalogue model.
channel_names: [red, green, blue]
# Remap raw mask values to class indices; unmapped values are ignored.
label_map: {0: 0, 1: 1, 2: 1, 255: 255}
# Multiply image values, for example digital numbers to reflectance.
scale: 0.0001
# Image value that marks missing pixels.
nodata: 0
```

When `classes` or `channel_names` differ from the catalogue entry, the model is built with the new output or input layout. Its starter weights then come from the same seed but its digest differs from the published one.

## Training

```bash
unbihexium train ship_detector_base --data dataset --epochs 50 --batch-size 8
```

Useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--variant` | from the model id | `tiny`, `base`, `large` or `mega` |
| `--chip-size` | tile size of the variant | side length of training chips in pixels |
| `--samples-per-epoch` | one per image | random chips per epoch; raise it for few large images |
| `--lr` | 0.001 | peak learning rate of AdamW |
| `--weight-decay` | 0.0001 | decoupled weight decay |
| `--device` | `auto` | `cpu`, `cuda`, `cuda:1` or `mps` |
| `--amp` | off | mixed precision on CUDA |
| `--patience` | none | stop after this many epochs without improvement |
| `--regression-loss` | `l1` | `l1`, `mse` or `huber` for regression targets |
| `--no-augment` | off | disable rotations, mirrors and radiometric jitter |
| `--output` | `runs/<model id>` | directory of checkpoints and history |

To fine-tune a model you trained before, pass its checkpoint instead of the model id:

```bash
unbihexium train runs/ship_detector_base/best.pt --data more_data --epochs 20 --lr 0.0003
```

### What happens during training

1. The per-band mean and standard deviation are estimated from up to 32 training chips. They are stored in the checkpoint and applied again at inference time, so you never normalise inputs yourself.
2. Every epoch draws new random chips, rotates and mirrors them (not for displacement fields) and, for detection and segmentation, jitters brightness and contrast.
3. AdamW optimises the task loss with a linear warm-up over the first epoch and cosine decay to 1 percent of the peak rate. Gradients are clipped to norm 10.
4. After every epoch the validation split is evaluated. The checkpoint with the best monitored metric is written to `best.pt`; the latest one to `last.pt`.

| Task | Loss | Monitored metric |
| --- | --- | --- |
| Detection | focal heat map loss, L1 size and offset (CenterNet) | mAP at IoU 0.5 |
| Segmentation, change detection | cross-entropy plus Dice | mean IoU |
| Dense and scene regression | L1 (or MSE, Huber), NaN targets skipped | RMSE |
| Enhancement, super-resolution | L1 | PSNR |

Without a validation split the training loss is monitored instead. Always keep an independent validation split; the training loss says nothing about accuracy on new images.

## Training from Python

```python
from unbihexium.ai.training import TrainConfig, train

config = TrainConfig(epochs=50, batch_size=8, chip_size=256, patience=10)
result = train("lulc_classifier_base", "dataset", config)
print(result.best_epoch, result.best_metrics["miou"], result.best_checkpoint)
```

`ChipDataset`, `Trainer` and `TaskLoss` are available for custom loops, and `FolderDataset` or `SyntheticDataset` can be replaced by any sequence of `Sample` records.

## Evaluation

```bash
unbihexium evaluate runs/ship_detector_base/best.pt --data dataset --split test
```

The command prints the metrics of the task as JSON: mAP at IoU 0.5 and 0.5 to 0.95 with precision and recall for detection; overall accuracy, per-class IoU and F1, mean IoU and Cohen's kappa for segmentation; MAE, RMSE, bias and R squared per output for regression; PSNR and SSIM for image-to-image models.

## Using the trained model

```bash
unbihexium predict runs/ship_detector_base/best.pt scene.tif ships.geojson
unbihexium zoo export runs/ship_detector_base/best.pt ship_detector.onnx
unbihexium predict ship_detector.onnx scene.tif ships.geojson --backend onnx
```

```python
from unbihexium.ai import ShipDetector

detector = ShipDetector(weights="runs/ship_detector_base/best.pt", threshold=0.4)
result = detector.predict("scene.tif")
print(result.count, result.to_geojson())
```

See [inference.md](inference.md) for tiling, thresholds and output formats.

## Responsible use

Report the accuracy of a trained model on independent reference data together with any result derived from it, and state the area, sensor and period of the training data. Follow [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md), in particular for detection models used for security, defence or border monitoring.
