# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/data.py
# Title       : Training datasets on disk and synthetic datasets
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy, SciPy, rasterio and
#               PyYAML
# =============================================================================
#
# Abstract
# --------
# Reads labelled training data for the model zoo tasks and produces Sample
# records (see transforms.py). The code does not depend on PyTorch; the
# training module wraps these datasets for its data loaders.
#
# Dataset layout
# --------------
#   dataset/
#     dataset.yaml             optional settings, see below
#     train/images/<id>.tif    image bands (.tif, .tiff, .npy or .npz)
#     train/labels/<id>.tif    target with the same file stem
#     val/images/ ...          validation split, same layout
#     test/images/ ...         optional test split
#
# Targets by task:
#
#   segmentation, change   single-band class raster (.tif, .npy, .png)
#   dense regression       K-band float raster, NaN marks missing values
#   enhancement            K-band float raster on the image grid
#   super-resolution       K-band raster on a grid scale times finer
#   detection              JSON {"boxes": [[x1, y1, x2, y2], ...],
#                          "labels": [...]} in pixel coordinates, or a
#                          GeoJSON FeatureCollection in map coordinates
#   scene regression       JSON {"values": [...]} or {"values": {name: v}},
#                          or one CSV file <split>/targets.csv with the
#                          columns id, <output 1>, <output 2>, ...
#
# Change detection images stack the bands of the first date followed by the
# bands of the second date. GeoJSON boxes are converted to pixels with the
# transform of the image; coordinates are assumed to be in the CRS of the
# image unless the collection declares "crs": "EPSG:<code>".
#
# dataset.yaml keys (all optional):
#
#   band_indices: [3, 2, 1, 7]   zero-based bands to read, in model order
#   classes: [ship]              class names; detection labels may use them
#   label_map: {0: 0, 1: 1, 255: 255}   remap raw mask values
#   scale: 0.0001                multiply image values (digital numbers to
#                                reflectance)
#   nodata: 0                    image value that marks missing pixels
#
# Synthetic data
# --------------
# SyntheticDataset generates learnable toy data for every trainable task:
# objects with class-specific spectra, class regions, changed regions between
# two dates, smooth targets that depend on the bands, degraded images and
# downsampled images. It needs no files and is used by the tests, the
# tutorials and the --synthetic option of the train command to check that a
# training setup works before real data is prepared.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# CSV target tables.
import csv

# JSON label files.
import json

# Hash of family names for synthetic spectra.
import zlib

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Dataset settings.
import yaml

# Array type annotations.
from numpy.typing import NDArray

# Smoothing of random fields for synthetic data.
from scipy.ndimage import gaussian_filter

# Samples, crops and the ignored label.
from unbihexium.ai.transforms import IGNORE_INDEX, Sample, crop_boxes

# Configuration and task of a model.
from unbihexium.zoo.catalog import Task
from unbihexium.zoo.config import BuildConfig  # Effective configuration.

# File extensions of image and dense label files, in lookup order.
RASTER_EXTENSIONS = (".tif", ".tiff", ".npy", ".npz", ".png", ".jpg")

# File extensions of detection and scene label files.
JSON_EXTENSIONS = (".json", ".geojson")

# Names of the dataset splits.
SPLITS = ("train", "val", "test")


# Whether a path has one of the supported raster extensions.
def _is_raster(path: Path) -> bool:
    # Case-insensitive comparison of the extension.
    return path.suffix.lower() in RASTER_EXTENSIONS


# Raised for missing or malformed dataset files.
class DatasetError(ValueError):
    # Plain subclass so that callers can catch dataset problems specifically.
    pass


# Rasterio window of a (top, left, height, width) tuple.
def _window(window: tuple[int, int, int, int]) -> Any:
    # Imported here so that NumPy-only datasets work without GDAL setup.
    from rasterio.windows import Window

    # Top, left, height and width.
    top, left, height, width = window
    # Rows and columns as start and stop indices.
    return Window.from_slices((top, top + height), (left, left + width))


# Read an array file as (bands, rows, cols), optionally only a window.
def read_raster(
    path: Path,  # File to read.
    window: tuple[int, int, int, int] | None = None,  # (top, left, height, width).
    bands: list[int] | None = None,  # Zero-based bands to read.
) -> NDArray[Any]:  # Array with a band axis.
    # Lower-case extension decides the reader.
    ext = path.suffix.lower()
    # NumPy arrays; memory mapping avoids reading unused parts.
    if ext == ".npy":
        # Memory-mapped array.
        data = np.load(path, mmap_mode="r")
    # Compressed NumPy archives.
    elif ext == ".npz":
        # Open the archive.
        with np.load(path) as archive:
            # First array, or the array named "arr_0", "image" or "label".
            key = next(k for k in ("image", "label", "arr_0", *archive.files) if k in archive)
            # Load it.
            data = archive[key]
    # Everything else is read with rasterio.
    else:
        # Imported here so that NumPy-only datasets work without GDAL setup.
        import rasterio

        # Open the file.
        with rasterio.open(path) as src:
            # Rasterio band indices are one-based.
            indexes = [b + 1 for b in bands] if bands is not None else None
            # Window in rasterio form.
            win = _window(window) if window else None
            # Read the requested bands and window.
            data = src.read(indexes=indexes, window=win)
        # Bands were already selected.
        bands = None
        # The window was already applied.
        window = None
    # Add a band axis to single-band arrays.
    if data.ndim == 2:
        # One band.
        data = data[None]
    # Select bands of NumPy arrays.
    if bands is not None:
        # Fancy indexing on the band axis.
        data = data[bands]
    # Cut the window from NumPy arrays.
    if window is not None:
        # Window bounds.
        top, left, height, width = window
        # Slice rows and columns.
        data = data[:, top : top + height, left : left + width]
    # Return an in-memory array.
    return np.asarray(data)


# Shape (bands, rows, cols) of an array file without reading its pixels.
def raster_shape(path: Path) -> tuple[int, int, int]:
    # Lower-case extension decides the reader.
    ext = path.suffix.lower()
    # NumPy files: header only through memory mapping.
    if ext in (".npy", ".npz"):
        # Shape of the array.
        shape = read_raster(path).shape if ext == ".npz" else np.load(path, mmap_mode="r").shape
    # Raster files: header of the dataset.
    else:
        # Imported here for NumPy-only datasets.
        import rasterio

        # Open the file.
        with rasterio.open(path) as src:
            # Bands, rows and columns.
            shape = (src.count, src.height, src.width)
    # Two-dimensional arrays have one band.
    return (1, *shape) if len(shape) == 2 else tuple(shape)  # type: ignore[return-value]


# Affine transform and CRS of a raster file, or None for NumPy files.
def raster_georeference(path: Path) -> tuple[tuple[float, ...], str] | None:
    # NumPy files carry no georeferencing.
    if path.suffix.lower() not in (".tif", ".tiff"):
        # Pixel coordinates only.
        return None
    # Imported here for NumPy-only datasets.
    import rasterio

    # Open the file.
    with rasterio.open(path) as src:
        # Six affine coefficients and the CRS string.
        return tuple(src.transform)[:6], str(src.crs) if src.crs else ""


# Find the file with a given stem and one of the given extensions.
def find_file(directory: Path, stem: str, extensions: tuple[str, ...]) -> Path | None:
    # Try the extensions in order.
    for ext in extensions:
        # Candidate path.
        candidate = directory / f"{stem}{ext}"
        # Return the first existing file.
        if candidate.is_file():
            # Found.
            return candidate
    # No file with that stem.
    return None


# Convert box labels given as names or numbers to class indices.
def resolve_labels(labels: list[Any], classes: list[str]) -> NDArray[np.int64]:
    # Class indices.
    out = []
    # Convert every label.
    for label in labels:
        # Names are looked up in the class list.
        if isinstance(label, str) and not label.isdigit():
            # Unknown names are an error.
            if label not in classes:
                # Explain the problem.
                raise DatasetError(f"unknown class {label!r}; known classes: {classes}")
            # Index of the name.
            out.append(classes.index(label))
        # Numbers are used as indices.
        else:
            # Integer index.
            out.append(int(label))
    # Return an int64 array.
    return np.asarray(out, dtype=np.int64)


# Read detection boxes from a JSON or GeoJSON file, in pixel coordinates.
def read_boxes(
    path: Path,  # Label file.
    classes: list[str],  # Class names of the model.
    georeference: tuple[tuple[float, ...], str] | None = None,  # Transform and CRS of the image.
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:  # Boxes and class indices.
    # Parse the file.
    data = json.loads(path.read_text(encoding="utf-8"))
    # Pixel boxes with labels.
    if isinstance(data, dict) and "boxes" in data:
        # Boxes as float32.
        boxes = np.asarray(data["boxes"], dtype=np.float32).reshape(-1, 4)
        # Labels default to the first class.
        labels = resolve_labels(data.get("labels", [0] * len(boxes)), classes)
        # Return the arrays.
        return boxes, labels
    # GeoJSON feature collections.
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        # Convert the features.
        return _geojson_boxes(data, classes, georeference)
    # Anything else is unsupported.
    raise DatasetError(f"{path}: expected {{'boxes': ...}} or a GeoJSON FeatureCollection")


# All coordinate pairs of a GeoJSON geometry.
def _coordinates(geometry: dict[str, Any]) -> NDArray[np.float64]:
    # Flatten nested coordinate lists down to pairs.
    flat = np.asarray(geometry["coordinates"], dtype=object)

    # Recursive flattening of irregular nesting (polygons with holes).
    def pairs(value: Any) -> list[list[float]]:
        # A pair of numbers is a coordinate.
        if len(value) and not isinstance(value[0], (list, tuple, np.ndarray)):
            # Longitude and latitude, or x and y.
            return [[float(value[0]), float(value[1])]]
        # Recurse into nested lists.
        return [p for item in value for p in pairs(item)]

    # Coordinates as an (N, 2) array.
    return np.asarray(pairs(flat.tolist()), dtype=np.float64)


# Convert GeoJSON features to pixel boxes.
def _geojson_boxes(
    data: dict[str, Any],  # Feature collection.
    classes: list[str],  # Class names.
    georeference: tuple[tuple[float, ...], str] | None,  # Transform and CRS of the image.
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:  # Boxes and class indices.
    # Boxes and labels collected from the features.
    boxes, labels = [], []
    # Optional coordinate transformer from the collection CRS to the image CRS.
    transformer = None
    # Declared CRS of the collection.
    declared = data.get("crs")
    # Reproject when both CRS are known and differ.
    if georeference and isinstance(declared, str) and declared != georeference[1]:
        # Imported here because only reprojected datasets need it.
        from pyproj import Transformer

        # Transformer with x, y order.
        transformer = Transformer.from_crs(declared, georeference[1], always_xy=True)
    # Inverse of the image transform maps coordinates to pixels.
    inverse = None
    # Compute the inverse for georeferenced images.
    if georeference:
        # Affine coefficients.
        a, b, c, d, e, f = georeference[0]
        # Matrix of the linear part.
        m = np.array([[a, b], [d, e]], dtype=np.float64)
        # Inverse matrix and translation.
        inverse = (np.linalg.inv(m), np.array([c, f], dtype=np.float64))
    # Convert every feature.
    for feature in data.get("features", []):
        # Skip features without geometry.
        if not feature.get("geometry"):
            # Nothing to convert.
            continue
        # Coordinates of the geometry.
        xy = _coordinates(feature["geometry"])
        # Reproject if needed.
        if transformer is not None:
            # Transform x and y.
            xs, ys = transformer.transform(xy[:, 0], xy[:, 1])
            # Stack the result.
            xy = np.stack([xs, ys], axis=1)
        # Map to pixel coordinates.
        if inverse is not None:
            # Solve the affine transform for (col, row).
            xy = (xy - inverse[1]) @ inverse[0].T
        # Bounding box in pixels.
        boxes.append([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])
        # Class from the properties, "class" or "label", default first class.
        props = feature.get("properties") or {}
        # Raw label.
        labels.append(props.get("class", props.get("label", 0)))
    # Arrays of boxes and labels.
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),  # Boxes.
        resolve_labels(labels, classes),  # Class indices.
    )  # End of the result.


# Labelled dataset in the folder layout described above.
class FolderDataset:
    # Index the samples of one split.
    def __init__(
        self,  # The dataset.
        root: str | Path,  # Dataset root.
        split: str,  # train, val or test.
        config: BuildConfig,  # Model configuration; decides the target type.
    ) -> None:  # The constructor returns nothing.
        # Root directory.
        self.root = Path(root)
        # Split name.
        self.split = split
        # Model configuration.
        self.config = config
        # Settings from dataset.yaml.
        self.settings = self._read_settings()
        # Image directory of the split.
        image_dir = self.root / split / "images"
        # A split without images is an error.
        if not image_dir.is_dir():
            # Explain the expected layout.
            raise DatasetError(f"{image_dir} does not exist; see docs/model_zoo/training.md")
        # Label directory of the split.
        self.label_dir = self.root / split / "labels"
        # Image files, sorted for reproducibility.
        self.images = sorted(p for p in image_dir.iterdir() if _is_raster(p))
        # An empty split is an error.
        if not self.images:
            # Explain the problem.
            raise DatasetError(f"no images in {image_dir}")
        # Scene targets from a CSV table, if present.
        self.table = self._read_table()
        # Class names for detection labels.
        self.classes = list(self.settings.get("classes") or config.outputs)

    # Read dataset.yaml.
    def _read_settings(self) -> dict[str, Any]:
        # Settings file.
        path = self.root / "dataset.yaml"
        # Missing file means defaults.
        if not path.is_file():
            # Empty settings.
            return {}
        # Parse the YAML file.
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        # The top level must be a mapping.
        if not isinstance(data, dict):
            # Explain the problem.
            raise DatasetError(f"{path} must contain a mapping")
        # Return the settings.
        return data

    # Read <split>/targets.csv for scene regression.
    def _read_table(self) -> dict[str, NDArray[np.float32]]:
        # Table file.
        path = self.root / self.split / "targets.csv"
        # Missing table.
        if not path.is_file():
            # No table targets.
            return {}
        # Targets per image id.
        table: dict[str, NDArray[np.float32]] = {}
        # Open the table.
        with path.open(newline="", encoding="utf-8") as fh:
            # Rows as dictionaries.
            for row in csv.DictReader(fh):
                # Image id.
                key = row.get("id") or row.get("image") or ""
                # Values in the order of the model outputs.
                values = [float(row[name]) for name in self.config.outputs]
                # Store them as float32.
                table[key] = np.asarray(values, dtype=np.float32)
        # Return the table.
        return table

    # Number of samples.
    def __len__(self) -> int:
        # One sample per image.
        return len(self.images)

    # Bands of the image files to read.
    @property
    def band_indices(self) -> list[int] | None:
        # Explicit selection from the settings.
        value = self.settings.get("band_indices")
        # List of integers or None.
        return [int(v) for v in value] if value is not None else None

    # Shape (bands, rows, cols) of an image.
    def shape(self, index: int) -> tuple[int, int, int]:
        # Header of the image file.
        c, h, w = raster_shape(self.images[index])
        # Selected bands.
        bands = self.band_indices
        # Number of bands after selection.
        return (len(bands) if bands is not None else c, h, w)

    # Read the image of a sample, scaled and with missing values as NaN.
    def _image(self, index: int, window: tuple[int, int, int, int] | None) -> NDArray[np.float32]:
        # Read the requested bands and window.
        image = read_raster(self.images[index], window, self.band_indices).astype(np.float32)
        # Value that marks missing pixels.
        nodata = self.settings.get("nodata")
        # Mark missing pixels.
        if nodata is not None:
            # Pixels where all bands equal the no-data value.
            missing = np.all(image == float(nodata), axis=0)
            # Set them to NaN.
            image[:, missing] = np.nan
        # Multiplicative scale, for example digital numbers to reflectance.
        scale = self.settings.get("scale")
        # Apply the scale.
        if scale is not None:
            # In-place multiplication.
            image *= float(scale)
        # The band count must match the model.
        if image.shape[0] != self.config.in_channels:
            # Explain the mismatch and the fix.
            raise DatasetError(
                f"{self.images[index].name}: {image.shape[0]} bands, the model expects "
                f"{self.config.in_channels} ({', '.join(self.config.channel_names)}); "
                "set band_indices in dataset.yaml"
            )  # End of the error.
        # Return the image.
        return image

    # Label file of a sample.
    def _label_path(self, index: int, extensions: tuple[str, ...]) -> Path:
        # File stem of the image.
        stem = self.images[index].stem
        # Find the label file.
        path = find_file(self.label_dir, stem, extensions)
        # Missing labels are an error.
        if path is None:
            # Explain the problem.
            raise DatasetError(f"no label file for {stem} in {self.label_dir}")
        # Return the path.
        return path

    # Load a sample, optionally only a window of the image.
    def load(self, index: int, window: tuple[int, int, int, int] | None = None) -> Sample:
        # Image window.
        image = self._image(index, window)
        # Sample name.
        name = self.images[index].stem
        # Task of the model.
        task = self.config.task
        # Class masks.
        if task in (Task.SEGMENTATION, Task.CHANGE_DETECTION):
            # Mask window.
            mask = read_raster(self._label_path(index, RASTER_EXTENSIONS), window)[0]
            # Optional remapping of raw values.
            mask = self._remap(mask.astype(np.int64))
            # Sample with a mask.
            return Sample(image=image, mask=mask, name=name)
        # Detection boxes.
        if task is Task.DETECTION:
            # Label file.
            path = self._label_path(index, JSON_EXTENSIONS)
            # Boxes in pixels of the full image.
            boxes, labels = read_boxes(path, self.classes, raster_georeference(self.images[index]))
            # Shift into the window.
            if window is not None:
                # Crop the boxes.
                boxes, labels = crop_boxes(boxes, labels, *window)  # type: ignore[assignment]
            # Sample with boxes.
            return Sample(image=image, boxes=boxes, labels=labels, name=name)
        # Scene targets.
        if task is Task.SCENE_REGRESSION:
            # Sample with a target vector.
            return Sample(image=image, vector=self._scene_target(index), name=name)
        # Dense targets on the image grid or a finer grid.
        scale = self.config.scale if task is Task.SUPER_RESOLUTION else 1
        # Target window on the target grid.
        target_window = tuple(v * scale for v in window) if window else None
        # Read the target.
        values = read_raster(self._label_path(index, RASTER_EXTENSIONS), target_window)  # type: ignore[arg-type]
        # The number of target bands must match the outputs.
        if values.shape[0] != self.config.out_channels:
            # Explain the mismatch.
            raise DatasetError(
                f"{name}: {values.shape[0]} target bands, expected {self.config.out_channels}"
            )  # End of the error.
        # Sample with dense targets.
        return Sample(image=image, values=values.astype(np.float32), name=name)

    # Apply label_map from the settings.
    def _remap(self, mask: NDArray[np.int64]) -> NDArray[np.int64]:
        # Mapping of raw values to class indices.
        mapping = self.settings.get("label_map")
        # Without a mapping the values are class indices.
        if not mapping:
            # Return the mask unchanged.
            return mask
        # Unmapped values are ignored.
        out = np.full_like(mask, IGNORE_INDEX)
        # Apply every entry.
        for raw, target in mapping.items():
            # Replace raw values.
            out[mask == int(raw)] = int(target)
        # Return the remapped mask.
        return out

    # Scene target of a sample from the table or a JSON file.
    def _scene_target(self, index: int) -> NDArray[np.float32]:
        # File stem.
        stem = self.images[index].stem
        # Table targets take precedence.
        if stem in self.table:
            # Values from the CSV file.
            return self.table[stem]
        # JSON label file.
        data = json.loads(self._label_path(index, JSON_EXTENSIONS).read_text(encoding="utf-8"))
        # Values as a list or a mapping.
        values = data.get("values", data)
        # Mappings are ordered by the model outputs.
        if isinstance(values, dict):
            # Values by output name.
            values = [values[name] for name in self.config.outputs]
        # Return float32 values.
        return np.asarray(values, dtype=np.float32).reshape(self.config.out_channels)

    # Load the full sample.
    def __getitem__(self, index: int) -> Sample:
        # Whole image.
        return self.load(index)


# Seeded smooth random field in [0, 1].
def _field(rng: np.random.Generator, shape: tuple[int, ...], sigma: float) -> NDArray[np.float64]:
    # Smoothed white noise.
    f = gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="wrap")
    # Rescale to [0, 1].
    return (f - f.min()) / max(float(f.max() - f.min()), 1e-12)


# Learnable toy data for a model configuration.
class SyntheticDataset:
    # Create the dataset.
    def __init__(
        self,  # The dataset.
        config: BuildConfig,  # Model configuration.
        length: int = 64,  # Number of samples.
        size: int = 64,  # Image height and width in pixels.
        seed: int = 0,  # Seed of the samples.
        noise: float = 0.02,  # Standard deviation of the image noise.
    ) -> None:  # The constructor returns nothing.
        # Spectral index models have no parameters to train.
        if not config.task.is_trainable:
            # Explain the problem.
            raise ValueError(f"{config.model_id} computes a fixed formula and is not trainable")
        # Model configuration.
        self.config = config
        # Number of samples.
        self.length = length
        # Image size.
        self.size = size
        # Seed of the samples.
        self.seed = seed
        # Image noise.
        self.noise = noise
        # Spectral signature per class, fixed per family.
        sig_rng = np.random.default_rng(zlib.crc32(config.family.encode()))
        # Change detection stacks two acquisition dates.
        dates = 2 if config.task is Task.CHANGE_DETECTION else 1
        # Bands per acquisition date.
        self.bands = config.in_channels // dates
        # Signatures of the classes (and of the background for detection).
        self.signatures = sig_rng.uniform(0.05, 0.95, (config.out_channels + 1, self.bands))
        # Band weights of the regression targets.
        self.weights = sig_rng.normal(0.0, 1.0, (config.out_channels, config.in_channels))

    # Number of samples.
    def __len__(self) -> int:
        # Fixed length.
        return self.length

    # Generate sample `index`.
    def __getitem__(self, index: int) -> Sample:
        # Independent generator per sample.
        rng = np.random.default_rng([self.seed, index])
        # Task of the model.
        task = self.config.task
        # Dispatch to the generator of the task.
        if task is Task.DETECTION:
            # Objects on a background.
            sample = self._detection(rng)
        # Class regions.
        elif task is Task.SEGMENTATION:
            # Regions with class spectra.
            sample = self._segmentation(rng)
        # Changed regions between two dates.
        elif task is Task.CHANGE_DETECTION:
            # Two dates with changes.
            sample = self._change(rng)
        # Targets that depend on the bands.
        elif task in (Task.DENSE_REGRESSION, Task.SCENE_REGRESSION):
            # Regression targets.
            sample = self._regression(rng)
        # Degraded images.
        elif task is Task.ENHANCEMENT:
            # Image-to-image pairs.
            sample = self._enhancement(rng)
        # Downsampled images.
        else:
            # Super-resolution pairs.
            sample = self._super_resolution(rng)
        # Name the sample.
        sample.name = f"synthetic_{index:05d}"
        # Return the sample.
        return sample

    # Image from a class map and the class signatures, with texture and noise.
    def _render(self, rng: np.random.Generator, classes: NDArray[Any]) -> NDArray[np.float32]:
        # Spectra of the pixels, shape (bands, H, W).
        image = np.moveaxis(self.signatures[classes], -1, 0)
        # Low-frequency brightness texture.
        texture = 0.1 * (_field(rng, classes.shape, 6.0) - 0.5)
        # Pixel noise.
        noise = rng.normal(0.0, self.noise, image.shape)
        # Combine and convert to float32.
        return (image + texture + noise).astype(np.float32)

    # Detection sample: rectangles with class spectra on a background.
    def _detection(self, rng: np.random.Generator) -> Sample:
        # Image size.
        s = self.size
        # Background class is the last signature.
        background = self.config.out_channels
        # Class map, initially background.
        classes = np.full((s, s), background, dtype=np.int64)
        # Number of objects.
        count = int(rng.integers(1, 6))
        # Boxes and labels.
        boxes, labels = [], []
        # Place the objects.
        for _ in range(count):
            # Box size between 6 and a quarter of the image.
            w, h = rng.integers(6, max(7, s // 4), size=2)
            # Top-left corner inside the image.
            x, y = int(rng.integers(0, s - w)), int(rng.integers(0, s - h))
            # Random class.
            k = int(rng.integers(0, self.config.out_channels))
            # Paint the object.
            classes[y : y + h, x : x + w] = k
            # Record the box.
            boxes.append([x, y, x + w, y + h])
            # Record the class.
            labels.append(k)
        # Render the image.
        image = self._render(rng, classes)
        # Sample with boxes; overlapping objects keep both boxes.
        return Sample(
            image=image,  # Image.
            boxes=np.asarray(boxes, dtype=np.float32),  # Boxes.
            labels=np.asarray(labels, dtype=np.int64),  # Classes.
        )  # End of the sample.

    # Segmentation sample: smooth class regions.
    def _segmentation(self, rng: np.random.Generator) -> Sample:
        # Number of classes.
        k = self.config.out_channels
        # Class with the largest smooth field wins at every pixel.
        classes = np.argmax(_field(rng, (k, self.size, self.size), 5.0), axis=0)
        # Sample with the class map.
        return Sample(image=self._render(rng, classes), mask=classes.astype(np.int64))

    # Change sample: two dates; changed regions take a new class spectrum.
    def _change(self, rng: np.random.Generator) -> Sample:
        # Number of change classes including no change.
        k = self.config.out_channels
        # Land cover of the first date.
        before = np.argmax(_field(rng, (k, self.size, self.size), 6.0), axis=0)
        # Changed regions: a smooth field above a threshold.
        changed = _field(rng, (self.size, self.size), 4.0) > 0.7
        # Change class of every changed pixel.
        kind = 1 + np.argmax(_field(rng, (max(k - 1, 1), self.size, self.size), 4.0), axis=0)
        # Reference: 0 for no change, otherwise the change class.
        mask = np.where(changed, kind, 0).astype(np.int64)
        # Land cover of the second date: changed pixels take their change class spectrum.
        after = np.where(changed, (before + kind) % (k + 1), before)
        # Both dates stacked on the band axis.
        image = np.concatenate([self._render(rng, before), self._render(rng, after)], axis=0)
        # Sample with the change map.
        return Sample(image=image, mask=mask)

    # Regression sample: targets are smooth functions of the bands.
    def _regression(self, rng: np.random.Generator) -> Sample:
        # Image bands: independent smooth fields.
        image = _field(rng, (self.config.in_channels, self.size, self.size), 3.0).astype(np.float32)
        # Linear combination of the centred bands per output.
        z = np.tensordot(self.weights, image - 0.5, axes=1) / np.sqrt(self.config.in_channels)
        # Bounds of the targets.
        lo, hi = self.config.value_range or (0.0, 1.0)
        # Squash into the bounds.
        values = (lo + (hi - lo) / (1 + np.exp(-4 * z))).astype(np.float32)
        # Add pixel noise to the image.
        image = image + rng.normal(0.0, self.noise, image.shape).astype(np.float32)
        # Scene models predict the mean over the image.
        if self.config.task is Task.SCENE_REGRESSION:
            # Sample with a target vector.
            return Sample(image=image, vector=values.mean(axis=(1, 2)).astype(np.float32))
        # Sample with a target map.
        return Sample(image=image, values=values)

    # Enhancement sample: the input is a degraded version of the target.
    def _enhancement(self, rng: np.random.Generator) -> Sample:
        # Number of output bands.
        k = self.config.out_channels
        # Displacement models learn a constant shift between two images.
        if set(self.config.outputs) >= {"dx", "dy"}:
            # Integer shift in pixels.
            dx, dy = (int(v) for v in rng.integers(-2, 3, size=2))
            # Reference image.
            half = self.config.in_channels // 2
            # Reference bands.
            ref = _field(rng, (half, self.size, self.size), 2.0)
            # Moved image: the reference shifted by (dx, dy).
            moved = np.roll(ref, (dy, dx), axis=(1, 2))
            # Stack both images.
            image = np.concatenate([ref, moved], axis=0).astype(np.float32)
            # Constant displacement field.
            values = np.zeros((k, self.size, self.size), dtype=np.float32)
            # Column shift.
            values[self.config.outputs.index("dx")] = dx
            # Row shift.
            values[self.config.outputs.index("dy")] = dy
            # Sample with the displacement field.
            return Sample(image=image, values=values)
        # Clean target bands.
        target = _field(rng, (k, self.size, self.size), 2.0).astype(np.float32)
        # Input bands: blurred and noisy copies of the targets, cycling over them.
        blurred = [gaussian_filter(target[c % k], 1.0) for c in range(self.config.in_channels)]
        # Additive noise of the degraded bands.
        noise = rng.normal(0, self.noise * 2, (self.config.in_channels, *target.shape[1:]))
        # Degraded input bands.
        image = (np.stack(blurred) + noise).astype(np.float32)
        # Sample with the clean target.
        return Sample(image=image, values=target)

    # Super-resolution sample: the input is a block average of the target.
    def _super_resolution(self, rng: np.random.Generator) -> Sample:
        # Upscaling factor.
        s = self.config.scale
        # High-resolution target.
        target = _field(rng, (self.config.out_channels, self.size * s, self.size * s), 2.0 * s)
        # Block average over s x s pixels.
        low = target.reshape(target.shape[0], self.size, s, self.size, s).mean(axis=(2, 4))
        # Input bands cycle over the target bands.
        image = np.stack([low[c % low.shape[0]] for c in range(self.config.in_channels)])
        # Sample with the high-resolution target.
        return Sample(image=image.astype(np.float32), values=target.astype(np.float32))


# Gaussian radius of a box so that a shifted box still has the given IoU.
def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    # Case 1 of CornerNet (Law and Deng, 2018): both corners move.
    b1 = height + width
    # Constant term.
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    # Radius of case 1.
    r1 = (b1 + np.sqrt(b1**2 - 4 * c1)) / 2
    # Case 2: the box shrinks.
    b2 = 2 * (height + width)
    # Constant term.
    c2 = (1 - min_overlap) * width * height
    # Radius of case 2.
    r2 = (b2 + np.sqrt(b2**2 - 16 * c2)) / 8
    # Case 3: the box grows.
    a3 = 4 * min_overlap
    # Linear term.
    b3 = -2 * min_overlap * (height + width)
    # Constant term.
    c3 = (min_overlap - 1) * width * height
    # Radius of case 3.
    r3 = (-b3 + np.sqrt(b3**2 - 4 * a3 * c3)) / (2 * a3)
    # The smallest radius satisfies all cases.
    return float(min(r1, r2, r3))


# CenterNet training targets for one image.
def encode_centernet(
    boxes: NDArray[Any] | None,  # (N, 4) boxes in input pixels.
    labels: NDArray[Any] | None,  # (N,) class ids.
    num_classes: int,  # Number of heat maps.
    height: int,  # Input height.
    width: int,  # Input width.
    stride: int,  # Output stride.
) -> dict[str, NDArray[np.float32]]:  # Heat map, size, offset and weight arrays.
    # Output grid size.
    h, w = height // stride, width // stride
    # Class heat maps.
    heatmap = np.zeros((num_classes, h, w), dtype=np.float32)
    # Box width and height at the centres, in output pixels.
    size = np.zeros((2, h, w), dtype=np.float32)
    # Sub-pixel offsets at the centres.
    offset = np.zeros((2, h, w), dtype=np.float32)
    # One where a box centre lies, zero elsewhere.
    weight = np.zeros((1, h, w), dtype=np.float32)
    # Images without objects have empty targets.
    if boxes is None or labels is None or len(boxes) == 0:
        # Return the empty targets.
        return {"heatmap": heatmap, "size": size, "offset": offset, "weight": weight}
    # Row and column grids of the output.
    ys, xs = np.mgrid[0:h, 0:w]
    # Encode every box.
    for (x1, y1, x2, y2), k in zip(np.asarray(boxes, dtype=np.float64), labels):
        # Box size in output pixels.
        bw, bh = (x2 - x1) / stride, (y2 - y1) / stride
        # Skip degenerate boxes.
        if bw <= 0 or bh <= 0:
            # Nothing to encode.
            continue
        # Box centre in output pixels.
        cx, cy = (x1 + x2) / 2 / stride, (y1 + y2) / 2 / stride
        # Integer cell of the centre.
        ix, iy = min(int(cx), w - 1), min(int(cy), h - 1)
        # Radius and standard deviation of the Gaussian peak.
        sigma = (2 * max(0.0, gaussian_radius(bh, bw)) + 1) / 6
        # Gaussian around the centre cell.
        g = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma**2))
        # Keep the maximum where peaks overlap.
        np.maximum(heatmap[int(k)], g.astype(np.float32), out=heatmap[int(k)])
        # Size target at the centre.
        size[:, iy, ix] = (bw, bh)
        # Offset target at the centre.
        offset[:, iy, ix] = (cx - ix, cy - iy)
        # Mark the centre.
        weight[0, iy, ix] = 1.0
    # Return the targets.
    return {"heatmap": heatmap, "size": size, "offset": offset, "weight": weight}


# =============================================================================
# End of module src/unbihexium/ai/data.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
