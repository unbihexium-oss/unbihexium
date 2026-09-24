# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/postprocessing/morphology.py
# Title       : Morphological cleaning, sieving and connected components
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy, SciPy and rasterio
# =============================================================================
#
# Abstract
# --------
# Cleaning of binary masks and class maps before they are published or
# vectorised:
#
#   structuring_element     square, disk or cross footprints
#   morphology_clean        opening, closing, erosion or dilation of masks
#   remove_small_objects    drop connected components below a size
#   fill_small_holes        fill background holes below a size
#   sieve                   minimum mapping unit for class maps: regions
#                           below the size take the class of their largest
#                           neighbour (GDAL sieve through rasterio)
#   majority_filter         modal filter of class maps, ignoring nodata
#   connected_components    label connected regions
#   component_statistics    area, bounding box, centroid and value
#                           statistics of every region
#
# Connectivity is 4 (edge neighbours) or 8 (edge and corner neighbours).
#
# References
# ----------
#   Serra, J. (1982). Image Analysis and Mathematical Morphology. Academic
#     Press.
#   Soille, P. (2003). Morphological Image Analysis: Principles and
#     Applications, 2nd ed. Springer.
#   Saura, S. (2002). Effects of minimum mapping unit on land cover data
#     spatial configuration and composition. International Journal of
#     Remote Sensing 23(22), 4853-4880.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Morphology, labelling and filters of SciPy.
from scipy import ndimage


# Structure of 4- or 8-connectivity for labelling.
def _connectivity(connectivity: int) -> NDArray[np.bool_]:
    # Only 4 and 8 are defined on a square grid.
    if connectivity not in (4, 8):
        # Explain the accepted values.
        raise ValueError("connectivity must be 4 or 8")
    # Rank one gives edge neighbours, rank two adds corners.
    return ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)


# Footprint of a given shape and size.
def structuring_element(size: int = 3, shape: str = "square") -> NDArray[np.bool_]:
    # The size must be a positive odd number so that the element is centred.
    if size < 1 or size % 2 == 0:
        # Explain the requirement.
        raise ValueError("size must be a positive odd integer")
    # Radius of the element.
    r = size // 2
    # Offsets from the centre.
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    # Full square.
    if shape == "square":
        # Every offset.
        return np.ones((size, size), dtype=bool)
    # Euclidean disk.
    if shape == "disk":
        # Offsets within the radius.
        return yy**2 + xx**2 <= r**2
    # Plus-shaped element.
    if shape == "cross":
        # Centre row and centre column.
        return (yy == 0) | (xx == 0)
    # Unknown shapes are rejected.
    raise ValueError("shape must be 'square', 'disk' or 'cross'")


# Binary morphology of a mask.
def morphology_clean(
    mask: NDArray[Any],  # Binary mask.
    operation: str = "open",  # open, close, erode or dilate.
    kernel_size: int = 3,  # Size of the structuring element.
    shape: str = "square",  # Shape of the structuring element.
    iterations: int = 1,  # Number of repetitions.
) -> NDArray[Any]:  # Cleaned mask in the input data type.
    # Operations by name.
    ops = {
        "open": ndimage.binary_opening,  # Erosion then dilation: removes specks.
        "close": ndimage.binary_closing,  # Dilation then erosion: fills gaps.
        "erode": ndimage.binary_erosion,  # Shrink objects.
        "dilate": ndimage.binary_dilation,  # Grow objects.
    }  # End of the operations.
    # The operation must be known.
    if operation not in ops:
        # Explain the accepted names.
        raise ValueError(f"operation must be one of {sorted(ops)}")
    # Mask as array.
    m = np.asarray(mask)
    # Structuring element.
    structure = structuring_element(kernel_size, shape)
    # Opening and closing border handling: pixels outside count as background.
    out = ops[operation](m > 0, structure=structure, iterations=iterations)
    # Keep the input data type.
    return out.astype(m.dtype)


# Remove connected components with fewer than min_size pixels.
def remove_small_objects(
    mask: NDArray[Any],  # Binary mask.
    min_size: int = 100,  # Smallest size that is kept, in pixels.
    connectivity: int = 8,  # 4 or 8.
) -> NDArray[Any]:  # Cleaned mask in the input data type.
    # Mask as array.
    m = np.asarray(mask)
    # Label the components.
    labels, _ = ndimage.label(m > 0, structure=_connectivity(connectivity))
    # Size of every label.
    sizes = np.bincount(labels.ravel())
    # Labels that are large enough.
    keep = sizes >= min_size
    # The background is never kept.
    keep[0] = False
    # Keep the pixels of large components.
    return keep[labels].astype(m.dtype)


# Fill holes of the background with fewer than max_size pixels.
def fill_small_holes(
    mask: NDArray[Any],  # Binary mask.
    max_size: int = 100,  # Largest hole that is filled, in pixels.
    connectivity: int = 4,  # Connectivity of the background.
) -> NDArray[Any]:  # Mask with small holes filled.
    # Mask as array.
    m = np.asarray(mask)
    # Label the background components.
    labels, _ = ndimage.label(m == 0, structure=_connectivity(connectivity))
    # Size of every background component.
    sizes = np.bincount(labels.ravel())
    # Components that touch the border are not holes.
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    # Holes small enough to fill.
    fill = sizes <= max_size
    # Border components and the foreground stay.
    fill[border] = False
    # Label zero is the foreground.
    fill[0] = False
    # Set the filled pixels.
    return ((m > 0) | fill[labels]).astype(m.dtype)


# Minimum mapping unit: merge small regions into their largest neighbour.
def sieve(
    labels: NDArray[Any],  # Class map.
    min_size: int,  # Smallest region that is kept, in pixels.
    connectivity: int = 4,  # 4 or 8.
    nodata: int | None = None,  # Value that is never changed nor merged into.
) -> NDArray[Any]:  # Sieved class map.
    # GDAL sieve through rasterio.
    from rasterio.features import sieve as gdal_sieve

    # Class map as array.
    arr = np.asarray(labels)
    # Class maps must hold integers.
    if not np.issubdtype(arr.dtype, np.integer):
        # Explain the requirement.
        raise ValueError("labels must be an integer array")
    # The connectivity must be valid.
    _connectivity(connectivity)
    # GDAL supports signed 32-bit integers for all class maps used here.
    work = arr.astype(np.int32)
    # Pixels excluded from sieving.
    mask = None if nodata is None else arr != nodata
    # Sieve the map.
    out = gdal_sieve(work, size=int(min_size), connectivity=connectivity, mask=mask)
    # Restore the nodata pixels, which GDAL leaves in place.
    if nodata is not None:
        # Keep the original values where masked.
        out = np.where(arr == nodata, work, out)
    # Return in the input data type.
    return out.astype(arr.dtype)


# Modal filter of a class map in a square window.
def majority_filter(
    labels: NDArray[Any],  # Class map.
    size: int = 3,  # Window size, odd.
    nodata: int | None = None,  # Label that does not vote and is kept.
) -> NDArray[Any]:  # Filtered class map.
    # Class map as array.
    arr = np.asarray(labels)
    # Class maps must hold integers.
    if not np.issubdtype(arr.dtype, np.integer):
        # Explain the requirement.
        raise ValueError("labels must be an integer array")
    # The window must be centred.
    if size < 1 or size % 2 == 0:
        # Explain the requirement.
        raise ValueError("size must be a positive odd integer")
    # Classes that vote.
    classes = np.unique(arr if nodata is None else arr[arr != nodata])
    # Nothing to filter without classes.
    if classes.size == 0:
        # Return a copy.
        return arr.copy()
    # One-hot layers of the classes, shape (K, H, W).
    onehot = (arr[None] == classes[:, None, None]).astype(np.float64)
    # Class fractions in the window; pixels outside the image do not vote.
    votes = ndimage.uniform_filter(onehot, size=(1, size, size), mode="constant")
    # Rounding removes floating point noise of the box filter.
    votes = np.round(votes * size * size)
    # Largest number of votes per pixel.
    best = votes.max(axis=0)
    # Index of the current label in the class list, -1 for nodata.
    current = np.searchsorted(classes, arr).clip(0, classes.size - 1)
    # Whether the current label is among the winners; ties keep it.
    keeps = np.take_along_axis(votes, current[None], axis=0)[0] == best
    # First winning class otherwise.
    winner = classes[np.argmax(votes, axis=0)]
    # Combine.
    out = np.where(keeps & (classes[current] == arr), arr, winner)
    # Nodata pixels are left untouched.
    if nodata is not None:
        # Restore them.
        out = np.where(arr == nodata, arr, out)
    # Return in the input data type.
    return out.astype(arr.dtype)


# Label connected regions of a mask, or of equal values in a class map.
def connected_components(
    image: NDArray[Any],  # Binary mask or class map.
    connectivity: int = 8,  # 4 or 8.
    background: int | None = 0,  # Value that is not labelled.
) -> tuple[NDArray[np.int32], int]:  # Label image and number of regions.
    # Input as array.
    arr = np.asarray(image)
    # Structure of the connectivity.
    structure = _connectivity(connectivity)
    # Output labels.
    out = np.zeros(arr.shape, dtype=np.int32)
    # Next free label.
    count = 0
    # Label every value separately so that touching classes stay apart.
    for value in np.unique(arr):
        # Skip the background.
        if background is not None and value == background:
            # Next value.
            continue
        # Regions of this value.
        lab, n = ndimage.label(arr == value, structure=structure)
        # Shift them past the labels used so far.
        out[lab > 0] = lab[lab > 0] + count
        # Advance the counter.
        count += n
    # Return the labels and their number.
    return out, count


# Statistics of every labelled region.
def component_statistics(
    labels: NDArray[Any],  # Label image from connected_components.
    values: NDArray[Any] | None = None,  # Optional image for mean, min and max.
    pixel_area: float = 1.0,  # Area of one pixel in map units.
    source: NDArray[Any] | None = None,  # Optional class map to report the class.
) -> list[dict[str, Any]]:  # One record per region, by label.
    # Labels as integer array.
    lab = np.asarray(labels).astype(np.int64)
    # Largest label.
    n = int(lab.max()) if lab.size else 0
    # No regions, no records.
    if n == 0:
        # Empty list.
        return []
    # Label numbers.
    index = np.arange(1, n + 1)
    # Pixel count per label.
    counts = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    # Row and column centroids.
    centroids = np.asarray(ndimage.center_of_mass(np.ones(lab.shape), lab, index)).reshape(-1, 2)
    # Bounding slices per label.
    boxes = ndimage.find_objects(lab, max_label=n)
    # Value statistics per label, when an image is given.
    stats: dict[str, NDArray[Any]] = {}
    # Compute them.
    if values is not None:
        # Values as float.
        v = np.asarray(values, dtype=np.float64)
        # Mean, minimum and maximum per label.
        stats = {
            "mean": np.asarray(ndimage.mean(v, lab, index)),  # Means.
            "min": np.asarray(ndimage.minimum(v, lab, index)),  # Minima.
            "max": np.asarray(ndimage.maximum(v, lab, index)),  # Maxima.
        }  # End of the statistics.
    # Records of every region.
    records = []
    # Build the records.
    for i, label in enumerate(index):
        # Bounding slices of the label, None when it has no pixels.
        box = boxes[i]
        # Skip label numbers without pixels.
        if counts[i] == 0 or box is None:
            # Next label.
            continue
        # Row and column slices of the region.
        rs, cs = box
        # Record of the region.
        rec: dict[str, Any] = {
            "label": int(label),  # Region number.
            "pixels": int(counts[i]),  # Size in pixels.
            "area": float(counts[i] * pixel_area),  # Size in map units.
            "bbox": (rs.start, cs.start, rs.stop, cs.stop),  # Row/column bounds.
            "centroid": (float(centroids[i, 0]), float(centroids[i, 1])),  # Row, column.
        }  # End of the record.
        # Add the value statistics.
        for key, stat in stats.items():
            # Statistic of this region.
            rec[key] = float(stat[i])
        # Add the class of the region.
        if source is not None:
            # Any pixel of the region carries its class.
            rec["value"] = np.asarray(source)[rs, cs][lab[rs, cs] == label][0].item()
        # Keep the record.
        records.append(rec)
    # Return the records.
    return records


# =============================================================================
# End of module src/unbihexium/postprocessing/morphology.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
