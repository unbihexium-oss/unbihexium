#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : examples/scripts/detect_ships.py
# Title       : Example: ship detection from satellite imagery
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires unbihexium
# =============================================================================
#
# Abstract
# --------
# Demonstrates how to detect ships in satellite imagery with the detection
# task API of Unbihexium and how to export the detections as a GeoJSON
# FeatureCollection. Each detection becomes a rectangular polygon built from
# its bounding box, with the class and the confidence as properties.
#
# Usage
# -----
#   python examples/scripts/detect_ships.py --input image.tif --output ships.geojson
#   python examples/scripts/detect_ships.py -i image.tif -o ships.geojson \
#       --model-id ship_detector_base
#
# Notes
# -----
# --model-id selects the model zoo model (a family such as ship_detector or
# a variant such as ship_detector_large); its task must be detection. When
# the input is georeferenced, the boxes are written in the map coordinates
# of the raster and the CRS is recorded in the named "crs" member of the
# GeoJSON 2008 specification, which unbihexium.io.reproject_geojson reads to
# convert the file to longitude and latitude. Inputs without a CRS or with
# the identity transform are written in pixel coordinates (column, row)
# with "crs" set to "pixel". Detections are model output and must be
# validated before they are used for decisions (see RESPONSIBLE_USE.md).
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Parse the command line options.
import argparse

# Represent the input file path.
from pathlib import Path

# Ship detection task API of Unbihexium.
from unbihexium.ai.detection import ShipDetector

# Raster container with pixel data and georeferencing metadata.
from unbihexium.core.raster import IDENTITY_TRANSFORM, Raster

# GeoJSON construction and validated writing.
from unbihexium.io.geojson import features_to_geojson, write_geojson

# Transforms of rasters without georeferencing: the library default and the
# identity that GDAL reports for plain images.
PIXEL_TRANSFORMS = {IDENTITY_TRANSFORM, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)}


# Whether a raster has a CRS and a transform other than the identity.
def is_georeferenced(raster: Raster) -> bool:
    # Metadata of the raster.
    meta = raster.metadata
    # Both a CRS and a real transform are needed for map coordinates.
    return meta is not None and bool(meta.crs) and tuple(meta.transform) not in PIXEL_TRANSFORMS


# Parse the options, run the detector and write the detections.
def main() -> None:
    # Command line interface of the example.
    parser = argparse.ArgumentParser(description="Detect ships in satellite image")
    # Path of the input GeoTIFF.
    parser.add_argument("--input", "-i", required=True, help="Input GeoTIFF file")
    # Path of the GeoJSON file to write.
    parser.add_argument("--output", "-o", required=True, help="Output GeoJSON file")
    # Minimum confidence for a detection to be kept, between 0 and 1.
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Detection threshold")
    # Model zoo model that runs the detection.
    parser.add_argument("--model-id", default="ship_detector_tiny", help="Model ID")
    # Parse sys.argv; argparse exits with a usage message on invalid input.
    args = parser.parse_args()

    # Tell the user which file is being read.
    print(f"Loading: {args.input}")
    # Open the raster.
    raster = Raster.from_file(Path(args.input))

    # Create the detector for the requested model and threshold.
    detector = ShipDetector(model=args.model_id, threshold=args.threshold)
    # Report the model that actually runs.
    print(f"Using model: {detector.model_id}")
    # Report the confidence threshold.
    print(f"Threshold: {args.threshold}")
    # Run the detection on the whole raster.
    result = detector.predict(raster)

    # Report the number of detections.
    print(f"Found {result.count} ships")

    # Map coordinates for georeferenced inputs, pixel coordinates otherwise.
    georeferenced = is_georeferenced(raster)
    # Features of the result in the chosen coordinates.
    features = result.to_geojson(pixel_coordinates=not georeferenced)["features"]
    # Collection with the named CRS member of the raster.
    if georeferenced:
        # CRS of the map boxes.
        geojson = features_to_geojson(features, crs=result.crs)
    # Pixel coordinates have no CRS.
    else:
        # Collection marked as pixel coordinates.
        geojson = {"type": "FeatureCollection", "crs": "pixel", "features": features}
    # Record the model as a foreign member.
    geojson["model_id"] = result.model_id

    # Tell the user where the result is written and in which coordinates.
    print(f"Saving: {args.output} ({result.crs if georeferenced else 'pixel coordinates'})")
    # Validate and write the file atomically.
    write_geojson(geojson, args.output)

    # Signal completion.
    print("Done!")


# Run the example when the file is executed as a script.
if __name__ == "__main__":
    # Start the example.
    main()

# =============================================================================
# End of module examples/scripts/detect_ships.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
