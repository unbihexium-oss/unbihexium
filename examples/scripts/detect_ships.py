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
# Demonstrates how to detect ships in satellite imagery with the AI detection
# module of Unbihexium and how to export the detections as a GeoJSON
# FeatureCollection. Each detection becomes a rectangular polygon built from
# its bounding box, with the class and the confidence as properties.
#
# Usage
# -----
#   python examples/scripts/detect_ships.py --input image.tif --output ships.geojson
#
# Notes
# -----
# The bounding boxes are written in the coordinates returned by the detector.
# The --model-id option is printed for information; the detector uses its
# default model. Detections are model output and must be validated before
# they are used for decisions (see RESPONSIBLE_USE.md).
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Parse the command line options.
import argparse

# Write the GeoJSON output.
import json

# Represent the input file path.
from pathlib import Path

# Ship detection model wrapper of Unbihexium.
from unbihexium.ai.detection import ShipDetector

# Raster container with pixel data and georeferencing metadata.
from unbihexium.core.raster import Raster


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
    # Identifier of the model zoo model to report.
    parser.add_argument("--model-id", default="ship_detector_tiny", help="Model ID")
    # Parse sys.argv; argparse exits with a usage message on invalid input.
    args = parser.parse_args()

    # Tell the user which file is being read.
    print(f"Loading: {args.input}")
    # Open the raster.
    raster = Raster.from_file(Path(args.input))

    # Report the model identifier.
    print(f"Using model: {args.model_id}")
    # Report the confidence threshold.
    print(f"Threshold: {args.threshold}")

    # Create the detector with the requested threshold.
    detector = ShipDetector(threshold=args.threshold)
    # Run the detection on the whole raster.
    result = detector.predict(raster)

    # Report the number of detections.
    print(f"Found {result.count} ships")

    # Convert to GeoJSON
    features = []
    # Build one GeoJSON Feature per detection.
    for det in result.detections:
        # bbox is (xmin, ymin, xmax, ymax).
        feature = {
            "type": "Feature",  # GeoJSON object type.
            "geometry": {  # Rectangle of the bounding box.
                "type": "Polygon",  # A single closed ring.
                "coordinates": [  # List of rings; only the exterior ring here.
                    [  # Exterior ring through the four corners of the box.
                        [det.bbox[0], det.bbox[1]],  # (xmin, ymin)
                        [det.bbox[2], det.bbox[1]],  # (xmax, ymin)
                        [det.bbox[2], det.bbox[3]],  # (xmax, ymax)
                        [det.bbox[0], det.bbox[3]],  # (xmin, ymax)
                        [det.bbox[0], det.bbox[1]],  # Back to the first point to close the ring.
                    ]  # End of the exterior ring.
                ],  # End of the ring list.
            },  # End of the geometry.
            "properties": {  # Attributes of the detection.
                "class_id": det.class_id,  # Numeric class identifier.
                "class_name": det.class_name,  # Human-readable class name.
                "confidence": det.confidence,  # Model confidence between 0 and 1.
            },  # End of the properties.
        }  # End of the feature.
        # Add the feature to the collection.
        features.append(feature)

    # Wrap the features in a FeatureCollection.
    geojson = {"type": "FeatureCollection", "features": features}

    # Tell the user where the result is written.
    print(f"Saving: {args.output}")
    # Open the output file for writing, replacing an existing file.
    with open(args.output, "w") as f:
        # Write indented JSON so that the file is readable.
        json.dump(geojson, f, indent=2)

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
