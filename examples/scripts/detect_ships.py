#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Example script: Ship detection from satellite imagery.

This script demonstrates how to detect ships in satellite imagery
using unbihexium's AI detection module.

Usage:
    python examples/scripts/detect_ships.py --input image.tif --output ships.geojson
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unbihexium.ai.detection import ShipDetector
from unbihexium.core.raster import Raster


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Detect ships in satellite image")
    parser.add_argument("--input", "-i", required=True, help="Input GeoTIFF file")
    parser.add_argument("--output", "-o", required=True, help="Output GeoJSON file")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--model-id", default="ship_detector_tiny", help="Model ID")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    raster = Raster.from_file(Path(args.input))

    print(f"Using model: {args.model_id}")
    print(f"Threshold: {args.threshold}")

    detector = ShipDetector(threshold=args.threshold)
    result = detector.predict(raster)

    print(f"Found {result.count} ships")

    # Convert to GeoJSON
    features = []
    for det in result.detections:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [det.bbox[0], det.bbox[1]],
                        [det.bbox[2], det.bbox[1]],
                        [det.bbox[2], det.bbox[3]],
                        [det.bbox[0], det.bbox[3]],
                        [det.bbox[0], det.bbox[1]],
                    ]
                ],
            },
            "properties": {
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": det.confidence,
            },
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    print(f"Saving: {args.output}")
    with open(args.output, "w") as f:
        json.dump(geojson, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
