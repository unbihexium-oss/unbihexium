#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Example script: NDVI calculation from satellite imagery.

This script demonstrates how to calculate NDVI from a multi-band
satellite image using unbihexium.

Usage:
    python examples/scripts/calculate_ndvi.py --input image.tif --output ndvi.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from unbihexium.core.index import compute_index
from unbihexium.core.raster import Raster
from unbihexium.io.geotiff import write_geotiff


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Calculate NDVI from satellite image")
    parser.add_argument("--input", "-i", required=True, help="Input GeoTIFF file")
    parser.add_argument("--output", "-o", required=True, help="Output NDVI GeoTIFF")
    parser.add_argument("--nir-band", type=int, default=4, help="NIR band (1-indexed)")
    parser.add_argument("--red-band", type=int, default=3, help="RED band (1-indexed)")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    raster = Raster.from_file(Path(args.input))
    raster.load()

    print(f"Image shape: {raster.shape}")
    print(f"CRS: {raster.metadata.crs}")

    # Extract bands (0-indexed)
    nir = raster.data[args.nir_band - 1]
    red = raster.data[args.red_band - 1]

    print("Calculating NDVI...")
    bands = {"NIR": nir, "RED": red}
    ndvi = compute_index("NDVI", bands)

    print(f"NDVI range: [{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}]")
    print(f"NDVI mean: {np.nanmean(ndvi):.3f}")

    # Save result
    print(f"Saving: {args.output}")
    write_geotiff(
        ndvi[np.newaxis, ...],  # Add band dimension
        Path(args.output),
        crs=raster.metadata.crs,
        transform=raster.metadata.transform,
    )

    print("Done!")


if __name__ == "__main__":
    main()
