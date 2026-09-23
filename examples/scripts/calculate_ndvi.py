#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : examples/scripts/calculate_ndvi.py
# Title       : Example: NDVI calculation from satellite imagery
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires unbihexium
# =============================================================================
#
# Abstract
# --------
# Demonstrates how to calculate the Normalized Difference Vegetation Index
# (NDVI) from a multi-band satellite image with Unbihexium and how to write
# the result as a single-band GeoTIFF that keeps the coordinate reference
# system and the geotransform of the input.
#
# Method
# ------
# NDVI = (NIR - RED) / (NIR + RED), computed per pixel from the near-infrared
# and red bands (Rouse et al., 1974, "Monitoring vegetation systems in the
# Great Plains with ERTS", Third ERTS Symposium, NASA SP-351, pp. 309-317).
# Values range from -1 to 1; dense green vegetation gives high values.
#
# Usage
# -----
#   python examples/scripts/calculate_ndvi.py --input image.tif --output ndvi.tif
#
# The default band numbers (NIR = 4, RED = 3) match a four-band image in the
# order blue, green, red, near-infrared. For a Sentinel-2 stack in the band
# order B1, B2, ..., B8, use --nir-band 8 --red-band 4.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Parse the command line options.
import argparse

# Represent the input and output file paths.
from pathlib import Path

# Compute summary statistics that ignore NaN pixels.
import numpy as np

# Spectral index calculator of Unbihexium.
from unbihexium.core.index import compute_index

# Raster container with pixel data and georeferencing metadata.
from unbihexium.core.raster import Raster

# GeoTIFF writer of Unbihexium.
from unbihexium.io.geotiff import write_geotiff


# Parse the options, compute NDVI and write the result.
def main() -> None:
    # Command line interface of the example.
    parser = argparse.ArgumentParser(description="Calculate NDVI from satellite image")
    # Path of the multi-band input GeoTIFF.
    parser.add_argument("--input", "-i", required=True, help="Input GeoTIFF file")
    # Path of the single-band NDVI GeoTIFF to write.
    parser.add_argument("--output", "-o", required=True, help="Output NDVI GeoTIFF")
    # Band number of the near-infrared band, counted from 1 as in GDAL.
    parser.add_argument("--nir-band", type=int, default=4, help="NIR band (1-indexed)")
    # Band number of the red band, counted from 1 as in GDAL.
    parser.add_argument("--red-band", type=int, default=3, help="RED band (1-indexed)")
    # Parse sys.argv; argparse exits with a usage message on invalid input.
    args = parser.parse_args()

    # Tell the user which file is being read.
    print(f"Loading: {args.input}")
    # Open the raster; this reads the metadata.
    raster = Raster.from_file(Path(args.input))
    # Read the pixel data into memory.
    raster.load()

    # Shape of the pixel array: (bands, rows, columns).
    print(f"Image shape: {raster.shape}")
    # Coordinate reference system of the input.
    print(f"CRS: {raster.metadata.crs}")

    # Extract bands (0-indexed)
    nir = raster.data[args.nir_band - 1]
    # Red band, converted from a 1-based band number to a 0-based index.
    red = raster.data[args.red_band - 1]

    # Announce the calculation.
    print("Calculating NDVI...")
    # compute_index looks the bands up by their standard names.
    bands = {"NIR": nir, "RED": red}
    # Per-pixel NDVI; pixels with NIR + RED = 0 become NaN.
    ndvi = compute_index("NDVI", bands)

    # Value range, ignoring NaN pixels.
    print(f"NDVI range: [{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}]")
    # Mean value, ignoring NaN pixels.
    print(f"NDVI mean: {np.nanmean(ndvi):.3f}")

    # Save result
    print(f"Saving: {args.output}")
    # Write a single-band GeoTIFF with the georeferencing of the input.
    write_geotiff(
        ndvi[np.newaxis, ...],  # Add band dimension
        Path(args.output),  # Output file path.
        crs=raster.metadata.crs,  # Same coordinate reference system as the input.
        transform=raster.metadata.transform,  # Same pixel-to-map geotransform.
    )  # End of the GeoTIFF write.

    # Signal completion.
    print("Done!")


# Run the example when the file is executed as a script.
if __name__ == "__main__":
    # Start the example.
    main()

# =============================================================================
# End of module examples/scripts/calculate_ndvi.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
