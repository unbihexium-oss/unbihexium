"""Zarr IO adapter for chunked array storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def read_zarr(
    path: str | Path,
    variable: str | None = None,
) -> tuple[NDArray[np.floating[Any]], dict[str, Any]]:
    """Read data from Zarr store.

    Args:
        path: Path to Zarr store.
        variable: Variable name to read (for xarray datasets).

    Returns:
        Tuple of (data array, attributes dict).
    """
    try:
        import zarr
    except ImportError as e:
        raise ImportError("zarr is required for Zarr support") from e

    path = Path(path)
    store = zarr.open(path, mode="r")

    if variable is not None:
        data = np.array(store[variable])
        attrs = dict(store[variable].attrs)
    else:
        if hasattr(store, "shape"):
            data = np.array(store)
            attrs = dict(store.attrs)
        else:
            # Group - get first array
            arrays = list(store.arrays())
            if arrays:
                name, arr = arrays[0]
                data = np.array(arr)
                attrs = dict(arr.attrs)
            else:
                raise ValueError("No arrays found in Zarr store")

    return data.astype(np.float32), attrs


def write_zarr(
    path: str | Path,
    data: NDArray[np.floating[Any]],
    chunks: tuple[int, ...] | None = None,
    attrs: dict[str, Any] | None = None,
    compressor: str = "zlib",
) -> Path:
    """Write data to Zarr store.

    Args:
        path: Output path.
        data: Data array.
        chunks: Chunk sizes. None for auto-chunking.
        attrs: Attributes to store.
        compressor: Compression codec.

    Returns:
        Path to the written store.
    """
    try:
        import zarr
        from numcodecs import Blosc, Zlib
    except ImportError as e:
        raise ImportError("zarr is required for Zarr support") from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if chunks is None:
        chunks = tuple(min(256, s) for s in data.shape)

    if compressor == "zlib":
        comp = Zlib(level=5)
    else:
        comp = Blosc(cname="lz4", clevel=5)

    z = zarr.open(
        str(path),
        mode="w",
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        compressor=comp,
    )
    z[:] = data

    if attrs:
        z.attrs.update(attrs)

    return path
