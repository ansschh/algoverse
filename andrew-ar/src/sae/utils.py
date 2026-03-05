"""
Utility functions for the SAE-Feature FAISS Benchmark pipeline.

Memmap loader, chunked iteration, checkpoint helpers.
"""

import json
import os
import numpy as np
from typing import Tuple, Iterator, Optional

from .config import EXTRACT_DTYPE


def model_dir(data_dir: str, model_name: str) -> str:
    """Return and ensure the per-model data directory exists."""
    d = os.path.join(data_dir, model_name)
    os.makedirs(d, exist_ok=True)
    return d


def load_memmap(path: str, shape: Tuple[int, ...],
                dtype: str = EXTRACT_DTYPE) -> np.memmap:
    """Load an existing memmap file as read-only."""
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def create_memmap(path: str, shape: Tuple[int, ...],
                  dtype: str = EXTRACT_DTYPE) -> np.memmap:
    """Create a new memmap file (or open existing for writing)."""
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def open_memmap_rw(path: str, shape: Tuple[int, ...],
                   dtype: str = EXTRACT_DTYPE) -> np.memmap:
    """Open an existing memmap file for read/write (resume)."""
    return np.memmap(path, dtype=dtype, mode="r+", shape=shape)


def iter_chunks(total: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    """Yield (start, end) index pairs for chunked iteration."""
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield start, end


def load_metadata(path: str) -> dict:
    """Load JSON metadata file, return empty dict if missing."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_metadata(path: str, data: dict):
    """Save JSON metadata file atomically."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def activation_path(data_dir: str, model_name: str) -> str:
    """Path to the activation memmap file."""
    return os.path.join(model_dir(data_dir, model_name), "activations.dat")


def activation_meta_path(data_dir: str, model_name: str) -> str:
    """Path to the activation metadata JSON."""
    return os.path.join(model_dir(data_dir, model_name), "activations_meta.json")


def token_ids_path(data_dir: str) -> str:
    """Path to the shared token_ids file."""
    return os.path.join(data_dir, "token_ids.npy")
