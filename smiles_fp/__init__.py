"""SmilesFp module."""

from __future__ import annotations

# Import RDKit to ensure the required libraries are loaded
import rdkit  # noqa: F401

from smiles_fp._smiles_fp import (
    bulk_tanimoto_mmap,
    bulk_tanimoto_parallel,
    load_fingerprints,
    save_fingerprints,
)
from smiles_fp.search import similarity_search

__version__ = "0.1.2025.09.3"

__all__ = [
    "bulk_tanimoto_mmap",
    "bulk_tanimoto_parallel",
    "load_fingerprints",
    "save_fingerprints",
    "similarity_search",
]
