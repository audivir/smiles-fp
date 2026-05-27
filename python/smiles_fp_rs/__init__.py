"""SmilesFp module."""

from __future__ import annotations

# initialize dynamic libraries
import rdkit  # noqa: F401

from smiles_fp_rs._smiles_fp_rs import (
    bulk_tanimoto_mmap,
    bulk_tanimoto_mmap_topk,
    bulk_tanimoto_parallel,
    bulk_tanimoto_parallel_topk,
    internal_tanimoto_mmap,
    internal_tanimoto_parallel,
    load_fingerprints,
    save_fingerprints,
)
from smiles_fp_rs.helpers import get_mols, get_morgan_fps, mol_from_smi, to_matrix
from smiles_fp_rs.search import similarity_search, windowed_bulk_tanimoto

__all__ = [
    "bulk_tanimoto_mmap",
    "bulk_tanimoto_mmap_topk",
    "bulk_tanimoto_parallel",
    "bulk_tanimoto_parallel_topk",
    "get_mols",
    "get_morgan_fps",
    "internal_tanimoto_mmap",
    "internal_tanimoto_parallel",
    "load_fingerprints",
    "mol_from_smi",
    "save_fingerprints",
    "similarity_search",
    "similarity_search",
    "to_matrix",
    "windowed_bulk_tanimoto",
]
