"""Rust-accelerated Tanimoto similarity search over RDKit Morgan fingerprints."""

from __future__ import annotations

# initialize dynamic libraries: RDKit libraries have no RUNPATH on Linux, so the dependencies of
# the linked libraries must already be loaded by the RDKit modules.
import rdkit  # noqa: F401
from rdkit import Chem  # noqa: F401
from rdkit.Chem import rdFingerprintGenerator  # noqa: F401

from smiles_fp._smiles_fp import (
    bulk_tanimoto_mmap,
    bulk_tanimoto_mmap_topk,
    bulk_tanimoto_parallel,
    bulk_tanimoto_parallel_topk,
    canonical_tautomers,
    clean,
    enumerate_tautomers,
    internal_tanimoto_mmap,
    internal_tanimoto_parallel,
    mols_from_smiles,
    mols_to_smiles,
    murcko_scaffolds,
    pick_canonical_tautomers,
    save_morgan_fps_from_smiles,
    score_tautomers,
)
from smiles_fp.helpers import (  # type: ignore[deprecated]
    get_mols,
    get_morgan_fps,
    load_fingerprints,
    load_fingerprints_as_ndarray,
    mol_from_smi,
    save_fingerprints,
    to_matrix,
)
from smiles_fp.search import similarity_search, windowed_bulk_tanimoto

__all__ = [
    "bulk_tanimoto_mmap",
    "bulk_tanimoto_mmap_topk",
    "bulk_tanimoto_parallel",
    "bulk_tanimoto_parallel_topk",
    "canonical_tautomers",
    "clean",
    "enumerate_tautomers",
    "get_mols",
    "get_morgan_fps",
    "internal_tanimoto_mmap",
    "internal_tanimoto_parallel",
    "load_fingerprints",
    "load_fingerprints_as_ndarray",
    "mol_from_smi",
    "mols_from_smiles",
    "mols_to_smiles",
    "murcko_scaffolds",
    "pick_canonical_tautomers",
    "save_fingerprints",
    "save_morgan_fps_from_smiles",
    "score_tautomers",
    "similarity_search",
    "to_matrix",
    "windowed_bulk_tanimoto",
]
