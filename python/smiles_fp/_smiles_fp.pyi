"""Type stubs for the compiled Rust module '_smiles_fp'.

This file provides type hints for static analysis tools (e.g., 'mypy',
'pyright') and enables autocompletion in IDEs for the functions defined
in Rust.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from _typeshed import StrPath
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

def save_fingerprints(
    py_fps: Sequence[ExplicitBitVect],
    filename: StrPath,
) -> None:
    """Saves a sequence of fingerprints to a binary file.

    All fingerprints must have the same length.

    Args:
        py_fps: A sequence (list, tuple, etc.) of RDKit fingerprints.
        filename: The path to the output file.
    """

def load_fingerprints(
    filename: StrPath,
) -> list[ExplicitBitVect]:
    """Loads a sequence of fingerprints from a binary file created with `save_fingerprints`.

    Args:
        filename: The path to the binary fingerprint file.

    Returns:
        A list of RDKit ExplicitBitVect objects.
    """

def bulk_tanimoto_parallel(
    py_fps: Sequence[ExplicitBitVect],
    py_fps2: Sequence[ExplicitBitVect],
    n_threads: int = -1,
    agg: Literal["mean", "max", "min", "full"] | None = None,
) -> NDArray[np.float64]:
    """Calculates Tanimoto similarities in parallel from RDKit fingerprints.

    Args:
        py_fps: The first sequence of RDKit fingerprints.
        py_fps2: The second sequence of RDKit fingerprints.
        n_threads: The number of threads to use. Defaults to -1 (auto-detect).
        agg: Aggregation method. Defaults to 'full'.

    Returns:
        A 2D NumPy array with the similarity matrix if agg is None/'full' or
        1D NumPy array of aggregated similarity scores.
    """

def bulk_tanimoto_parallel_topk(
    py_fps: Sequence[ExplicitBitVect],
    py_fps2: Sequence[ExplicitBitVect],
    k: int = 10,
    n_threads: int = -1,
) -> tuple[NDArray[np.uint32], NDArray[np.float64]]:
    """Finds the top-K highest similarity scores for each query in memory.

    Args:
        py_fps: Sequence of fingerprints for the query.
        py_fps2: Sequence of fingerprints for the database to query from.
        k: Number of most-similar molecules to query for.
        n_threads: The number of threads to use. Defaults to -1 (auto-detect).

    Returns:
        A tuple with a 2D NumPy array containing the indices of the similarities and
        a 2D NumPy array containing the similarity scores.
    """

def bulk_tanimoto_mmap(
    path1: StrPath,
    path2: StrPath,
    n_threads: int = -1,
    agg: Literal["mean", "max", "min", "full"] | None = None,
    db_offset: int = 0,
    db_limit: int = 0,
) -> NDArray[np.float64]:
    """Calculates Tanimoto similarities in parallel from binary files using memory-mapping.

    Args:
        path1: Path to the query binary fingerprint file.
        path2: Path to the database binary fingerprint file.
        n_threads: The number of threads to use. Defaults to -1 (auto-detect).
        agg: Aggregation method. Defaults to 'full'.
        db_offset: Starting index in the second sequence to process (for windowing).
        db_limit: Number of items in the second sequence to process (0 means all).

    Returns:
        A 2D NumPy array with the similarity matrix if agg is None/'full' or
        1D NumPy array of aggregated similarity scores.
    """

def bulk_tanimoto_mmap_topk(
    path1: StrPath,
    path2: StrPath,
    k: int = 10,
    n_threads: int = -1,
    db_offset: int = 0,
    db_limit: int = 0,
) -> tuple[NDArray[np.uint32], NDArray[np.float64]]:
    """Finds the top-K highest similarity scores for each query using memory-mapping.

    Args:
        path1: Path to the query binary fingerprint file.
        path2: Path to the database binary fingerprint file.
        k: Number of most-similar molecules to query for. Defaults to 10.
        n_threads: The number of threads to use. Defaults to -1 (auto-detect).
        db_offset: Starting index in the database to process (for windowing).
        db_limit: Number of items in the database to process (0 means all).

    Returns:
        A tuple with a 2D NumPy array containing the indices of the similarities and
        a 2D NumPy array containing the similarity scores.
    """

def internal_tanimoto_parallel(
    py_fps: Sequence[ExplicitBitVect],
    n_threads: int = -1,
    agg: Literal["mean", "max", "min", "full"] | None = None,
) -> NDArray[np.float64]:
    """Calculates internal (pairwise) similarities for a dataset in parallel.

    Args:
        py_fps: The sequence of RDKit fingerprints.
        n_threads: The number of threads to use. Defaults to -1 (auto-detect).
        agg: Aggregation method. Defaults to 'full'.

    Returns:
        If agg is 'full' or None, returns a 1D condensed distance matrix array
        (size: N * (N-1) / 2). Can be converted to a full symmetric 2D matrix
        using scipy.spatial.distance.squareform.
        Otherwise, returns a 1D array of the aggregated similarity scores.
    """

def internal_tanimoto_mmap(
    path: StrPath,
    n_threads: int = -1,
    agg: Literal["mean", "max", "min", "full"] | None = None,
) -> NDArray[np.float64]:
    """Calculates internal (pairwise) similarities from a binary file using memory-mapping.

    Args:
        path: Path to the binary fingerprint file.
        n_threads: The number of threads to use. Defaults to -1 (auto-detect).
        agg: Aggregation method. Defaults to 'full'.

    Returns:
        If agg is 'full' or None, returns a 1D condensed distance matrix array
        (size: N * (N-1) / 2). Can be converted to a full symmetric 2D matrix
        using scipy.spatial.distance.squareform.
        Otherwise, returns a 1D array of the aggregated similarity scores.
    """

def mols_from_smiles(
    smiles: Sequence[str],
    sanitize: bool = True,
    *,
    n_threads: int = -1,
) -> list[Chem.Mol | None]:
    """Parses SMILES to RDKit molecules multi-threaded.

    Args:
        smiles: Sequence of SMILES strings to parse.
        sanitize: Whether to sanitize and remove Hs.
        n_threads: The number of threads to use. Defaults to -1 (auto-detect).

    Returns:
        A list with a molecule per SMILES, or None where parsing failed.
    """

def mols_to_smiles(
    mols: Sequence[Chem.Mol],
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int | Sequence[int | None] | None = None,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    *,
    ignore_atom_map_numbers: bool = False,
    n_threads: int = -1,
) -> list[str]:
    """Writes SMILES for RDKit molecules multi-threaded.

    Holds the GIL, so other Python threads cannot modify the molecules. A GIL-released RDKit
    function running on them in another thread (e.g. `GetFingerprints`) is not blocked, so the
    molecules must not be modified by it during conversion. RDKit logging redirected to Python
    (e.g. `rdBase.LogToPythonLogger`) can deadlock, as a worker thread that logs waits for the GIL.
    `doRandom` is not supported because RDKit's random generator is global and not thread-safe.
    A molecule that appears several times is converted once per distinct `rooted_at_atom`,
    and re-written at each position.

    Args:
        mols: A sequence of RDKit molecules to convert.
        isomeric_smiles: Include stereochemistry and isotope information.
        kekule_smiles: Use the Kekule form (no aromatic bonds).
        rooted_at_atom: If non-negative, start the SMILES at this atom (overrides `canonical`).
            One index for all molecules, or a sequence with one index (or None) per molecule.
        canonical: Canonicalize the output.
        all_bonds_explicit: Write the symbol of every bond.
        all_hs_explicit: Write the H count of every atom.
        ignore_atom_map_numbers: Ignore atom map numbers when canonicalizing.
        n_threads: The number of threads to use. Defaults to -1 (auto-detect).

    Returns:
        A list with each molecule's SMILES.

    Raises:
        TypeError: If an item is not a molecule.
        ValueError: If a `rooted_at_atom` sequence does not have one item per molecule.
        RuntimeError: If RDKit cannot write a molecule (e.g. `rooted_at_atom` out of range).
    """
