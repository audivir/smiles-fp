"""Type stubs for the compiled Rust module '_smiles_fp_rs'.

This file provides type hints for static analysis tools (e.g., 'mypy', 'pyright')
and enables autocompletion in IDEs for the functions defined in Rust.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from _typeshed import StrPath
from numpy.typing import NDArray
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

def save_fingerprints(
    py_fps: Sequence[ExplicitBitVect],
    filename: StrPath,
) -> None:
    """Save a sequence of fingerprints to a binary file.

    All fingerprints must have the same length.

    Args:
        py_fps: A sequence (list, tuple, etc.) of RDKit fingerprints.
        filename: The path to the output file.
    """

def load_fingerprints(
    filename: StrPath,
) -> list[ExplicitBitVect]:
    """Load a sequence of fingerprints from a binary file created with `save_fingerprints`.

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
    """Calculate Tanimoto similarities in parallel from RDKit fingerprints.

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
    """Find the top-K highest similarity scores for each query in memory.

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
    """Calculate Tanimoto similarities in parallel from binary files using memory-mapping.

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
    """Find the top-K highest similarity scores for each query using memory-mapping.

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
    """Calculate internal (pairwise) similarities for a dataset in parallel.

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
    """Calculate internal (pairwise) similarities from a binary file using memory-mapping.

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
