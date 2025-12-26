"""Type stubs for the compiled C++ module 'smiles_fp'.

This file provides type hints for static analysis tools (e.g., mypy, pyright)
and enables autocompletion in IDEs for the functions defined in C++.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

def save_fingerprints(py_fps: Sequence[ExplicitBitVect], filename: str) -> None:
    """Saves a sequence of fingerprints to a binary file.

    All fingerprints must be of the same length.

    Args:
        py_fps: A sequence (list, tuple, etc.) of RDKit ExplicitBitVect objects.
        filename: The path to the output file.
    """

def load_fingerprints(filename: str) -> list[ExplicitBitVect]:
    """Loads a sequence of fingerprints from a binary file.

    Args:
        filename: The path to the binary fingerprint file.

    Returns:
        A list of RDKit ExplicitBitVect objects.
    """

def bulk_tanimoto_parallel(
    py_fps: Sequence[ExplicitBitVect],
    py_fps2: Sequence[ExplicitBitVect],
    num_threads: int = -1,
) -> npt.NDArray[np.float64]:
    """Calculates Tanimoto similarities in parallel from RDKit fp objects.

    Args:
        py_fps: The first sequence of RDKit ExplicitBitVect objects.
        py_fps2: The second sequence of RDKit ExplicitBitVect objects.
        num_threads: The number of threads to use. Defaults to -1 (auto-detect).

    Returns:
        A 1D NumPy array of double-precision similarity scores, flattened row-major.
    """

def bulk_tanimoto_mmap(
    filename1: str, filename2: str, num_threads: int = -1
) -> npt.NDArray[np.float64]:
    """Calculates Tanimoto similarities between two binary fingerprint files.

    Uses memory-mapping. Faster and more memory-efficient for large datasets than
    using the ExplicitBitVect objects directly.

    Args:
        filename1: Path to the first binary fingerprint file.
        filename2: Path to the second binary fingerprint file.
        num_threads: The number of threads to use. Defaults to -1 (auto-detect).

    Returns:
        A 1D NumPy array of double-precision similarity scores, flattened row-major.
    """
