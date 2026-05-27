"""Helper functions for SmilesFp package."""

from __future__ import annotations

from multiprocessing import cpu_count
from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rdFPGen
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence

    import numpy as np
    from numpy.typing import NDArray
    from rdkit.DataStructs import ExplicitBitVect


def to_matrix(vector: NDArray[np.float64], n: int, diagonal: float = 0.0) -> NDArray[np.float64]:
    """Convert condensed distance vector to its corresponding matrix.

    Args:
        vector: The condensed 1D distance vector
        n: Number of points
        diagonal: Value to set the diagonal to.

    Returns:
        The 2D matrix with 0.0 on the diagonal.
    """
    import numpy as np

    if vector.ndim != 1:
        raise ValueError("must be 1D array")
    if vector.size != n * (n - 1) // 2:
        raise ValueError("unexpected size")
    if n < 2:  # noqa: PLR2004
        arr = np.empty(shape=(n, n), dtype=np.float64)
    else:
        import scipy

        arr = scipy.spatial.distance.squareform(vector, "tomatrix")

    np.fill_diagonal(arr, diagonal)

    return arr


def mol_from_smi(smi: str) -> Chem.Mol:
    """Convert a SMILES string to a RDKit molecule.

    Args:
        smi: SMILES to convert.

    Returns:
        A RDKit Molecule

    Raises:
        ValueError: If invalid SMILES are provide
    """
    if mol := Chem.MolFromSmiles(smi):
        return mol
    raise ValueError(f"Could not convert {smi} to a molecule.")  # pragma: no cover


def get_mols(smis: Iterable[str], n_jobs: int = -1, verbose: bool = False) -> list[Chem.Mol]:
    """Convert a batch of SMILES strings to RDKit molecules.

    Args:
        smis: Iterable of SMILES strings to convert.
        n_jobs: Number of jobs. Defaults to -1 (auto-detect).
        verbose: If a progress bar should be printed.

    Returns:
        A list of RDKit molecules

    Raises:
        ValueError: If invalid SMILES are provide
    """
    import joblib

    n_jobs = n_jobs if n_jobs > 0 else cpu_count()

    with joblib.Parallel(
        n_jobs=n_jobs,
        return_as="generator",
    ) as p:
        mol_gen: Generator[Chem.Mol] = p(joblib.delayed(mol_from_smi)(smi) for smi in smis)

        if verbose:  # pragma: no cover
            mol_gen = tqdm(mol_gen)
        return list(mol_gen)


def get_morgan_fps(
    mols: Sequence[ExplicitBitVect], radius: int = 2, n_bits: int = 2048, n_threads: int = -1
) -> tuple[ExplicitBitVect, ...]:
    """Convert RDKit molecules to Morgan fingerprints.

    Args:
        mols: A sequence of RDKit molecules.
        radius: Radius of the fingerprint.
        n_bits: Size of the fingerprint.
        n_threads: Number of threads. Defaults to -1 (auto-detect).

    Returns:
        A tuple of Morgan fingerprints.
    """
    n_threads = n_threads if n_threads > 0 else cpu_count()
    morgan_gen = rdFPGen.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps: tuple[ExplicitBitVect, ...] = morgan_gen.GetFingerprints(mols, numThreads=n_threads)
    return fps
