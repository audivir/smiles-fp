"""Helper functions for conversion of RDKit molecules and NumPy arrays."""

from __future__ import annotations

from collections.abc import Sequence
from multiprocessing import cpu_count
from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rdFPGen

from smiles_fp._smiles_fp import mols_from_smiles

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    from numpy.typing import NDArray
    from rdkit.DataStructs import ExplicitBitVect


def to_matrix(vector: NDArray[np.float64], n: int, diagonal: float = 0.0) -> NDArray[np.float64]:
    """Converts a condensed distance vector to its corresponding matrix.

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
    """Converts a SMILES to a RDKit molecule, fails on an invalid SMILES.

    Args:
        smi: SMILES to convert.

    Returns:
        A RDKit Molecule

    Raises:
        ValueError: If an invalid SMILES is provided
    """
    mol: Chem.Mol | None = Chem.MolFromSmiles(smi)
    if mol:
        return mol
    raise ValueError(f"Could not convert {smi} to a molecule.")


def get_mols(
    smis: Iterable[str], n_jobs: int = -1, verbose: bool = False, *, sanitize: bool = True
) -> list[Chem.Mol]:
    """Converts a batch of SMILES to RDKit molecules, fails on any invalid SMILES.

    Args:
        smis: Iterable of SMILES strings to convert.
        n_jobs: Number of threads. Defaults to -1 (auto-detect).
        verbose: Ignored.
        sanitize: Whether to sanitize and remove Hs.

    Returns:
        A list of RDKit molecules

    Raises:
        ValueError: If any invalid SMILES are provided
    """
    del verbose  # unused
    if not isinstance(smis, Sequence):
        smis = list(smis)
    mols = mols_from_smiles(smis, sanitize=sanitize, n_threads=n_jobs)
    failed = [smi for smi, mol in zip(smis, mols, strict=True) if mol is None]
    if failed:
        raise ValueError(
            f"Could not convert {len(failed)} SMILES to molecules: {', '.join(failed)}"
        )
    return mols  # type: ignore[return-value]


def get_morgan_fps(
    mols: Sequence[Chem.Mol], radius: int = 2, n_bits: int = 2048, n_threads: int = -1
) -> tuple[ExplicitBitVect, ...]:
    """Converts RDKit molecules to Morgan fingerprints.

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
