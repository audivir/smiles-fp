"""Helper functions for conversion of RDKit molecules and NumPy arrays."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rdFPGen

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    from _typeshed import StrPath
    from numpy.typing import NDArray
    from rdkit.DataStructs import ExplicitBitVect


HEADER_SIZE = 8
ARRAY_NDIM = 2


def save_fingerprints(
    py_fps: Sequence[ExplicitBitVect] | NDArray[np.uint8],
    filename: StrPath,
) -> None:
    """Saves a sequence of fingerprints or a NDArray to a binary file.

    All fingerprints must have the same length.

    Args:
        py_fps: A sequence of RDKit fingerprints or an uint8 NDArray.
        filename: The path to the output file.

    Raises:
        TypeError: If an array is not 2D with dtype uint8.
        ValueError: If an array has values other than 0 or 1 or no bits per fingerprint.
    """
    import numpy as np

    if not isinstance(py_fps, np.ndarray):
        from smiles_fp._smiles_fp import _save_fingerprints

        _save_fingerprints(py_fps, filename)
        return

    if py_fps.ndim != ARRAY_NDIM or py_fps.dtype != np.uint8:
        raise TypeError("Fingerprint arrays must be 2D with dtype uint8")
    n_fps, n_bits = py_fps.shape
    if n_fps >= 2**32 or n_bits >= 2**32:
        raise ValueError("Too many fingerprints or bits for the file format (max 2^32-1)")
    if n_fps == 0:
        n_bits = 0
    elif n_bits == 0:
        raise ValueError("Fingerprints must have at least one bit")
    elif py_fps.max() > 1:
        raise ValueError("Fingerprints must only contain the values 0 and 1")

    # pads to whole 64-bit blocks, as bits are stored as little-endian blocks.
    padded: NDArray[np.uint8] = np.pad(py_fps, ((0, 0), (0, -n_bits % 64)))
    blocks = np.packbits(padded, axis=1, bitorder="little").view("<u8")
    path = Path(filename)
    try:
        with path.open("wb") as file:
            file.write(struct.pack("<II", n_fps, n_bits))
            blocks.tofile(file)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def load_fingerprints_as_ndarray(
    filename: StrPath,
) -> NDArray[np.uint8]:
    """Loads a sequence of fingerprints as NDArray from a file created with `save_fingerprints`.

    Args:
        filename: The path to the binary fingerprint file.

    Returns:
        A unpacked uint8 NDArray with the fingerprint data.

    Raises:
        ValueError: If the file is truncated or corrupt.
    """
    import numpy as np

    with Path(filename).open("rb") as file:
        header = file.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            raise ValueError("Corrupt or truncated fingerprint file: missing header")
        n_fps, n_bits = struct.unpack("<II", header)
        if n_fps == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        n_blocks = -(-n_bits // 64)
        blocks = np.fromfile(file, dtype="<u8", count=n_fps * n_blocks)
    if blocks.size != n_fps * n_blocks:
        raise ValueError(
            f"Corrupt or truncated fingerprint file: header declares {n_fps} fingerprints of "
            f"{n_bits} bits"
        )
    bits = np.unpackbits(blocks.view(np.uint8).reshape(n_fps, -1), axis=1, bitorder="little")
    return np.ascontiguousarray(bits[:, :n_bits])


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
    from smiles_fp._smiles_fp import mols_from_smiles

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
