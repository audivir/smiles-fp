"""Helper functions for conversion of RDKit molecules and NumPy arrays."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, get_args, overload

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rdFPGen
from tqdm.auto import tqdm
from typing_extensions import deprecated

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    from _typeshed import StrPath
    from numpy.typing import NDArray
    from rdkit.DataStructs import ExplicitBitVect

    from smiles_fp._smiles_fp import Fingerprints

FpOutput: TypeAlias = Literal["bitvect", "unpacked", "packed"]
"""Format of returned fingerprints: RDKit bit vectors, a uint8 NDArray with one bit per column,
or a uint64 NDArray of packed 64-bit blocks in little bit order."""

OUTPUTS: tuple[FpOutput, ...] = get_args(FpOutput)
HEADER_SIZE = 8
ARRAY_NDIM = 2


def pack_bits(bits: NDArray[np.uint8]) -> NDArray[np.uint64]:
    """Packs a 2D array of 0 and 1 into little-endian 64-bit blocks per row.

    Args:
        bits: A uint8 NDArray with one bit per column.

    Returns:
        The blocks, with the last block of each row padded with zeros.

    Raises:
        ValueError: If the array has values other than 0 or 1.
    """
    import numpy as np

    if bits.max() > 1:
        raise ValueError("Fingerprints must only contain the values 0 and 1")
    # pads to whole 64-bit blocks, as bits are stored as little-endian blocks.
    padded: NDArray[np.uint8] = np.pad(bits, ((0, 0), (0, -bits.shape[1] % 64)))
    return np.packbits(padded, axis=1, bitorder="little").view("<u8")


def check_blocks(blocks: NDArray[np.uint64], n_bits: int) -> NDArray[np.uint64]:
    """Checks packed blocks against the number of bits and converts them to little-endian.

    Args:
        blocks: A uint64 NDArray with the blocks of each fingerprint per row.
        n_bits: The number of bits per fingerprint.

    Returns:
        The contiguous little-endian blocks.

    Raises:
        ValueError: If the number of blocks does not fit `n_bits` or the padding is not zero.
    """
    import numpy as np

    n_blocks = -(-n_bits // 64)
    if blocks.shape[1] != n_blocks:
        raise ValueError(
            f"{n_bits} bits need {n_blocks} blocks, but the array has {blocks.shape[1]}"
        )
    little = np.ascontiguousarray(blocks, dtype="<u8")
    padding = -n_bits % 64
    if padding and (little[:, -1] >> np.uint64(64 - padding)).any():
        raise ValueError("The padding bits beyond n_bits must be zero")
    return little


def save_fingerprints(
    py_fps: Fingerprints,
    filename: StrPath,
    *,
    n_bits: int | None = None,
) -> None:
    """Saves a sequence of fingerprints or a NDArray to a binary file.

    All fingerprints must have the same length.

    Args:
        py_fps: A sequence of RDKit fingerprints or an uint8 or uint64 NDArray.
        filename: The path to the output file.
        n_bits: The number of bits per fingerprint of a packed uint64 NDArray, whose last block
            may be padded. Required for those arrays and not allowed for other inputs.

    Raises:
        TypeError: If an array is not 2D with dtype uint8 or uint64.
        ValueError: If an array has values other than 0 or 1 or no bits per fingerprint, or if
            `n_bits` is missing, unexpected, does not match the blocks, or the padding is not zero.
    """
    import numpy as np

    is_packed = isinstance(py_fps, np.ndarray) and py_fps.dtype.type is np.uint64
    if is_packed and n_bits is None:
        raise ValueError("n_bits is required for packed uint64 arrays")
    if not is_packed and n_bits is not None:
        raise ValueError("n_bits is only used for packed uint64 arrays")

    if not isinstance(py_fps, np.ndarray):
        from smiles_fp._smiles_fp import _save_fingerprints

        _save_fingerprints(py_fps, filename)
        return

    if py_fps.ndim != ARRAY_NDIM or py_fps.dtype.type not in {np.uint8, np.uint64}:
        raise TypeError("Fingerprint arrays must be 2D with dtype uint8 or uint64")
    n_fps, width = py_fps.shape
    bits_per_fp = width if n_bits is None else n_bits
    if n_fps >= 2**32 or bits_per_fp >= 2**32:
        raise ValueError("Too many fingerprints or bits for the file format (max 2^32-1)")

    blocks: NDArray[np.uint64]
    if n_fps == 0:
        bits_per_fp = 0
        blocks = np.zeros((0, 0), dtype="<u8")
    elif bits_per_fp < 1:
        raise ValueError("Fingerprints must have at least one bit")
    elif is_packed:
        blocks = check_blocks(np.asarray(py_fps, dtype=np.uint64), bits_per_fp)
    else:
        blocks = pack_bits(np.asarray(py_fps, dtype=np.uint8))

    path = Path(filename)
    try:
        with path.open("wb") as file:
            file.write(struct.pack("<II", n_fps, bits_per_fp))
            blocks.tofile(file)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def unpack_blocks(blocks: NDArray[np.uint64], n_bits: int) -> NDArray[np.uint8]:
    """Unpacks little-endian 64-bit blocks into a uint8 NDArray with one bit per column.

    Args:
        blocks: A 2D uint64 NDArray with the blocks of each fingerprint per row.
        n_bits: The number of bits per fingerprint.

    Returns:
        A contiguous uint8 NDArray with `n_bits` columns.
    """
    import numpy as np

    bits = np.unpackbits(blocks.view(np.uint8), axis=1, bitorder="little")
    return np.ascontiguousarray(bits[:, :n_bits])


def check_output(output: str) -> None:
    """Raises a ValueError if `output` is not one of the fingerprint formats."""
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {', '.join(OUTPUTS)}, not {output!r}")


@overload
def load_fingerprints(
    filename: StrPath, *, output: Literal["bitvect"] = "bitvect"
) -> list[ExplicitBitVect]: ...
@overload
def load_fingerprints(filename: StrPath, *, output: Literal["unpacked"]) -> NDArray[np.uint8]: ...
@overload
def load_fingerprints(filename: StrPath, *, output: Literal["packed"]) -> NDArray[np.uint64]: ...
def load_fingerprints(
    filename: StrPath, *, output: FpOutput = "bitvect"
) -> list[ExplicitBitVect] | NDArray[np.uint8] | NDArray[np.uint64]:
    """Loads a sequence of fingerprints from a binary file created with `save_fingerprints`.

    Args:
        filename: The path to the binary fingerprint file.
        output: The format of the fingerprints: a list of RDKit bit vectors, an unpacked uint8
            NDArray with one bit per column, or a packed uint64 NDArray of 64-bit blocks, whose
            last block may be padded with zeros.

    Returns:
        A list of RDKit ExplicitBitVect objects or a NDArray.

    Raises:
        ValueError: If `output` is unknown or the file is truncated or corrupt.
    """
    import numpy as np

    check_output(output)
    if output == "bitvect":
        from smiles_fp._smiles_fp import _load_fingerprints

        return _load_fingerprints(filename)

    with Path(filename).open("rb") as file:
        header = file.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            raise ValueError("Corrupt or truncated fingerprint file: missing header")
        n_fps, n_bits = struct.unpack("<II", header)
        if n_fps == 0:
            return np.zeros((0, 0), dtype=np.uint8 if output == "unpacked" else np.uint64)
        n_blocks = -(-n_bits // 64)
        blocks = np.fromfile(file, dtype="<u8", count=n_fps * n_blocks)
    if blocks.size != n_fps * n_blocks:
        raise ValueError(
            f"Corrupt or truncated fingerprint file: header declares {n_fps} fingerprints of "
            f"{n_bits} bits"
        )
    matrix = blocks.reshape(n_fps, n_blocks)
    return unpack_blocks(matrix, n_bits) if output == "unpacked" else matrix


@deprecated("Use `load_fingerprints(filename, output='unpacked')` instead.")
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
    return load_fingerprints(filename, output="unpacked")


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
        n_jobs: Number of threads, where -1 detects the count automatically.
        verbose: Whether to show a progress bar with ETA.
        sanitize: Whether to sanitize and remove Hs.

    Returns:
        A list of RDKit molecules

    Raises:
        ValueError: If any invalid SMILES are provided
    """
    from smiles_fp._smiles_fp import mols_from_smiles

    if not isinstance(smis, Sequence):
        smis = list(smis)
    mols = mols_from_smiles(smis, sanitize=sanitize, n_threads=n_jobs, verbose=verbose)
    failed = [smi for smi, mol in zip(smis, mols, strict=True) if mol is None]
    if failed:
        raise ValueError(
            f"Could not convert {len(failed)} SMILES to molecules: {', '.join(failed)}"
        )
    return mols  # type: ignore[return-value]


@overload
def get_morgan_fps(
    mols: Sequence[Chem.Mol],
    radius: int = 2,
    n_bits: int = 2048,
    n_threads: int = -1,
    *,
    verbose: bool = False,
    output: Literal["bitvect"] = "bitvect",
) -> tuple[ExplicitBitVect, ...]: ...
@overload
def get_morgan_fps(
    mols: Sequence[Chem.Mol],
    radius: int = 2,
    n_bits: int = 2048,
    n_threads: int = -1,
    *,
    verbose: bool = False,
    output: Literal["unpacked"],
) -> NDArray[np.uint8]: ...
@overload
def get_morgan_fps(
    mols: Sequence[Chem.Mol],
    radius: int = 2,
    n_bits: int = 2048,
    n_threads: int = -1,
    *,
    verbose: bool = False,
    output: Literal["packed"],
) -> NDArray[np.uint64]: ...
def get_morgan_fps(  # noqa: PLR0913
    mols: Sequence[Chem.Mol],
    radius: int = 2,
    n_bits: int = 2048,
    n_threads: int = -1,
    *,
    verbose: bool = False,
    output: FpOutput = "bitvect",
) -> tuple[ExplicitBitVect, ...] | NDArray[np.uint8] | NDArray[np.uint64]:
    """Converts RDKit molecules to Morgan fingerprints.

    Args:
        mols: A sequence of RDKit molecules.
        radius: Radius of the fingerprint.
        n_bits: Size of the fingerprint.
        n_threads: Number of threads, where -1 detects the count automatically.
        verbose: Whether to show a progress bar with ETA. Fingerprints are then computed in
            chunks, which is slightly slower.
        output: The format of the fingerprints: a tuple of RDKit bit vectors, an unpacked uint8
            NDArray with one bit per column, or a packed uint64 NDArray of 64-bit blocks, whose
            last block may be padded with zeros.

    Returns:
        A tuple of Morgan fingerprints or a NDArray.

    Raises:
        ValueError: If `output` is unknown.
    """
    import numpy as np

    check_output(output)
    n_threads = n_threads if n_threads > 0 else cpu_count()
    morgan_gen = rdFPGen.GetMorganGenerator(radius=radius, fpSize=n_bits)
    # RDKit has no progress hook, so a verbose run advances the bar once per chunk of molecules.
    step = max(1000, n_threads * 64) if verbose else max(len(mols), 1)
    parts: list[tuple[ExplicitBitVect, ...]] = []
    with tqdm(
        total=len(mols),
        desc="Fingerprinting molecules",
        unit="mol",
        disable=not (verbose and mols),
    ) as bar:
        for start in range(0, len(mols), step):
            chunk = mols[start : start + step]
            parts.append(morgan_gen.GetFingerprints(chunk, numThreads=n_threads))
            bar.update(len(chunk))
    fps = parts[0] if len(parts) == 1 else tuple(fp for part in parts for fp in part)
    if output == "bitvect":
        return fps

    from smiles_fp._smiles_fp import _fingerprints_to_blocks

    blocks = (
        _fingerprints_to_blocks(fps) if fps else np.zeros((0, -(-n_bits // 64)), dtype=np.uint64)
    )
    return unpack_blocks(blocks, n_bits) if output == "unpacked" else blocks
