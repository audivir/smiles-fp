"""Pure-Python drop-in for the `smiles_fp` package: copy this file next to your code.

`import smiles_fp` then works without Rust, C++ or a wheel. Only `rdkit`, `numpy`, `scipy` and
`tqdm` are needed. The functions, signatures, results and errors equal those of the compiled
package, but everything runs in Python on one thread: `n_threads` is accepted and ignored, as RDKit
calls from Python hold the GIL. Use the compiled package for speed.

The submodules `smiles_fp.helpers`, `smiles_fp.search` and `smiles_fp._smiles_fp` exist as aliases.
"""

from __future__ import annotations

import struct
import sys
import types
from collections.abc import Collection, Iterable, Iterator, Sequence
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, get_args

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rdFPGen
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from tqdm.auto import tqdm
from typing_extensions import deprecated

if TYPE_CHECKING:
    from _typeshed import StrPath
    from numpy.typing import NDArray

T = TypeVar("T")

FpOutput: TypeAlias = Literal["bitvect", "unpacked", "packed"]
Agg: TypeAlias = Literal["mean", "max", "min", "full"]
TautomerStatus: TypeAlias = Literal[
    "completed", "max_tautomers_reached", "max_transforms_reached", "canceled"
]

OUTPUTS: tuple[FpOutput, ...] = get_args(FpOutput)
STATUS: tuple[TautomerStatus, ...] = get_args(TautomerStatus)
HEADER_SIZE = 8
ARRAY_NDIM = 2
MAX_FILE_COUNT = 2**32
# upper bound of the elements of a temporary array, which bounds the memory of a similarity chunk.
CHUNK_ELEMENTS = 4_000_000
# RDKit has no progress hook, so a verbose fingerprint run advances once per chunk of molecules.
MIN_FP_STEP = 1000

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


def _bar(
    items: Iterable[T], total: int, desc: str, verbose: bool, unit: str = "mol"
) -> Iterator[T]:
    """Iterates over `items`, with a progress bar if `verbose`."""
    if not verbose or total == 0:
        yield from items
        return
    yield from tqdm(items, total=total, desc=desc, unit=unit)


def _check_output(output: str) -> None:
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {', '.join(OUTPUTS)}, not {output!r}")


def _want_mol(output: str) -> bool:
    """Reads the `output` argument, which is True for molecules."""
    if output not in {"smiles", "mol"}:
        raise ValueError(f"Unknown output '{output}', expected 'smiles' or 'mol'")
    return output == "mol"


# molecules


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


def mols_from_smiles(
    smiles: Sequence[str],
    sanitize: bool = True,
    *,
    n_threads: int = -1,  # noqa: ARG001
    verbose: bool = False,
) -> list[Chem.Mol | None]:
    """Parses SMILES to RDKit molecules.

    Args:
        smiles: Sequence of SMILES strings to parse.
        sanitize: Whether to sanitize and remove Hs.
        n_threads: Ignored.
        verbose: Whether to show a progress bar with ETA while processing.

    Returns:
        A list with a molecule per SMILES, or None where parsing failed.

    Raises:
        TypeError: If `smiles` is a string or an item is not a string.
    """
    if isinstance(smiles, str) or not all(isinstance(s, str) for s in smiles):
        raise TypeError("smiles must be a sequence of strings")
    parsed = (Chem.MolFromSmiles(s, sanitize=sanitize) for s in smiles)
    return list(_bar(parsed, len(smiles), "Parsing SMILES", verbose))


def get_mols(
    smis: Iterable[str], n_jobs: int = -1, verbose: bool = False, *, sanitize: bool = True
) -> list[Chem.Mol]:
    """Converts a batch of SMILES to RDKit molecules, fails on any invalid SMILES.

    Args:
        smis: Iterable of SMILES strings to convert.
        n_jobs: Ignored.
        verbose: Whether to show a progress bar with ETA.
        sanitize: Whether to sanitize and remove Hs.

    Returns:
        A list of RDKit molecules

    Raises:
        ValueError: If any invalid SMILES are provided
    """
    if not isinstance(smis, Sequence):
        smis = list(smis)
    mols = mols_from_smiles(smis, sanitize=sanitize, n_threads=n_jobs, verbose=verbose)
    failed = [smi for smi, mol in zip(smis, mols, strict=True) if mol is None]
    if failed:
        raise ValueError(
            f"Could not convert {len(failed)} SMILES to molecules: {', '.join(failed)}"
        )
    return mols  # type: ignore[return-value]


def _roots(arg: int | Sequence[int | None] | None, n: int) -> list[int]:
    """Reads `rooted_at_atom`, which is -1 where no root is set."""
    if arg is None:
        return [-1] * n
    if isinstance(arg, int):
        return [arg] * n
    if not isinstance(arg, Sequence):
        raise TypeError("rooted_at_atom must be an int, None or a sequence of int | None")
    if not all(r is None or isinstance(r, int) for r in arg):
        raise TypeError("rooted_at_atom must be an int, None or a sequence of int | None")
    if len(arg) != n:
        raise ValueError(f"rooted_at_atom has {len(arg)} items for {n} molecules")
    return [-1 if r is None else r for r in arg]


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
    n_threads: int = -1,  # noqa: ARG001
    verbose: bool = False,
) -> list[str]:
    """Writes SMILES for RDKit molecules.

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
        n_threads: Ignored.
        verbose: Whether to show a progress bar with ETA while processing.

    Returns:
        A list with each molecule's SMILES.

    Raises:
        TypeError: If an item is not a molecule.
        ValueError: If a `rooted_at_atom` sequence does not have one item per molecule.
        RuntimeError: If RDKit cannot write a molecule (e.g. `rooted_at_atom` out of range).
    """
    mols = list(mols)
    if not all(isinstance(m, Chem.Mol) for m in mols):
        raise TypeError("mols must be a sequence of molecules")
    roots = _roots(rooted_at_atom, len(mols))
    smiles: list[str] = []
    for i, (mol, root) in enumerate(
        _bar(zip(mols, roots, strict=True), len(mols), "Writing SMILES", verbose)
    ):
        try:
            smiles.append(
                Chem.MolToSmiles(
                    mol,
                    isomericSmiles=isomeric_smiles,
                    kekuleSmiles=kekule_smiles,
                    rootedAtAtom=root,
                    canonical=canonical,
                    allBondsExplicit=all_bonds_explicit,
                    allHsExplicit=all_hs_explicit,
                    ignoreAtomMapNumbers=ignore_atom_map_numbers,
                )
            )
        except Exception as e:  # noqa: PERF203
            raise RuntimeError(f"MolToSmiles failed for molecule {i}: {e}") from e
    return smiles


# cleaning, scaffolds and tautomers

# the inputs of a batch: SMILES (None for invalid ones) or molecules.
Inputs: TypeAlias = "list[str | None] | list[Chem.Mol]"


def _is_smiles_input(inputs: Sequence[Any]) -> bool | None:
    """Tells whether the first item makes the batch SMILES; None for an empty batch."""
    if len(inputs) == 0:
        return None
    return inputs[0] is None or isinstance(inputs[0], str)


def _read_inputs(inputs: Sequence[Any]) -> tuple[bool, Inputs] | None:
    """Reads SMILES, where None and empty strings are invalid later, or molecules.

    Returns None for no items. Molecules are copied, so the inputs stay unchanged.
    """
    is_smiles = _is_smiles_input(inputs)
    if is_smiles is None:
        return None
    if is_smiles:
        if not all(s is None or isinstance(s, str) for s in inputs):
            raise TypeError("inputs must be all SMILES or all molecules")
        return True, list(inputs)
    if not all(isinstance(m, Chem.Mol) for m in inputs):
        raise TypeError("inputs must be all SMILES or all molecules")
    return False, [Chem.Mol(m) for m in inputs]


def _parse(smiles: str | None) -> Chem.Mol | None:
    """Parses SMILES, where None, the empty string and an empty molecule are invalid."""
    if not smiles:
        return None
    mol: Chem.Mol | None = Chem.MolFromSmiles(smiles)
    return mol if mol is not None and mol.GetNumAtoms() > 0 else None


def _reparses(smiles: str) -> bool:
    return _parse(smiles) is not None


def _finish(mol: Chem.Mol | None, want_mol: bool, verify: bool) -> str | Chem.Mol | None:
    """Returns a new molecule as SMILES, or keeps it as molecule.

    With `verify`, a result whose SMILES do not parse to a molecule with atoms is invalid.
    """
    if mol is None:
        return None
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:  # noqa: BLE001
        return None
    if verify and not _reparses(smiles):
        return None
    return mol if want_mol else smiles


def _apply(
    inputs: Sequence[Any],
    output: str,
    desc: str,
    verbose: bool,
    op: Any,
    *,
    verify: bool,
) -> list[Any]:
    """Applies `op`, which returns a molecule or None, to every input."""
    want_mol = _want_mol(output)
    read = _read_inputs(inputs)
    if read is None:
        return []
    is_smiles, items = read
    results: list[Any] = []
    for item in _bar(items, len(items), desc, verbose):
        mol = _parse(item) if is_smiles else item
        if mol is None:
            results.append(None)
            continue
        try:
            results.append(_finish(op(mol), want_mol, verify))
        except Exception:  # noqa: BLE001
            results.append(None)
    return results


_SALTS: SaltRemover | None = None


def _salt_remover() -> SaltRemover:
    global _SALTS  # noqa: PLW0603
    if _SALTS is None:
        _SALTS = SaltRemover()  # type: ignore[no-untyped-call]
    return _SALTS


def _strip(mol: Chem.Mol, dont_remove_everything: bool) -> Chem.Mol | None:
    try:
        stripped: Chem.Mol = _salt_remover().StripMol(  # type: ignore[no-untyped-call]
            Chem.Mol(mol), dontRemoveEverything=dont_remove_everything
        )
    except ValueError:
        return None
    return stripped


def _strip_salts(
    mols: Sequence[Chem.Mol],
    dont_remove_everything: bool = False,
    *,
    n_threads: int = -1,  # noqa: ARG001
    verbose: bool = False,
) -> list[Chem.Mol | None]:
    """Removes the default salts from copies of molecules like `SaltRemover().StripMol`.

    Args:
        mols: A sequence of RDKit molecules, which stay unchanged.
        dont_remove_everything: Whether to keep the last salt if all fragments would be removed.
        n_threads: Ignored.
        verbose: Whether to show a progress bar with ETA while processing.

    Returns:
        A list with a stripped molecule per input, or None where sanitizing it failed.

    Raises:
        TypeError: If an item is not a molecule.
    """
    mols = list(mols)
    if not all(isinstance(m, Chem.Mol) for m in mols):
        raise TypeError("mols must be a sequence of molecules")
    return [
        _strip(m, dont_remove_everything) for m in _bar(mols, len(mols), "Stripping salts", verbose)
    ]


def _clean_mol(  # noqa: C901
    mol: Chem.Mol,
    *,
    largest_fragment: bool,
    no_salts: bool,
    no_isotopes: bool,
    no_stereo: bool,
    allowed_elements: Collection[int] | None,
    neutralize: bool,
) -> Chem.Mol | None:
    """Applies the selected steps to a molecule, which is None if it is invalid afterwards."""
    if mol.GetNumAtoms() == 0:
        return None
    try:
        if largest_fragment:
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = Chem.RemoveAllHs(mol)
        if mol.GetNumAtoms() == 0:
            return None
        if no_salts:
            stripped = _strip(mol, False)
            if stripped is None or stripped.GetNumAtoms() == 0:
                return None
            mol = stripped
        if neutralize:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        if no_isotopes:
            for atom in mol.GetAtoms():  # type: ignore[no-untyped-call,unused-ignore]
                atom.SetIsotope(0)
        if no_stereo:
            Chem.RemoveStereochemistry(mol)
        if allowed_elements and any(
            atom.GetAtomicNum() not in allowed_elements
            for atom in mol.GetAtoms()  # type: ignore[no-untyped-call,unused-ignore]
        ):
            return None
    except Exception:  # noqa: BLE001
        return None
    return mol if mol.GetNumAtoms() > 0 else None


def _write_checked(mol: Chem.Mol, isomeric: bool, want_mol: bool, verify: bool) -> Any:
    """Writes the SMILES of a cleaned molecule and, with `verify`, checks that it parses again.

    With `want_mol` the molecule is the one parsed from the written SMILES.
    """
    out = Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    if not want_mol and not verify:
        return out or None
    # the written SMILES may not parse, e.g. F[P-](F)(F)(F)(F)F fails with uncharging.
    check = _parse(out)
    if check is None:
        return None
    return check if want_mol else out


def clean(
    inputs: Sequence[Any],
    output: str | None = None,
    *,
    largest_fragment: bool = False,
    no_salts: bool = False,
    no_isotopes: bool = False,
    no_stereo: bool = False,
    isomeric: bool = True,
    allowed_elements: Collection[int] | None = None,
    neutralize: bool = False,
    verify: bool = True,
    inplace: bool = False,
    n_threads: int = -1,  # noqa: ARG001
    verbose: bool = False,
) -> list[Any]:
    """Cleans SMILES or molecules and returns canonical SMILES or molecules.

    Parses SMILES, removes explicit hydrogens, applies the selected steps, writes the SMILES, and
    parses it again as a check. Molecules are copied first and stay unchanged, and with
    `output="mol"` the cleaned copies are returned without parsing.

    Args:
        inputs: SMILES strings, where None and empty strings are invalid, or molecules.
        output: Whether to return canonical SMILES or molecules. For SMILES inputs the molecules
            are parsed from the written SMILES.
        largest_fragment: Whether to keep only the largest fragment, like `LargestFragmentChooser`.
            Runs before hydrogens and salts are removed.
        no_salts: Whether to remove salts with the default patterns of `SaltRemover`.
        no_isotopes: Whether to remove isotopes. Implied if `isomeric` is False.
        no_stereo: Whether to remove stereochemistry. Implied if `isomeric` is False.
        isomeric: Whether to keep isomeric information in the SMILES.
        allowed_elements: Atomic numbers. If set, molecules with other elements are invalid.
        neutralize: Whether to neutralize charges. Groups of neutral species, like nitro, stay.
        verify: Whether to parse the written SMILES again as a check.
        inplace: Whether to clean molecules in place instead of copying them. An invalid molecule
            stays unchanged.
        n_threads: Ignored.
        verbose: Whether to show a progress bar with ETA while processing.

    Returns:
        With `inplace`, whether each molecule is valid. Otherwise a list with one result per
        input, or None where it is invalid, nothing is left after cleaning, or RDKit fails.

    Raises:
        TypeError: If the items are neither all SMILES nor all molecules, or `inplace` is set for
            SMILES.
        ValueError: If `output` is unknown or not "mol" with `inplace`.
    """
    want_mol = _want_mol(output or ("mol" if inplace else "smiles"))
    if inplace and not want_mol:
        raise ValueError("inplace requires output 'mol'")
    if inplace and _is_smiles_input(inputs):
        raise TypeError("inplace requires molecules")
    read = _read_inputs(inputs) if not inplace else _read_in_place(inputs)
    if read is None:
        return []
    is_smiles, items = read
    # SMILES leave isotopes and stereochemistry to the writer, molecules that are not written
    # lose them here.
    steps = {
        "largest_fragment": largest_fragment,
        "no_salts": no_salts,
        "no_isotopes": no_isotopes if is_smiles and isomeric else (no_isotopes or not isomeric),
        "no_stereo": no_stereo if is_smiles and isomeric else (no_stereo or not isomeric),
        "allowed_elements": allowed_elements,
        "neutralize": neutralize,
    }
    if is_smiles and not isomeric:
        steps["no_isotopes"] = steps["no_stereo"] = False
    results: list[Any] = []
    for item in _bar(items, len(items), "Cleaning", verbose):
        mol = _parse(item) if is_smiles else item
        cleaned = None if mol is None else _clean_mol(mol, **steps)  # type: ignore[arg-type]
        if inplace:
            results.append(_replace(item, cleaned))  # type: ignore[arg-type]
        elif cleaned is None:
            results.append(None)
        elif not is_smiles and want_mol:
            results.append(cleaned)
        else:
            results.append(_safe_write(cleaned, isomeric, want_mol, verify))
    return results


def _read_in_place(inputs: Sequence[Any]) -> tuple[bool, Inputs] | None:
    """Reads molecules without copying them."""
    if len(inputs) == 0:
        return None
    if not all(isinstance(m, Chem.Mol) for m in inputs):
        raise TypeError("inputs must be all SMILES or all molecules")
    return False, list(inputs)


def _replace(mol: Chem.Mol, cleaned: Chem.Mol | None) -> bool:
    """Overwrites a molecule with its cleaned version, if there is one."""
    if cleaned is None:
        return False
    mol.__init__(cleaned)  # type: ignore[misc]
    return True


def _safe_write(mol: Chem.Mol, isomeric: bool, want_mol: bool, verify: bool) -> Any:
    try:
        return _write_checked(mol, isomeric, want_mol, verify)
    except Exception:  # noqa: BLE001
        return None


def _murcko(mol: Chem.Mol, generic: bool, include_chirality: bool) -> Chem.Mol:
    scaffold: Chem.Mol = MurckoScaffold.GetScaffoldForMol(mol)  # type: ignore[no-untyped-call]
    if generic:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)  # type: ignore[no-untyped-call]
    if not include_chirality:
        for atom in scaffold.GetAtoms():  # type: ignore[no-untyped-call,unused-ignore]
            atom.SetIsotope(0)
        Chem.RemoveStereochemistry(scaffold)
    return scaffold


def murcko_scaffolds(
    inputs: Sequence[Any],
    output: str = "smiles",
    *,
    generic: bool = False,
    include_chirality: bool = True,
    n_threads: int = -1,  # noqa: ARG001
    verbose: bool = False,
) -> list[Any]:
    """Computes Murcko scaffolds like `MurckoScaffold.GetScaffoldForMol`.

    Molecules are copied first and stay unchanged.

    Args:
        inputs: SMILES strings, where None and empty strings are invalid, or molecules.
        output: Whether to return canonical SMILES or scaffold molecules.
        generic: Whether to make the scaffold generic like `MakeScaffoldGeneric`, where all atoms
            become carbon and all bonds single.
        include_chirality: Whether to keep stereochemistry and isotopes. Without them, the SMILES
            equal those of `MurckoScaffoldSmiles` with its default `includeChirality=False`.
        n_threads: Ignored.
        verbose: Whether to show a progress bar with ETA while processing.

    Returns:
        A list with one scaffold per input, or None where it is invalid or RDKit fails. The
        scaffold of a molecule without rings is empty (an empty string or a molecule without atoms).

    Raises:
        TypeError: If the items are neither all SMILES nor all molecules.
        ValueError: If `output` is unknown.
    """
    return _apply(
        inputs,
        output,
        "Murcko scaffolds",
        verbose,
        lambda mol: _murcko(mol, generic, include_chirality),
        verify=False,
    )


def _enumerator(
    max_tautomers: int,
    max_transforms: int,
    remove_sp3_stereo: bool,
    remove_bond_stereo: bool,
    remove_isotopic_hs: bool,
    reassign_stereo: bool,
    v1: bool,
) -> rdMolStandardize.TautomerEnumerator:
    enumerator = (
        rdMolStandardize.GetV1TautomerEnumerator() if v1 else rdMolStandardize.TautomerEnumerator()
    )
    enumerator.SetMaxTautomers(max_tautomers)
    enumerator.SetMaxTransforms(max_transforms)
    enumerator.SetRemoveSp3Stereo(remove_sp3_stereo)
    enumerator.SetRemoveBondStereo(remove_bond_stereo)
    # older RDKit versions cannot change the removal of isotopic hydrogens from Python.
    setter = getattr(enumerator, "SetRemoveIsotopicHs", None)
    if setter is not None:
        setter(remove_isotopic_hs)
    elif not remove_isotopic_hs:
        raise NotImplementedError(
            "This RDKit cannot disable remove_isotopic_hs from Python, use the compiled package"
        )
    enumerator.SetReassignStereo(reassign_stereo)
    return enumerator


def canonical_tautomers(
    inputs: Sequence[Any],
    output: str = "smiles",
    *,
    max_tautomers: int = 1000,
    max_transforms: int = 1000,
    remove_sp3_stereo: bool = True,
    remove_bond_stereo: bool = True,
    remove_isotopic_hs: bool = True,
    reassign_stereo: bool = True,
    v1: bool = False,
    n_threads: int = -1,  # noqa: ARG001
    verbose: bool = False,
) -> list[Any]:
    """Canonicalizes tautomers like `TautomerEnumerator.Canonicalize`.

    Molecules are copied first and stay unchanged. The defaults are those of RDKit.

    Args:
        inputs: SMILES strings, where None and empty strings are invalid, or molecules.
        output: Whether to return SMILES or molecules.
        max_tautomers: The maximum number of tautomers to enumerate per molecule.
        max_transforms: The maximum number of transformations to apply per molecule.
        remove_sp3_stereo: Whether to remove stereochemistry of sp3 atoms that tautomerization
            can change.
        remove_bond_stereo: Whether to remove stereochemistry of bonds that tautomerization can
            change.
        remove_isotopic_hs: Whether to remove isotopic hydrogens that tautomerization can move.
        reassign_stereo: Whether to assign the stereochemistry again after tautomerization.
        v1: Whether to use the transformations of the older `GetV1TautomerEnumerator`.
        n_threads: Ignored.
        verbose: Whether to show a progress bar with ETA while processing.

    Returns:
        A list with one tautomer per input, or None where it is invalid, RDKit fails, or the
        SMILES of the tautomer does not parse again.

    Raises:
        TypeError: If the items are neither all SMILES nor all molecules.
        ValueError: If `output` is unknown.
    """
    enumerator = _enumerator(
        max_tautomers,
        max_transforms,
        remove_sp3_stereo,
        remove_bond_stereo,
        remove_isotopic_hs,
        reassign_stereo,
        v1,
    )
    return _apply(
        inputs, output, "Canonical tautomers", verbose, enumerator.Canonicalize, verify=True
    )


def enumerate_tautomers(
    inputs: Sequence[Any],
    output: str = "smiles",
    *,
    max_tautomers: int = 1000,
    max_transforms: int = 1000,
    remove_sp3_stereo: bool = True,
    remove_bond_stereo: bool = True,
    remove_isotopic_hs: bool = True,
    reassign_stereo: bool = True,
    v1: bool = False,
    n_threads: int = -1,  # noqa: ARG001
    verbose: bool = False,
) -> list[Any]:
    """Enumerates all tautomers like `TautomerEnumerator.Enumerate`.

    Molecules are copied first and stay unchanged.

    Args:
        inputs: SMILES strings, where None and empty strings are invalid, or molecules.
        output: Whether to return SMILES or molecules.
        max_tautomers: The maximum number of tautomers to enumerate per molecule.
        max_transforms: The maximum number of transformations to apply per molecule.
        remove_sp3_stereo: Whether to remove stereochemistry of sp3 atoms that tautomerization
            can change.
        remove_bond_stereo: Whether to remove stereochemistry of bonds that tautomerization can
            change.
        remove_isotopic_hs: Whether to remove isotopic hydrogens that tautomerization can move.
        reassign_stereo: Whether to assign the stereochemistry again after tautomerization.
        v1: Whether to use the transformations of the older `GetV1TautomerEnumerator`.
        n_threads: Ignored.
        verbose: Whether to show a progress bar with ETA while processing.

    Returns:
        A list with one tuple per input of its tautomers, sorted by SMILES, and the status of the
        enumeration, which tells whether a limit stopped it. None where it is invalid or RDKit
        fails.

    Raises:
        TypeError: If the items are neither all SMILES nor all molecules.
        ValueError: If `output` is unknown.
    """
    want_mol = _want_mol(output)
    enumerator = _enumerator(
        max_tautomers,
        max_transforms,
        remove_sp3_stereo,
        remove_bond_stereo,
        remove_isotopic_hs,
        reassign_stereo,
        v1,
    )
    read = _read_inputs(inputs)
    if read is None:
        return []
    is_smiles, items = read
    results: list[Any] = []
    for item in _bar(items, len(items), "Enumerating tautomers", verbose):
        mol = _parse(item) if is_smiles else item
        if mol is None:
            results.append(None)
            continue
        try:
            found = enumerator.Enumerate(mol)
            status = STATUS[int(found.status)]
            tautomers = [_finish(t, want_mol, verify=False) for t in found.tautomers]
        except Exception:  # noqa: BLE001
            results.append(None)
            continue
        results.append(([t for t in tautomers if t is not None], status))
    return results


def score_tautomers(
    inputs: Sequence[Any],
    *,
    n_threads: int = -1,  # noqa: ARG001
    verbose: bool = False,
) -> list[int | None]:
    """Scores tautomers like `TautomerEnumerator.ScoreTautomer`.

    Molecules are copied first and stay unchanged.

    Args:
        inputs: SMILES strings, where None and empty strings are invalid, or molecules.
        n_threads: Ignored.
        verbose: Whether to show a progress bar with ETA while processing.

    Returns:
        A list with the score per input, where higher is better, or None where it is invalid or
        RDKit fails.

    Raises:
        TypeError: If the items are neither all SMILES nor all molecules.
    """
    read = _read_inputs(inputs)
    if read is None:
        return []
    is_smiles, items = read
    scores: list[int | None] = []
    for item in _bar(items, len(items), "Scoring tautomers", verbose):
        mol = _parse(item) if is_smiles else item
        try:
            scores.append(
                None if mol is None else int(rdMolStandardize.TautomerEnumerator.ScoreTautomer(mol))
            )
        except Exception:  # noqa: BLE001
            scores.append(None)
    return scores


def _pick(enumerator: rdMolStandardize.TautomerEnumerator, group: Sequence[Any]) -> Chem.Mol | None:
    """Picks the best tautomer of one group, which is None if the group is invalid."""
    read = _read_inputs(group)
    if read is None:
        return None
    is_smiles, items = read
    # an invalid item makes the group invalid.
    mols = [_parse(i) if is_smiles else i for i in items]
    if any(m is None for m in mols):
        return None
    try:
        return enumerator.PickCanonical(mols)
    except Exception:  # noqa: BLE001
        return None


def pick_canonical_tautomers(
    groups: Sequence[Sequence[Any]],
    output: str = "smiles",
    *,
    n_threads: int = -1,  # noqa: ARG001
    verbose: bool = False,
) -> list[Any]:
    """Picks the best tautomer of each group like `PickCanonical`.

    Molecules are copied first and stay unchanged.

    Args:
        groups: Groups of tautomers as SMILES strings or molecules.
        output: Whether to return SMILES or molecules.
        n_threads: Ignored.
        verbose: Whether to show a progress bar with ETA while processing.

    Returns:
        A list with the best tautomer per group, or None where the group is empty, has an invalid
        item, or RDKit fails.

    Raises:
        TypeError: If the items are neither all SMILES nor all molecules.
        ValueError: If `output` is unknown.
    """
    want_mol = _want_mol(output)
    groups = list(groups)
    # reads every group first, so a wrong item type raises before any work.
    for group in groups:
        _read_inputs(group)
    enumerator = _enumerator(1000, 1000, True, True, True, True, False)
    return [
        _finish(_pick(enumerator, group), want_mol, verify=True)
        for group in _bar(groups, len(groups), "Picking canonical tautomers", verbose, "group")
    ]


# fingerprint files and arrays


def _n_blocks(n_bits: int) -> int:
    return -(-n_bits // 64)


def _pack_bits(bits: NDArray[np.uint8]) -> NDArray[np.uint64]:
    """Packs a 2D array of 0 and 1 into little-endian 64-bit blocks per row."""
    if bits.max() > 1:
        raise ValueError("Fingerprints must only contain the values 0 and 1")
    # pads to whole 64-bit blocks, as bits are stored as little-endian blocks.
    padded: NDArray[np.uint8] = np.pad(bits, ((0, 0), (0, -bits.shape[1] % 64)))
    return np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little")).view("<u8")


def _check_blocks(blocks: NDArray[np.uint64], n_bits: int) -> NDArray[np.uint64]:
    """Checks packed blocks against the number of bits and converts them to little-endian."""
    n_blocks = _n_blocks(n_bits)
    if blocks.shape[1] != n_blocks:
        raise ValueError(
            f"{n_bits} bits need {n_blocks} blocks, but the array has {blocks.shape[1]}"
        )
    little = np.ascontiguousarray(blocks, dtype="<u8")
    padding = -n_bits % 64
    if padding and (little[:, -1] >> np.uint64(64 - padding)).any():
        raise ValueError("The padding bits beyond n_bits must be zero")
    return little


def _unpack_blocks(blocks: NDArray[np.uint64], n_bits: int) -> NDArray[np.uint8]:
    """Unpacks little-endian 64-bit blocks into a uint8 NDArray with one bit per column."""
    bits = np.unpackbits(blocks.view(np.uint8), axis=1, bitorder="little")
    return np.ascontiguousarray(bits[:, :n_bits])


def _bit_vects_to_blocks(fps: Sequence[ExplicitBitVect]) -> tuple[NDArray[np.uint64], int]:
    """Copies the bits of bit vectors into packed 64-bit blocks and returns them with `n_bits`."""
    if not all(isinstance(fp, ExplicitBitVect) for fp in fps):
        raise TypeError("fingerprints must be RDKit ExplicitBitVect objects")
    if len(fps) == 0:
        return np.zeros((0, 0), dtype="<u8"), 0
    n_bits = fps[0].GetNumBits()
    if any(fp.GetNumBits() != n_bits for fp in fps):
        raise ValueError("Inconsistent fingerprint lengths")
    width = _n_blocks(n_bits) * 8
    rows = np.zeros((len(fps), width), dtype=np.uint8)
    for i, fp in enumerate(fps):
        raw = DataStructs.BitVectToBinaryText(fp)
        rows[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return rows.view("<u8"), n_bits


def _write_blocks(path: StrPath, blocks: NDArray[np.uint64], n_fps: int, n_bits: int) -> None:
    """Writes the file: a header, then the little-endian blocks. No file is left on failure."""
    target = Path(path)
    try:
        with target.open("wb") as file:
            file.write(struct.pack("<II", n_fps, n_bits))
            blocks.astype("<u8", copy=False).tofile(file)
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def save_fingerprints(
    py_fps: Sequence[ExplicitBitVect] | NDArray[np.uint8] | NDArray[np.uint64],
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
        OSError: If the file cannot be opened for sequences of RDKit fingerprints.
    """
    is_packed = isinstance(py_fps, np.ndarray) and py_fps.dtype.type is np.uint64
    if is_packed and n_bits is None:
        raise ValueError("n_bits is required for packed uint64 arrays")
    if not is_packed and n_bits is not None:
        raise ValueError("n_bits is only used for packed uint64 arrays")

    if not isinstance(py_fps, np.ndarray):
        blocks, bits = _bit_vects_to_blocks(py_fps)
        try:
            _write_blocks(filename, blocks, len(py_fps), bits)
        except OSError as e:
            raise OSError(f"Could not open {filename}: {e}") from e
        return

    if py_fps.ndim != ARRAY_NDIM or py_fps.dtype.type not in {np.uint8, np.uint64}:
        raise TypeError("Fingerprint arrays must be 2D with dtype uint8 or uint64")
    n_fps, width = py_fps.shape
    bits_per_fp = width if n_bits is None else n_bits
    if n_fps >= MAX_FILE_COUNT or bits_per_fp >= MAX_FILE_COUNT:
        raise ValueError("Too many fingerprints or bits for the file format (max 2^32-1)")

    blocks_out: NDArray[np.uint64]
    if n_fps == 0:
        bits_per_fp = 0
        blocks_out = np.zeros((0, 0), dtype="<u8")
    elif bits_per_fp < 1:
        raise ValueError("Fingerprints must have at least one bit")
    elif is_packed:
        blocks_out = _check_blocks(np.asarray(py_fps, dtype=np.uint64), bits_per_fp)
    else:
        blocks_out = _pack_bits(np.asarray(py_fps, dtype=np.uint8))
    _write_blocks(filename, blocks_out, n_fps, bits_per_fp)


def _read_header(path: StrPath) -> tuple[int, int, int]:
    """Reads and checks the header: returns the number of fingerprints and bits, and the size."""
    size = Path(path).stat().st_size
    if size < HEADER_SIZE:
        raise ValueError("Invalid or empty binary footprint file")
    with Path(path).open("rb") as file:
        n_fps, n_bits = struct.unpack("<II", file.read(HEADER_SIZE))
    if n_fps and size < HEADER_SIZE + n_fps * _n_blocks(n_bits) * 8:
        raise ValueError(
            f"Corrupt or truncated fingerprint file: header declares {n_fps} fingerprints of "
            f"{n_bits} bits, but the file has only {size} bytes"
        )
    return n_fps, n_bits, size


def _read_blocks(path: StrPath, offset: int = 0, limit: int = 0) -> tuple[NDArray[np.uint64], int]:
    """Reads the blocks of fingerprints `offset` to `offset + limit` (0 for all) of a file."""
    n_fps, n_bits, _ = _read_header(path)
    if n_fps == 0:
        return np.zeros((0, 0), dtype=np.uint64), 0
    n_blocks = _n_blocks(n_bits)
    start = min(offset, n_fps)
    count = n_fps - start if limit == 0 else min(limit, n_fps - start)
    with Path(path).open("rb") as file:
        file.seek(HEADER_SIZE + start * n_blocks * 8)
        blocks = np.fromfile(file, dtype="<u8", count=count * n_blocks)
    return blocks.astype(np.uint64, copy=False).reshape(count, n_blocks), n_bits


def _bit_vects_from_bits(bits: NDArray[np.uint8]) -> list[ExplicitBitVect]:
    fps: list[ExplicitBitVect] = []
    for row in bits:
        fp = ExplicitBitVect(bits.shape[1])
        fp.SetBitsFromList(np.flatnonzero(row).tolist())
        fps.append(fp)
    return fps


def load_fingerprints(
    filename: StrPath, *, output: FpOutput = "bitvect"
) -> Any:  # list[ExplicitBitVect] | NDArray[np.uint8] | NDArray[np.uint64]
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
    _check_output(output)
    path = Path(filename)
    if path.stat().st_size < HEADER_SIZE:
        raise ValueError("Corrupt or truncated fingerprint file: missing header")
    blocks, n_bits = _read_blocks(path)
    if blocks.shape[0] == 0:
        return [] if output == "bitvect" else np.zeros((0, 0), dtype=_DTYPES[output])
    if output == "packed":
        return blocks
    bits = _unpack_blocks(blocks, n_bits)
    return _bit_vects_from_bits(bits) if output == "bitvect" else bits


_DTYPES = {"unpacked": np.uint8, "packed": np.uint64}


@deprecated("Use `load_fingerprints(filename, output='unpacked')` instead.")
def load_fingerprints_as_ndarray(filename: StrPath) -> NDArray[np.uint8]:
    """Loads a sequence of fingerprints as NDArray from a file created with `save_fingerprints`.

    Args:
        filename: The path to the binary fingerprint file.

    Returns:
        A unpacked uint8 NDArray with the fingerprint data.

    Raises:
        ValueError: If the file is truncated or corrupt.
    """
    return load_fingerprints(filename, output="unpacked")  # type: ignore[no-any-return]


def to_matrix(vector: NDArray[np.float64], n: int, diagonal: float = 0.0) -> NDArray[np.float64]:
    """Converts a condensed distance vector to its corresponding matrix.

    Args:
        vector: The condensed 1D distance vector
        n: Number of points
        diagonal: Value to set the diagonal to.

    Returns:
        The 2D matrix with 0.0 on the diagonal.
    """
    if vector.ndim != 1:
        raise ValueError("must be 1D array")
    if vector.size != n * (n - 1) // 2:
        raise ValueError("unexpected size")
    if n < 2:  # noqa: PLR2004
        arr = np.empty(shape=(n, n), dtype=np.float64)
    else:
        import scipy.spatial.distance

        arr = scipy.spatial.distance.squareform(vector, "tomatrix")

    np.fill_diagonal(arr, diagonal)

    return arr


def get_morgan_fps(
    mols: Sequence[Chem.Mol],
    radius: int = 2,
    n_bits: int = 2048,
    n_threads: int = -1,
    *,
    verbose: bool = False,
    output: FpOutput = "bitvect",
) -> Any:  # tuple[ExplicitBitVect, ...] | NDArray[np.uint8] | NDArray[np.uint64]
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
    _check_output(output)
    n_threads = n_threads if n_threads > 0 else cpu_count()
    morgan_gen = rdFPGen.GetMorganGenerator(radius=radius, fpSize=n_bits)
    step = max(MIN_FP_STEP, n_threads * 64) if verbose else max(len(mols), 1)
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
    blocks = (
        _bit_vects_to_blocks(fps)[0] if fps else np.zeros((0, _n_blocks(n_bits)), dtype=np.uint64)
    )
    return _unpack_blocks(blocks, n_bits) if output == "unpacked" else blocks


def save_morgan_fps_from_smiles(
    smiles: Sequence[str],
    filename: StrPath,
    radius: int = 2,
    n_bits: int = 2048,
    *,
    n_threads: int = -1,  # noqa: ARG001
    verbose: bool = False,
) -> None:
    """Computes Morgan fingerprints of SMILES and saves them to a binary file.

    Args:
        smiles: Sequence of SMILES strings to parse and sanitize.
        filename: The path to the output file.
        radius: Radius of the fingerprint.
        n_bits: Size of the fingerprint.
        n_threads: Ignored.
        verbose: Whether to show a progress bar with ETA while processing.

    Raises:
        ValueError: If a SMILES is invalid (no file is written) or `n_bits` is zero.
    """
    if n_bits == 0:
        raise ValueError("n_bits must be positive")
    if len(smiles) >= MAX_FILE_COUNT:
        raise ValueError("Too many fingerprints for the file format (max 2^32-1)")
    generator = rdFPGen.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps: list[ExplicitBitVect] = []
    for i, smi in enumerate(_bar(smiles, len(smiles), "Fingerprinting SMILES", verbose)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Could not convert SMILES {i}: {smi}")
        fps.append(generator.GetFingerprint(mol))
    blocks, bits = _bit_vects_to_blocks(fps)
    _write_blocks(filename, blocks, len(fps), bits)


# similarity

_POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def _popcount(blocks: NDArray[np.uint64]) -> NDArray[np.int64]:
    """Counts the set bits per block, summed over the last axis."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(blocks).sum(axis=-1, dtype=np.int64)
    return _POPCOUNT[blocks.view(np.uint8)].sum(axis=-1, dtype=np.int64)  # pragma: no cover


class _Fps:
    """Packed fingerprints: blocks per row and the bits per fingerprint, if known."""

    def __init__(self, blocks: NDArray[np.uint64], n_bits: int | None) -> None:
        self.blocks = blocks
        self.n_bits = n_bits

    @property
    def n_fps(self) -> int:
        return int(self.blocks.shape[0])


_ARRAY_TYPE_ERROR = (
    "Fingerprint arrays must be 2D with dtype uint8 (bits) or uint64 (packed blocks)"
)


def _extract(obj: Any) -> _Fps:
    """Reads bit vectors or an array of bits or packed blocks."""
    if isinstance(obj, np.ndarray):
        if obj.ndim != ARRAY_NDIM:
            raise TypeError(_ARRAY_TYPE_ERROR)
        if obj.dtype == np.dtype(np.uint8):
            n_fps, n_bits = obj.shape
            if n_fps == 0:
                return _Fps(np.zeros((0, _n_blocks(n_bits)), dtype=np.uint64), n_bits)
            if n_bits == 0:
                raise ValueError("Fingerprints must have at least one bit")
            if obj.max() > 1:
                raise ValueError("Fingerprint arrays must contain only 0 and 1")
            return _Fps(_pack_bits(obj).astype(np.uint64, copy=False), n_bits)
        if obj.dtype == np.dtype(np.uint64) and obj.dtype.isnative:
            return _Fps(np.ascontiguousarray(obj), None)
        raise TypeError(_ARRAY_TYPE_ERROR)
    blocks, n_bits = _bit_vects_to_blocks(list(obj))
    if blocks.shape[0] == 0:
        return _Fps(blocks.astype(np.uint64), n_bits)
    return _Fps(blocks.astype(np.uint64, copy=False), n_bits)


def _check_same_bits(a: _Fps, b: _Fps) -> None:
    bits_differ = a.n_bits is not None and b.n_bits is not None and a.n_bits != b.n_bits
    if bits_differ or a.blocks.shape[1] != b.blocks.shape[1]:
        raise ValueError("Fingerprint bit mismatch")


def _parse_agg(agg: str | None) -> str:
    value = (agg or "full").lower()
    if value not in {"mean", "max", "min", "full"}:
        raise ValueError("Invalid aggregation type. Choose 'full', 'mean', 'max', or 'min'.")
    return value


def _similarity_rows(
    left: NDArray[np.uint64], right: NDArray[np.uint64]
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """Yields the Tanimoto similarities of chunks of rows of `left` to all of `right`."""
    pop_right = _popcount(right)
    pop_left = _popcount(left)
    step = max(1, CHUNK_ELEMENTS // max(1, right.shape[0] * right.shape[1]))
    for start in range(0, left.shape[0], step):
        rows = left[start : start + step]
        common = _popcount(rows[:, None, :] & right[None, :, :])
        total = pop_left[start : start + step, None] + pop_right[None, :]
        denominator = total - common
        # no bits at all, in either fingerprint, count as dissimilar.
        sim = np.where(denominator == 0, 0.0, common / np.where(denominator == 0, 1, denominator))
        yield start, sim


def _aggregate(sim: NDArray[np.float64], agg: str) -> NDArray[np.float64]:
    result: NDArray[np.float64] = getattr(np, agg if agg != "mean" else "mean")(sim, axis=1)
    return result


def _bulk(left: _Fps, right: _Fps, agg: str, n_threads: int) -> NDArray[np.float64]:  # noqa: ARG001
    n1, n2 = left.n_fps, right.n_fps
    if n1 == 0 or n2 == 0:
        return np.zeros((n1, n2) if agg == "full" else n1, dtype=np.float64)
    _check_same_bits(left, right)
    if agg == "full":
        out = np.empty((n1, n2), dtype=np.float64)
        for start, sim in _similarity_rows(left.blocks, right.blocks):
            out[start : start + sim.shape[0]] = sim
        return out
    agged = np.empty(n1, dtype=np.float64)
    for start, sim in _similarity_rows(left.blocks, right.blocks):
        agged[start : start + sim.shape[0]] = _aggregate(sim, agg)
    return agged


def _empty_topk(n1: int) -> tuple[NDArray[np.uint32], NDArray[np.float64]]:
    return np.zeros((n1, 0), dtype=np.uint32), np.zeros((n1, 0), dtype=np.float64)


def _topk(
    left: _Fps, right: _Fps, k: int, offset: int = 0
) -> tuple[NDArray[np.uint32], NDArray[np.float64]]:
    n1, n2 = left.n_fps, right.n_fps
    if n1 == 0 or n2 == 0:
        return _empty_topk(n1)
    _check_same_bits(left, right)
    k_actual = min(k, n2)
    indices = np.zeros((n1, k_actual), dtype=np.uint32)
    scores = np.zeros((n1, k_actual), dtype=np.float64)
    if k_actual == 0:
        return indices, scores
    for start, sim in _similarity_rows(left.blocks, right.blocks):
        # a stable sort puts the lower index first among equal scores.
        order = np.argsort(-sim, axis=1, kind="stable")[:, :k_actual]
        rows = slice(start, start + sim.shape[0])
        indices[rows] = order + offset
        scores[rows] = np.take_along_axis(sim, order, axis=1)
    return indices, scores


def bulk_tanimoto_parallel(
    py_fps: Any,
    py_fps2: Any,
    n_threads: int = -1,
    agg: Agg | None = None,
) -> NDArray[np.float64]:
    """Calculates Tanimoto similarities from RDKit fingerprints or arrays.

    Args:
        py_fps: The first sequence of RDKit fingerprints or NDArray.
        py_fps2: The second sequence of RDKit fingerprints or NDArray.
        n_threads: Ignored.
        agg: Aggregation method.

    Returns:
        A 2D NumPy array with the similarity matrix if agg is None/'full' or
        1D NumPy array of aggregated similarity scores.

    Raises:
        TypeError: If an array is not 2D with dtype uint8 or uint64.
        ValueError: If the fingerprint lengths differ or a uint8 array has values other than 0 or 1.
    """
    mode = _parse_agg(agg)
    return _bulk(_extract(py_fps), _extract(py_fps2), mode, n_threads)


def bulk_tanimoto_parallel_topk(
    py_fps: Any,
    py_fps2: Any,
    k: int = 10,
    n_threads: int = -1,  # noqa: ARG001
) -> tuple[NDArray[np.uint32], NDArray[np.float64]]:
    """Finds the top-K highest similarity scores for each query in memory.

    Args:
        py_fps: Sequence of fingerprints or NDArray for the query.
        py_fps2: Sequence of fingerprints or NDArray for the database to query from.
        k: Number of most-similar molecules to query for.
        n_threads: Ignored.

    Returns:
        A tuple with a 2D NumPy array containing the indices of the similarities and
        a 2D NumPy array containing the similarity scores.

    Raises:
        TypeError: If an array is not 2D with dtype uint8 or uint64.
        ValueError: If the fingerprint lengths differ or a uint8 array has values other than 0 or 1.
    """
    return _topk(_extract(py_fps), _extract(py_fps2), k)


def _mmap_pair(path1: StrPath, path2: StrPath, offset: int, limit: int) -> tuple[_Fps, _Fps, int]:
    """Reads the query file and a window of the database file, and the window start."""
    blocks1, bits1 = _read_blocks(path1)
    n_db = _read_header(path2)[0]
    blocks2, bits2 = _read_blocks(path2, offset, limit)
    return _Fps(blocks1, bits1), _Fps(blocks2, bits2), min(offset, n_db)


def bulk_tanimoto_mmap(
    path1: StrPath,
    path2: StrPath,
    n_threads: int = -1,
    agg: Agg | None = None,
    db_offset: int = 0,
    db_limit: int = 0,
) -> NDArray[np.float64]:
    """Calculates Tanimoto similarities from binary files.

    Args:
        path1: Path to the query binary fingerprint file.
        path2: Path to the database binary fingerprint file.
        n_threads: Ignored.
        agg: Aggregation method.
        db_offset: Starting index in the second sequence to process (for windowing).
        db_limit: Number of items in the second sequence to process (0 means all).

    Returns:
        A 2D NumPy array with the similarity matrix if agg is None/'full' or
        1D NumPy array of aggregated similarity scores.
    """
    mode = _parse_agg(agg)
    left, right, _ = _mmap_pair(path1, path2, db_offset, db_limit)
    return _bulk(left, right, mode, n_threads)


def bulk_tanimoto_mmap_topk(
    path1: StrPath,
    path2: StrPath,
    k: int = 10,
    n_threads: int = -1,  # noqa: ARG001
    db_offset: int = 0,
    db_limit: int = 0,
) -> tuple[NDArray[np.uint32], NDArray[np.float64]]:
    """Finds the top-K highest similarity scores for each query from binary files.

    Args:
        path1: Path to the query binary fingerprint file.
        path2: Path to the database binary fingerprint file.
        k: Number of most-similar molecules to query for.
        n_threads: Ignored.
        db_offset: Starting index in the database to process (for windowing).
        db_limit: Number of items in the database to process (0 means all).

    Returns:
        A tuple with a 2D NumPy array containing the indices of the similarities and
        a 2D NumPy array containing the similarity scores.
    """
    left, right, start = _mmap_pair(path1, path2, db_offset, db_limit)
    return _topk(left, right, k, start)


def _internal(fps: _Fps, agg: str) -> NDArray[np.float64]:
    n = fps.n_fps
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if agg == "full":
        out = np.zeros(n * (n - 1) // 2, dtype=np.float64)
        offset = 0
        for start, sim in _similarity_rows(fps.blocks, fps.blocks):
            for row in range(sim.shape[0]):
                i = start + row
                size = n - i - 1
                out[offset : offset + size] = sim[row, i + 1 :]
                offset += size
        return out
    out = np.zeros(n, dtype=np.float64)
    if n < ARRAY_NDIM:
        return out
    for start, sim in _similarity_rows(fps.blocks, fps.blocks):
        rows = np.arange(sim.shape[0])
        sim[rows, start + rows] = {"mean": 0.0, "max": 0.0, "min": 1.0}[agg]
        out[start : start + sim.shape[0]] = (
            sim.sum(axis=1) / (n - 1) if agg == "mean" else getattr(sim, agg)(axis=1)
        )
    return out


def internal_tanimoto_parallel(
    py_fps: Any,
    n_threads: int = -1,  # noqa: ARG001
    agg: Agg | None = None,
) -> NDArray[np.float64]:
    """Calculates internal (pairwise) similarities for a dataset.

    Args:
        py_fps: The sequence of RDKit fingerprints or NDArray.
        n_threads: Ignored.
        agg: Aggregation method.

    Returns:
        If agg is 'full' or None, returns a 1D condensed distance matrix array
        (size: N * (N-1) / 2). Can be converted to a full symmetric 2D matrix
        using scipy.spatial.distance.squareform.
        Otherwise, returns a 1D array of the aggregated similarity scores.

    Raises:
        TypeError: If an array is not 2D with dtype uint8 or uint64.
        ValueError: If a uint8 array has values other than 0 or 1.
    """
    mode = _parse_agg(agg)
    return _internal(_extract(py_fps), mode)


def internal_tanimoto_mmap(
    path: StrPath,
    n_threads: int = -1,  # noqa: ARG001
    agg: Agg | None = None,
) -> NDArray[np.float64]:
    """Calculates internal (pairwise) similarities from a binary file.

    Args:
        path: Path to the binary fingerprint file.
        n_threads: Ignored.
        agg: Aggregation method.

    Returns:
        If agg is 'full' or None, returns a 1D condensed distance matrix array
        (size: N * (N-1) / 2). Can be converted to a full symmetric 2D matrix
        using scipy.spatial.distance.squareform.
        Otherwise, returns a 1D array of the aggregated similarity scores.
    """
    mode = _parse_agg(agg)
    blocks, n_bits = _read_blocks(path)
    return _internal(_Fps(blocks, n_bits), mode)


def windowed_bulk_tanimoto(
    query: StrPath,
    db: StrPath,
    window_size: int = 10_000,
    n_threads: int = -1,
    agg: Literal["max", "min", "full", "mean"] | None = None,
) -> Iterator[NDArray[np.float64]]:
    """Yields chunked similarity matrices to prevent out-of-memory errors.

    Args:
        query: Path to the query binary fingerprint file.
        db: Path to the database binary fingerprint file.
        window_size: Size of the window
        n_threads: Ignored.
        agg: Aggregation method, where "mean" is not supported.

    Returns:
        A generator with array window (shape depending on aggregation)
    """
    if agg == "mean":
        raise ValueError("Mean unsupported")

    with Path(db).open("rb") as f:
        db_size = int.from_bytes(f.read(4), "little")

    for offset in range(0, db_size, window_size):
        yield bulk_tanimoto_mmap(
            query, db, n_threads=n_threads, agg=agg, db_offset=offset, db_limit=window_size
        )


def similarity_search(
    query_ids: Sequence[T],
    query: StrPath,
    db_ids: Sequence[T],
    db: StrPath,
    k: int = 10,
    n_threads: int = cpu_count(),
) -> dict[T, list[tuple[T, float]]]:
    """Searches for the top k similar fingerprints in a list of fingerprints.

    Args:
        query_ids: Any identifier for each query fingerprint.
        query: Path to the query binary fingerprint file.
        db_ids: Any identifier for each comparison fingerprint.
        db: Path to the database binary fingerprint file.
        k: Number of most-similar molecules to query for.
        n_threads: Ignored.

    Returns:
        A dictionary of query identifiers to a list of
        tuples of (identifier, similarity) ranked by similarity.
    """
    idx, scores = bulk_tanimoto_mmap_topk(query, db, k, n_threads)

    results: dict[T, list[tuple[T, float]]] = {}
    for q_id, q_idx, q_scores in zip(query_ids, idx, scores, strict=True):
        top_results: list[tuple[T, float]] = []
        for ix, score in zip(q_idx, q_scores, strict=True):
            top_results.append((db_ids[ix], score))
        results[q_id] = top_results

    return results


# the submodules of the package


def _alias(name: str, members: dict[str, Any]) -> None:
    """Registers a submodule of `smiles_fp` that exposes `members`."""
    module = types.ModuleType(f"smiles_fp.{name}")
    module.__dict__.update(members)
    sys.modules[module.__name__] = module
    setattr(sys.modules[__name__], name, module)


def _register_submodules() -> None:
    namespace = globals()
    helpers = [
        "FpOutput",
        "get_mols",
        "get_morgan_fps",
        "load_fingerprints",
        "load_fingerprints_as_ndarray",
        "mol_from_smi",
        "save_fingerprints",
        "to_matrix",
    ]
    _alias("helpers", {name: namespace[name] for name in helpers})
    _alias(
        "search",
        {name: namespace[name] for name in ("similarity_search", "windowed_bulk_tanimoto")},
    )
    compiled = [name for name in __all__ if name not in {*helpers, "similarity_search"}]
    compiled += ["_strip_salts"]
    members = {name: namespace[name] for name in compiled if name in namespace}
    members["_save_fingerprints"] = save_fingerprints
    members["_load_fingerprints"] = load_fingerprints
    members["_fingerprints_to_blocks"] = lambda fps: _bit_vects_to_blocks(fps)[0]
    _alias("_smiles_fp", members)


if __name__ == "smiles_fp":
    _register_submodules()
