"""Compares `clean` with the cleaning pipeline written in RDKit Python."""

from __future__ import annotations

from itertools import product
from typing import Literal, TypedDict

import pytest
from conftest import read_smis
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
from rdkit.Chem.SaltRemover import SaltRemover
from smiles_fp import clean, mols_from_smiles

rdBase.DisableLog("rdApp.*")

ALLOWED = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 52, 53}
EDGE_CASES = (
    "",
    "not a smiles(",
    "[H][H]",
    "[H]",
    "[2H]C",
    "[13CH3][C@H](N)C(=O)O",
    "C[C@@H](F)Cl.[Na+].[Cl-]",
    "C/C=C/C",
    "CC(=O)[O-]",
    "CC(=O)[O-].[Na+]",
    "C[N+](C)(C)C.[Cl-]",
    "[O-][N+](=O)c1ccccc1",
    "OC(=O)C=CC(=O)O.CCN",
    "[Fe+2].[Cl-].[Cl-]",
    "C[Si](C)(C)C",
    "[Se]",
    "[Na+].[Na+]",
    "CCO.CCO",
)
SUFFIXES = ("", ".Cl", ".[Na+].[Cl-]")
FLAGS = ("no_salts", "no_isotopes", "no_stereo", "isomeric", "neutralize")


class Flags(TypedDict):
    no_salts: bool
    no_isotopes: bool
    no_stereo: bool
    isomeric: bool
    neutralize: bool


def smiles_flags(flags: Flags) -> Flags:
    """Returns the flags of the original pipeline, which leaves both to the SMILES writer."""
    iso = flags["isomeric"]
    return {
        **flags,
        "no_isotopes": iso and flags["no_isotopes"],
        "no_stereo": iso and flags["no_stereo"],
    }


def mol_flags(flags: Flags) -> Flags:
    """Returns the flags for molecules that are not written, where non-isomeric implies both."""
    iso = flags["isomeric"]
    return {
        **flags,
        "no_isotopes": flags["no_isotopes"] or not iso,
        "no_stereo": flags["no_stereo"] or not iso,
    }


def clean_mol_reference(mol: Chem.Mol, allowed: set[int] | None, flags: Flags) -> Chem.Mol | None:
    """Mirrors the steps of the pipeline, treating an RDKit exception as an invalid input."""
    mol = Chem.RemoveAllHs(mol)
    if mol.GetNumAtoms() == 0:
        return None
    if flags["no_salts"]:
        try:
            mol = SaltRemover().StripMol(mol)  # type: ignore[no-untyped-call]
        except ValueError:  # pragma: no cover
            return None
        if mol.GetNumAtoms() == 0:
            return None
    if flags["neutralize"]:
        mol = Uncharger().uncharge(mol)
    atoms = mol.GetAtoms()  # type: ignore[no-untyped-call,unused-ignore]
    if flags["no_isotopes"]:
        for atom in atoms:
            atom.SetIsotope(0)
    if flags["no_stereo"]:
        Chem.RemoveStereochemistry(mol)
    if allowed and any(atom.GetAtomicNum() not in allowed for atom in atoms):
        return None
    return mol


def write_reference(mol: Chem.Mol, flags: Flags) -> str | None:
    out = Chem.MolToSmiles(mol, isomericSmiles=flags["isomeric"])
    check: Chem.Mol | None = Chem.MolFromSmiles(out)
    return (
        out if out and check is not None and check.GetNumAtoms() > 0 else None
    )  # pragma: no cover


def clean_reference(smiles: str, allowed: set[int] | None, flags: Flags) -> str | None:
    if not smiles:
        return None
    parsed: Chem.Mol | None = Chem.MolFromSmiles(smiles)
    if parsed is None or parsed.GetNumAtoms() == 0:
        return None
    mol = clean_mol_reference(parsed, allowed, smiles_flags(flags))
    return None if mol is None else write_reference(mol, flags)


def one_mol(smiles: str) -> Chem.Mol:
    mol = mols_from_smiles([smiles])[0]
    assert mol is not None
    return mol


def corpus() -> list[str]:
    smis = [s.split()[0] for s in read_smis()[:60]]
    return [*EDGE_CASES, *(s + suffix for s in smis for suffix in SUFFIXES)]


@pytest.mark.parametrize("allowed", [None, ALLOWED], ids=["any", "allowed"])
@pytest.mark.parametrize("values", list(product([False, True], repeat=len(FLAGS))))
def test_matches_reference(values: tuple[bool, ...], allowed: set[int] | None) -> None:
    flags: Flags = dict(zip(FLAGS, values, strict=True))  # type: ignore[assignment]
    smis = corpus()
    expected = [clean_reference(s, allowed, flags) for s in smis]
    smiles = clean(smis, allowed_elements=allowed, **flags)
    assert smiles == expected
    mols = clean(smis, "mol", allowed_elements=allowed, **flags)
    assert [Chem.MolToSmiles(m) if m else None for m in mols] == expected


@pytest.mark.parametrize("n_threads", [1, 3])
def test_thread_counts_agree(n_threads: int) -> None:
    smis = corpus()
    assert clean(smis, no_salts=True, n_threads=n_threads) == clean(smis, no_salts=True)


def test_none_and_empty_are_invalid() -> None:
    assert clean([None, "", "CCO"]) == [None, None, "CCO"]


def test_empty_input() -> None:
    assert clean([]) == []
    assert clean([], "mol") == []


def test_unknown_output() -> None:
    with pytest.raises(ValueError, match="Unknown output"):
        clean(["CCO"], "inchi")  # type: ignore[call-overload]


@pytest.mark.parametrize("allowed", [None, ALLOWED], ids=["any", "allowed"])
@pytest.mark.parametrize("values", list(product([False, True], repeat=len(FLAGS))))
def test_mols_match_reference(values: tuple[bool, ...], allowed: set[int] | None) -> None:
    flags: Flags = dict(zip(FLAGS, values, strict=True))  # type: ignore[assignment]
    mols = [m for m in mols_from_smiles(corpus()) if m is not None]
    before = [Chem.MolToSmiles(m) for m in mols]
    cleaned = [clean_mol_reference(Chem.Mol(m), allowed, mol_flags(flags)) for m in mols]
    expected_mols = [Chem.MolToSmiles(m) if m else None for m in cleaned]
    expected_smiles = [write_reference(m, flags) if m else None for m in cleaned]
    result = clean(mols, "mol", allowed_elements=allowed, **flags)
    assert [Chem.MolToSmiles(m) if m else None for m in result] == expected_mols
    assert clean(mols, allowed_elements=allowed, **flags) == expected_smiles
    assert [Chem.MolToSmiles(m) for m in mols] == before


def test_repeated_molecule_gets_own_copies() -> None:
    mol = one_mol("CCN.Cl")
    first, second = clean([mol, mol], "mol", no_salts=True)
    assert first is not second
    assert Chem.MolToSmiles(first) == Chem.MolToSmiles(second) == "CCN"


def test_mixed_inputs_raise() -> None:
    mol = one_mol("CCO")
    with pytest.raises(TypeError):
        clean(["CCO", mol])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        clean([mol, "CCO"])  # type: ignore[arg-type]


@pytest.mark.parametrize("allowed", [None, ALLOWED], ids=["any", "allowed"])
@pytest.mark.parametrize("values", list(product([False, True], repeat=len(FLAGS))))
def test_inplace_matches_copy(values: tuple[bool, ...], allowed: set[int] | None) -> None:
    flags: Flags = dict(zip(FLAGS, values, strict=True))  # type: ignore[assignment]
    mols = [m for m in mols_from_smiles(corpus()) if m is not None]
    before = [Chem.MolToSmiles(m) for m in mols]
    copies = clean(mols, "mol", allowed_elements=allowed, **flags)
    valid = clean(mols, allowed_elements=allowed, inplace=True, **flags)
    assert valid == [c is not None for c in copies]
    expected = [Chem.MolToSmiles(c) if c else b for c, b in zip(copies, before, strict=True)]
    # invalid molecules stay unchanged.
    assert [Chem.MolToSmiles(m) for m in mols] == expected


def test_inplace_keeps_object_usable() -> None:
    mol = one_mol("CCN.Cl")
    assert clean([mol], no_salts=True, inplace=True) == [True]
    assert mol.GetNumAtoms() == 3
    assert Chem.MolToSmiles(Chem.Mol(mol)) == "CCN"
    assert Chem.MolToSmiles(Chem.RWMol(mol)) == "CCN"


def test_inplace_repeated_molecule() -> None:
    mol = one_mol("CCN.Cl")
    assert clean([mol, mol], no_salts=True, inplace=True) == [True, True]
    assert Chem.MolToSmiles(mol) == "CCN"


def test_inplace_argument_errors() -> None:
    mol = one_mol("CCO")
    assert clean([], inplace=True) == []
    with pytest.raises(TypeError, match="inplace requires molecules"):
        clean(["CCO"], "mol", inplace=True)  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="requires output"):
        clean([mol], "smiles", inplace=True)  # type: ignore[call-overload]


@pytest.mark.parametrize("output", ["smiles", "mol"])
def test_skipping_verification_keeps_results(output: Literal["smiles", "mol"]) -> None:
    mols = [m for m in mols_from_smiles(corpus()) if m is not None]
    for inputs in (corpus(), mols):
        checked = clean(inputs, output, no_salts=True)
        unchecked = clean(inputs, output, no_salts=True, verify=False)
        as_text = [Chem.MolToSmiles(x) if isinstance(x, Chem.Mol) else x for x in unchecked]
        assert as_text == [Chem.MolToSmiles(x) if isinstance(x, Chem.Mol) else x for x in checked]


@pytest.mark.parametrize("inplace", [False, True])
def test_non_isomeric_molecules_lose_isotopes_and_stereo(inplace: bool) -> None:
    mol = one_mol("[13CH3][C@H](F)Cl")
    if inplace:
        assert clean([mol], isomeric=False, inplace=True) == [True]
        result = mol
    else:
        (result,) = (m for m in clean([mol], "mol", isomeric=False) if m is not None)
    assert Chem.MolToSmiles(result) == "CC(F)Cl"
