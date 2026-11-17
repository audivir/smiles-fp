"""Compares `murcko_scaffolds` with `MurckoScaffold.GetScaffoldForMol` of RDKit."""

from __future__ import annotations

import pytest
from conftest import read_smis
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from smiles_fp import mols_from_smiles, murcko_scaffolds

rdBase.DisableLog("rdApp.*")

EDGE_CASES = (
    "",
    "not a smiles(",
    "CCO",
    "C",
    "c1ccccc1CCN",
    "C1CC1C(=O)O.[Na+]",
    "c1ccc(cc1)-c1ccncc1",
    "C[C@H]1CC[C@@H](C)CC1",
    "[13CH3]c1ccccc1",
    "O=C1CCC(=O)N1",
    "C1CCC2(CC1)CCCC2",
    "[H][H]",
)


def smiles_of(mol: Chem.Mol | None) -> str | None:
    return None if mol is None else Chem.MolToSmiles(mol)  # type: ignore[no-any-return,unused-ignore]


def scaffold_reference(mol: Chem.Mol) -> str | None:
    """Returns the SMILES of the scaffold, or None where RDKit raises like `murcko_scaffolds`."""
    scaffold: Chem.Mol | None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)  # type: ignore[no-untyped-call,unused-ignore]
    except RuntimeError:  # pragma: no cover
        scaffold = None
    return smiles_of(scaffold)


MURCKO_WORKS = scaffold_reference(Chem.MolFromSmiles("c1ccccc1")) is not None
"""Whether RDKit computes scaffolds, which `MurckoDecompose` of 2026.9.2 does not on macOS."""


def expect(scaffold: str) -> str | None:
    """Returns the expected scaffold, which is None where RDKit cannot compute any."""
    return scaffold if MURCKO_WORKS else None


def corpus() -> list[str]:
    return [*EDGE_CASES, *(s.split()[0] for s in read_smis()[:300])]


def smiles_reference(smiles: str) -> str | None:
    mol: Chem.Mol | None = Chem.MolFromSmiles(smiles) if smiles else None
    return None if mol is None else scaffold_reference(mol)


@pytest.mark.parametrize("n_threads", [1, 4])
def test_smiles_match_reference(n_threads: int) -> None:
    smis = corpus()
    expected = [smiles_reference(s) for s in smis]
    assert murcko_scaffolds(smis, n_threads=n_threads) == expected
    mols = murcko_scaffolds(smis, "mol", n_threads=n_threads)
    assert [smiles_of(m) for m in mols] == expected


def test_molecules_match_reference_and_stay_unchanged() -> None:
    mols = [m for m in mols_from_smiles(corpus()) if m is not None]
    before = [Chem.MolToSmiles(m) for m in mols]
    expected = [scaffold_reference(Chem.Mol(m)) for m in mols]
    assert murcko_scaffolds(mols) == expected
    assert [smiles_of(m) for m in murcko_scaffolds(mols, "mol")] == expected
    assert [Chem.MolToSmiles(m) for m in mols] == before


def test_acyclic_is_empty_but_not_invalid() -> None:
    assert murcko_scaffolds(["CCO", "c1ccccc1C"]) == [expect(""), expect("c1ccccc1")]
    (empty,) = murcko_scaffolds(["CCO"], "mol")
    assert (empty is not None) == MURCKO_WORKS
    assert empty is None or empty.GetNumAtoms() == 0


def test_none_and_empty_are_invalid() -> None:
    assert murcko_scaffolds([None, "", "c1ccccc1"]) == [None, None, expect("c1ccccc1")]


def test_repeated_molecule_gets_own_scaffolds() -> None:
    mol = mols_from_smiles(["c1ccccc1C"])[0]
    assert mol is not None
    first, second = murcko_scaffolds([mol, mol], "mol")
    assert (first is not second) == MURCKO_WORKS
    assert [smiles_of(first), smiles_of(second)] == [expect("c1ccccc1")] * 2


def test_empty_input_and_errors() -> None:
    assert murcko_scaffolds([]) == []
    with pytest.raises(ValueError, match="Unknown output"):
        murcko_scaffolds(["c1ccccc1"], "inchi")  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        murcko_scaffolds(["c1ccccc1", 1])  # type: ignore[arg-type]


OPTION_CASES = (
    "c1ncccc1CCN",
    "C[C@H]1CC[C@@H](C)CC1",
    "[13CH3]c1ccccc1",
    "C1[C@](Cl)(F)O1",
    "C1[NH2+]C1",
    "c1[nH]ccc1C",
    "C/C=C/c1ccccc1/C=C/C",
    "O=C1CCC(=O)N1",
    "[H]c1ccccc1[H]",
    "CCO",
)


def option_reference(smiles: str, *, generic: bool, include_chirality: bool) -> str | None:
    """Mirrors `MakeScaffoldGeneric` and `MurckoScaffoldSmiles` of RDKit."""
    mol: Chem.Mol = Chem.MolFromSmiles(smiles)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)  # type: ignore[no-untyped-call,unused-ignore]
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)  # type: ignore[no-untyped-call,unused-ignore]
    except RuntimeError:  # pragma: no cover
        return None
    return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)  # type: ignore[no-any-return,unused-ignore]


@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("include_chirality", [False, True])
def test_options_match_reference(generic: bool, include_chirality: bool) -> None:
    smis = [*OPTION_CASES, *(s.split()[0] for s in read_smis()[:150])]
    expected = [
        None
        if not MURCKO_WORKS
        else option_reference(s, generic=generic, include_chirality=include_chirality)
        for s in smis
    ]
    assert murcko_scaffolds(smis, generic=generic, include_chirality=include_chirality) == expected
    mols = murcko_scaffolds(smis, "mol", generic=generic, include_chirality=include_chirality)
    assert [smiles_of(m) for m in mols] == expected
    parsed = [m for m in mols_from_smiles(smis) if m is not None]
    assert murcko_scaffolds(parsed, generic=generic, include_chirality=include_chirality) == [
        e for s, e in zip(smis, expected, strict=True) if Chem.MolFromSmiles(s) is not None
    ]


def test_options_change_the_result() -> None:
    chiral = "Cc1cccc2c1OCC[C@@H]2NC(C)C1(C)CCCO1"
    assert murcko_scaffolds([chiral]) == [expect("c1ccc2c(c1)OCC[C@@H]2NCC1CCCO1")]
    assert murcko_scaffolds([chiral], include_chirality=False) == [
        expect("c1ccc2c(c1)OCCC2NCC1CCCO1")
    ]
    assert murcko_scaffolds(["c1ncccc1C"], generic=True) == [expect("C1CCCCC1")]
