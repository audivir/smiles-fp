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
