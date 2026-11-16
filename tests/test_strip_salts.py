"""Compares the salt stripper with `SaltRemover` of RDKit."""

from __future__ import annotations

import pytest
from conftest import read_smis
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from smiles_fp import get_mols
from smiles_fp._smiles_fp import _strip_salts

REMOVER = SaltRemover()  # type: ignore[no-untyped-call]

EDGE_CASES = (
    "CC(=O)[O-].[Na+]",
    "CC(=O)O.[Na]",
    "Cl.Cl",
    "[Na+].[Cl-]",
    "[Na+].[Na+]",
    "CN(Br)Cl.Cl",
    "CCN.Cl.Br",
    "c1ccccc1.O",
    "C[NH+](C)C.[Cl-]",
    "OC(=O)C=CC(=O)O.CCN",
    "CCO.CCO",
    "OC(=O)C(O)C(O)C(=O)O.C1CCCCC1NC1CCCCC1",
    "O",
    "C",
)
SUFFIXES = ("", ".Cl", ".[Na+].[Cl-]", ".O.CC(=O)O")


def strip_reference(mol: Chem.Mol, keep_last: bool) -> Chem.Mol | None:
    try:
        return REMOVER.StripMol(Chem.Mol(mol), dontRemoveEverything=keep_last)  # type: ignore[no-untyped-call,no-any-return]
    except ValueError:  # pragma: no cover
        return None


def smiles_of(mol: Chem.Mol | None) -> str | None:
    return None if mol is None else Chem.MolToSmiles(mol)


def salty_mols() -> list[Chem.Mol]:
    smis = [s.split()[0] for s in read_smis()[:300]]
    smis = [*EDGE_CASES, *(s + suffix for s in smis for suffix in SUFFIXES)]
    return get_mols(smis)


@pytest.mark.parametrize("keep_last", [False, True])
@pytest.mark.parametrize("n_threads", [1, 4])
def test_matches_salt_remover(keep_last: bool, n_threads: int) -> None:
    mols = salty_mols()
    result = _strip_salts(mols, keep_last, n_threads=n_threads)
    expected = [strip_reference(m, keep_last) for m in mols]
    assert [smiles_of(m) for m in result] == [smiles_of(m) for m in expected]


def test_removes_salts() -> None:
    mols = get_mols(["CCN.Cl", "CC(=O)[O-].[Na+]", "CCO.CCO"])
    assert [smiles_of(m) for m in _strip_salts(mols)] == ["CCN", "", "CCO.CCO"]


def test_inputs_stay_unchanged() -> None:
    mols = get_mols(["CCN.Cl"])
    _strip_salts(mols)
    assert Chem.MolToSmiles(mols[0]) == "CCN.Cl"


def test_repeated_molecule_gets_own_copies() -> None:
    mol = get_mols(["CCN.Cl"])[0]
    first, second = _strip_salts([mol, mol])
    assert first is not second
    assert smiles_of(first) == smiles_of(second) == "CCN"


def test_empty_and_invalid_input() -> None:
    assert _strip_salts([]) == []
    with pytest.raises(TypeError):
        _strip_salts(["CCN"])  # type: ignore[list-item]
