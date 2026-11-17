"""Compares `canonical_tautomers` with `TautomerEnumerator.Canonicalize` of RDKit."""

from __future__ import annotations

import pytest
from conftest import read_smis
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from smiles_fp import canonical_tautomers, mols_from_smiles

rdBase.DisableLog("rdApp.*")

EDGE_CASES = (
    "",
    "not a smiles(",
    "CCO",
    "O=C1CCCCC1",
    "OC1=CCCCC1",
    "Oc1ccccn1",
    "O=c1cccc[nH]1",
    "CC(=O)CC(=O)C",
    "Nc1ncnc2[nH]cnc12",
    "C[C@H](O)C(=O)C",
    "[Na+].[Cl-]",
)


def corpus() -> list[str]:
    return [*EDGE_CASES, *(s.split()[0] for s in read_smis()[:150])]


def enumerator() -> TautomerEnumerator:
    result = TautomerEnumerator()
    result.SetMaxTautomers(10_000)
    result.SetMaxTransforms(10_000)
    return result


def smiles_of(mol: Chem.Mol | None) -> str | None:
    return None if mol is None else Chem.MolToSmiles(mol)  # type: ignore[no-any-return,unused-ignore]


def reference(mol: Chem.Mol | None) -> str | None:
    """Returns the tautomer SMILES, or None where it is invalid or fails the sanity check."""
    if mol is None:
        return None
    smiles = smiles_of(enumerator().Canonicalize(mol))
    checked: Chem.Mol | None = Chem.MolFromSmiles(smiles) if smiles else None
    return smiles if checked is not None and checked.GetNumAtoms() > 0 else None


def smiles_reference(smiles: str) -> str | None:
    return reference(Chem.MolFromSmiles(smiles) if smiles else None)


@pytest.mark.parametrize("n_threads", [1, 4])
def test_smiles_match_reference(n_threads: int) -> None:
    smis = corpus()
    expected = [smiles_reference(s) for s in smis]
    assert canonical_tautomers(smis, n_threads=n_threads) == expected
    mols = canonical_tautomers(smis, "mol", n_threads=n_threads)
    assert [smiles_of(m) for m in mols] == expected


def test_molecules_match_reference_and_stay_unchanged() -> None:
    mols = [m for m in mols_from_smiles(corpus()) if m is not None]
    before = [Chem.MolToSmiles(m) for m in mols]
    expected = [reference(Chem.Mol(m)) for m in mols]
    assert canonical_tautomers(mols) == expected
    assert [smiles_of(m) for m in canonical_tautomers(mols, "mol")] == expected
    assert [Chem.MolToSmiles(m) for m in mols] == before


def test_tautomers_are_merged() -> None:
    assert len(set(canonical_tautomers(["O=C1CCCCC1", "OC1=CCCCC1"]))) == 1


def test_limits_are_used() -> None:
    smis = ["Nc1ncnc2[nH]cnc12", "O=C1CC(=O)CC(=O)C1"]
    small = canonical_tautomers(smis, max_tautomers=2, max_transforms=2)
    mols = [Chem.MolFromSmiles(s) for s in smis]
    low = TautomerEnumerator()
    low.SetMaxTautomers(2)
    low.SetMaxTransforms(2)
    assert small == [smiles_of(low.Canonicalize(m)) for m in mols]


def test_none_and_empty_are_invalid() -> None:
    assert canonical_tautomers([None, "", "CCO"]) == [None, None, "CCO"]


def test_empty_input_and_errors() -> None:
    assert canonical_tautomers([]) == []
    with pytest.raises(ValueError, match="Unknown output"):
        canonical_tautomers(["CCO"], "inchi")  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        canonical_tautomers(["CCO", 1])  # type: ignore[arg-type]
