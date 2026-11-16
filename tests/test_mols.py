"""Tests for the threaded SMILES <-> molecule conversion."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, overload

import conftest
import pytest
import smiles_fp
from rdkit import Chem
from typing_extensions import override

SMILES = list(conftest.get_cached_mols())[:300]


def _signature(mol: Chem.Mol) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Comparable description of an (also unsanitized) molecule."""
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]  # type: ignore[no-untyped-call]
    bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble())
        for b in mol.GetBonds()  # type: ignore[no-untyped-call]
    ]
    return atoms, bonds


@pytest.mark.parametrize("n_threads", [1, 4, -1])
def test_mols_from_smiles_matches_rdkit(n_threads: int) -> None:
    mols = smiles_fp.mols_from_smiles(SMILES, n_threads=n_threads)
    assert len(mols) == len(SMILES)
    for smi, mol in zip(SMILES, mols, strict=True):
        assert mol is not None
        assert isinstance(mol, Chem.Mol)
        assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(Chem.MolFromSmiles(smi))


@pytest.mark.parametrize("sanitize", [True, False])
def test_mols_from_smiles_sanitize_flag(sanitize: bool) -> None:
    mols = smiles_fp.mols_from_smiles(SMILES, sanitize, n_threads=2)
    for smi, mol in zip(SMILES, mols, strict=True):
        assert mol is not None
        assert _signature(mol) == _signature(Chem.MolFromSmiles(smi, sanitize))


def test_mols_from_smiles_unsanitized_keeps_hs_and_ring_info() -> None:
    [mol] = smiles_fp.mols_from_smiles(["[H]C([H])([H])O"], False)
    assert mol is not None
    assert mol.GetNumAtoms() == 5  # explicit Hs are not removed without sanitization
    [mol] = smiles_fp.mols_from_smiles(["[H]C([H])([H])O"], True)
    assert mol is not None
    assert mol.GetNumAtoms() == 2


def test_mols_from_smiles_invalid_gives_none() -> None:
    smis = ["CCO", "not a smiles", "C1CC", "c1ccccc1"]
    mols = smiles_fp.mols_from_smiles(smis)
    assert [m is None for m in mols] == [m is None for m in map(Chem.MolFromSmiles, smis)]
    assert [m is None for m in mols] == [False, True, True, False]


def test_mols_from_smiles_empty_and_name() -> None:
    assert smiles_fp.mols_from_smiles([]) == []
    [mol] = smiles_fp.mols_from_smiles(["CCO ethanol"])
    assert mol is not None
    assert mol.GetProp("_Name") == "ethanol"


def test_mols_from_smiles_rejects_non_strings() -> None:
    with pytest.raises(TypeError):
        smiles_fp.mols_from_smiles([1, 2])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        smiles_fp.mols_from_smiles("CCO")


def test_mols_from_smiles_survives_many_batches() -> None:
    for _ in range(50):
        mols = smiles_fp.mols_from_smiles(SMILES, n_threads=8)
        assert all(m is not None for m in mols)


@pytest.mark.parametrize("n_threads", [1, 4, -1])
def test_mols_to_smiles_matches_rdkit(n_threads: int) -> None:
    mols = list(conftest.get_cached_mols().values())[:300]
    assert smiles_fp.mols_to_smiles(mols, n_threads=n_threads) == [
        Chem.MolToSmiles(m) for m in mols
    ]


MOLS = list(conftest.get_cached_mols().values())[:300]


@pytest.mark.parametrize(
    ("this", "rdkit"),
    [
        ({}, {}),
        ({"isomeric_smiles": False}, {"isomericSmiles": False}),
        ({"kekule_smiles": True}, {"kekuleSmiles": True}),
        ({"canonical": False}, {"canonical": False}),
        ({"rooted_at_atom": 0}, {"rootedAtAtom": 0}),
        ({"rooted_at_atom": 2, "canonical": False}, {"rootedAtAtom": 2, "canonical": False}),
        ({"all_bonds_explicit": True}, {"allBondsExplicit": True}),
        ({"all_hs_explicit": True}, {"allHsExplicit": True}),
        (
            {"isomeric_smiles": False, "kekule_smiles": True, "all_hs_explicit": True},
            {"isomericSmiles": False, "kekuleSmiles": True, "allHsExplicit": True},
        ),
    ],
)
@pytest.mark.parametrize("n_threads", [1, 4])
def test_mols_to_smiles_forwards_arguments(
    this: dict[str, Any], rdkit: dict[str, Any], n_threads: int
) -> None:
    got = smiles_fp.mols_to_smiles(MOLS, n_threads=n_threads, **this)
    assert got == [Chem.MolToSmiles(m, **rdkit) for m in MOLS]


def test_mols_to_smiles_ignore_atom_map_numbers() -> None:
    mols = [Chem.MolFromSmiles(s) for s in ("[CH3:2][CH2:1]O", "[CH3:1][CH2:2]O", "OCC")]
    for flag in (True, False):
        assert smiles_fp.mols_to_smiles(mols, ignore_atom_map_numbers=flag) == [
            Chem.MolToSmiles(m, ignoreAtomMapNumbers=flag) for m in mols
        ]


def test_mols_to_smiles_positional_arguments_follow_rdkit_order() -> None:
    got = smiles_fp.mols_to_smiles(MOLS, False, True, 1, False, True, True)
    assert got == [Chem.MolToSmiles(m, False, True, 1, False, True, True) for m in MOLS]
    with pytest.raises(TypeError):  # ignore_atom_map_numbers and n_threads are keyword-only
        smiles_fp.mols_to_smiles(MOLS, True, False, -1, True, False, False, True)  # type: ignore[call-arg]


def test_mols_to_smiles_has_no_do_random() -> None:
    with pytest.raises(TypeError):
        smiles_fp.mols_to_smiles(MOLS, do_random=True)  # type: ignore[call-arg]


def test_mols_to_smiles_rooted_at_atom_out_of_range_raises() -> None:
    with pytest.raises(RuntimeError, match="must be less than the number of atoms"):
        smiles_fp.mols_to_smiles([Chem.MolFromSmiles("CCO")], rooted_at_atom=99)


@pytest.mark.parametrize("n_threads", [2, 12])
def test_mols_to_smiles_repeated_molecule_objects(n_threads: int) -> None:
    """Writing the same ROMol from several threads at once crashed RDKit before this was fixed."""
    big = max(MOLS, key=lambda m: m.GetNumAtoms())
    mols = [big] * 5_000 + MOLS + [big] * 5_000
    got = smiles_fp.mols_to_smiles(mols, n_threads=n_threads)
    assert got == [Chem.MolToSmiles(m) for m in mols]


def test_mols_to_smiles_error_reports_first_position_of_repeated_molecule() -> None:
    mol = Chem.MolFromSmiles("CCO")
    with pytest.raises(RuntimeError, match="molecule 0"):
        smiles_fp.mols_to_smiles([mol, mol], rooted_at_atom=99)


def _expected_rooted(mols: list[Chem.Mol], roots: list[int | None]) -> list[str]:
    return [
        Chem.MolToSmiles(m, rootedAtAtom=-1 if r is None else r)
        for m, r in zip(mols, roots, strict=True)
    ]


@pytest.mark.parametrize("n_threads", [1, 4])
def test_mols_to_smiles_rooted_at_atom_per_molecule(n_threads: int) -> None:
    roots: list[int | None] = [None if i % 4 == 3 else i % 3 for i in range(len(MOLS))]
    got = smiles_fp.mols_to_smiles(MOLS, rooted_at_atom=roots, n_threads=n_threads)
    assert got == _expected_rooted(MOLS, roots)
    assert smiles_fp.mols_to_smiles(MOLS, rooted_at_atom=tuple(roots)) == got


def test_mols_to_smiles_rooted_at_atom_none_and_int() -> None:
    default = [Chem.MolToSmiles(m) for m in MOLS]
    assert smiles_fp.mols_to_smiles(MOLS, rooted_at_atom=None) == default
    assert smiles_fp.mols_to_smiles(MOLS, rooted_at_atom=[None] * len(MOLS)) == default
    assert smiles_fp.mols_to_smiles(MOLS, rooted_at_atom=-1) == default
    assert smiles_fp.mols_to_smiles(MOLS, rooted_at_atom=2) == _expected_rooted(
        MOLS, [2] * len(MOLS)
    )


@pytest.mark.parametrize("n_threads", [2, 12])
def test_mols_to_smiles_same_molecule_with_different_roots(n_threads: int) -> None:
    """One ROMol with many roots is still written by a single thread (no shared-molecule crash)."""
    big = max(MOLS, key=lambda m: m.GetNumAtoms())
    mols = [big] * 4_000 + MOLS + [big] * 4_000
    roots: list[int | None] = [i % m.GetNumAtoms() if i % 5 else None for i, m in enumerate(mols)]
    got = smiles_fp.mols_to_smiles(mols, rooted_at_atom=roots, n_threads=n_threads)
    assert got == _expected_rooted(mols, roots)


@pytest.mark.parametrize("n_threads", [1, 4, 12])
def test_mols_to_smiles_few_molecules_many_roots_uses_copies(n_threads: int) -> None:
    # few distinct molecules cannot keep all threads busy, so repeated ones are written from copies.
    pair = sorted(MOLS, key=lambda m: m.GetNumAtoms())[-2:]
    mols = [pair[i % 2] for i in range(6_000)]
    roots: list[int | None] = [i % m.GetNumAtoms() for i, m in enumerate(mols)]
    roots[::7] = [None] * len(roots[::7])
    got = smiles_fp.mols_to_smiles(mols, rooted_at_atom=roots, n_threads=n_threads)
    assert got == _expected_rooted(mols, roots)


@pytest.mark.parametrize("n_threads", [1, 3, 32])
def test_mols_to_smiles_single_molecule_all_roots_matches_rdkit(n_threads: int) -> None:
    peptide = Chem.MolFromSmiles("NCC(=O)" * 12 + "O")
    roots = list(range(peptide.GetNumAtoms()))
    got = smiles_fp.mols_to_smiles(
        [peptide] * len(roots), rooted_at_atom=roots, n_threads=n_threads
    )
    assert got == [Chem.MolToSmiles(peptide, rootedAtAtom=r) for r in roots]


def test_mols_to_smiles_rooted_at_atom_length_mismatch_raises() -> None:
    mols = MOLS[:3]
    with pytest.raises(ValueError, match="rooted_at_atom has 2 items for 3 molecules"):
        smiles_fp.mols_to_smiles(mols, rooted_at_atom=[0, 1])
    with pytest.raises(ValueError, match="rooted_at_atom has 0 items for 3 molecules"):
        smiles_fp.mols_to_smiles(mols, rooted_at_atom=[])


@pytest.mark.parametrize("bad", ["01", ["a", 0, 0], [0.5, 0, 0], {0, 1, 2}, object()])
def test_mols_to_smiles_rooted_at_atom_wrong_type_raises(bad: Any) -> None:
    with pytest.raises(TypeError, match="rooted_at_atom must be"):
        smiles_fp.mols_to_smiles(MOLS[:3], rooted_at_atom=bad)


def test_mols_to_smiles_rooted_at_atom_out_of_range_item_reports_position() -> None:
    mol = Chem.MolFromSmiles("CCO")
    with pytest.raises(RuntimeError, match="molecule 2"):
        smiles_fp.mols_to_smiles([mol, mol, mol], rooted_at_atom=[0, 1, 99])


def test_mols_to_smiles_roundtrip() -> None:
    mols = smiles_fp.mols_from_smiles(SMILES)
    smis = smiles_fp.mols_to_smiles(mols)  # type: ignore[arg-type]
    again = smiles_fp.mols_to_smiles(smiles_fp.mols_from_smiles(smis))  # type: ignore[arg-type]
    assert smis == again


def test_mols_to_smiles_accepts_tuples_and_empty() -> None:
    mols = tuple(list(conftest.get_cached_mols().values())[:5])
    assert smiles_fp.mols_to_smiles(mols) == [Chem.MolToSmiles(m) for m in mols]
    assert smiles_fp.mols_to_smiles([]) == []


def test_mols_to_smiles_rejects_non_mols() -> None:
    with pytest.raises(TypeError):
        smiles_fp.mols_to_smiles(["CCO"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        smiles_fp.mols_to_smiles([None])  # type: ignore[list-item]


class _FreshMols(Sequence[Chem.Mol]):
    """Creates a new molecule on every access: only the extracted references keep them alive."""

    def __init__(self, smis: list[str]) -> None:
        self.smis = smis

    @overload
    def __getitem__(self, i: int) -> Chem.Mol: ...
    @overload
    def __getitem__(self, i: slice) -> Sequence[Chem.Mol]: ...
    @override
    def __getitem__(self, i: int | slice) -> Chem.Mol | Sequence[Chem.Mol]:
        if isinstance(i, slice):  # pragma: no cover
            raise NotImplementedError
        return Chem.MolFromSmiles(self.smis[i])

    @override
    def __len__(self) -> int:
        return len(self.smis)


def test_mols_to_smiles_keeps_temporary_molecules_alive() -> None:
    smis = SMILES[:200]
    expected = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smis]
    fresh = _FreshMols(smis)
    assert len(fresh) == len(smis)
    assert smiles_fp.mols_to_smiles(fresh, n_threads=4) == expected


def test_get_mols_forwards_to_mols_from_smiles_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mols = smiles_fp.get_mols(SMILES[:50], n_jobs=2)
    assert [Chem.MolToSmiles(m) for m in mols] == [
        Chem.MolToSmiles(m) for m in smiles_fp.mols_from_smiles(SMILES[:50])
    ]
    unsanitized = smiles_fp.get_mols(["[H]C([H])([H])O"], sanitize=False)
    assert unsanitized[0].GetNumAtoms() == 5


def test_get_mols_accepts_any_iterable() -> None:
    assert len(smiles_fp.get_mols(iter(SMILES[:5]))) == 5


def test_get_mols_raises_on_invalid_smiles() -> None:
    with pytest.raises(
        ValueError, match="Could not convert 2 SMILES to molecules: bad one, bad two"
    ):
        smiles_fp.get_mols(["CCO", "bad one", "c1ccccc1", "bad two"])


def test_mol_from_smi() -> None:
    assert Chem.MolToSmiles(smiles_fp.mol_from_smi("OCC")) == "CCO"
    with pytest.raises(ValueError, match="Could not convert not a smiles"):
        smiles_fp.mol_from_smi("not a smiles")
