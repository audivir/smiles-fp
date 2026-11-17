"""Compares and benchmarks the tautomer functions against `TautomerEnumerator` of RDKit."""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest
from conftest import read_smis
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from smiles_fp import (
    canonical_tautomers,
    enumerate_tautomers,
    mols_from_smiles,
    pick_canonical_tautomers,
    score_tautomers,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pytest_benchmark.fixture import BenchmarkFixture

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
    "C[C@H](C(=O)C)C(=O)C",
    "C/C=C/C(=O)C",
    "O/C=C/C(C)=O",
    "[2H]OC=CC",
    "OC(=O)C=CC",
    "[Na+].[Cl-]",
)
BOOLS = ("remove_sp3_stereo", "remove_bond_stereo", "remove_isotopic_hs", "reassign_stereo")
THREADS = (1, 4, 8)


def corpus() -> list[str]:
    return [*EDGE_CASES, *(s.split()[0] for s in read_smis()[:150])]


DEFAULTS: dict[str, Any] = {
    "max_tautomers": 1000,
    "max_transforms": 1000,
    "remove_sp3_stereo": True,
    "remove_bond_stereo": True,
    "remove_isotopic_hs": True,
    "reassign_stereo": True,
    "v1": False,
}


def enumerator(**kwargs: Any) -> TautomerEnumerator:
    """Builds an enumerator from the options of the functions with the defaults of RDKit."""
    opts = {**DEFAULTS, **kwargs}
    result = rdMolStandardize.GetV1TautomerEnumerator() if opts["v1"] else TautomerEnumerator()
    result.SetMaxTautomers(opts["max_tautomers"])
    result.SetMaxTransforms(opts["max_transforms"])
    result.SetRemoveSp3Stereo(opts["remove_sp3_stereo"])
    result.SetRemoveBondStereo(opts["remove_bond_stereo"])
    result.SetReassignStereo(opts["reassign_stereo"])
    # older RDKit versions cannot change the removal of isotopic hydrogens.
    setter = getattr(result, "SetRemoveIsotopicHs", None)
    if setter is not None:  # pragma: no cover
        setter(opts["remove_isotopic_hs"])
    elif not opts["remove_isotopic_hs"]:  # pragma: no cover
        pytest.skip("RDKit cannot disable the removal of isotopic hydrogens")
    return result


def smiles_of(mol: Chem.Mol | None) -> str | None:
    return None if mol is None else Chem.MolToSmiles(mol)  # type: ignore[no-any-return,unused-ignore]


def parse(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles) if smiles else None


def is_valid(smiles: str | None) -> bool:
    """Mirrors the sanity check of `get_tautomer`, which rejects an empty molecule."""
    mol = parse(smiles) if smiles else None
    return mol is not None and mol.GetNumAtoms() > 0


def reference(mol: Chem.Mol | None, **kwargs: Any) -> str | None:
    """Returns the tautomer SMILES, or None where it is invalid or fails the sanity check."""
    if mol is None:
        return None
    smiles = smiles_of(enumerator(**kwargs).Canonicalize(mol))
    return smiles if is_valid(smiles) else None


def reference_enumerate(smiles: str, **kwargs: Any) -> tuple[list[str], str] | None:
    mol = parse(smiles)
    if mol is None:
        return None
    result = enumerator(**kwargs).Enumerate(mol)
    # the status is named like `TautomerEnumeratorStatus.MaxTautomersReached`.
    name = str(result.status).rsplit(".", 1)[-1]
    snake = "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")
    return [Chem.MolToSmiles(t) for t in result.tautomers], snake


def smiles_reference(smiles: str, **kwargs: Any) -> str | None:
    return reference(parse(smiles), **kwargs)


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


@pytest.mark.parametrize("flags", list(itertools.product([True, False], repeat=len(BOOLS))))
@pytest.mark.parametrize("v1", [False, True])
def test_options_match_reference(flags: tuple[bool, ...], v1: bool) -> None:
    kwargs: dict[str, Any] = dict(zip(BOOLS, flags, strict=True), v1=v1)
    smis = list(EDGE_CASES)
    assert canonical_tautomers(smis, **kwargs) == [smiles_reference(s, **kwargs) for s in smis]


def test_options_change_the_result() -> None:
    smis = ["CC(=O)[C@H](C)C(=O)O"]
    assert canonical_tautomers(smis, remove_sp3_stereo=False) != canonical_tautomers(smis)


def test_limits_match_reference() -> None:
    smis = ["Nc1ncnc2[nH]cnc12", "O=C1CC(=O)CC(=O)C1"]
    kwargs: dict[str, Any] = {"max_tautomers": 2, "max_transforms": 2}
    assert canonical_tautomers(smis, **kwargs) == [smiles_reference(s, **kwargs) for s in smis]


def test_none_and_empty_are_invalid() -> None:
    assert canonical_tautomers([None, "", "CCO"]) == [None, None, "CCO"]


@pytest.mark.parametrize("v1", [False, True])
def test_enumerate_matches_reference(v1: bool) -> None:
    smis = corpus()
    expected = [reference_enumerate(s, v1=v1) for s in smis]
    assert enumerate_tautomers(smis, v1=v1) == expected
    mols = enumerate_tautomers(smis, "mol", v1=v1)
    assert [
        None if r is None else ([Chem.MolToSmiles(m) for m in r[0]], r[1]) for r in mols
    ] == expected


def test_enumerate_reports_a_reached_limit() -> None:
    kwargs: dict[str, Any] = {"max_tautomers": 2}
    (result,) = enumerate_tautomers(["O=C1CC(=O)CC(=O)C1"], **kwargs)
    assert result is not None
    assert result[1] == "max_tautomers_reached"
    assert len(result[0]) == 2
    assert result == reference_enumerate("O=C1CC(=O)CC(=O)C1", **kwargs)


def test_enumerate_molecules_stay_unchanged() -> None:
    mols = [m for m in mols_from_smiles(corpus()) if m is not None]
    before = [Chem.MolToSmiles(m) for m in mols]
    assert len(enumerate_tautomers(mols, "mol")) == len(mols)
    assert [Chem.MolToSmiles(m) for m in mols] == before


def test_score_matches_reference() -> None:
    smis = corpus()
    assert score_tautomers(smis) == [score_reference(s) for s in smis]
    mols = [m for m in mols_from_smiles(smis) if m is not None]
    assert score_tautomers(mols) == [TautomerEnumerator.ScoreTautomer(m) for m in mols]
    assert score_tautomers([None, ""]) == [None, None]


def score_reference(smiles: str) -> int | None:
    mol = parse(smiles)
    return None if mol is None else TautomerEnumerator.ScoreTautomer(mol)


def reference_pick(group: Sequence[str]) -> str | None:
    mols = [parse(s) for s in group]
    if not mols or any(m is None for m in mols):
        return None
    picked = enumerator().PickCanonical([m for m in mols if m is not None])
    return smiles_of(picked) if is_valid(smiles_of(picked)) else None


def groups() -> list[list[str]]:
    smis = [s for s in EDGE_CASES if s and parse(s) is not None]
    found = [r[0] for r in enumerate_tautomers(smis) if r is not None]
    return [*found, [], ["CCO", "not a smiles("], ["CCO"], ["OC1=CCCCC1", "O=C1CCCCC1"]]


def test_pick_matches_reference() -> None:
    grouped = groups()
    expected = [reference_pick(g) for g in grouped]
    assert pick_canonical_tautomers(grouped) == expected
    mols = pick_canonical_tautomers(grouped, "mol")
    assert [smiles_of(m) for m in mols] == expected


def test_pick_molecules_match_reference_and_stay_unchanged() -> None:
    grouped = [g for g in groups() if g and all(parse(s) for s in g)]
    mol_groups = [[Chem.MolFromSmiles(s) for s in g] for g in grouped]
    before = [[Chem.MolToSmiles(m) for m in g] for g in mol_groups]
    assert pick_canonical_tautomers(mol_groups) == [reference_pick(g) for g in grouped]
    assert [[Chem.MolToSmiles(m) for m in g] for g in mol_groups] == before


def test_empty_input_and_errors() -> None:
    assert canonical_tautomers([]) == []
    assert enumerate_tautomers([]) == []
    assert score_tautomers([]) == []
    assert pick_canonical_tautomers([]) == []
    with pytest.raises(ValueError, match="Unknown output"):
        canonical_tautomers(["CCO"], "inchi")  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="Unknown output"):
        enumerate_tautomers(["CCO"], "inchi")  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="Unknown output"):
        pick_canonical_tautomers([["CCO"]], "inchi")  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        canonical_tautomers(["CCO", 1])  # type: ignore[arg-type]


def serial_canonical(smis: Sequence[str]) -> list[str | None]:  # pragma: no cover
    return [smiles_reference(s) for s in smis]


def thread_pool_canonical(smis: Sequence[str]) -> list[str | None]:  # pragma: no cover
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(smiles_reference, smis))


def joblib_canonical(smis: Sequence[str]) -> list[str | None]:  # pragma: no cover
    joblib = pytest.importorskip("joblib")
    parallel = joblib.Parallel(n_jobs=8, return_as="list")
    return parallel(joblib.delayed(smiles_reference)(smi) for smi in smis)  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    "canonicalize",
    [serial_canonical, thread_pool_canonical, joblib_canonical],
    ids=["rdkit_serial", "thread_pool", "joblib"],
)
def test_benchmark_canonical_python(  # pragma: no cover
    canonicalize: Callable[[Sequence[str]], list[str | None]],
    benchmark_smis: list[str],
    benchmark: BenchmarkFixture,
) -> None:
    result = benchmark(canonicalize, benchmark_smis)

    assert result == [smiles_reference(s) for s in benchmark_smis]


@pytest.mark.parametrize("n_threads", THREADS)
def test_benchmark_canonical_native(  # pragma: no cover
    n_threads: int, benchmark_smis: list[str], benchmark: BenchmarkFixture
) -> None:
    result = benchmark(canonical_tautomers, benchmark_smis, n_threads=n_threads)

    assert result == [smiles_reference(s) for s in benchmark_smis]


@pytest.mark.parametrize("n_threads", THREADS)
def test_benchmark_enumerate_native(  # pragma: no cover
    n_threads: int, benchmark_smis: list[str], benchmark: BenchmarkFixture
) -> None:
    result = benchmark(enumerate_tautomers, benchmark_smis, n_threads=n_threads)

    assert result == [reference_enumerate(s) for s in benchmark_smis]


@pytest.mark.parametrize("n_threads", THREADS)
def test_benchmark_score_native(  # pragma: no cover
    n_threads: int, benchmark_smis: list[str], benchmark: BenchmarkFixture
) -> None:
    result = benchmark(score_tautomers, benchmark_smis, n_threads=n_threads)

    assert result == [score_reference(s) for s in benchmark_smis]
