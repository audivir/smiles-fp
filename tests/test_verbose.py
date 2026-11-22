"""Tests the progress bar of the functions with a `verbose` flag."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import conftest
import pytest
import smiles_fp
from rdkit import Chem
from smiles_fp._smiles_fp import _strip_salts

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

N = 300

CASES: dict[str, tuple[str, str, Callable[..., Any]]] = {
    # name: (input kind, bar description, call)
    "mols_from_smiles": ("smiles", "Parsing SMILES", smiles_fp.mols_from_smiles),
    "mols_to_smiles": ("mols", "Writing SMILES", smiles_fp.mols_to_smiles),
    "clean_smiles": ("smiles", "Cleaning", smiles_fp.clean),
    "clean_mols": ("mols", "Cleaning", lambda x, **kw: smiles_fp.clean(x, "mol", **kw)),
    "clean_inplace": ("mols", "Cleaning", lambda x, **kw: smiles_fp.clean(x, inplace=True, **kw)),
    "strip_salts": ("mols", "Stripping salts", _strip_salts),
    "murcko_smiles": ("smiles", "Murcko scaffolds", smiles_fp.murcko_scaffolds),
    "murcko_mols": ("mols", "Murcko scaffolds", smiles_fp.murcko_scaffolds),
    "canonical_tautomers": ("smiles", "Canonical tautomers", smiles_fp.canonical_tautomers),
    "enumerate_tautomers": ("smiles", "Enumerating tautomers", smiles_fp.enumerate_tautomers),
    "score_tautomers": ("smiles", "Scoring tautomers", smiles_fp.score_tautomers),
}


@pytest.fixture
def smis() -> list[str]:
    return [smi.split()[0] for smi in conftest.read_smis()[:N]]


def bar_lines(err: str) -> list[str]:
    """Returns the distinct redraws of the bar."""
    return [line for line in err.replace("\r", "\n").split("\n") if line.strip()]


def assert_finished(err: str, desc: str, total: int) -> None:
    lines = bar_lines(err)
    assert lines, "no progress bar was written"
    assert desc in lines[-1]
    assert f"{total}/{total}" in lines[-1]
    assert "100%" in lines[-1]


def normalized(result: Any) -> Any:
    """Makes results comparable, as molecules have no equality."""
    if isinstance(result, Chem.Mol):
        return Chem.MolToSmiles(result)
    if isinstance(result, tuple | list):
        return [normalized(item) for item in result]
    return result


@pytest.mark.parametrize("case", CASES)
def test_verbose_shows_finished_bar_and_keeps_result(
    case: str, smis: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    kind, desc, func = CASES[case]

    def inputs() -> Any:
        return smiles_fp.get_mols(smis) if kind == "mols" else smis

    expected = normalized(func(inputs()))
    assert capsys.readouterr().err == ""

    assert normalized(func(inputs(), verbose=True)) == expected
    assert_finished(capsys.readouterr().err, desc, len(smis))


def test_verbose_pick_canonical_tautomers(capsys: pytest.CaptureFixture[str]) -> None:
    groups = [["CC(=O)C", "CC(O)=C"], ["Oc1ccccn1", "O=c1cccc[nH]1"]]
    expected = smiles_fp.pick_canonical_tautomers(groups)
    assert smiles_fp.pick_canonical_tautomers(groups, verbose=True) == expected
    assert_finished(capsys.readouterr().err, "Picking canonical tautomers", len(groups))


@pytest.mark.parametrize("n_threads", [1, 4])
def test_verbose_save_morgan_fps_from_smiles(
    smis: list[str], tmp_path: Path, n_threads: int, capsys: pytest.CaptureFixture[str]
) -> None:
    quiet, loud = tmp_path / "quiet.fp", tmp_path / "loud.fp"
    smiles_fp.save_morgan_fps_from_smiles(smis, quiet, n_threads=n_threads)
    smiles_fp.save_morgan_fps_from_smiles(smis, loud, n_threads=n_threads, verbose=True)
    assert loud.read_bytes() == quiet.read_bytes()
    assert_finished(capsys.readouterr().err, "Fingerprinting SMILES", len(smis))


def test_verbose_get_mols(smis: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    smiles_fp.get_mols(smis, verbose=True)
    assert_finished(capsys.readouterr().err, "Parsing SMILES", len(smis))


@pytest.mark.parametrize("output", ["bitvect", "unpacked", "packed"])
def test_verbose_get_morgan_fps_is_chunked_but_identical(
    output: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    import numpy as np

    # chunks hold at least 1000 molecules, so this takes three.
    mols = smiles_fp.get_mols([smi.split()[0] for smi in conftest.read_smis()[:2500]])
    expected = smiles_fp.get_morgan_fps(mols, output=output)
    actual = smiles_fp.get_morgan_fps(mols, output=output, verbose=True)
    if output == "bitvect":
        assert [list(a.GetOnBits()) for a in actual] == [list(e.GetOnBits()) for e in expected]
    else:
        np.testing.assert_array_equal(actual, expected)
    assert_finished(capsys.readouterr().err, "Fingerprinting molecules", len(mols))


def test_verbose_bar_moves_while_gil_is_held(capsys: pytest.CaptureFixture[str]) -> None:
    # the bar is redrawn by the calling thread, so it moves also for functions that hold the GIL.
    # distinct molecule objects are needed, as repeated ones are written once, and one thread
    # keeps the call slow enough for several redraws.
    smis = [smi.split()[0] for smi in conftest.read_smis()] * 5
    smiles_fp.mols_to_smiles(smiles_fp.get_mols(smis), n_threads=1, verbose=True)
    lines = bar_lines(capsys.readouterr().err)
    counts = {line.split("|")[1].split("[")[0].strip() for line in lines if "|" in line}
    assert len(counts) > 2, "the bar was only drawn at the start and end"


@pytest.mark.parametrize(
    "call",
    [
        lambda: smiles_fp.mols_from_smiles([], verbose=True),
        lambda: smiles_fp.get_morgan_fps([], verbose=True),
    ],
    ids=["mols_from_smiles", "get_morgan_fps"],
)
def test_verbose_empty_input_shows_no_bar(
    call: Callable[[], object], capsys: pytest.CaptureFixture[str]
) -> None:
    call()
    assert capsys.readouterr().err == ""
