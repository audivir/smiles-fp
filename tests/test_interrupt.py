"""Tests that Ctrl-C stops the long-running functions quickly."""

from __future__ import annotations

import contextlib
import signal
import time
from typing import TYPE_CHECKING, Any

import conftest
import pytest
import smiles_fp
from smiles_fp._smiles_fp import _strip_salts

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

DELAY = 0.2
"""Seconds until the simulated Ctrl-C."""
LIMIT = 1.0
"""Seconds within which a call must stop, where each call needs several seconds in one thread."""
N_COPIES = 10


CASES: dict[str, tuple[str, Callable[..., Any]]] = {
    # name: (input kind, call), where only mols_to_smiles and strip_salts hold the GIL.
    "mols_from_smiles": (
        "smiles",
        lambda x, **kw: smiles_fp.mols_from_smiles(x, n_threads=1, **kw),
    ),
    "clean_smiles_to_mols": (
        "smiles",
        lambda x, **kw: smiles_fp.clean(x, "mol", n_threads=1, **kw),
    ),
    "clean_mols_copy": ("mols", lambda x, **kw: smiles_fp.clean(x, "mol", n_threads=1, **kw)),
    "mols_to_smiles": ("mols", lambda x, **kw: smiles_fp.mols_to_smiles(x, n_threads=1, **kw)),
    "strip_salts": ("mols", lambda x, **kw: _strip_salts(x, n_threads=1, **kw)),
    "murcko": ("smiles", lambda x, **kw: smiles_fp.murcko_scaffolds(x, n_threads=1, **kw)),
    "canonical_tautomers": (
        "smiles",
        lambda x, **kw: smiles_fp.canonical_tautomers(x, n_threads=1, **kw),
    ),
}


@pytest.fixture(scope="module")
def big_smis() -> list[str]:
    return [smi.split()[0] for smi in conftest.read_smis()] * N_COPIES


@pytest.fixture(scope="module")
def big_mols(big_smis: list[str]) -> list[Any]:
    return smiles_fp.get_mols(big_smis)


@contextlib.contextmanager
def interrupt_after(delay: float) -> Iterator[None]:
    """Delivers a real signal that Python turns into `KeyboardInterrupt`, like Ctrl-C.

    A timer thread would not do, as it cannot run while a call holds the GIL.
    """

    def handler(*_: object) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, delay)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def assert_interrupted(call: Callable[[], object]) -> None:
    start = time.perf_counter()
    with pytest.raises(KeyboardInterrupt), interrupt_after(DELAY):
        call()
    elapsed = time.perf_counter() - start
    assert elapsed < LIMIT, f"stopped only after {elapsed:.2f} s"


@pytest.mark.parametrize("verbose", [False, True], ids=["quiet", "verbose"])
@pytest.mark.parametrize("case", CASES)
def test_ctrl_c_stops_call(
    case: str,
    verbose: bool,
    big_smis: list[str],
    big_mols: list[Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    kind, func = CASES[case]
    data = big_smis if kind == "smiles" else big_mols
    assert_interrupted(lambda: func(data, verbose=verbose))
    capsys.readouterr()


def test_ctrl_c_stops_inplace_clean(big_smis: list[str]) -> None:
    mols = smiles_fp.get_mols(big_smis)
    assert_interrupted(lambda: smiles_fp.clean(mols, inplace=True, n_threads=1))
    # the molecules stay usable, as the in-place work done so far remains.
    assert all(mol.GetNumAtoms() > 0 for mol in mols[:100])


def test_ctrl_c_stops_save_morgan_fps(big_smis: list[str], tmp_path: Any) -> None:
    path = tmp_path / "fps.fp"
    assert_interrupted(lambda: smiles_fp.save_morgan_fps_from_smiles(big_smis, path, n_threads=1))
    assert not path.exists()


def test_functions_work_after_interrupt(big_smis: list[str], big_mols: list[Any]) -> None:
    assert_interrupted(lambda: smiles_fp.mols_to_smiles(big_mols, n_threads=1))
    from rdkit import Chem

    smis = big_smis[:50]
    mols = smiles_fp.get_mols(smis)
    assert smiles_fp.mols_to_smiles(mols) == [Chem.MolToSmiles(mol) for mol in mols]
    expected = [Chem.CanonSmiles(smi) for smi in smis]  # type: ignore[no-untyped-call]
    assert smiles_fp.clean(smis) == expected
