"""Configures the tests and benchmarks."""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

DROPIN = Path(__file__).parents[1] / "dropin" / "smiles_fp.py"
# these test the compiled package itself: its generated docs and its handling of interrupts.
COMPILED_ONLY = ("test_docs.py", "test_interrupt.py")

if os.environ.get("SMILES_FP_DROPIN"):  # pragma: no cover
    # runs the tests against the pure-Python dropin, which must give the same results.
    spec = importlib.util.spec_from_file_location("smiles_fp", DROPIN)
    assert spec is not None
    assert spec.loader is not None
    dropin = importlib.util.module_from_spec(spec)
    sys.modules["smiles_fp"] = dropin
    spec.loader.exec_module(dropin)
    collect_ignore = list(COMPILED_ONLY)

from smiles_fp import get_mols, get_morgan_fps  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from numpy.typing import NDArray
    from rdkit import Chem
    from rdkit.DataStructs import ExplicitBitVect

N_SMILES = 10_000
TEST_SIZES = (0, 1, 100)

BENCHMARK_SIZES = (2_500,)

# n_bits = 0 fails with RDKit
FP_SIZES = (1, 512)

FIXTURES = Path(__file__).parent / "fixtures"
TEST_SMILES_PATH = FIXTURES / "test.smiles"

if max(*TEST_SIZES, *BENCHMARK_SIZES) > N_SMILES:  # pragma: no cover
    raise ValueError("More SMILES requested for benchmark than processed")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:  # noqa: ARG001  # pragma: no cover
    """Skips what the dropin cannot do, e.g. settings that this RDKit hides from Python."""
    try:
        return (yield)
    except NotImplementedError as e:
        pytest.skip(str(e))


def zero_array(dims: int, value: int = 0) -> NDArray[np.float64]:
    return np.zeros(shape=tuple(value for _ in range(dims)), dtype=np.float64)


def fps_to_bits(fps: Sequence[ExplicitBitVect]) -> NDArray[np.uint8]:
    """Converts fingerprints to a 2D uint8 array with one bit per column."""
    if not fps:
        return np.zeros((0, 1), dtype=np.uint8)
    rows = [np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0") for fp in fps]  # type: ignore[no-untyped-call,unused-ignore]
    return np.array(rows, dtype=np.uint8)


def bits_to_packed(bits: NDArray[np.uint8]) -> NDArray[np.uint64]:
    """Packs a 2D uint8 array of bits into little-endian 64-bit blocks per row."""
    padded = np.pad(bits, ((0, 0), (0, -bits.shape[1] % 64)))
    return np.packbits(padded, axis=1, bitorder="little").view("<u8")


@cache
def read_smis() -> list[str]:
    smis: list[str] = []
    with TEST_SMILES_PATH.open() as f:
        for _ in range(N_SMILES):
            line = f.readline()
            if not line:
                raise ValueError("Not enough SMILES available")  # pragma: no cover
            smis.append(line)
    return smis


@pytest.fixture(params=TEST_SIZES)
def test_smis(request: pytest.FixtureRequest) -> list[str]:
    n_smis: int = request.param
    return read_smis()[:n_smis]


@cache
def get_cached_mols() -> dict[str, Chem.Mol]:
    smis = read_smis()
    mols = get_mols(smis)
    return dict(zip(smis, mols, strict=True))


@cache
def get_cached_fps(n_bits: int = 2048) -> dict[str, ExplicitBitVect]:
    mols = get_cached_mols()
    fps = get_morgan_fps(list(mols.values()), n_bits=n_bits)
    return dict(zip(mols, fps, strict=True))


@pytest.fixture(params=FP_SIZES)
def test_fps(test_smis: list[str], request: pytest.FixtureRequest) -> list[ExplicitBitVect]:
    n_bits: int = request.param
    fps = get_cached_fps(n_bits)
    return [fps[s] for s in test_smis]


@pytest.fixture(params=BENCHMARK_SIZES)
def benchmark_smis(request: pytest.FixtureRequest) -> list[str]:  # pragma: no cover
    n_smis: int = request.param
    return read_smis()[:n_smis]


@pytest.fixture
def benchmark_mols(benchmark_smis: list[str]) -> list[Chem.Mol]:  # pragma: no cover
    mols = get_cached_mols()
    return [mols[s] for s in benchmark_smis]


@pytest.fixture(params=BENCHMARK_SIZES)
def benchmark_fps(request: pytest.FixtureRequest) -> list[ExplicitBitVect]:  # pragma: no cover
    n_smis: int = request.param
    smis = read_smis()[:n_smis]
    fps = get_cached_fps()
    return [fps[s] for s in smis]
