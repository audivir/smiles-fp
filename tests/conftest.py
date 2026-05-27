"""Configure the tests and benchmarks."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from smiles_fp_rs import get_mols, get_morgan_fps

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from rdkit import Chem
    from rdkit.DataStructs import ExplicitBitVect

N_SMILES = 10_000
TEST_SIZES = (0, 1, 7, 16, 100)
TEST_SIZES = (0, 1, 100)

BENCHMARK_SIZES = (2_500,)

# n_bits = 0 fails with RDKit
FP_SIZES = (1, 31, 32, 64, 127, 512)
FP_SIZES = (1, 512)

PARENT = Path(__file__).parent
TEST_SMILES_PATH = PARENT / "test.smiles"

if max(*TEST_SIZES, *BENCHMARK_SIZES) > N_SMILES:  # pragma: no cover
    raise ValueError("More SMILES requested for benchmark than processed")


def zero_array(dims: int, value: int = 0) -> NDArray[np.float64]:
    return np.zeros(shape=tuple(value for _ in range(dims)), dtype=np.float64)


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
def benchmark_fps(request: pytest.FixtureRequest) -> list[ExplicitBitVect]:  # pragma: no cover
    n_smis: int = request.param
    smis = read_smis()[:n_smis]
    fps = get_cached_fps()
    return [fps[s] for s in smis]
