"""Test and benchmark saving and loading."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import conftest
import pytest
import smiles_fp_rs

if TYPE_CHECKING:
    from pytest_benchmark import BenchmarkFixture
    from rdkit.DataStructs import ExplicitBitVect


def _func_caller(
    fps: list[ExplicitBitVect],
    tmp_path: Path,
    mode: Literal["load", "save"],
    typ: type[str | Path] = Path,
) -> list[ExplicitBitVect]:
    path = typ(tmp_path / "test_fps.bin")

    if mode == "load":
        return smiles_fp_rs.load_fingerprints(path)
    smiles_fp_rs.save_fingerprints(fps, path)
    return fps


@pytest.mark.parametrize("strpath", [str, Path])
@pytest.mark.parametrize("mod", ["cpp", "rust"])
def test_save_and_load_fingerprints_input(
    strpath: type[str | Path],
    mod: Literal["cpp", "rust"],
    tmp_path: Path,
) -> None:
    test_fps = list(conftest.get_cached_fps().values())[:64]
    _func_caller(test_fps, tmp_path, mode="save", typ=strpath)
    loaded = _func_caller(test_fps, tmp_path, mode="load", typ=strpath)
    assert test_fps == loaded


def test_save_and_load_fingerprints(
    test_fps: list[ExplicitBitVect],
    tmp_path: Path,
) -> None:
    _func_caller(test_fps, tmp_path, mode="save")
    loaded = _func_caller(test_fps, tmp_path, mode="load")
    assert test_fps == loaded


@pytest.mark.parametrize("mod", ["cpp", "rust"])
def test_benchmark_save_fingerprints(  # pragma: no cover
    mod: Literal["cpp", "rust"],
    benchmark_fps: list[ExplicitBitVect],
    tmp_path: Path,
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(_func_caller, tmp_path=tmp_path, fps=benchmark_fps, mode="save")
    loaded = _func_caller(benchmark_fps, tmp_path, mode="load")
    assert benchmark_fps == loaded


@pytest.mark.parametrize("mod", ["cpp", "rust"])
def test_benchmark_load_fingerprints(  # pragma: no cover
    mod: Literal["cpp", "rust"],
    benchmark_fps: list[ExplicitBitVect],
    tmp_path: Path,
    benchmark: BenchmarkFixture,
) -> None:
    _func_caller(benchmark_fps, tmp_path, mode="save")
    loaded = benchmark(_func_caller, tmp_path=tmp_path, fps=benchmark_fps, mode="load")
    assert benchmark_fps == loaded


def _pickle(  # pragma: no cover
    fps: list[ExplicitBitVect],
    path: Path,
) -> None:
    with path.open("wb") as f:
        pickle.dump(fps, f)


def _unpickle(  # pragma: no cover
    path: Path,
) -> list[ExplicitBitVect]:
    with path.open("rb") as f:
        return pickle.load(f)  # type: ignore[no-any-return]


def test_benchmark_pickle_fingerprints(  # pragma: no cover
    benchmark_fps: list[ExplicitBitVect], tmp_path: Path, benchmark: BenchmarkFixture
) -> None:
    path = tmp_path / "test_fps.bin"
    benchmark(_pickle, benchmark_fps, path)
    loaded = _unpickle(path)
    assert benchmark_fps == loaded


def test_benchmark_unpickle_fingerprints(  # pragma: no cover
    benchmark_fps: list[ExplicitBitVect],
    tmp_path: Path,
    benchmark: BenchmarkFixture,
) -> None:
    path = tmp_path / "test_fps.bin"
    _pickle(benchmark_fps, path)
    loaded = benchmark(_unpickle, path)
    assert benchmark_fps == loaded
