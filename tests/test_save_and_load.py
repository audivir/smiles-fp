"""Tests and benchmarks saving and loading."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import conftest
import pytest
import smiles_fp

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from rdkit.DataStructs import ExplicitBitVect


def _func_caller(
    fps: list[ExplicitBitVect],
    tmp_path: Path,
    mode: Literal["load", "save"],
    typ: type[str | Path] = Path,
) -> list[ExplicitBitVect]:
    path = typ(tmp_path / "test_fps.bin")

    if mode == "load":
        return smiles_fp.load_fingerprints(path)
    smiles_fp.save_fingerprints(fps, path)
    return fps


@pytest.mark.parametrize("strpath", [str, Path])
def test_save_and_load_fingerprints_input(
    strpath: type[str | Path],
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


def test_benchmark_save_fingerprints(  # pragma: no cover
    benchmark_fps: list[ExplicitBitVect],
    tmp_path: Path,
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(_func_caller, tmp_path=tmp_path, fps=benchmark_fps, mode="save")
    loaded = _func_caller(benchmark_fps, tmp_path, mode="load")
    assert benchmark_fps == loaded


def test_benchmark_load_fingerprints(  # pragma: no cover
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


def _fp(smi: str, n_bits: int) -> ExplicitBitVect:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as rdFPGen

    return rdFPGen.GetMorganGenerator(radius=2, fpSize=n_bits).GetFingerprint(  # type: ignore[no-any-return]
        Chem.MolFromSmiles(smi)
    )


def test_save_fingerprints_unwritable_path_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Could not open"):
        smiles_fp.save_fingerprints([_fp("CCO", 512)], tmp_path / "missing_dir" / "fps.bin")


def test_save_fingerprints_mixed_lengths_raises_and_leaves_no_file(tmp_path: Path) -> None:
    path = tmp_path / "fps.bin"
    with pytest.raises(ValueError, match="Inconsistent fingerprint lengths"):
        smiles_fp.save_fingerprints([_fp("CCO", 512), _fp("CCO", 1024)], path)
    assert not path.exists()


def test_save_fingerprints_wrong_item_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "fps.bin"
    with pytest.raises(TypeError):
        smiles_fp.save_fingerprints([_fp("CCO", 512), "not a fingerprint"], path)  # type: ignore[list-item]
    assert not path.exists()


def test_load_fingerprints_truncated_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "fps.bin"
    smiles_fp.save_fingerprints([_fp("CCO", 512), _fp("CCN", 512)], path)
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match="truncated"):
        smiles_fp.load_fingerprints(path)


def test_load_fingerprints_huge_header_is_rejected_before_allocating(tmp_path: Path) -> None:
    path = tmp_path / "fps.bin"
    path.write_bytes((2**32 - 1).to_bytes(4, "little") + (8192).to_bytes(4, "little"))
    with pytest.raises(ValueError, match="truncated"):
        smiles_fp.load_fingerprints(path)
