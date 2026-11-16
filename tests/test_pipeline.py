"""Tests and benchmarks the fused SMILES to fingerprint file pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import conftest
import pytest
import smiles_fp

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture


def separate_pipeline(
    smis: list[str], path: Path, radius: int = 2, n_bits: int = 2048
) -> None:  # pragma: no cover
    mols = smiles_fp.get_mols(smis, n_jobs=8)
    fps = smiles_fp.get_morgan_fps(mols, radius=radius, n_bits=n_bits, n_threads=8)
    smiles_fp.save_fingerprints(fps, path)


@pytest.mark.parametrize(("radius", "n_bits"), [(2, 2048), (0, 64), (1, 100), (3, 1), (4, 4097)])
@pytest.mark.parametrize("n_threads", [1, 3, 12])
def test_save_morgan_fps_from_smiles_is_byte_identical_to_separate_steps(
    tmp_path: Path, radius: int, n_bits: int, n_threads: int
) -> None:
    smis = conftest.read_smis()[:200]
    separate_pipeline(smis, tmp_path / "separate.fp", radius, n_bits)
    smiles_fp.save_morgan_fps_from_smiles(
        smis, tmp_path / "fused.fp", radius, n_bits, n_threads=n_threads
    )

    assert (tmp_path / "fused.fp").read_bytes() == (tmp_path / "separate.fp").read_bytes()


def test_save_morgan_fps_from_smiles_loads_back_as_bit_vectors(tmp_path: Path) -> None:
    smis = conftest.read_smis()[:50]
    smiles_fp.save_morgan_fps_from_smiles(smis, str(tmp_path / "fps.fp"))

    loaded = smiles_fp.load_fingerprints(tmp_path / "fps.fp")
    expected = smiles_fp.get_morgan_fps(smiles_fp.get_mols(smis))
    assert [list(fp.GetOnBits()) for fp in loaded] == [list(fp.GetOnBits()) for fp in expected]


def test_save_morgan_fps_from_smiles_invalid_smiles_raises_and_leaves_no_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fps.fp"
    with pytest.raises(ValueError, match="Could not convert SMILES 1: not a smiles"):
        smiles_fp.save_morgan_fps_from_smiles(["CCO", "not a smiles", "c1ccccc1"], path)
    assert not path.exists()


def test_save_morgan_fps_from_smiles_without_smiles_writes_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.fp"
    smiles_fp.save_morgan_fps_from_smiles([], path)
    assert smiles_fp.load_fingerprints(path) == []


def test_save_morgan_fps_from_smiles_zero_bits_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_bits must be positive"):
        smiles_fp.save_morgan_fps_from_smiles(["CCO"], tmp_path / "fps.fp", n_bits=0)


def test_save_morgan_fps_from_smiles_unwritable_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        smiles_fp.save_morgan_fps_from_smiles(["CCO"], tmp_path / "missing_dir" / "fps.fp")


def test_benchmark_separate_pipeline(  # pragma: no cover
    benchmark_smis: list[str], tmp_path: Path, benchmark: BenchmarkFixture
) -> None:
    benchmark(separate_pipeline, benchmark_smis, tmp_path / "separate.fp")


@pytest.mark.parametrize("n_threads", [1, 4, 8])
def test_benchmark_fused_pipeline(  # pragma: no cover
    n_threads: int, benchmark_smis: list[str], tmp_path: Path, benchmark: BenchmarkFixture
) -> None:
    separate_pipeline(benchmark_smis, tmp_path / "separate.fp")

    benchmark(
        smiles_fp.save_morgan_fps_from_smiles,
        benchmark_smis,
        tmp_path / "fused.fp",
        n_threads=n_threads,
    )

    assert (tmp_path / "fused.fp").read_bytes() == (tmp_path / "separate.fp").read_bytes()
