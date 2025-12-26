"""Test and benchmark the `smiles_fp` package."""

from __future__ import annotations

import os
import pickle
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rdFPGen
from tqdm import tqdm

import smiles_fp

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy.typing as npt
    from pytest_benchmark.fixture import BenchmarkFixture
    from rdkit.DataStructs import ExplicitBitVect

PARENT = Path(__file__).parent
TEST_SMILES_PATH = PARENT / "test.smiles"
SIZES = [1_000]  # [100, 1_000, 10_000, 50_000]
THREADS = [8]  # [1, 2, 4, 8, cpu_count()]
MAX_FPS = 100_000


def mol_from_smi(smi: str) -> Chem.Mol:
    """Convert a SMILES string to a molecule."""
    if mol := Chem.MolFromSmiles(smi):
        return mol
    raise ValueError(f"Could not convert {smi} to a molecule.")


@pytest.fixture(params=SIZES)
def test_smis(
    request: pytest.FixtureRequest,
) -> list[str]:
    """Read the first N SMILES strings from the TEST_SMILES_PATH file."""
    N: int = request.param  # noqa: N806
    smis: list[str] = []
    with TEST_SMILES_PATH.open() as f:
        for _ in range(N):
            line = f.readline()
            if not line:
                break
            smis.append(line)
    return smis


@pytest.fixture
def test_fps(
    test_smis: list[str],
) -> list[ExplicitBitVect]:
    """Generate fingerprints from the test SMILES strings."""
    with joblib.Parallel(
        n_jobs=os.cpu_count(),
        return_as="generator",
    ) as p:
        mol_gen: Generator[Chem.Mol] = p(joblib.delayed(mol_from_smi)(smi) for smi in test_smis)
        mols = list(tqdm(mol_gen))
    morgan_gen = rdFPGen.GetMorganGenerator(radius=2, fpSize=2048)
    fps: tuple[ExplicitBitVect, ...] = morgan_gen.GetFingerprints(mols, numThreads=cpu_count())
    return list(fps)


@pytest.fixture
def test_sims(
    test_fps: list[ExplicitBitVect],
) -> list[float]:
    """Calculate similarities between the first and second half of the test fingerprints."""
    half = len(test_fps) // 2
    nested_sims: list[list[float]] = [
        DataStructs.BulkTanimotoSimilarity(fp, test_fps[half:]) for fp in test_fps[:half]
    ]
    return [sim for sublist in nested_sims for sim in sublist]


def test_save_and_load_fingerprints_str(
    test_fps: list[ExplicitBitVect],
    tmp_path: Path,
) -> None:
    smiles_fp.save_fingerprints(
        test_fps,
        str(tmp_path / "test_fps.bin"),
    )
    loaded = smiles_fp.load_fingerprints(str(tmp_path / "test_fps.bin"))
    assert test_fps == loaded


def test_save_fingerprints(
    test_fps: list[ExplicitBitVect],
    tmp_path: Path,
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(
        smiles_fp.save_fingerprints,
        test_fps,
        tmp_path / "test_fps.bin",
    )
    loaded = smiles_fp.load_fingerprints(tmp_path / "test_fps.bin")
    assert test_fps == loaded


def test_load_fingerprints(
    test_fps: list[ExplicitBitVect],
    tmp_path: Path,
    benchmark: BenchmarkFixture,
) -> None:
    smiles_fp.save_fingerprints(
        test_fps,
        tmp_path / "test_fps.bin",
    )
    loaded = benchmark(
        smiles_fp.load_fingerprints,
        tmp_path / "test_fps.bin",
    )
    assert test_fps == loaded


def _pickle(
    fps: list[ExplicitBitVect],
    path: Path,
) -> None:
    with path.open("wb") as f:
        pickle.dump(fps, f)


def _unpickle(
    path: Path,
) -> list[ExplicitBitVect]:
    with path.open("rb") as f:
        return pickle.load(f)  # type: ignore[no-any-return] # noqa: S301


def test_pickle_fingerprints(
    test_fps: list[ExplicitBitVect], tmp_path: Path, benchmark: BenchmarkFixture
) -> None:
    benchmark(
        _pickle,
        test_fps,
        tmp_path / "test_fps.bin",
    )
    loaded = _unpickle(tmp_path / "test_fps.bin")
    assert test_fps == loaded


def test_unpickle_fingerprints(
    test_fps: list[ExplicitBitVect],
    tmp_path: Path,
    benchmark: BenchmarkFixture,
) -> None:
    _pickle(
        test_fps,
        tmp_path / "test_fps.bin",
    )
    loaded = benchmark(
        _unpickle,
        tmp_path / "test_fps.bin",
    )
    assert test_fps == loaded


def _single(
    fps1: list[ExplicitBitVect],
    fps2: list[ExplicitBitVect],
) -> list[float]:
    return [DataStructs.TanimotoSimilarity(fp1, fp2) for fp1 in fps1 for fp2 in fps2]


def _bulk(
    fps1: list[ExplicitBitVect],
    fps2: list[ExplicitBitVect],
) -> list[float]:
    return [s for fp in fps1 for s in DataStructs.BulkTanimotoSimilarity(fp, fps2)]


@pytest.mark.skip("Takes too long")
def test_single_tanimoto(
    test_fps: list[ExplicitBitVect],
    test_sims: list[float],
    benchmark: BenchmarkFixture,
) -> None:
    half = len(test_fps) // 2
    sims = benchmark(
        _single,
        test_fps[:half],
        test_fps[half:],
    )

    np.testing.assert_array_equal(sims, test_sims)


def test_bulk_tanimoto(
    test_fps: list[ExplicitBitVect],
    test_sims: list[float],
    benchmark: BenchmarkFixture,
) -> None:
    half = len(test_fps) // 2
    sims = benchmark(
        _bulk,
        test_fps[:half],
        test_fps[half:],
    )

    np.testing.assert_array_equal(sims, test_sims)


@pytest.mark.parametrize("num_threads", THREADS)
def test_bulk_tanimoto_parallel(
    num_threads: int,
    test_fps: list[ExplicitBitVect],
    test_sims: list[float],
    benchmark: BenchmarkFixture,
) -> None:
    if len(test_fps) > MAX_FPS:
        pytest.skip("bulk_tanimo_parallel is too memory intensive, skipping")

    half = len(test_fps) // 2
    sims = benchmark(
        smiles_fp.bulk_tanimoto_parallel,
        test_fps[:half],
        test_fps[half:],
        num_threads,
    )

    np.testing.assert_array_equal(sims, test_sims)


def _bulk_mmap(
    fps1: list[ExplicitBitVect],
    fps2: list[ExplicitBitVect],
    num_threads: int,
    tmp_path: Path,
    pathlike: type[str | Path] = Path,
) -> npt.NDArray[np.float64]:
    smiles_fp.save_fingerprints(
        fps1,
        pathlike(tmp_path / "test_fps_first.bin"),
    )
    smiles_fp.save_fingerprints(
        fps2,
        pathlike(tmp_path / "test_fps_second.bin"),
    )

    return smiles_fp.bulk_tanimoto_mmap(
        pathlike(tmp_path / "test_fps_first.bin"),
        pathlike(tmp_path / "test_fps_second.bin"),
        num_threads=num_threads,
    )


def test_bulk_tanimoto_mmap_str(
    test_fps: list[ExplicitBitVect],
    test_sims: list[float],
    tmp_path: Path,
) -> None:
    half = len(test_fps) // 2
    sims = _bulk_mmap(
        test_fps[:half],
        test_fps[half:],
        8,
        tmp_path,
        str,
    )
    np.testing.assert_array_equal(sims, test_sims)


@pytest.mark.parametrize("num_threads", THREADS)
def test_bulk_tanimoto_mmap(
    num_threads: int,
    test_fps: list[ExplicitBitVect],
    test_sims: list[float],
    tmp_path: Path,
    benchmark: BenchmarkFixture,
) -> None:
    half = len(test_fps) // 2
    sims = benchmark(
        _bulk_mmap,
        test_fps[:half],
        test_fps[half:],
        num_threads,
        tmp_path,
    )

    np.testing.assert_array_equal(sims, test_sims)
