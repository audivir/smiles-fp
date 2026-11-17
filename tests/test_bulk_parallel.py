"""Tests and benchmarks parallel similarity calculation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import conftest
import numpy as np
import pytest
import smiles_fp
from rdkit import DataStructs

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray
    from pytest_benchmark.fixture import BenchmarkFixture
    from rdkit.DataStructs import ExplicitBitVect

THREADS = (1, 4, 8)
MAX_FPS = 100_000


def get_sims(fps: list[ExplicitBitVect]) -> NDArray[np.float64]:
    if len(fps) == 0:
        return conftest.zero_array(2)
    return np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])


@pytest.fixture
def test_sims(
    test_fps: list[ExplicitBitVect],
) -> NDArray[np.float64]:
    """Calculates similarities between the first and second half of the test fingerprints."""
    return get_sims(test_fps)


@pytest.fixture
def benchmark_sims(  # pragma: no cover
    benchmark_fps: list[ExplicitBitVect],
) -> NDArray[np.float64]:
    """Calculates similarities between the first and second half of the test fingerprints."""
    return get_sims(benchmark_fps)


def func_caller(  # noqa: PLR0913,PLR0917
    fps: list[ExplicitBitVect],
    n_threads: int = 8,
    k: int | None = None,
    agg: Literal["mean", "max", "min", "full"] | None = None,
    db_offset: int | None = None,
    db_limit: int | None = None,
    typ: type[str | Path] = Path,
    tmp_path: Path | None = None,
    mmap: bool = False,
    internal: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.uint32], NDArray[np.float64]]:
    kwargs: dict[str, Any] = {}

    if k is not None and internal:  # pragma: no cover
        raise ValueError("k and internal are mutually exclusive")
    kwargs["n_threads"] = n_threads

    for key, val in (("agg", agg), ("db_offset", db_offset), ("db_limit", db_limit)):
        if val is None:
            continue
        kwargs[key] = val

    if mmap:
        if not tmp_path:  # pragma: no cover
            raise ValueError("need tmp path for mmap")
        path = typ(tmp_path / "test_fps.bin")
        smiles_fp.save_fingerprints(fps, path)

        if k is not None:
            return smiles_fp.bulk_tanimoto_mmap_topk(path, path, k, **kwargs)
        if internal:
            return smiles_fp.internal_tanimoto_mmap(path, **kwargs)
        return smiles_fp.bulk_tanimoto_mmap(path, path, **kwargs)

    if k is not None:
        return smiles_fp.bulk_tanimoto_parallel_topk(fps, fps, k, **kwargs)
    if internal:
        return smiles_fp.internal_tanimoto_parallel(fps, **kwargs)
    return smiles_fp.bulk_tanimoto_parallel(fps, fps, **kwargs)


def rdkit_single(  # pragma: no cover
    fps1: list[ExplicitBitVect],
    fps2: list[ExplicitBitVect],
) -> NDArray[np.float64]:
    return np.array([[DataStructs.TanimotoSimilarity(fp1, fp2) for fp2 in fps2] for fp1 in fps1])


def rdkit_bulk(  # pragma: no cover
    fps1: list[ExplicitBitVect],
    fps2: list[ExplicitBitVect],
) -> NDArray[np.float64]:
    return np.array([DataStructs.BulkTanimotoSimilarity(fp1, fps2) for fp1 in fps1])


@pytest.mark.parametrize("n_threads", THREADS)
@pytest.mark.parametrize("mmap", [True, False], ids=["mmap", "direct"])
def test_bulk_tanimoto_parallel(
    test_fps: list[ExplicitBitVect],
    n_threads: int,
    mmap: bool,
    test_sims: NDArray[np.float64],
    tmp_path: Path,
) -> None:
    sims = func_caller(test_fps, n_threads, tmp_path=tmp_path, mmap=mmap)
    np.testing.assert_array_equal(sims, test_sims)


@pytest.mark.parametrize("strpath", [str, Path])
def test_bulk_tanimoto_mmap_input(
    strpath: type[str | Path],
    tmp_path: Path,
) -> None:
    test_fps = list(conftest.get_cached_fps().values())[:64]
    test_sims = get_sims(test_fps)
    sims = func_caller(test_fps, typ=strpath, tmp_path=tmp_path, mmap=True)
    np.testing.assert_array_equal(sims, test_sims)


@pytest.mark.parametrize("func", [rdkit_single, rdkit_bulk], ids=["single", "bulk"])
def test_benchmark_rdkit_tanimoto(  # pragma: no cover
    func: Callable[[Sequence[ExplicitBitVect], Sequence[ExplicitBitVect]], NDArray[np.float64]],
    benchmark_fps: list[ExplicitBitVect],
    benchmark_sims: NDArray[np.float64],
    benchmark: BenchmarkFixture,
) -> None:
    if func == rdkit_single:
        pytest.skip(reason="Takes too long")

    sims = benchmark(
        func,
        benchmark_fps,
        benchmark_fps,
    )

    np.testing.assert_array_equal(sims, benchmark_sims)


@pytest.mark.parametrize("mmap", [True, False], ids=["mmap", "direct"])
@pytest.mark.parametrize("internal", [True, False], ids=["in", "ext"])
@pytest.mark.parametrize("n_threads", THREADS)
def test_benchmark_bulk_tanimoto_parallel(  # noqa: PLR0913,PLR0917 # pragma: no cover
    mmap: bool,
    internal: bool,
    n_threads: int,
    benchmark_fps: list[ExplicitBitVect],
    benchmark_sims: NDArray[np.float64],
    benchmark: BenchmarkFixture,
    tmp_path: Path,
) -> None:
    if len(benchmark_fps) > MAX_FPS:
        pytest.skip("bulk_tanimoto_parallel is too memory intensive, skipping")

    sims = benchmark(
        func_caller,
        fps=benchmark_fps,
        n_threads=n_threads,
        tmp_path=tmp_path,
        mmap=mmap,
        internal=internal,
    )

    if internal:
        sims = smiles_fp.to_matrix(sims, len(benchmark_fps))
        np.fill_diagonal(sims, 1.0)

    np.testing.assert_array_equal(sims, benchmark_sims)


@pytest.mark.parametrize("n", [0, 1, 100])
@pytest.mark.parametrize("mmap", [True, False], ids=["mmap", "direct"])
def test_internal_tanimoto(n: int, mmap: bool, tmp_path: Path) -> None:
    test_fps = list(conftest.get_cached_fps().values())[:n]
    test_sims = np.array(get_sims(test_fps))
    np.fill_diagonal(test_sims, 0.0)

    sims = func_caller(test_fps, tmp_path=tmp_path, mmap=mmap, internal=True)
    if isinstance(sims, tuple):  # pragma: no cover
        raise TypeError("top k result instead of similarity matrix")
    sims = smiles_fp.to_matrix(sims, n)

    np.testing.assert_allclose(sims, test_sims)


@pytest.mark.parametrize("n", [0, 1, 100])
@pytest.mark.parametrize("agg", ["mean", "max", "min"])
@pytest.mark.parametrize("mmap", [True, False], ids=["mmap", "direct"])
def test_internal_tanimoto_agg(
    n: int,
    agg: Literal["mean", "max", "min"],
    mmap: bool,
    tmp_path: Path,
) -> None:
    test_fps = list(conftest.get_cached_fps().values())[:n]

    vector = func_caller(test_fps, tmp_path=tmp_path, mmap=mmap, internal=True)
    if isinstance(vector, tuple):  # pragma: no cover
        raise TypeError("top k result instead of similarity matrix")
    full_matrix = smiles_fp.to_matrix(vector, n)

    rust_agg = func_caller(test_fps, agg=agg, tmp_path=tmp_path, mmap=mmap, internal=True)

    if full_matrix.size == 0:
        np.testing.assert_array_equal(rust_agg, conftest.zero_array(1))
    elif agg == "mean":
        if n == 1:
            np.testing.assert_array_equal(rust_agg, conftest.zero_array(1, 1))
        else:
            np.testing.assert_allclose(
                rust_agg, full_matrix.sum(axis=1) / (full_matrix.shape[1] - 1)
            )
    elif agg == "max":
        np.testing.assert_array_equal(rust_agg, full_matrix.max(axis=1))
    elif agg == "min":
        if n == 1:
            np.testing.assert_array_equal(rust_agg, conftest.zero_array(1, 1))
        else:
            np.fill_diagonal(full_matrix, 1.0)
            np.testing.assert_array_equal(rust_agg, full_matrix.min(axis=1))
    else:  # pragma: no cover
        raise RuntimeError("unexpected")


@pytest.mark.parametrize("agg", ["mean", "max", "min"])
@pytest.mark.parametrize("mmap", [True, False], ids=["mmap", "direct"])
def test_bulk_tanimoto_agg(
    agg: Literal["mean", "max", "min"],
    mmap: bool,
    test_fps: list[ExplicitBitVect],
    tmp_path: Path,
) -> None:
    full_matrix = func_caller(test_fps, tmp_path=tmp_path, mmap=mmap)

    if isinstance(full_matrix, tuple):  # pragma: no cover
        raise TypeError("top k result instead of similarity matrix")

    rust_agg = func_caller(test_fps, agg=agg, tmp_path=tmp_path, mmap=mmap)

    if full_matrix.size == 0:
        np.testing.assert_array_equal(rust_agg, conftest.zero_array(1))
    elif agg == "mean":
        np.testing.assert_allclose(rust_agg, full_matrix.mean(axis=1))
    elif agg == "max":
        np.testing.assert_array_equal(rust_agg, full_matrix.max(axis=1))
    elif agg == "min":
        np.testing.assert_array_equal(rust_agg, full_matrix.min(axis=1))
    else:  # pragma: no cover
        raise RuntimeError("unexpected")


@pytest.mark.parametrize("k", [1, 5], ids=["k1", "k5"])
@pytest.mark.parametrize("mmap", [True, False], ids=["mmap", "direct"])
def test_top_k_search(k: int, mmap: bool, test_fps: list[ExplicitBitVect], tmp_path: Path) -> None:
    full_matrix = func_caller(test_fps, tmp_path=tmp_path, mmap=mmap)
    idx, scores = func_caller(test_fps, k=k, tmp_path=tmp_path, mmap=mmap)

    for i in range(len(full_matrix)):
        # Numpy argsort (ascending), take last K, reverse to descending
        expected_idx = np.argsort(full_matrix[i])[-k:][::-1]
        expected_scores = full_matrix[i][expected_idx]

        # 1. Verify the mathematical values of the scores are exactly correct
        np.testing.assert_allclose(np.sort(scores[i]), np.sort(expected_scores))

        # 2. Verify the indices are correct by indexing back into the full matrix
        # If the index is wrong, this fetched score will not match the returned score.
        actual_scores_at_idx = full_matrix[i][idx[i]]
        np.testing.assert_allclose(actual_scores_at_idx, scores[i])

        # 3. Guardrail: Verify the heap didn't return duplicate indices
        assert len(set(idx[i])) == len(idx[i]), f"Duplicate indices found in Top-K for query {i}"


def test_windowing_offsets(
    test_fps: list[ExplicitBitVect],
    tmp_path: Path,
) -> None:
    full_matrix = func_caller(test_fps, tmp_path=tmp_path, mmap=True)

    if isinstance(full_matrix, tuple):  # pragma: no cover
        raise TypeError("top k result instead of similarity matrix")

    total_db_fps = full_matrix.shape[1]
    chunk_size = max(1, total_db_fps // 2)

    # Process just the second chunk
    chunked_matrix = func_caller(
        test_fps,
        db_offset=chunk_size,
        db_limit=chunk_size,
        tmp_path=tmp_path,
        mmap=True,
    )

    # Slice the full matrix natively in numpy to simulate the chunk
    expected_slice = full_matrix[:, chunk_size : chunk_size + chunk_size]

    np.testing.assert_array_equal(chunked_matrix, expected_slice)


def bit_vect(n_bits: int) -> ExplicitBitVect:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as rdFPGen

    return rdFPGen.GetMorganGenerator(radius=2, fpSize=n_bits).GetFingerprint(  # type: ignore[no-any-return]
        Chem.MolFromSmiles("CCO")
    )


@pytest.mark.parametrize("func", ["bulk", "bulk_topk", "internal"])
def test_parallel_invalid_input_raises_instead_of_empty_result(func: str) -> None:
    mixed = [bit_vect(512), bit_vect(1024)]
    good = [bit_vect(512)]
    with pytest.raises(ValueError, match="Inconsistent fingerprint lengths"):  # noqa: PT012
        if func == "bulk":
            smiles_fp.bulk_tanimoto_parallel(mixed, good)
        elif func == "bulk_topk":
            smiles_fp.bulk_tanimoto_parallel_topk(good, mixed, 1)
        else:
            smiles_fp.internal_tanimoto_parallel(mixed)
    with pytest.raises(TypeError):
        smiles_fp.bulk_tanimoto_parallel(["nope"], good)  # type: ignore[list-item]


@pytest.mark.parametrize("func", ["bulk", "bulk_topk", "internal"])
def test_mmap_truncated_file_raises(func: str, tmp_path: Path) -> None:
    path = tmp_path / "fps.bin"
    smiles_fp.save_fingerprints([bit_vect(512), bit_vect(512)], path)
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match="truncated"):  # noqa: PT012
        if func == "bulk":
            smiles_fp.bulk_tanimoto_mmap(path, path)
        elif func == "bulk_topk":
            smiles_fp.bulk_tanimoto_mmap_topk(path, path, 1)
        else:
            smiles_fp.internal_tanimoto_mmap(path)
