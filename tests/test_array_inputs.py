"""Tests and benchmarks array inputs of the in-memory similarity functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, get_args

import conftest
import numpy as np
import pytest
import smiles_fp
from smiles_fp.helpers import FpOutput

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_benchmark.fixture import BenchmarkFixture
    from rdkit import Chem
    from rdkit.DataStructs import ExplicitBitVect

Agg: TypeAlias = Literal["mean", "max", "min"]
Layout: TypeAlias = Literal["fortran", "strided", "readonly", "big_endian"]
Func: TypeAlias = Literal["bulk", "bulk_topk", "internal"]
BulkFunc: TypeAlias = Literal["bulk", "bulk_topk"]

KINDS: tuple[FpOutput, ...] = tuple(kind for kind in get_args(FpOutput) if kind != "bitvect")
AGGS: tuple[Agg | None, ...] = (None, *get_args(Agg))
LAYOUTS: tuple[Layout, ...] = get_args(Layout)
TOP_K = 5
N_THREADS = 4


def as_kind(fps: list[ExplicitBitVect], kind: FpOutput) -> Any:
    if kind == "bitvect":  # pragma: no cover
        return fps
    bits = conftest.fps_to_bits(fps)
    return bits if kind == "unpacked" else conftest.bits_to_packed(bits)


def to_layout(array: np.ndarray[Any, Any], layout: Layout) -> np.ndarray[Any, Any]:
    if layout == "fortran":
        return np.asfortranarray(array)
    if layout == "strided":
        return np.repeat(array, 2, axis=0)[::2]
    if layout == "readonly":
        copy = array.copy()
        copy.setflags(write=False)
        return copy
    return array.astype(">u8")  # big endian, packed only


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("agg", AGGS)
def test_bulk_arrays_match_bit_vectors(
    test_fps: list[ExplicitBitVect], kind: FpOutput, agg: Agg | None
) -> None:
    arr = as_kind(test_fps, kind)
    expected = smiles_fp.bulk_tanimoto_parallel(test_fps, test_fps, N_THREADS, agg)
    np.testing.assert_array_equal(
        smiles_fp.bulk_tanimoto_parallel(arr, arr, N_THREADS, agg), expected
    )
    # mixed inputs
    np.testing.assert_array_equal(
        smiles_fp.bulk_tanimoto_parallel(test_fps, arr, N_THREADS, agg),
        expected,
    )
    np.testing.assert_array_equal(
        smiles_fp.bulk_tanimoto_parallel(arr, test_fps, N_THREADS, agg),
        expected,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_bulk_topk_arrays_match_bit_vectors(
    test_fps: list[ExplicitBitVect], kind: FpOutput
) -> None:
    arr = as_kind(test_fps, kind)
    idx, scores = smiles_fp.bulk_tanimoto_parallel_topk(test_fps, test_fps, TOP_K, N_THREADS)
    arr_idx, arr_scores = smiles_fp.bulk_tanimoto_parallel_topk(arr, arr, TOP_K, N_THREADS)
    np.testing.assert_array_equal(arr_idx, idx)
    np.testing.assert_array_equal(arr_scores, scores)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("agg", AGGS)
def test_internal_arrays_match_bit_vectors(
    test_fps: list[ExplicitBitVect], kind: FpOutput, agg: Agg | None
) -> None:
    arr = as_kind(test_fps, kind)
    expected = smiles_fp.internal_tanimoto_parallel(test_fps, N_THREADS, agg)
    np.testing.assert_array_equal(
        smiles_fp.internal_tanimoto_parallel(arr, N_THREADS, agg), expected
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n_bits", [1, 63, 64, 65, 2000, 2048])
def test_bulk_array_sizes_match_numpy(kind: FpOutput, n_bits: int) -> None:
    rng = np.random.default_rng(n_bits)
    left = rng.integers(0, 2, (20, n_bits), dtype=np.uint8)
    right = rng.integers(0, 2, (30, n_bits), dtype=np.uint8)
    inter = (left[:, None, :] & right[None, :, :]).sum(axis=2)
    union = (left[:, None, :] | right[None, :, :]).sum(axis=2)
    expected = np.where(union == 0, 0.0, inter / np.maximum(union, 1))

    if kind == "packed":
        left, right = conftest.bits_to_packed(left), conftest.bits_to_packed(right)
    np.testing.assert_allclose(smiles_fp.bulk_tanimoto_parallel(left, right, N_THREADS), expected)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("kind", KINDS)
def test_array_layouts_give_the_same_result(
    test_fps: list[ExplicitBitVect], kind: FpOutput, layout: Layout
) -> None:
    if layout == "big_endian" and kind == "unpacked":
        pytest.skip("byte order only exists for packed blocks")
    arr = as_kind(test_fps, kind)
    if layout == "big_endian":
        # a non-native byte order is not a packed uint64 array we can read in place.
        with pytest.raises(TypeError, match=r"dtype uint8 .* uint64"):
            smiles_fp.internal_tanimoto_parallel(to_layout(arr, layout))
        return
    laid_out = to_layout(arr, layout)
    np.testing.assert_array_equal(
        smiles_fp.internal_tanimoto_parallel(laid_out, N_THREADS),
        smiles_fp.internal_tanimoto_parallel(arr, N_THREADS),
    )


@pytest.mark.parametrize("func", get_args(Func))
@pytest.mark.parametrize(
    ("array", "error", "match"),
    [
        (np.zeros((3, 8), dtype=np.float64), TypeError, r"uint8 \(bits\) or uint64"),
        (np.zeros((3, 8), dtype=np.bool_), TypeError, r"uint8 \(bits\) or uint64"),
        (np.zeros((3, 8), dtype=np.uint32), TypeError, r"uint8 \(bits\) or uint64"),
        (np.zeros(8, dtype=np.uint8), TypeError, r"uint8 \(bits\) or uint64"),
        (np.zeros((2, 3, 8), dtype=np.uint8), TypeError, r"uint8 \(bits\) or uint64"),
        (np.full((3, 8), 2, dtype=np.uint8), ValueError, "only 0 and 1"),
        (np.zeros((3, 0), dtype=np.uint8), ValueError, "at least one bit"),
    ],
    ids=["float", "bool", "uint32", "1d", "3d", "value", "zero_bits"],
)
def test_invalid_arrays_raise(
    func: Func, array: np.ndarray[Any, Any], error: type[Exception], match: str
) -> None:
    caller: dict[Func, Callable[[], Any]] = {
        "bulk": lambda: smiles_fp.bulk_tanimoto_parallel(array, array),
        "bulk_topk": lambda: smiles_fp.bulk_tanimoto_parallel_topk(array, array),
        "internal": lambda: smiles_fp.internal_tanimoto_parallel(array),
    }
    with pytest.raises(error, match=match):
        caller[func]()


@pytest.mark.parametrize("func", get_args(BulkFunc))
@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.zeros((2, 512), dtype=np.uint8), np.zeros((2, 256), dtype=np.uint8)),
        (np.zeros((2, 8), dtype=np.uint64), np.zeros((2, 4), dtype=np.uint64)),
        (np.zeros((2, 512), dtype=np.uint8), np.zeros((2, 4), dtype=np.uint64)),
    ],
    ids=["bits", "blocks", "bits_and_blocks"],
)
def test_mismatched_array_lengths_raise(
    func: BulkFunc, left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]
) -> None:
    caller = (
        smiles_fp.bulk_tanimoto_parallel
        if func == "bulk"
        else smiles_fp.bulk_tanimoto_parallel_topk
    )
    with pytest.raises(ValueError, match="bit mismatch"):
        caller(left, right)


def test_array_and_bit_vector_of_different_lengths_raise() -> None:
    fps = list(conftest.get_cached_fps(512).values())[:3]
    with pytest.raises(ValueError, match="bit mismatch"):
        smiles_fp.bulk_tanimoto_parallel(fps, np.zeros((3, 256), dtype=np.uint8))


@pytest.mark.parametrize("kind", KINDS)
def test_empty_arrays(kind: FpOutput) -> None:
    empty: np.ndarray[Any, Any] = np.zeros((0, 8 if kind == "packed" else 512), dtype=np.uint8)
    if kind == "packed":
        empty = empty.astype(np.uint64)
    full = as_kind(list(conftest.get_cached_fps().values())[:3], kind)

    assert smiles_fp.bulk_tanimoto_parallel(empty, full).shape == (0, 3)
    assert smiles_fp.bulk_tanimoto_parallel(full, empty).shape == (3, 0)
    assert smiles_fp.bulk_tanimoto_parallel(empty, empty, agg="max").shape == (0,)
    idx, scores = smiles_fp.bulk_tanimoto_parallel_topk(empty, full)
    assert idx.shape == scores.shape == (0, 0)
    assert smiles_fp.internal_tanimoto_parallel(empty).shape == (0,)


@pytest.mark.parametrize("n_bits", [1, 63, 64, 65, 512, 2048])
def test_get_morgan_fps_arrays_match_bit_vectors(n_bits: int) -> None:
    mols = list(conftest.get_cached_mols().values())[:50]
    bits = conftest.fps_to_bits(list(smiles_fp.get_morgan_fps(mols, n_bits=n_bits)))

    unpacked = smiles_fp.get_morgan_fps(mols, n_bits=n_bits, output="unpacked")
    packed = smiles_fp.get_morgan_fps(mols, n_bits=n_bits, output="packed")

    assert unpacked.dtype == np.uint8
    assert packed.dtype == np.uint64
    np.testing.assert_array_equal(unpacked, bits)
    np.testing.assert_array_equal(packed, conftest.bits_to_packed(bits))


@pytest.mark.parametrize(("output", "shape"), [("unpacked", (0, 100)), ("packed", (0, 2))])
def test_get_morgan_fps_without_molecules_returns_empty_array(
    output: FpOutput, shape: tuple[int, int]
) -> None:
    assert np.shape(smiles_fp.get_morgan_fps([], n_bits=100, output=output)) == shape


def test_get_morgan_fps_unknown_output_raises() -> None:
    with pytest.raises(ValueError, match="output must be one of"):
        smiles_fp.get_morgan_fps([], output="array")  # type: ignore[call-overload]


@pytest.mark.parametrize("kind", ["bitvect", *KINDS])
@pytest.mark.parametrize("internal", [True, False], ids=["in", "ext"])
def test_benchmark_array_inputs(  # pragma: no cover
    kind: FpOutput,
    internal: bool,
    benchmark_fps: list[ExplicitBitVect],
    benchmark: BenchmarkFixture,
) -> None:
    data = as_kind(benchmark_fps, kind)
    if internal:
        result = benchmark(smiles_fp.internal_tanimoto_parallel, data, 8, "max")
        expected = smiles_fp.internal_tanimoto_parallel(benchmark_fps, 8, "max")
    else:
        result = benchmark(smiles_fp.bulk_tanimoto_parallel, data, data, 8, "max")
        expected = smiles_fp.bulk_tanimoto_parallel(benchmark_fps, benchmark_fps, 8, "max")

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("layout", ["fortran", "strided"])
@pytest.mark.parametrize("kind", KINDS)
def test_benchmark_array_layouts(  # pragma: no cover
    kind: FpOutput,
    layout: Layout,
    benchmark_fps: list[ExplicitBitVect],
    benchmark: BenchmarkFixture,
) -> None:
    arr = to_layout(as_kind(benchmark_fps, kind), layout)
    result = benchmark(smiles_fp.internal_tanimoto_parallel, arr, 8, "max")

    np.testing.assert_array_equal(
        result, smiles_fp.internal_tanimoto_parallel(benchmark_fps, 8, "max")
    )


@pytest.mark.parametrize("output", ["bitvect", "unpacked", "packed"])
def test_benchmark_get_morgan_fps_outputs(  # pragma: no cover
    output: FpOutput,
    benchmark_mols: list[Chem.Mol],
    benchmark: BenchmarkFixture,
) -> None:
    fps = benchmark(smiles_fp.get_morgan_fps, benchmark_mols, output=output)

    assert len(fps) == len(benchmark_mols)
