"""Tests and benchmarks saving and loading."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import conftest
import numpy as np
import pytest
import smiles_fp

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from rdkit.DataStructs import ExplicitBitVect
    from smiles_fp.helpers import FpOutput


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


def bit_array(n_fps: int, n_bits: int, seed: int = 0) -> np.ndarray[Any, Any]:
    return np.random.default_rng(seed).integers(0, 2, size=(n_fps, n_bits), dtype=np.uint8)


@pytest.mark.parametrize("n_bits", [1, 63, 64, 65, 100, 2048])
def test_array_roundtrip_matches_bit_vectors(tmp_path: Path, n_bits: int) -> None:
    bits = bit_array(37, n_bits)
    path = tmp_path / "array.fp"
    smiles_fp.save_fingerprints(bits, path)

    loaded = smiles_fp.load_fingerprints(path)
    assert [list(fp.GetOnBits()) for fp in loaded] == [list(np.flatnonzero(row)) for row in bits]
    np.testing.assert_array_equal(smiles_fp.load_fingerprints(path, output="unpacked"), bits)


def test_array_and_bit_vector_files_are_identical(tmp_path: Path) -> None:
    fps = [_fp(smi, 2048) for smi in conftest.read_smis()[:50]]
    bit_strings = [fp.ToBitString() for fp in fps]  # type: ignore[no-untyped-call,unused-ignore]
    bits = np.array([list(bit_string) for bit_string in bit_strings], dtype=np.uint8)
    smiles_fp.save_fingerprints(fps, tmp_path / "fps.fp")
    smiles_fp.save_fingerprints(bits, tmp_path / "array.fp")

    assert (tmp_path / "fps.fp").read_bytes() == (tmp_path / "array.fp").read_bytes()
    np.testing.assert_array_equal(
        smiles_fp.load_fingerprints(tmp_path / "fps.fp", output="unpacked"), bits
    )


def test_array_save_accepts_non_contiguous_arrays(tmp_path: Path) -> None:
    view = bit_array(40, 200)[::2, ::-1]
    smiles_fp.save_fingerprints(view, tmp_path / "view.fp")
    np.testing.assert_array_equal(
        smiles_fp.load_fingerprints(tmp_path / "view.fp", output="unpacked"), view
    )


def test_array_save_without_fingerprints_writes_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.fp"
    smiles_fp.save_fingerprints(np.zeros((0, 128), dtype=np.uint8), path)
    assert smiles_fp.load_fingerprints(path) == []
    assert smiles_fp.load_fingerprints(path, output="unpacked").shape == (0, 0)


@pytest.mark.parametrize(
    ("bits", "error", "match"),
    [
        (np.full((3, 8), 2, dtype=np.uint8), ValueError, "only contain the values 0 and 1"),
        (np.zeros((3, 0), dtype=np.uint8), ValueError, "at least one bit"),
        (np.zeros((3, 8), dtype=np.bool_), TypeError, "2D with dtype uint8"),
        (np.zeros((3, 8), dtype=np.float64), TypeError, "2D with dtype uint8"),
        (np.zeros(8, dtype=np.uint8), TypeError, "2D with dtype uint8"),
        (np.zeros((2, 3, 8), dtype=np.uint8), TypeError, "2D with dtype uint8"),
    ],
    ids=["value", "zero_bits", "bool", "float", "1d", "3d"],
)
def test_array_save_invalid_input_raises_and_leaves_no_file(
    tmp_path: Path, bits: np.ndarray[Any, Any], error: type[Exception], match: str
) -> None:
    path = tmp_path / "fps.fp"
    with pytest.raises(error, match=match):
        smiles_fp.save_fingerprints(bits, path)
    assert not path.exists()


def test_array_save_unwritable_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        smiles_fp.save_fingerprints(bit_array(2, 8), tmp_path / "missing_dir" / "fps.fp")


def test_array_save_too_many_fingerprints_raises(tmp_path: Path) -> None:
    # a broadcast view needs no memory, so the size limit is checkable.
    too_many = np.broadcast_to(np.uint8(0), (2**32, 1))
    with pytest.raises(ValueError, match=r"max 2\^32-1"):
        smiles_fp.save_fingerprints(too_many, tmp_path / "fps.fp")


@pytest.mark.parametrize("n_bits", [1, 63, 64, 65, 100, 2048])
def test_packed_save_matches_unpacked_file(tmp_path: Path, n_bits: int) -> None:
    bits = bit_array(37, n_bits)
    smiles_fp.save_fingerprints(bits, tmp_path / "bits.fp")
    smiles_fp.save_fingerprints(
        conftest.bits_to_packed(bits), tmp_path / "packed.fp", n_bits=n_bits
    )

    assert (tmp_path / "bits.fp").read_bytes() == (tmp_path / "packed.fp").read_bytes()
    np.testing.assert_array_equal(
        smiles_fp.load_fingerprints(tmp_path / "packed.fp", output="unpacked"), bits
    )


@pytest.mark.parametrize("layout", ["fortran", "big_endian", "strided"])
def test_packed_save_accepts_other_layouts(
    tmp_path: Path, layout: Literal["fortran", "big_endian", "strided"]
) -> None:
    bits = bit_array(40, 200)
    packed = conftest.bits_to_packed(bits)
    if layout == "fortran":
        packed = np.asfortranarray(packed)
    elif layout == "big_endian":
        packed = packed.astype(">u8")
    else:
        packed = np.repeat(packed, 2, axis=0)[::2]
    smiles_fp.save_fingerprints(packed, tmp_path / "packed.fp", n_bits=200)
    np.testing.assert_array_equal(
        smiles_fp.load_fingerprints(tmp_path / "packed.fp", output="unpacked"), bits
    )


def test_packed_save_without_fingerprints_writes_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.fp"
    smiles_fp.save_fingerprints(np.zeros((0, 2), dtype=np.uint64), path, n_bits=128)
    assert path.read_bytes() == bytes(8)


def test_packed_save_ignores_padding_of_a_bit_vector_equivalent(tmp_path: Path) -> None:
    fps = [_fp(smi, 2048) for smi in conftest.read_smis()[:20]]
    path = tmp_path / "fps.fp"
    smiles_fp.save_fingerprints(fps, path)
    blocks = np.fromfile(path, dtype="<u8", offset=8).reshape(len(fps), -1)
    smiles_fp.save_fingerprints(blocks, tmp_path / "packed.fp", n_bits=2048)
    assert path.read_bytes() == (tmp_path / "packed.fp").read_bytes()


@pytest.mark.parametrize(
    ("array", "n_bits", "match"),
    [
        (np.zeros((3, 2), dtype=np.uint64), None, "n_bits is required"),
        (np.zeros((3, 2), dtype=np.uint64), 0, "at least one bit"),
        (np.zeros((3, 2), dtype=np.uint64), 64, "64 bits need 1 blocks, but the array has 2"),
        (np.zeros((3, 2), dtype=np.uint64), 128 + 64, "bits need 3 blocks, but the array has 2"),
        (np.zeros((3, 2), dtype=np.uint64), 2**32, r"max 2\^32-1"),
        (np.zeros((3, 8), dtype=np.uint8), 8, "n_bits is only used for packed"),
        (np.full((3, 2), 2**63, dtype=np.uint64), 100, "padding bits"),
        (np.full((3, 1), 2, dtype=np.uint64), 1, "padding bits"),
    ],
    ids=[
        "missing",
        "zero",
        "too_many_blocks",
        "too_few_blocks",
        "huge",
        "uint8",
        "padding",
        "one_bit_padding",
    ],
)
def test_packed_save_invalid_input_raises_and_leaves_no_file(
    tmp_path: Path, array: np.ndarray[Any, Any], n_bits: int | None, match: str
) -> None:
    path = tmp_path / "fps.fp"
    with pytest.raises(ValueError, match=match):
        smiles_fp.save_fingerprints(array, path, n_bits=n_bits)
    assert not path.exists()


def test_save_bit_vectors_with_n_bits_raises(tmp_path: Path) -> None:
    fps = [_fp(smi, 64) for smi in conftest.read_smis()[:3]]
    with pytest.raises(ValueError, match="n_bits is only used for packed"):
        smiles_fp.save_fingerprints(fps, tmp_path / "fps.fp", n_bits=64)
    assert not (tmp_path / "fps.fp").exists()


@pytest.mark.parametrize("n_bits", [1, 63, 64, 65, 100, 2048])
def test_load_packed_matches_blocks_and_roundtrips(tmp_path: Path, n_bits: int) -> None:
    bits = bit_array(37, n_bits)
    smiles_fp.save_fingerprints(bits, tmp_path / "bits.fp")

    packed = smiles_fp.load_fingerprints(tmp_path / "bits.fp", output="packed")
    assert packed.dtype == np.uint64
    np.testing.assert_array_equal(packed, conftest.bits_to_packed(bits))

    smiles_fp.save_fingerprints(packed, tmp_path / "packed.fp", n_bits=n_bits)
    assert (tmp_path / "bits.fp").read_bytes() == (tmp_path / "packed.fp").read_bytes()


def test_load_packed_without_fingerprints_returns_empty_array(tmp_path: Path) -> None:
    smiles_fp.save_fingerprints([], tmp_path / "empty.fp")
    packed = smiles_fp.load_fingerprints(tmp_path / "empty.fp", output="packed")
    assert packed.shape == (0, 0)
    assert packed.dtype == np.uint64


def test_load_fingerprints_as_ndarray_is_a_deprecated_forward(tmp_path: Path) -> None:
    bits = bit_array(37, 100)
    smiles_fp.save_fingerprints(bits, tmp_path / "bits.fp")

    with pytest.warns(DeprecationWarning, match="output='unpacked'"):
        loaded = smiles_fp.load_fingerprints_as_ndarray(  # type: ignore[deprecated]
            tmp_path / "bits.fp"
        )

    np.testing.assert_array_equal(loaded, bits)


def test_load_fingerprints_unknown_output_raises(tmp_path: Path) -> None:
    smiles_fp.save_fingerprints(bit_array(2, 8), tmp_path / "fps.fp")
    with pytest.raises(ValueError, match="output must be one of"):
        smiles_fp.load_fingerprints(tmp_path / "fps.fp", output="array")  # type: ignore[call-overload]


@pytest.mark.parametrize("output", ["unpacked", "packed"])
def test_load_fingerprints_array_truncated_file_raises(tmp_path: Path, output: FpOutput) -> None:
    path = tmp_path / "fps.fp"
    smiles_fp.save_fingerprints(bit_array(4, 64), path)
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match="truncated"):
        smiles_fp.load_fingerprints(path, output=output)


@pytest.mark.parametrize("output", ["unpacked", "packed"])
def test_load_fingerprints_array_missing_header_raises(tmp_path: Path, output: FpOutput) -> None:
    path = tmp_path / "fps.fp"
    path.write_bytes(b"\x01\x00")
    with pytest.raises(ValueError, match="missing header"):
        smiles_fp.load_fingerprints(path, output=output)


def test_benchmark_array_save(  # pragma: no cover
    tmp_path: Path, benchmark: BenchmarkFixture
) -> None:
    bits = bit_array(conftest.BENCHMARK_SIZES[0], 2048)
    benchmark(smiles_fp.save_fingerprints, bits, tmp_path / "bench.fp")

    np.testing.assert_array_equal(
        smiles_fp.load_fingerprints(tmp_path / "bench.fp", output="unpacked"), bits
    )


def test_benchmark_packed_save(  # pragma: no cover
    tmp_path: Path, benchmark: BenchmarkFixture
) -> None:
    bits = bit_array(conftest.BENCHMARK_SIZES[0], 2048)
    packed = conftest.bits_to_packed(bits)
    benchmark(smiles_fp.save_fingerprints, packed, tmp_path / "bench.fp", n_bits=2048)

    np.testing.assert_array_equal(
        smiles_fp.load_fingerprints(tmp_path / "bench.fp", output="unpacked"), bits
    )


def test_benchmark_array_load(  # pragma: no cover
    tmp_path: Path, benchmark: BenchmarkFixture
) -> None:
    bits = bit_array(conftest.BENCHMARK_SIZES[0], 2048)
    smiles_fp.save_fingerprints(bits, tmp_path / "bench.fp")

    loaded = benchmark(smiles_fp.load_fingerprints, tmp_path / "bench.fp", output="unpacked")

    np.testing.assert_array_equal(loaded, bits)


def test_benchmark_packed_load(  # pragma: no cover
    tmp_path: Path, benchmark: BenchmarkFixture
) -> None:
    bits = bit_array(conftest.BENCHMARK_SIZES[0], 2048)
    smiles_fp.save_fingerprints(bits, tmp_path / "bench.fp")
    loaded = benchmark(smiles_fp.load_fingerprints, tmp_path / "bench.fp", output="packed")

    np.testing.assert_array_equal(loaded, conftest.bits_to_packed(bits))
