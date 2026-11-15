"""Benchmarks the conversion between SMILES and RDKit molecules."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest
import smiles_fp
from rdkit import Chem
from smiles_fp.helpers import mol_from_smi

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture

THREADS = (1, 4, 8)


def serial_parse(smis: Sequence[str]) -> list[Chem.Mol | None]:  # pragma: no cover
    return [Chem.MolFromSmiles(smi) for smi in smis]


def thread_pool_parse(smis: Sequence[str]) -> list[Chem.Mol | None]:  # pragma: no cover
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(Chem.MolFromSmiles, smis))


def joblib_parse(smis: Sequence[str]) -> list[Chem.Mol | None]:  # pragma: no cover
    joblib = pytest.importorskip("joblib")
    parallel = joblib.Parallel(n_jobs=8, return_as="list")
    return parallel(joblib.delayed(mol_from_smi)(smi) for smi in smis)  # type: ignore[no-any-return]


def serial_write(mols: Sequence[Chem.Mol]) -> list[str]:  # pragma: no cover
    return [Chem.MolToSmiles(mol) for mol in mols]


def thread_pool_write(mols: Sequence[Chem.Mol]) -> list[str]:  # pragma: no cover
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(Chem.MolToSmiles, mols))


@pytest.mark.parametrize(
    "parse",
    [serial_parse, thread_pool_parse, joblib_parse],
    ids=["rdkit_serial", "thread_pool", "joblib"],
)
def test_benchmark_parse_python(  # pragma: no cover
    parse: Callable[[Sequence[str]], list[Chem.Mol | None]],
    benchmark_smis: list[str],
    benchmark_mols: list[Chem.Mol],
    benchmark: BenchmarkFixture,
) -> None:
    mols = benchmark(parse, benchmark_smis)

    assert [Chem.MolToSmiles(m) for m in mols if m] == serial_write(benchmark_mols)


@pytest.mark.parametrize("n_threads", THREADS)
def test_benchmark_mols_from_smiles(  # pragma: no cover
    n_threads: int,
    benchmark_smis: list[str],
    benchmark_mols: list[Chem.Mol],
    benchmark: BenchmarkFixture,
) -> None:
    mols = benchmark(smiles_fp.mols_from_smiles, benchmark_smis, n_threads=n_threads)

    assert [Chem.MolToSmiles(m) for m in mols if m] == serial_write(benchmark_mols)


@pytest.mark.parametrize("n_threads", THREADS)
def test_benchmark_multithreaded_smiles_mol_supplier(  # pragma: no cover
    n_threads: int,
    benchmark_smis: list[str],
    benchmark_mols: list[Chem.Mol],
    benchmark: BenchmarkFixture,
    tmp_path: Path,
) -> None:
    path = tmp_path / "smis.smi"
    path.write_text("\n".join(benchmark_smis) + "\n")

    def read() -> list[Chem.Mol]:
        supplier = Chem.MultithreadedSmilesMolSupplier(
            str(path), titleLine=False, nameColumn=-1, numWriterThreads=n_threads
        )
        # the supplier yields a few trailing None entries.
        return [mol for mol in supplier if mol is not None]  # type: ignore[redundant-expr]

    mols = benchmark(read)

    # the supplier does not guarantee the input order.
    assert sorted(Chem.MolToSmiles(m) for m in mols) == sorted(serial_write(benchmark_mols))


@pytest.mark.parametrize(
    "write", [serial_write, thread_pool_write], ids=["rdkit_serial", "thread_pool"]
)
def test_benchmark_write_python(  # pragma: no cover
    write: Callable[[Sequence[Chem.Mol]], list[str]],
    benchmark_mols: list[Chem.Mol],
    benchmark: BenchmarkFixture,
) -> None:
    smis = benchmark(write, benchmark_mols)

    assert smis == serial_write(benchmark_mols)


@pytest.mark.parametrize("n_threads", THREADS)
def test_benchmark_mols_to_smiles(  # pragma: no cover
    n_threads: int,
    benchmark_mols: list[Chem.Mol],
    benchmark: BenchmarkFixture,
) -> None:
    smis = benchmark(smiles_fp.mols_to_smiles, benchmark_mols, n_threads=n_threads)

    assert smis == serial_write(benchmark_mols)
