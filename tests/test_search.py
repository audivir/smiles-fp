"""Test high-level search and windowing API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import conftest
import numpy as np
import pytest
import smiles_fp_rs
from smiles_fp_rs.search import similarity_search, windowed_bulk_tanimoto

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def temp_db(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary query and database files."""
    test_fps = list(conftest.get_cached_fps().values())[:64]
    quarter = len(test_fps) // 4
    q_path = tmp_path / "query.bin"
    db_path = tmp_path / "db.bin"

    smiles_fp_rs.save_fingerprints(test_fps[:quarter], q_path)
    smiles_fp_rs.save_fingerprints(test_fps[quarter:], db_path)
    return q_path, db_path


def test_similarity_search(temp_db: tuple[Path, Path]) -> None:
    """Verify similarity_search returns correct dict structure and top-k."""
    q_path, db_path = temp_db

    with q_path.open("rb") as f:
        q_size = int.from_bytes(f.read(4), "little")

    with db_path.open("rb") as f:
        db_size = int.from_bytes(f.read(4), "little")

    # Create simple identifiers
    q_ids = [f"q{i}" for i in range(q_size)]
    db_ids = [f"db{i}" for i in range(db_size)]

    k = 3
    results = similarity_search(q_ids, q_path, db_ids, db_path, k=k)

    assert len(results) == q_size
    for matches in results.values():
        assert len(matches) == k
        # Ensure scores are sorted descending
        scores = [m[1] for m in matches]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("agg", ["full", "max", "min"])
def test_windowed_bulk_tanimoto(
    agg: Literal["full", "max", "min"],
    temp_db: tuple[Path, Path],
) -> None:
    q_path, db_path = temp_db

    full_matrix = smiles_fp_rs.bulk_tanimoto_mmap(q_path, db_path, agg=agg)
    chunks = list(windowed_bulk_tanimoto(q_path, db_path, window_size=5, agg=agg))

    stacked = np.hstack(chunks) if agg == "full" else np.stack(chunks, axis=1)

    if agg == "max":
        windowed = np.max(stacked, axis=1)
    elif agg == "min":
        windowed = np.min(stacked, axis=1)
    else:
        windowed = stacked

    np.testing.assert_allclose(full_matrix, windowed)


def test_windowed_bulk_tanimoto_mean() -> None:
    with pytest.raises(ValueError, match="Mean unsupported"):
        list(windowed_bulk_tanimoto("q.bin", "db.bin", agg="mean"))
