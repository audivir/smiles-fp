"""Search for similar molecules."""

from __future__ import annotations

from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

from smiles_fp_rs._smiles_fp_rs import bulk_tanimoto_mmap, bulk_tanimoto_mmap_topk

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    import numpy as np
    from _typeshed import StrPath
    from numpy.typing import NDArray

T = TypeVar("T")


def windowed_bulk_tanimoto(
    query: StrPath,
    db: StrPath,
    window_size: int = 10_000,
    n_threads: int = -1,
    agg: Literal["max", "min", "full"] | None = None,
) -> Generator[NDArray[np.float64]]:
    """Yield chunked similarity matrices to prevent out-of-memory errors.

    Args:
        query: Path to the query binary fingerprint file.
        db: Path to the database binary fingerprint file.
        window_size: Size of the window
        n_threads: The number of threads to use. Defaults to -1 (auto-detect).
        agg: Aggregation method. Defaults to 'full'.

    Returns:
        A generator with array window (shape depending on aggregation)
    """
    if agg == "mean":
        raise ValueError("Mean unsupported")

    db = Path(db)
    query = Path(query)

    with db.open("rb") as f:
        db_size = int.from_bytes(f.read(4), "little")

    for offset in range(0, db_size, window_size):
        yield bulk_tanimoto_mmap(
            query,
            db,
            n_threads=n_threads,
            agg=agg,
            db_offset=offset,
            db_limit=window_size,
        )


def similarity_search(  # noqa: PLR0913
    query_ids: Sequence[T],
    query: StrPath,
    db_ids: Sequence[T],
    db: StrPath,
    k: int = 10,
    n_threads: int = cpu_count(),
) -> dict[T, list[tuple[T, float]]]:
    """Search for the top k similar fingerprints in a list of fingerprints.

    Args:
        query_ids: Any identifier for each query fingerprint.
        query: Path to the query binary fingerprint file.
        db_ids: Any identifier for each comparison fingerprint.
        db: Path to the database binary fingerprint file.
        k: Number of most-similar molecules to query for.
        n_threads: The number of threads to use. Defaults to -1 (auto-detect).

    Returns:
        A dictionary of query identifiers to a list of
        tuples of (identifier, similarity) ranked by similarity.
    """
    idx, scores = bulk_tanimoto_mmap_topk(query, db, k, n_threads)

    results: dict[T, list[tuple[T, float]]] = {}
    for q_id, q_idx, q_scores in zip(query_ids, idx, scores, strict=True):
        top_results: list[tuple[T, float]] = []
        for ix, score in zip(q_idx, q_scores, strict=True):
            top_results.append((db_ids[ix], score))
        results[q_id] = top_results

    return results
