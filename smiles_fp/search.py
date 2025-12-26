"""Search for similar molecules."""

from __future__ import annotations

import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from smiles_fp._smiles_fp import bulk_tanimoto_mmap, save_fingerprints

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rdkit.DataStructs import ExplicitBitVect

T = TypeVar("T")


def similarity_search(
    query_ids: Sequence[T],
    query: Sequence[ExplicitBitVect] | str | Path,
    fps_ids: Sequence[T],
    fps: Sequence[ExplicitBitVect] | str | Path,
    k: int = 10,
    num_threads: int = cpu_count(),
) -> dict[T, list[tuple[T, float]]]:
    """Search for the top k similar fingerprints in a list of fingerprints.

    Args:
        query_ids: Any identifier for each query fingerprint.
        query: The query fingerprints either as a list of fingerprints
            or a path to SmilesFP binary file.
        fps_ids: Any identifier for each comparison fingerprint.
        fps: The comparison fingerprints either as a list of fingerprints
            or a path to SmilesFP binary file.
        k: The number of similar fingerprints to return.
        num_threads: The number of threads to use.

    Returns:
        A dictionary of query identifiers to a list of
        tuples of (identifier, similarity) ranked by similarity.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        if isinstance(query, str | Path):
            query_path = str(query)
        else:
            query_path = str(tmpdir / "query.bin")
            save_fingerprints(query, query_path)
        if isinstance(fps, str | Path):
            fps_path = str(fps)
        else:
            fps_path = str(tmpdir / "fps.bin")
            save_fingerprints(fps, str(fps_path))
        sims = bulk_tanimoto_mmap(
            query_path,
            fps_path,
            num_threads,
        )

    k = min(k, len(fps_ids))
    sims = sims.reshape(len(query_ids), len(fps_ids))
    top_k = sims.argsort(axis=-1)[:, -k:]  # reverse (k, k-1, ..., 1)
    top_k = top_k[:, ::-1]  # right order (1, 2, ..., k)

    result: dict[T, list[tuple[T, float]]] = {}
    for ix, top_k_ixs in enumerate(top_k):
        top_results: list[tuple[T, float]] = []
        for rank in range(k):
            fps_ix = top_k_ixs[rank]
            top_results.append(
                (
                    fps_ids[fps_ix],
                    sims[ix, fps_ix],
                )
            )
        result[query_ids[ix]] = top_results

    return result
