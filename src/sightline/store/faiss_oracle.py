"""Exact brute-force oracle. Never a serving path; it exists to be right.

Every recall number this project publishes is measured against this store. Its
job is to produce the *complete* correct answer for a query under a filter —
the true top-k of the permitted subset, with no approximation anywhere — so that
a disagreement with Qdrant or pgvector is attributable to the backend and not to
the measurement.

``IndexFlatIP``: inner product over unit-length vectors, which is cosine. No
graph, no quantisation, no tuning knobs, nothing to get subtly wrong.

**This is not a serving path and must not become one.** :data:`NEVER_SERVE`
exists so a test can assert the serving API never imports this module. There is
deliberately no runtime import guard, because the one legitimate in-process user
*is* a serving process: the drift checker (guardrail 13) samples served queries
and replays them here, and a hard guard would make that job impossible to write.
The control is a source-level test, not a thrown exception.

FAISS is an optional extra, and the oracle does not require it. With
``faiss-cpu`` installed :class:`FaissOracleStore` mirrors the vectors into an
``IndexFlatIP`` and filters with an ``IDSelectorBatch``, which is materially
faster on a 70k corpus. Without it, :func:`build_oracle` returns a
:class:`~sightline.store.memory.MemoryVectorStore`, whose search is *also*
exhaustive and *also* exact — identical results, more wall clock. FR-11 requires
the numpy-only path to work, so that is the one CI runs by default.

The filter runs before the distance computation on both paths. An oracle that
post-filtered would be measuring the baseline arm's bug and calling it ground
truth.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from sightline.store.base import StoreStats, UncheckedHit, VectorStore
from sightline.store.memory import MemoryVectorStore, first_matching_token, normalise
from sightline.types import Chunk, FilterPlan

__all__ = [
    "FaissOracleStore",
    "build_oracle",
    "faiss_available",
    "NEVER_SERVE",
    "ORACLE_EXTRA_HINT",
]

#: Read by the test that asserts the serving API cannot reach this module.
NEVER_SERVE = True

ORACLE_EXTRA_HINT = (
    "the FAISS oracle needs faiss-cpu, which is an optional extra: "
    "pip install 'sightline[oracle]' — or use build_oracle(), which falls back to "
    "the numpy store, whose search is equally exact"
)

try:  # pragma: no cover - exercised only by the optional-extra CI job
    import faiss
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]


def faiss_available() -> bool:
    """Whether the ``oracle`` extra is installed."""
    return faiss is not None


def build_oracle(dim: int | None = None, *, prefer_faiss: bool = True) -> VectorStore:
    """Return the fastest exact store available.

    Args:
        dim: Vector dimensionality, inferred on first upsert if omitted.
        prefer_faiss: Set false to force the numpy path, which is what the
            differential test does when it wants both paths compared.

    Returns:
        A :class:`FaissOracleStore` if FAISS is installed and wanted, otherwise a
        :class:`~sightline.store.memory.MemoryVectorStore`. Both are exhaustive
        and exact, and must return identical hits for identical input — a test
        asserts exactly that when the extra is present.
    """
    if prefer_faiss and faiss_available():
        return FaissOracleStore(dim)
    return MemoryVectorStore(dim, name="oracle-numpy")


class FaissOracleStore:
    """``IndexFlatIP`` exact search, filtered to the permitted subset first.

    Implements :class:`~sightline.store.base.VectorStore`. Bookkeeping — chunk
    metadata, grant-token bitsets, plan compilation — is delegated to
    :class:`~sightline.store.memory.MemoryVectorStore`, which already owns the
    definition of what a plan admits. Only the scoring is FAISS's.

    The index is rebuilt from the reference store's matrix whenever it changes.
    ``IndexFlatIP`` has no training and no graph, so a rebuild is a memcpy, and a
    rebuild is simpler than maintaining ``remove_ids``/``add_with_ids`` parity
    across upserts. For a measurement instrument, simple and obviously correct
    beats incremental.
    """

    name = "faiss-oracle"

    def __init__(self, dim: int | None = None) -> None:
        """Create the oracle.

        Raises:
            ImportError: If ``faiss-cpu`` is not installed. Use :func:`build_oracle`
                to get the numpy fallback instead.
        """
        if faiss is None:
            raise ImportError(ORACLE_EXTRA_HINT)
        self._mem = MemoryVectorStore(dim, name=self.name)
        self._index: Any | None = None
        self._dirty = True

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        self._mem.upsert(chunks, vectors)
        self._dirty = True

    def _ensure_index(self) -> Any | None:
        if not self._dirty and self._index is not None:
            return self._index
        dim = self._mem.dim()
        if dim is None or len(self._mem) == 0:
            return None
        index = faiss.IndexFlatIP(dim)
        # np.ascontiguousarray because matrix() hands back a read-only view and
        # FAISS wants a contiguous, writable-in-its-own-right float32 buffer.
        index.add(np.ascontiguousarray(self._mem.matrix(), dtype=np.float32))
        self._index = index
        self._dirty = False
        return index

    def search(
        self, query_vector: Sequence[float], plan: FilterPlan, k: int
    ) -> list[UncheckedHit]:
        """The true top-k of the permitted subset. Ground truth, by construction."""
        if k <= 0:
            return []
        index = self._ensure_index()
        if index is None:
            return []
        rows = self._mem.candidate_rows(plan)
        if rows is not None and rows.size == 0:
            return []

        query = np.ascontiguousarray(
            normalise(query_vector, self._mem.dim()).reshape(1, -1), dtype=np.float32
        )
        limit = k if rows is None else min(k, int(rows.size))

        try:
            params = None
            keep_alive = None
            if rows is not None:
                # IndexFlat ids are row positions, and the reference store's rows
                # are copied into the index in order, so the selector speaks the
                # same id space as candidate_rows().
                keep_alive = np.ascontiguousarray(rows, dtype=np.int64)
                selector = faiss.IDSelectorBatch(
                    int(keep_alive.size), faiss.swig_ptr(keep_alive)
                )
                params = faiss.SearchParameters()
                params.sel = selector
            scores, ids = index.search(query, limit, params=params)
            del keep_alive
        except (TypeError, AttributeError, RuntimeError):  # pragma: no cover
            # Older FAISS builds do not accept SearchParameters on IndexFlat.
            # The numpy path is exact too, so falling back costs time, not truth.
            return self._mem.search(query_vector, plan, k)

        hits: list[UncheckedHit] = []
        for score, row in zip(scores[0].tolist(), ids[0].tolist()):
            if row < 0:  # FAISS pads with -1 when fewer than `limit` are selected.
                continue
            chunk_id, object_ref, text, tokens = self._mem.row_info(int(row))
            hits.append(
                UncheckedHit(
                    chunk_id=chunk_id,
                    score=float(score),
                    text=text,
                    object_ref=object_ref,
                    matched_token=first_matching_token(plan, tokens),
                )
            )
        return hits

    def count_matching(self, plan: FilterPlan) -> int:
        return self._mem.count_matching(plan)

    def stats(self) -> StoreStats:
        inner = self._mem.stats()
        return StoreStats(
            n_vectors=inner.n_vectors,
            resident_bytes=inner.resident_bytes * 2,  # the matrix exists twice over.
            disk_bytes=0,
            backend=self.name,
            detail={
                **inner.detail,
                "index": "IndexFlatIP",
                "exact": "true",
                "never_serve": "true",
                "role": "ground truth for recall measurement",
            },
        )

    def __len__(self) -> int:
        return len(self._mem)
