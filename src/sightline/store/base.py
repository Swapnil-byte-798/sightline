"""The one interface every vector backend implements — and the type that makes
a permission bug hard to write.

Three backends sit behind this: Qdrant (the serving path), pgvector with Postgres
row-level security (a second, independent enforcement layer), and a brute-force
FAISS index (the oracle — never served, used to prove the others return the
*complete* correct answer under a filter).

THE CENTRAL RULE OF THIS SYSTEM
-------------------------------
**The index is a hint. The database is the authority.**

A vector index is a denormalised copy of permission state, so it is always
slightly stale. Every system that treats it as authoritative has a window in
which a revoked user still retrieves. Sightline closes that window by refusing to
make it representable: :meth:`VectorStore.search` returns ``UncheckedHit``, and
the answer builder accepts only ``Hit``. The sole way to turn one into the other
is :func:`recheck`, which consults the live tuple store.

The consequence is the security argument for the whole product: **a stale index
loses results, it never leaks them.** A document whose permissions were tightened
since indexing is dropped at recheck. A document whose permissions were loosened
is simply missed until reindex — an availability cost, deliberately chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from sightline.types import Chunk, FilterPlan

__all__ = ["UncheckedHit", "Hit", "StoreStats", "VectorStore", "Rechecker"]


@dataclass(frozen=True, slots=True)
class UncheckedHit:
    """What an index returns. Not safe to show anyone yet.

    There is deliberately no method on this class that produces a ``Hit``. You
    must go through :func:`sightline.authz.recheck.recheck`, which needs the
    tuple store — so the dangerous path requires an import that looks wrong.
    """

    chunk_id: str
    score: float
    text: str
    object_ref: str
    #: The grant token the index matched on. Advisory; recheck does not trust it.
    matched_token: str | None = None


@dataclass(frozen=True, slots=True)
class Hit:
    """An ``UncheckedHit`` that survived re-checking against live permissions.

    Constructing one of these outside ``recheck()`` is possible in Python — this
    is a convention enforced by review and by a test that greps for it, not by
    the runtime. The test is ``tests/test_no_unchecked_construction.py``.
    """

    chunk_id: str
    score: float
    text: str
    object_ref: str
    why_allowed: str
    #: Policy epoch at the moment of the re-check, recorded on the response.
    checked_at_epoch: int


@dataclass(frozen=True, slots=True)
class StoreStats:
    """What the store costs. Reported per backend so the trade-off is legible."""

    n_vectors: int
    resident_bytes: int
    disk_bytes: int
    backend: str
    detail: dict[str, str]


@runtime_checkable
class VectorStore(Protocol):
    """A vector index that can restrict a search to what a principal may see.

    The contract that matters is on :meth:`search`: the hits MUST be the top-k
    over the *permitted subset*, not the top-k over everything with forbidden
    entries removed afterwards. Those two differ, and the size of the difference
    is this project's headline chart. ``PostFilterStore`` implements the wrong
    one deliberately, as the baseline arm.
    """

    name: str

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Insert or replace chunks. Must be idempotent on ``chunk.id``."""
        ...

    def search(
        self, query_vector: Sequence[float], plan: FilterPlan, k: int
    ) -> list[UncheckedHit]:
        """Top-k over the subset ``plan`` admits. Push the filter INTO the index."""
        ...

    def count_matching(self, plan: FilterPlan) -> int:
        """How many chunks the plan admits. Drives strategy selection."""
        ...

    def stats(self) -> StoreStats:
        ...


@runtime_checkable
class Rechecker(Protocol):
    """Turns index output into servable results by consulting live permissions."""

    def recheck(
        self, principal_ref: str, hits: Sequence[UncheckedHit]
    ) -> tuple[list[Hit], int]:
        """Return surviving hits and the count dropped as stale."""
        ...
