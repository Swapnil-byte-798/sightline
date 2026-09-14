"""A chunk/vector length mismatch must raise, not truncate.

Without ``strict=True``, ``zip(chunks, vectors)`` stops at the shorter input and
says nothing. The visible symptom would be documents missing from the index, or
worse, present but scored against a neighbour's vector. Both look like "retrieval
is a bit poor" rather than like a bug, which is how they survive.
"""

from __future__ import annotations

import numpy as np
import pytest

from sightline.store.memory import MemoryVectorStore
from sightline.types import Chunk, GrantToken, ObjectRef

DIM = 16


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            id=f"c{i}",
            object=ObjectRef("doc", str(i)),
            text=f"document {i}",
            grant_tokens=frozenset({GrantToken("g0")}),
        )
        for i in range(n)
    ]


def _vectors(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    return [rng.normal(size=DIM).astype("float32") for _ in range(n)]


def test_more_chunks_than_vectors_raises():
    store = MemoryVectorStore(dim=DIM)
    with pytest.raises(ValueError):
        store.upsert(_chunks(5), _vectors(3))


def test_more_vectors_than_chunks_raises():
    store = MemoryVectorStore(dim=DIM)
    with pytest.raises(ValueError):
        store.upsert(_chunks(3), _vectors(5))


def test_matched_lengths_still_work():
    """The guard must not break the normal path."""
    store = MemoryVectorStore(dim=DIM)
    store.upsert(_chunks(4), _vectors(4))
    assert store.stats().n_vectors == 4
