"""The baseline arm must actually exhibit the bug we claim to have measured.

``PostFilterStore`` implements the wrong design on purpose: search everything,
then drop what the principal may not see. If that arm ever stopped losing recall,
the project's headline chart would be measuring nothing, so the collapse itself
needs a test.
"""

from __future__ import annotations

import numpy as np
import pytest

from sightline.store.memory import MemoryVectorStore
from sightline.store.postfilter import PostFilterStore
from sightline.types import FilterPlan, GrantToken, PlanStrategy, PrincipalRef
from _world import make_corpus

DIM, K, TRIALS = 32, 10, 25
GROUPS = [f"g{i}" for i in range(20)]


def _plan(n_groups: int) -> FilterPlan:
    return FilterPlan(
        principal=PrincipalRef("user", "dana"),
        strategy=PlanStrategy.GRANT_TOKENS,
        grant_tokens=frozenset(GrantToken(g) for g in GROUPS[:n_groups]),
        epoch=1,
    )


def _mean_returned(store, plan) -> float:
    rng = np.random.default_rng(11)
    total = 0
    for _ in range(TRIALS):
        q = rng.normal(size=DIM).astype("float32")
        total += len(store.search(q, plan, K))
    return total / TRIALS


@pytest.fixture
def stores():
    chunks, vectors = make_corpus(1_000, GROUPS, dim=DIM)
    pre, post = MemoryVectorStore(dim=DIM), PostFilterStore(dim=DIM)
    pre.upsert(chunks, vectors)
    post.upsert(chunks, vectors)
    return pre, post


def test_prefilter_always_fills_k(stores):
    """Searching only the permitted set returns k whenever k documents exist."""
    pre, _ = stores
    assert _mean_returned(pre, _plan(1)) == pytest.approx(K)


def test_postfilter_collapses_at_low_visibility(stores):
    """One group in twenty is 5% visibility. Post-filtering must return far under k."""
    _, post = stores
    assert _mean_returned(post, _plan(1)) < K / 4


def test_postfilter_recovers_at_full_visibility(stores):
    """The two agree when the principal can see everything, which bounds the claim.

    Without this, a broken post-filter implementation would also pass the test
    above, and we would be publishing a chart of our own bug.
    """
    pre, post = stores
    assert _mean_returned(post, _plan(len(GROUPS))) == pytest.approx(
        _mean_returned(pre, _plan(len(GROUPS)))
    )


def test_collapse_is_monotonic_in_visibility(stores):
    """More visibility must never return fewer results."""
    _, post = stores
    counts = [_mean_returned(post, _plan(n)) for n in (1, 2, 5, 10, 20)]
    assert counts == sorted(counts), counts
