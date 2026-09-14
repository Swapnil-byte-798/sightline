"""The most important test in the repository.

An empty permission set must mean "see nothing", never "see everything". This is
an easy accident: an empty collection is falsy, so ``if tokens:`` quietly becomes
"skip the filter", and the system opens up completely at exactly the moment it
should close. See docs/adr/0003-fail-closed.md.
"""

from __future__ import annotations

import pytest

from sightline.store.memory import MemoryVectorStore
from sightline.types import FilterPlan, PlanStrategy, PrincipalRef

from _world import make_corpus

DIM = 32


@pytest.fixture
def loaded_store() -> MemoryVectorStore:
    chunks, vectors = make_corpus(200, ["g0", "g1", "g2", "g3"], dim=DIM)
    s = MemoryVectorStore(dim=DIM)
    s.upsert(chunks, vectors)
    return s


def _plan(**kw) -> FilterPlan:
    return FilterPlan(principal=PrincipalRef("user", "nobody"), epoch=1, **kw)


def test_empty_grant_tokens_returns_nothing(loaded_store):
    """A principal holding no grant tokens sees zero documents, not all of them."""
    plan = _plan(strategy=PlanStrategy.GRANT_TOKENS, grant_tokens=frozenset())
    hits = loaded_store.search([0.0] * DIM, plan, 10)
    assert hits == [], f"empty token set leaked {len(hits)} results"


def test_empty_grant_tokens_counts_zero(loaded_store):
    plan = _plan(strategy=PlanStrategy.GRANT_TOKENS, grant_tokens=frozenset())
    assert loaded_store.count_matching(plan) == 0


def test_empty_explicit_ids_returns_nothing(loaded_store):
    """Same rule for the enumerate strategy: an empty id list admits nothing."""
    plan = _plan(strategy=PlanStrategy.ENUMERATE, explicit_ids=frozenset())
    assert loaded_store.search([0.0] * DIM, plan, 10) == []


def test_unknown_token_returns_nothing(loaded_store):
    """A token nobody was granted must not match by accident."""
    plan = _plan(strategy=PlanStrategy.GRANT_TOKENS, grant_tokens=frozenset({"g-does-not-exist"}))
    assert loaded_store.search([0.0] * DIM, plan, 10) == []
