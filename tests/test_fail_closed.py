"""The most important test in the repository.

An empty permission set means **see nothing**. Never "see everything", never "no
filter applies", never "unrestricted". This is an easy accident and a total one:
an empty collection is falsy, so ``if tokens:`` quietly becomes "skip the filter"
and the system opens up completely at exactly the moment it should close. Planted
mutant M6 is that one-character mistake; ADR 0003 is the decision.

Every layer that can express a filter is asserted here separately — the plan
predicate, the store's own admission rule, the actual search, the count, the
baseline arm, and the compiler that produces plans in the first place — because
"empty means nobody" has to be true in all of them and each one re-derives it
from a slightly different angle.

The principal used throughout is ``user:dave``, who holds no tuple anywhere in
the fixture world. If any assertion in this file starts failing, stop reading the
rest of the suite: nothing else it says about permissions is worth anything.
"""

from __future__ import annotations

import pytest
from _world import make_corpus

from sightline.store.memory import MemoryVectorStore, admits, plan_bindings
from sightline.types import Chunk, FilterPlan, GrantToken, ObjectRef, PlanStrategy, PrincipalRef

authz = pytest.importorskip("sightline.authz", reason="sightline.authz is not present yet")

compile_plan = authz.compile_plan
plan_admits = authz.plan_admits

DIM = 32
NOBODY = PrincipalRef("user", "nobody")
GROUPS = ["g0", "g1", "g2", "g3"]


@pytest.fixture
def loaded_store() -> MemoryVectorStore:
    chunks, vectors = make_corpus(200, GROUPS, dim=DIM)
    store = MemoryVectorStore(dim=DIM)
    store.upsert(chunks, vectors)
    return store


def _plan(**kw) -> FilterPlan:
    return FilterPlan(principal=NOBODY, epoch=1, **kw)


def _query() -> list[float]:
    """A query vector, not the zero vector.

    The zero vector scores identically against everything and would make a
    "returned nothing" assertion pass for the wrong reason.
    """
    return [1.0] + [0.0] * (DIM - 1)


# --------------------------------------------------------------------------
# The predicate
# --------------------------------------------------------------------------


def test_empty_token_set_is_a_binding_condition_not_an_absent_one():
    """``plan_bindings`` is where "empty means nobody" is decided once."""
    by_ids, by_tokens = plan_bindings(
        _plan(strategy=PlanStrategy.GRANT_TOKENS, grant_tokens=frozenset())
    )
    assert by_tokens is True, "an empty token set must still bind; falsy is not absent"
    by_ids, _ = plan_bindings(_plan(strategy=PlanStrategy.ENUMERATE, explicit_ids=frozenset()))
    assert by_ids is True


@pytest.mark.parametrize(
    "plan",
    [
        _plan(strategy=PlanStrategy.GRANT_TOKENS, grant_tokens=frozenset()),
        _plan(strategy=PlanStrategy.ENUMERATE, explicit_ids=frozenset()),
        _plan(strategy=PlanStrategy.EXACT_SCAN),
    ],
    ids=["no-tokens", "no-ids", "no-conditions-at-all"],
)
def test_plan_predicate_admits_nothing(plan):
    """``plan_admits`` and ``admits`` agree, and both say no."""
    assert plan_admits(plan, "7", ["g0"]) is False
    assert admits(plan, "doc:7", ["g0"]) is False


def test_only_unfiltered_admits_everything():
    """``UNFILTERED`` is the one strategy that means "no condition", and the
    compiler will not emit it without being told the corpus size."""
    everything = _plan(strategy=PlanStrategy.UNFILTERED)
    assert plan_admits(everything, "7", []) is True
    assert admits(everything, "doc:7", []) is True


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


def test_empty_grant_tokens_returns_nothing(loaded_store):
    """A principal holding no grant tokens sees zero documents, not all of them."""
    plan = _plan(strategy=PlanStrategy.GRANT_TOKENS, grant_tokens=frozenset())
    hits = loaded_store.search(_query(), plan, 10)
    assert hits == [], f"empty token set leaked {len(hits)} results"


def test_empty_grant_tokens_counts_zero(loaded_store):
    plan = _plan(strategy=PlanStrategy.GRANT_TOKENS, grant_tokens=frozenset())
    assert loaded_store.count_matching(plan) == 0


def test_empty_explicit_ids_returns_nothing(loaded_store):
    """Same rule for the enumerate strategy: an empty id list admits nothing."""
    plan = _plan(strategy=PlanStrategy.ENUMERATE, explicit_ids=frozenset())
    assert loaded_store.search(_query(), plan, 10) == []
    assert loaded_store.count_matching(plan) == 0


def test_unknown_token_returns_nothing(loaded_store):
    """A token nobody was granted must not match by accident."""
    plan = _plan(
        strategy=PlanStrategy.GRANT_TOKENS,
        grant_tokens=frozenset({GrantToken("g-does-not-exist")}),
    )
    assert loaded_store.search(_query(), plan, 10) == []


def test_a_chunk_with_no_tokens_is_visible_to_nobody(loaded_store):
    """Empty *on the chunk* means nobody either (FR-15).

    A document that expands to no granting userset is not "public", it is
    unreachable, and ingest is supposed to reject it rather than index it. If one
    gets in anyway, no plan may admit it.
    """
    loaded_store.upsert(
        [
            Chunk(
                id="orphan",
                object=ObjectRef("doc", "orphan"),
                text="nobody was granted this",
                grant_tokens=frozenset(),
            )
        ],
        [[1.0] * DIM],
    )
    for n in range(1, len(GROUPS) + 1):
        plan = _plan(
            strategy=PlanStrategy.GRANT_TOKENS,
            grant_tokens=frozenset(GrantToken(g) for g in GROUPS[:n]),
        )
        assert all(h.chunk_id != "orphan" for h in loaded_store.search(_query(), plan, 50))


def test_the_store_is_not_vacuously_empty(loaded_store):
    """Prove the fixture can return results, so the assertions above mean something.

    Without this, a store that returned ``[]`` for every query in existence would
    pass every other test in this file.
    """
    plan = _plan(
        strategy=PlanStrategy.GRANT_TOKENS, grant_tokens=frozenset({GrantToken("g0")})
    )
    assert len(loaded_store.search(_query(), plan, 10)) == 10


# --------------------------------------------------------------------------
# The baseline arm fails closed too
# --------------------------------------------------------------------------


def test_post_filter_baseline_also_fails_closed(loaded_store):
    """The wrong design is wrong about recall, not about permissions.

    Keeping this straight matters for the evaluation: if the baseline leaked as
    well as under-retrieved, the headline chart would be measuring two bugs and
    attributing both to one.
    """
    postfilter = pytest.importorskip(
        "sightline.store.postfilter", reason="the baseline arm is not present yet"
    )
    chunks, vectors = make_corpus(200, GROUPS, dim=DIM)
    store = postfilter.PostFilterStore(dim=DIM)
    store.upsert(chunks, vectors)
    plan = _plan(strategy=PlanStrategy.GRANT_TOKENS, grant_tokens=frozenset())
    assert store.search(_query(), plan, 10) == []


# --------------------------------------------------------------------------
# The compiler
# --------------------------------------------------------------------------


def test_a_principal_with_no_tuples_compiles_to_a_deny_all_plan(world):
    """dave's plan admits nothing, and says so in every field.

    Note the strategy: ``ENUMERATE`` of nothing, not ``UNFILTERED``. "No
    constraint" and "no permitted documents" are one refactor apart and must
    never be spelled the same way.
    """
    plan = compile_plan(world, PrincipalRef("user", "dave"))
    assert plan.strategy is PlanStrategy.ENUMERATE
    assert plan.explicit_ids == frozenset()
    assert plan.grant_tokens == frozenset()
    assert plan.estimated_cardinality == 0
    assert plan_admits(plan, "merger", ["anything"]) is False


def test_deny_all_plan_returns_nothing_from_a_real_index(world):
    """End to end: compile dave, search a populated index, get nothing."""
    store = MemoryVectorStore(dim=DIM)
    chunks, vectors = make_corpus(50, GROUPS, dim=DIM)
    store.upsert(chunks, vectors)
    plan = compile_plan(world, PrincipalRef("user", "dave"))
    assert store.search(_query(), plan, 10) == []
    assert store.count_matching(plan) == 0


def test_unfiltered_requires_being_told_the_corpus_size(world):
    """Without a corpus size, ``UNFILTERED`` is not reachable at all.

    The tuple store only knows about documents somebody wrote a tuple for, and
    the dangerous document is precisely the one nobody did. "This principal can
    see everything I know about" is not the same claim as "this principal can see
    everything", and the compiler is not allowed to confuse them.
    """
    alice = PrincipalRef("user", "alice")
    assert compile_plan(world, alice).strategy is not PlanStrategy.UNFILTERED
    # Even claiming a corpus far larger than the tuple store knows about must not
    # widen the plan.
    assert (
        compile_plan(world, alice, corpus_size=10_000).strategy is not PlanStrategy.UNFILTERED
    )


def test_revoking_the_last_grant_returns_to_deny_all(world):
    """The transition into the empty state is the one that gets missed."""
    bob = PrincipalRef("user", "bob")
    assert compile_plan(world, bob).explicit_ids == frozenset({"bob-notes"})
    world.delete("doc:bob-notes#viewer@user:bob")
    after = compile_plan(world, bob)
    assert after.explicit_ids == frozenset()
    assert after.grant_tokens == frozenset()
    assert plan_admits(after, "bob-notes", []) is False
