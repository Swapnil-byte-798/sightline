"""The index is a hint. The database is the authority.

``recheck()`` is where the hint meets the authority, and it is the reason a stale
index in this system loses results instead of leaking them. Everything here is a
variation on one experiment: index the corpus under one policy, change the
policy, and ask what a query returns.

The asymmetry being tested, stated once:

* **Permissions tightened since indexing** — the index still offers the chunk,
  ``recheck`` drops it, and the drop is counted. Cost: nothing.
* **Permissions loosened since indexing** — the index does not offer the chunk
  at all, ``recheck`` never sees it, and the result is missing until reindex.
  Cost: availability, deliberately chosen.

A system that got this backwards would look identical in a demo.
"""

from __future__ import annotations

from importlib import import_module

import pytest

from sightline.store.base import Hit, UncheckedHit
from sightline.store.memory import MemoryVectorStore
from sightline.types import Chunk, ObjectRef, PrincipalRef

authz = pytest.importorskip("sightline.authz", reason="sightline.authz is not present yet")

check = authz.check
compile_plan = authz.compile_plan
derive_grant_token = authz.derive_grant_token
grant_tokens_for_object = authz.grant_tokens_for_object
recheck = authz.recheck
LiveRechecker = authz.LiveRechecker

DIM = 8
INDEXED_DOCS = ("doc:merger", "doc:handbook", "doc:bob-notes", "doc:payroll")


def _vector(i: int) -> list[float]:
    """Distinct, deterministic, and never all-zero. Content does not matter here."""
    return [1.0 if j == i % DIM else 0.25 for j in range(DIM)]


@pytest.fixture
def indexed(world):
    """A vector store stamped from the tuple store *as it is right now*.

    Every test below then changes the tuple store and leaves the index alone,
    which is exactly the condition the index is always in: slightly behind.
    """
    store = MemoryVectorStore(dim=DIM)
    chunks = []
    vectors = []
    for i, ref in enumerate(INDEXED_DOCS):
        obj = ObjectRef.parse(ref)
        chunks.append(
            Chunk(
                id=f"{obj.id}#0",
                object=obj,
                text=f"the contents of {ref}",
                grant_tokens=grant_tokens_for_object(world, obj),
            )
        )
        vectors.append(_vector(i))
    store.upsert(chunks, vectors)
    return store


def _unchecked(object_ref: str, chunk_id: str | None = None, token: str | None = None):
    return UncheckedHit(
        chunk_id=chunk_id or f"{ObjectRef.parse(object_ref).id}#0",
        score=0.9,
        text=f"the contents of {object_ref}",
        object_ref=object_ref,
        matched_token=token,
    )


# --------------------------------------------------------------------------
# The happy path, so the unhappy ones mean something
# --------------------------------------------------------------------------


def test_permitted_hits_survive_and_carry_their_derivation(world):
    survivors, dropped = recheck(world, "user:alice", [_unchecked("doc:merger")])
    assert dropped == 0
    assert len(survivors) == 1
    hit = survivors[0]
    assert isinstance(hit, Hit)
    assert hit.object_ref == "doc:merger"
    assert hit.why_allowed, "a survivor with no derivation cannot answer 'why am I allowed?'"
    assert hit.checked_at_epoch == world.epoch()


def test_recheck_returns_hit_not_uncheckedhit(world):
    """The type is the contract. ``Hit`` is the only thing the answer builder takes."""
    survivors, _ = recheck(world, "user:alice", [_unchecked("doc:merger")])
    assert type(survivors[0]) is Hit


# --------------------------------------------------------------------------
# A stale index loses results
# --------------------------------------------------------------------------


def test_revoked_grant_is_dropped_at_recheck(world, indexed):
    """Tighten the ACL after indexing; the index still offers it, recheck removes it."""
    alice = PrincipalRef("user", "alice")
    plan = compile_plan(world, alice)
    before = indexed.search(_vector(0), plan, 10)
    assert any(h.object_ref == "doc:merger" for h in before)

    world.delete("doc:merger#viewer@group:legal#member")

    # The index is untouched and still returns the chunk against the OLD plan,
    # which is precisely the window every "the index is authoritative" system
    # leaves open.
    stale_hits = indexed.search(_vector(0), plan, 10)
    assert any(h.object_ref == "doc:merger" for h in stale_hits)

    survivors, dropped = recheck(world, "user:alice", stale_hits)
    assert dropped >= 1
    assert all(h.object_ref != "doc:merger" for h in survivors)


def test_a_revoked_principal_gets_nothing_at_all(world, indexed):
    """Remove alice from the group entirely and every one of her hits dies."""
    alice = PrincipalRef("user", "alice")
    plan = compile_plan(world, alice)
    stale_hits = indexed.search(_vector(0), plan, 10)
    assert stale_hits, "fixture is broken: alice could see nothing before the revocation"

    world.delete("group:legal#member@user:alice")
    # doc:draft is hers by ownership, not by the group, so revoke that too.
    world.delete("doc:draft#owner@user:alice")

    survivors, dropped = recheck(world, "user:alice", stale_hits)
    assert survivors == []
    assert dropped == len(stale_hits)


def test_drops_are_counted_not_silent(world, indexed):
    """A silent drop is indistinguishable from a retrieval bug, so it is counted."""
    plan = compile_plan(world, PrincipalRef("user", "alice"))
    hits = indexed.search(_vector(0), plan, 10)
    world.delete("doc:handbook#viewer@group:everyone#member")
    survivors, dropped = recheck(world, "user:alice", hits)
    assert dropped == len(hits) - len(survivors)
    assert dropped > 0


def test_loosened_permissions_lose_results_rather_than_leak_them(world, indexed):
    """The other half of the asymmetry, and the cost this design accepts.

    Granting access to an already-indexed document does not make it findable: its
    chunks were stamped with the old token set. ``check()`` says yes and the query
    returns nothing until reindex. That is an availability bug on purpose.
    """
    dave = PrincipalRef("user", "dave")
    world.write("doc:merger#viewer@user:dave")
    assert check(world, ObjectRef("doc", "merger"), "viewer", dave).allowed is True

    plan = compile_plan(world, dave)
    hits = indexed.search(_vector(0), plan, 10)
    assert hits == [], "the index cannot know about a grant written after it was built"

    # And after a reindex of that one document, it is findable again.
    obj = ObjectRef("doc", "merger")
    indexed.upsert(
        [
            Chunk(
                id="merger#0",
                object=obj,
                text="the contents of doc:merger",
                grant_tokens=grant_tokens_for_object(world, obj),
            )
        ],
        [_vector(0)],
    )
    reindexed = indexed.search(_vector(0), compile_plan(world, dave), 10)
    assert [h.object_ref for h in reindexed] == ["doc:merger"]


# --------------------------------------------------------------------------
# What recheck refuses to trust
# --------------------------------------------------------------------------


def test_matched_token_is_not_trusted(world):
    """M4. The token the index matched is advisory and may name a dead group.

    Here it names a group the principal genuinely belongs to, attached to a
    document that group was never granted. Trusting the field would turn one
    forged payload into read access.
    """
    forged = _unchecked(
        "doc:bob-notes",
        token=str(derive_grant_token(PrincipalRef("group", "legal", "member"), "viewer")),
    )
    survivors, dropped = recheck(world, "user:alice", [forged])
    assert survivors == []
    assert dropped == 1


def test_every_hit_is_checked_not_just_the_first_page(world):
    """M10. Batching means "de-duplicate by object", never "check the first k"."""
    permitted = [_unchecked("doc:merger", chunk_id=f"merger#{i}") for i in range(30)]
    forbidden = [_unchecked("doc:bob-notes", chunk_id=f"bob#{i}") for i in range(30)]
    interleaved = [h for pair in zip(permitted, forbidden, strict=True) for h in pair]

    survivors, dropped = recheck(world, "user:alice", interleaved)
    assert len(survivors) == 30
    assert dropped == 30
    assert all(h.object_ref == "doc:merger" for h in survivors)
    # Order is preserved, so the store's ranking survives the authorisation pass.
    assert [h.chunk_id for h in survivors] == [f"merger#{i}" for i in range(30)]


def test_one_check_per_distinct_object(world, monkeypatch):
    """Thirty chunks of one document are one question, asked once.

    Asserted because it is the only optimisation in this file, and an
    optimisation nobody tests is an optimisation somebody later "fixes" into a
    correctness bug.
    """
    # ``import sightline.authz.recheck as m`` binds the re-exported *function*,
    # because the package exports a name that shadows its own submodule.
    # import_module goes to sys.modules and gets the module either way.
    recheck_module = import_module("sightline.authz.recheck")

    calls: list[str] = []
    real = recheck_module.check

    def counting(store, obj, relation, principal, **kwargs):
        calls.append(f"{obj}#{relation}")
        return real(store, obj, relation, principal, **kwargs)

    monkeypatch.setattr(recheck_module, "check", counting)
    hits = [_unchecked("doc:merger", chunk_id=f"merger#{i}") for i in range(30)]
    hits += [_unchecked("doc:handbook", chunk_id=f"handbook#{i}") for i in range(30)]
    survivors, _ = recheck(world, "user:alice", hits)

    assert len(survivors) == 60
    assert len(calls) == 2, f"one check per distinct object, got {len(calls)}"


def test_malformed_object_reference_is_denied(world):
    """An index entry we cannot even name is not an entry we serve."""
    junk = UncheckedHit(chunk_id="x", score=1.0, text="", object_ref="not-a-ref")
    survivors, dropped = recheck(world, "user:alice", [junk])
    assert survivors == []
    assert dropped == 1


def test_recheck_of_nothing_is_nothing(world):
    assert recheck(world, "user:dave", []) == ([], 0)


def test_principal_with_no_tuples_survives_nothing(world, indexed):
    """dave holds nothing. Every hit a forged plan could produce dies here."""
    everything = [_unchecked(ref) for ref in INDEXED_DOCS]
    survivors, dropped = recheck(world, "user:dave", everything)
    assert survivors == []
    assert dropped == len(everything)


def test_live_rechecker_is_the_same_function_with_a_store_attached(world):
    """The protocol exists so the pipeline depends on an interface, not so that
    recheck becomes swappable for something that does less."""
    rechecker = LiveRechecker(world)
    survivors, dropped = rechecker.recheck("user:alice", [_unchecked("doc:merger")])
    assert len(survivors) == 1 and dropped == 0
    assert rechecker.recheck("user:bob", [_unchecked("doc:merger")]) == ([], 1)
