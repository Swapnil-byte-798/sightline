"""A permission write invalidates every cached plan, immediately.

The epoch is a monotonic counter bumped by every policy write. A
:class:`FilterPlan` carries the epoch it was compiled under, and a plan older
than the live policy is stale and must not be served. That is the entire
mechanism, and it is deliberately this small: invalidation that requires
remembering to call it is invalidation that does not happen.

Two planted mutants live in this file's blast radius:

* **M1** — ``is_stale()`` always returns ``False``. "The recheck catches it
  anyway" is a sentence that gets said in review and sounds reasonable.
* **M8** — the plan cache keyed on the principal alone, with a TTL. The TTL looks
  short in a meeting and is five minutes of a revoked user reading documents.

The cache key contains the epoch, so a write makes every existing key
unreachable and stale entries age out of the LRU on their own. The tests below
check that from both ends: the key, and the behaviour a user would see.
"""

from __future__ import annotations

import dataclasses
import sqlite3

import pytest

from sightline.store.memory import MemoryVectorStore
from sightline.types import Chunk, FilterPlan, ObjectRef, PlanStrategy, PrincipalRef

authz = pytest.importorskip("sightline.authz", reason="sightline.authz is not present yet")

MemoryTupleStore = authz.MemoryTupleStore
SQLiteTupleStore = authz.SQLiteTupleStore
NamespaceConfig = authz.NamespaceConfig
PlanCache = authz.PlanCache
PlanCompiler = authz.PlanCompiler
compile_plan = authz.compile_plan

ALICE = PrincipalRef("user", "alice")
DIM = 32


# --------------------------------------------------------------------------
# The counter
# --------------------------------------------------------------------------


def test_is_stale_compares_against_the_live_epoch():
    """M1 in one assertion. A plan from an older policy is stale; equal is not."""
    plan = FilterPlan(principal=ALICE, strategy=PlanStrategy.ENUMERATE, epoch=7)
    assert plan.is_stale(8) is True
    assert plan.is_stale(7) is False
    assert plan.is_stale(6) is False, "a plan from the future is a clock bug, not staleness"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_every_policy_write_moves_the_epoch(backend, world):
    store = world if backend == "memory" else SQLiteTupleStore(":memory:")
    if backend == "sqlite":
        store.write("doc:merger#viewer@group:legal#member")

    start = store.epoch()
    assert store.write("doc:new#viewer@user:alice") > start
    assert store.delete("doc:new#viewer@user:alice") > start + 1
    # A no-op write still bumps: the caller cannot tell whether it changed
    # anything, and a bump that only sometimes happens is a bump nobody can
    # reason about.
    before = store.epoch()
    assert store.write("doc:merger#viewer@group:legal#member") > before


def test_redefining_a_namespace_moves_the_epoch(world):
    """A namespace change alters what already-stored tuples *mean*.

    No tuple moved and every compiled plan is still wrong, which is exactly the
    case a tuple-count-based invalidation would miss.
    """
    before = world.epoch()
    world.define_namespace(NamespaceConfig("doc", {"viewer": authz.This()}))
    assert world.epoch() > before


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------


def test_the_cache_key_contains_the_epoch():
    """M8. Key on the principal alone and a permission change is invisible."""
    a = PlanCache.key(ALICE, 1, "viewer", "doc")
    b = PlanCache.key(ALICE, 2, "viewer", "doc")
    assert a != b


def test_a_write_makes_the_cached_entry_unreachable(world):
    cache = PlanCache()
    epoch = world.epoch()
    plan = compile_plan(world, ALICE, epoch=epoch)
    cache.put(plan, "viewer", "doc")
    assert cache.get(ALICE, epoch, "viewer", "doc") is plan

    world.write("doc:anything#viewer@user:zed")
    assert cache.get(ALICE, world.epoch(), "viewer", "doc") is None, (
        "the cache served a plan compiled under the previous policy"
    )


def test_a_stale_plan_is_refused_even_from_a_key_that_looks_fresh(world):
    """Belt and braces: the key says fresh, the plan says otherwise, deny.

    This branch only fires if somebody mis-stamps a plan, which is a bug — and
    the cheap defence against that bug is one comparison on the read path.
    """
    cache = PlanCache()
    mis_stamped = dataclasses.replace(compile_plan(world, ALICE), epoch=0)
    cache._entries[PlanCache.key(ALICE, 5, "viewer", "doc")] = mis_stamped  # noqa: SLF001
    assert cache.get(ALICE, 5, "viewer", "doc") is None


def test_the_cache_hits_when_nothing_has_changed(world):
    """The invalidation must not be so eager that the cache is decorative."""
    compiler = PlanCompiler(world)
    first = compiler.compile(ALICE)
    second = compiler.compile(ALICE)
    assert first is second
    assert compiler.cache.hits == 1


def test_the_compiler_recompiles_after_a_write(world):
    """And the new plan reflects the new policy, not just a new epoch."""
    compiler = PlanCompiler(world)
    before = compiler.compile(ALICE)
    assert "merger" in before.explicit_ids

    world.delete("doc:merger#viewer@group:legal#member")
    after = compiler.compile(ALICE)

    assert after is not before
    assert after.epoch > before.epoch
    assert "merger" not in after.explicit_ids


def test_a_grant_is_visible_on_the_next_compile(world):
    """The other direction. Cache invalidation that only works for revocations
    is a support ticket every time somebody is added to a group."""
    compiler = PlanCompiler(world)
    dave = PrincipalRef("user", "dave")
    assert compiler.compile(dave).explicit_ids == frozenset()
    world.write("doc:handbook#viewer@user:dave")
    assert compiler.compile(dave).explicit_ids == frozenset({"handbook"})


def test_the_cache_evicts_rather_than_growing_without_bound(world):
    cache = PlanCache(max_entries=4)
    for i in range(10):
        cache.put(
            FilterPlan(principal=PrincipalRef("user", f"u{i}"), strategy=PlanStrategy.ENUMERATE),
            "viewer",
            "doc",
        )
    assert len(cache) == 4


# --------------------------------------------------------------------------
# What a user would see
# --------------------------------------------------------------------------


@pytest.fixture
def retriever(world):
    """A pipeline whose index holds exactly one document: ``doc:merger``."""
    retrieve = pytest.importorskip(
        "sightline.retrieve", reason="sightline.retrieve is not present yet"
    )
    embed = pytest.importorskip(
        "sightline.ingest.embed", reason="sightline.ingest.embed is not present yet"
    )
    settings_mod = pytest.importorskip("sightline.settings")

    embedder = embed.HashEmbedder(dim=DIM, quiet=True)
    text = "the merger memo, which alice may read until the moment she may not"
    obj = ObjectRef("doc", "merger")
    store = MemoryVectorStore(dim=DIM)
    store.upsert(
        [
            Chunk(
                id="merger#0",
                object=obj,
                text=text,
                grant_tokens=authz.grant_tokens_for_object(world, obj),
            )
        ],
        [embedder.encode([text])[0]],
    )
    base = settings_mod.Settings()
    settings = base.with_(
        retrieval=dataclasses.replace(base.retrieval, refusal_floor_ms=5.0)
    )
    return retrieve.Retriever(
        tuple_store=world, vector_store=store, embedder=embedder, settings=settings
    )


def test_revocation_takes_effect_on_the_very_next_query(retriever, world):
    """The end-to-end version, through a warm plan cache.

    No sleep, no TTL, no "eventually". The write bumps the epoch, the next query
    reads the new epoch, and the cached plan is unreachable by construction.
    """
    first = retriever.search("merger memo", ALICE)
    assert [h.object_ref for h in first.hits] == ["doc:merger"]
    assert retriever.compiler.cache.hits >= 0  # warm it, then invalidate it

    world.delete("doc:merger#viewer@group:legal#member")

    second = retriever.search("merger memo", ALICE)
    assert second.hits == ()
    assert second.refused is True
    assert second.epoch > first.epoch


def test_a_plan_that_cannot_be_refreshed_refuses_rather_than_serving(retriever, world):
    """When recompilation is disabled, a stale plan is a 409, not a result.

    Simulated with a compiler that insists on stamping the previous epoch —
    which is what a replica lagging behind the primary looks like from here.
    """
    errors = pytest.importorskip("sightline.errors")

    class LaggingCompiler(PlanCompiler):
        def compile(self, principal, *, epoch=None):
            plan = super().compile(principal, epoch=epoch)
            return dataclasses.replace(plan, epoch=max(plan.epoch - 1, 0))

    retriever.compiler = LaggingCompiler(world)
    retriever.settings = retriever.settings.with_(
        retrieval=dataclasses.replace(retriever.settings.retrieval, recompile_on_stale=False)
    )
    with pytest.raises(errors.StalePolicy):
        retriever.search("merger memo", ALICE)


# --------------------------------------------------------------------------
# The increment is part of the write
# --------------------------------------------------------------------------


class _EpochBumpFails:
    """A connection proxy that fails only on the epoch increment.

    Stands in for a crash between the tuple write and the counter bump. The
    transaction semantics are the real connection's, so the rollback under test
    is sqlite's and not a fake.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, *args):
        if "UPDATE policy" in sql:
            raise sqlite3.OperationalError("induced crash between write and epoch bump")
        return self._conn.execute(sql, *args)

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, *exc):
        return self._conn.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_the_epoch_increment_is_inside_the_write_transaction():
    """M15. A crash between the two must leave neither, never just the tuple.

    A permission change that landed without a bump is invisible to every cached
    plan — the leak this counter exists to prevent, arriving through the door
    marked "bookkeeping".
    """
    store = SQLiteTupleStore(":memory:")
    store.write("doc:public#viewer@group:everyone#member")
    epoch_before = store.epoch()

    real_conn = store._conn  # noqa: SLF001
    store._conn = _EpochBumpFails(real_conn)  # noqa: SLF001
    try:
        with pytest.raises(sqlite3.OperationalError):
            store.write("doc:secret#viewer@user:mallory")
    finally:
        store._conn = real_conn  # noqa: SLF001

    assert store.epoch() == epoch_before
    assert store.read(ObjectRef("doc", "secret"), "viewer") == [], (
        "the tuple write committed without its epoch bump: every cached plan is "
        "now entitled to ignore a permission change that really happened"
    )
