"""The differential oracle, as a property test: the plan may never out-permit
``check()``.

``FilterPlan`` is a cache of an answer ``check()`` computes slowly. The two are
allowed to disagree in exactly one direction. A plan that admits a document
``check()`` denies is a **breach** — the index would hand the answer builder a
candidate the authority refuses, and the only thing left between that and a leak
is ``recheck()``, which is a second line of defence and not an excuse for a
broken first one. A plan that *misses* a document ``check()`` allows is a recall
bug: it costs availability, it is bounded by the split depth budget, and it is
reported rather than asserted on.

So the invariant is one-sided and absolute:

    false_allow == 0

There is no acceptable non-zero value and no rate to tune. ``false_deny`` is
printed and bounded loosely, because the subject/object depth split *guarantees*
some false denies on deep graphs and a strict assertion there would be a test
that fails for a documented design decision.

The test runs twice over the same generated policies:

* against the plan as compiled, whatever strategy it chose, and
* against the plan forced to ``GRANT_TOKENS``.

The second run is the one that matters. Small corpora compile to ``ENUMERATE``,
where the plan carries document ids and agreement is nearly tautological; the
token filter is what production actually executes at scale, and it is the path
where a subject-side and object-side walk have to meet in the middle without
ever overshooting.

Hypothesis drives it when installed. When it is not — the suite's floor is
pydantic, fastapi, numpy, httpx — the same generator runs from a seeded
``random.Random`` over a fixed list of seeds, so the property is still checked on
every machine and CI is still reproducible.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from sightline.types import ObjectRef, PlanStrategy, PrincipalRef

authz = pytest.importorskip("sightline.authz", reason="sightline.authz is not present yet")

MemoryTupleStore = authz.MemoryTupleStore
check = authz.check
compile_plan = authz.compile_plan
plan_admits = authz.plan_admits
grant_tokens_for_object = authz.grant_tokens_for_object

try:  # pragma: no cover - availability differs per environment, both paths tested
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    HAVE_HYPOTHESIS = True
except ImportError:  # pragma: no cover
    HAVE_HYPOTHESIS = False


# --------------------------------------------------------------------------
# Policy generation
# --------------------------------------------------------------------------


def random_policy(
    rng: random.Random,
    *,
    n_users: int,
    n_groups: int,
    n_docs: int,
    n_folders: int,
    n_edges: int,
) -> list[str]:
    """A random Zanzibar policy, including shapes that are meant to hurt.

    Cycles between groups, self-membership, folder chains, direct user grants,
    grants to a group nobody is in, and adversarially similar group names are all
    reachable. Nothing here is filtered for "sensible": the invariant under test
    must hold on policies no admin would write, because those are the ones real
    directories accumulate.
    """
    users = [f"user:u{i}" for i in range(n_users)]
    groups = [f"group:g{i}" for i in range(n_groups)]
    # Two names where one is a prefix of the other. Planted mutant M9 lives here.
    groups.append("group:g0-interns")
    docs = [f"doc:d{i}" for i in range(n_docs)]
    folders = [f"folder:f{i}" for i in range(n_folders)]

    rows: set[str] = set()
    for _ in range(n_edges):
        kind = rng.random()
        if kind < 0.30:  # person into group
            rows.add(f"{rng.choice(groups)}#member@{rng.choice(users)}")
        elif kind < 0.50:  # group into group (may close a cycle; that is the point)
            rows.add(f"{rng.choice(groups)}#member@{rng.choice(groups)}#member")
        elif kind < 0.70:  # document granted to a group
            rel = rng.choice(("viewer", "editor", "owner"))
            rows.add(f"{rng.choice(docs)}#{rel}@{rng.choice(groups)}#member")
        elif kind < 0.80:  # document granted to a person directly
            rel = rng.choice(("viewer", "owner"))
            rows.add(f"{rng.choice(docs)}#{rel}@{rng.choice(users)}")
        elif kind < 0.90 and folders:  # folder granted to a group
            rows.add(f"{rng.choice(folders)}#viewer@{rng.choice(groups)}#member")
        elif folders:  # containment, both spellings, sometimes folder-in-folder
            child = rng.choice(docs + folders)
            rows.add(f"{child}#parent@{rng.choice(folders)}")
    return sorted(rows)


@dataclasses.dataclass(frozen=True, slots=True)
class Disagreement:
    """One (principal, object) pair where plan and authority differ."""

    principal: str
    object: str
    plan_admits: bool
    check_allows: bool
    strategy: str

    def __str__(self) -> str:
        verdict = "FALSE ALLOW" if self.plan_admits else "false deny"
        return (
            f"{verdict}: {self.principal} / {self.object} "
            f"(plan={self.plan_admits}, check={self.check_allows}, "
            f"strategy={self.strategy})"
        )


def audit_policy(rows: list[str], *, force_tokens: bool) -> tuple[list[Disagreement], int, int]:
    """Compare every principal against every document. Returns the disagreements.

    Args:
        rows: Tuples to load into a fresh store.
        force_tokens: Rewrite each plan to ``GRANT_TOKENS`` before asking it.
            This is the production filter, and on a fixture this small it is the
            only path that says anything.

    Returns:
        ``(disagreements, n_pairs, n_allowed)``. ``n_allowed`` is the number of
        pairs the authority permitted — the honest denominator for a false-deny
        rate, and the number that proves the run was not vacuous.
    """
    store = MemoryTupleStore(rows)
    principals = sorted(
        {
            str(t.principal)
            for t in store.iter_tuples()
            if t.principal.relation is None and t.principal.namespace == "user"
        }
    )
    objects = sorted({str(t.object) for t in store.iter_tuples() if t.object.namespace == "doc"})
    if not principals or not objects:
        return [], 0, 0

    tokens = {obj: grant_tokens_for_object(store, ObjectRef.parse(obj)) for obj in objects}

    out: list[Disagreement] = []
    n_pairs = 0
    n_allowed = 0
    for who in principals:
        principal = PrincipalRef.parse(who)
        plan = compile_plan(store, principal, corpus_size=len(objects))
        if force_tokens:
            plan = dataclasses.replace(plan, strategy=PlanStrategy.GRANT_TOKENS)
        for obj in objects:
            n_pairs += 1
            ref = ObjectRef.parse(obj)
            admitted = plan_admits(plan, ref.id, tokens[obj])
            allowed = check(store, ref, "viewer", principal).allowed
            n_allowed += int(allowed)
            if admitted != allowed:
                out.append(
                    Disagreement(who, obj, admitted, allowed, plan.strategy.value)
                )
    return out, n_pairs, n_allowed


def assert_no_false_allow(rows: list[str], *, force_tokens: bool) -> tuple[int, int]:
    """The invariant. Returns ``(n_allowed, n_false_deny)`` for the caller's tally."""
    disagreements, _, n_allowed = audit_policy(rows, force_tokens=force_tokens)
    breaches = [d for d in disagreements if d.plan_admits]
    assert not breaches, (
        "the compiled plan admits what check() denies, which is a breach:\n  "
        + "\n  ".join(str(b) for b in breaches[:10])
        + "\n\npolicy:\n  "
        + "\n  ".join(rows)
    )
    return n_allowed, len(disagreements) - len(breaches)


# --------------------------------------------------------------------------
# The property, without hypothesis
# --------------------------------------------------------------------------

#: Fixed seeds, so a failure is a reproduction rather than a report. Widen this
#: list rather than making the corpus bigger: more shapes beat bigger shapes.
SEEDS = tuple(range(40))


@pytest.mark.parametrize("force_tokens", [False, True], ids=["as-compiled", "grant-tokens"])
def test_plan_never_admits_what_check_denies(force_tokens):
    """Forty seeded random policies, every principal against every document."""
    total_allowed = 0
    total_false_deny = 0
    for seed in SEEDS:
        rng = random.Random(seed)
        rows = random_policy(
            rng,
            n_users=rng.randint(1, 5),
            n_groups=rng.randint(1, 5),
            n_docs=rng.randint(1, 6),
            n_folders=rng.randint(0, 3),
            n_edges=rng.randint(4, 24),
        )
        allowed, false_deny = assert_no_false_allow(rows, force_tokens=force_tokens)
        total_allowed += allowed
        total_false_deny += false_deny

    # A property test that never generated an allow proves nothing at all.
    assert total_allowed > 50, (
        f"only {total_allowed} permitted pairs were generated; the sweep is close to "
        "vacuous and would pass against a compiler that denies everything"
    )
    # False denies are permitted by design (the split depth budget costs recall on
    # deep graphs) but a compiler that denies most of what it should allow is a
    # different bug wearing the same clothes.
    assert total_false_deny <= total_allowed * 0.10, (
        f"{total_false_deny} false denies against {total_allowed} allowed pairs: "
        "that is a recall collapse, not depth-budget loss"
    )


def test_the_sweep_would_catch_a_deliberate_over_permit():
    """Prove the harness can fail. A gate nobody has seen fail is a decoration.

    A plan with every token in the corpus is exactly the mistake this whole test
    exists to catch, so feeding one in must produce a breach.
    """
    rows = [
        "group:legal#member@user:alice",
        "doc:d1#viewer@group:legal#member",
        "doc:d2#viewer@group:secret#member",
    ]
    store = MemoryTupleStore(rows)
    every_token = frozenset().union(
        *(
            grant_tokens_for_object(store, ObjectRef("doc", d))
            for d in ("d1", "d2")
        )
    )
    over_permissive = dataclasses.replace(
        compile_plan(store, PrincipalRef("user", "alice")),
        strategy=PlanStrategy.GRANT_TOKENS,
        grant_tokens=every_token,
    )
    admitted = plan_admits(
        over_permissive, "d2", grant_tokens_for_object(store, ObjectRef("doc", "d2"))
    )
    denied = not check(store, ObjectRef("doc", "d2"), "viewer", PrincipalRef("user", "alice")).allowed
    assert admitted and denied, "the false-allow detector no longer detects a false allow"


# --------------------------------------------------------------------------
# The same property, with hypothesis
# --------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_HYPOTHESIS, reason="hypothesis is an optional dev dependency")
def test_plan_never_admits_what_check_denies_hypothesis():
    """Hypothesis shrinks a breach to the smallest policy that still breaches.

    The seeded sweep above catches the same class of bug; this one produces a
    two-tuple reproduction instead of a twenty-tuple one, which is the difference
    between a fix in an hour and a fix in a day.
    """

    @given(
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        n_users=st.integers(min_value=1, max_value=5),
        n_groups=st.integers(min_value=1, max_value=5),
        n_docs=st.integers(min_value=1, max_value=6),
        n_folders=st.integers(min_value=0, max_value=3),
        n_edges=st.integers(min_value=1, max_value=30),
        force_tokens=st.booleans(),
    )
    @settings(
        max_examples=150,
        deadline=None,  # check() over a deep graph is slow by design
        suppress_health_check=[HealthCheck.too_slow],
    )
    def prop(seed, n_users, n_groups, n_docs, n_folders, n_edges, force_tokens):
        rows = random_policy(
            random.Random(seed),
            n_users=n_users,
            n_groups=n_groups,
            n_docs=n_docs,
            n_folders=n_folders,
            n_edges=n_edges,
        )
        assert_no_false_allow(rows, force_tokens=force_tokens)

    prop()


# --------------------------------------------------------------------------
# The real oracle, when it is present
# --------------------------------------------------------------------------


def test_oracle_agrees_on_the_fixture_world(world):
    """``authz.oracle`` is the shipped version of this test. Run it here too.

    CI runs it as a blocking gate over a large generated corpus; this keeps it
    honest on the small one, where a regression shows up as a two-line diff.
    """
    oracle = pytest.importorskip(
        "sightline.authz.oracle", reason="sightline.authz.oracle is not present yet"
    )
    result = oracle.run_oracle(world, n_pairs=400, seed=1)
    assert result.false_allow == 0, result.summary()
    assert result.token_false_allow == 0, result.summary()
    assert not result.breached, result.summary()
    assert result.n_allowed > 0, "the oracle sampled no permitted pairs; the run is vacuous"
