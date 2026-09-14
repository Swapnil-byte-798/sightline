"""The differential oracle: does the compiled plan agree with the authority?

:func:`~sightline.authz.compile.compile_plan` is a projection of
:func:`~sightline.authz.check.check`, and a projection can be wrong in two
directions. This module samples ``(principal, object)`` pairs, asks both, and
reports the disagreements **asymmetrically**, because they are not the same kind
of thing:

* **false_allow** — the plan admits an object ``check()`` denies. This is a
  **breach**. Target exactly zero, CI-blocking, no tolerance, no "flaky test"
  interpretation available. One is a failed build.
* **false_deny** — the plan misses an object ``check()`` allows. This is a
  quality bug. Reported as a rate with a target below 0.1%, and it has real
  causes that are written down rather than hidden: the split depth budget in
  ``check.py`` gives up recall past eight levels on either side, and the
  enumeration cap gives up exactness past
  :data:`~sightline.authz.compile.MAX_ENUMERATION` objects.

Publishing that asymmetry is the point. A system that reports one number for
"accuracy" has averaged a breach together with a missing search result.

WHAT THIS DELIBERATELY DOES NOT MEASURE
---------------------------------------
Index staleness. Tokens are re-derived from the live store at the current epoch
rather than read out of a vector payload, so a disagreement here is a *compiler*
bug and nothing else. Staleness between the index and the store is recheck's
job, it is covered by the differential leak test in ``tests/``, and mixing the
two into one number would let a compiler bug hide behind "the index was behind".
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from sightline.authz.check import check
from sightline.authz.compile import (
    DEFAULT_NAMESPACE,
    DEFAULT_RELATION,
    compile_plan,
    grant_token_key,
    grant_tokens_for_object,
    plan_admits,
    principal_closure,
    reachable_objects,
)
from sightline.authz.tuples import TupleStore
from sightline.types import FilterPlan, ObjectRef, PrincipalRef, Relation

__all__ = [
    "FALSE_ALLOW_TARGET",
    "FALSE_DENY_TARGET",
    "NOBODY",
    "STRATA_BOUNDS",
    "OraclePair",
    "OracleResult",
    "Stratum",
    "run_oracle",
]

#: Not a threshold. A count, and the only acceptable one.
FALSE_ALLOW_TARGET = 0

#: A rate, because recall degrades gracefully and secrecy does not.
FALSE_DENY_TARGET = 0.001

#: Strata over group-closure size: how many usersets a principal transitively
#: holds. Stratifying on this is the difference between a sample that exercises
#: nested-group expansion and a sample that is 95% people in one group.
STRATA_BOUNDS: tuple[tuple[str, int, int], ...] = (
    ("0", 0, 0),
    ("1-2", 1, 2),
    ("3-8", 3, 8),
    ("9-32", 9, 32),
    ("33+", 33, 1 << 30),
)

#: A principal with no tuples at all, injected into every run. ADR 0003 says
#: empty means deny everything; the cheapest way to keep that honest is to make
#: sure the sample always contains somebody who is owed nothing.
NOBODY = PrincipalRef("user", "__oracle_nobody__")

# Disagreements are kept for the report, but a run that produces 40,000 of them
# does not need 40,000 of them printed.
_MAX_KEPT = 50


@dataclass(frozen=True, slots=True)
class OraclePair:
    """One sampled comparison. Serialisable, so a failure is reproducible."""

    principal: str
    object: str
    plan_allows: bool
    #: The grant-token condition evaluated on its own, regardless of which
    #: strategy the plan actually chose. See :func:`run_oracle` for why both
    #: verdicts are recorded.
    token_allows: bool
    check_allows: bool
    strategy: str
    closure_size: int
    stratum: str

    @property
    def false_allow(self) -> bool:
        return self.plan_allows and not self.check_allows

    @property
    def false_deny(self) -> bool:
        return self.check_allows and not self.plan_allows

    @property
    def token_false_allow(self) -> bool:
        return self.token_allows and not self.check_allows

    @property
    def token_false_deny(self) -> bool:
        return self.check_allows and not self.token_allows

    def to_dict(self) -> dict[str, object]:
        return {
            "principal": self.principal,
            "object": self.object,
            "plan_allows": self.plan_allows,
            "token_allows": self.token_allows,
            "check_allows": self.check_allows,
            "strategy": self.strategy,
            "closure_size": self.closure_size,
            "stratum": self.stratum,
        }


@dataclass(frozen=True, slots=True)
class Stratum:
    """Per-stratum totals, so a failure says *where* the compiler is wrong."""

    name: str
    n_principals: int
    n_pairs: int
    n_allowed: int
    false_allow: int
    false_deny: int
    token_false_allow: int = 0
    token_false_deny: int = 0

    @property
    def false_deny_rate(self) -> float:
        return self.false_deny / self.n_allowed if self.n_allowed else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_principals": self.n_principals,
            "n_pairs": self.n_pairs,
            "n_allowed": self.n_allowed,
            "false_allow": self.false_allow,
            "false_deny": self.false_deny,
            "false_deny_rate": round(self.false_deny_rate, 6),
            "token_false_allow": self.token_false_allow,
            "token_false_deny": self.token_false_deny,
        }


@dataclass(frozen=True, slots=True)
class OracleResult:
    """The structured verdict. ``to_dict()`` is what the eval harness commits."""

    n_pairs: int
    n_principals: int
    n_objects: int
    epoch: int
    relation: Relation
    namespace: str
    seed: int
    false_allow: int
    false_deny: int
    #: Denominator is the number of pairs ``check()`` allowed, not the number of
    #: pairs sampled. A false-deny rate over everything sampled would shrink
    #: simply by sampling more documents nobody can read.
    n_allowed: int
    #: The same comparison for the grant-token condition alone. On a small
    #: corpus the plan chooses ENUMERATE and these are the only numbers that say
    #: anything about the filter production actually runs.
    token_false_allow: int = 0
    token_false_deny: int = 0
    strata: tuple[Stratum, ...] = ()
    breaches: tuple[OraclePair, ...] = ()
    misses: tuple[OraclePair, ...] = field(default=(), repr=False)

    @property
    def false_deny_rate(self) -> float:
        return self.false_deny / self.n_allowed if self.n_allowed else 0.0

    @property
    def token_false_deny_rate(self) -> float:
        return self.token_false_deny / self.n_allowed if self.n_allowed else 0.0

    @property
    def breached(self) -> bool:
        """Any false allow at all, on either path. No acceptable non-zero value."""
        return (
            self.false_allow > FALSE_ALLOW_TARGET
            or self.token_false_allow > FALSE_ALLOW_TARGET
        )

    @property
    def passed(self) -> bool:
        return (
            not self.breached
            and self.false_deny_rate <= FALSE_DENY_TARGET
            and self.token_false_deny_rate <= FALSE_DENY_TARGET
        )

    def summary(self) -> str:
        return (
            f"oracle: {self.n_pairs} pairs, {self.n_allowed} allowed by check, "
            f"false_allow={self.false_allow}+{self.token_false_allow}tok "
            f"(target {FALSE_ALLOW_TARGET}), "
            f"false_deny={self.false_deny} ({self.false_deny_rate:.4%}) / "
            f"{self.token_false_deny} tok ({self.token_false_deny_rate:.4%}), "
            f"target {FALSE_DENY_TARGET:.1%} "
            f"-> {'PASS' if self.passed else 'FAIL'}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "n_pairs": self.n_pairs,
            "n_principals": self.n_principals,
            "n_objects": self.n_objects,
            "epoch": self.epoch,
            "relation": self.relation,
            "namespace": self.namespace,
            "seed": self.seed,
            "false_allow": self.false_allow,
            "false_allow_target": FALSE_ALLOW_TARGET,
            "false_deny": self.false_deny,
            "n_allowed": self.n_allowed,
            "false_deny_rate": round(self.false_deny_rate, 6),
            "false_deny_target": FALSE_DENY_TARGET,
            "token_false_allow": self.token_false_allow,
            "token_false_deny": self.token_false_deny,
            "token_false_deny_rate": round(self.token_false_deny_rate, 6),
            "breached": self.breached,
            "passed": self.passed,
            "strata": [s.to_dict() for s in self.strata],
            "breaches": [p.to_dict() for p in self.breaches],
            "misses": [p.to_dict() for p in self.misses],
        }


def _stratum_of(size: int) -> str:
    for name, lo, hi in STRATA_BOUNDS:
        if lo <= size <= hi:
            return name
    return STRATA_BOUNDS[-1][0]  # pragma: no cover - bounds are exhaustive


def _candidate_principals(store: TupleStore, limit: int) -> list[PrincipalRef]:
    """Concrete principals appearing anywhere in the tuple graph, plus NOBODY.

    Usersets are excluded as *subjects to sample*: the question the product
    answers is "may this person see this", and a group is not a person. Groups
    are still exercised — they are the path the walk takes to get to the person.
    """
    seen: dict[str, PrincipalRef] = {str(NOBODY): NOBODY}
    for t in store.iter_tuples():
        if t.principal.relation is None:
            seen.setdefault(str(t.principal), t.principal)
    # Sorted, and truncated after sorting. A store backed by a set iterates in
    # hash order, which varies per process; a sampler seeded on that is
    # reproducible only by accident, and "rerun it and see" is not a debugging
    # strategy for a permission failure.
    return [seen[k] for k in sorted(seen)][:limit]


def run_oracle(
    store: TupleStore,
    *,
    n_pairs: int = 1000,
    seed: int = 0,
    relation: Relation = DEFAULT_RELATION,
    namespace: str = DEFAULT_NAMESPACE,
    principals: Sequence[PrincipalRef] | None = None,
    objects: Sequence[ObjectRef] | None = None,
    max_principals: int = 2000,
    allowed_bias: float = 0.5,
) -> OracleResult:
    """Sample ``(principal, object)`` pairs and compare plan against ``check()``.

    ``allowed_bias`` is the fraction of pairs drawn from the principal's own
    reachable set rather than uniformly from the corpus. It defaults to half,
    and the reason is not subtle: on a realistic corpus a uniform sample almost
    never lands on a document the principal may read, so the false-deny
    denominator would be near zero and the published rate would be a flattering
    artefact of the sampler. Biasing toward allowed pairs measures recall
    honestly; the uniform half is what catches false allows.

    Deterministic for a given ``seed`` and store, so a CI failure is a
    reproduction rather than a report.
    """
    rng = random.Random(seed)
    epoch = store.epoch()
    key = grant_token_key()

    pool_principals = (
        list(principals) if principals is not None else _candidate_principals(store, max_principals)
    )
    pool_objects = (
        list(objects) if objects is not None else sorted(store.list_objects(namespace), key=str)
    )
    if not pool_principals or not pool_objects:
        return OracleResult(
            n_pairs=0,
            n_principals=len(pool_principals),
            n_objects=len(pool_objects),
            epoch=epoch,
            relation=relation,
            namespace=namespace,
            seed=seed,
            false_allow=0,
            false_deny=0,
            n_allowed=0,
        )

    # Stratify principals by group-closure size. The closure is computed once
    # per principal and reused for sampling, so this costs one reverse walk per
    # principal rather than one per pair.
    buckets: dict[str, list[PrincipalRef]] = {name: [] for name, _, _ in STRATA_BOUNDS}
    closures: dict[str, int] = {}
    reachable: dict[str, list[str]] = {}
    for p in pool_principals:
        closure = principal_closure(store, p)
        size = closure.size
        closures[str(p)] = size
        buckets[_stratum_of(size)].append(p)
        ids, _ = reachable_objects(store, closure, relation=relation, namespace=namespace)
        reachable[str(p)] = sorted(ids)

    live = [name for name, _, _ in STRATA_BOUNDS if buckets[name]]
    per_stratum = max(n_pairs // len(live), 1)

    plans: dict[str, FilterPlan] = {}
    token_cache: dict[str, frozenset[str]] = {}
    pairs: list[OraclePair] = []

    for name in live:
        for _ in range(per_stratum):
            principal = rng.choice(buckets[name])
            pkey = str(principal)
            mine = reachable[pkey]
            if mine and rng.random() < allowed_bias:
                obj = ObjectRef(namespace, rng.choice(mine))
            else:
                obj = rng.choice(pool_objects)

            plan = plans.get(pkey)
            if plan is None:
                plan = compile_plan(
                    store, principal, relation=relation, namespace=namespace, epoch=epoch, key=key
                )
                plans[pkey] = plan

            okey = str(obj)
            tokens = token_cache.get(okey)
            if tokens is None:
                tokens = frozenset(
                    str(t) for t in grant_tokens_for_object(store, obj, relation, key=key)
                )
                token_cache[okey] = tokens

            plan_verdict = plan_admits(plan, obj.id, tokens)
            # Evaluated separately from the plan's own strategy: on a fixture
            # corpus every plan is ENUMERATE, so without this line the token
            # filter — the one that runs in production against a real index —
            # would never be compared against the authority at all.
            token_verdict = bool(plan.grant_tokens & tokens)
            truth = check(store, obj, relation, principal).allowed
            pairs.append(
                OraclePair(
                    principal=pkey,
                    object=okey,
                    plan_allows=plan_verdict,
                    token_allows=token_verdict,
                    check_allows=truth,
                    strategy=plan.strategy.value,
                    closure_size=closures[pkey],
                    stratum=name,
                )
            )

    return _summarise(pairs, buckets, epoch, relation, namespace, seed, len(pool_objects))


def _summarise(
    pairs: list[OraclePair],
    buckets: dict[str, list[PrincipalRef]],
    epoch: int,
    relation: Relation,
    namespace: str,
    seed: int,
    n_objects: int,
) -> OracleResult:
    strata: list[Stratum] = []
    for name, members in buckets.items():
        if not members:
            continue
        mine = [p for p in pairs if p.stratum == name]
        strata.append(
            Stratum(
                name=name,
                n_principals=len(members),
                n_pairs=len(mine),
                n_allowed=sum(1 for p in mine if p.check_allows),
                false_allow=sum(1 for p in mine if p.false_allow),
                false_deny=sum(1 for p in mine if p.false_deny),
                token_false_allow=sum(1 for p in mine if p.token_false_allow),
                token_false_deny=sum(1 for p in mine if p.token_false_deny),
            )
        )
    breaches = [p for p in pairs if p.false_allow or p.token_false_allow]
    misses = [p for p in pairs if p.false_deny or p.token_false_deny]
    return OracleResult(
        n_pairs=len(pairs),
        n_principals=sum(len(m) for m in buckets.values()),
        n_objects=n_objects,
        epoch=epoch,
        relation=relation,
        namespace=namespace,
        seed=seed,
        false_allow=sum(1 for p in pairs if p.false_allow),
        false_deny=sum(1 for p in pairs if p.false_deny),
        n_allowed=sum(1 for p in pairs if p.check_allows),
        token_false_allow=sum(1 for p in pairs if p.token_false_allow),
        token_false_deny=sum(1 for p in pairs if p.token_false_deny),
        strata=tuple(strata),
        breaches=tuple(breaches[:_MAX_KEPT]),
        misses=tuple(misses[:_MAX_KEPT]),
    )
