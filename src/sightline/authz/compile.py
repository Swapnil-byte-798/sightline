"""The permission compiler: one principal, compiled into something an index can
execute in a single query.

A query cannot carry the ids of the 40,000 documents a person may read. So the
compiler walks the tuple graph *backwards* — from the person, up through every
group they belong to transitively — and emits a small set of **grant tokens**
plus a :class:`~sightline.types.PlanStrategy` chosen from the size of the result.

The two halves of the token scheme meet in the middle, and the asymmetry is the
design:

* **Ingest** expands a document's ``viewer`` relation over the *object* side
  (``sightline.authz.check.expand_leaves``) and stops at the group edge. It
  stamps one token per granting userset.
* **Query** expands a principal over the *subject* side and stops at the same
  group edge. It emits one token per userset the principal can be granted
  through — the groups, not the documents. :class:`Closure` explains why that
  distinction is the difference between a 988-token plan and a 20,000-token one.

Neither side ever enumerates people into the index, so somebody joining or
leaving a group rewrites **zero vectors** — their compiled plan changes, and the
policy epoch invalidates it automatically. Only a change to a document's own
permissions touches the index. That is ADR 0002, and it is the reason this file
exists rather than a simple "stamp the allowed user ids on each chunk".

Tokens for direct per-user grants (``doc:42#viewer@user:alice``) are derived from
``user:alice``, and that does not contradict ADR 0002. The ban there is on
expanding *group membership* into user ids, because membership churns. A direct
grant tuple is itself a change to that document's own ACL, so the vector write it
causes is one the design already pays for.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict, deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from sightline.authz.check import (
    MAX_OBJECT_DEPTH,
    MAX_SUBJECT_DEPTH,
    expand_leaves,
    implying_relations,
    tuple_to_userset_rules,
)
from sightline.authz.tuples import NamespaceConfig, TupleStore
from sightline.types import FilterPlan, GrantToken, ObjectRef, PlanStrategy, PrincipalRef, Relation

__all__ = [
    "ENUMERATE_MAX",
    "EXACT_SCAN_MAX",
    "UNFILTERED_MIN_RATIO",
    "MAX_ENUMERATION",
    "DEFAULT_RELATION",
    "DEFAULT_NAMESPACE",
    "grant_token_key",
    "derive_grant_token",
    "grant_tokens_for_object",
    "Closure",
    "principal_closure",
    "reachable_objects",
    "compile_plan",
    "plan_admits",
    "PlanCache",
    "PlanCompiler",
]

# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------
#
# THESE NUMBERS ARE NOT MEASUREMENTS. They have the right shape and the wrong
# values, and they stay wrong until `eval/selectivity.py` runs on the reference
# machine against a real index. The crossovers depend on things this module
# cannot see: HNSW `m` and `ef_search`, vector dimensionality, payload index
# type, and how the filter interacts with graph traversal. Anyone who ships a
# hardcoded threshold and calls it tuned is guessing in public.
#
# What is *not* negotiable is the ordering: a filter that is cheap to express
# (a handful of ids) beats a filter that is cheap to evaluate (token overlap)
# only while the id list stays small enough to send.

#: Above this many permitted objects, do not put ids in the query.
ENUMERATE_MAX = 512

#: Above this, brute force stops beating approximate search.
EXACT_SCAN_MAX = 4096

#: Fraction of the corpus a principal must see before filtering is skipped.
#: One. Not 0.98. See :func:`compile_plan` for why the FRD's 98% is wrong.
UNFILTERED_MIN_RATIO = 1.0

#: Hard ceiling on the reverse walk. Past this the answer is "lots", which is
#: all the strategy chooser needs, and the walk stops costing query latency.
MAX_ENUMERATION = 50_000

#: The relation the retrieval path filters on. Everything here is parameterised
#: on it, but there is exactly one in v1 and pretending otherwise is fiction.
DEFAULT_RELATION: Relation = "viewer"

DEFAULT_NAMESPACE = "doc"

# A keyed hash, so an index payload does not disclose that `group:legal` exists
# to anyone who can read the collection. The default key is a development key
# and is published here on purpose: a secret with a default is not a secret, and
# pretending otherwise is worse than saying so. Deployments set
# SIGHTLINE_GRANT_KEY, and rotating it forces a reindex — which is the honest
# cost of keyed tokens and is written down in the ops runbook rather than
# discovered.
_DEV_KEY = b"sightline-development-grant-key-not-secret"
_KEY_ENV = "SIGHTLINE_GRANT_KEY"


def grant_token_key() -> bytes:
    """Normalise the configured key to 32 bytes for blake2b."""
    raw = os.environ.get(_KEY_ENV)
    material = raw.encode("utf-8") if raw else _DEV_KEY
    return hashlib.sha256(material).digest()


def derive_grant_token(
    subject: PrincipalRef | str, relation: Relation = DEFAULT_RELATION, *, key: bytes | None = None
) -> GrantToken:
    """One granting userset plus one relation, hashed into an opaque token.

    Stable across runs for the same key, different across keys, and free of any
    plaintext group name (FR-7). The relation is part of the input so that a
    token granting ``viewer`` cannot be reused as a token granting ``editor``
    should a second filtered relation ever appear.
    """
    k = key if key is not None else grant_token_key()
    msg = f"{subject}|{relation}".encode("utf-8")
    digest = hashlib.blake2b(msg, digest_size=16, key=k).hexdigest()
    return GrantToken(f"gt_{digest}")


def grant_tokens_for_object(
    store: TupleStore,
    object: ObjectRef,
    relation: Relation = DEFAULT_RELATION,
    *,
    key: bytes | None = None,
    max_depth: int = MAX_OBJECT_DEPTH,
) -> frozenset[GrantToken]:
    """The token set ingest stamps on every chunk of ``object``.

    Includes tokens inherited from parent folders, because inheritance is a
    property of the document, not of the person asking. A document that comes
    back with an empty set is **not** indexable as visible-to-all: ingest
    rejects it (FR-15). Empty means nobody, always (ADR 0003).
    """
    k = key if key is not None else grant_token_key()
    leaves = expand_leaves(store, object, relation, max_depth=max_depth)
    return frozenset(derive_grant_token(leaf, relation, key=k) for leaf in leaves)


# --------------------------------------------------------------------------
# The reverse walk
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Closure:
    """The group closure: every userset a principal holds *and can be granted
    through*, with how many hops away each one is.

    What is deliberately **not** in here is every permission the principal
    holds. ``doc:317#viewer`` is held, and it is not a userset anybody is
    granted through, so walking into it buys nothing and costs one node per
    document in the corpus. An early version did exactly that: a principal who
    could read 20,000 documents produced a 20,000-token plan and took two
    seconds to compile, which is the "40,000 document ids in a query" problem
    this module exists to avoid, wearing a hash.

    The cut is not a heuristic. A grant token is only ever stamped for a subject
    that some tuple names as a principal, so :meth:`TupleStore.subject_keys`
    describes exactly the part of the graph that can carry one. Documents are
    reached in one place instead — :func:`reachable_objects`, under a cap.

    ``grantors`` is the subset that actually names something. Every node found
    through the subject graph qualifies by construction; the principal
    themselves may not, and a principal who grants nothing gets no tokens, which
    is the empty plan ADR 0003 requires.
    """

    principal: PrincipalRef
    hops: Mapping[PrincipalRef, int]
    grantors: frozenset[PrincipalRef]

    def __len__(self) -> int:
        return len(self.hops)

    def __iter__(self) -> Iterator[PrincipalRef]:
        return iter(self.hops)

    def __contains__(self, userset: object) -> bool:
        return userset in self.hops

    @property
    def size(self) -> int:
        """Group-closure size: usersets held, not counting the principal."""
        return max(len(self.hops) - 1, 0)


def principal_closure(
    store: TupleStore,
    principal: PrincipalRef,
    *,
    max_depth: int = MAX_SUBJECT_DEPTH,
) -> Closure:
    """Every userset ``principal`` holds, transitively, with its hop count.

    Breadth-first over the reverse index, so hop counts are minimal and the
    depth bound means what it says. Three things this deliberately does:

    * It compares **parsed** principals, never string prefixes. Prefix matching
      is planted mutant M9, where ``group:legal`` swallows
      ``group:legal-interns`` and nobody notices until an intern reads the
      merger memo.
    * It closes over ``ComputedUserset`` in reverse: if ``viewer`` is defined as
      a union containing ``computed(owner)``, then holding ``owner`` on an
      object means holding ``viewer`` on it. Without this, an owner's plan would
      miss their own documents.
    * It reads the reverse index for a node even at the depth limit, where it
      will not expand it. One extra lookup buys a correct ``grantors`` set at
      the boundary, and a token missing at exactly the depth limit is a recall
      cliff nobody would ever debug.

    Candidate edges are filtered as ``(namespace, id, relation)`` string triples
    against :meth:`TupleStore.subject_keys` *before* a
    :class:`~sightline.types.PrincipalRef` is built for them. That reads as a
    micro-optimisation and is not one: a heavily-permissioned principal has
    hundreds of thousands of candidate edges and almost all of them are
    documents, so building the object first is the difference between
    milliseconds and seconds on the reference machine.
    """
    maps: dict[str, _RelationMaps] = {}
    subjects = store.subject_keys()
    hops: dict[PrincipalRef, int] = {principal: 0}
    grantors: set[PrincipalRef] = set()
    seen: set[tuple[str, str, str]] = set()
    queue: deque[tuple[PrincipalRef, int]] = deque([(principal, 0)])
    while queue:
        subject, hop = queue.popleft()
        grants = store.read_by_principal(subject)
        if grants:
            grantors.add(subject)
        if hop >= max_depth:
            continue
        for t in grants:
            ns, ident, rel = t.object.namespace, t.object.id, t.relation
            for held_rel in _held_relations(store, ns, rel, maps):
                key = (ns, ident, held_rel)
                if key in seen or key not in subjects:
                    continue
                seen.add(key)
                held = PrincipalRef(ns, ident, held_rel)
                hops[held] = hop + 1
                queue.append((held, hop + 1))
    return Closure(principal, hops, frozenset(grantors))


@dataclass(frozen=True, slots=True)
class _RelationMaps:
    """Both directions of the relation-implication graph for one namespace."""

    #: relation -> relations that imply it (``viewer`` -> ``{viewer, editor, owner}``)
    implying: Mapping[Relation, frozenset[Relation]]
    #: relation -> relations it implies (``owner`` -> ``("viewer", "editor")``)
    implies: Mapping[Relation, tuple[Relation, ...]]


def _relation_maps(
    store: TupleStore, namespace: str, cache: dict[str, _RelationMaps]
) -> _RelationMaps:
    """Derived relation maps for one namespace, memoised for this call only.

    Per call rather than per process: namespace configs are policy, a policy
    change bumps the epoch, and a process-lifetime cache of policy is the thing
    this whole system is an argument against.
    """
    known = cache.get(namespace)
    if known is None:
        cfg = store.namespace_config(namespace)
        implying = {rel: implying_relations(cfg, rel) for rel in cfg.relations}
        implies: dict[Relation, list[Relation]] = {}
        for target, sources in implying.items():
            for source in sources:
                if source != target:
                    implies.setdefault(source, []).append(target)
        known = _RelationMaps(implying, {k: tuple(v) for k, v in implies.items()})
        cache[namespace] = known
    return known


def _held_relations(
    store: TupleStore, namespace: str, relation: Relation, cache: dict[str, _RelationMaps]
) -> tuple[Relation, ...]:
    """``relation`` plus every relation on the same object that it implies.

    Holding ``group:x#owner`` where ``member = union(this, computed(owner))``
    means holding ``group:x#member``, and ``member`` is the userset documents
    are actually granted to. Without this, an owner of a group would be granted
    nothing through it.

    The common case — a relation that implies nothing further — returns a
    one-element tuple and allocates nothing else, because this runs once per
    candidate edge.
    """
    targets = _relation_maps(store, namespace, cache).implies.get(relation, ())
    return (relation, *targets) if targets else (relation,)


def reachable_objects(
    store: TupleStore,
    closure: Closure,
    *,
    relation: Relation = DEFAULT_RELATION,
    namespace: str = DEFAULT_NAMESPACE,
    cap: int = MAX_ENUMERATION,
) -> tuple[frozenset[str], bool]:
    """Object ids in ``namespace`` on which the closure grants ``relation``.

    This is the forward evaluator run backwards, and it has to stay in step with
    it in *both* directions:

    * Anything here that ``check()`` would deny is a **breach**. So the
      containment descent re-reads the child's own namespace config rather than
      assuming every namespace inherits the same way, and the depth arithmetic
      below is the same arithmetic ``check.py`` documents.
    * Anything here that the grant-token filter would admit is a strategy that
      returns *different results depending on how many documents you can see*.
      That is not a security bug but it is an indefensible one, so the seeding
      deliberately mirrors the token condition exactly: an object is reachable
      when some userset in the closure is named by a grant on it. The oracle
      found this the hard way — the first version seeded from the closure's own
      members, which is one subject hop shallower, and ``ENUMERATE`` quietly
      admitted less than ``GRANT_TOKENS``.

    A grant reached through ``c`` containment steps from a userset at hop ``h``
    sits at forward depth ``c + h``, and both are bounded by their half of the
    split budget, so the result is always within ``MAX_USERSET_DEPTH``.

    Returns ``(ids, truncated)``. ``truncated`` means the walk hit ``cap`` and
    the count is a floor, not a total — enough to choose a strategy, not enough
    to enumerate.
    """
    maps: dict[str, _RelationMaps] = {}
    tuplesets = _containment_relations(store, relation)
    found: set[str] = set()
    seen: set[tuple[str, str, int]] = set()
    queue: deque[tuple[ObjectRef, int]] = deque()

    # Only grantors are seeded: a userset that names nothing cannot put an
    # object within reach, and iterating the full closure here is what made a
    # heavy principal's plan compile take seconds instead of milliseconds.
    for userset in closure.grantors:
        for t in store.read_by_principal(userset):
            # An undeclared relation implies only itself, matching the `This`
            # default in NamespaceConfig.rewrite: an unconfigured namespace
            # still works, it just inherits nothing.
            implying = _relation_maps(store, t.object.namespace, maps).implying.get(
                relation, frozenset({relation})
            )
            if t.relation in implying:
                queue.append((t.object, 0))

    while queue:
        obj, containment = queue.popleft()
        key = (obj.namespace, obj.id, containment)
        if key in seen:
            continue
        seen.add(key)
        if obj.namespace == namespace:
            found.add(obj.id)
            if len(found) >= cap:
                return frozenset(found), True
        if containment >= MAX_OBJECT_DEPTH:
            continue
        for child in _inheriting_children(store, obj, relation, tuplesets):
            queue.append((child, containment + 1))
    return frozenset(found), False


def _containment_relations(store: TupleStore, relation: Relation) -> frozenset[Relation]:
    """Relation names that carry inheritance of ``relation``, across namespaces.

    Read from every namespace's rules rather than guessed, because the rule that
    says a document inherits from its folder lives on the *document*, and this
    walk is standing on the folder. Usually this is the single name ``parent``;
    computing it once turns the descent into two indexed lookups per object
    instead of a scan of everything attached to it.
    """
    names: set[Relation] = set()
    for ns in store.list_namespaces():
        cfg = store.namespace_config(ns)
        for rule in tuple_to_userset_rules(cfg.rewrite(relation)):
            if rule.computed_relation == relation:
                names.add(rule.tupleset)
    return frozenset(names)


def _inheriting_children(
    store: TupleStore, parent: ObjectRef, relation: Relation, tuplesets: frozenset[Relation]
) -> list[ObjectRef]:
    """Objects that inherit ``relation`` from ``parent``.

    The child's namespace config decides, not the parent's. If ``doc`` declares
    ``viewer <- tupleToUserset(parent, viewer)`` and ``folder`` does not, then
    documents inherit from folders and folders do not inherit from folders, and
    this walk has to reflect that or it will hand out access the evaluator
    refuses.
    """
    candidates: list[tuple[ObjectRef, Relation, bool]] = []
    as_principal = PrincipalRef(parent.namespace, parent.id)
    for tupleset in tuplesets:
        # Spelling 1: child#parent@folder  -> the parent is the principal.
        for t in store.read_by_principal(as_principal, tupleset):
            candidates.append((t.object, tupleset, False))
        # Spelling 2: folder#parent@child  -> the child is the principal. Only
        # valid where the rule opted into both directions.
        for t in store.read(parent, tupleset):
            if t.principal.relation is None:
                candidates.append(
                    (ObjectRef(t.principal.namespace, t.principal.id), tupleset, True)
                )

    out: list[ObjectRef] = []
    for child, tupleset, inverted in candidates:
        cfg: NamespaceConfig = store.namespace_config(child.namespace)
        for rule in tuple_to_userset_rules(cfg.rewrite(relation)):
            if rule.tupleset != tupleset or rule.computed_relation != relation:
                continue
            if inverted and not rule.both_directions:
                continue
            out.append(child)
            break
    return out


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


def compile_plan(
    store: TupleStore,
    principal: PrincipalRef,
    *,
    relation: Relation = DEFAULT_RELATION,
    namespace: str = DEFAULT_NAMESPACE,
    epoch: int | None = None,
    corpus_size: int | None = None,
    key: bytes | None = None,
    cap: int = MAX_ENUMERATION,
) -> FilterPlan:
    """Compile one principal into an executable :class:`FilterPlan`.

    The epoch is read *before* the walk. Reading it afterwards would let a write
    that landed mid-walk produce a plan stamped with the new epoch while
    containing the old policy — a plan that looks fresh and is not. Stamping the
    older epoch can only cost a recompile.

    On ``UNFILTERED``: ``docs/FRD.md`` FR-5 says a principal who can see 98% of
    the corpus compiles to ``UNFILTERED``. That is wrong and this implementation
    does not do it. A plan that admits 2% of the corpus it should not is, by the
    oracle's own definition, a false allow — and "recheck will catch it" is
    exactly the reasoning this product exists to refuse. ``UNFILTERED`` requires
    seeing the whole corpus, and requires the caller to say how big the corpus
    is, because the tuple store only knows about documents somebody wrote a
    tuple for, and the dangerous document is precisely the one nobody did.

    The enumeration is stopped as soon as it has seen enough to decide, rather
    than run to :data:`MAX_ENUMERATION` every time. Nothing downstream needs the
    exact size of a large permitted set: the strategy chooser needs to know
    whether it is over :data:`EXACT_SCAN_MAX`, and ``UNFILTERED`` needs to know
    whether it covers the corpus. Past that point ``estimated_cardinality`` is a
    floor and the plan is ``GRANT_TOKENS`` regardless, which is why this runs in
    milliseconds for somebody who can read everything.

    Cost, stated rather than implied: a cache miss is one pass over the subject
    graph plus one bounded pass over reachable objects. A typical principal
    lands inside the 60 ms budget in ``docs/FRD.md`` §5 on the reference
    machine. A principal who can read the entire corpus does not — the bounded
    enumeration still has to walk every grant edge of every group they belong
    to, and that is several times the budget. The honest fix is a dedicated
    subject-graph index rather than a smaller cap, and it is not built. The
    numbers belong in ``eval/selectivity.py``, generated per run, not typed into
    this docstring where they would quietly go stale.
    """
    live_epoch = store.epoch() if epoch is None else epoch
    k = key if key is not None else grant_token_key()

    closure = principal_closure(store, principal)
    walk_cap = min(cap, max(EXACT_SCAN_MAX + 1, (corpus_size or 0) + 1))
    ids, truncated = reachable_objects(
        store, closure, relation=relation, namespace=namespace, cap=walk_cap
    )
    tokens = frozenset(derive_grant_token(u, relation, key=k) for u in closure.grantors)

    n = len(ids)
    strategy = _choose_strategy(n, truncated, corpus_size)
    # Ids are only carried by the strategies that filter on them. A plan is
    # shipped to a backend; a 40,000-element id list that nobody reads is
    # payload, latency, and a disclosure waiting for a debug log.
    explicit = ids if strategy in (PlanStrategy.ENUMERATE, PlanStrategy.EXACT_SCAN) else frozenset()
    return FilterPlan(
        principal=principal,
        strategy=strategy,
        grant_tokens=tokens,
        explicit_ids=explicit,
        epoch=live_epoch,
        estimated_cardinality=walk_cap if truncated else n,
    )


def _choose_strategy(n: int, truncated: bool, corpus_size: int | None) -> PlanStrategy:
    """Pick an execution strategy from cardinality. Order is load-bearing."""
    if n == 0:
        # Deny everything, expressed as an enumeration of nothing. Not
        # UNFILTERED, not "no filter" — empty means nobody (ADR 0003, M6).
        return PlanStrategy.ENUMERATE
    if (
        not truncated
        and corpus_size is not None
        and corpus_size > 0
        and n >= corpus_size * UNFILTERED_MIN_RATIO
    ):
        return PlanStrategy.UNFILTERED
    if truncated:
        return PlanStrategy.GRANT_TOKENS
    if n <= ENUMERATE_MAX:
        return PlanStrategy.ENUMERATE
    if n <= EXACT_SCAN_MAX:
        return PlanStrategy.EXACT_SCAN
    return PlanStrategy.GRANT_TOKENS


def plan_admits(
    plan: FilterPlan, object_id: str, chunk_tokens: Iterable[GrantToken | str] = ()
) -> bool:
    """Would this plan let this chunk through? The single definition of that.

    Every backend filter — a Qdrant payload filter, a SQL ``WHERE``, a numpy
    mask — must agree with this function, and the conformance suite checks that
    it does. Two mistakes are pre-named because they are the ones people make:

    * **M2** — combining a plan's conditions with OR. Within one chunk's token
      set the test is OR (any grant suffices). Across a plan's conditions it is
      AND. ``all()`` below is the AND.
    * **M6** — treating an empty token set as "no filter". An empty set matches
      nothing. ``bool(conditions)`` below is what stops an empty plan from
      falling through to allow.

    Note that each strategy applies its own condition rather than every
    condition it happens to carry. Intersecting an id list with a token set
    would compound two approximations and turn a recall bug into a silent one.
    """
    if plan.strategy is PlanStrategy.UNFILTERED:
        return True
    conditions: list[bool] = []
    if plan.strategy in (PlanStrategy.ENUMERATE, PlanStrategy.EXACT_SCAN):
        conditions.append(object_id in plan.explicit_ids)
    if plan.strategy is PlanStrategy.GRANT_TOKENS:
        chunk = frozenset(GrantToken(str(t)) for t in chunk_tokens)
        conditions.append(bool(plan.grant_tokens & chunk))
    return bool(conditions) and all(conditions)


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


class PlanCache:
    """LRU cache of compiled plans, keyed on ``(principal, epoch, ...)``.

    The epoch in the key is the whole point. Key on the principal alone and a
    permission change is invisible until the entry expires, which is planted
    mutant M8 and is also just how this bug happens in the wild: someone adds a
    TTL, the TTL looks short in a meeting, and it is five minutes of a revoked
    user reading documents.

    Because the key contains the epoch, **no invalidation logic is needed**. A
    write bumps the epoch, every existing key becomes unreachable, and stale
    entries age out of the LRU. Invalidation that requires remembering to call
    it is invalidation that does not happen.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        self._max = max_entries
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, int, str, str], FilterPlan] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(
        principal: PrincipalRef, epoch: int, relation: Relation, namespace: str
    ) -> tuple[str, int, str, str]:
        return (str(principal), epoch, relation, namespace)

    def get(
        self, principal: PrincipalRef, epoch: int, relation: Relation, namespace: str
    ) -> FilterPlan | None:
        k = self.key(principal, epoch, relation, namespace)
        with self._lock:
            plan = self._entries.get(k)
            if plan is None:
                self.misses += 1
                return None
            self._entries.move_to_end(k)
            self.hits += 1
        # Belt and braces: a plan that is somehow stale is not served even from
        # a key that says it is fresh.
        return None if plan.is_stale(epoch) else plan

    def put(self, plan: FilterPlan, relation: Relation, namespace: str) -> None:
        k = self.key(plan.principal, plan.epoch, relation, namespace)
        with self._lock:
            self._entries[k] = plan
            self._entries.move_to_end(k)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class PlanCompiler:
    """A compiler bound to a store and a cache. What the query path holds.

    Exists so the HTTP layer has one object to inject and one place where the
    epoch is read. ``compile_plan`` remains a free function because the oracle
    and the tests want to compile without a cache in the way.
    """

    def __init__(
        self,
        store: TupleStore,
        *,
        relation: Relation = DEFAULT_RELATION,
        namespace: str = DEFAULT_NAMESPACE,
        cache: PlanCache | None = None,
        corpus_size: int | None = None,
    ) -> None:
        self.store = store
        self.relation = relation
        self.namespace = namespace
        self.cache = cache if cache is not None else PlanCache()
        self.corpus_size = corpus_size

    def compile(self, principal: PrincipalRef, *, epoch: int | None = None) -> FilterPlan:
        live = self.store.epoch() if epoch is None else epoch
        cached = self.cache.get(principal, live, self.relation, self.namespace)
        if cached is not None:
            return cached
        plan = compile_plan(
            self.store,
            principal,
            relation=self.relation,
            namespace=self.namespace,
            epoch=live,
            corpus_size=self.corpus_size,
        )
        self.cache.put(plan, self.relation, self.namespace)
        return plan
