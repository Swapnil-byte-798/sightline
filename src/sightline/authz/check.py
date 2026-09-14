"""``Check()`` — the authoritative answer to "may this principal do this?".

Everything else in Sightline is a cache, an index, or an approximation of this
function. It is allowed to be the slowest thing in the repo and it is not
allowed to be wrong. When a compiled plan and this evaluator disagree, this one
is right by definition, and :mod:`sightline.authz.oracle` exists to find the
disagreements.

Three properties matter more than speed:

**It terminates.** Group graphs in real organisations contain cycles — "leads"
contains "seniors" contains "leads", written two years apart by two admins. A
cycle terminates its branch and denies; it does not raise and it does not grant.

**It is bounded.** Expansion stops at :data:`MAX_USERSET_DEPTH` and the limit
resolves toward *deny*, recording ``depth_limit_reached`` in the decision.
Truncating toward allow is planted mutant M5, and it is the kind of bug that
only shows up on the one customer with a 20-deep group hierarchy.

**It shows its work.** :class:`~sightline.types.Decision` carries the derivation
path as a list of tuple strings, in order, each of which a test (or a human, or
the admin console) can replay one at a time. An access decision nobody can audit
is a liability dressed as a feature.

THE DEPTH BUDGET IS SPLIT, ON PURPOSE
-------------------------------------
Token stamping (``expand`` over the object side, at ingest) and plan compilation
(the closure over the subject side, at query time) run at different moments and
neither can see how much depth the other consumed. If both were allowed the full
:data:`MAX_USERSET_DEPTH`, a token match could imply a path of up to twice that
depth — a path ``check()`` would refuse — and the compiler would be *more*
permissive than the authority. That is a false allow, which this project calls a
breach.

So the budget is split: :data:`MAX_OBJECT_DEPTH` for the object side,
:data:`MAX_SUBJECT_DEPTH` for the subject side, and they sum to
:data:`MAX_USERSET_DEPTH`. Any token match therefore implies a derivation within
``check()``'s own bound. The cost is recall: a grant that needs more than
:data:`MAX_OBJECT_DEPTH` folder levels or more than :data:`MAX_SUBJECT_DEPTH`
nested groups is visible to ``check()`` and invisible to the index. The oracle
reports that as a false *deny*, which is a quality bug, not a breach.

Say the consequence plainly, because ``docs/FRD.md`` §5 claims "nesting depth up
to 16" as a capacity number: ``check()`` meets that and the **search path does
not**. Search tops out at eight nested groups. A nine-deep group chain is
answered correctly by ``/v1/explain`` and returns nothing from ``/v1/query``,
and the oracle prints that as a false-deny rate rather than leaving it to be
discovered. Meeting the published number is a one-line change — raise
:data:`MAX_USERSET_DEPTH` to 32 and split it 16/16 — and it costs deeper walks
on every check. It is not made here because nobody has measured what that costs
on the reference machine, and a limit raised to match a document rather than a
measurement is how the latency budget gets spent by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sightline.authz.tuples import (
    ComputedUserset,
    NamespaceConfig,
    Rewrite,
    This,
    TupleToUserset,
    Union,
)
from sightline.types import Decision, ObjectRef, PrincipalRef, Relation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sightline.authz.tuples import TupleStore

__all__ = [
    "DEPTH_LIMIT_NOTE",
    "MAX_OBJECT_DEPTH",
    "MAX_SUBJECT_DEPTH",
    "MAX_USERSET_DEPTH",
    "Explanation",
    "UsersetTree",
    "check",
    "expand",
    "expand_leaves",
    "explain",
    "implying_relations",
    "tuple_to_userset_rules",
]

#: Hard bound on userset expansion. Reaching it denies.
MAX_USERSET_DEPTH = 16

#: Object-side share of the budget (folder nesting, computed relations).
MAX_OBJECT_DEPTH = 8

#: Subject-side share of the budget (nested groups).
MAX_SUBJECT_DEPTH = 8

# If a future edit breaks this, the compiler can out-permit the authority. Fail
# at import rather than at 3am.
assert MAX_OBJECT_DEPTH + MAX_SUBJECT_DEPTH == MAX_USERSET_DEPTH

#: Exact string a denial carries when the bound stopped the walk. Tested for.
DEPTH_LIMIT_NOTE = "depth_limit_reached"

# A denial over a wide graph can try thousands of edges. The derivation is for a
# human, so the diagnostic list is capped and says that it was capped.
_MAX_DENY_NOTES = 64


def implying_relations(cfg: NamespaceConfig, relation: Relation) -> frozenset[Relation]:
    """Relations on the same object that imply ``relation``.

    ``viewer = union(this, computed(editor))`` and ``editor = union(this,
    computed(owner))`` means holding ``owner`` implies ``viewer``; this returns
    ``{viewer, editor, owner}``. Used by plan compilation to normalise a held
    relation to the one the index filters on, so the object side and the subject
    side meet at the same name.
    """
    implying: set[Relation] = {relation}
    frontier = [relation]
    while frontier:
        rel = frontier.pop()
        for target in _computed_targets(cfg.rewrite(rel)):
            # `viewer = union(this, computed(editor))` says an editor is a
            # viewer, so `editor` implies `viewer`. The arrow in the config
            # points from the weaker relation to the stronger one, and reading
            # it the other way round is the bug the oracle caught first.
            if target not in implying:
                implying.add(target)
                frontier.append(target)
    return frozenset(implying)


def _computed_targets(rw: Rewrite) -> list[Relation]:
    """Relations reachable from ``rw`` by ``ComputedUserset`` on the same object."""
    match rw:
        case ComputedUserset(relation=rel):
            return [rel]
        case Union(children=kids):
            out: list[Relation] = []
            for kid in kids:
                out.extend(_computed_targets(kid))
            return out
        case _:
            return []


def tuple_to_userset_rules(rw: Rewrite) -> list[TupleToUserset]:
    """Every ``tupleToUserset`` rule inside a rewrite, flattened.

    Public because plan compilation has to walk inheritance in reverse, and it
    must read the *same* rules this evaluator reads. Two independent notions of
    what inherits from what is how a compiler out-permits its authority.
    """
    match rw:
        case TupleToUserset():
            return [rw]
        case Union(children=kids):
            out: list[TupleToUserset] = []
            for kid in kids:
                out.extend(tuple_to_userset_rules(kid))
            return out
        case _:
            return []


# --------------------------------------------------------------------------
# check()
# --------------------------------------------------------------------------


class _Evaluator:
    """One evaluation. Memoised within the call and thrown away after it.

    Memoisation is per call rather than per store because the store is live: a
    cache that outlives the call is a cache that can serve a revoked answer, and
    the entire product is an argument against exactly that.
    """

    def __init__(self, store: TupleStore, principal: PrincipalRef, max_depth: int) -> None:
        self._store = store
        self._principal = principal
        self._max_depth = max_depth
        self.checked = 0
        self.deepest = 0
        self._notes: list[str] = []
        self._stack: set[tuple[str, str, str]] = set()
        # key -> (allowed, path, height). ``height`` is how much depth the
        # subtree consumed below this node; see _reuse().
        self._memo: dict[tuple[str, str, str], tuple[bool, tuple[str, ...], int]] = {}
        self._truncations = 0

    # -- diagnostics -------------------------------------------------------
    def note(self, text: str) -> None:
        if len(self._notes) < _MAX_DENY_NOTES:
            self._notes.append(text)
        elif len(self._notes) == _MAX_DENY_NOTES:
            self._notes.append(f"... diagnostics truncated at {_MAX_DENY_NOTES} entries")

    @property
    def notes(self) -> tuple[str, ...]:
        return tuple(self._notes)

    # -- evaluation --------------------------------------------------------
    def holds(self, obj: ObjectRef, rel: Relation, depth: int) -> tuple[bool, tuple[str, ...], int]:
        """Does ``self._principal`` hold ``rel`` on ``obj``?

        Returns ``(allowed, derivation_path, height)``. ``height`` is the extra
        depth the answer consumed below this node, and it is what makes the memo
        safe to reuse at a different depth.
        """
        if depth > self._max_depth:
            self._truncations += 1
            self.note(f"{DEPTH_LIMIT_NOTE} at {obj}#{rel} (limit {self._max_depth})")
            return (False, (), 0)
        # Recorded after the bound check, so a depth-limited walk reports the
        # limit rather than one past it. Explain renders this number.
        self.deepest = max(self.deepest, depth)

        key = (obj.namespace, obj.id, rel)
        if key in self._stack:
            # A cycle terminates the branch. It is not an error and it is not a
            # grant: if the only way to reach the principal is through a loop,
            # there is no derivation.
            self._truncations += 1
            self.note(f"cycle: {obj}#{rel}")
            return (False, (), 0)

        cached = self._memo.get(key)
        if cached is not None and self._reuse(cached, depth):
            return cached

        self._stack.add(key)
        before = self._truncations
        try:
            cfg = self._store.namespace_config(obj.namespace)
            result = self._eval(cfg.rewrite(rel), obj, rel, depth)
        finally:
            self._stack.discard(key)

        if self._truncations == before:
            # Only complete answers are cacheable. A result produced under a
            # truncated walk is an artefact of where we happened to be in the
            # graph, and reusing it elsewhere would be wrong in both directions.
            self._memo[key] = result
        return result

    def _reuse(self, cached: tuple[bool, tuple[str, ...], int], depth: int) -> bool:
        """May a memoised answer be reused at this depth?

        A complete *deny* is reusable anywhere: less budget can only explore
        less. A complete *allow* is reusable only if its derivation still fits —
        otherwise a shallow success would smuggle a too-deep path into a context
        where ``check()`` should have refused, and the depth bound would be a
        suggestion rather than a bound.
        """
        allowed, _, height = cached
        if not allowed:
            return True
        return depth + height <= self._max_depth

    def _eval(
        self, rw: Rewrite, obj: ObjectRef, rel: Relation, depth: int
    ) -> tuple[bool, tuple[str, ...], int]:
        match rw:
            case This():
                return self._eval_this(obj, rel, depth)
            case ComputedUserset(relation=other):
                # Same object, so no graph hop and no depth charge. The cycle
                # stack still bounds it: relations are finite per namespace.
                return self.holds(obj, other, depth)
            case TupleToUserset():
                return self._eval_ttu(rw, obj, depth)
            case Union(children=kids):
                height = 0
                for kid in kids:
                    allowed, path, kid_height = self._eval(kid, obj, rel, depth)
                    height = max(height, kid_height)
                    if allowed:
                        return (True, path, kid_height)
                return (False, (), height)
        raise TypeError(f"unknown rewrite: {rw!r}")  # pragma: no cover - defensive

    def _eval_this(
        self, obj: ObjectRef, rel: Relation, depth: int
    ) -> tuple[bool, tuple[str, ...], int]:
        tuples = self._store.read(obj, rel)
        self.checked += len(tuples)
        height = 0
        # Direct grants first: cheapest, and the most common shape.
        for t in tuples:
            if t.principal == self._principal:
                return (True, (str(t),), 0)
        for t in tuples:
            if t.principal.relation is None:
                continue  # a concrete principal that is not ours
            sub_obj = ObjectRef(t.principal.namespace, t.principal.id)
            allowed, path, sub_height = self.holds(sub_obj, t.principal.relation, depth + 1)
            height = max(height, 1 + sub_height)
            if allowed:
                return (True, (str(t), *path), 1 + sub_height)
        if not tuples:
            self.note(f"no tuples for {obj}#{rel}")
        else:
            # Name the edges that were tried, not just the failure. "Denied" on
            # its own is unactionable for the admin who has to fix it.
            self.note(f"no grant to {self._principal} in {obj}#{rel} ({len(tuples)} tried)")
        return (False, (), height)

    def _eval_ttu(
        self, rw: TupleToUserset, obj: ObjectRef, depth: int
    ) -> tuple[bool, tuple[str, ...], int]:
        height = 0
        for parent, via in self._containers(rw, obj):
            allowed, path, sub_height = self.holds(parent, rw.computed_relation, depth + 1)
            height = max(height, 1 + sub_height)
            if allowed:
                return (True, (via, *path), 1 + sub_height)
        return (False, (), height)

    def _containers(self, rw: TupleToUserset, obj: ObjectRef) -> list[tuple[ObjectRef, str]]:
        """Objects ``obj`` inherits from, with the tuple string that says so.

        Both spellings of the containment tuple are read; see
        :class:`~sightline.authz.tuples.TupleToUserset` for why.
        """
        out: list[tuple[ObjectRef, str]] = []
        forward = self._store.read(obj, rw.tupleset)
        self.checked += len(forward)
        for t in forward:
            out.append((ObjectRef(t.principal.namespace, t.principal.id), str(t)))
        if rw.both_directions:
            inverted = self._store.read_by_principal(
                PrincipalRef(obj.namespace, obj.id), rw.tupleset
            )
            self.checked += len(inverted)
            for t in inverted:
                out.append((t.object, str(t)))
        return out


def check(
    store: TupleStore,
    object: ObjectRef,
    relation: Relation,
    principal: PrincipalRef,
    *,
    max_depth: int = MAX_USERSET_DEPTH,
) -> Decision:
    """Authoritative permission check. Default deny.

    Absence of a tuple is denial — never "unknown", never "allow". The returned
    :class:`~sightline.types.Decision` carries the derivation in ``why``: for an
    allow that is an ordered list of tuple strings which replay to the same
    answer one at a time; for a deny it is the list of edges that were tried,
    including ``depth_limit_reached`` and ``cycle:`` markers, because "why not"
    is the question an admin actually asks.
    """
    ev = _Evaluator(store, principal, max_depth)
    allowed, path, _ = ev.holds(object, relation, 0)
    why = path if allowed else ev.notes
    return Decision(allowed=allowed, why=why, checked_tuples=ev.checked)


# --------------------------------------------------------------------------
# expand()
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UsersetTree:
    """The object-side expansion of one relation: who grants it, structurally.

    The leaves are **subjects as written** — ``group:legal#member``,
    ``user:alice`` — and group membership is deliberately *not* followed. That
    asymmetry is the whole grant-token design (ADR 0002): the object side stops
    at the group edge, the subject side starts from the person and walks up to
    it, and the two meet at the userset. Expanding membership here would put
    people's ids in the index and turn every membership change into a vector
    rewrite storm.
    """

    object: ObjectRef
    relation: Relation
    rewrite: str
    subjects: tuple[PrincipalRef, ...] = ()
    children: tuple[UsersetTree, ...] = ()
    #: ``"cycle"`` or ``"depth"`` when this branch was cut short.
    truncated: str | None = None

    def leaves(self) -> frozenset[PrincipalRef]:
        """Every granting subject in the tree, flattened."""
        found = set(self.subjects)
        for kid in self.children:
            found |= kid.leaves()
        return frozenset(found)

    def to_dict(self) -> dict[str, object]:
        return {
            "object": str(self.object),
            "relation": self.relation,
            "rewrite": self.rewrite,
            "subjects": [str(s) for s in self.subjects],
            "children": [k.to_dict() for k in self.children],
            "truncated": self.truncated,
        }


def expand(
    store: TupleStore,
    object: ObjectRef,
    relation: Relation,
    *,
    max_depth: int = MAX_OBJECT_DEPTH,
    _seen: frozenset[tuple[str, str, str]] = frozenset(),
    _depth: int = 0,
) -> UsersetTree:
    """Expand one relation on one object into the subjects that grant it.

    Bounded by :data:`MAX_OBJECT_DEPTH` rather than :data:`MAX_USERSET_DEPTH`;
    the module docstring explains why the budget is split.
    """
    key = (object.namespace, object.id, relation)
    if key in _seen:
        return UsersetTree(object, relation, "this", truncated="cycle")
    if _depth > max_depth:
        return UsersetTree(object, relation, "this", truncated="depth")
    cfg = store.namespace_config(object.namespace)
    return _expand_rw(
        store, cfg.rewrite(relation), object, relation, max_depth, _seen | {key}, _depth
    )


def _expand_rw(
    store: TupleStore,
    rw: Rewrite,
    object: ObjectRef,
    relation: Relation,
    max_depth: int,
    seen: frozenset[tuple[str, str, str]],
    depth: int,
) -> UsersetTree:
    match rw:
        case This():
            subjects = tuple(t.principal for t in store.read(object, relation))
            return UsersetTree(object, relation, "this", subjects=subjects)
        case ComputedUserset(relation=other):
            # No depth charge: same object, and the seen-set bounds it.
            kid = expand(store, object, other, max_depth=max_depth, _seen=seen, _depth=depth)
            return UsersetTree(object, relation, "computed_userset", children=(kid,))
        case TupleToUserset(tupleset=tupleset, computed_relation=computed, both_directions=both):
            kids: list[UsersetTree] = []
            for parent in _containers(store, object, tupleset, both):
                kids.append(
                    expand(
                        store,
                        parent,
                        computed,
                        max_depth=max_depth,
                        _seen=seen,
                        _depth=depth + 1,
                    )
                )
            return UsersetTree(object, relation, "tuple_to_userset", children=tuple(kids))
        case Union(children=children):
            kids2 = [
                _expand_rw(store, kid, object, relation, max_depth, seen, depth)
                for kid in children
            ]
            return UsersetTree(object, relation, "union", children=tuple(kids2))
    raise TypeError(f"unknown rewrite: {rw!r}")  # pragma: no cover - defensive


def _containers(
    store: TupleStore, object: ObjectRef, tupleset: Relation, both: bool
) -> list[ObjectRef]:
    out = [ObjectRef(t.principal.namespace, t.principal.id) for t in store.read(object, tupleset)]
    if both:
        out.extend(
            t.object
            for t in store.read_by_principal(PrincipalRef(object.namespace, object.id), tupleset)
        )
    return out


def expand_leaves(
    store: TupleStore,
    object: ObjectRef,
    relation: Relation,
    *,
    max_depth: int = MAX_OBJECT_DEPTH,
) -> frozenset[PrincipalRef]:
    """The granting subjects of ``object#relation``. This is what gets stamped.

    Ingest turns each of these into one grant token. Note what is not here: no
    user expansion, no document id lists, and no "everyone" special case.
    """
    return expand(store, object, relation, max_depth=max_depth).leaves()


@dataclass(frozen=True, slots=True)
class Explanation:
    """``/v1/explain`` in one object: the decision plus the tree it came from."""

    allowed: bool
    why: tuple[str, ...]
    checked_tuples: int
    depth_reached: int
    tree: UsersetTree = field(
        compare=False, repr=False, default_factory=lambda: UsersetTree(ObjectRef("", ""), "")
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "why": list(self.why),
            "checked_tuples": self.checked_tuples,
            "depth_reached": self.depth_reached,
            "tree": self.tree.to_dict(),
        }


def explain(
    store: TupleStore,
    object: ObjectRef,
    relation: Relation,
    principal: PrincipalRef,
    *,
    max_depth: int = MAX_USERSET_DEPTH,
) -> Explanation:
    """Full derivation for an allow *or* a deny.

    A denial that says only "denied" is useless to the admin who has to fix it,
    so the failed branches are included. This is admin-only at the HTTP layer:
    the tree names groups, and naming groups to someone who cannot see the
    object is its own small disclosure.
    """
    ev = _Evaluator(store, principal, max_depth)
    allowed, path, _ = ev.holds(object, relation, 0)
    return Explanation(
        allowed=allowed,
        why=path if allowed else ev.notes,
        checked_tuples=ev.checked,
        depth_reached=ev.deepest,
        tree=expand(store, object, relation),
    )
