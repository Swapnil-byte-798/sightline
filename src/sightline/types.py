"""Core domain types.

Everything in Sightline is defined in terms of three nouns:

* a **principal** — someone asking a question (a person, a service account),
* an **object**   — a thing that can be seen (a document, a folder, a group),
* a **relation**  — the named edge between them (``viewer``, ``member``, ``parent``).

That triple is a *tuple*, and it is the only representation of permission in the
system. There is no ``tenant_id`` column and no ``is_public`` boolean: both are
shorthands that stop being expressible the moment a real organisation has nested
groups, and both are how permission bugs get shipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NewType

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Decision",
    "FilterPlan",
    "ObjectRef",
    "PlanStrategy",
    "PrincipalRef",
    "RefusalReason",
    "Relation",
    "Tuple_",
]

# A grant token is the unit the vector index actually filters on. Compiling a
# principal's permissions down to a *set of these* is what makes authorisation
# expressible as a payload filter rather than a 40,000-element id list.
GrantToken = NewType("GrantToken", str)


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """``namespace:id`` — e.g. ``doc:1042``, ``group:legal``, ``folder:hr``."""

    namespace: str
    id: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.namespace}:{self.id}"

    @classmethod
    def parse(cls, raw: str) -> ObjectRef:
        ns, _, ident = raw.partition(":")
        if not ns or not ident:
            raise ValueError(f"malformed object ref: {raw!r} (want 'namespace:id')")
        return cls(ns, ident)


@dataclass(frozen=True, slots=True)
class PrincipalRef:
    """Who is asking. A principal may itself be a userset (``group:legal#member``)."""

    namespace: str
    id: str
    relation: str | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        base = f"{self.namespace}:{self.id}"
        return f"{base}#{self.relation}" if self.relation else base

    @classmethod
    def parse(cls, raw: str) -> PrincipalRef:
        subject, _, rel = raw.partition("#")
        ns, _, ident = subject.partition(":")
        if not ns or not ident:
            raise ValueError(f"malformed principal ref: {raw!r}")
        return cls(ns, ident, rel or None)


Relation = str


@dataclass(frozen=True, slots=True)
class Tuple_:
    """``object#relation@principal`` — 'doc:42#viewer@group:legal#member'.

    Named with a trailing underscore because ``tuple`` is a builtin and shadowing
    it inside this package would be a small cruelty to every future reader.
    """

    object: ObjectRef
    relation: Relation
    principal: PrincipalRef

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.object}#{self.relation}@{self.principal}"

    @classmethod
    def parse(cls, raw: str) -> Tuple_:
        obj_part, _, subj_part = raw.partition("@")
        obj_raw, _, rel = obj_part.partition("#")
        if not rel or not subj_part:
            raise ValueError(f"malformed tuple: {raw!r} (want 'ns:id#relation@ns:id')")
        return cls(ObjectRef.parse(obj_raw), rel, PrincipalRef.parse(subj_part))


class PlanStrategy(str, Enum):
    """How the compiled plan will be executed against the index.

    The strategy is *chosen from measured cardinality*, not hardcoded. Which one
    fires for a given principal is recorded on every response, because "why was
    this query slow" is unanswerable otherwise.
    """

    #: Few enough allowed documents to list their ids outright.
    ENUMERATE = "enumerate"
    #: Filter on compiled grant tokens; the index prunes during graph traversal.
    GRANT_TOKENS = "grant_tokens"
    #: Principal can see (nearly) everything; skip filtering entirely.
    UNFILTERED = "unfiltered"
    #: Allowed set is small enough that exact scan beats approximate search.
    EXACT_SCAN = "exact_scan"


@dataclass(frozen=True, slots=True)
class FilterPlan:
    """A principal's permissions, compiled into something an index can execute.

    This is the centre of the system. ``Check()`` is authoritative but too slow to
    run per candidate document; this is its precomputed, cacheable projection.
    The two must agree, and ``authz.oracle`` exists to prove they do: a plan that
    admits a document ``Check()`` would deny is a **breach**, not a bug.
    """

    principal: PrincipalRef
    strategy: PlanStrategy
    grant_tokens: frozenset[GrantToken] = frozenset()
    explicit_ids: frozenset[str] = frozenset()
    #: Monotonic counter for the policy the plan was compiled from. A plan built
    #: under an older epoch than the live policy is stale and must not be served.
    epoch: int = 0
    #: Estimated number of documents this plan admits, used to pick the strategy.
    estimated_cardinality: int = 0

    def is_stale(self, current_epoch: int) -> bool:
        return self.epoch < current_epoch


@dataclass(frozen=True, slots=True)
class Decision:
    """The result of an authorisation check, with its derivation.

    ``why`` is not decoration. An access decision a human cannot audit is a
    liability, and the admin console renders this path directly.
    """

    allowed: bool
    why: tuple[str, ...] = ()
    checked_tuples: int = 0


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text plus the object it belongs to."""

    id: str
    object: ObjectRef
    text: str
    #: Grant tokens stamped at ingest. The index filters on exactly these.
    grant_tokens: frozenset[GrantToken] = frozenset()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    object: ObjectRef
    score: float
    #: Which grant token admitted this chunk. Populated so the UI can answer
    #: "why am I allowed to see this?" without a second authorisation pass.
    why_allowed: str | None = None


class RefusalReason(str, Enum):
    NO_PERMITTED_EVIDENCE = "no_permitted_evidence"
    NO_EVIDENCE_AT_ALL = "no_evidence_at_all"
    UNGROUNDED = "ungrounded"
    INJECTION_DETECTED = "injection_detected"
    BUDGET_EXCEEDED = "budget_exceeded"
    STALE_POLICY = "stale_policy"


@dataclass(frozen=True, slots=True)
class Answer:
    """Grounded or refused. There is no third state.

    Note what is *absent*: a free-text citation field. The synthesiser returns
    chunk ids only, and citations are reconstructed from the retrieved set, so a
    citation to a document the principal cannot see is unrepresentable rather
    than merely discouraged.
    """

    text: str
    citations: tuple[Citation, ...] = ()
    refused: bool = False
    refusal_reason: RefusalReason | None = None
    #: Set when the system declines to say whether evidence merely exists.
    #: Distinguishing "no such document" from "not for you" leaks the corpus.
    existence_protected: bool = False
    strategy: PlanStrategy | None = None
    epoch: int = 0
