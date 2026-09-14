"""The spine: compile plan, search, recheck, guard, generate, reconstruct.

Everything else in this package is a component. This is the order they run in,
and the order is the product:

    epoch -> plan -> [stale? empty?] -> embed -> search (UncheckedHit)
          -> recheck (Hit) -> guardrails -> generate -> citations -> audit

Two structural facts do the security work, and neither of them is a check that
somebody could forget to call.

**You cannot reach generation without passing through recheck.** The synthesiser
entry point, :meth:`ProviderChain.answer`, accepts ``Sequence[Hit]``, and ``Hit``
is produced in exactly one place in the codebase — ``authz.recheck.recheck()``.
This module holds no other path to it: there is no branch that builds a Hit, no
flag that skips the recheck call, and :func:`_assert_rechecked` makes the
type-level rule a runtime one as well. Skipping recheck here is planted mutant
M3, and it is meant to be difficult to write by accident.

**Every response reports how it was authorised.** The plan strategy that fired,
the policy epoch it ran under, and the number of candidates the live check threw
away go on the answer, in the trace, in the metrics and in the audit row. The
last of those, ``dropped_at_recheck``, is the stale-index gap made visible per
request — and it is the number whose *disappearance* across all traffic means
recheck has quietly become a no-op.

On refusals: ``NO_PERMITTED_EVIDENCE`` (the plan admits nothing, or everything
found was dropped) and ``NO_EVIDENCE_AT_ALL`` (the corpus really has nothing)
produce byte-identical output. Distinguishing them enumerates the corpus one
question at a time, and document titles and project code names are often the
sensitive part. The distinction survives only in the audit log. The timing
channel between the two is narrowed by a constant-time floor and is **not
closed**; see :class:`~sightline.settings.RetrievalSettings`.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from sightline import obs
from sightline.audit import AuditRecord, AuditSink, MemoryAuditLog, hash_query
from sightline.authz.compile import DEFAULT_RELATION, PlanCompiler
from sightline.authz.recheck import LiveRechecker
from sightline.authz.tuples import TupleStore
from sightline.errors import (
    AuthorityUnavailable,
    BudgetExceeded,
    IndexUnavailable,
    StalePolicy,
)
from sightline.generate import Claim, Generation, ProviderChain
from sightline.settings import Settings
from sightline.store.base import Hit, UncheckedHit, VectorStore
from sightline.types import (
    Answer,
    Citation,
    FilterPlan,
    ObjectRef,
    PlanStrategy,
    PrincipalRef,
    RefusalReason,
)

__all__ = [
    "INJECTION_PATTERNS",
    "BuiltinGuardrails",
    "Diagnostics",
    "Guardrails",
    "PipelineEvent",
    "QueryResult",
    "Retriever",
    "SearchResult",
    "default_guardrails",
]


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------

#: Instruction-shaped content in retrieved text. This is a pattern matcher, it is
#: described everywhere as a pattern matcher, and no detection rate is claimed
#: for it, because claiming one would be dishonest: translate the instruction
#: into Polish or spell it with homoglyphs and it walks straight past.
#:
#: The reason that is survivable: the model has no tools, no network, no second
#: retrieval pass, and the permission filter was compiled before the query was
#: read. An injection that succeeds completely still cannot widen the candidate
#: set. This scan is a speed bump in front of a wall, not the wall.
INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_previous", re.compile(r"(?i)ignore\s+(all\s+)?(your\s+|the\s+)?previous\s+"
                                   r"(instructions?|rules?|prompts?)")),
    ("disregard", re.compile(r"(?i)disregard\s+(all\s+)?(your\s+|the\s+)?(above|prior|previous)")),
    ("ai_address", re.compile(r"(?i)(note|message|instruction)s?\s+(for|to)\s+the\s+"
                              r"(ai|assistant|language model|llm|bot)")),
    ("role_override", re.compile(r"(?i)you\s+are\s+now\s+(a|an|the)\s+")),
    ("system_forge", re.compile(r"(?i)<\s*/?\s*(system|assistant)\s*>|\[/?INST\]|"
                                r"###\s*(system|instruction)")),
    ("permission_claim", re.compile(r"(?i)(has been|have been|is|are)\s+granted\s+"
                                    r"(temporary\s+)?(access|permission|clearance)")),
    ("retrieval_override", re.compile(r"(?i)(ignore|bypass|override)\s+(your\s+)?"
                                      r"(retrieval\s+)?(restrictions?|filters?|permissions?)")),
    ("exfiltrate", re.compile(r"(?i)(send|post|upload|email)\s+(this|the\s+\w+)\s+to\s+"
                              r"(https?://|\S+@)")),
    ("secrecy", re.compile(r"(?i)do\s+not\s+(mention|reveal|tell|disclose)\s+"
                           r"(this|these|the)\s+(instruction|note|message)")),
    ("marker_forge", re.compile(r"<<<\s*END_SIGHTLINE_DOCUMENT\s*>>>")),
    ("invisible_text", re.compile(r"[​‌‍⁠﻿]{4,}")),
)


@runtime_checkable
class Guardrails(Protocol):
    """The two checks the pipeline runs on the rechecked set.

    Deliberately narrow. Everything else on the FRD's guardrail list is either
    structural (the ``Hit`` type, citation reconstruction) or lives in the
    authorisation layer (depth limit, cycle guard), and a guardrail implemented
    as a type is a guardrail that cannot be mocked out.
    """

    def scan_injection(self, texts: Sequence[str]) -> tuple[bool, str]:
        """Return ``(hit, pattern_name)``. Never returns the matching text."""
        ...

    def ground(self, claims: Sequence[Claim], known_ids: frozenset[str]) -> tuple[Claim, ...]:
        """Drop claims whose evidence is not in the rechecked set."""
        ...


class BuiltinGuardrails:
    """Pattern scan and structural grounding, with no model call anywhere.

    No model call is a design constraint, not a cost saving: a scanner built from
    a language model is itself a thing you can prompt-inject, so the detector
    would inherit the exact vulnerability it exists to catch.
    """

    def scan_injection(self, texts: Sequence[str]) -> tuple[bool, str]:
        for text in texts:
            for name, pattern in INJECTION_PATTERNS:
                if pattern.search(text):
                    # The name of the pattern, never the span it matched. The
                    # matched span is document content, and document content in a
                    # refusal payload is planted mutant M12.
                    return True, name
        return False, ""

    def ground(self, claims: Sequence[Claim], known_ids: frozenset[str]) -> tuple[Claim, ...]:
        """Structural grounding: a claim survives only with surviving evidence.

        This is not a semantic entailment check and does not pretend to be one.
        It enforces the property that actually matters — every sentence shown to
        the reader is attributed to a chunk that passed the live permission check
        on this request — and leaves "does the sentence follow from the chunk" to
        the model that wrote it. A claim whose ids were all dropped is removed
        entirely rather than shown uncited, because an uncited sentence is
        indistinguishable from one the model invented.
        """
        out: list[Claim] = []
        for claim in claims:
            kept = tuple(cid for cid in claim.chunk_ids if cid in known_ids)
            if kept:
                out.append(Claim(claim.text, kept))
        return tuple(out)


def default_guardrails() -> Guardrails:
    """Use :mod:`sightline.guardrails` when it exposes the hooks, else the builtin.

    The guardrails package owns the real scanner; this module owns the order
    things run in. The adapter exists so the spine is never unguarded during
    development of either half — the fallback is a working implementation, not a
    no-op, which is the difference between a seam and a hole.
    """
    try:
        import sightline.guardrails as pkg
    except ImportError:  # pragma: no cover - the package is in-tree
        return BuiltinGuardrails()
    scan = getattr(pkg, "scan_injection", None)
    ground = getattr(pkg, "ground", None)
    if callable(scan) and callable(ground):
        return _AdaptedGuardrails(scan, ground)
    return BuiltinGuardrails()


class _AdaptedGuardrails:
    """Wraps two module-level functions into the :class:`Guardrails` protocol."""

    def __init__(self, scan: Any, ground: Any) -> None:
        self._scan = scan
        self._ground = ground

    def scan_injection(self, texts: Sequence[str]) -> tuple[bool, str]:
        return self._scan(texts)

    def ground(self, claims: Sequence[Claim], known_ids: frozenset[str]) -> tuple[Claim, ...]:
        return self._ground(claims, known_ids)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """Per-request timing and decision counts, returned inline on every response.

    Inline as well as in traces, because a caller with no tracing backend still
    needs to know why their query was slow, and because ``dropped_at_recheck``
    reaching them directly means nobody has to take our word for it.
    """

    candidates: int = 0
    dropped_at_recheck: int = 0
    plan_ms: float = 0.0
    search_ms: float = 0.0
    recheck_ms: float = 0.0
    guard_ms: float = 0.0
    synth_ms: float = 0.0
    total_ms: float = 0.0
    plan_cache: str = "miss"
    provider: str = "none"
    degraded: bool = False
    guardrails_fired: tuple[str, ...] = ()
    #: The epoch moved between compiling the plan and re-checking. Not an error:
    #: recheck reads live tuples, so the served results are correct for the newer
    #: policy. Recorded because it explains an otherwise surprising drop count.
    policy_moved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "dropped_at_recheck": self.dropped_at_recheck,
            "plan_ms": round(self.plan_ms, 3),
            "search_ms": round(self.search_ms, 3),
            "recheck_ms": round(self.recheck_ms, 3),
            "guard_ms": round(self.guard_ms, 3),
            "synth_ms": round(self.synth_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "plan_cache": self.plan_cache,
            "provider": self.provider,
            "degraded": self.degraded,
            "guardrails_fired": list(self.guardrails_fired),
            "policy_moved": self.policy_moved,
        }


@dataclass(frozen=True, slots=True)
class QueryResult:
    """An :class:`~sightline.types.Answer` plus everything an operator wants."""

    answer: Answer
    diagnostics: Diagnostics
    request_id: str = ""
    audit_seq: int | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    """``/v1/search``: rechecked hits, no generation. Same guarantees."""

    hits: tuple[Hit, ...]
    strategy: PlanStrategy
    epoch: int
    diagnostics: Diagnostics
    refused: bool = False
    refusal_reason: RefusalReason | None = None
    existence_protected: bool = False
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """One server-sent event from ``/v1/ask``.

    The stream carries *stages*, not model tokens. Streaming tokens would put
    text in front of the reader before grounding and citation reconstruction had
    run, and a refusal you have already displayed half of is not a refusal.
    """

    stage: str
    data: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


def _assert_rechecked(hits: Sequence[Any]) -> None:
    """Runtime half of the ``Hit``-only rule (M3).

    The type annotation is the documentation; this is the enforcement. It costs
    an isinstance per hit, which is nothing next to a permission check, and it
    turns "somebody passed the raw index output straight through" from a subtle
    leak into an immediate, loud failure.
    """
    for hit in hits:
        if not isinstance(hit, Hit):
            raise TypeError(
                f"expected Hit (the output of recheck()), got {type(hit).__name__}: "
                "the index is a hint, the database is the authority"
            )


def _admits_nothing(plan: FilterPlan) -> bool:
    """Whether this plan can possibly match a chunk.

    Empty means nobody (ADR 0003). Treating an empty token set as "no filter, so
    match everything" is planted mutant M6, and it is an easy accident because an
    empty set is falsy. Written as an explicit predicate so the dangerous reading
    has no place to hide.
    """
    if plan.strategy is PlanStrategy.UNFILTERED:
        return False
    return not plan.grant_tokens and not plan.explicit_ids


class Retriever:
    """Holds the components and runs them in the one supported order.

    Construct with everything injected; :func:`sightline.api.build_retriever`
    does the wiring from settings. Nothing here reaches for a global, so a test
    can drive the whole pipeline with an in-memory store and a fake provider.
    """

    def __init__(
        self,
        *,
        tuple_store: TupleStore,
        vector_store: VectorStore,
        embedder: Any,
        compiler: PlanCompiler | None = None,
        rechecker: LiveRechecker | None = None,
        generator: ProviderChain | None = None,
        guardrails: Guardrails | None = None,
        audit: AuditSink | None = None,
        settings: Settings | None = None,
        relation: str = DEFAULT_RELATION,
    ) -> None:
        self.tuple_store = tuple_store
        self.vector_store = vector_store
        self.embedder = embedder
        self.settings = settings or Settings()
        self.relation = relation
        self.compiler = compiler or PlanCompiler(tuple_store, relation=relation)
        self.rechecker = rechecker or LiveRechecker(tuple_store, relation=relation)
        self.generator = generator or ProviderChain([], settings=self.settings.generation)
        self.guardrails = guardrails or default_guardrails()
        self.audit = audit or MemoryAuditLog()

    # -- policy ------------------------------------------------------------
    def epoch(self) -> int:
        """Read the live policy epoch, or refuse.

        Unreadable epoch is unreadable policy, and the FRD is explicit that this
        refuses rather than degrades. There is no cached value to fall back to,
        deliberately: a cached epoch makes a stale plan look fresh, which is the
        one thing the epoch exists to prevent.
        """
        try:
            return self.tuple_store.epoch()
        except Exception as exc:
            raise AuthorityUnavailable(f"policy epoch unreadable: {exc}") from exc

    def plan_for(self, principal: PrincipalRef) -> tuple[FilterPlan, str, int]:
        """Compile (or reuse) a plan. Returns ``(plan, cache_result, live_epoch)``.

        The staleness check runs *after* compilation against a freshly read
        epoch, so a write that landed during the compile is caught. One recompile
        is attempted; a second staleness means the policy is moving faster than
        we can compile, and the honest answer to that is a 409 telling the caller
        to retry rather than a loop that never converges.
        """
        live = self.epoch()
        # The cache's own hit counter, read as a delta. Counting entries instead
        # would call an eviction a hit. Under concurrency another request can
        # move the counter between these two reads, so the label is a sample and
        # not an accounting; it is a metric label, not a security decision.
        before = self.compiler.cache.hits
        plan = self.compiler.compile(principal, epoch=live)
        cache = "hit" if self.compiler.cache.hits > before else "miss"

        live_now = self.epoch()
        if plan.is_stale(live_now):
            if not self.settings.retrieval.recompile_on_stale:
                raise StalePolicy(plan_epoch=plan.epoch, live_epoch=live_now)
            plan = self.compiler.compile(principal, epoch=live_now)
            cache = "miss"
            live_now = self.epoch()
            if plan.is_stale(live_now):
                raise StalePolicy(plan_epoch=plan.epoch, live_epoch=live_now)
        return plan, cache, live_now

    # -- search ------------------------------------------------------------
    def search(
        self,
        question: str,
        principal: PrincipalRef,
        *,
        k: int | None = None,
        request_id: str = "",
    ) -> SearchResult:
        """Plan, search, recheck. No generation, same guarantees.

        Every hit returned has passed the live permission check. The count that
        did not is in ``diagnostics.dropped_at_recheck``.
        """
        rid = request_id or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        k = self.validate_request(question, k, self.settings.retrieval.default_max_context_chars)

        with obs.span("v1.search", **{"request.id": rid}) as sp:
            plan, cache, live, plan_ms = self._plan_stage(principal, sp)
            if _admits_nothing(plan):
                self._floor(started)
                diag = self._diag(
                    started, plan_ms=plan_ms, plan_cache=cache,
                    fired=("empty_permitted_set",),
                )
                self._audit_row(
                    rid, principal, question, plan, diag,
                    internal_reason=RefusalReason.NO_PERMITTED_EVIDENCE, route="/v1/search",
                )
                return SearchResult(
                    (), plan.strategy, plan.epoch, diag,
                    refused=True,
                    refusal_reason=RefusalReason.NO_PERMITTED_EVIDENCE,
                    existence_protected=True,
                    request_id=rid,
                )

            unchecked, search_ms = self._search_stage(question, plan, k, sp)
            hits, dropped, recheck_ms, moved = self._recheck_stage(principal, unchecked, plan, sp)

            diag = self._diag(
                started,
                plan_ms=plan_ms,
                candidates=len(unchecked),
                dropped=dropped,
                search_ms=search_ms,
                recheck_ms=recheck_ms,
                plan_cache=cache,
                policy_moved=moved or plan.epoch != live,
            )
            refused = not hits
            if refused:
                self._floor(started)
            self._audit_row(
                rid, principal, question, plan, diag,
                retrieved=tuple(h.object_ref for h in unchecked),
                shown=tuple(h.object_ref for h in hits),
                internal_reason=(
                    self._empty_reason(unchecked) if refused else None
                ),
                route="/v1/search",
            )
            obs.queries_total.inc(
                strategy=plan.strategy.value, outcome="refused" if refused else "hits"
            )
            return SearchResult(
                tuple(hits), plan.strategy, plan.epoch, diag,
                refused=refused,
                refusal_reason=RefusalReason.NO_PERMITTED_EVIDENCE if refused else None,
                existence_protected=refused,
                request_id=rid,
            )

    # -- ask ---------------------------------------------------------------
    def ask(
        self,
        question: str,
        principal: PrincipalRef,
        *,
        k: int | None = None,
        max_context_chars: int | None = None,
        request_id: str = "",
    ) -> QueryResult:
        """The whole pipeline. Grounded or refused; there is no third state."""
        for event in self.events(
            question, principal, k=k, max_context_chars=max_context_chars,
            request_id=request_id,
        ):
            if event.stage == "result":
                return event.data["result"]
        raise AssertionError("the pipeline always ends in a result event")

    def events(
        self,
        question: str,
        principal: PrincipalRef,
        *,
        k: int | None = None,
        max_context_chars: int | None = None,
        request_id: str = "",
    ) -> Iterator[PipelineEvent]:
        """Run the pipeline, yielding a stage event as each one completes.

        ``/v1/ask`` renders these as server-sent events. The final event carries
        the finished :class:`QueryResult`; every earlier event carries counts and
        stage names only — never chunk text, and never a partial answer.
        """
        rid = request_id or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        max_chars = max_context_chars or self.settings.retrieval.default_max_context_chars
        k = self.validate_request(question, k, max_chars)

        with obs.span("v1.ask", **{"request.id": rid}) as sp:
            yield PipelineEvent("accepted", {"request_id": rid, "k": k})

            plan, cache, live, plan_ms = self._plan_stage(principal, sp)
            yield PipelineEvent(
                "plan",
                {
                    "strategy": plan.strategy.value,
                    "epoch": plan.epoch,
                    "cache": cache,
                    "n_grant_tokens": len(plan.grant_tokens),
                    "n_explicit_ids": len(plan.explicit_ids),
                    "estimated_cardinality": plan.estimated_cardinality,
                },
            )

            if _admits_nothing(plan):
                self._floor(started)
                diag = self._diag(
                    started, plan_ms=plan_ms, plan_cache=cache,
                    fired=("empty_permitted_set",),
                )
                answer = self._empty_answer(plan)
                seq = self._audit_row(
                    rid, principal, question, plan, diag,
                    internal_reason=RefusalReason.NO_PERMITTED_EVIDENCE,
                )
                obs.refusals_total.inc(reason=RefusalReason.NO_PERMITTED_EVIDENCE.value)
                obs.queries_total.inc(strategy=plan.strategy.value, outcome="refused")
                yield PipelineEvent("result", {"result": QueryResult(answer, diag, rid, seq)})
                return

            unchecked, search_ms = self._search_stage(question, plan, k, sp)
            yield PipelineEvent("search", {"candidates": len(unchecked)})

            hits, dropped, recheck_ms, moved = self._recheck_stage(principal, unchecked, plan, sp)
            yield PipelineEvent(
                "recheck",
                {"presented": len(unchecked), "kept": len(hits), "dropped": dropped},
            )

            if not hits:
                self._floor(started)
                diag = self._diag(
                    started, plan_ms=plan_ms, candidates=len(unchecked), dropped=dropped,
                    search_ms=search_ms, recheck_ms=recheck_ms, plan_cache=cache,
                    policy_moved=moved,
                )
                answer = self._empty_answer(plan)
                seq = self._audit_row(
                    rid, principal, question, plan, diag,
                    retrieved=tuple(h.object_ref for h in unchecked),
                    internal_reason=self._empty_reason(unchecked),
                )
                obs.refusals_total.inc(reason=RefusalReason.NO_PERMITTED_EVIDENCE.value)
                obs.queries_total.inc(strategy=plan.strategy.value, outcome="refused")
                yield PipelineEvent("result", {"result": QueryResult(answer, diag, rid, seq)})
                return

            # Guardrails run on the rechecked set only. Scanning before recheck
            # would mean a document the caller may not see could refuse their
            # query, which is both a denial-of-service and a side channel that
            # says "something you cannot read exists and it is odd".
            guard_started = time.perf_counter()
            injected, pattern = self.guardrails.scan_injection([h.text for h in hits])
            guard_ms = (time.perf_counter() - guard_started) * 1000.0
            sp.set("guardrail.injection_hit", injected)
            yield PipelineEvent("guardrails", {"injection_hit": injected})

            if injected:
                diag = self._diag(
                    started, plan_ms=plan_ms, candidates=len(unchecked),
                    dropped=dropped, search_ms=search_ms,
                    recheck_ms=recheck_ms, guard_ms=guard_ms, plan_cache=cache,
                    fired=(f"injection:{pattern}",), policy_moved=moved,
                )
                answer = Answer(
                    text=_REFUSAL_INJECTION,
                    refused=True,
                    refusal_reason=RefusalReason.INJECTION_DETECTED,
                    strategy=plan.strategy,
                    epoch=plan.epoch,
                )
                seq = self._audit_row(
                    rid, principal, question, plan, diag,
                    retrieved=tuple(h.object_ref for h in unchecked),
                    internal_reason=RefusalReason.INJECTION_DETECTED,
                )
                obs.refusals_total.inc(reason=RefusalReason.INJECTION_DETECTED.value)
                obs.queries_total.inc(strategy=plan.strategy.value, outcome="refused")
                yield PipelineEvent("result", {"result": QueryResult(answer, diag, rid, seq)})
                return

            synth_started = time.perf_counter()
            _assert_rechecked(hits)
            generation = self.generator.answer(
                question, hits, max_context_chars=max_chars,
            )
            synth_ms = (time.perf_counter() - synth_started) * 1000.0

            by_id = {h.chunk_id: h for h in hits}
            claims = self.guardrails.ground(generation.claims, frozenset(by_id))
            citations = _reconstruct_citations(claims, by_id)
            ids_dropped = _count_dropped_ids(generation.claims, frozenset(by_id))
            obs.record_generation(
                sp,
                provider=generation.provider,
                chunks_in=len(hits),
                claims_out=len(generation.claims),
                citations=len(citations),
                ids_dropped=ids_dropped,
                degraded=generation.degraded,
            )
            yield PipelineEvent(
                "synth",
                {
                    "provider": generation.provider,
                    "claims": len(claims),
                    "citations": len(citations),
                    "degraded": generation.degraded,
                },
            )

            fired: list[str] = []
            if ids_dropped:
                fired.append("citation_reconstruction")
            if generation.degraded:
                fired.append("degraded_provider")

            if not claims or not citations:
                diag = self._diag(
                    started, plan_ms=plan_ms, candidates=len(unchecked),
                    dropped=dropped, search_ms=search_ms,
                    recheck_ms=recheck_ms, guard_ms=guard_ms, synth_ms=synth_ms,
                    plan_cache=cache, provider=generation.provider,
                    degraded=generation.degraded, fired=tuple(fired + ["ungrounded"]),
                    policy_moved=moved,
                )
                answer = Answer(
                    text=_REFUSAL_UNGROUNDED,
                    refused=True,
                    refusal_reason=RefusalReason.UNGROUNDED,
                    strategy=plan.strategy,
                    epoch=plan.epoch,
                )
                seq = self._audit_row(
                    rid, principal, question, plan, diag,
                    retrieved=tuple(h.object_ref for h in unchecked),
                    internal_reason=RefusalReason.UNGROUNDED,
                    provider=generation.provider, degraded=generation.degraded,
                )
                obs.refusals_total.inc(reason=RefusalReason.UNGROUNDED.value)
                obs.queries_total.inc(strategy=plan.strategy.value, outcome="refused")
                yield PipelineEvent("result", {"result": QueryResult(answer, diag, rid, seq)})
                return

            answer = Answer(
                text=" ".join(c.text for c in claims),
                citations=citations,
                refused=False,
                strategy=plan.strategy,
                epoch=plan.epoch,
            )
            diag = self._diag(
                started, plan_ms=plan_ms, candidates=len(unchecked),
                dropped=dropped, search_ms=search_ms,
                recheck_ms=recheck_ms, guard_ms=guard_ms, synth_ms=synth_ms, plan_cache=cache,
                provider=generation.provider, degraded=generation.degraded,
                fired=tuple(fired), policy_moved=moved,
            )
            shown = tuple(dict.fromkeys(str(c.object) for c in citations))
            seq = self._audit_row(
                rid, principal, question, plan, diag,
                retrieved=tuple(h.object_ref for h in unchecked),
                shown=shown,
                provider=generation.provider,
                degraded=generation.degraded,
            )
            obs.queries_total.inc(strategy=plan.strategy.value, outcome="grounded")
            yield PipelineEvent("result", {"result": QueryResult(answer, diag, rid, seq)})

    # -- stages ------------------------------------------------------------
    def _plan_stage(self, principal: PrincipalRef, sp: obs.SpanHandle
                    ) -> tuple[FilterPlan, str, int, float]:
        """Returns ``(plan, cache_result, live_epoch, elapsed_ms)``.

        The elapsed time is returned rather than stashed on ``self``. One
        :class:`Retriever` serves every concurrent request, so per-request state
        on the instance is a data race that shows up as another user's timings in
        your diagnostics — and, if anyone later put a plan there, as something
        much worse.
        """
        started = time.perf_counter()
        with obs.span("authz.compile_plan") as plan_span:
            plan, cache, live = self.plan_for(principal)
            obs.record_plan(plan_span, plan, cache=cache)
        elapsed = time.perf_counter() - started
        obs.stage_seconds.observe(elapsed, stage="plan")
        # Attributes on the request span too, but not through record_plan: that
        # increments the cache counter, and one compile is one cache event.
        sp.set_many(
            {
                "authz.strategy": plan.strategy.value,
                "authz.cache": cache,
                "authz.n_grant_tokens": len(plan.grant_tokens),
                "policy.epoch": plan.epoch,
            }
        )
        return plan, cache, live, elapsed * 1000.0

    def _search_stage(self, question: str, plan: FilterPlan, k: int, sp: obs.SpanHandle
                      ) -> tuple[list[UncheckedHit], float]:
        started = time.perf_counter()
        try:
            vector = self._embed(question)
            with obs.span("store.search") as search_span:
                unchecked = list(self.vector_store.search(vector, plan, k))
                obs.record_search(
                    search_span,
                    backend=getattr(self.vector_store, "name", "unknown"),
                    k=k,
                    returned=len(unchecked),
                )
        except Exception as exc:
            raise IndexUnavailable(f"search failed: {type(exc).__name__}") from exc
        elapsed = time.perf_counter() - started
        obs.stage_seconds.observe(elapsed, stage="search")
        obs.record_search(
            sp, backend=getattr(self.vector_store, "name", "unknown"), k=k,
            returned=len(unchecked),
        )
        return unchecked, elapsed * 1000.0

    def _recheck_stage(
        self, principal: PrincipalRef, unchecked: Sequence[UncheckedHit], plan: FilterPlan,
        sp: obs.SpanHandle,
    ) -> tuple[list[Hit], int, float, bool]:
        started = time.perf_counter()
        with obs.span("authz.recheck"):
            hits, dropped = self.rechecker.recheck(str(principal), unchecked)
        _assert_rechecked(hits)
        elapsed = time.perf_counter() - started
        obs.stage_seconds.observe(elapsed, stage="recheck")
        epoch_now = hits[0].checked_at_epoch if hits else self.epoch()
        obs.record_recheck(
            sp, strategy=plan.strategy.value, presented=len(unchecked), dropped=dropped,
            epoch=epoch_now,
        )
        return hits, dropped, elapsed * 1000.0, epoch_now != plan.epoch

    def _embed(self, question: str) -> Sequence[float]:
        matrix = self.embedder.encode([question])
        return np.asarray(matrix, dtype=np.float32)[0]

    # -- helpers -----------------------------------------------------------
    def validate_request(self, question: str, k: int | None, max_chars: int) -> int:
        """Caps, checked before any work. Over budget refuses; it never trims.

        Trimming to fit would mean serving a request with fewer permission checks
        than it needed, which is the failure mode this whole system is about
        (SR-7).

        Public because the streaming endpoint has to run it *before* the
        response starts: once the first byte of a 200 is on the wire, a 429 is
        no longer expressible, and an error the client can only discover by
        parsing a truncated stream is not an error anybody handles.
        """
        cfg = self.settings.retrieval
        if not question or not question.strip():
            raise BudgetExceeded("empty question")
        if len(question) > cfg.max_question_chars:
            raise BudgetExceeded(
                f"question is {len(question)} chars, cap is {cfg.max_question_chars}"
            )
        resolved = cfg.default_k if k is None else k
        if resolved < 1 or resolved > cfg.max_k:
            raise BudgetExceeded(f"k must be between 1 and {cfg.max_k}, got {resolved}")
        if max_chars > cfg.max_context_chars:
            raise BudgetExceeded(
                f"max_context_chars {max_chars} exceeds the cap {cfg.max_context_chars}"
            )
        return resolved

    def _floor(self, started: float) -> None:
        """Hold a refusal open to a constant floor.

        The empty-plan path returns in about a millisecond; the searched-and-
        found-nothing path takes tens. Identical bodies with a 60x timing
        difference is still an oracle for "a document exists that you may not
        see". This narrows it. It does not close it — a patient attacker with
        thousands of timed probes still wins, and the honest fix for that threat
        model is a decoy search, which costs real work on the cheapest path in
        the system. The trade is documented rather than hidden.
        """
        floor = self.settings.retrieval.refusal_floor_ms / 1000.0
        remaining = floor - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)

    def _empty_answer(self, plan: FilterPlan) -> Answer:
        """The one refusal both empty-handed paths return, byte for byte.

        One constructor, called from both sites, so the two cannot drift apart in
        a later edit. Divergence here is planted mutant M11.
        """
        return Answer(
            text=_REFUSAL_EMPTY,
            citations=(),
            refused=True,
            refusal_reason=RefusalReason.NO_PERMITTED_EVIDENCE,
            existence_protected=True,
            strategy=plan.strategy,
            epoch=plan.epoch,
        )

    @staticmethod
    def _empty_reason(unchecked: Sequence[UncheckedHit]) -> RefusalReason:
        """The true reason, for the audit log only.

        The index found candidates and the live check removed all of them
        (``NO_PERMITTED_EVIDENCE``), or the index found nothing at all
        (``NO_EVIDENCE_AT_ALL``). The user sees the same words either way.
        """
        return (
            RefusalReason.NO_PERMITTED_EVIDENCE if unchecked
            else RefusalReason.NO_EVIDENCE_AT_ALL
        )

    def _diag(
        self,
        started: float,
        *,
        plan_ms: float = 0.0,
        candidates: int = 0,
        dropped: int = 0,
        search_ms: float = 0.0,
        recheck_ms: float = 0.0,
        guard_ms: float = 0.0,
        synth_ms: float = 0.0,
        plan_cache: str = "miss",
        provider: str = "none",
        degraded: bool = False,
        fired: tuple[str, ...] = (),
        policy_moved: bool = False,
    ) -> Diagnostics:
        total = time.perf_counter() - started
        obs.stage_seconds.observe(total, stage="total")
        return Diagnostics(
            candidates=candidates,
            dropped_at_recheck=dropped,
            plan_ms=plan_ms,
            search_ms=search_ms,
            recheck_ms=recheck_ms,
            guard_ms=guard_ms,
            synth_ms=synth_ms,
            total_ms=total * 1000.0,
            plan_cache=plan_cache,
            provider=provider,
            degraded=degraded,
            guardrails_fired=fired,
            policy_moved=policy_moved,
        )

    def _audit_row(
        self,
        request_id: str,
        principal: PrincipalRef,
        question: str,
        plan: FilterPlan,
        diag: Diagnostics,
        *,
        retrieved: tuple[str, ...] = (),
        shown: tuple[str, ...] = (),
        internal_reason: RefusalReason | None = None,
        provider: str | None = None,
        degraded: bool = False,
        route: str = "/v1/ask",
    ) -> int | None:
        """Write one audit row. Never raises into the request path.

        ``audit_required`` decides whether a sink failure fails the request, and
        it lives in settings rather than here because it is a deployment policy:
        a regulated customer wants the request to fail, a demo does not.
        """
        record = AuditRecord(
            principal=str(principal),
            query_hash=hash_query(question, self.settings.audit.hmac_key.reveal()),
            strategy=plan.strategy.value,
            epoch=plan.epoch,
            n_candidates=diag.candidates,
            n_dropped_at_recheck=diag.dropped_at_recheck,
            retrieved_objects=tuple(dict.fromkeys(retrieved)),
            shown_objects=tuple(dict.fromkeys(shown)),
            guardrails_fired=diag.guardrails_fired,
            refusal_reason=internal_reason.value if internal_reason else None,
            provider=provider,
            degraded=degraded,
            request_id=request_id,
            route=route,
        )
        try:
            row = self.audit.append(record)
        except Exception:
            obs.audit_appends_total.inc(outcome="error")
            if self.settings.audit.required:
                raise
            return None
        obs.audit_appends_total.inc(outcome="ok")
        return row.seq


# --------------------------------------------------------------------------
# Refusal text
# --------------------------------------------------------------------------

#: One string, used by both empty-handed refusals. It says nothing about whether
#: anything exists, because saying so is the leak (FR-20).
_REFUSAL_EMPTY = (
    "I could not find anything I can show you that answers this question."
)

_REFUSAL_INJECTION = (
    "I stopped answering this question: retrieved content contained "
    "instruction-shaped text, which this system treats as a reason to refuse "
    "rather than a reason to comply."
)

_REFUSAL_UNGROUNDED = (
    "I could not produce an answer that is supported by the documents I am "
    "allowed to show you."
)


def _reconstruct_citations(claims: Sequence[Claim], by_id: dict[str, Hit]) -> tuple[Citation, ...]:
    """Build citations by looking chunk ids up in the rechecked set.

    Never parsed out of generated text — that is planted mutant M14. The map is
    built from ``Hit`` objects, so a citation to a chunk the principal cannot see
    is not merely discouraged, it has no construction path: the id is not in the
    map, so nothing is emitted.

    ``why_allowed`` comes from the recheck derivation, not from the index's
    advisory ``matched_token`` (FR-21, and trusting the token is M4).
    """
    out: list[Citation] = []
    seen: set[str] = set()
    for claim in claims:
        for chunk_id in claim.chunk_ids:
            hit = by_id.get(chunk_id)
            if hit is None or chunk_id in seen:
                continue
            seen.add(chunk_id)
            try:
                obj = ObjectRef.parse(hit.object_ref)
            except ValueError:
                continue
            out.append(
                Citation(
                    chunk_id=hit.chunk_id,
                    object=obj,
                    score=hit.score,
                    why_allowed=hit.why_allowed,
                )
            )
    return tuple(out)


def _count_dropped_ids(claims: Sequence[Claim], known: frozenset[str]) -> int:
    """How many chunk ids the model named that were not in the rechecked set.

    Non-zero means the synthesiser invented an identifier or cited something that
    was dropped between prompt and response. Worth a metric rather than a log
    line: a sudden rise is a model regression, and a steady non-zero is a bug.
    """
    return sum(1 for c in claims for cid in c.chunk_ids if cid not in known)


def generation_is_grounded(generation: Generation, known: frozenset[str]) -> bool:
    """Whether any claim survives citation reconstruction. Used by tests."""
    return any(any(cid in known for cid in c.chunk_ids) for c in generation.claims)
