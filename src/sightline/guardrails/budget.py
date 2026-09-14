"""Budgets that degrade the answer, never the checks.

Suppose recheck returns ten hits averaging 1,200 characters against an 8,000
character context budget. Three responses were available:

===========================================  ========================================
Response                                     Verdict
===========================================  ========================================
Drop chunks until it fits, answer normally   Wrong. Evidence was silently discarded,
                                             possibly the piece that changes the
                                             answer, and the user cannot tell
Skip the recheck on the overflow to save     Catastrophic. This is planted mutant M10
time
Degrade the *answer*, never the *checks*     Correct
===========================================  ========================================

So this module implements a ladder:

1. **Full synthesis** over the whole rechecked set.
2. **Extractive** — return the top rechecked passages verbatim, each with its
   real citation, and no generated prose. This rung is the one that matters: the
   passages are permitted, the citations are real, and the only thing lost is
   the summary. It also costs zero provider tokens, which is why it is the right
   answer to an exhausted token budget rather than a 429.
3. **Refuse** with ``BUDGET_EXCEEDED`` and HTTP 429.

The invariant across the whole ladder: **no rung is reached by skipping a
permission check.** Every function here takes ``Hit``, which only
:func:`sightline.authz.recheck.recheck` can produce, so the tempting
optimisation is not expressible without an import that looks wrong.

WHY THE EXTRACTIVE RUNG IS NOT THE SAME SIN AS "DROP CHUNKS UNTIL IT FITS"
--------------------------------------------------------------------------
Both show the user less than everything. The difference is that an extractive
answer says how many passages it is showing and that it is showing them raw, so
the user knows the summary is missing and can ask a narrower question. A
synthesised answer built from a silently truncated context looks exactly like a
complete one. The lie is the problem, not the truncation.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
* **Shared state.** :class:`TokenLedger` is in-memory and per-process. Run four
  workers and you have four times the budget. A real deployment needs Redis or a
  Postgres row with a transaction, and that is a deployment concern this repo
  does not pretend to solve — but the consequence is written down rather than
  discovered on the invoice.
* **A real tokenizer.** :func:`estimate_tokens` is characters divided by four,
  rounded up. A tokenizer would be exact and would pull ``tokenizers`` into the
  core path, which the guardrail latency budget and the dependency promise both
  forbid. The estimate is biased high on purpose: see its docstring.
* **Per-principal rate limiting.** Tokens per day is a cost control. Requests
  per second is a different control, it belongs in front of the application, and
  conflating them produces a limiter that is wrong about both.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from sightline.guardrails.grounding import citations_for, refuse
from sightline.store.base import Hit
from sightline.types import Answer, PlanStrategy, RefusalReason

__all__ = [
    "CHARS_PER_TOKEN",
    "EXTRACTIVE_NOTICE",
    "Budget",
    "BudgetDecision",
    "Reservation",
    "Rung",
    "TokenLedger",
    "clamp_k",
    "context_chars",
    "decide",
    "estimate_tokens",
    "extractive_answer",
    "refuse_over_budget",
]


class Rung(str, Enum):
    """Which rung of the degradation ladder this request lands on."""

    FULL = "full"
    EXTRACTIVE = "extractive"
    REFUSE = "refuse"


#: Rough bytes-per-token for English in a BPE vocabulary. Real value drifts
#: between 3.5 and 4.5 for prose and is closer to 2 for code, JSON and any
#: non-Latin script. Everything here rounds *up* and adds a margin, because the
#: cost of over-estimating is an answer that degrades one rung early, and the
#: cost of under-estimating is a provider call that exceeds a budget somebody is
#: paying for.
CHARS_PER_TOKEN = 4

#: Prefixed to every extractive answer. The user must be able to tell that no
#: model wrote this and that it is passages rather than an answer; an extractive
#: answer that looks synthesised is the failure this rung exists to avoid.
EXTRACTIVE_NOTICE = (
    "Answering from the source passages directly, without a written summary, "
    "because this request is at its limit. The passages below are the ones you "
    "are permitted to see, in relevance order."
)


@dataclass(frozen=True, slots=True)
class Budget:
    """Caps for one deployment. All of them are hard; none of them raise.

    Attributes:
        max_k: Hard ceiling on the number of candidates requested from the
            index. Applied *before* search, because recheck cost is linear in k
            and an unbounded k is a denial-of-service vector against the tuple
            store (PRD risk row: "recheck becomes a denial-of-service vector").
        max_context_chars: Ceiling on retrieved characters put in front of the
            model. 8,000 is roughly 2,000 tokens, matching the FRD default.
        per_principal_daily_tokens: One person's daily allowance. Sized so a
            heavy user cannot spend the organisation's day by lunchtime.
        global_daily_tokens: The whole deployment's daily allowance, the number
            that stops a runaway loop being a four-figure invoice.
        max_request_tokens: Ceiling on a single request's estimated spend.
        expected_output_tokens: Reserved for the model's reply, since the spend
            is not known until after the call and the check happens before it.
        usd_per_1k_tokens: Optional, for reporting a cost alongside the tokens.
            Priced per deployment because every provider charges differently and
            a hardcoded number here would be stale within a quarter.
    """

    max_k: int = 50
    max_context_chars: int = 8_000
    per_principal_daily_tokens: int = 200_000
    global_daily_tokens: int = 5_000_000
    max_request_tokens: int = 4_000
    expected_output_tokens: int = 400
    usd_per_1k_tokens: float | None = None


def estimate_tokens(text: str) -> int:
    """Characters over four, rounded up. An estimate, and it says so.

    Deliberately biased high. An under-estimate means the provider call happens
    and the budget is already blown by the time the real count arrives; an
    over-estimate means one request degrades to extractive slightly early, which
    is a worse answer rather than an unplanned cost.
    """
    if not text:
        return 0
    return -(-len(text) // CHARS_PER_TOKEN)


def clamp_k(requested_k: int, budget: Budget) -> int:
    """Clamp k *before* the search, never after.

    Clamping after search would mean discarding rechecked evidence, which is the
    thing this module exists to refuse to do. Clamping before means the index
    was only ever asked for what the system can afford to check.
    """
    return max(1, min(int(requested_k), budget.max_k))


def context_chars(hits: Sequence[Hit]) -> int:
    """Characters the rechecked set would contribute to the prompt.

    Counts the per-chunk framing as well as the text, because the delimiters and
    chunk id labels from :func:`sightline.guardrails.injection.context_block` are
    real tokens somebody pays for.
    """
    # ~24 characters of "[chunk c_1042]" plus newlines, per chunk.
    return sum(len(hit.text) for hit in hits) + 24 * len(hits)


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """Which rung, why, and what it would cost. Computed before any provider call."""

    rung: Rung
    why: str
    estimated_tokens: int
    context_chars: int
    #: Set when the FULL rung was taken and tokens were reserved. Settle it with
    #: the real usage, or release it if the call never happened.
    reservation: Reservation | None = None

    @property
    def refused(self) -> bool:
        return self.rung is Rung.REFUSE

    def estimated_usd(self, budget: Budget) -> float | None:
        if budget.usd_per_1k_tokens is None:
            return None
        return self.estimated_tokens / 1000 * budget.usd_per_1k_tokens


@dataclass(slots=True)
class Reservation:
    """Tokens held against a principal between the check and the real usage.

    Reserve-then-settle rather than check-then-record, because check-then-record
    lets N concurrent requests all read the same "remaining" and all proceed.
    The reservation is the estimate; :meth:`settle` replaces it with the truth.
    """

    principal: str
    day: str
    estimated: int
    _ledger: TokenLedger
    _closed: bool = False

    def settle(self, actual_tokens: int) -> None:
        """Replace the estimate with the provider's reported usage."""
        if self._closed:
            return
        self._closed = True
        self._ledger._adjust(self.principal, self.day, actual_tokens - self.estimated)

    def release(self) -> None:
        """Give the tokens back. Call this when the provider call never happened."""
        self.settle(0)


def _utc_day() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


@dataclass
class TokenLedger:
    """Per-principal and global daily token counters. In memory, per process.

    Thread-safe, because FastAPI runs sync handlers in a thread pool and two
    requests from the same principal will hit this at the same time on the first
    day anybody uses the system.

    Not durable and not shared. A restart forgives the day's spend and four
    workers multiply every budget by four. Both are stated in the module
    docstring rather than left for the invoice to reveal.
    """

    budget: Budget = field(default_factory=Budget)
    #: Injectable so tests do not have to wait for midnight UTC.
    clock: Callable[[], str] = _utc_day
    _per_principal: dict[tuple[str, str], int] = field(default_factory=dict)
    _global: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def spent(self, principal_ref: str, day: str | None = None) -> int:
        with self._lock:
            return self._per_principal.get((principal_ref, day or self.clock()), 0)

    def spent_globally(self, day: str | None = None) -> int:
        with self._lock:
            return self._global.get(day or self.clock(), 0)

    def remaining(self, principal_ref: str) -> tuple[int, int]:
        """``(principal_remaining, global_remaining)`` for today. Never negative."""
        day = self.clock()
        with self._lock:
            mine = self.budget.per_principal_daily_tokens - self._per_principal.get(
                (principal_ref, day), 0
            )
            everyone = self.budget.global_daily_tokens - self._global.get(day, 0)
        return max(0, mine), max(0, everyone)

    def reserve(self, principal_ref: str, tokens: int) -> Reservation | None:
        """Hold ``tokens`` against today's budgets, or return ``None`` if they do not fit.

        ``None`` is not an error and must not become one. The caller degrades to
        the extractive rung, which spends no provider tokens at all — a user at
        their limit still gets permitted passages with real citations.
        """
        day = self.clock()
        with self._lock:
            mine = self._per_principal.get((principal_ref, day), 0)
            everyone = self._global.get(day, 0)
            if mine + tokens > self.budget.per_principal_daily_tokens:
                return None
            if everyone + tokens > self.budget.global_daily_tokens:
                return None
            self._per_principal[(principal_ref, day)] = mine + tokens
            self._global[day] = everyone + tokens
            self._prune(day)
        return Reservation(principal=principal_ref, day=day, estimated=tokens, _ledger=self)

    def _adjust(self, principal_ref: str, day: str, delta: int) -> None:
        with self._lock:
            key = (principal_ref, day)
            self._per_principal[key] = max(0, self._per_principal.get(key, 0) + delta)
            self._global[day] = max(0, self._global.get(day, 0) + delta)

    def _prune(self, today: str, keep_days: int = 3) -> None:
        """Drop counters older than a few days so this dict stays bounded.

        Called under the lock. Three days rather than one because a deployment
        spanning timezones will still be writing yesterday's bucket when the UTC
        date rolls, and because the memory is a few hundred bytes per principal.
        """
        if len(self._global) <= keep_days:
            return
        keep = set(sorted(self._global, reverse=True)[:keep_days]) | {today}
        self._global = {day: n for day, n in self._global.items() if day in keep}
        self._per_principal = {
            key: n for key, n in self._per_principal.items() if key[1] in keep
        }


def decide(
    principal_ref: str,
    hits: Sequence[Hit],
    *,
    question: str = "",
    budget: Budget | None = None,
    ledger: TokenLedger | None = None,
) -> BudgetDecision:
    """Pick a rung. Always called **before** the provider call, never after.

    Note the type of ``hits``: :class:`~sightline.store.base.Hit`, meaning every
    candidate here already passed :func:`sightline.authz.recheck.recheck`. There
    is no overload that takes ``UncheckedHit``, so "skip the recheck on the
    overflow to save time" (planted mutant M10) cannot be written against this
    function without an import that gives it away in review.

    Args:
        principal_ref: Who is asking, e.g. ``user:alice``. Used as the ledger key
            and nothing else — it never reaches the filter, which was compiled
            long before this point.
        hits: The complete rechecked set. Complete: passing a truncated list here
            is the silent-evidence-loss bug, committed one frame earlier.
        question: The user's query, counted toward the request's token estimate.
        budget: Caps. Defaults to :class:`Budget`'s defaults.
        ledger: Daily counters. Omit to skip the daily budgets entirely and check
            only the per-request caps, which is what a single-shot script wants.

    Returns:
        A :class:`BudgetDecision`. When the rung is ``FULL`` the decision carries
        a live :class:`Reservation` that the caller must settle or release.
    """
    caps = budget or Budget()
    chars = context_chars(hits)
    estimated = (
        estimate_tokens(question)
        + -(-chars // CHARS_PER_TOKEN)
        + caps.expected_output_tokens
    )

    if not hits:
        # Nothing to be extractive *from*. This is an evidence outcome, not a
        # budget one, and the caller should have refused before reaching here;
        # returning REFUSE keeps the function total rather than raising.
        return BudgetDecision(
            rung=Rung.REFUSE,
            why="no rechecked evidence to answer or quote from",
            estimated_tokens=estimated,
            context_chars=chars,
        )

    if chars > caps.max_context_chars:
        return BudgetDecision(
            rung=Rung.EXTRACTIVE,
            why=f"context {chars} chars over cap {caps.max_context_chars}",
            estimated_tokens=estimated,
            context_chars=chars,
        )

    if estimated > caps.max_request_tokens:
        return BudgetDecision(
            rung=Rung.EXTRACTIVE,
            why=f"estimated {estimated} tokens over per-request cap {caps.max_request_tokens}",
            estimated_tokens=estimated,
            context_chars=chars,
        )

    if ledger is None:
        return BudgetDecision(
            rung=Rung.FULL,
            why="within per-request caps; no daily ledger in use",
            estimated_tokens=estimated,
            context_chars=chars,
        )

    reservation = ledger.reserve(principal_ref, estimated)
    if reservation is None:
        mine, everyone = ledger.remaining(principal_ref)
        return BudgetDecision(
            rung=Rung.EXTRACTIVE,
            why=f"daily budget exhausted (principal {mine}, global {everyone} left)",
            estimated_tokens=estimated,
            context_chars=chars,
        )

    return BudgetDecision(
        rung=Rung.FULL,
        why="within all budgets",
        estimated_tokens=estimated,
        context_chars=chars,
        reservation=reservation,
    )


def extractive_answer(
    hits: Sequence[Hit],
    *,
    budget: Budget | None = None,
    strategy: PlanStrategy | None = None,
    epoch: int = 0,
    redact_egress: bool = False,
) -> Answer:
    """Rung two: permitted passages verbatim, real citations, no generated prose.

    Still correct and still useful. Every passage passed recheck, every citation
    was reconstructed from the rechecked set, and no model saw any of it — which
    is also why this rung is the right answer to an exhausted *token* budget and
    not merely to an oversized context.

    Passages are included in relevance order until the character cap is reached,
    and the notice states how many of how many are shown. That disclosure is the
    entire difference between this and silently truncating a synthesised answer.

    Args:
        hits: The complete rechecked set, in relevance order.
        budget: Caps. ``max_context_chars`` bounds the answer body too.
        strategy: Recorded on the response.
        epoch: Policy epoch, recorded on the response.
        redact_egress: Run PII redaction over the quoted passages. Off by
            default, and the reason is the position in
            :mod:`sightline.guardrails.pii`: this text is going to a reader the
            authorisation layer already cleared, and redacting a permitted
            answer harms the permitted reader without protecting anyone.
            Egress redaction is mandatory for logs and traces, which is a
            different call site.

    Returns:
        A grounded (not refused) :class:`~sightline.types.Answer`, or a
        ``BUDGET_EXCEEDED`` refusal if not even one passage fits.
    """
    caps = budget or Budget()
    body: list[str] = []
    used_ids: list[str] = []
    spent = len(EXTRACTIVE_NOTICE)

    for hit in hits:
        entry = f"[{hit.chunk_id}] {hit.text.strip()}"
        if spent + len(entry) > caps.max_context_chars and used_ids:
            break
        if spent + len(entry) > caps.max_context_chars:
            # Not even the top passage fits. Truncating it would quote a
            # sentence fragment as if it were the evidence, which is worse than
            # saying no, so this is where the ladder ends.
            return refuse_over_budget(strategy=strategy, epoch=epoch)
        body.append(entry)
        used_ids.append(hit.chunk_id)
        spent += len(entry)

    if not used_ids:
        return refuse_over_budget(strategy=strategy, epoch=epoch)

    passages = "\n\n".join(body)
    if redact_egress:
        from sightline.guardrails.pii import redact_for_egress

        passages = redact_for_egress(passages).text

    shown = f"Showing {len(used_ids)} of {len(hits)} permitted passages."
    return Answer(
        text=f"{EXTRACTIVE_NOTICE} {shown}\n\n{passages}",
        citations=citations_for(hits, used_ids),
        refused=False,
        strategy=strategy,
        epoch=epoch,
    )


def refuse_over_budget(
    *, strategy: PlanStrategy | None = None, epoch: int = 0
) -> Answer:
    """Rung three. The HTTP layer maps this to 429.

    Reached only when the extractive rung cannot fit a single passage. It is the
    bottom of the ladder and it should be rare; if it is not rare, the fix is a
    larger ``max_context_chars`` or smaller chunks, not a skipped check.
    """
    return refuse(RefusalReason.BUDGET_EXCEEDED, strategy=strategy, epoch=epoch)
