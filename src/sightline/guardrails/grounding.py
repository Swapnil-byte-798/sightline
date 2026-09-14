"""Grounding, citation reconstruction, and two refusals that must look identical.

THE USUAL APPROACH, AND WHY IT DOES NOT WORK
--------------------------------------------
Most systems ask nicely: *"cite your sources using [doc:N]"*. The model complies
most of the time and invents a plausible identifier the rest of the time, and you
get a footnote pointing at a document that does not exist — or worse, one that
exists and the reader may not see.

Sightline makes that unrepresentable. The synthesiser's return type is
:class:`Claim`: text plus **chunk ids only**, never a formatted citation string.
Citations are then *reconstructed* by looking each id up in the rechecked set —
the hits that passed the live permission check for this principal on this
request. An id that is not in that map has nothing to map to, so it produces
nothing. Look at :class:`sightline.types.Answer`: there is no free-text citation
field for a fabricated one to live in, and the absence is the feature. Parsing
citations out of generated text is planted mutant M14/M15.

BE PRECISE ABOUT WHAT THIS BUYS
-------------------------------
Structural reconstruction makes a *fabricated identifier* impossible. It does
not make a *wrong attribution* impossible: the model can attach a real,
permitted chunk id to a claim that chunk does not support. That is a different
failure and it needs :func:`support_score`, which is a lexical overlap test, is
approximate, and is honestly the weakest guardrail in the system. Do not let the
type-level guarantee launder the heuristic one. The two are kept in separate
functions here so that nobody can confuse them by accident.

EXISTENCE PROTECTION
--------------------
If "there is no such document" and "there is a document, but not for you"
produce different output, an attacker enumerates the corpus one question at a
time — and document titles and project code names are frequently the sensitive
part. So ``NO_EVIDENCE_AT_ALL`` and ``NO_PERMITTED_EVIDENCE`` produce a
byte-identical :class:`~sightline.types.Answer`, carrying one canonical reason.
The true reason is returned separately, for the audit log only. See
:func:`refuse_no_evidence`.

THE TIMING CHANNEL, STATED RATHER THAN HIDDEN
---------------------------------------------
Byte-equality does not close it. The empty-plan short circuit (guardrail 3)
returns in about a millisecond; the searched-and-found-nothing path takes about
sixty. The bodies match exactly and the clock is a perfect oracle. Three options
existed: a constant-time floor on the refusal path, a decoy search that is
discarded, or documenting the channel and accepting it. This module implements
the floor (:func:`existence_floor`), because the decoy costs real work on the
cheapest path in the system. The floor narrows the channel to the jitter of
``time.sleep`` and does not eliminate it — a patient attacker with thousands of
timed probes still wins, and against that threat model the decoy is the answer.
The acceptance test asserts the difference sits below the harness noise floor
rather than asserting it is zero, because a wall-clock test that asserts zero is
a test that lies.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
* **A model-based entailment check.** It would be better at the job and it puts
  a second promptable component in the path of untrusted text. If one is added
  it belongs behind the same "may only raise severity" rule as the injection
  classifier.
* **Sentence splitting of generated prose.** The synthesiser returns claims
  already separated. Re-splitting its output would mean parsing generated text,
  which is exactly the habit this module exists to break.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from sightline.store.base import Hit
from sightline.types import Answer, Citation, ObjectRef, PlanStrategy, RefusalReason

__all__ = [
    "Claim",
    "ClaimVerdict",
    "GroundingResult",
    "MIN_SUPPORT",
    "EXISTENCE_PROTECTED_TEXT",
    "EXISTENCE_PROTECTED_REASON",
    "REFUSAL_FLOOR_SECONDS",
    "REFUSAL_TEXT",
    "support_score",
    "citations_for",
    "reconstruct",
    "enforce",
    "build_answer",
    "refuse",
    "refuse_no_evidence",
    "existence_floor",
]


@dataclass(frozen=True, slots=True)
class Claim:
    """One assertion plus the chunk ids the synthesiser says support it.

    This is the synthesiser's entire output vocabulary. There is no field for a
    formatted citation, a URL, a title or a page number, and adding one would
    reopen the fabrication hole this design closes: any such field is free text,
    and free text about a document is a claim the permission layer never checked.
    """

    text: str
    chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """What happened to one claim, in enough detail to explain a refusal."""

    claim: Claim
    kept_ids: tuple[str, ...]
    #: Ids the model returned that are not in the rechecked set. Either
    #: hallucinated, or belonging to a document this principal cannot see. Both
    #: are dropped, and the two are indistinguishable here on purpose.
    dropped_ids: tuple[str, ...]
    support: float
    supported: bool
    why: str


@dataclass(frozen=True, slots=True)
class GroundingResult:
    verdicts: tuple[ClaimVerdict, ...]
    citations: tuple[Citation, ...]

    @property
    def kept(self) -> tuple[ClaimVerdict, ...]:
        return tuple(v for v in self.verdicts if v.supported and v.kept_ids)

    @property
    def grounded(self) -> bool:
        """True when at least one claim survived with at least one real citation."""
        return bool(self.kept)

    @property
    def dropped_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for verdict in self.verdicts:
            for chunk_id in verdict.dropped_ids:
                if chunk_id not in seen:
                    seen.append(chunk_id)
        return tuple(seen)

    def text(self) -> str:
        return " ".join(v.claim.text.strip() for v in self.kept if v.claim.text.strip())

    def summary(self) -> str:
        """For the log. Counts and ids, never claim text or chunk text."""
        return (
            f"claims={len(self.verdicts)} kept={len(self.kept)} "
            f"citations={len(self.citations)} dropped_ids={len(self.dropped_ids)}"
        )


#: Fraction of a claim's content words that must appear in its cited chunks.
#: Tuned by hand on a handful of examples, which is exactly as rigorous as it
#: sounds. It is a floor against wholesale invention, not a measurement of
#: entailment, and no accuracy number is published for it anywhere in this repo.
MIN_SUPPORT = 0.45

# Words carrying no evidential weight. Short list on purpose: a long stopword
# list starts deleting the words that decide meaning ("not", "except", "only").
_STOPWORDS = frozenset(
    """
    a an and are as at be been by for from had has have in into is it its of on
    or that the their there these they this to was were what when where which
    who will with your you our we
    """.split()
)

_WORD = re.compile(r"[\w£$€%.,/-]+")
_NUMERIC = re.compile(r"^[£$€]?\d[\d.,/%-]*$")
_NEGATIONS = frozenset(
    {"no", "not", "never", "none", "cannot", "without", "except", "excluding", "denied"}
)


def _tokens(text: str) -> list[str]:
    return [t.strip(".,").casefold() for t in _WORD.findall(text) if t.strip(".,")]


def _content(tokens: Iterable[str]) -> set[str]:
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def _normalise_number(token: str) -> str:
    """``£1,200`` and ``1200`` compare equal; ``26%`` and ``26`` do not."""
    return re.sub(r"[\u00a3$\u20ac,]", "", token).rstrip(".")


def _numbers(tokens: Sequence[str]) -> set[str]:
    """Normalised numeric tokens."""
    return {
        _normalise_number(t) for t in tokens if _NUMERIC.match(t) and _normalise_number(t)
    }


def _number_units(tokens: Sequence[str]) -> set[str]:
    """Number-with-its-next-word pairs: ``26 weeks``, ``12 months``.

    The bare-number check is not enough and the failure is worth spelling out.
    A claim of "12 weeks" cited against text reading "26 weeks at full pay for
    employees with 12 months service" passes a bare-number check, because 12 is
    right there — in a different sentence, attached to a different unit. Pairing
    each number with the word after it catches that, which is the single most
    valuable line in this function.

    It over-refuses when the evidence spells a number in words ("twenty-six
    weeks"). That direction is the safe one and it is not free: an answer that
    was actually correct gets refused. Stated rather than tuned away.
    """
    pairs: set[str] = set()
    for index, token in enumerate(tokens[:-1]):
        if not _NUMERIC.match(token):
            continue
        following = tokens[index + 1]
        if following.isalpha() and len(following) > 1:
            pairs.add(f"{_normalise_number(token)} {following}")
    return pairs


def support_score(claim_text: str, evidence: str) -> tuple[float, str]:
    """Lexical overlap between a claim and its cited text. The weakest guardrail.

    Three signals, none of them entailment:

    1. Content-word overlap, as a fraction of the claim's content words.
    2. **Numeric agreement**, at two levels: every number in the claim must
       appear in the evidence, and every number must appear *with the same unit
       after it*. "26 weeks" versus "12 weeks" is the failure mode that actually
       hurts people, a lexical overlap score barely notices the difference, and
       a bare-number check misses it whenever the wrong number happens to occur
       elsewhere in the chunk. See :func:`_number_units`.
    3. A negation-mismatch penalty. A claim that says "not entitled" cited
       against text with no negation in it is suspicious. Crude — it cannot tell
       which clause the negation attaches to — so it lowers the score rather
       than deciding on its own.

    Args:
        claim_text: One claim from the synthesiser.
        evidence: The concatenated text of the chunks it cited.

    Returns:
        ``(score, why)`` with score in ``[0, 1]``. ``why`` is a short phrase for
        the audit log; it quotes neither the claim nor the evidence.
    """
    claim_tokens = _tokens(claim_text)
    evidence_tokens = _tokens(evidence)
    claim_content = _content(claim_tokens)
    if not claim_content:
        # A claim with no content words is not grounded, it is punctuation.
        return 0.0, "claim has no content words"
    evidence_content = _content(evidence_tokens)

    overlap = len(claim_content & evidence_content) / len(claim_content)

    missing = _numbers(claim_tokens) - _numbers(evidence_tokens)
    if missing:
        return 0.0, f"{len(missing)} number(s) in the claim absent from the cited text"

    mismatched = _number_units(claim_tokens) - _number_units(evidence_tokens)
    if mismatched:
        return 0.0, f"{len(mismatched)} number/unit pair(s) not found in the cited text"

    negated_claim = bool({t for t in claim_tokens} & _NEGATIONS)
    negated_evidence = bool({t for t in evidence_tokens} & _NEGATIONS)
    if negated_claim != negated_evidence:
        overlap *= 0.6
        return overlap, f"overlap {overlap:.2f} after negation-mismatch penalty"

    return overlap, f"overlap {overlap:.2f}"


def citations_for(
    hits: Sequence[Hit], chunk_ids: Iterable[str] | None = None
) -> tuple[Citation, ...]:
    """Build citations from rechecked hits. The only way a citation is ever made.

    Args:
        hits: Hits that survived :func:`sightline.authz.recheck.recheck`. Nothing
            else is acceptable: a citation built from an ``UncheckedHit`` is a
            citation to a document nobody confirmed this principal can see.
        chunk_ids: Restrict to these ids, in the order given. ``None`` cites
            every hit, which is what the extractive path wants.

    Returns:
        Citations, deduplicated by chunk id, in the requested order.
    """
    by_id = {hit.chunk_id: hit for hit in hits}
    wanted = list(chunk_ids) if chunk_ids is not None else [h.chunk_id for h in hits]
    out: list[Citation] = []
    seen: set[str] = set()
    for chunk_id in wanted:
        hit = by_id.get(chunk_id)
        if hit is None or chunk_id in seen:
            continue
        seen.add(chunk_id)
        out.append(
            Citation(
                chunk_id=hit.chunk_id,
                object=ObjectRef.parse(hit.object_ref),
                score=hit.score,
                why_allowed=hit.why_allowed,
            )
        )
    return tuple(out)


def reconstruct(
    claims: Sequence[Claim], hits: Sequence[Hit]
) -> tuple[tuple[Claim, ...], tuple[str, ...]]:
    """Drop every chunk id the model returned that is not in the rechecked set.

    Guardrail 8. Returns the surviving claims (those with at least one real id
    left) and the dropped ids. Dropping is silent to the user by design: telling
    them "your answer cited c_9999, which you cannot see" would confirm that
    ``c_9999`` exists, which is the enumeration leak existence protection is
    there to prevent.
    """
    permitted = {hit.chunk_id for hit in hits}
    kept: list[Claim] = []
    dropped: list[str] = []
    for claim in claims:
        surviving = tuple(cid for cid in claim.chunk_ids if cid in permitted)
        for cid in claim.chunk_ids:
            if cid not in permitted and cid not in dropped:
                dropped.append(cid)
        if surviving:
            kept.append(Claim(text=claim.text, chunk_ids=surviving))
    return tuple(kept), tuple(dropped)


def enforce(
    claims: Sequence[Claim],
    hits: Sequence[Hit],
    *,
    min_support: float = MIN_SUPPORT,
) -> GroundingResult:
    """Run both grounding controls over the synthesiser's output.

    Order matters. Reconstruction (structural, exact) runs first, so the support
    check (heuristic, approximate) only ever sees chunk text this principal is
    allowed to see. Reversing them would score claims against forbidden
    evidence, which is a leak even if the score is discarded.

    Args:
        claims: Straight from the synthesiser. Text plus chunk ids.
        hits: The rechecked set for this request.
        min_support: Overlap floor. See :data:`MIN_SUPPORT` for how little that
            number means.

    Returns:
        A :class:`GroundingResult`. Check :attr:`GroundingResult.grounded`
        before building an answer; if it is false, refuse with
        ``RefusalReason.UNGROUNDED``.
    """
    by_id: Mapping[str, Hit] = {hit.chunk_id: hit for hit in hits}
    verdicts: list[ClaimVerdict] = []
    cited_order: list[str] = []

    for claim in claims:
        kept_ids = tuple(cid for cid in claim.chunk_ids if cid in by_id)
        dropped_ids = tuple(cid for cid in claim.chunk_ids if cid not in by_id)
        if not kept_ids:
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    kept_ids=(),
                    dropped_ids=dropped_ids,
                    support=0.0,
                    supported=False,
                    why="no cited chunk survived recheck",
                )
            )
            continue
        evidence = "\n".join(by_id[cid].text for cid in kept_ids)
        score, why = support_score(claim.text, evidence)
        supported = score >= min_support
        if supported:
            cited_order.extend(cid for cid in kept_ids if cid not in cited_order)
        verdicts.append(
            ClaimVerdict(
                claim=claim,
                kept_ids=kept_ids,
                dropped_ids=dropped_ids,
                support=score,
                supported=supported,
                why=why,
            )
        )

    return GroundingResult(
        verdicts=tuple(verdicts),
        citations=citations_for(hits, cited_order),
    )


def build_answer(
    claims: Sequence[Claim],
    hits: Sequence[Hit],
    *,
    strategy: PlanStrategy | None = None,
    epoch: int = 0,
    min_support: float = MIN_SUPPORT,
) -> tuple[Answer, GroundingResult]:
    """Turn claims into an :class:`~sightline.types.Answer`, or refuse.

    There is no third state. An answer with zero citations and ``refused=False``
    is impossible by construction here, and a test asserts it (FR-19).

    Returns:
        ``(answer, result)``. The result is audit material — it carries claim
        text and support scores — and must not be serialised into the response.
    """
    result = enforce(claims, hits, min_support=min_support)
    if not result.grounded:
        return (
            refuse(RefusalReason.UNGROUNDED, strategy=strategy, epoch=epoch),
            result,
        )
    return (
        Answer(
            text=result.text(),
            citations=result.citations,
            refused=False,
            strategy=strategy,
            epoch=epoch,
        ),
        result,
    )


#: What the user sees for both empty-handed outcomes. One string, one constant,
#: referenced twice, so that a future edit cannot make them diverge — which is
#: precisely planted mutant M11.
EXISTENCE_PROTECTED_TEXT = (
    "I could not find anything I am able to show you for that question."
)

#: The reason stamped on the response for both. The permitted-set reason is the
#: canonical one because it never asserts anything about the corpus: saying "no
#: permitted evidence" is true whether or not the document exists, whereas
#: "no evidence at all" would be a statement about what the corpus contains.
EXISTENCE_PROTECTED_REASON = RefusalReason.NO_PERMITTED_EVIDENCE

#: Floor for the empty-handed refusal path, seconds. Sized to sit above the
#: searched-and-found-nothing path on the reference machine (~60 ms). Too low
#: and it closes nothing; too high and it becomes the latency story.
REFUSAL_FLOOR_SECONDS = 0.075

#: Refusal bodies. Fixed strings: no chunk text, no document names, no counts,
#: no matched spans. Planted mutant M12 is document content appearing in an
#: error or refusal payload, and the way that mutant gets written is by someone
#: helpfully interpolating a detail into one of these.
REFUSAL_TEXT: dict[RefusalReason, str] = {
    RefusalReason.NO_PERMITTED_EVIDENCE: EXISTENCE_PROTECTED_TEXT,
    RefusalReason.NO_EVIDENCE_AT_ALL: EXISTENCE_PROTECTED_TEXT,
    RefusalReason.UNGROUNDED: (
        "I found material but could not support an answer from it, so I am not "
        "going to guess."
    ),
    # Naming injection tells an attacker who planted a document that it was
    # retrieved. That is a real signal and it is accepted: the alternative is a
    # generic refusal that makes this guardrail impossible to operate, and the
    # attacker already knows what they wrote.
    RefusalReason.INJECTION_DETECTED: (
        "A retrieved document contained content that appears to be instructions "
        "aimed at this assistant, so this query was not answered."
    ),
    RefusalReason.BUDGET_EXCEEDED: (
        "This request is over its budget. Narrow the question or try again later."
    ),
    RefusalReason.STALE_POLICY: (
        "Permissions changed while this query was running. Retry."
    ),
}


def refuse(
    reason: RefusalReason,
    *,
    strategy: PlanStrategy | None = None,
    epoch: int = 0,
) -> Answer:
    """Build a refusal. Never carries evidence, never carries chunk text.

    For the two empty-handed reasons use :func:`refuse_no_evidence` instead: it
    normalises the reason so the response cannot distinguish them.
    """
    if reason in (RefusalReason.NO_PERMITTED_EVIDENCE, RefusalReason.NO_EVIDENCE_AT_ALL):
        answer, _ = refuse_no_evidence(actual=reason, strategy=strategy, epoch=epoch)
        return answer
    return Answer(
        text=REFUSAL_TEXT[reason],
        citations=(),
        refused=True,
        refusal_reason=reason,
        strategy=strategy,
        epoch=epoch,
    )


def refuse_no_evidence(
    *,
    actual: RefusalReason,
    strategy: PlanStrategy | None = None,
    epoch: int = 0,
) -> tuple[Answer, RefusalReason]:
    """The two empty-handed refusals, made indistinguishable.

    "There is no such document" and "there is a document but not for you" return
    the same bytes: same text, same reason, same citations, same flags. If they
    did not, anyone who can phrase a question can enumerate the corpus, and
    document titles and project code names are often the sensitive part.

    Args:
        actual: The true reason. Goes to the audit log and nowhere else.
        strategy: Recorded on the response. Note that the strategy itself is a
            weak signal about the principal's own permissions, not about the
            corpus, which is why it is not suppressed.
        epoch: Policy epoch at refusal time.

    Returns:
        ``(answer, audit_reason)``. Serialise the answer; log the reason. Never
        put ``audit_reason`` in a response payload.
    """
    answer = Answer(
        text=EXISTENCE_PROTECTED_TEXT,
        citations=(),
        refused=True,
        refusal_reason=EXISTENCE_PROTECTED_REASON,
        existence_protected=True,
        strategy=strategy,
        epoch=epoch,
    )
    return answer, actual


@contextmanager
def existence_floor(seconds: float = REFUSAL_FLOOR_SECONDS) -> Iterator[None]:
    """Hold the empty-handed refusal path open for a fixed minimum duration.

    Wrap the *whole* short-circuit path, from the moment the request is accepted
    to the moment the refusal is built, so that the one-millisecond "your plan
    admits nothing" answer and the sixty-millisecond "we searched and found
    nothing" answer take comparable time on the wall clock.

    This narrows the channel to scheduler jitter. It does not close it, and no
    sleep-based approach can: ``time.sleep`` guarantees a minimum, not a
    distribution. Closing it properly means running the decoy search. The cost
    of that is real work on the cheapest path in the system, which is why it is
    not the default.

    Usage::

        with existence_floor():
            answer, audit_reason = refuse_no_evidence(actual=reason)
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        remaining = seconds - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)
