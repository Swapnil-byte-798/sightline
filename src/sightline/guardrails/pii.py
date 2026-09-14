"""PII detection and redaction, at two moments that are not substitutes.

Personally identifiable information can be removed on the way in, on the way
out, or both. They are different controls with different costs:

============================  ==============================  ==========================
\\                             Redact at ingest                Redact at egress
============================  ==============================  ==========================
When                          Before chunking and embedding   After synthesis, before
                                                              the response, logs, traces
What the index holds          Redacted text only              The original text
Effect on retrieval           Destructive; you cannot          None
                              retrieve what is not there
Effect on a permitted reader  Harms them — HR cannot get an    None
                              answer about a salary either
Protects against              The index leaking: backups, a    A permitted answer copied
                              compromised vector store         into a lower-trust store
Reversible                    No                               Yes
============================  ==============================  ==========================

THE MISTAKE THIS MODULE REFUSES TO MAKE
---------------------------------------
Redaction cannot be selective by reader. It happens once, for everybody. If you
find yourself redacting because "not everyone should see that", you have
written an access-control rule in the wrong layer and it will be wrong for the
people who are allowed. **Permission is the control; redaction is hygiene.**

So :data:`INGEST_KINDS` is deliberately short: card numbers, national insurance
and social security numbers, IBANs, and live credentials — things that should
not be in an AI pipeline under *any* permission. Everything else is left in the
index, where the authorisation layer decides who sees it.

:data:`EGRESS_KINDS` is wide, and the surface it protects is not the answer. It
is the debugging. Planted mutant M12 is "chunk text appears in an error or
refusal payload", because that is the realistic way document content escapes:
log lines, span attributes, exception messages. Use :func:`redact_for_log` on
anything that is about to be written somewhere with weaker access control than
the index.

THE ENRON QUESTION
------------------
This repo indexes the Enron email corpus, which is not a synthetic dataset. It
is half a million real messages from real people, released by a regulator, and
it contains their home phone numbers, their family's medical news, their credit
card numbers and their affairs. None of them consented to being a benchmark.
Ingest-time redaction is partly a security control and partly the only ethical
answer available: the alternative is to build a searchable index of private
correspondence and call it a demo. It does not make the corpus consensual. It
removes the categories that would do concrete harm if the index leaked, which is
the most that can be done without abandoning the only realistic permission-shaped
corpus that is free to use.

OPTIONAL: MICROSOFT PRESIDIO
----------------------------
:class:`PresidioDetector` wraps ``presidio-analyzer`` when it is installed, and
adds the recogniser categories a regex cannot do — person names, locations,
dates of birth. It is deliberately **not** an extra in ``pyproject.toml``: it
pulls spaCy and a language model, which is several hundred megabytes on a laptop
with 8 GB of RAM, and the core promise of this repo is that the whole test suite
runs on pydantic, fastapi, numpy and httpx. Install it yourself if you want it.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
* **Name detection by regex.** Capitalised-word heuristics flag every product
  name and miss every lowercase signature. Names need a model; that is Presidio.
* **Bare nine-digit SSN matching.** Without separators it is indistinguishable
  from an order number, and the false positive rate on a corporate corpus is
  intolerable. Formatted SSNs are matched; unformatted ones are not, and that
  gap is stated rather than hidden.
* **Reversible tokenisation with a vault.** Placeholders here are one-way. A
  reversible scheme needs key management this project does not have.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

__all__ = [
    "EGRESS_KINDS",
    "FINGERPRINT_SALT_ENV",
    "INGEST_KINDS",
    "CompositeDetector",
    "Detector",
    "PiiKind",
    "PiiMatch",
    "PresidioDetector",
    "Redaction",
    "RegexDetector",
    "detect",
    "load_detector",
    "redact",
    "redact_chunks",
    "redact_for_egress",
    "redact_for_ingest",
    "redact_for_log",
]


class PiiKind(str, Enum):
    """What was found. A ``str`` enum because these end up in metrics labels.

    Cardinality is bounded and small, which is the requirement for a metric
    label — unlike the matched value itself, which never becomes one.
    """

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IBAN = "iban"
    IP_ADDRESS = "ip_address"
    SECRET = "secret"
    #: Presidio only. Regexes do not detect names and this module does not pretend.
    PERSON = "person"
    LOCATION = "location"


#: Removed before anything is embedded. Short on purpose: these are the
#: categories that should not be in an AI pipeline under any permission.
#: Everything absent from this set is left in the index for the authorisation
#: layer to protect, which is the layer that can be right about it.
INGEST_KINDS: frozenset[PiiKind] = frozenset(
    {PiiKind.CREDIT_CARD, PiiKind.SSN, PiiKind.IBAN, PiiKind.SECRET}
)

#: Removed on the way out, and on the way into any log, trace or error payload.
#: Wider, because egress redaction costs the permitted reader nothing they
#: needed — they can open the document.
EGRESS_KINDS: frozenset[PiiKind] = frozenset(
    {
        PiiKind.CREDIT_CARD,
        PiiKind.SSN,
        PiiKind.IBAN,
        PiiKind.SECRET,
        PiiKind.EMAIL,
        PiiKind.PHONE,
        PiiKind.IP_ADDRESS,
        PiiKind.PERSON,
        PiiKind.LOCATION,
    }
)

FINGERPRINT_SALT_ENV = "SIGHTLINE_REDACTION_SALT"

# Default salt is a constant, not a per-process random value, and that is a
# deliberate trade. A random salt would make placeholders unstable across runs,
# so re-ingesting an unchanged document would produce different text, a
# different hash and a rewritten vector — turning a hygiene control into a
# reindex storm. The cost is that a 24-bit fingerprint of a known-format value
# (an email address) is trivially reversible by dictionary attack. The
# fingerprint is a correlation handle for a human reading redacted text, not a
# secret. Set SIGHTLINE_REDACTION_SALT if you need it to be unlinkable.
_DEFAULT_SALT = "sightline/v1"


@dataclass(frozen=True, slots=True)
class PiiMatch:
    """One finding. ``text`` is the raw value and is audit-only.

    Never put a ``PiiMatch`` in a response, a log line or a span attribute:
    the whole point of the module is that this value does not travel. Use
    :meth:`Redaction.counts` for anything that leaves the process.
    """

    kind: PiiKind
    start: int
    end: int
    text: str
    detector: str = "regex"

    def placeholder(self, salt: str | None = None) -> str:
        """``[EMAIL:a1b2c3]`` — stable per distinct value, one-way.

        The fingerprint keeps coreference readable: two mentions of the same
        address redact to the same token, so "he emailed [EMAIL:a1b2c3] twice"
        still says something. A bare ``[EMAIL]`` loses that and reads worse.
        """
        material = f"{salt or os.environ.get(FINGERPRINT_SALT_ENV) or _DEFAULT_SALT}"
        digest = hashlib.blake2s(
            (material + self.kind.value + self.text.casefold()).encode("utf-8"),
            digest_size=4,
        ).hexdigest()[:6]
        return f"[{self.kind.name}:{digest}]"


@dataclass(frozen=True, slots=True)
class Redaction:
    """Redacted text plus what was taken out of it."""

    text: str
    matches: tuple[PiiMatch, ...] = ()
    original_chars: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.matches)

    def counts(self) -> dict[str, int]:
        """Kind -> count. The only form of this result safe to log."""
        totals: dict[str, int] = {}
        for match in self.matches:
            totals[match.kind.value] = totals.get(match.kind.value, 0) + 1
        return totals

    def summary(self) -> str:
        if not self.matches:
            return f"clean ({self.original_chars} chars)"
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts().items()))
        return f"redacted {parts} from {self.original_chars} chars"


def _luhn(digits: str) -> bool:
    """Card check digit. Rejects the order numbers that look like cards."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = ord(char) - 48
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _iban_valid(candidate: str) -> bool:
    """ISO 13616 mod-97. Without it, every uppercase token is an IBAN."""
    compact = candidate.replace(" ", "").upper()
    if len(compact) < 15 or len(compact) > 34:
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    if not numeric.isdigit():
        return False
    return int(numeric) % 97 == 1


def _valid_ip(candidate: str) -> bool:
    parts = candidate.split(".")
    return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


@dataclass(frozen=True, slots=True)
class _Pattern:
    kind: PiiKind
    pattern: re.Pattern[str]
    #: Extra test applied to the matched text. A regex that matches everything
    #: shaped like a card number is a regex that redacts your invoice numbers.
    validate: Callable[[str], bool] | None = None


def _no_validation(_: str) -> bool:
    return True


# Order matters only for overlapping shapes: SECRET before EMAIL so that a
# token containing an "@" is reported as the credential it is.
_PATTERNS: tuple[_Pattern, ...] = (
    _Pattern(
        PiiKind.SECRET,
        re.compile(
            r"-----BEGIN [A-Z ]{0,32}PRIVATE KEY-----"
            r"|\bAKIA[0-9A-Z]{16}\b"
            r"|\bgh[pousr]_[A-Za-z0-9]{30,}\b"
            r"|\bxox[baprs]-[A-Za-z0-9-]{10,}\b"
            r"|\b(?:sk|pk|rk)-[A-Za-z0-9_-]{20,}\b"
            r"|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
        ),
    ),
    _Pattern(
        PiiKind.EMAIL,
        re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    ),
    _Pattern(
        PiiKind.SSN,
        # Structurally valid US SSNs only: no 000/666/9xx area, no 00 group, no
        # 0000 serial. Unformatted nine-digit runs are not matched — see the
        # module docstring for why that gap is deliberate.
        re.compile(r"\b(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}\b"),
    ),
    _Pattern(
        PiiKind.CREDIT_CARD,
        re.compile(r"\b\d(?:[ -]?\d){12,18}\b"),
        lambda text: _luhn(re.sub(r"\D", "", text)),
    ),
    _Pattern(
        PiiKind.IBAN,
        re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{2,4}){2,8}\b"),
        _iban_valid,
    ),
    _Pattern(
        PiiKind.PHONE,
        # Three narrow shapes rather than one broad one. Phone numbers are the
        # worst category in this file: every long digit run in a corporate
        # corpus is a ticket, an invoice or a trade id. This is why PHONE is in
        # EGRESS_KINDS and not in INGEST_KINDS.
        re.compile(
            r"(?<![\w.])\+\d{1,3}[ .-]?(?:\(?\d{1,4}\)?[ .-]?){2,5}\d{2,4}(?![\w.])"
            r"|(?<![\w.])\(\d{3}\)[ .-]?\d{3}[ .-]\d{4}(?![\w.])"
            r"|(?<![\w.])\d{3}[ .-]\d{3}[ .-]\d{4}(?![\w.])"
        ),
    ),
    _Pattern(
        PiiKind.IP_ADDRESS,
        re.compile(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?![\w.])"),
        _valid_ip,
    ),
)


@runtime_checkable
class Detector(Protocol):
    """Anything that can find PII in a string."""

    name: str

    def detect(self, text: str, kinds: Iterable[PiiKind] | None = None) -> list[PiiMatch]:
        ...


class RegexDetector:
    """Pattern and checksum detector. No dependencies, no downloads, no model.

    Every numeric category is checksum-validated (Luhn for cards, mod-97 for
    IBANs, octet range for IP addresses, the Social Security Administration's
    structural rules for SSNs), because the difference between a detector and a
    nuisance is precisely whether it fires on invoice numbers.
    """

    name = "regex"

    def detect(self, text: str, kinds: Iterable[PiiKind] | None = None) -> list[PiiMatch]:
        """Find matches, longest-first, without overlaps.

        Args:
            text: The text to scan.
            kinds: Restrict to these categories. ``None`` means every category
                this detector supports.

        Returns:
            Matches sorted by position. Overlaps are resolved in favour of the
            longer match, so a card number inside a longer digit run is not also
            reported as a phone number.
        """
        wanted = set(kinds) if kinds is not None else None
        found: list[PiiMatch] = []
        for spec in _PATTERNS:
            if wanted is not None and spec.kind not in wanted:
                continue
            validate = spec.validate or _no_validation
            for match in spec.pattern.finditer(text):
                value = match.group(0)
                if not validate(value):
                    continue
                found.append(
                    PiiMatch(
                        kind=spec.kind,
                        start=match.start(),
                        end=match.end(),
                        text=value,
                        detector=self.name,
                    )
                )
        return _resolve_overlaps(found)


def _resolve_overlaps(matches: Sequence[PiiMatch]) -> list[PiiMatch]:
    """Longest match wins; ties broken by earlier start. Deterministic on purpose."""
    ordered = sorted(matches, key=lambda m: (-(m.end - m.start), m.start))
    kept: list[PiiMatch] = []
    for candidate in ordered:
        if any(candidate.start < k.end and k.start < candidate.end for k in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda m: m.start)


_PRESIDIO_MISSING = (
    "PresidioDetector needs presidio-analyzer and a spaCy model. It is not an "
    "extra in pyproject.toml on purpose: it is several hundred megabytes and the "
    "core promise is that the suite runs on pydantic, fastapi, numpy and httpx. "
    "Install it yourself with: pip install presidio-analyzer && "
    "python -m spacy download en_core_web_lg"
)

#: Presidio entity name -> our kind. Anything not in this map is ignored rather
#: than passed through, so a Presidio upgrade cannot silently widen what gets
#: redacted at ingest.
_PRESIDIO_KINDS = {
    "EMAIL_ADDRESS": PiiKind.EMAIL,
    "PHONE_NUMBER": PiiKind.PHONE,
    "US_SSN": PiiKind.SSN,
    "CREDIT_CARD": PiiKind.CREDIT_CARD,
    "IBAN_CODE": PiiKind.IBAN,
    "IP_ADDRESS": PiiKind.IP_ADDRESS,
    "PERSON": PiiKind.PERSON,
    "LOCATION": PiiKind.LOCATION,
}


class PresidioDetector:
    """Optional wrapper over ``presidio-analyzer``.

    Worth it for one thing the regexes cannot do at all: person and location
    names. Costs a spaCy pipeline per process and roughly two orders of
    magnitude more time per document than :class:`RegexDetector`, which on the
    reference machine is the difference between ingesting the corpus overnight
    and ingesting it over a weekend.

    Raises:
        ImportError: If Presidio is not installed. The message names the install.
    """

    name = "presidio"

    def __init__(self, *, language: str = "en", min_score: float = 0.5) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as exc:  # pragma: no cover - needs the optional dep
            raise ImportError(_PRESIDIO_MISSING) from exc
        self._engine = AnalyzerEngine()
        self._language = language
        self._min_score = min_score

    def detect(self, text: str, kinds: Iterable[PiiKind] | None = None) -> list[PiiMatch]:
        wanted = set(kinds) if kinds is not None else None
        results = self._engine.analyze(text=text, language=self._language)
        found: list[PiiMatch] = []
        for result in results:
            kind = _PRESIDIO_KINDS.get(result.entity_type)
            if kind is None or result.score < self._min_score:
                continue
            if wanted is not None and kind not in wanted:
                continue
            found.append(
                PiiMatch(
                    kind=kind,
                    start=result.start,
                    end=result.end,
                    text=text[result.start : result.end],
                    detector=self.name,
                )
            )
        return _resolve_overlaps(found)


class CompositeDetector:
    """Union of several detectors, overlaps resolved once at the end.

    Union, never intersection. Two detectors disagreeing about whether a string
    is a card number should end with it redacted, because the cost of a false
    positive is a hole in a sentence and the cost of a false negative is a card
    number in a vector database.
    """

    name = "composite"

    def __init__(self, *detectors: Detector) -> None:
        if not detectors:
            raise ValueError("CompositeDetector needs at least one detector")
        self.detectors = detectors

    def detect(self, text: str, kinds: Iterable[PiiKind] | None = None) -> list[PiiMatch]:
        found: list[PiiMatch] = []
        for detector in self.detectors:
            found.extend(detector.detect(text, kinds))
        return _resolve_overlaps(found)


_DEFAULT_DETECTOR = RegexDetector()


def load_detector(*, use_presidio: bool = False) -> Detector:
    """Build the detector for this process.

    Args:
        use_presidio: Add Presidio on top of the regexes. Off by default so CI
            downloads nothing.

    Returns:
        A :class:`Detector`. Always includes :class:`RegexDetector`, because the
        checksum-validated categories are the ones that matter at ingest and a
        model is not more reliable at arithmetic.
    """
    if not use_presidio:
        return _DEFAULT_DETECTOR
    return CompositeDetector(_DEFAULT_DETECTOR, PresidioDetector())


def detect(
    text: str,
    kinds: Iterable[PiiKind] | None = None,
    *,
    detector: Detector | None = None,
) -> list[PiiMatch]:
    """Find PII without changing anything. Matches carry raw values; do not log them."""
    return (detector or _DEFAULT_DETECTOR).detect(text, kinds)


def redact(
    text: str,
    kinds: Iterable[PiiKind] | None = None,
    *,
    detector: Detector | None = None,
    salt: str | None = None,
) -> Redaction:
    """Replace every match with a stable one-way placeholder.

    Replacement runs back-to-front so earlier offsets stay valid, which is the
    kind of detail that is obvious until someone refactors it into a forward
    loop and every second redaction lands one character off.

    Args:
        text: Text to redact.
        kinds: Categories to remove. Use :data:`INGEST_KINDS` or
            :data:`EGRESS_KINDS` rather than inventing a set at the call site.
        detector: Override the default regex detector.
        salt: Fingerprint salt. See :data:`FINGERPRINT_SALT_ENV`.

    Returns:
        A :class:`Redaction`. ``matches`` holds the raw values for audit; only
        ``counts()`` is safe to emit.
    """
    engine = detector or _DEFAULT_DETECTOR
    matches = engine.detect(text, kinds)
    redacted = text
    for match in sorted(matches, key=lambda m: m.start, reverse=True):
        redacted = redacted[: match.start] + match.placeholder(salt) + redacted[match.end :]
    return Redaction(text=redacted, matches=tuple(matches), original_chars=len(text))


def redact_for_ingest(text: str, *, detector: Detector | None = None) -> Redaction:
    """Destructive redaction, applied before chunking and embedding.

    Only :data:`INGEST_KINDS`. What is removed here cannot be retrieved by
    anybody, including the people who are allowed to see it, so the set stays
    small on purpose. If you are about to widen it because "not everyone should
    see that", write a tuple instead.
    """
    return redact(text, INGEST_KINDS, detector=detector)


def redact_for_egress(text: str, *, detector: Detector | None = None) -> Redaction:
    """Non-destructive redaction, applied to generated text before it is returned.

    The original is still in the index, so this costs the permitted reader
    nothing they cannot get by opening the document.
    """
    return redact(text, EGRESS_KINDS, detector=detector)


def redact_for_log(text: str, *, max_chars: int = 200, detector: Detector | None = None) -> str:
    """Redact and truncate anything about to cross into a lower-trust store.

    Logs, traces, error payloads and refusal bodies. Two controls, because PII
    is not the only thing that should not be in a log line: the truncation caps
    how much document text can escape through a debugging surface even when the
    detector misses everything. This is the direct answer to planted mutant M12.
    """
    cleaned = redact(text, EGRESS_KINDS, detector=detector).text
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "…"


def redact_chunks(
    chunks: Iterable[object], *, detector: Detector | None = None
) -> tuple[list[str], dict[str, int]]:
    """Redact a batch of chunk texts at ingest, returning texts and total counts.

    Takes anything with a ``text`` attribute so that it works on ``Chunk`` and
    on whatever the ingest pipeline is holding mid-transform.
    """
    texts: list[str] = []
    totals: dict[str, int] = {}
    for chunk in chunks:
        result = redact_for_ingest(str(getattr(chunk, "text", chunk)), detector=detector)
        texts.append(result.text)
        for kind, count in result.counts().items():
            totals[kind] = totals.get(kind, 0) + count
    return texts, totals
