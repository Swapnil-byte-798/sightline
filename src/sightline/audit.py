"""Append-only, hash-chained audit log.

Each row commits to its predecessor: ``digest = sha256(prev_digest ||
canonical_json(row))``. Change one field of one historical row and every digest
after it stops matching, so tampering is *detectable* by anyone who can read the
file. Be precise about what that does and does not buy:

* It detects edits, reordering and mid-file truncation.
* It does **not** prevent them, and it does not detect truncation of the tail —
  lopping off the last N rows leaves a perfectly valid chain. Closing that needs
  an external anchor: periodically publishing the head digest somewhere the
  process running this code cannot rewrite. That anchor is **not built**, and
  saying so is better than implying an integrity property this file does not
  have.

**What is recorded, and the one field pair that matters.** The schema is
``(ts, principal, query_hash, strategy, epoch, n_candidates,
n_dropped_at_recheck, refusal_reason, retrieved_objects, shown_objects, ...)``.
``retrieved`` versus ``shown`` is the security-interesting part: the difference
between them is exactly what the live permission recheck threw away. A log that
records only what was shown cannot answer "was the index stale for this user" or
"did this request touch a document it should not have retrieved", which is the
question an incident review opens with.

**Query text is hashed, never stored** (SR-12). An audit log of everyone's
questions is a new liability created out of nothing. The hash is keyed (HMAC),
because a plain SHA-256 of a short question is trivially reversed by hashing a
candidate list — the same reason you do not store unsalted password hashes.
Consequence, stated rather than hidden: hashes are not comparable across
deployments, and rotating the key breaks correlation with older rows.

**Everything is redacted on the way in.** Audit logs, like traces, land in
tooling with a weaker access model than the document store. Planted mutant M12 is
"chunk text appears in an error or refusal payload"; this file is the other half
of that surface, and it never stores chunk text at all — only object references,
which are already names, not contents.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

from sightline.errors import AuditChainBroken, AuditUnavailable

__all__ = [
    "GENESIS_DIGEST",
    "AuditRecord",
    "AuditRow",
    "ChainVerification",
    "AuditSink",
    "MemoryAuditLog",
    "JsonlAuditLog",
    "canonical_json",
    "chain_digest",
    "verify_chain",
    "hash_query",
    "redact",
    "redact_for_egress",
    "REDACTIONS",
]

#: The zero digest the first row commits to. Not an empty string: an empty
#: previous digest and a missing previous digest hash identically, and a
#: distinguishable genesis makes "someone deleted row 1" visible.
GENESIS_DIGEST = "0" * 64


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------

def _luhn(digits: str) -> bool:
    """Luhn checksum. Without it, every long number is a 'card number'."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = ord(ch) - 48
        if i % 2:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _card(match: re.Match[str]) -> str:
    digits = re.sub(r"[^0-9]", "", match.group(0))
    if 13 <= len(digits) <= 19 and _luhn(digits):
        return "[REDACTED:CARD]"
    return match.group(0)


#: ``(name, pattern, replacement)``. Order matters: the credential patterns run
#: before the generic ones so an API key containing an ``@`` is not mistaken for
#: an email address.
#:
#: This is a pattern matcher and is described as one. It has false positives (a
#: support ticket quoting a fake card number) and false negatives (a key format
#: invented after this list was written). Claiming a detection rate for it would
#: be dishonest, so none is claimed. It is hygiene on a surface that should not
#: have carried the data in the first place, not a control.
REDACTIONS: tuple[tuple[str, re.Pattern[str], Any], ...] = (
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
        "[REDACTED:PRIVATE_KEY]",
    ),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED:AWS_KEY]"),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"), "[REDACTED:TOKEN]"),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "[REDACTED:TOKEN]"),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}"), "[REDACTED:TOKEN]"),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"),
        "[REDACTED:JWT]",
    ),
    ("card", re.compile(r"\b(?:\d[ -]*?){13,19}\b"), _card),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED:SSN]"),
    (
        "iban",
        re.compile(r"\b[A-Z]{2}\d{2}[ ]?(?:[A-Z0-9]{4}[ ]?){2,7}[A-Z0-9]{1,4}\b"),
        "[REDACTED:IBAN]",
    ),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"), "[REDACTED:EMAIL]"),
    (
        "phone",
        re.compile(r"(?<![\w.])\+?\d[\d ().-]{8,16}\d(?![\w.])"),
        "[REDACTED:PHONE]",
    ),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED:IP]"),
)

#: The narrow set: material that should not be in an AI pipeline under *any*
#: permission. See the table in ``docs/guide/07``: redaction is hygiene,
#: permission is the control. Stripping a salary figure before generation would
#: break the answer for the person in HR who is entirely allowed to have it, so
#: this set stops at live credentials and card numbers.
_EGRESS_KINDS = frozenset({"private_key", "aws_key", "github_token", "openai_key", "bearer",
                           "jwt", "card"})


def redact(text: str, *, kinds: Iterable[str] | None = None) -> str:
    """Strip personally identifying and credential-shaped substrings.

    Used on anything that crosses into a lower-trust store: the audit log, log
    lines, span attributes, error detail.

    Args:
        text: Free text. Non-strings are returned unchanged by the callers.
        kinds: Restrict to these redaction names. ``None`` runs all of them.

    Returns:
        The text with matches replaced by ``[REDACTED:KIND]`` markers. The
        markers are deliberately visible: a silently shortened log line makes
        the reader distrust the whole file.
    """
    if not text:
        return text
    wanted = None if kinds is None else frozenset(kinds)
    out = text
    for name, pattern, repl in REDACTIONS:
        if wanted is not None and name not in wanted:
            continue
        out = pattern.sub(repl, out)
    return out


def redact_for_egress(text: str) -> str:
    """Strip only what must never leave the process, for the LLM provider path.

    Retrieved document text goes to a third-party model. Running the full
    :func:`redact` over it would delete the answer for a reader who is permitted
    to have it — redaction cannot be selective by reader, which is exactly why it
    is not an access control. So this pass removes credentials and card numbers
    only: the things that have no business in any prompt regardless of who is
    asking.
    """
    return redact(text, kinds=_EGRESS_KINDS)


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def canonical_json(obj: Mapping[str, Any]) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace, escaped non-ASCII.

    Determinism is the whole security property. Two processes that serialise the
    same row differently produce different digests and a chain that looks broken,
    so nothing here may depend on dict order, locale, or float formatting.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")


def chain_digest(prev_digest: str, row: Mapping[str, Any]) -> str:
    """``sha256(prev_digest || 0x0a || canonical_json(row))``, hex.

    The separator matters: without a delimiter, ``prev="ab", row=...`` and
    ``prev="a", row=b...`` could hash identically. It is a one-byte fix for a
    length-extension-shaped ambiguity, and it costs nothing.
    """
    h = hashlib.sha256()
    h.update(prev_digest.encode("ascii"))
    h.update(b"\n")
    h.update(canonical_json(row))
    return h.hexdigest()


def hash_query(text: str, key: bytes | str = b"") -> str:
    """Keyed hash of a question. The question itself is never stored.

    An unkeyed hash of "what is our parental leave policy" is recoverable by
    anyone willing to hash a list of plausible questions, which defeats the point
    of not storing the text. With a deployment key it is not, at the cost of
    hashes being incomparable across deployments — the right trade, since
    cross-deployment correlation of employees' questions is not a feature anybody
    asked for.
    """
    material = key.encode("utf-8") if isinstance(key, str) else key
    normalised = " ".join(text.split()).lower().encode("utf-8")
    if not material:
        # Unkeyed fallback, domain-separated so it cannot be confused with a
        # keyed digest. Development only; production sets SIGHTLINE_AUDIT_HMAC_KEY.
        return "u:" + hashlib.sha256(b"sightline/query/v1\n" + normalised).hexdigest()[:32]
    return "k:" + hmac.new(material, normalised, hashlib.sha256).hexdigest()[:32]


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One served request, as the log will remember it.

    Note what is absent: the question, the answer, and any chunk text. What is
    present is enough to reconstruct *which policy state served whom, by which
    strategy, and what the live check rejected* — and not enough to reconstruct
    what anybody asked.
    """

    principal: str
    query_hash: str
    strategy: str
    epoch: int
    n_candidates: int = 0
    n_dropped_at_recheck: int = 0
    #: Object references the index offered, before the live recheck.
    retrieved_objects: tuple[str, ...] = ()
    #: Object references that were actually cited to the user. The set difference
    #: against ``retrieved_objects`` is the stale-index gap for this request.
    shown_objects: tuple[str, ...] = ()
    guardrails_fired: tuple[str, ...] = ()
    refusal_reason: str | None = None
    provider: str | None = None
    degraded: bool = False
    request_id: str = ""
    route: str = "/v1/query"
    ts: float = field(default_factory=time.time)

    def to_row(self) -> dict[str, Any]:
        """The canonical dict that the digest commits to.

        Explicit and sorted rather than ``asdict()``: the chain's meaning depends
        on the exact field set, so adding a field must be a deliberate edit here,
        not a side effect of adding an attribute above.
        """
        return {
            "ts": round(self.ts, 6),
            "route": self.route,
            "request_id": self.request_id,
            "principal": self.principal,
            "query_hash": self.query_hash,
            "strategy": self.strategy,
            "epoch": self.epoch,
            "n_candidates": self.n_candidates,
            "n_dropped_at_recheck": self.n_dropped_at_recheck,
            "retrieved_objects": list(self.retrieved_objects),
            "shown_objects": list(self.shown_objects),
            "guardrails_fired": list(self.guardrails_fired),
            "refusal_reason": self.refusal_reason,
            "provider": self.provider,
            "degraded": self.degraded,
        }

    def redacted(self) -> "AuditRecord":
        """Apply redaction before the row is written.

        Object references and principals are identifiers, not free text, but an
        identifier derived from an email address is still an email address —
        which is exactly the shape the Enron corpus produces. So they go through
        the pass too.
        """
        return replace(
            self,
            principal=redact(self.principal),
            retrieved_objects=tuple(redact(o) for o in self.retrieved_objects),
            shown_objects=tuple(redact(o) for o in self.shown_objects),
            guardrails_fired=tuple(redact(g) for g in self.guardrails_fired),
        )


@dataclass(frozen=True, slots=True)
class AuditRow:
    """A record as stored: sequence number, previous digest, own digest."""

    seq: int
    prev: str
    digest: str
    row: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {"seq": self.seq, "prev": self.prev, "digest": self.digest, "row": self.row},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @classmethod
    def from_json(cls, line: str) -> "AuditRow":
        blob = json.loads(line)
        return cls(
            seq=int(blob["seq"]),
            prev=str(blob["prev"]),
            digest=str(blob["digest"]),
            row=dict(blob["row"]),
        )


@dataclass(frozen=True, slots=True)
class ChainVerification:
    """The result of walking the chain. ``ok`` is the only thing to branch on."""

    ok: bool
    checked: int
    head: str = GENESIS_DIGEST
    broken_at: int | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": self.checked,
            "head": self.head,
            "broken_at": self.broken_at,
            "reason": self.reason,
        }


def verify_chain(rows: Sequence[AuditRow], *, start_digest: str = GENESIS_DIGEST,
                 start_seq: int = 0) -> ChainVerification:
    """Recompute every digest and check the links.

    Args:
        rows: Rows in write order.
        start_digest: Digest the first row must commit to. Pass the head of an
            earlier segment to verify a page in the middle of a long log.
        start_seq: Sequence number the first row must carry, for the same reason.

    Returns:
        A :class:`ChainVerification`. It reports rather than raises, because the
        caller — ``GET /v1/audit`` — wants to *show* the break, and an exception
        would hand it a stack trace instead of a row number.
    """
    prev = start_digest
    expect = start_seq
    for row in rows:
        if row.seq != expect:
            return ChainVerification(False, expect - start_seq, prev, row.seq,
                                     f"sequence jumped: expected {expect}, found {row.seq}")
        if row.prev != prev:
            return ChainVerification(False, expect - start_seq, prev, row.seq,
                                     "row does not commit to its predecessor")
        recomputed = chain_digest(prev, row.row)
        if recomputed != row.digest:
            return ChainVerification(False, expect - start_seq, prev, row.seq,
                                     "row content does not match its digest")
        prev = row.digest
        expect += 1
    return ChainVerification(True, expect - start_seq, prev)


# --------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------


@runtime_checkable
class AuditSink(Protocol):
    """Where rows go. Append and read; there is no update and no delete.

    The absence of a delete method is the interface doing its job. A sink that
    can rewrite history is not an audit log, whatever the file is called.
    """

    def append(self, record: AuditRecord) -> AuditRow:
        ...

    def read(self, *, limit: int = 100, after_seq: int = -1) -> list[AuditRow]:
        ...

    def verify(self) -> ChainVerification:
        ...

    @property
    def head(self) -> str:
        ...

    def __len__(self) -> int:
        ...


class MemoryAuditLog:
    """In-process chain. The default, and what the test suite uses.

    Bounded by ``max_rows`` so a long-running demo does not eat the laptop. Note
    the consequence and do not pretend otherwise: dropping the oldest rows means
    a full verification from genesis is no longer possible, so the drop point is
    recorded and :meth:`verify` checks from there.
    """

    def __init__(self, max_rows: int = 100_000) -> None:
        self._lock = threading.Lock()
        self._rows: list[AuditRow] = []
        self._head = GENESIS_DIGEST
        self._next_seq = 0
        self._max_rows = max_rows
        self._dropped_to_digest = GENESIS_DIGEST
        self._dropped_to_seq = 0

    def append(self, record: AuditRecord) -> AuditRow:
        row_dict = record.redacted().to_row()
        with self._lock:
            digest = chain_digest(self._head, row_dict)
            row = AuditRow(self._next_seq, self._head, digest, row_dict)
            self._rows.append(row)
            self._head = digest
            self._next_seq += 1
            if len(self._rows) > self._max_rows:
                dropped = self._rows.pop(0)
                self._dropped_to_digest = dropped.digest
                self._dropped_to_seq = dropped.seq + 1
            return row

    def read(self, *, limit: int = 100, after_seq: int = -1) -> list[AuditRow]:
        with self._lock:
            rows = [r for r in self._rows if r.seq > after_seq]
        return rows[:limit]

    def verify(self) -> ChainVerification:
        with self._lock:
            rows = list(self._rows)
            start, seq = self._dropped_to_digest, self._dropped_to_seq
        return verify_chain(rows, start_digest=start, start_seq=seq)

    @property
    def head(self) -> str:
        with self._lock:
            return self._head

    def __len__(self) -> int:
        with self._lock:
            return self._next_seq


class JsonlAuditLog:
    """One JSON object per line, opened append-only.

    ``O_APPEND`` means concurrent writers cannot interleave a partial line, and a
    line shorter than the OS write granularity lands atomically. That is a
    property of the platform, not of this code, which is why ``fsync`` is a
    setting: without it a power cut can lose the tail, and losing the tail is the
    truncation case the chain cannot detect.
    """

    def __init__(self, path: str | Path, *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AuditUnavailable(f"cannot create audit directory {self.path.parent}") from exc
        self._head, self._next_seq = self._read_tail()

    def _read_tail(self) -> tuple[str, int]:
        """Resume the chain from an existing file.

        A trailing partial line (the power-cut case) is refused rather than
        silently skipped: continuing a chain past a line nobody can verify makes
        every later digest meaningless.
        """
        if not self.path.exists():
            return GENESIS_DIGEST, 0
        last: AuditRow | None = None
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    last = AuditRow.from_json(line)
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    raise AuditChainBroken(
                        lineno, f"unparseable audit row at line {lineno} of {self.path}"
                    ) from exc
        if last is None:
            return GENESIS_DIGEST, 0
        return last.digest, last.seq + 1

    def append(self, record: AuditRecord) -> AuditRow:
        row_dict = record.redacted().to_row()
        with self._lock:
            digest = chain_digest(self._head, row_dict)
            row = AuditRow(self._next_seq, self._head, digest, row_dict)
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(fd, (row.to_json() + "\n").encode("utf-8"))
                    if self.fsync:
                        os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError as exc:
                raise AuditUnavailable(f"audit append failed: {exc}") from exc
            self._head = digest
            self._next_seq += 1
            return row

    def _iter_rows(self) -> Iterator[AuditRow]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield AuditRow.from_json(line)

    def read(self, *, limit: int = 100, after_seq: int = -1) -> list[AuditRow]:
        out: list[AuditRow] = []
        for row in self._iter_rows():
            if row.seq > after_seq:
                out.append(row)
                if len(out) >= limit:
                    break
        return out

    def verify(self) -> ChainVerification:
        return verify_chain(list(self._iter_rows()))

    @property
    def head(self) -> str:
        with self._lock:
            return self._head

    def __len__(self) -> int:
        with self._lock:
            return self._next_seq
