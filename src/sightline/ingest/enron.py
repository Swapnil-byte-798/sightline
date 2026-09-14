"""The Enron connector: real messages, and permissions derived from real headers.

WHY THIS CORPUS
---------------
Every permission demo I have read assigns ACLs at random. A random ACL is a
coin-flip dressed as an org chart: it has no group structure, no nesting, no
correlation between who can see a document and what the document says, and so a
filtered-search benchmark built on it measures nothing except how fast a bitmap
intersects.

The Enron corpus has access structure that nobody invented. Each of the ~150
mailboxes is a real principal. A message in Rick Buy's mailbox is a message Rick
Buy could actually read, in 2001, because it was in his mail. The To and Cc lines
are real distribution decisions made by real people under deadline. The internal
distribution lists (``all.employees@enron.com`` and friends) are real groups with
real, observable membership. That is the only reason this corpus is here. The
email content is incidental.

WHAT THE DERIVATION CLAIMS, AND WHAT IT DOES NOT
------------------------------------------------
The tuples this module emits **are** the ground truth for the harness: every
number in the repo is computed against them, and ``authz.check`` is the authority
over them. What is *not* claimed is that they reconstruct Enron's actual 2001
access-control lists. Nobody can check that, and pretending otherwise would be
the same sin as random ACLs with better marketing.

The derivation, and where each step is a simplification:

1. **A mailbox owner is a viewer of every message in their mailbox.** This one
   is solid — the message was in their mail.
2. **The sender is a viewer.** Also solid.
3. **To and Cc recipients are viewers.** Solid for the addresses that resolve to
   a person; an address with no mailbox in the corpus becomes a principal with
   no other tuples, which is correct but unexercised.
4. **Bcc is not modelled.** The maildir headers rarely carry it, and inferring it
   from mailbox presence would manufacture edges. Named, not built.
5. **Distribution-list addresses become groups**, with membership inferred from
   which mailbox owners hold a copy of a message sent to that list. That is an
   observation, not a guess — the copy is in their maildir. It *under*-counts:
   only ~150 of Enron's ~20,000 employees have a mailbox here, so every derived
   DL is a small sample of the real one.
6. **Recipient sets that recur become groups.** If the same set of four people
   is addressed together on twenty messages, real life almost certainly had a
   list or a team behind it, and granting by that set is what the org was doing
   whether or not it had a name. This is the weakest inferential step and it is
   here for a structural reason: the compiler needs nested groups to exist, and
   fabricating a group hierarchy would put the thing under test into the fixture.
7. **Groups nest by set containment.** A derived group whose members are a strict
   subset of another's is attached as a member userset of the larger. Real orgs
   nest by reporting line; this nests by co-occurrence. It produces genuine
   multi-hop usersets to compile through, which is what the harness needs, and it
   is not a claim about Enron's org chart.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
* **Thread reconstruction.** ``In-Reply-To`` chains would let a reply inherit a
  parent's ACL via ``TupleToUserset``. Tempting, and wrong: a reply routinely
  drops recipients, and inheriting the parent's viewers would silently
  *broaden* access. The one place this module could have invented a leak.
* **Attachment extraction.** Nothing in the corpus needs it.
* **Folder objects.** ``doc#parent@folder`` inheritance is exercised by the
  synthetic fixtures, where the hierarchy is known, not inferred.
"""

from __future__ import annotations

import email
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from email import policy
from email.message import EmailMessage
from email.utils import getaddresses
from pathlib import Path

from sightline.types import ObjectRef, PrincipalRef, Tuple_

__all__ = [
    "DEFAULT_MAX_MESSAGES",
    "MIN_GROUP_SIZE",
    "MIN_GROUP_SUPPORT",
    "TARGET_CHUNKS",
    "VIEWER",
    "AclDerivation",
    "DerivationConfig",
    "EnronMessage",
    "dedupe_messages",
    "derive_acl",
    "document_text",
    "is_distribution_list",
    "iter_enronqa",
    "iter_enronqa_questions",
    "iter_maildir",
    "normalise_address",
    "principal_for_address",
    "quote_boundaries",
    "sample_messages",
]

#: Corpus size target. Sized to fit a free Qdrant tier and a 6-hour CI job, not
#: to be impressive. 70k chunks at 384 dims is ~107 MB of float32, which fits in
#: 8 GB alongside everything else; a bigger corpus would change no conclusion in
#: this repo and would stop the benchmark being reproducible by a reader.
TARGET_CHUNKS = 70_000

#: Messages to read before stopping. Enron bodies average a little under two
#: 256-token chunks each after quoting is split off, so this lands near the
#: chunk target. It is an estimate and the pipeline stops on the *chunk* count,
#: which is the number that actually matters.
DEFAULT_MAX_MESSAGES = 40_000

#: Smallest recipient set that may become a group. Two people is a conversation.
MIN_GROUP_SIZE = 3

#: How many messages a recipient set must recur on before it is called a group.
#: Below this it is a coincidence, and a group per coincidence would give the
#: compiler tens of thousands of one-message usersets to walk.
MIN_GROUP_SUPPORT = 4

#: Only one filtered relation exists in v1.
VIEWER = "viewer"

_ENRON_DOMAINS = frozenset({"enron.com", "enron.net", "ect.enron.com", "enron.co.uk"})

# Under-detects on purpose. A false positive turns a person into a group, which
# invents membership; a false negative leaves a list as an individual principal,
# which merely loses a group. Losing structure is the cheaper error.
_DL_LOCAL_RE = re.compile(
    r"""^(
        all[._-].*        | # all.employees, all-houston
        .*[._-]dl         | # something_dl
        dl[._-].*         | # dl.traders
        .*employees.*     |
        .*announcements?.*|
        .*distribution.*  |
        .*[._-]team       |
        everyone          |
        staff             |
        undisclosed[._-]recipients
    )$""",
    re.IGNORECASE | re.VERBOSE,
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Where a quoted reply or forward starts. Chunks never straddle one of these, so
# a forwarded thread does not smear four messages into one embedding.
_QUOTE_MARKERS = (
    "-----Original Message-----",
    "---------------------- Forwarded by",
    "----- Forwarded by",
    "-----Forwarded by",
    "\n> ",
)

_BODY_KEYS = ("email", "body", "text", "message", "content", "email_body", "document")
_MAILBOX_KEYS = ("user", "mailbox", "owner", "inbox", "username")
_PATH_KEYS = ("file", "path", "file_name", "filename", "email_file", "id")
_QUESTION_KEYS = ("questions", "question", "queries", "query")
_ANSWER_KEYS = ("answers", "answer", "gold_answers", "responses")


# --------------------------------------------------------------------------
# Principals
# --------------------------------------------------------------------------


def _slug(raw: str) -> str:
    return _SLUG_RE.sub("-", raw.strip().lower()).strip("-")


def normalise_address(raw: str) -> str:
    """Lowercase, de-quote and strip an address down to ``local@domain``.

    The corpus contains the same human as ``Jeff.Skilling@enron.com``,
    ``jeff.skilling@enron.com`` and ``"Skilling, Jeff" <jskilling@enron.com>``.
    Case folding merges the first two. The third is a genuinely different
    address and is left alone: merging aliases would require an identity
    resolver, and a wrong identity resolver silently merges two people's
    permissions, which is the exact failure this whole repo exists to prevent.
    """
    addr = raw.strip().strip("<>").strip()
    if "<" in addr and ">" in addr:
        addr = addr[addr.rindex("<") + 1 : addr.rindex(">")]
    addr = addr.strip().strip("'\"").lower()
    # Enron's exporters emit these three spellings of the same separator.
    addr = addr.replace("..", ".").replace(" ", "")
    return addr


def is_distribution_list(address: str) -> bool:
    """Heuristic: does this address name a list rather than a person?

    See ``_DL_LOCAL_RE``. Deliberately conservative; see the module docstring for
    which direction the errors point.
    """
    local, _, domain = address.partition("@")
    if not local:
        return False
    if domain and domain not in _ENRON_DOMAINS and "enron" not in domain:
        return False
    return _DL_LOCAL_RE.match(local) is not None


def principal_for_address(address: str, *, registry: dict[str, str] | None = None) -> PrincipalRef:
    """Map an email address to a stable principal.

    Internal addresses become ``user:<local-part>``; external ones keep their
    domain (``user:alice-at-example-com``) so that an outside counterparty is
    never confused with an employee of the same first name. Distribution lists
    become ``group:dl_<slug>#member``.

    Args:
        address: Already normalised by :func:`normalise_address`.
        registry: Optional slug -> address map used to detect collisions. Two
            different addresses that slugify to the same string get a hash
            suffix on the second. Without this, ``a.b@x`` and ``a-b@x`` would
            become one principal and quietly merge two people's access — a
            close cousin of planted mutant M9.

    Returns:
        A :class:`PrincipalRef`.
    """
    local, _, domain = address.partition("@")
    if is_distribution_list(address):
        return PrincipalRef("group", f"dl_{_slug(local)}", "member")
    base = _slug(local) if domain in _ENRON_DOMAINS else _slug(address.replace("@", "-at-"))
    if not base:
        base = hashlib.blake2b(address.encode(), digest_size=6).hexdigest()
    if registry is not None:
        seen = registry.get(base)
        if seen is None:
            registry[base] = address
        elif seen != address:
            suffix = hashlib.blake2b(address.encode(), digest_size=3).hexdigest()
            base = f"{base}-{suffix}"
            registry.setdefault(base, address)
    return PrincipalRef("user", base)


def mailbox_principal(mailbox: str) -> PrincipalRef:
    """The owner of a maildir directory. ``maildir/skilling-j`` -> ``user:skilling-j``.

    Mailbox slugs and address local parts do not agree, so ``user:skilling-j``
    and ``user:jeff.skilling`` are two principals for one human. That is left
    uncorrected on purpose: it costs recall on a query asked as one of them, it
    never grants either of them anything the other has, and the alternative is an
    identity-resolution heuristic that fails in the granting direction.
    """
    return PrincipalRef("user", _slug(mailbox))


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnronMessage:
    """One message, possibly present in several mailboxes.

    The same message sits in the sender's ``sent`` folder and every recipient's
    ``inbox``, so identity is the ``Message-ID``, not the file path. Deduplicating
    on it does two things: it stops the index holding six copies of the same
    text, and it makes each copy's mailbox owner a *viewer of one document*
    rather than an owner of a private near-duplicate. The second is the reason
    it matters — it is where the corpus's multi-principal ACLs come from.
    """

    message_id: str
    subject: str
    sender: str
    to: tuple[str, ...]
    cc: tuple[str, ...]
    date: str
    body: str
    mailboxes: tuple[str, ...] = ()
    source_paths: tuple[str, ...] = ()

    @property
    def object(self) -> ObjectRef:
        """``doc:<16 hex>`` from the message id. Stable across ingest runs."""
        digest = hashlib.blake2b(self.message_id.encode("utf-8"), digest_size=8).hexdigest()
        return ObjectRef("doc", digest)

    @property
    def recipients(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for a in (*self.to, *self.cc):
            seen.setdefault(a, None)
        return tuple(seen)


def _addresses(msg: EmailMessage, header: str) -> tuple[str, ...]:
    raw = msg.get_all(header, [])
    out: list[str] = []
    seen: set[str] = set()
    for _, addr in getaddresses([str(r) for r in raw]):
        norm = normalise_address(addr)
        if "@" not in norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return tuple(out)


def _body_of(msg: EmailMessage) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    return str(part.get_content())
                except (LookupError, UnicodeDecodeError):
                    continue
        return ""
    try:
        return str(msg.get_content())
    except (LookupError, UnicodeDecodeError):
        payload = msg.get_payload(decode=True)
        return payload.decode("latin-1", "replace") if isinstance(payload, bytes) else ""


def _parse_rfc822(raw: str, *, mailbox: str, path: str) -> EnronMessage | None:
    try:
        msg = email.message_from_string(raw, policy=policy.default)
    except Exception:  # pragma: no cover - malformed source files
        return None
    if not isinstance(msg, EmailMessage):  # pragma: no cover - policy guarantees it
        return None
    body = _body_of(msg).strip()
    if not body:
        return None
    sender_list = _addresses(msg, "From")
    sender = sender_list[0] if sender_list else ""
    mid = str(msg.get("Message-ID", "")).strip()
    if not mid:
        # No Message-ID: fall back to content addressing so the same message
        # found in two mailboxes still deduplicates.
        mid = "sha:" + hashlib.blake2b(
            f"{sender}|{msg.get('Date', '')}|{body}".encode(), digest_size=16
        ).hexdigest()
    return EnronMessage(
        message_id=mid,
        subject=str(msg.get("Subject", "")).strip(),
        sender=sender,
        to=_addresses(msg, "To"),
        cc=_addresses(msg, "Cc"),
        date=str(msg.get("Date", "")).strip(),
        body=body,
        mailboxes=(mailbox,) if mailbox else (),
        source_paths=(path,) if path else (),
    )


def iter_maildir(root: Path | str, *, limit: int | None = None) -> Iterator[EnronMessage]:
    """Walk the raw CMU maildir, yielding one message per file.

    Args:
        root: The directory containing the per-person mailboxes (usually named
            ``maildir``). Either the ``maildir`` directory itself or its parent.
        limit: Stop after this many parsed messages.

    Yields:
        :class:`EnronMessage`, one per file, with ``mailboxes`` holding the
        single owner that file belonged to. Duplicates across mailboxes are
        merged later by :func:`dedupe_messages`, not here, because merging needs
        the whole stream and this needs to stay O(1) in memory.

    Note:
        Files are walked in sorted order. An unsorted ``os.walk`` would make the
        corpus depend on filesystem inode ordering, and then a truncated run
        would ingest a different 40,000 messages on every machine — which is
        enough to make a published recall number irreproducible.
    """
    base = Path(root)
    if (base / "maildir").is_dir():
        base = base / "maildir"
    if not base.is_dir():
        raise FileNotFoundError(f"no maildir at {base}")
    count = 0
    for mailbox_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        mailbox = mailbox_dir.name
        for path in sorted(mailbox_dir.rglob("*")):
            if not path.is_file():
                continue
            try:
                raw = path.read_text("utf-8", errors="replace")
            except OSError:  # pragma: no cover - unreadable file
                continue
            msg = _parse_rfc822(raw, mailbox=mailbox, path=str(path.relative_to(base)))
            if msg is None:
                continue
            yield msg
            count += 1
            if limit is not None and count >= limit:
                return


def _first_key(record: dict[str, object], keys: Sequence[str]) -> object | None:
    for k in keys:
        if k in record and record[k] not in (None, "", []):
            return record[k]
    return None


def iter_enronqa(path: Path | str, *, limit: int | None = None) -> Iterator[EnronMessage]:
    """Read the EnronQA export (``MichaelR207/enron_qa_0922``) as messages.

    The dataset ships one record per email with the raw RFC822 text and the
    mailbox it came from, plus question/answer pairs used by ``eval/``. Field
    names have moved between revisions, so the keys are probed rather than
    hardcoded and an unrecognised schema raises with the keys it actually saw.
    Silently yielding nothing from a file that parsed fine is the failure mode
    that costs an afternoon.

    Args:
        path: A ``.jsonl`` file, or a directory of them.
        limit: Stop after this many messages.

    Yields:
        :class:`EnronMessage`.
    """
    count = 0
    for record in _iter_jsonl(path):
        raw = _first_key(record, _BODY_KEYS)
        if raw is None:
            raise ValueError(
                "no email body field in EnronQA record; looked for "
                f"{_BODY_KEYS}, record has {sorted(record)}"
            )
        mailbox = str(_first_key(record, _MAILBOX_KEYS) or "")
        src = str(_first_key(record, _PATH_KEYS) or "")
        text = str(raw)
        msg = _parse_rfc822(text, mailbox=_slug(mailbox), path=src)
        if msg is None:
            # Some revisions store the body already stripped of headers. Keep it
            # as a document rather than dropping it; the mailbox owner is still a
            # real principal and that is the part this repo cares about.
            body = text.strip()
            if not body:
                continue
            mid = "sha:" + hashlib.blake2b(body.encode("utf-8"), digest_size=16).hexdigest()
            msg = EnronMessage(
                message_id=mid,
                subject=str(record.get("subject", "") or ""),
                sender="",
                to=(),
                cc=(),
                date="",
                body=body,
                mailboxes=(_slug(mailbox),) if mailbox else (),
                source_paths=(src,) if src else (),
            )
        yield msg
        count += 1
        if limit is not None and count >= limit:
            return


def iter_enronqa_questions(path: Path | str) -> Iterator[tuple[str, str, str]]:
    """Yield ``(mailbox, question, answer)`` from an EnronQA export.

    The eval harness asks these *as the mailbox owner*, which is the whole point
    of using this dataset: every question has a principal attached who provably
    could read the answer, so a filtered-retrieval result is checkable rather
    than plausible.
    """
    for record in _iter_jsonl(path):
        mailbox = _slug(str(_first_key(record, _MAILBOX_KEYS) or ""))
        questions = _first_key(record, _QUESTION_KEYS)
        answers = _first_key(record, _ANSWER_KEYS)
        qs = questions if isinstance(questions, list) else [questions]
        ans = answers if isinstance(answers, list) else [answers]
        for i, q in enumerate(qs):
            if not q:
                continue
            a = ans[i] if i < len(ans) and ans[i] else ""
            yield mailbox, str(q), str(a)


def _iter_jsonl(path: Path | str) -> Iterator[dict[str, object]]:
    p = Path(path)
    files = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no .jsonl files under {p}")
    for f in files:
        with f.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record


def dedupe_messages(messages: Iterable[EnronMessage]) -> list[EnronMessage]:
    """Merge copies of the same message, unioning their mailboxes.

    This is where the corpus gets its multi-principal ACLs: one message held in
    six mailboxes becomes one document with six mailbox-owner viewers, instead of
    six documents with one viewer each. Six single-viewer documents would make
    every principal's permitted set disjoint from everyone else's, and a
    filtered-search benchmark on disjoint sets is a benchmark with no filter
    selectivity to measure.

    Returns:
        Messages in first-seen order, so a truncated run is a prefix of a full
        one and the two are comparable.
    """
    merged: dict[str, EnronMessage] = {}
    for msg in messages:
        existing = merged.get(msg.message_id)
        if existing is None:
            merged[msg.message_id] = msg
            continue
        boxes = tuple(dict.fromkeys((*existing.mailboxes, *msg.mailboxes)))
        paths = tuple(dict.fromkeys((*existing.source_paths, *msg.source_paths)))
        # Keep the copy with the most header information; forwarded copies often
        # lose the Cc line, and losing a Cc loses a viewer.
        best = existing if len(existing.recipients) >= len(msg.recipients) else msg
        merged[msg.message_id] = EnronMessage(
            message_id=best.message_id,
            subject=best.subject or existing.subject,
            sender=best.sender or existing.sender,
            to=best.to,
            cc=best.cc,
            date=best.date or existing.date,
            body=best.body,
            mailboxes=boxes,
            source_paths=paths,
        )
    return list(merged.values())


# --------------------------------------------------------------------------
# Document text
# --------------------------------------------------------------------------


def document_text(msg: EnronMessage) -> str:
    """Render a message as the text that gets chunked and embedded.

    Headers are included because "who sent this and when" is half of what people
    ask an email search. They are rendered as plain lines, not as a JSON blob,
    because the embedding model was trained on prose.
    """
    lines = []
    if msg.subject:
        lines.append(f"Subject: {msg.subject}")
    if msg.sender:
        lines.append(f"From: {msg.sender}")
    if msg.to:
        lines.append(f"To: {', '.join(msg.to)}")
    if msg.cc:
        lines.append(f"Cc: {', '.join(msg.cc)}")
    if msg.date:
        lines.append(f"Date: {msg.date}")
    header = "\n".join(lines)
    return f"{header}\n\n{msg.body}" if header else msg.body


def quote_boundaries(text: str) -> list[int]:
    """Character offsets where a quoted reply or forward begins.

    Passed to the chunker as hard boundaries. A forwarded thread is several
    messages in one file, and a chunk spanning the seam embeds a blend of two
    conversations and cites it as one. The ACL is unaffected — every part of the
    file carries the containing document's tokens — so this is a retrieval
    quality decision, not a security one.
    """
    offsets: set[int] = set()
    for marker in _QUOTE_MARKERS:
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            offsets.add(idx)
            start = idx + 1
    header_end = text.find("\n\n")
    if header_end > 0:
        offsets.add(header_end + 2)
    return sorted(o for o in offsets if 0 < o < len(text))


# --------------------------------------------------------------------------
# ACL derivation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DerivationConfig:
    """Knobs on the ACL derivation. All of them change what the corpus *is*."""

    min_group_size: int = MIN_GROUP_SIZE
    min_group_support: int = MIN_GROUP_SUPPORT
    #: Turn recurring recipient sets into groups (step 6 in the module docstring).
    synthesise_recipient_groups: bool = True
    #: Nest derived groups by set containment (step 7).
    nest_groups: bool = True
    #: Skip messages with no derivable viewer at all. They are not indexable
    #: (ADR 0003: empty means nobody), so the alternative to skipping is
    #: crashing, and a handful of headerless files should not stop an ingest.
    skip_unowned: bool = True


@dataclass(slots=True)
class AclDerivation:
    """Everything the derivation produced, with counters to argue about.

    Attributes:
        tuples: Every permission fact, ready for ``TupleStore.apply``.
        viewers_by_object: Per-document grants, for the ingest loop.
        groups: Derived group object id -> the member principals it stands for,
            fully expanded (after nesting is applied the *tuples* hold fewer
            direct members; this stays expanded so the oracle can check the
            walk arrives at the same set).
        stats: Counters. Printed by the ingest CLI and asserted on in tests.
    """

    tuples: list[Tuple_] = field(default_factory=list)
    viewers_by_object: dict[str, frozenset[str]] = field(default_factory=dict)
    groups: dict[str, frozenset[str]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)

    def tuple_strings(self) -> list[str]:
        return [str(t) for t in self.tuples]


def _group_id(members: Sequence[str]) -> str:
    digest = hashlib.blake2b("\x00".join(sorted(members)).encode(), digest_size=6).hexdigest()
    return f"rs_{digest}"


def _viewer_tuple(obj: ObjectRef, principal: PrincipalRef) -> Tuple_:
    return Tuple_(obj, VIEWER, principal)


def derive_acl(
    messages: Sequence[EnronMessage], config: DerivationConfig | None = None
) -> AclDerivation:
    """Turn messages into Zanzibar tuples.

    Args:
        messages: Deduplicated messages (run :func:`dedupe_messages` first, or
            the same message appears as several documents and the group
            statistics count it several times).
        config: Derivation knobs.

    Returns:
        An :class:`AclDerivation`.

    Note:
        Two passes. The first collects recipient sets and distribution-list
        sightings across the whole corpus, because a group is defined by
        recurrence and recurrence is not visible one message at a time. The
        second emits tuples. That means the whole message list is held in
        memory — at 40,000 messages it is a few hundred MB, which the reference
        machine has. A streaming version would need a two-pass file read and is
        not worth the complexity at this size.
    """
    cfg = config or DerivationConfig()
    registry: dict[str, str] = {}

    # -- pass 1: population statistics -------------------------------------
    set_support: Counter[frozenset[str]] = Counter()
    dl_members: dict[str, set[str]] = defaultdict(set)
    per_message: list[tuple[EnronMessage, set[str], set[str], set[str]]] = []

    for msg in messages:
        owners = {str(mailbox_principal(m)) for m in msg.mailboxes if m}
        people: set[str] = set(owners)
        recipient_dls: set[str] = set()
        grant_only_dls: set[str] = set()
        if msg.sender:
            sender_ref = principal_for_address(msg.sender, registry=registry)
            if sender_ref.namespace == "group":
                # A list in the From line (``hr.announcements@enron.com``) still
                # grants — somebody behind that alias sent it — but it gets no
                # membership inference. Receiving *from* a list is not evidence
                # of being *on* it, and treating it as such would put every
                # employee in the announcements group.
                grant_only_dls.add(sender_ref.id)
            else:
                people.add(str(sender_ref))
        for addr in msg.recipients:
            ref = principal_for_address(addr, registry=registry)
            if ref.namespace == "group":
                recipient_dls.add(ref.id)
            else:
                people.add(str(ref))
        # A mailbox owner holding a copy of a list-addressed message is
        # observable evidence they were on that list. This is the only
        # membership signal in the corpus that is not an inference.
        for dl in recipient_dls:
            dl_members[dl].update(owners)
        if len(people) >= cfg.min_group_size:
            set_support[frozenset(people)] += 1
        per_message.append((msg, owners, people, recipient_dls | grant_only_dls))

    # -- group table -------------------------------------------------------
    groups: dict[str, set[str]] = {dl: set(mem) for dl, mem in dl_members.items() if mem}
    set_to_group: dict[frozenset[str], str] = {}
    if cfg.synthesise_recipient_groups:
        for members, support in set_support.items():
            if support < cfg.min_group_support or len(members) < cfg.min_group_size:
                continue
            gid = _group_id(sorted(members))
            set_to_group[members] = gid
            groups[gid] = set(members)

    expanded = {gid: frozenset(mem) for gid, mem in groups.items()}
    nesting: list[tuple[str, str]] = []
    direct: dict[str, set[str]] = {gid: set(mem) for gid, mem in groups.items()}
    if cfg.nest_groups:
        nesting = _nest_groups(expanded, direct)

    # -- pass 2: tuples ----------------------------------------------------
    tuples: list[Tuple_] = []
    seen: set[Tuple_] = set()

    def emit(t: Tuple_) -> None:
        if t not in seen:
            seen.add(t)
            tuples.append(t)

    for gid, members in sorted(direct.items()):
        gobj = ObjectRef("group", gid)
        for member in sorted(members):
            emit(Tuple_(gobj, "member", PrincipalRef.parse(member)))
    for parent, child in nesting:
        emit(
            Tuple_(
                ObjectRef("group", parent),
                "member",
                PrincipalRef("group", child, "member"),
            )
        )

    viewers_by_object: dict[str, frozenset[str]] = {}
    n_skipped = 0
    n_group_grants = 0
    n_direct_grants = 0

    for msg, owners, people, dls in per_message:
        if not people and not dls:
            n_skipped += 1
            continue
        obj = msg.object
        granted: set[str] = set()
        # Distribution lists first: one tuple standing for many people is the
        # shape the compiler is built for, and using it wherever it is available
        # is the difference between a realistic token count and a fake one.
        for dl in sorted(dls):
            principal = PrincipalRef("group", dl, "member")
            emit(_viewer_tuple(obj, principal))
            granted.add(str(principal))
            n_group_grants += 1
        frozen = frozenset(people)
        gid = set_to_group.get(frozen)
        if gid is not None:
            principal = PrincipalRef("group", gid, "member")
            emit(_viewer_tuple(obj, principal))
            granted.add(str(principal))
            n_group_grants += 1
            # The mailbox owners still get a direct grant. Their access does not
            # depend on the synthesised group being right, and a synthesised
            # group is the one part of this derivation that could be wrong.
            for owner in sorted(owners):
                emit(_viewer_tuple(obj, PrincipalRef.parse(owner)))
                granted.add(owner)
                n_direct_grants += 1
        else:
            for person in sorted(people):
                emit(_viewer_tuple(obj, PrincipalRef.parse(person)))
                granted.add(person)
                n_direct_grants += 1
        if not granted and cfg.skip_unowned:
            n_skipped += 1
            continue
        viewers_by_object[str(obj)] = frozenset(granted)

    stats = {
        "messages": len(messages),
        "documents": len(viewers_by_object),
        "skipped_no_viewer": n_skipped,
        "tuples": len(tuples),
        "groups": len(groups),
        "groups_from_distribution_lists": len(dl_members),
        "groups_from_recipient_sets": len(set_to_group),
        "nested_group_edges": len(nesting),
        "grants_via_group": n_group_grants,
        "grants_direct": n_direct_grants,
        "distinct_principals": len({p for v in viewers_by_object.values() for p in v}),
    }
    return AclDerivation(
        tuples=tuples,
        viewers_by_object=viewers_by_object,
        groups=expanded,
        stats=stats,
    )


def _nest_groups(
    expanded: dict[str, frozenset[str]], direct: dict[str, set[str]]
) -> list[tuple[str, str]]:
    """Attach each group's largest strict subgroups as member usersets.

    Mutates ``direct`` in place, removing members now reachable through a nested
    group. The result is a real multi-hop userset graph for the compiler to walk
    rather than a flat one-hop fixture.

    Cycles are impossible because a child is only nested under a strictly larger
    parent, and "strictly larger" is a well-founded order. That is the whole
    cycle argument and it is written down because the obvious alternative
    (subset without the size check) admits ``A ⊆ B ⊆ A`` for equal sets and
    produces a two-node loop that ``check()`` bounds out of rather than
    reporting.
    """
    by_size = sorted(expanded, key=lambda g: (-len(expanded[g]), g))
    edges: list[tuple[str, str]] = []
    for parent in by_size:
        remaining = direct[parent]
        parent_size = len(expanded[parent])
        while True:
            best: str | None = None
            best_size = 1
            for child in by_size:
                if child == parent:
                    continue
                members = expanded[child]
                size = len(members)
                if size <= best_size or size >= parent_size:
                    continue
                if members <= remaining:
                    best, best_size = child, size
            if best is None:
                break
            edges.append((parent, best))
            remaining -= expanded[best]
    return edges


# --------------------------------------------------------------------------
# A fixture, clearly labelled as one
# --------------------------------------------------------------------------


def sample_messages() -> list[EnronMessage]:
    """Eight hand-written messages with Enron-shaped headers. NOT THE CORPUS.

    This exists so the derivation has a test subject on a machine with no
    corpus downloaded. It is written to contain exactly the structures the
    derivation is supposed to find — one distribution list, one recurring
    recipient set, one nested pair, one message present in two mailboxes — and
    therefore it proves the code does what it says and proves nothing at all
    about Enron. Numbers in the README come from the real corpus or they do not
    get published.
    """
    core = ("kay.mann@enron.com", "sara.shackleton@enron.com", "tana.jones@enron.com")
    body = (
        "The counterparty wants the netting annex redrafted before the "
        "confirmation goes out. I have marked up section 4 and attached the "
        "revised schedule. Please review before Thursday's call."
    )
    out: list[EnronMessage] = []
    for i in range(5):
        out.append(
            EnronMessage(
                message_id=f"<sample-core-{i}@enron.com>",
                subject=f"Re: ISDA schedule, draft {i + 1}",
                sender="mark.taylor@enron.com",
                to=core,
                cc=(),
                date=f"Mon, {10 + i} Sep 2001 09:1{i}:00 -0500",
                body=f"{body} (revision {i + 1})",
                mailboxes=("mann-k",),
            )
        )
    out.append(
        EnronMessage(
            message_id="<sample-wider@enron.com>",
            subject="Legal team sync",
            sender="mark.taylor@enron.com",
            to=(*core, "carol.clair@enron.com", "elizabeth.sager@enron.com"),
            cc=(),
            date="Tue, 18 Sep 2001 14:02:00 -0500",
            body="Agenda for the weekly legal sync: open confirmations, "
            "the pending novation, and headcount for Q4.",
            mailboxes=("mann-k", "shackleton-s"),
        )
    )
    for i in range(4):
        out.append(
            EnronMessage(
                message_id=f"<sample-wider-{i}@enron.com>",
                subject=f"Legal team sync, week {i + 2}",
                sender="mark.taylor@enron.com",
                to=(*core, "carol.clair@enron.com", "elizabeth.sager@enron.com"),
                cc=(),
                date=f"Tue, {25 + i} Sep 2001 14:02:00 -0500",
                body=f"Weekly legal sync notes, week {i + 2}. Open confirmations "
                "and the pending novation remain outstanding.",
                mailboxes=("mann-k",),
            )
        )
    out.append(
        EnronMessage(
            message_id="<sample-allemp@enron.com>",
            subject="Benefits enrolment closes Friday",
            sender="hr.announcements@enron.com",
            to=("all.employees@enron.com",),
            cc=(),
            date="Wed, 26 Sep 2001 08:00:00 -0500",
            body="Open enrolment for the 2002 benefits year closes on Friday. "
            "Elections not made by then will roll over unchanged.",
            mailboxes=("mann-k", "shackleton-s", "jones-t"),
        )
    )
    return out
