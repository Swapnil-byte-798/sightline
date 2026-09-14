"""Chunking: one document in, a list of :class:`~sightline.types.Chunk` out.

Three properties matter here and nothing else does.

**1. A chunk never spans two documents.** This is not a quality concern, it is a
security one. A chunk carries exactly one object's grant tokens; text from two
objects in one chunk would carry the union of two ACLs, which is a leak with a
plausible-sounding explanation attached. The API makes it unrepresentable:
:func:`chunk_document` takes a single :class:`ObjectRef` and a single string, and
there is no function anywhere in this package that concatenates documents.

**2. Chunk ids are deterministic.** Re-running ingest over unchanged input must
produce byte-identical ids, because ``VectorStore.upsert`` is idempotent on
``chunk.id`` and that is the whole resumability story (FR-15). The id is a hash
of ``(object, ordinal, tokenizer name, chunk text)``.

The tokenizer name is in there deliberately. Chunk boundaries depend on how text
is tokenised, so the approximate tokenizer and the real WordPiece tokenizer cut
the same document in different places. Without the tokenizer name in the hash,
switching between them would silently produce two overlapping copies of the
corpus that look like one. With it, they are visibly different ids and the
collection rebuild is forced rather than forgotten.

**3. Overlap, because retrieval cuts sentences in half.** A fixed stride of
``target - overlap`` tokens, with the cut snapped backwards to a sentence end
when one is nearby.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
* **Semantic / recursive chunking.** Splitting on embedding-similarity drift is
  better and is measurable, so it belongs behind an eval, not in v1 by assertion.
* **Deletion of superseded chunks.** Content-addressed ids mean an edited
  document produces *new* ids; the old ones do not get overwritten, they get
  orphaned. ``VectorStore`` has no ``delete`` in v1, so the orphan survives until
  the collection is rebuilt. That is a real, unfixed hole and it is written down
  here rather than discovered later. It does not leak: an orphan chunk keeps the
  tokens it was stamped with, and recheck still runs against live tuples.
* **Table / code-block awareness.** Email bodies do not need it. A SharePoint
  connector would.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from sightline.types import Chunk, GrantToken, ObjectRef

__all__ = [
    "DEFAULT_MIN_TOKENS",
    "DEFAULT_OVERLAP_TOKENS",
    "DEFAULT_TARGET_TOKENS",
    "SENTENCE_SNAP_SLACK",
    "ApproxWordTokenizer",
    "Chunker",
    "ChunkingConfig",
    "HFTokenizer",
    "SpanTokenizer",
    "TextSpan",
    "chunk_document",
    "chunk_id_for",
    "default_tokenizer",
    "split_text",
]

# --------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------
#
# 256 tokens is not a tuned number. It is the size at which all-MiniLM-L6-v2
# stops truncating (its trained max sequence length is 256 WordPiece tokens, and
# text past that is silently dropped by the model, not by us). Chunking larger
# than the model's window is the single most common way to ship an index where a
# third of every chunk was never embedded. Tuning this downwards is an eval
# question; tuning it upwards is a bug.

#: Target chunk length in tokens of whatever tokenizer is in use.
DEFAULT_TARGET_TOKENS = 256

#: Tokens repeated between adjacent chunks. Roughly two sentences of English.
DEFAULT_OVERLAP_TOKENS = 48

#: Below this a trailing fragment is dropped if the previous chunk already
#: covers it. A 4-token chunk is an embedding of noise with a citation attached.
DEFAULT_MIN_TOKENS = 16

#: How far back a cut may be moved to land on a sentence end.
SENTENCE_SNAP_SLACK = 32

# Words, contractions and hyphenations as one unit; every other non-space
# character as its own. This overcounts punctuation-heavy text relative to
# WordPiece and undercounts long rare words, which is why it is called
# "approximate" in its name and why the name is hashed into every chunk id.
_TOKEN_RE = re.compile(r"\w+(?:[.'’-]\w+)*|[^\w\s]", re.UNICODE)

_SENTENCE_END_RE = re.compile(r"[.!?…]['\"’”)\]]*$")


@runtime_checkable
class SpanTokenizer(Protocol):
    """Anything that can say where the tokens of a string start and end.

    Character spans, not token strings: the chunker slices the *original* text so
    that whitespace, casing and line breaks survive into the chunk. Detokenising
    WordPiece back into prose would quietly rewrite the corpus.
    """

    name: str

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Half-open ``(start, end)`` character spans, in order, non-overlapping."""
        ...


class ApproxWordTokenizer:
    """Regex tokenizer with zero dependencies.

    This exists so that chunking — and therefore the whole test suite — runs with
    only the core dependencies installed. It is not WordPiece and does not claim
    to be; on ordinary English prose it lands within roughly 25% of MiniLM's
    count, which is close enough to keep chunks inside the model window given the
    256-token target, and not close enough to publish a token count from.
    """

    name = "approx-word-v1"

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in _TOKEN_RE.finditer(text)]


class HFTokenizer:
    """The real WordPiece tokenizer, when the ``embed`` extra is installed.

    Truncation is explicitly disabled. The tokenizer's own truncation would cut
    a long document at 256 tokens and return nothing about the rest, so a
    100-page document would index its first paragraph and silently lose the
    other ninety-nine.
    """

    def __init__(self, tokenizer_path: str) -> None:
        try:
            from tokenizers import Tokenizer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ImportError(
                "HFTokenizer needs the 'tokenizers' package. "
                "Install it with: pip install 'sightline[embed]'"
            ) from exc
        self._tok = Tokenizer.from_file(tokenizer_path)
        self._tok.no_truncation()
        self._tok.no_padding()
        self.name = f"hf:{hashlib.blake2b(tokenizer_path.encode(), digest_size=6).hexdigest()}"

    def spans(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok.encode(text, add_special_tokens=False)
        # Some normalisers emit zero-width spans for stripped characters. They
        # would create empty chunks at a boundary, so drop them here.
        return [(s, e) for s, e in enc.offsets if e > s]


def default_tokenizer() -> SpanTokenizer:
    """The approximate tokenizer, always. Never probes for the optional one.

    Silently upgrading the tokenizer when an extra happens to be installed would
    make chunk ids depend on the machine's pip history. Pass an
    :class:`HFTokenizer` explicitly when you want it.
    """
    return ApproxWordTokenizer()


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Chunk geometry. Frozen because it is hashed into the collection identity.

    Changing any field here changes every chunk id in the corpus, which means a
    full reindex. That is the correct cost and it should be obvious, so the
    config is a value, not a pile of keyword arguments threaded through calls.
    """

    target_tokens: int = DEFAULT_TARGET_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    min_tokens: int = DEFAULT_MIN_TOKENS
    snap_slack: int = SENTENCE_SNAP_SLACK

    def __post_init__(self) -> None:
        if self.target_tokens < 1:
            raise ValueError("target_tokens must be >= 1")
        if not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError(
                f"overlap_tokens must be in [0, target_tokens); "
                f"got {self.overlap_tokens} with target {self.target_tokens}. "
                "An overlap >= target never advances and would loop forever."
            )
        if self.min_tokens < 0:
            raise ValueError("min_tokens must be >= 0")
        if self.snap_slack < 0:
            raise ValueError("snap_slack must be >= 0")

    @property
    def stride(self) -> int:
        return self.target_tokens - self.overlap_tokens

    def fingerprint(self) -> str:
        """Short stable digest of the geometry, for cache and collection names."""
        raw = f"{self.target_tokens}:{self.overlap_tokens}:{self.min_tokens}:{self.snap_slack}"
        return hashlib.blake2b(raw.encode(), digest_size=6).hexdigest()


@dataclass(frozen=True, slots=True)
class TextSpan:
    """A prospective chunk: text plus where it came from in the source string."""

    ordinal: int
    text: str
    char_start: int
    char_end: int
    n_tokens: int


def chunk_id_for(object: ObjectRef, ordinal: int, text: str, tokenizer_name: str) -> str:
    """Deterministic id for one chunk.

    Content-addressed on purpose: two ingest runs over unchanged input produce
    the same ids, so ``upsert`` is a no-op and a run that died halfway through
    can simply be re-run. The object and ordinal are in the hash as well as the
    text, so two documents that happen to contain the same paragraph get
    distinct chunks — sharing one would mean sharing one ACL.
    """
    payload = f"{object}\x00{ordinal}\x00{tokenizer_name}\x00{text}".encode()
    return f"ch_{hashlib.blake2b(payload, digest_size=16).hexdigest()}"


def _ends_sentence(text: str, span: tuple[int, int]) -> bool:
    return _SENTENCE_END_RE.search(text[span[0] : span[1]]) is not None


def _snap_back(
    text: str, spans: Sequence[tuple[int, int]], start: int, end: int, slack: int
) -> int:
    """Move a cut backwards onto a sentence end, if one is within ``slack``.

    Returns ``end`` unchanged when no sentence boundary is close enough. Never
    moves the cut to or before ``start``: a zero-length chunk would not advance.
    """
    floor = max(start + 1, end - slack)
    for j in range(end, floor - 1, -1):
        if _ends_sentence(text, spans[j - 1]):
            return j
    return end


def _boundary_limit(
    spans: Sequence[tuple[int, int]], start: int, boundaries: Sequence[int]
) -> int | None:
    """Index of the first token at or after the next hard boundary past ``start``.

    Hard boundaries are char offsets a chunk must never straddle — used by the
    Enron connector to keep a message's headers, its body and each quoted reply
    in separate chunks, so that a quoted forward does not smear one message's
    text into another's chunk. Returns ``None`` when no boundary applies.
    """
    if not boundaries:
        return None
    start_char = spans[start][0]
    for b in boundaries:
        if b <= start_char:
            continue
        for idx in range(start, len(spans)):
            if spans[idx][0] >= b:
                return idx if idx > start else None
        return None
    return None


def split_text(
    text: str,
    *,
    config: ChunkingConfig | None = None,
    tokenizer: SpanTokenizer | None = None,
    boundaries: Sequence[int] = (),
) -> list[TextSpan]:
    """Split one document's text into overlapping spans.

    Args:
        text: The document body. Sliced, never rewritten.
        config: Chunk geometry.
        tokenizer: How to count tokens. Defaults to the dependency-free
            approximate tokenizer.
        boundaries: Sorted character offsets no chunk may cross. Overlap is
            **not** carried across a boundary, because the point of the boundary
            is that the two sides are different material.

    Returns:
        Spans in document order, with ``ordinal`` counting from zero.
    """
    cfg = config or ChunkingConfig()
    tok = tokenizer or default_tokenizer()
    spans = tok.spans(text)
    if not spans:
        return []

    bounds = sorted(b for b in boundaries if 0 < b < len(text))
    out: list[TextSpan] = []
    i = 0
    n = len(spans)
    while i < n:
        end = min(i + cfg.target_tokens, n)
        limit = _boundary_limit(spans, i, bounds)
        hit_boundary = limit is not None and limit < end
        if limit is not None:
            end = min(end, limit)
        if end < n and not hit_boundary:
            end = _snap_back(text, spans, i, end, cfg.snap_slack)

        char_start = spans[i][0]
        char_end = spans[end - 1][1]
        piece = text[char_start:char_end]
        n_tokens = end - i

        redundant = bool(out) and char_end <= out[-1].char_end
        if not redundant and (n_tokens >= cfg.min_tokens or not out):
            out.append(TextSpan(len(out), piece, char_start, char_end, n_tokens))

        if end >= n:
            break
        # No overlap across a hard boundary; the next chunk starts clean.
        nxt = end if hit_boundary else max(end - cfg.overlap_tokens, i + 1)
        i = max(nxt, i + 1)
    return out


def chunk_document(
    object: ObjectRef,
    text: str,
    *,
    grant_tokens: frozenset[GrantToken],
    metadata: dict[str, str] | None = None,
    config: ChunkingConfig | None = None,
    tokenizer: SpanTokenizer | None = None,
    boundaries: Sequence[int] = (),
) -> list[Chunk]:
    """Chunk one document and stamp every chunk with the document's tokens.

    Args:
        object: The single object these chunks belong to. One object, always.
        text: Its body.
        grant_tokens: The document's compiled grant tokens, from
            :func:`sightline.authz.compile.grant_tokens_for_object`.
        metadata: Extra string fields copied onto every chunk.
        config: Chunk geometry.
        tokenizer: Token span provider.
        boundaries: Hard char offsets no chunk may straddle.

    Returns:
        Chunks in document order.

    Raises:
        ValueError: If ``grant_tokens`` is empty. Empty means nobody, always
            (ADR 0003), so a chunk nobody can see must not reach the index —
            not because indexing it would leak today, but because the only way
            it becomes visible later is a bug in the filter, and there is no
            reason to leave that ammunition lying in the collection. FR-15.
    """
    if not grant_tokens:
        raise ValueError(
            f"refusing to chunk {object}: zero grant tokens. "
            "An empty token set means nobody may see this document (ADR 0003); "
            "it is never a synonym for 'everyone'. Fix the document's ACL or "
            "skip the document."
        )
    cfg = config or ChunkingConfig()
    tok = tokenizer or default_tokenizer()
    base = dict(metadata or {})
    chunks: list[Chunk] = []
    for span in split_text(text, config=cfg, tokenizer=tok, boundaries=boundaries):
        meta = dict(base)
        meta.update(
            {
                "ordinal": str(span.ordinal),
                "n_tokens": str(span.n_tokens),
                "char_start": str(span.char_start),
                "char_end": str(span.char_end),
                "tokenizer": tok.name,
                "chunking": cfg.fingerprint(),
            }
        )
        chunks.append(
            Chunk(
                id=chunk_id_for(object, span.ordinal, span.text, tok.name),
                object=object,
                text=span.text,
                grant_tokens=grant_tokens,
                metadata=meta,
            )
        )
    return chunks


@dataclass(slots=True)
class Chunker:
    """A configured chunker. Holds the tokenizer so it is constructed once.

    Building an :class:`HFTokenizer` reads and parses a 700 KB JSON file. Doing
    that per document turns a 40-minute ingest into an overnight one.
    """

    config: ChunkingConfig = field(default_factory=ChunkingConfig)
    tokenizer: SpanTokenizer | None = None

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = default_tokenizer()

    @property
    def tokenizer_name(self) -> str:
        assert self.tokenizer is not None
        return self.tokenizer.name

    def with_tokenizer(self, tokenizer: SpanTokenizer) -> Chunker:
        return replace(self, tokenizer=tokenizer)

    def split(self, text: str, *, boundaries: Sequence[int] = ()) -> list[TextSpan]:
        return split_text(
            text, config=self.config, tokenizer=self.tokenizer, boundaries=boundaries
        )

    def chunk(
        self,
        object: ObjectRef,
        text: str,
        *,
        grant_tokens: frozenset[GrantToken],
        metadata: dict[str, str] | None = None,
        boundaries: Sequence[int] = (),
    ) -> list[Chunk]:
        return chunk_document(
            object,
            text,
            grant_tokens=grant_tokens,
            metadata=metadata,
            config=self.config,
            tokenizer=self.tokenizer,
            boundaries=boundaries,
        )
