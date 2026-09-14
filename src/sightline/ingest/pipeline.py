"""Ingest orchestration: parse, chunk, embed, stamp, upsert — resumably.

This runs in two places and both of them interrupt it. On free CI there is a
six-hour job cap. On the reference laptop the lid closes. So the pipeline is
built around the assumption that it will be killed partway through and started
again, and the design follows from that:

* **Chunk ids are content-addressed**, so ``upsert`` on an already-indexed chunk
  is a no-op rather than a duplicate.
* **Vectors are cached on disk by content hash**, so a restart re-embeds nothing
  it already embedded, which is where all the time goes.
* **A checkpoint records completed object ids**, appended after the vectors for
  that object are durably written. The order matters: vectors first, checkpoint
  second. Crash between them and the object is re-ingested, costing a cache hit
  and an idempotent upsert. The other order would mark an object done whose
  vectors never landed, and the document would be missing from the index with
  nothing to indicate it.

ORDER OF OPERATIONS, AND WHY
----------------------------
Tuples are written to the authority **before** any vector is. Grant tokens are
computed from the tuple store, so an object whose tuples are not in yet would
compile to the empty set and be rejected. Rejected, not indexed-as-public — that
is ADR 0003, and this pipeline is where it bites: a corpus where the ACL import
half-failed produces an ingest that refuses most of its documents and says so,
rather than an index that quietly works for everyone.

The epoch is recorded at the start and again at the end. If policy changed
during the run, some chunks carry tokens compiled under the older epoch. That is
safe and it is the central rule doing its job: the index is stale, recheck is
authoritative, and a stale chunk is dropped rather than served. The report says
so explicitly instead of leaving the reader to work it out.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
* **Parallel workers.** Two cores, and the embedder already saturates them.
  A process pool would add pickling cost and a second failure mode for no gain
  on the machine this is measured on.
* **A queue / broker.** v1 ingests from a local corpus. FR-17's reindex is a
  function call, not a job. Named as missing rather than mocked.
* **Deletion of superseded chunks.** ``VectorStore`` has no ``delete`` in v1.
  See ``chunk.py`` for the same hole from the other side.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sightline.authz.compile import DEFAULT_RELATION, grant_tokens_for_object
from sightline.authz.tuples import TupleStore
from sightline.ingest.chunk import Chunker, ChunkingConfig
from sightline.ingest.embed import Embedder, EmbeddingStats, VectorCache, embed_all
from sightline.ingest.enron import (
    AclDerivation,
    DerivationConfig,
    EnronMessage,
    dedupe_messages,
    derive_acl,
    document_text,
    quote_boundaries,
)
from sightline.store.base import VectorStore
from sightline.types import Chunk, ObjectRef, Relation

__all__ = [
    "Checkpoint",
    "IngestConfig",
    "IngestReport",
    "ProgressReporter",
    "SourceDocument",
    "documents_from_messages",
    "ingest_messages",
    "main",
    "reindex_object",
    "run_ingest",
]


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One object's text, ready to chunk. The pipeline's only input shape.

    Deliberately not "a file path": the connector decides what a document is.
    Making the pipeline take paths would push mailbox deduplication and header
    rendering into it, and then the Enron-specific reasoning would be somewhere
    a SharePoint connector has to read.
    """

    object: ObjectRef
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    #: Char offsets no chunk may straddle. See ``enron.quote_boundaries``.
    boundaries: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class IngestConfig:
    """Everything that changes what the index contains.

    Changing any of ``chunking``, ``relation`` or the embedder identity
    invalidates every chunk id or every vector, so all three end up in the run
    fingerprint that names the cache directory. A cache shared between two
    configurations is a cache that hands back vectors from the wrong model.
    """

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    relation: Relation = DEFAULT_RELATION
    #: Chunks embedded and upserted per batch. 512 chunks of 384 float32 is
    #: 780 KB — small enough that a kill loses under a second of work.
    batch_chunks: int = 512
    #: Rows per model forward pass.
    embed_batch: int = 32
    #: Stop once this many chunks have been written. ``None`` for no limit.
    max_chunks: int | None = None
    #: Stop after this many documents. ``None`` for no limit.
    max_documents: int | None = None
    #: Where the vector cache and checkpoint live.
    work_dir: Path | None = None
    #: Emit progress lines to stderr.
    progress: bool = True
    #: Seconds between progress lines. Not per batch: a progress line per batch
    #: on a six-hour CI job is 40,000 lines nobody reads and a log nobody keeps.
    progress_interval: float = 5.0


@dataclass(slots=True)
class IngestReport:
    """What the run did. Every field is counted, none is estimated.

    ``epoch_start`` and ``epoch_end`` differing is not an error. It means policy
    moved during ingest, so some chunks are stamped against an older view. The
    index is a hint; recheck is the authority. The report states it so that the
    fact is in the log rather than in someone's head.
    """

    documents_seen: int = 0
    documents_indexed: int = 0
    documents_skipped_resumed: int = 0
    documents_rejected_no_tokens: int = 0
    documents_empty_text: int = 0
    chunks_written: int = 0
    vectors_written: int = 0
    seconds: float = 0.0
    epoch_start: int = 0
    epoch_end: int = 0
    embedder: str = ""
    embedder_is_semantic: bool = False
    embed_seconds: float = 0.0
    embed_cached: int = 0
    embed_computed: int = 0
    embed_deduplicated: int = 0
    truncated_at_limit: bool = False

    @property
    def chunks_per_second(self) -> float:
        return self.chunks_written / self.seconds if self.seconds > 0 else 0.0

    @property
    def policy_moved(self) -> bool:
        return self.epoch_end != self.epoch_start

    def to_dict(self) -> dict[str, object]:
        return {
            "documents_seen": self.documents_seen,
            "documents_indexed": self.documents_indexed,
            "documents_skipped_resumed": self.documents_skipped_resumed,
            "documents_rejected_no_tokens": self.documents_rejected_no_tokens,
            "documents_empty_text": self.documents_empty_text,
            "chunks_written": self.chunks_written,
            "vectors_written": self.vectors_written,
            "seconds": round(self.seconds, 2),
            "chunks_per_second": round(self.chunks_per_second, 1),
            "epoch_start": self.epoch_start,
            "epoch_end": self.epoch_end,
            "policy_moved_during_ingest": self.policy_moved,
            "embedder": self.embedder,
            "embedder_is_semantic": self.embedder_is_semantic,
            "embed_seconds": round(self.embed_seconds, 2),
            "embed_cached": self.embed_cached,
            "embed_computed": self.embed_computed,
            "embed_deduplicated": self.embed_deduplicated,
            "truncated_at_limit": self.truncated_at_limit,
        }

    def __str__(self) -> str:
        flag = "" if self.embedder_is_semantic else "  [NOT SEMANTIC EMBEDDER]"
        moved = "  [policy moved during ingest]" if self.policy_moved else ""
        return (
            f"{self.documents_indexed} docs, {self.chunks_written} chunks, "
            f"{self.chunks_per_second:.1f} chunks/s, "
            f"{self.documents_rejected_no_tokens} rejected for zero grant tokens, "
            f"epoch {self.epoch_start}->{self.epoch_end}{moved}{flag}"
        )

    def _absorb(self, stats: EmbeddingStats) -> None:
        self.embedder = stats.embedder
        self.embedder_is_semantic = stats.is_semantic
        self.embed_seconds += stats.seconds
        self.embed_cached += stats.n_cached
        self.embed_computed += stats.n_computed
        self.embed_deduplicated += stats.n_deduplicated


class Checkpoint:
    """Append-only record of which objects are fully indexed.

    A JSONL file of ``{"o": "doc:abc"}``, flushed and fsynced after each batch.
    Append-only rather than a rewritten set, because rewriting a 40,000-entry
    file after every batch is both slow and a window in which a kill leaves a
    truncated file. An append that dies mid-line leaves one torn line, which the
    loader drops.

    A no-op instance (``path=None``) is a valid Checkpoint. Tests and one-shot
    reindexes should not have to create a directory to run.
    """

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path is not None else None
        self._done: set[str] = set()
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._done.add(str(json.loads(line)["o"]))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    def __contains__(self, object_ref: str) -> bool:
        return object_ref in self._done

    def __len__(self) -> int:
        return len(self._done)

    def mark(self, object_refs: Sequence[str]) -> None:
        fresh = [o for o in object_refs if o not in self._done]
        if not fresh:
            return
        self._done.update(fresh)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for o in fresh:
                fh.write(json.dumps({"o": o}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


class ProgressReporter:
    """Throttled one-line progress to stderr, with a rate and an ETA.

    stderr, not stdout: the ingest CLI prints its report as JSON on stdout, and
    a progress line in the middle of it makes the report unparseable.
    """

    def __init__(
        self,
        enabled: bool = True,
        interval: float = 5.0,
        total: int | None = None,
        stream=None,
    ) -> None:
        self.enabled = enabled
        self.interval = interval
        self.total = total
        self.stream = stream if stream is not None else sys.stderr
        self._start = time.perf_counter()
        self._last = 0.0

    def update(self, done: int, *, suffix: str = "", force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if not force and now - self._last < self.interval:
            return
        self._last = now
        elapsed = max(now - self._start, 1e-9)
        rate = done / elapsed
        parts = [f"{done}"]
        if self.total:
            pct = 100.0 * done / self.total
            remaining = (self.total - done) / rate if rate > 0 else 0.0
            parts.append(f"/{self.total} ({pct:.1f}%) eta {remaining / 60:.1f}m")
        parts.append(f"{rate:.1f}/s")
        if suffix:
            parts.append(suffix)
        print("  ingest: " + " ".join(parts), file=self.stream, flush=True)

    def done(self, done: int, *, suffix: str = "") -> None:
        self.update(done, suffix=suffix, force=True)


def _cache_for(config: IngestConfig, embedder: Embedder) -> VectorCache | None:
    """Cache directory keyed by embedder identity, dim and chunk geometry.

    All three are in the path. Sharing one cache across two embedders would hand
    back MiniLM vectors for a HashEmbedder run, which is not a bug that announces
    itself — the numbers just come out wrong.
    """
    if config.work_dir is None:
        return None
    name = f"{embedder.name}-d{embedder.dim}-c{config.chunking.fingerprint()}"
    return VectorCache(Path(config.work_dir) / "vectors" / name, dim=embedder.dim)


def documents_from_messages(messages: Iterable[EnronMessage]) -> Iterator[SourceDocument]:
    """Render Enron messages as pipeline input.

    Kept here rather than in ``enron.py`` because it is the seam between the
    connector's vocabulary (messages, mailboxes) and the pipeline's (objects,
    text). A second connector adds a second function like this one and changes
    nothing else.
    """
    for msg in messages:
        text = document_text(msg)
        yield SourceDocument(
            object=msg.object,
            text=text,
            metadata={
                "subject": msg.subject[:200],
                "sender": msg.sender,
                "date": msg.date,
                "mailboxes": ",".join(msg.mailboxes),
                "source": "enron",
            },
            boundaries=tuple(quote_boundaries(text)),
        )


def run_ingest(
    documents: Iterable[SourceDocument],
    *,
    tuple_store: TupleStore,
    vector_store: VectorStore,
    embedder: Embedder,
    config: IngestConfig | None = None,
    chunker: Chunker | None = None,
    checkpoint: Checkpoint | None = None,
    on_report: Callable[[IngestReport], None] | None = None,
) -> IngestReport:
    """Chunk, embed and upsert documents whose tuples are already written.

    Args:
        documents: Source documents, streamed. The pipeline never holds more
            than one batch of chunks at a time.
        tuple_store: The authority. Grant tokens are compiled from it per
            document, so its tuples must already be written.
        vector_store: Where vectors land. Must be idempotent on ``chunk.id``.
        embedder: Any :class:`~sightline.ingest.embed.Embedder`.
        config: Batch sizes, limits, work directory.
        chunker: Pre-built chunker, so an ``HFTokenizer`` is loaded once.
        checkpoint: Resumption state. Defaults to one under ``work_dir``.
        on_report: Called with the report after each batch, for live dashboards.

    Returns:
        An :class:`IngestReport`.

    Note:
        A document whose compiled token set is empty is **rejected and counted**,
        never indexed. Empty means nobody (ADR 0003). If the rejected count is
        most of the corpus, the ACL import failed, and this is the number that
        says so.
    """
    cfg = config or IngestConfig()
    chk = chunker or Chunker(config=cfg.chunking)
    cp = checkpoint
    if cp is None:
        cp_path = Path(cfg.work_dir) / "checkpoint.jsonl" if cfg.work_dir else None
        cp = Checkpoint(cp_path)
    cache = _cache_for(cfg, embedder)

    report = IngestReport(epoch_start=tuple_store.epoch())
    report.embedder = embedder.name
    report.embedder_is_semantic = embedder.is_semantic
    reporter = ProgressReporter(
        enabled=cfg.progress, interval=cfg.progress_interval, total=cfg.max_chunks
    )
    started = time.perf_counter()

    pending: list[Chunk] = []
    pending_objects: list[str] = []

    def flush() -> None:
        if not pending:
            return
        vectors, stats = embed_all(
            embedder,
            [c.text for c in pending],
            cache=cache,
            batch_size=cfg.embed_batch,
        )
        vector_store.upsert(pending, vectors)
        report.chunks_written += len(pending)
        report.vectors_written += len(pending)
        report._absorb(stats)
        # Vectors are durable before the checkpoint says so. Crash in between
        # and the object is redone from cache; the reverse would lose it.
        cp.mark(pending_objects)
        pending.clear()
        pending_objects.clear()
        if on_report is not None:
            on_report(report)

    for doc in documents:
        report.documents_seen += 1
        ref = str(doc.object)
        if ref in cp:
            report.documents_skipped_resumed += 1
            continue
        if not doc.text.strip():
            report.documents_empty_text += 1
            continue
        tokens = grant_tokens_for_object(tuple_store, doc.object, cfg.relation)
        if not tokens:
            report.documents_rejected_no_tokens += 1
            continue
        chunks = chk.chunk(
            doc.object,
            doc.text,
            grant_tokens=tokens,
            metadata=doc.metadata,
            boundaries=doc.boundaries,
        )
        if not chunks:
            report.documents_empty_text += 1
            continue
        pending.extend(chunks)
        pending_objects.append(ref)
        report.documents_indexed += 1

        if len(pending) >= cfg.batch_chunks:
            flush()
            reporter.update(report.chunks_written, suffix=f"{report.documents_indexed} docs")
        if cfg.max_chunks is not None and report.chunks_written + len(pending) >= cfg.max_chunks:
            report.truncated_at_limit = True
            break
        if cfg.max_documents is not None and report.documents_indexed >= cfg.max_documents:
            report.truncated_at_limit = True
            break

    flush()
    report.seconds = time.perf_counter() - started
    report.epoch_end = tuple_store.epoch()
    reporter.done(report.chunks_written, suffix=f"{report.documents_indexed} docs")
    return report


def ingest_messages(
    messages: Sequence[EnronMessage],
    *,
    tuple_store: TupleStore,
    vector_store: VectorStore,
    embedder: Embedder,
    config: IngestConfig | None = None,
    derivation: DerivationConfig | None = None,
    chunker: Chunker | None = None,
) -> tuple[IngestReport, AclDerivation]:
    """End to end: dedupe, derive tuples, write them, then index.

    The tuple write happens in **one** ``apply`` call, so the whole corpus ACL
    lands under a single epoch bump. Writing them one at a time would bump the
    epoch tens of thousands of times, which is correct but would invalidate every
    compiled plan on every write and make the ingest look like a permission
    storm in the metrics.

    Returns:
        ``(report, derivation)``. The derivation's ``stats`` are what the README
        quotes for corpus shape, and CI re-derives them to check the number has
        not drifted.
    """
    unique = dedupe_messages(messages)
    acl = derive_acl(unique, derivation)
    tuple_store.apply(writes=acl.tuples)
    report = run_ingest(
        documents_from_messages(unique),
        tuple_store=tuple_store,
        vector_store=vector_store,
        embedder=embedder,
        config=config,
        chunker=chunker,
    )
    return report, acl


def reindex_object(
    document: SourceDocument,
    *,
    tuple_store: TupleStore,
    vector_store: VectorStore,
    embedder: Embedder,
    config: IngestConfig | None = None,
    chunker: Chunker | None = None,
) -> int:
    """Re-stamp and re-upsert one document's chunks. Returns vectors written.

    This is FR-17. A document's own ACL changing costs exactly its chunk count in
    vector writes; a *membership* change costs zero, because membership is never
    in a token. Do not call this on a membership change — the correct response to
    one is nothing at all, and the test that proves it asserts this function's
    return value stays zero because it was never called.

    Raises:
        ValueError: If the document now compiles to zero grant tokens. Losing
            every grant means nobody may see it, and the right response is an
            explicit failure the caller handles (delete the chunks) rather than a
            silent skip that leaves the old, more permissive chunks in place.
    """
    cfg = config or IngestConfig()
    chk = chunker or Chunker(config=cfg.chunking)
    tokens = grant_tokens_for_object(tuple_store, document.object, cfg.relation)
    if not tokens:
        raise ValueError(
            f"{document.object} compiles to zero grant tokens; it is no longer "
            "visible to anyone (ADR 0003). Delete its chunks rather than "
            "reindexing them."
        )
    chunks = chk.chunk(
        document.object,
        document.text,
        grant_tokens=tokens,
        metadata=document.metadata,
        boundaries=document.boundaries,
    )
    if not chunks:
        return 0
    cache = _cache_for(cfg, embedder)
    vectors, _ = embed_all(
        embedder, [c.text for c in chunks], cache=cache, batch_size=cfg.embed_batch
    )
    vector_store.upsert(chunks, vectors)
    return len(chunks)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m sightline.ingest.pipeline`` — build an index from a corpus.

    Prints the report as JSON on stdout and progress on stderr, so
    ``... 2>/dev/null | jq`` works and CI can diff the report against the number
    committed in the README.
    """
    import argparse

    from sightline.authz.tuples import SQLiteTupleStore
    from sightline.ingest.embed import load_embedder
    from sightline.ingest.enron import (
        DEFAULT_MAX_MESSAGES,
        TARGET_CHUNKS,
        iter_enronqa,
        iter_maildir,
    )
    from sightline.store import STORE_NAMES, get_store

    parser = argparse.ArgumentParser(description="Sightline ingest")
    parser.add_argument("corpus", help="maildir root, or an EnronQA .jsonl file/dir")
    parser.add_argument("--work-dir", default=".sightline", help="cache and checkpoint")
    parser.add_argument("--tuples-db", default=None, help="SQLite tuple store path")
    parser.add_argument("--store", default="memory", choices=STORE_NAMES)
    parser.add_argument(
        "--format",
        default="auto",
        choices=("auto", "maildir", "enronqa"),
        help="auto treats a file or a directory containing .jsonl as EnronQA, "
        "anything else as a maildir. Say it explicitly when the directory holds "
        "both, because the auto rule picks EnronQA and will not mention it.",
    )
    parser.add_argument("--embedder", default="auto", choices=("auto", "onnx", "hash"))
    parser.add_argument("--max-messages", type=int, default=DEFAULT_MAX_MESSAGES)
    parser.add_argument("--max-chunks", type=int, default=TARGET_CHUNKS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    corpus = Path(args.corpus)
    fmt = args.format
    if fmt == "auto":
        fmt = "enronqa" if corpus.is_file() or any(corpus.glob("*.jsonl")) else "maildir"
    source = (
        iter_enronqa(corpus, limit=args.max_messages)
        if fmt == "enronqa"
        else iter_maildir(corpus, limit=args.max_messages)
    )
    print(f"  ingest: reading {corpus} as {fmt}", file=sys.stderr, flush=True)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    tuple_store: TupleStore = SQLiteTupleStore(args.tuples_db or work / "tuples.db")
    report, acl = ingest_messages(
        list(source),
        tuple_store=tuple_store,
        vector_store=get_store(args.store),
        embedder=load_embedder(args.embedder),
        config=IngestConfig(
            work_dir=work, max_chunks=args.max_chunks, progress=not args.quiet
        ),
    )
    print(json.dumps({"ingest": report.to_dict(), "corpus": acl.stats}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
