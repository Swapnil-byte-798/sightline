"""Ingest: how text and permissions get into the index together.

Read the modules in this order:

1. :mod:`~sightline.ingest.enron` — the connector. Real messages, and tuples
   derived from real headers. The corpus is here because its access structure
   was not invented: mailboxes are principals, To/Cc lines are grants, and the
   internal distribution lists are groups with observable membership. Every
   simplification in the derivation is named in that module's docstring.
2. :mod:`~sightline.ingest.chunk` — token-aware chunking with overlap,
   deterministic ids, and one hard rule: a chunk never spans two documents,
   because a chunk carries exactly one object's grant tokens.
3. :mod:`~sightline.ingest.embed` — ONNX Runtime and all-MiniLM-L6-v2, plus a
   deterministic hash embedder so the test suite needs no downloads. The hash
   embedder is **not semantic** and says so in three separate places.
4. :mod:`~sightline.ingest.pipeline` — orchestration, with progress and
   resumability, because this runs under a six-hour CI cap and on a laptop that
   sleeps.

THE RULE THIS PACKAGE ENFORCES
------------------------------
Grant tokens are compiled from the tuple store *per document* and stamped on its
chunks. A document that compiles to zero tokens is **rejected**, never indexed.
Empty means nobody, always (ADR 0003). Treating an empty token set as "visible
to everyone" is planted mutant M6, and the place it would be introduced is here,
so the rejection is a ``ValueError`` in :func:`~sightline.ingest.chunk.chunk_document`
rather than a filtered list comprehension somebody can delete.

Nothing in this package ever stamps a user id or a document id list onto a
chunk. Tokens come from group edges (ADR 0002), which is why a person joining or
leaving a group rewrites zero vectors.
"""

from sightline.ingest.chunk import (
    DEFAULT_MIN_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    ApproxWordTokenizer,
    Chunker,
    ChunkingConfig,
    HFTokenizer,
    SpanTokenizer,
    TextSpan,
    chunk_document,
    chunk_id_for,
    default_tokenizer,
    split_text,
)
from sightline.ingest.embed import (
    EMBED_DIM,
    HASH_EMBEDDER_WARNING,
    MODEL_ID,
    MODEL_MAX_TOKENS,
    Embedder,
    EmbeddingStats,
    HashEmbedder,
    OnnxEmbedder,
    VectorCache,
    download_model,
    embed_all,
    load_embedder,
    model_dir,
)
from sightline.ingest.enron import (
    DEFAULT_MAX_MESSAGES,
    MIN_GROUP_SIZE,
    MIN_GROUP_SUPPORT,
    TARGET_CHUNKS,
    AclDerivation,
    DerivationConfig,
    EnronMessage,
    dedupe_messages,
    derive_acl,
    document_text,
    is_distribution_list,
    iter_enronqa,
    iter_enronqa_questions,
    iter_maildir,
    normalise_address,
    principal_for_address,
    quote_boundaries,
    sample_messages,
)
from sightline.ingest.pipeline import (
    Checkpoint,
    IngestConfig,
    IngestReport,
    ProgressReporter,
    SourceDocument,
    documents_from_messages,
    ingest_messages,
    reindex_object,
    run_ingest,
)

__all__ = [
    # chunk
    "ChunkingConfig",
    "Chunker",
    "TextSpan",
    "SpanTokenizer",
    "ApproxWordTokenizer",
    "HFTokenizer",
    "default_tokenizer",
    "split_text",
    "chunk_document",
    "chunk_id_for",
    "DEFAULT_TARGET_TOKENS",
    "DEFAULT_OVERLAP_TOKENS",
    "DEFAULT_MIN_TOKENS",
    # embed
    "Embedder",
    "HashEmbedder",
    "OnnxEmbedder",
    "EmbeddingStats",
    "VectorCache",
    "embed_all",
    "load_embedder",
    "download_model",
    "model_dir",
    "EMBED_DIM",
    "MODEL_ID",
    "MODEL_MAX_TOKENS",
    "HASH_EMBEDDER_WARNING",
    # enron
    "EnronMessage",
    "AclDerivation",
    "DerivationConfig",
    "derive_acl",
    "dedupe_messages",
    "document_text",
    "quote_boundaries",
    "iter_maildir",
    "iter_enronqa",
    "iter_enronqa_questions",
    "normalise_address",
    "principal_for_address",
    "is_distribution_list",
    "sample_messages",
    "TARGET_CHUNKS",
    "DEFAULT_MAX_MESSAGES",
    "MIN_GROUP_SIZE",
    "MIN_GROUP_SUPPORT",
    # pipeline
    "SourceDocument",
    "IngestConfig",
    "IngestReport",
    "Checkpoint",
    "ProgressReporter",
    "run_ingest",
    "ingest_messages",
    "reindex_object",
    "documents_from_messages",
]
