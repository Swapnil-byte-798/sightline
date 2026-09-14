"""Embedding: text to 384 float32s, on a laptop with no GPU and no torch.

ONNX RUNTIME, NEVER TORCH
------------------------
The reference machine is a 2015 Intel MacBook Pro running macOS x86_64, where
PyTorch has shipped no wheels past 2.2.x. Any dependency that pulls torch makes
this repo unbuildable on the machine it was written on, which is a good enough
reason on its own. The better reason is that ``onnxruntime`` is 40 MB, starts in
under a second and does not need a CUDA story to explain itself. Sentence
embeddings are a fixed-graph forward pass; a training framework is the wrong
tool for it.

The model is ``sentence-transformers/all-MiniLM-L6-v2``: 384 dimensions, 256
token window, 6 layers. It is not the best retrieval model available and it is
not claimed to be. It is the one that runs at a usable rate on two 2015 cores.

THE FALLBACK
------------
:class:`HashEmbedder` exists so the test suite runs with zero downloads. It is
**not semantic**. Read its docstring before using it for anything.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
* **Query/passage instruction prefixes.** MiniLM was not trained with them; some
  newer models are, and bolting a prefix onto a model that never saw one makes
  the numbers worse while looking like tuning.
* **Matryoshka truncation to 256 or 128 dims.** Worth measuring, not worth
  asserting. The store is dimension-agnostic, so it can be added behind an eval.
* **A GPU path.** There is no GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "EMBED_DIM",
    "HASH_EMBEDDER_WARNING",
    "MODEL_ID",
    "MODEL_MAX_TOKENS",
    "REQUIRED_MODEL_FILES",
    "Embedder",
    "EmbeddingStats",
    "HashEmbedder",
    "OnnxEmbedder",
    "VectorCache",
    "download_model",
    "embed_all",
    "load_embedder",
    "model_dir",
]

#: all-MiniLM-L6-v2 output width. Every store in this repo is built around it.
EMBED_DIM = 384

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

#: The model's trained sequence length. Text past this is dropped by the model,
#: not by us, which is why ``chunk.py`` targets 256 tokens and not 512.
MODEL_MAX_TOKENS = 256

_HF_BASE = "https://huggingface.co/{model}/resolve/main/{path}"

#: What a local model directory must contain to be usable.
REQUIRED_MODEL_FILES = ("tokenizer.json", "model.onnx")

# Quantised exports are published under several names and the set changes
# between repo revisions, so probing beats hardcoding one filename that 404s six
# months from now. Note the omission: no avx512 variant is listed, because the
# reference machine is Broadwell and does not have avx512 — an int8 model built
# for it would either refuse to load or run slower than fp32. Whether int8 is
# actually faster here is a measurement `eval/` makes, not an assumption.
_INT8_CANDIDATES = ("model_int8.onnx", "model_quantized.onnx", "model_qint8_arm64.onnx")

HASH_EMBEDDER_WARNING = (
    "HashEmbedder is in use. It is a hashing trick, not a language model: it has "
    "no notion of meaning, and 'car' and 'automobile' are as unrelated to it as "
    "'car' and 'ostrich'. Retrieval numbers measured with it are measurements of "
    "lexical overlap and MUST NOT be published as retrieval quality."
)

_TOKEN_SPLIT_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Text to unit vectors.

    Every implementation returns **L2-normalised** rows, because every store in
    this repo scores with a dot product and calls it cosine. Normalising at the
    edge means no store has to remember to.
    """

    name: str
    dim: int
    #: ``False`` marks an embedder whose output does not encode meaning. The eval
    #: harness refuses to publish a number produced by one, and this flag is how
    #: it knows. A boolean is checkable; a docstring warning is not.
    is_semantic: bool

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), dim)`` float32, L2-normalised, rows aligned to input."""
        ...


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    """Unit rows. A zero row stays zero rather than becoming NaN.

    A zero vector scores 0.0 against every query, which is the honest answer for
    a chunk with no signal in it. NaN would poison a top-k comparison instead.
    """
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (mat / norms).astype(np.float32, copy=False)


# --------------------------------------------------------------------------
# The fallback
# --------------------------------------------------------------------------


class HashEmbedder:
    """DETERMINISTIC HASHING. NOT SEMANTIC. NOT FOR PUBLISHED NUMBERS.

    This is the feature-hashing trick over word unigrams and bigrams: each token
    is hashed to a dimension and a sign, and the vector is the signed sum. Two
    documents that share words land near each other. Two documents that mean the
    same thing in different words land nowhere near each other, because nothing
    in this class has ever seen a sentence.

    It exists for exactly one reason: the test suite must run on a fresh checkout
    with no network, no model download and no optional extras, and a permission
    test that needs a 90 MB download is a permission test that gets skipped.

    **The specific mistake this is guarding against.** A previous repo of mine
    shipped a fallback embedder just like this one, benchmarked against it
    because the real model was slow to set up, and published the recall numbers.
    They were numbers about string overlap wearing the costume of semantic
    search. Hence three guards here, all of them deliberately annoying:

    * ``is_semantic = False``, which the eval harness checks before writing any
      result file;
    * a ``RuntimeWarning`` on first construction, once per process;
    * ``name`` starts with ``hash-`` so it shows up in every stats line, cache
      key and collection name that records which embedder produced a vector.

    Do not remove any of the three.
    """

    is_semantic = False
    _warned = False

    def __init__(self, dim: int = EMBED_DIM, *, seed: int = 0, quiet: bool = False) -> None:
        self.dim = dim
        self.seed = seed
        self.name = f"hash-{dim}-s{seed}"
        if not quiet and not HashEmbedder._warned:
            HashEmbedder._warned = True
            warnings.warn(HASH_EMBEDDER_WARNING, RuntimeWarning, stacklevel=2)

    def _units(self, text: str) -> Iterator[str]:
        words = _TOKEN_SPLIT_RE.findall(text.lower())
        yield from words
        for a, b in zip(words, words[1:], strict=False):  # bigrams: ragged by design
            yield f"{a}_{b}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        prefix = self.seed.to_bytes(4, "little")
        for row, text in enumerate(texts):
            vec = out[row]
            for unit in self._units(text):
                digest = hashlib.blake2b(
                    prefix + unit.encode("utf-8"), digest_size=8
                ).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vec[idx] += sign
        return _l2_normalise(out)


# --------------------------------------------------------------------------
# The real one
# --------------------------------------------------------------------------


def model_dir() -> Path:
    """Where model files live. ``SIGHTLINE_MODEL_DIR`` overrides.

    Defaults under the user cache rather than inside the repo, so a checkout
    stays small and ``git status`` stays readable.
    """
    raw = os.environ.get("SIGHTLINE_MODEL_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "sightline" / "models" / MODEL_ID.split("/")[-1]


def download_model(
    dest: Path | None = None,
    *,
    include_int8: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Fetch the ONNX export and tokenizer from the Hugging Face CDN.

    Explicit, never automatic. An ingest run that quietly pulls 90 MB the first
    time it is invoked is an ingest run that fails in CI for reasons nobody can
    read off the log.

    Args:
        dest: Target directory. Defaults to :func:`model_dir`.
        include_int8: Also try the quantised exports. Failures here are not
            fatal: the filenames differ across repo revisions, and fp32 works.
        progress: Called with one human-readable line per file.

    Returns:
        The directory the files landed in.
    """
    import httpx

    target = dest or model_dir()
    target.mkdir(parents=True, exist_ok=True)
    wanted = [("tokenizer.json", "tokenizer.json"), ("onnx/model.onnx", "model.onnx")]
    if include_int8:
        wanted += [(f"onnx/{n}", n) for n in _INT8_CANDIDATES]

    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        for remote, local in wanted:
            path = target / local
            if path.exists() and path.stat().st_size > 0:
                if progress:
                    progress(f"have {local}")
                continue
            url = _HF_BASE.format(model=MODEL_ID, path=remote)
            resp = client.get(url)
            if resp.status_code != 200:
                if local in REQUIRED_MODEL_FILES:
                    raise RuntimeError(
                        f"could not fetch required model file {remote} "
                        f"({resp.status_code}) from {url}"
                    )
                if progress:
                    progress(f"skip {local} (not published at this revision)")
                continue
            path.write_bytes(resp.content)
            if progress:
                progress(f"got  {local} ({len(resp.content) / 1e6:.1f} MB)")
    return target


def _resolve_model_file(directory: Path, prefer_int8: bool) -> Path:
    if prefer_int8:
        for name in _INT8_CANDIDATES:
            candidate = directory / name
            if candidate.exists():
                return candidate
    fp32 = directory / "model.onnx"
    if fp32.exists():
        return fp32
    raise FileNotFoundError(
        f"no ONNX model in {directory}. Expected one of "
        f"{('model.onnx', *_INT8_CANDIDATES)}. Fetch them with "
        "`python -m sightline.ingest.embed --download`, or point "
        "SIGHTLINE_MODEL_DIR at a directory that has them."
    )


class OnnxEmbedder:
    """all-MiniLM-L6-v2 through onnxruntime, with mean pooling.

    Mean pooling over the attention mask, then L2 normalise: that is what
    sentence-transformers does for this checkpoint, and using the ``[CLS]``
    vector instead — which is what you get if you grab the second output tensor
    without checking — produces embeddings that are subtly, unfixably worse
    while still looking like embeddings.

    Threads default to the physical core count minus nothing clever, because the
    reference machine has two cores and any thread-tuning heuristic written for
    a 64-core server makes it slower.
    """

    is_semantic = True

    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        prefer_int8: bool = True,
        max_tokens: int = MODEL_MAX_TOKENS,
        intra_op_threads: int | None = None,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
            from tokenizers import Tokenizer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ImportError(
                "OnnxEmbedder needs onnxruntime and tokenizers. "
                "Install them with: pip install 'sightline[embed]'. "
                "Do not install torch; this repo does not use it and the "
                "reference machine (macOS x86_64) has no wheels for it."
            ) from exc

        self.directory = Path(directory) if directory is not None else model_dir()
        tok_path = self.directory / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(
                f"no tokenizer.json in {self.directory}. Fetch the model with "
                "`python -m sightline.ingest.embed --download`."
            )
        self.model_path = _resolve_model_file(self.directory, prefer_int8)
        self.max_tokens = max_tokens
        self.dim = EMBED_DIM
        self.quantised = self.model_path.name != "model.onnx"
        self.name = f"minilm-l6-v2-{'int8' if self.quantised else 'fp32'}"

        self._tok = Tokenizer.from_file(str(tok_path))
        self._tok.enable_truncation(max_length=max_tokens)
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            opts.intra_op_num_threads = intra_op_threads
        self._sess = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        # Exports disagree about whether token_type_ids is an input. Feed what
        # the graph actually declares instead of what the tutorial says.
        self._input_names = {i.name for i in self._sess.get_inputs()}
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        encs = self._tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = ids
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = mask
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)

        # onnxruntime sessions are thread-safe, but the tokenizer's padding
        # state is not, and the batch shape is shared mutable state on it.
        with self._lock:
            outputs = self._sess.run(None, feed)

        hidden = next((o for o in outputs if getattr(o, "ndim", 0) == 3), None)
        if hidden is None:  # pragma: no cover - would mean a different export
            raise RuntimeError(
                f"{self.model_path.name} produced no [batch, seq, dim] tensor; "
                "this is not an all-MiniLM-L6-v2 sentence-encoder export."
            )
        m = mask.astype(np.float32)[:, :, None]
        summed = (hidden.astype(np.float32) * m).sum(axis=1)
        counts = np.maximum(m.sum(axis=1), 1e-9)
        return _l2_normalise(summed / counts)


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------


class VectorCache:
    """Content-addressed vector cache: one blob, one index, append only.

    Keyed on ``blake2b(embedder name | dim | text)``, so re-chunking a document
    whose text did not change costs no model time, and a run killed by a closing
    laptop lid resumes without re-embedding what it already did.

    **Layout.** One ``vectors.f32`` of raw little-endian float32 rows and one
    ``index.jsonl`` of ``{"k": key, "r": row}``. Not one file per vector: 70,000
    small files on HFS+ costs more in directory metadata than the vectors
    themselves, and deleting them takes minutes.

    **The cost of append-only.** Re-embedding the same key (a model upgrade under
    an unchanged name, say) appends a second row and leaks the first. The index
    keeps the newest, so reads are correct and the file is merely fat.
    :meth:`compact` rewrites it. Nothing calls compact automatically, because a
    rewrite that runs during an ingest is a rewrite that can be interrupted
    halfway.
    """

    def __init__(self, directory: Path | str, dim: int = EMBED_DIM) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._blob_path = self.directory / "vectors.f32"
        self._index_path = self.directory / "index.jsonl"
        self._index: dict[str, int] = {}
        self._rows = 0
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self._blob_path.exists():
            size = self._blob_path.stat().st_size
            self._rows = size // (self.dim * 4)
        if not self._index_path.exists():
            return
        with self._index_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A torn last line is what an interrupted run leaves behind.
                    # Dropping it costs one re-embed; refusing to start costs the
                    # whole run.
                    continue
                row = int(rec["r"])
                if row < self._rows:
                    self._index[str(rec["k"])] = row

    @staticmethod
    def key(embedder_name: str, dim: int, text: str) -> str:
        payload = f"{embedder_name}\x00{dim}\x00{text}".encode()
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def __len__(self) -> int:
        return len(self._index)

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        """Look up several keys with one file read. Missing keys are absent."""
        rows = {k: self._index[k] for k in keys if k in self._index}
        if not rows:
            return {}
        with self._lock, self._blob_path.open("rb") as fh:
            out: dict[str, np.ndarray] = {}
            for k, row in sorted(rows.items(), key=lambda kv: kv[1]):
                fh.seek(row * self.dim * 4)
                buf = fh.read(self.dim * 4)
                if len(buf) < self.dim * 4:
                    continue
                out[k] = np.frombuffer(buf, dtype="<f4").copy()
            return out

    def put_many(self, items: Sequence[tuple[str, np.ndarray]]) -> None:
        """Append vectors and their index entries, then flush both.

        Blob first, index second. If the process dies between them the vector is
        orphaned and gets recomputed — wasted work. The other order would index a
        row that does not exist and hand back garbage, which is not wasted work,
        it is wrong answers.
        """
        if not items:
            return
        with self._lock:
            with self._blob_path.open("ab") as blob:
                for _, vec in items:
                    arr = np.asarray(vec, dtype="<f4").reshape(-1)
                    if arr.size != self.dim:
                        raise ValueError(
                            f"cache dim mismatch: got {arr.size}, expected {self.dim}"
                        )
                    blob.write(arr.tobytes())
                blob.flush()
                os.fsync(blob.fileno())
            with self._index_path.open("a", encoding="utf-8") as idx:
                for k, _ in items:
                    idx.write(json.dumps({"k": k, "r": self._rows}) + "\n")
                    self._index[k] = self._rows
                    self._rows += 1
                idx.flush()
                os.fsync(idx.fileno())

    def compact(self) -> int:
        """Rewrite blob and index keeping only live rows. Returns rows dropped."""
        with self._lock:
            live = sorted(self._index.items(), key=lambda kv: kv[1])
            if not live:
                return 0
            tmp_blob = self._blob_path.with_suffix(".f32.tmp")
            tmp_index = self._index_path.with_suffix(".jsonl.tmp")
            dropped = self._rows - len(live)
            with (
                self._blob_path.open("rb") as src,
                tmp_blob.open("wb") as dst,
                tmp_index.open("w", encoding="utf-8") as idx,
            ):
                for new_row, (k, old_row) in enumerate(live):
                    src.seek(old_row * self.dim * 4)
                    dst.write(src.read(self.dim * 4))
                    idx.write(json.dumps({"k": k, "r": new_row}) + "\n")
            tmp_blob.replace(self._blob_path)
            tmp_index.replace(self._index_path)
            self._index = {k: i for i, (k, _) in enumerate(live)}
            self._rows = len(live)
            return dropped


# --------------------------------------------------------------------------
# Batched embedding with measured throughput
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmbeddingStats:
    """What the embed step actually cost. Reported, never estimated.

    ``n_computed`` counts texts that actually went through a forward pass.
    Copies served from the disk cache are ``n_cached``; copies served from
    another identical string in the same call are ``n_deduplicated``. Three
    counters rather than one because ``texts_per_second`` is divided by
    ``n_computed`` only: a run reporting 40,000 texts/sec because the corpus is
    full of repeated signature blocks is a throughput number about a dictionary,
    published as a number about a model.
    """

    embedder: str
    is_semantic: bool
    n_texts: int
    n_cached: int
    n_computed: int
    n_chars: int
    seconds: float
    batch_size: int
    n_deduplicated: int = 0

    @property
    def texts_per_second(self) -> float:
        return self.n_computed / self.seconds if self.seconds > 0 else 0.0

    @property
    def chars_per_second(self) -> float:
        return self.n_chars / self.seconds if self.seconds > 0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.n_cached / self.n_texts if self.n_texts else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "embedder": self.embedder,
            "is_semantic": self.is_semantic,
            "n_texts": self.n_texts,
            "n_cached": self.n_cached,
            "n_computed": self.n_computed,
            "n_deduplicated": self.n_deduplicated,
            "seconds": round(self.seconds, 3),
            "texts_per_second": round(self.texts_per_second, 1),
            "chars_per_second": round(self.chars_per_second, 1),
            "batch_size": self.batch_size,
        }

    def __str__(self) -> str:
        flag = "" if self.is_semantic else "  [NOT SEMANTIC]"
        return (
            f"{self.embedder}: {self.n_computed} computed, {self.n_cached} cached, "
            f"{self.n_deduplicated} deduplicated, "
            f"{self.texts_per_second:.1f} texts/s over {self.seconds:.1f}s{flag}"
        )


def embed_all(
    embedder: Embedder,
    texts: Sequence[str],
    *,
    cache: VectorCache | None = None,
    batch_size: int = 32,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, EmbeddingStats]:
    """Embed a list of texts, using the cache for anything already computed.

    Args:
        embedder: Any :class:`Embedder`.
        texts: Inputs. Duplicates are embedded once and copied, which matters:
            email corpora are full of identical signature blocks and quoted
            replies, and on Enron this is worth roughly a fifth of the run.
        cache: Optional disk cache. Reads and writes go through it.
        batch_size: Rows per forward pass. 32 fits comfortably in 8 GB at 256
            tokens; larger batches stop helping on two cores.
        progress: Called as ``(done, total)`` after each batch.

    Returns:
        ``(vectors, stats)`` with one row per input, in input order.
    """
    n = len(texts)
    dim = embedder.dim
    out = np.zeros((n, dim), dtype=np.float32)
    if n == 0:
        return out, EmbeddingStats(embedder.name, embedder.is_semantic, 0, 0, 0, 0, 0.0, batch_size)

    # Deduplicate first: identical text has identical embedding by definition.
    unique: dict[str, list[int]] = {}
    for i, t in enumerate(texts):
        unique.setdefault(t, []).append(i)
    uniq_texts = list(unique.keys())

    keys = [VectorCache.key(embedder.name, dim, t) for t in uniq_texts]
    hits: dict[str, np.ndarray] = cache.get_many(keys) if cache is not None else {}

    pending: list[int] = []
    n_cached = 0
    for u, (text, key) in enumerate(zip(uniq_texts, keys, strict=True)):
        vec = hits.get(key)
        if vec is None:
            pending.append(u)
            continue
        for i in unique[text]:
            out[i] = vec
        n_cached += len(unique[text])

    n_chars = 0
    elapsed = 0.0
    done = 0
    n_computed = 0
    for start in range(0, len(pending), batch_size):
        batch_idx = pending[start : start + batch_size]
        batch = [uniq_texts[u] for u in batch_idx]
        t0 = time.perf_counter()
        vectors = embedder.encode(batch)
        elapsed += time.perf_counter() - t0
        if vectors.shape != (len(batch), dim):
            raise ValueError(
                f"{embedder.name}.encode returned {vectors.shape}, "
                f"expected {(len(batch), dim)}"
            )
        for u, text, vec in zip(batch_idx, batch, vectors, strict=True):
            for i in unique[text]:
                out[i] = vec
            n_chars += len(text)
            n_computed += 1
        if cache is not None:
            cache.put_many([(keys[u], v) for u, v in zip(batch_idx, vectors, strict=True)])
        done += len(batch)
        if progress is not None:
            progress(done, len(pending))

    stats = EmbeddingStats(
        embedder=embedder.name,
        is_semantic=embedder.is_semantic,
        n_texts=n,
        n_cached=n_cached,
        n_computed=n_computed,
        n_chars=n_chars,
        seconds=elapsed,
        batch_size=batch_size,
        n_deduplicated=n - n_cached - n_computed,
    )
    return out, stats


def load_embedder(
    kind: str = "auto",
    *,
    directory: Path | str | None = None,
    prefer_int8: bool = True,
    quiet: bool = False,
) -> Embedder:
    """Construct an embedder by name.

    Args:
        kind: ``"onnx"`` (raises if unavailable), ``"hash"`` (always works,
            never semantic), or ``"auto"``.
        directory: Model directory for the ONNX path.
        prefer_int8: Use a quantised export if one is present.
        quiet: Suppress the hash-embedder warning. For tests that assert on it.

    Returns:
        An :class:`Embedder`.

    Note:
        ``"auto"`` falls back to :class:`HashEmbedder` when onnxruntime or the
        model files are missing, and it is loud about it. Falling back silently
        is exactly how a repo ends up publishing hash-embedder recall.
    """
    kind = kind.lower()
    if kind == "hash":
        return HashEmbedder(quiet=quiet)
    if kind == "onnx":
        return OnnxEmbedder(directory, prefer_int8=prefer_int8)
    if kind != "auto":
        raise ValueError(f"unknown embedder kind: {kind!r} (want 'onnx', 'hash' or 'auto')")
    try:
        return OnnxEmbedder(directory, prefer_int8=prefer_int8)
    except (ImportError, FileNotFoundError) as exc:
        warnings.warn(
            f"falling back to HashEmbedder: {exc}. {HASH_EMBEDDER_WARNING}",
            RuntimeWarning,
            stacklevel=2,
        )
        return HashEmbedder(quiet=quiet)


def _measure(embedder: Embedder, texts: Iterable[str], batch_size: int) -> EmbeddingStats:
    vecs, stats = embed_all(embedder, list(texts), batch_size=batch_size)
    del vecs
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m sightline.ingest.embed`` — download the model, or benchmark it.

    Deliberately tiny. Ingest has its own entry point; this one exists so the
    model fetch is a single obvious command in the README rather than a
    side effect of the first real run.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Sightline embedder utilities")
    parser.add_argument("--download", action="store_true", help="fetch model files")
    parser.add_argument("--bench", type=int, default=0, help="embed N synthetic texts")
    parser.add_argument("--kind", default="auto", choices=("auto", "onnx", "hash"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.download:
        target = download_model(progress=print)
        print(f"model directory: {target}")
    if args.bench:
        emb = load_embedder(args.kind)
        corpus = [
            f"Message {i}: the quarterly gas nomination schedule was revised "
            f"after the pipeline outage and needs counterparty sign-off."
            for i in range(args.bench)
        ]
        print(_measure(emb, corpus, args.batch_size))
    if not args.download and not args.bench:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
