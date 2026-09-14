"""Pure-Python reference vector store: numpy only, no services, exact cosine.

This is the backend the entire test suite and CI run against, so it is the one
that has to be *right* rather than fast. It is also the definition of correct
behaviour that the approximate backends are measured against: when a Qdrant
result disagrees with this file, this file wins the argument.

Two things matter here.

**The filter runs before scoring, not after.** :meth:`MemoryVectorStore.search`
computes the admitted row set first and scores only those rows. That is the whole
difference between this module and :mod:`sightline.store.postfilter`, and it is
why this one returns k results when k permitted chunks exist.

**Plan semantics live here, once.** :func:`plan_bindings`, :func:`admits` and
:func:`first_matching_token` are the single definition of what a
:class:`~sightline.types.FilterPlan` admits, and every other backend imports
them rather than reimplementing the rules. Two of the planted mutants (M2 —
OR where AND was meant, M6 — an empty token set matching everything) are
mistakes you make once per backend if each backend re-derives the rules.

Storage layout, and why: vectors are one contiguous float32 matrix so scoring is
a single BLAS call, and a chunk's grant tokens are a packed uint64 bitset row so
the token filter is a vectorised AND rather than 70,000 Python set
intersections. At 70k chunks by 384 dims that is ~103 MB of vectors and under
1 MB of permission bits, which fits the 8 GB reference machine with room for the
process that is asking.

Not built, deliberately: deletion (the protocol has no delete; a reindex
rewrites), persistence (this is a reference implementation, not a database), and
any epoch awareness. A store cannot know whether a plan is stale — the caller
compares ``plan.epoch`` against the live policy epoch and refuses. Pushing that
check down here would put the authority in the index, which is the one thing
this system is built not to do.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from sightline.store.base import StoreStats, UncheckedHit
from sightline.types import Chunk, FilterPlan, GrantToken, PlanStrategy

__all__ = [
    "MemoryVectorStore",
    "plan_bindings",
    "admits",
    "first_matching_token",
    "normalise",
    "normalise_batch",
]

#: Bits per word in the grant-token bitset. uint64 because numpy's bitwise ops on
#: uint64 are the widest single-instruction path available without SIMD tricks.
_WORD_BITS = 64


def plan_bindings(plan: FilterPlan) -> tuple[bool, bool]:
    """Which of a plan's two conditions are binding: ``(explicit_ids, grant_tokens)``.

    The rules, stated once so no backend has to guess:

    * ``UNFILTERED`` binds nothing. The plan compiler only emits it for a
      principal it has proved can see (nearly) the whole corpus.
    * ``ENUMERATE`` always binds the id condition, *even when the id set is
      empty*. Empty means "this principal may see no object", not "no constraint".
    * ``GRANT_TOKENS`` always binds the token condition, on the same reasoning.
      An empty token set matching everything is planted mutant M6.
    * ``EXACT_SCAN`` is a statement about execution, not about the admitted set,
      so it binds whichever conditions the plan actually carries.
    * A plan carrying *both* conditions binds both, and they combine with AND.
      Combining them with OR is planted mutant M2.

    Returns:
        A pair of booleans: whether the explicit-id condition and the grant-token
        condition must each hold for a chunk to be a candidate.
    """
    if plan.strategy is PlanStrategy.UNFILTERED:
        return (False, False)
    by_ids = plan.strategy is PlanStrategy.ENUMERATE or bool(plan.explicit_ids)
    by_tokens = plan.strategy is PlanStrategy.GRANT_TOKENS or bool(plan.grant_tokens)
    return (by_ids, by_tokens)


def _id_condition_holds(plan: FilterPlan, object_ref: str) -> bool:
    """Does ``object_ref`` appear in the plan's enumerated set?

    Two spellings are accepted: the canonical ``namespace:id`` form, and a bare
    ``id`` for entries that carry no namespace. The bare form is ambiguous by
    construction — an entry ``"42"`` admits ``doc:42`` and ``folder:42`` alike —
    so plan compilers should emit namespaced entries. It is accepted anyway
    because refusing it silently drops results, and a store that silently drops
    results is indistinguishable from a store that is broken.
    """
    ids = plan.explicit_ids
    if object_ref in ids:
        return True
    _, sep, ident = object_ref.partition(":")
    return bool(sep) and ident in ids


def admits(plan: FilterPlan, object_ref: str, grant_tokens: Iterable[str]) -> bool:
    """Would ``plan`` make this chunk a candidate?

    Advisory only, in the same sense the whole index is advisory: a ``True`` here
    means "worth scoring", never "safe to show". The answer builder still only
    accepts a ``Hit``, and only ``recheck()`` produces one.
    """
    by_ids, by_tokens = plan_bindings(plan)
    if not by_ids and not by_tokens:
        # UNFILTERED admits everything. Every other strategy that reaches this
        # branch carries no condition at all, which we read as "admits nothing":
        # losing results is recoverable, leaking them is not.
        return plan.strategy is PlanStrategy.UNFILTERED
    if by_ids and not _id_condition_holds(plan, object_ref):
        return False
    if by_tokens and not (plan.grant_tokens & frozenset(grant_tokens)):
        return False
    return True


def first_matching_token(plan: FilterPlan, grant_tokens: Iterable[str]) -> str | None:
    """The token that admitted a chunk, for ``UncheckedHit.matched_token``.

    Reported so the UI can eventually answer "why am I allowed to see this",
    and sorted so the value is stable across runs and across backends — a
    conformance suite comparing two backends' hits should not fail on set
    iteration order. Recheck does not trust this value; trusting it is planted
    mutant M4.
    """
    overlap = plan.grant_tokens & frozenset(grant_tokens)
    if not overlap:
        return None
    return min(overlap)


def normalise_batch(
    vectors: Sequence[Sequence[float]], dim: int | None = None
) -> list[np.ndarray]:
    """Unit-normalise a batch in one pass where the batch is rectangular.

    Worth the branch: normalising 70,000 vectors one call at a time costs seconds
    of pure numpy dispatch overhead on the reference machine, and ingest is the
    operation a reindex has to finish before a revocation takes effect. Ragged
    input (different lengths, which is a caller bug about to be reported) falls
    back to the per-vector path so the error names the offending row.
    """
    try:
        # np.array, not np.asarray: normalisation is in-place below, and a caller
        # who hands in their own float32 matrix should not get it rescaled under
        # them as a side effect of indexing.
        block = np.array(vectors, dtype=np.float32)
    except (ValueError, TypeError):
        block = None
    if block is None or block.ndim != 2:
        return [normalise(v, dim) for v in vectors]
    if dim is not None and block.shape[1] != dim:
        raise ValueError(f"vectors have dim {block.shape[1]}, store holds dim {dim}")
    norms = np.linalg.norm(block, axis=1)
    np.divide(block, np.where(norms > 0.0, norms, 1.0)[:, None], out=block)
    return list(block)


def normalise(vector: Sequence[float] | np.ndarray, dim: int | None = None) -> np.ndarray:
    """Return ``vector`` as a unit-length float32 row.

    Cosine similarity is a dot product once both sides are unit length, so
    normalising at ingest turns every search into one matrix multiply. A
    zero-length vector is left as zeros rather than raising: it scores 0.0
    against everything, which is the honest answer for a chunk whose embedding
    failed, and is easier to notice than a crash in an ingest loop.
    """
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if dim is not None and arr.shape[0] != dim:
        raise ValueError(f"vector has dim {arr.shape[0]}, store holds dim {dim}")
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = arr / np.float32(norm)
    return arr


class MemoryVectorStore:
    """Exact cosine search over an in-process numpy matrix, filtered first.

    Implements :class:`~sightline.store.base.VectorStore`. Honours every
    :class:`~sightline.types.PlanStrategy`:

    ``ENUMERATE``
        Gathers candidate rows from an object-ref posting list and scores only
        those. For a principal who can see 40 documents this touches 40 rows,
        not 70,000.
    ``GRANT_TOKENS``
        Vectorised bitset intersection over the packed token matrix, then scores
        the survivors.
    ``UNFILTERED``
        Skips the filter entirely and scores the whole matrix.
    ``EXACT_SCAN``
        Identical results to ``GRANT_TOKENS`` here, because this store has no
        approximate path to fall back from — every search it performs is exact.
        The strategy is still honoured rather than rejected, so that a test may
        legitimately assert strict top-k equality against the oracle on this
        path. That assertion is only ever valid on an exact path (FR-14).
    """

    def __init__(self, dim: int | None = None, *, name: str = "memory") -> None:
        """Create an empty store.

        Args:
            dim: Vector dimensionality. Inferred from the first upsert if omitted.
            name: Backend name reported in :class:`~sightline.store.base.StoreStats`.
                Overridable so wrappers (the post-filter baseline) can report
                themselves honestly.
        """
        self.name = name
        self._dim: int | None = dim
        self._n = 0
        self._cap = 0
        self._vecs: np.ndarray = np.zeros((0, dim or 0), dtype=np.float32)

        self._row_of: dict[str, int] = {}
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._object_refs: list[str] = []
        self._row_tokens: list[tuple[str, ...]] = []
        self._rows_by_object: dict[str, list[int]] = {}

        # Grant tokens as a packed bitset: one uint64 row per chunk, one bit per
        # distinct token in the corpus. Set intersection becomes `& then .any()`.
        self._token_id: dict[str, int] = {}
        self._words = 1
        self._bits: np.ndarray = np.zeros((0, 1), dtype=np.uint64)

        self._text_bytes = 0

    # ---------------------------------------------------------------- writes

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Insert or replace chunks, keyed on ``chunk.id``.

        Idempotent by contract: re-upserting the same id overwrites the row in
        place rather than appending, so a reindex of a changed document does not
        leave the old permissions behind. Getting that wrong is how a revoked
        document survives in an index.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks and {len(vectors)} vectors")
        if not chunks:
            return

        prepared = normalise_batch(vectors, self._dim)
        if self._dim is None:
            self._dim = int(prepared[0].shape[0])
            self._vecs = np.zeros((self._cap, self._dim), dtype=np.float32)
        for vec in prepared:
            if vec.shape[0] != self._dim:
                raise ValueError(f"vector has dim {vec.shape[0]}, store holds dim {self._dim}")

        self._ensure_capacity(self._n + len(chunks))
        for chunk, vec in zip(chunks, prepared):
            row = self._row_of.get(chunk.id)
            if row is None:
                row = self._n
                self._n += 1
                self._row_of[chunk.id] = row
                self._chunk_ids.append(chunk.id)
                self._texts.append(chunk.text)
                self._object_refs.append("")
                self._row_tokens.append(())
            else:
                self._retract_row(row)
                self._texts[row] = chunk.text

            object_ref = str(chunk.object)
            self._object_refs[row] = object_ref
            self._rows_by_object.setdefault(object_ref, []).append(row)
            self._text_bytes += len(chunk.text.encode("utf-8"))

            tokens = tuple(sorted(str(t) for t in chunk.grant_tokens))
            self._row_tokens[row] = tokens
            self._set_token_bits(row, tokens)

            self._vecs[row] = vec

    def _retract_row(self, row: int) -> None:
        """Undo the indexing side effects of a row before it is overwritten."""
        old_ref = self._object_refs[row]
        if old_ref:
            postings = self._rows_by_object.get(old_ref)
            if postings is not None:
                remaining = [r for r in postings if r != row]
                if remaining:
                    self._rows_by_object[old_ref] = remaining
                else:
                    del self._rows_by_object[old_ref]
        self._text_bytes -= len(self._texts[row].encode("utf-8"))
        self._bits[row] = 0

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._cap:
            return
        new_cap = max(16, self._cap * 2)
        while new_cap < needed:
            new_cap *= 2
        vecs = np.zeros((new_cap, self._dim or 0), dtype=np.float32)
        if self._n:
            vecs[: self._n] = self._vecs[: self._n]
        self._vecs = vecs
        bits = np.zeros((new_cap, self._words), dtype=np.uint64)
        if self._n:
            bits[: self._n] = self._bits[: self._n]
        self._bits = bits
        self._cap = new_cap

    def compact(self) -> int:
        """Trim the capacity slack left by geometric growth. Returns bytes freed.

        Doubling keeps ingest amortised O(1) but leaves up to half the vector
        matrix unused — at 70k by 384 that is ~90 MB of nothing on a machine with
        8 GB. A bulk ingest should call this once when it finishes. It is not
        automatic, because doing it inside ``upsert`` would turn every batch into
        a full copy.
        """
        if self._cap == self._n:
            return 0
        before = self._vecs.nbytes + self._bits.nbytes
        # .copy(), not ascontiguousarray: the slice is already C-contiguous, so
        # ascontiguousarray hands back a *view* whose .base is the oversized
        # array. Nothing would be freed and the return value would be a lie.
        self._vecs = self._vecs[: self._n].copy()
        self._bits = self._bits[: self._n].copy()
        self._cap = self._n
        return int(before - (self._vecs.nbytes + self._bits.nbytes))

    def _set_token_bits(self, row: int, tokens: Sequence[str]) -> None:
        for token in tokens:
            tid = self._token_id.get(token)
            if tid is None:
                tid = len(self._token_id)
                self._token_id[token] = tid
                self._grow_token_words(tid)
            self._bits[row, tid // _WORD_BITS] |= np.uint64(1 << (tid % _WORD_BITS))

    def _grow_token_words(self, tid: int) -> None:
        needed = tid // _WORD_BITS + 1
        if needed <= self._words:
            return
        bits = np.zeros((self._bits.shape[0], needed), dtype=np.uint64)
        bits[:, : self._words] = self._bits
        self._bits = bits
        self._words = needed

    # --------------------------------------------------------------- reads

    def search(
        self, query_vector: Sequence[float], plan: FilterPlan, k: int
    ) -> list[UncheckedHit]:
        """Top-k over the subset ``plan`` admits.

        Candidates are resolved *first*, then scored. The result is the true
        top-k of the permitted subset, so a principal with 12 permitted chunks
        and k=10 gets 10 results, not however many of the global top 10 happened
        to be theirs.

        Returns ``UncheckedHit``, never ``Hit``: these have not been re-checked
        against live permissions and are not safe to show anyone.
        """
        if k <= 0 or self._n == 0 or self._dim is None:
            return []
        query = normalise(query_vector, self._dim)

        rows = self._candidate_rows(plan)
        scores, row_of_local, n_admitted = self._score(query, rows)
        take = min(k, n_admitted)
        if take <= 0:
            return []
        if take < scores.shape[0]:
            selected = np.argpartition(-scores, take - 1)[:take]
        else:
            selected = np.arange(scores.shape[0])
        # Sort the partition by ascending index first, then stable-sort by
        # descending score: exact ties resolve to insertion order, identically on
        # every run. A conformance suite that compares two backends cannot
        # tolerate arbitrary tie order.
        #
        # "Exact tie" means bitwise-equal float32, which is narrower than it
        # sounds: BLAS blocks a matrix-vector product and handles the ragged tail
        # differently, so two rows holding the *same* vector can score one ulp
        # apart. Duplicate-vector fixtures should assert on the returned set, not
        # on the order within it.
        selected = np.sort(selected)
        selected = selected[np.argsort(-scores[selected], kind="stable")]

        hits: list[UncheckedHit] = []
        for local in selected:
            local_i = int(local)
            score = float(scores[local_i])
            if not np.isfinite(score):
                continue
            row = local_i if row_of_local is None else int(row_of_local[local_i])
            hits.append(
                UncheckedHit(
                    chunk_id=self._chunk_ids[row],
                    score=score,
                    text=self._texts[row],
                    object_ref=self._object_refs[row],
                    matched_token=first_matching_token(plan, self._row_tokens[row]),
                )
            )
        return hits

    def count_matching(self, plan: FilterPlan) -> int:
        """How many chunks the plan admits. Drives strategy selection upstream."""
        rows = self._candidate_rows(plan)
        return self._n if rows is None else int(rows.size)

    def stats(self) -> StoreStats:
        resident = int(self._vecs.nbytes + self._bits.nbytes) + self._text_bytes
        return StoreStats(
            n_vectors=self._n,
            resident_bytes=resident,
            disk_bytes=0,
            backend=self.name,
            detail={
                "exact": "true",
                "dim": str(self._dim or 0),
                "distinct_grant_tokens": str(len(self._token_id)),
                "token_bitset_words": str(self._words),
                "distinct_objects": str(len(self._rows_by_object)),
                # Reported, not hidden: geometric growth leaves slack, and
                # compact() is how a finished ingest gives it back.
                "capacity_slack_bytes": str(
                    int((self._cap - self._n) * (self._vecs.shape[1] * 4 + self._words * 8))
                ),
                "note": "filter applied before scoring; no approximate path",
            },
        )

    def __len__(self) -> int:
        return self._n

    # ------------------------------------------------- accessors for the oracle
    #
    # The exact oracle in `faiss_oracle` keeps this store as its bookkeeping
    # layer and mirrors only the vectors into FAISS, so it needs read access to
    # the matrix, the row metadata and the compiled candidate set. Three narrow
    # public methods beat one module reaching into another's underscores.

    def matrix(self) -> np.ndarray:
        """Read-only view of the live vector rows, shape ``(len(self), dim)``."""
        view = self._vecs[: self._n]
        view.flags.writeable = False
        return view

    def row_info(self, row: int) -> tuple[str, str, str, tuple[str, ...]]:
        """``(chunk_id, object_ref, text, grant_tokens)`` for one row."""
        return (
            self._chunk_ids[row],
            self._object_refs[row],
            self._texts[row],
            self._row_tokens[row],
        )

    def candidate_rows(self, plan: FilterPlan) -> np.ndarray | None:
        """Rows a plan admits, or ``None`` meaning every row (``UNFILTERED``)."""
        return self._candidate_rows(plan)

    def dim(self) -> int | None:
        """Vector dimensionality, or ``None`` before the first upsert."""
        return self._dim

    # ------------------------------------------------------------- internals

    def _candidate_rows(self, plan: FilterPlan) -> np.ndarray | None:
        """Rows the plan admits. ``None`` means every row (UNFILTERED).

        ``None`` rather than ``arange(n)`` because materialising 70,000 indices
        to mean "no filter" costs a copy of the whole vector matrix downstream.
        """
        by_ids, by_tokens = plan_bindings(plan)
        empty = np.empty(0, dtype=np.int64)
        if not by_ids and not by_tokens:
            if plan.strategy is PlanStrategy.UNFILTERED:
                return None
            return empty  # no condition and not unfiltered: admit nothing.

        if by_ids:
            rows = self._rows_for_ids(plan.explicit_ids)
            if by_tokens and rows.size:
                rows = rows[self._token_mask(rows, plan.grant_tokens)]
            return rows

        mask = self._token_mask(None, plan.grant_tokens)
        return np.nonzero(mask)[0].astype(np.int64, copy=False)

    def _rows_for_ids(self, explicit_ids: frozenset[str]) -> np.ndarray:
        if not explicit_ids:
            return np.empty(0, dtype=np.int64)
        rows: list[int] = []
        for entry in explicit_ids:
            if ":" in entry:
                rows.extend(self._rows_by_object.get(entry, ()))
            else:
                # Bare id: admit it under any namespace. See _id_condition_holds.
                for ref, postings in self._rows_by_object.items():
                    if ref.partition(":")[2] == entry:
                        rows.extend(postings)
        if not rows:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.asarray(rows, dtype=np.int64))

    def _token_mask(
        self, rows: np.ndarray | None, tokens: frozenset[GrantToken]
    ) -> np.ndarray:
        """Boolean mask over ``rows`` (or over every row) of token-set overlap."""
        width = self._n if rows is None else int(rows.size)
        plan_bits = self._plan_bits(tokens)
        if plan_bits is None:
            return np.zeros(width, dtype=bool)
        block = self._bits[: self._n] if rows is None else self._bits[rows]
        return np.bitwise_and(block, plan_bits).any(axis=1)

    def _plan_bits(self, tokens: frozenset[GrantToken]) -> np.ndarray | None:
        """Pack a plan's tokens into this store's bit layout.

        ``None`` when the plan holds no token this corpus has ever seen — which
        includes the empty token set. Both mean zero candidates.
        """
        if not tokens:
            return None
        bits = np.zeros(self._words, dtype=np.uint64)
        hit = False
        for token in tokens:
            tid = self._token_id.get(str(token))
            if tid is None:
                continue  # a token no chunk carries cannot admit a chunk.
            bits[tid // _WORD_BITS] |= np.uint64(1 << (tid % _WORD_BITS))
            hit = True
        return bits if hit else None

    def _score(
        self, query: np.ndarray, rows: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None, int]:
        """Cosine scores, the local-to-row mapping, and how many rows are admitted.

        Two shapes, chosen by size. A small candidate set is gathered into a
        compact matrix and scored directly. A large one is scored in place over
        the full matrix with non-candidates set to ``-inf``, because gathering
        50,000 rows of 384 float32s costs 76 MB of copy on a machine with 8 GB,
        and one BLAS call over the full matrix is cheaper than the copy.

        The admitted count is returned rather than recovered later with an
        ``isfinite`` pass: it is already known exactly on every branch, and the
        pass would be an extra O(n) scan on the hot path for no information.
        """
        n = self._n
        if rows is None:
            return self._vecs[:n] @ query, None, n
        if rows.size == 0:
            return np.empty(0, dtype=np.float32), rows, 0
        if rows.size * 2 > n:
            scores = self._vecs[:n] @ query
            keep = np.zeros(n, dtype=bool)
            keep[rows] = True
            scores = np.where(keep, scores, -np.inf).astype(np.float32, copy=False)
            return scores, None, int(rows.size)
        return self._vecs[rows] @ query, rows, int(rows.size)
