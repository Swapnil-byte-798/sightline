"""The wrong way to do permissioned retrieval, implemented on purpose.

READ THIS BEFORE USING ANYTHING IN THIS MODULE
----------------------------------------------
:class:`PostFilterStore` retrieves the global top-k **ignoring permissions**, and
only then drops the results the principal may not see. It returns fewer than k
results, and the shortfall grows as the principal's share of the corpus shrinks.

This is not a bug to be fixed. It is the **baseline arm** of the evaluation, and
producing its recall collapse is the reason this project exists. It is also,
almost verbatim, what most RAG tutorials ship and what a competent engineer
writes on the first afternoon: retrieve, then filter, because filtering a list is
obvious and pushing a filter into an ANN graph is not.

Why it collapses. A principal who may see 1% of a 1M-chunk corpus asks for k=10.
The global top 10 contains, in expectation, 0.1 permitted chunks. So the answer
is built from zero or one pieces of evidence, while thousands of relevant,
permitted chunks sit unretrieved. The user sees "I don't know" and concludes the
AI is useless; nobody sees the actual cause. Correctness of the *filter* is not
the problem here — this store never returns a forbidden chunk. Recall is the
problem, and recall failures are silent.

The tempting patch is ``overfetch``: retrieve 10k candidates and hope 10 survive.
It is supported here (see the constructor) precisely so the evaluation can show
what it buys — the collapse moves right, it does not go away, and the latency
cost is linear in the overfetch factor while the recall gain is not.

What this store must never do: pad the survivors back up to k from the
unfiltered pool. That turns a silent recall failure into an actual leak, and it
is planted mutant **M7**. There is no code path here that can do it.

Guard. Importing this module inside a serving process raises. The signal is
``uvicorn`` in ``sys.modules``: pytest imports the FastAPI app but never
uvicorn, so the guard fires in a real server and stays quiet in CI. Set
``SIGHTLINE_ALLOW_POSTFILTER=1`` to override, which exists so the override is a
deliberate, greppable act rather than a default.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace

from sightline.store.base import StoreStats, UncheckedHit, VectorStore
from sightline.store.memory import MemoryVectorStore, admits, first_matching_token
from sightline.types import Chunk, FilterPlan, PlanStrategy

__all__ = [
    "ALLOW_ENV_VAR",
    "NOT_FOR_PRODUCTION",
    "PostFilterStore",
    "PostFilterTrace",
    "assert_not_serving",
]

#: Read by the docs build and by a test that asserts this module is labelled.
NOT_FOR_PRODUCTION = True

ALLOW_ENV_VAR = "SIGHTLINE_ALLOW_POSTFILTER"

#: Modules whose presence means "this process is serving traffic". uvicorn is the
#: right signal because the test suite drives FastAPI through its ASGI transport
#: and never imports a server.
_SERVING_SIGNALS = ("uvicorn",)


def assert_not_serving(context: str = "sightline.store.postfilter") -> None:
    """Raise if the baseline arm is being loaded inside a serving process.

    Args:
        context: What is being guarded, quoted in the error.

    Raises:
        RuntimeError: If a server runtime is loaded and the override is unset.
    """
    if os.environ.get(ALLOW_ENV_VAR, "") not in ("", "0", "false", "False"):
        return
    present = [name for name in _SERVING_SIGNALS if name in sys.modules]
    if present:
        raise RuntimeError(
            f"{context} is the deliberately-wrong baseline arm and must never serve "
            f"traffic; it was loaded in a process that has imported {present[0]}. "
            f"Use sightline.store.memory or the Qdrant backend. To run the evaluation "
            f"inside such a process anyway, set {ALLOW_ENV_VAR}=1 and explain why in "
            f"the commit message."
        )


assert_not_serving()


@dataclass(frozen=True, slots=True)
class PostFilterTrace:
    """What one post-filtered search actually did, for the evaluation harness.

    ``returned`` is what the caller got; ``requested`` is what it asked for. The
    gap between them, plotted against permission density, is the headline chart.
    """

    requested_k: int
    retrieved: int
    returned: int
    dropped: int

    @property
    def shortfall(self) -> int:
        """How many results the caller asked for and did not get."""
        return max(0, self.requested_k - self.returned)


class PostFilterStore:
    """Global top-k, then drop the forbidden ones. The baseline arm.

    Implements :class:`~sightline.store.base.VectorStore` so the conformance
    suite can run against it unmodified — it is a *correct* store in the sense
    that it never returns a chunk the plan forbids. It is an incorrect store in
    the sense that matters: its results are not the top-k of the permitted
    subset.
    """

    def __init__(
        self,
        delegate: VectorStore | None = None,
        *,
        dim: int | None = None,
        overfetch: int = 1,
        name: str = "postfilter",
    ) -> None:
        """Wrap a store and throw away its ability to filter during search.

        Args:
            delegate: The store doing the actual vector work. Defaults to an
                in-process :class:`~sightline.store.memory.MemoryVectorStore`,
                whose filtering this class then declines to use.
            dim: Passed to the default delegate.
            overfetch: Retrieve ``overfetch * k`` global candidates before
                dropping. 1 is the honest tutorial baseline. Values above 1 are
                the usual production patch and are supported so the evaluation
                can measure what they do and do not fix.
            name: Backend name reported in stats.

        Raises:
            ValueError: If ``overfetch`` is below 1.
        """
        assert_not_serving(f"{type(self).__name__}()")
        if overfetch < 1:
            raise ValueError(f"overfetch must be >= 1, got {overfetch}")
        self.name = name
        self.overfetch = overfetch
        self._delegate: VectorStore = delegate or MemoryVectorStore(dim)
        # The delegate returns UncheckedHit, which carries no grant tokens, so
        # the drop step needs its own copy of the permission payload. That extra
        # bookkeeping is itself an argument against post-filtering: the filter
        # has to live somewhere, and here it lives in the application.
        self._tokens: dict[str, tuple[str, ...]] = {}
        self.last_trace: PostFilterTrace | None = None

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        self._delegate.upsert(chunks, vectors)
        for chunk in chunks:
            self._tokens[chunk.id] = tuple(sorted(str(t) for t in chunk.grant_tokens))

    def search(
        self, query_vector: Sequence[float], plan: FilterPlan, k: int
    ) -> list[UncheckedHit]:
        """Retrieve globally, then drop. Returns **at most** k, often far fewer.

        The plan is used only to decide what to throw away, never to decide what
        to retrieve. That one line is the entire difference from a correct store,
        and it is worth a few hundred million dollars of stalled AI rollouts.
        """
        if k <= 0:
            self.last_trace = PostFilterTrace(k, 0, 0, 0)
            return []

        # Deliberately blind retrieval. The principal is preserved so nothing
        # downstream can mistake this for a different user's query, but the
        # strategy is forced to UNFILTERED and both conditions are stripped.
        blind = FilterPlan(
            principal=plan.principal,
            strategy=PlanStrategy.UNFILTERED,
            epoch=plan.epoch,
            estimated_cardinality=plan.estimated_cardinality,
        )
        candidates = self._delegate.search(query_vector, blind, k * self.overfetch)

        survivors: list[UncheckedHit] = []
        for hit in candidates:
            tokens = self._tokens.get(hit.chunk_id, ())
            if not admits(plan, hit.object_ref, tokens):
                continue
            # The delegate stamped matched_token against the blind plan, where it
            # is meaningless. Restamp it from the real plan: this store is wrong
            # about recall, and only about recall. Leaving the advisory field
            # broken too would let a conformance failure be blamed on the wrong
            # defect.
            survivors.append(replace(hit, matched_token=first_matching_token(plan, tokens)))
        # Truncate, never pad. There is no branch below this line that reaches
        # back into `candidates` (planted mutant M7).
        kept = survivors[:k]
        self.last_trace = PostFilterTrace(
            requested_k=k,
            retrieved=len(candidates),
            returned=len(kept),
            dropped=len(candidates) - len(survivors),
        )
        return kept

    def search_with_trace(
        self, query_vector: Sequence[float], plan: FilterPlan, k: int
    ) -> tuple[list[UncheckedHit], PostFilterTrace]:
        """:meth:`search`, plus the counts the recall-collapse chart is drawn from."""
        hits = self.search(query_vector, plan, k)
        assert self.last_trace is not None  # set by search on every path
        return hits, self.last_trace

    def count_matching(self, plan: FilterPlan) -> int:
        """Delegated, and therefore *correct*.

        The baseline is wrong about retrieval, not about counting. Making it lie
        here too would muddy the evaluation: we want to show that a system which
        knows exactly how many chunks the principal may see still fails to
        retrieve them.
        """
        return self._delegate.count_matching(plan)

    def stats(self) -> StoreStats:
        inner = self._delegate.stats()
        return StoreStats(
            n_vectors=inner.n_vectors,
            resident_bytes=inner.resident_bytes,
            disk_bytes=inner.disk_bytes,
            backend=self.name,
            detail={
                **inner.detail,
                "not_for_production": "true",
                "arm": "baseline",
                "filter_stage": "after_retrieval",
                "overfetch": str(self.overfetch),
                "delegate": inner.backend,
            },
        )
