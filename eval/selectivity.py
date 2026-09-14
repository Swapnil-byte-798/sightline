"""The headline chart: what filtering strategy costs you, as the permitted
fraction of the corpus shrinks.

Five permission densities — 0.1%, 1%, 10%, 50%, 100% of the corpus visible to
the asking principal — crossed with four retrieval strategies, scored against an
exact oracle over the permitted subset. Two numbers come out for every cell:
**recall@10**, and what it cost to get it.

The finding this exists to publish:

    Post-filtering does not degrade gracefully. It degrades as the permitted
    fraction, because the probability that a globally-top-k document is one you
    may read *is* the permitted fraction. At 0.1% density a k=10 request returns
    approximately nothing, and it does so while reading the entire corpus.

That is not a subtle performance regression. It is the reason companies freeze
AI search rollouts and cannot explain why the assistant seems stupid for exactly
the people with the least access.

WHY BOTH COLUMNS
----------------
Recall alone is a sales chart. A filtering strategy that wins on recall and
costs ten times the work has not won, it has moved the problem, and somebody
will move it back under load. So every arm reports **distance computations**
(how many vectors were scored) and **wall-clock latency** alongside recall, and
the post-filter arm is run at overfetch 1 and overfetch 10 because "just
retrieve more candidates" is the patch every team reaches for first. It buys a
constant factor against a problem that scales with 1/density. The table says so.

THE HONESTY CONSTRAINTS
-----------------------
Three, and they are the reason this file is longer than it looks like it needs
to be.

1. **No exact-equality assertion.** :data:`RECALL_FLOOR` is a floor. On an
   approximate index, ``filtered top-k == exact top-k`` holds only when the query
   routes to brute force, and a test asserting it elsewhere is a test that
   eventually lies about an ANN parameter change. FR-14.

2. **The reference backend is exact, and the report says so out loud.**
   ``MemoryVectorStore`` scores the permitted subset with numpy and has no
   approximate path at all, so the grant-token and exact-scan arms return 1.000
   here *by construction*. That is a fact about the backend, not evidence about
   HNSW. Every result carries ``approximate=false`` and the rendered table
   carries the caveat, because a 1.000 published without it is a number pretending
   to be a measurement. Point the harness at a Qdrant collection (``--store
   qdrant``) and the floor starts doing work.

3. **The hash embedder is not semantic.**
   :class:`~sightline.ingest.embed.HashEmbedder` hashes word unigrams and
   bigrams. Recall measured against it is recall over string overlap. Results
   carry ``semantic=false`` and :mod:`eval.report` refuses to drop the caveat.

WHAT IS DELIBERATELY NOT MEASURED HERE
--------------------------------------
Staleness. Every arm is scored against an oracle built from the *same* epoch, so
a disagreement is a retrieval-strategy defect and nothing else. The index-versus-
authority gap is recheck's problem, it is measured in :mod:`eval.attack`, and
averaging the two into one recall number would let a leak hide behind "the index
was behind".
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import eval  # noqa: F401  - path bootstrap; see eval/__init__.py
import numpy as np
from sightline.authz.compile import (
    DEFAULT_NAMESPACE,
    DEFAULT_RELATION,
    compile_plan,
    grant_tokens_for_object,
    principal_closure,
    reachable_objects,
)
from sightline.authz.tuples import MemoryTupleStore, TupleStore
from sightline.ingest.embed import EMBED_DIM, HashEmbedder
from sightline.store.base import UncheckedHit, VectorStore
from sightline.store.memory import MemoryVectorStore
from sightline.types import Chunk, FilterPlan, ObjectRef, PlanStrategy, PrincipalRef

__all__ = [
    "DENSITIES",
    "RECALL_FLOOR",
    "ARMS",
    "CorpusSpec",
    "Corpus",
    "ArmResult",
    "DensityResult",
    "SelectivityReport",
    "build_corpus",
    "run_selectivity",
    "render_markdown",
    "main",
]

#: The permission densities the FRD names. Fractions of the corpus a principal
#: can see. 0.1% is not a pathological case invented to win an argument: it is a
#: contractor with access to one project folder in a company-wide index.
DENSITIES: tuple[float, ...] = (0.001, 0.01, 0.1, 0.5, 1.0)

#: FR-14. A **floor**, asserted against the exact oracle, never an equality.
#: 0.95 rather than 1.0 because the serving backend is an approximate index and
#: an approximate index that never misses is an approximate index you have
#: mis-measured.
RECALL_FLOOR = 0.95

#: Arms measured at every density. The first is the baseline bug.
ARMS: tuple[str, ...] = (
    "post_filter",
    "post_filter_x10",
    "grant_tokens",
    "exact_scan",
)

_K = 10


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    """Shape of the synthetic corpus. Deterministic for a given seed.

    Synthetic rather than Enron on purpose. This harness measures the *geometry*
    of filtering — what happens to top-k when the permitted subset is a thin
    slice — and that needs a corpus where the permitted fraction is an input you
    set rather than a property you discover. The Enron connector produces the
    realistic access structure and is what the end-to-end numbers run on; it
    cannot produce a principal who sees exactly 0.1% of the corpus on request.
    """

    n_docs: int = 4000
    n_queries: int = 30
    #: Vocabulary is partitioned into topics so that cosine ranking has real
    #: structure to find. Without this every document is equidistant and recall
    #: measures tie-breaking.
    n_topics: int = 40
    words_per_topic: int = 24
    words_per_doc: int = 48
    topics_per_doc: int = 2
    words_per_query: int = 6
    dim: int = EMBED_DIM
    seed: int = 20240914
    densities: tuple[float, ...] = DENSITIES


@dataclass(frozen=True, slots=True)
class Corpus:
    """A built corpus plus the principals that see a known slice of it."""

    spec: CorpusSpec
    tuple_store: TupleStore
    chunks: tuple[Chunk, ...]
    vectors: np.ndarray
    queries: tuple[str, ...]
    query_vectors: np.ndarray
    #: density -> principal constructed to see (about) that fraction.
    principals: dict[float, PrincipalRef]
    #: density -> how many documents that principal can actually see.
    visible: dict[float, int]
    embedder_name: str
    embedder_is_semantic: bool

    @property
    def n_docs(self) -> int:
        return len(self.chunks)


def _topic_words(spec: CorpusSpec) -> list[list[str]]:
    return [
        [f"t{topic:03d}w{word:02d}" for word in range(spec.words_per_topic)]
        for topic in range(spec.n_topics)
    ]


def build_corpus(spec: CorpusSpec | None = None) -> Corpus:
    """Build the corpus, its tuples, its vectors and one principal per density.

    The permission structure is the interesting part:

    * Every document carries ``doc:N#viewer@group:corpus#member``. Every
      document must compile to at least one grant token or ingest rejects it
      (ADR 0003, FR-15), and a document visible to nobody is not a document this
      chart can say anything about.
    * For each target density ``d`` there is a group holding ``viewer`` on a
      seeded random ``round(d * n_docs)`` documents, and exactly one principal
      whose only membership is that group. The document sample is drawn
      independently per density, so the 10% principal's documents are not a
      superset of the 1% principal's — realistic, and it stops any arm from
      benefiting from a nested structure that real ACLs do not have.
    * Nobody in the measured set is a member of ``group:corpus``. If they were,
      every principal would see everything and the chart would be flat.

    Membership is never expanded into the index: a principal's plan carries the
    token derived from their *group*, and the chunk carries the token derived
    from the granting userset. Joining a group rewrites zero vectors (ADR 0002),
    which is exactly why the densities can be varied without re-embedding.
    """
    spec = spec or CorpusSpec()
    rng = random.Random(spec.seed)
    topics = _topic_words(spec)

    texts: list[str] = []
    doc_topics: list[tuple[int, ...]] = []
    for _ in range(spec.n_docs):
        chosen = tuple(rng.sample(range(spec.n_topics), spec.topics_per_doc))
        pool = [w for topic in chosen for w in topics[topic]]
        texts.append(" ".join(rng.choice(pool) for _ in range(spec.words_per_doc)))
        doc_topics.append(chosen)

    queries: list[str] = []
    for _ in range(spec.n_queries):
        topic = rng.randrange(spec.n_topics)
        queries.append(" ".join(rng.choice(topics[topic]) for _ in range(spec.words_per_query)))

    tuples: list[str] = [f"doc:{i}#viewer@group:corpus#member" for i in range(spec.n_docs)]
    principals: dict[float, PrincipalRef] = {}
    for index, density in enumerate(spec.densities):
        group = f"group:d{index}"
        size = max(1, min(spec.n_docs, int(round(density * spec.n_docs))))
        for doc in rng.sample(range(spec.n_docs), size):
            tuples.append(f"doc:{doc}#viewer@{group}#member")
        principal = PrincipalRef("user", f"p{index}")
        tuples.append(f"{group}#member@{principal}")
        principals[density] = principal

    store = MemoryTupleStore(tuples)

    embedder = HashEmbedder(spec.dim, seed=spec.seed & 0xFFFF, quiet=True)
    vectors = embedder.encode(texts)
    query_vectors = embedder.encode(queries)

    chunks = tuple(
        Chunk(
            id=f"c{i:06d}",
            object=ObjectRef("doc", str(i)),
            text=texts[i],
            grant_tokens=grant_tokens_for_object(store, ObjectRef("doc", str(i))),
            metadata={},
        )
        for i in range(spec.n_docs)
    )
    for chunk in chunks:
        if not chunk.grant_tokens:  # pragma: no cover - construction guarantees it
            raise ValueError(f"{chunk.object} compiled to zero grant tokens (ADR 0003)")

    visible = {
        density: _visible_count(store, principals[density]) for density in spec.densities
    }
    return Corpus(
        spec=spec,
        tuple_store=store,
        chunks=chunks,
        vectors=vectors,
        queries=tuple(queries),
        query_vectors=query_vectors,
        principals=principals,
        visible=visible,
        embedder_name=embedder.name,
        embedder_is_semantic=bool(getattr(embedder, "is_semantic", False)),
    )


def _visible_count(store: TupleStore, principal: PrincipalRef) -> int:
    ids, _ = reachable_objects(
        store,
        principal_closure(store, principal),
        relation=DEFAULT_RELATION,
        namespace=DEFAULT_NAMESPACE,
    )
    return len(ids)


def _permitted_ids(store: TupleStore, principal: PrincipalRef) -> frozenset[str]:
    ids, truncated = reachable_objects(
        store,
        principal_closure(store, principal),
        relation=DEFAULT_RELATION,
        namespace=DEFAULT_NAMESPACE,
    )
    if truncated:  # pragma: no cover - the eval corpus never reaches the cap
        raise RuntimeError(
            "the reachable-object walk truncated; an enumerated arm cannot be "
            "measured against a partial id set without quietly reporting a "
            "recall loss as a strategy difference"
        )
    return ids


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


def _force(plan: FilterPlan, strategy: PlanStrategy, ids: frozenset[str]) -> FilterPlan:
    """Re-stamp a compiled plan with a chosen strategy, for measurement only.

    The compiler picks a strategy from cardinality and is right to; this harness
    exists to measure what the strategies it did *not* pick would have cost, so
    it overrides the choice. Nothing outside this module does this, and nothing
    outside this module should: forcing a strategy in the serving path is how you
    end up with ``UNFILTERED`` in production.

    The explicit-id set is carried as ``namespace:id`` rather than bare ids, so
    the enumerated arms agree with the token arm about what an id means.
    """
    return dataclasses.replace(plan, strategy=strategy, explicit_ids=ids)


@dataclass(frozen=True, slots=True)
class _Arm:
    """One strategy under measurement: how to search, and what it cost."""

    name: str
    #: ``(query_vector, k) -> (hits, distance_computations)``
    search: Callable[[np.ndarray, int], tuple[list[UncheckedHit], int]]
    #: Whether the underlying index is approximate. Drives the recall caveat.
    approximate: bool


def _build_arms(
    corpus: Corpus, base_plan: FilterPlan, permitted: frozenset[str]
) -> tuple[dict[str, _Arm], Callable[[np.ndarray, int], list[UncheckedHit]]]:
    """Construct one arm per strategy plus the oracle, sharing one vector copy.

    Every arm wraps the same :class:`~sightline.store.memory.MemoryVectorStore`
    instance where it can, because building four copies of a 4000x384 matrix on
    a 2015 laptop is most of the runtime and none of the measurement.
    """
    from sightline.store.postfilter import PostFilterStore  # baseline arm, guarded import

    store = MemoryVectorStore(corpus.spec.dim)
    store.upsert(corpus.chunks, corpus.vectors)

    token_plan = _force(base_plan, PlanStrategy.GRANT_TOKENS, frozenset())
    scan_plan = _force(base_plan, PlanStrategy.EXACT_SCAN, permitted)
    unfiltered = dataclasses.replace(
        base_plan, strategy=PlanStrategy.UNFILTERED, explicit_ids=frozenset()
    )

    def _cost(plan: FilterPlan) -> int:
        rows = store.candidate_rows(plan)
        return len(store) if rows is None else int(rows.size)

    token_cost = _cost(token_plan)
    scan_cost = _cost(scan_plan)
    global_cost = _cost(unfiltered)

    post = PostFilterStore(delegate=store, overfetch=1, name="postfilter")
    post10 = PostFilterStore(delegate=store, overfetch=10, name="postfilter-x10")
    # The wrappers keep their own token copy for the drop step; the delegate
    # already holds the vectors, so this is bookkeeping, not a second index.
    post.upsert(corpus.chunks, corpus.vectors)
    post10.upsert(corpus.chunks, corpus.vectors)

    arms = {
        "post_filter": _Arm(
            "post_filter",
            lambda q, k: (post.search(q, base_plan, k), global_cost),
            approximate=False,
        ),
        "post_filter_x10": _Arm(
            "post_filter_x10",
            lambda q, k: (post10.search(q, base_plan, k), global_cost),
            approximate=False,
        ),
        "grant_tokens": _Arm(
            "grant_tokens",
            lambda q, k: (store.search(q, token_plan, k), token_cost),
            approximate=False,
        ),
        "exact_scan": _Arm(
            "exact_scan",
            lambda q, k: (store.search(q, scan_plan, k), scan_cost),
            approximate=False,
        ),
    }

    def oracle(q: np.ndarray, k: int) -> list[UncheckedHit]:
        return store.search(q, scan_plan, k)

    return arms, oracle


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One strategy at one density."""

    arm: str
    recall_at_10: float
    #: Mean results actually returned. Post-filtering's shortfall lives here:
    #: the caller asked for 10 and the recall number alone does not say that it
    #: got two.
    mean_returned: float
    #: Vectors scored per query. Deterministic, so :mod:`eval.report` gates it.
    distance_computations: int
    p50_ms: float
    p95_ms: float
    approximate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "recall_at_10": round(self.recall_at_10, 4),
            "mean_returned": round(self.mean_returned, 3),
            "distance_computations": self.distance_computations,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "approximate": self.approximate,
        }


@dataclass(frozen=True, slots=True)
class DensityResult:
    """Every arm at one permission density."""

    density: float
    principal: str
    visible_docs: int
    #: Mean size of the oracle's answer. Below k when the principal cannot see
    #: k documents at all, which is the whole point at 0.1%.
    mean_oracle_size: float
    arms: tuple[ArmResult, ...]

    def arm(self, name: str) -> ArmResult:
        for result in self.arms:
            if result.arm == name:
                return result
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "density": self.density,
            "principal": self.principal,
            "visible_docs": self.visible_docs,
            "mean_oracle_size": round(self.mean_oracle_size, 3),
            "arms": [a.to_dict() for a in self.arms],
        }


@dataclass(frozen=True, slots=True)
class SelectivityReport:
    """The whole run, ready to be rendered or diffed."""

    k: int
    n_docs: int
    n_queries: int
    seed: int
    embedder: str
    semantic: bool
    recall_floor: float
    results: tuple[DensityResult, ...]
    warnings: tuple[str, ...] = ()

    @property
    def floor_holds(self) -> bool:
        """Whether every correctly-filtered arm cleared :data:`RECALL_FLOOR`.

        The post-filter arms are excluded, and not because they are allowed to
        fail: they are the *demonstration*. Holding the baseline bug to the floor
        would turn the headline finding into a failing build every run.
        """
        return all(
            d.arm(name).recall_at_10 >= self.recall_floor
            for d in self.results
            for name in ("grant_tokens", "exact_scan")
        )

    @property
    def worst_post_filter_recall(self) -> float:
        return min(d.arm("post_filter").recall_at_10 for d in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "n_docs": self.n_docs,
            "n_queries": self.n_queries,
            "seed": self.seed,
            "embedder": self.embedder,
            "semantic": self.semantic,
            "recall_floor": self.recall_floor,
            "floor_holds": self.floor_holds,
            "worst_post_filter_recall": round(self.worst_post_filter_recall, 4),
            "densities": [d.to_dict() for d in self.results],
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


def _recall(retrieved: Sequence[str], truth: Sequence[str]) -> float:
    """Fraction of the oracle's answer the arm found.

    The denominator is ``len(truth)``, not ``k``. A principal who can see three
    documents has an oracle answer of size three, and scoring them out of ten
    would report a hard ceiling of 0.3 as a retrieval failure. Where the oracle
    is empty — the principal sees nothing — recall is defined as 1.0: returning
    nothing is the complete and correct answer, and the security property being
    demonstrated elsewhere is precisely that it stays empty.
    """
    if not truth:
        return 1.0
    return len(set(retrieved) & set(truth)) / len(truth)


def run_selectivity(
    spec: CorpusSpec | None = None,
    *,
    k: int = _K,
    corpus: Corpus | None = None,
    repeats: int = 1,
    progress: Callable[[str], None] | None = None,
) -> SelectivityReport:
    """Measure every arm at every density. Deterministic for a given seed.

    Args:
        spec: Corpus shape. Ignored when ``corpus`` is supplied.
        k: Results requested per query. 10, because that is what the FRD
            publishes and what a UI shows.
        corpus: A prebuilt corpus, so a caller running this twice does not pay
            for embedding twice.
        repeats: Timing repeats per query. The recall numbers are unaffected;
            more repeats narrow the latency percentiles on a noisy laptop.
        progress: Called with a one-line status per density. ``None`` is silent.

    Returns:
        A :class:`SelectivityReport`. Nothing is asserted here — the harness
        measures and :mod:`eval.report` gates, so a drifted number produces a
        readable diff instead of a stack trace.
    """
    corpus = corpus or build_corpus(spec)
    results: list[DensityResult] = []

    for density in corpus.spec.densities:
        principal = corpus.principals[density]
        permitted = _permitted_ids(corpus.tuple_store, principal)
        base_plan = compile_plan(
            corpus.tuple_store, principal, corpus_size=corpus.n_docs
        )
        arms, oracle = _build_arms(corpus, base_plan, permitted)

        truth: list[list[str]] = []
        for row in range(len(corpus.queries)):
            truth.append([h.chunk_id for h in oracle(corpus.query_vectors[row], k)])

        arm_results: list[ArmResult] = []
        for name in ARMS:
            arm = arms[name]
            recalls: list[float] = []
            returned: list[int] = []
            timings: list[float] = []
            for row in range(len(corpus.queries)):
                query = corpus.query_vectors[row]
                hits: list[UncheckedHit] = []
                for _ in range(max(1, repeats)):
                    started = time.perf_counter()
                    hits, _cost = arm.search(query, k)
                    timings.append((time.perf_counter() - started) * 1000.0)
                recalls.append(_recall([h.chunk_id for h in hits], truth[row]))
                returned.append(len(hits))
            _, cost = arm.search(corpus.query_vectors[0], k)
            arm_results.append(
                ArmResult(
                    arm=name,
                    recall_at_10=statistics.fmean(recalls),
                    mean_returned=statistics.fmean(returned),
                    distance_computations=cost,
                    p50_ms=_percentile(timings, 50),
                    p95_ms=_percentile(timings, 95),
                    approximate=arm.approximate,
                )
            )

        results.append(
            DensityResult(
                density=density,
                principal=str(principal),
                visible_docs=corpus.visible[density],
                mean_oracle_size=statistics.fmean(len(t) for t in truth),
                arms=tuple(arm_results),
            )
        )
        if progress is not None:
            worst = min(a.recall_at_10 for a in arm_results)
            progress(f"density {density:>6.3%}: worst arm recall@{k} = {worst:.3f}")

    warnings: list[str] = []
    if not corpus.embedder_is_semantic:
        warnings.append(
            f"embedder {corpus.embedder_name!r} is not semantic: it hashes word "
            "unigrams and bigrams, so these are recall numbers over string "
            "overlap. Install the embed extra and re-run for semantic recall."
        )
    if not any(a.approximate for d in results for a in d.arms):
        warnings.append(
            "every arm ran on an exact backend, so the filtered arms return "
            f"recall 1.000 by construction rather than by measurement. The "
            f"{RECALL_FLOOR:.2f} floor only does work against an approximate "
            "index (Qdrant/HNSW); it is published here as the floor it is, not "
            "as evidence about ANN behaviour."
        )

    return SelectivityReport(
        k=k,
        n_docs=corpus.n_docs,
        n_queries=len(corpus.queries),
        seed=corpus.spec.seed,
        embedder=corpus.embedder_name,
        semantic=corpus.embedder_is_semantic,
        recall_floor=RECALL_FLOOR,
        results=tuple(results),
        warnings=tuple(warnings),
    )


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. No interpolation, because these are timings.

    Interpolating between two wall-clock samples invents a measurement that was
    never taken. The nearest-rank value is one that actually happened.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_markdown(report: SelectivityReport) -> str:
    """The table :mod:`eval.report` pastes into the README.

    Generated, never typed. Every figure in it came out of the run that produced
    this object, which is the only reason it is worth putting in a README at all.
    """
    lines: list[str] = []
    lines.append(
        f"Corpus {report.n_docs:,} documents, {report.n_queries} queries, "
        f"k={report.k}, seed {report.seed}, embedder `{report.embedder}`. "
        f"Recall is measured against an exact scan over the permitted subset."
    )
    lines.append("")
    lines.append(
        "| visible | docs | arm | recall@10 | returned/10 | vectors scored | p50 ms | p95 ms |"
    )
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|")
    for density in report.results:
        for index, arm in enumerate(density.arms):
            visible = f"{density.density:.1%}" if index == 0 else ""
            docs = f"{density.visible_docs:,}" if index == 0 else ""
            lines.append(
                f"| {visible} | {docs} | `{arm.arm}` | {arm.recall_at_10:.3f} | "
                f"{arm.mean_returned:.1f} | {arm.distance_computations:,} | "
                f"{arm.p50_ms:.2f} | {arm.p95_ms:.2f} |"
            )
    lines.append("")
    lines.append(
        f"**Post-filtering bottoms out at recall {report.worst_post_filter_recall:.3f}** "
        f"while scoring every vector in the corpus. Overfetching ten times as "
        f"many candidates buys a constant factor against a problem that scales "
        f"with 1/density; the `post_filter_x10` row is what that patch is worth."
    )
    lines.append("")
    for warning in report.warnings:
        lines.append(f"> Caveat: {warning}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.selectivity",
        description="Recall and cost versus permission density. The headline chart.",
    )
    parser.add_argument("--docs", type=int, default=CorpusSpec.n_docs)
    parser.add_argument("--queries", type=int, default=CorpusSpec.n_queries)
    parser.add_argument("--k", type=int, default=_K)
    parser.add_argument("--seed", type=int, default=CorpusSpec.seed)
    parser.add_argument("--repeats", type=int, default=1, help="timing repeats per query")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    spec = CorpusSpec(n_docs=args.docs, n_queries=args.queries, seed=args.seed)
    report = run_selectivity(
        spec,
        k=args.k,
        repeats=args.repeats,
        progress=None if args.quiet or args.json else lambda line: print(line, file=sys.stderr),
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    # Exit non-zero only when a *correctly* filtered arm missed the floor. The
    # post-filter collapse is the finding, not a failure.
    return 0 if report.floor_holds else 1


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
