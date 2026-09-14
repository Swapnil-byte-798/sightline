"""Qdrant backend: the serving path, with the permission filter inside traversal.

The point of this file is one sentence: **the filter is applied during graph
traversal, not after it.** Qdrant evaluates payload conditions while walking the
HNSW graph, so the search returns k results from the permitted subset rather
than k global results of which some survive. That is the difference between this
module and :mod:`sightline.store.postfilter`, and it is the reason a correct
permissioned assistant is possible at all on an approximate index.

Index layout, and why it is unusual
-----------------------------------
The collection is created with ``hnsw_config.m = 0`` and
``hnsw_config.payload_m = 16``. That disables the *global* HNSW graph entirely
and builds a separate subgraph per indexed payload value. A query carrying a
grant-token filter then traverses only the subgraphs for the tokens it holds,
which is why filtered search stays fast at high selectivity instead of degrading
into a scan with a predicate bolted on.

The cost is stated plainly: with a multi-valued field like ``grant_tokens`` a
point joins one subgraph per token it carries, so build time and index memory
scale with the average number of tokens per chunk. That is acceptable because of
ADR 0002 — tokens come from *groups*, so the count per chunk is small and stable,
and it does not grow when people join or leave those groups.

``grant_tokens`` is indexed as a keyword field with ``is_tenant=True``. The
honest caveat: Qdrant's tenant optimisation co-locates points on disk by a
tenant key and assumes roughly one value per point. With a multi-valued field the
on-disk co-location is weaker than the single-tenant case; the ``payload_m``
subgraphs, not ``is_tenant``, are what actually make the filter cheap. The flag
is set because it is still the correct declaration of intent and costs nothing.

What the local-mode test does *not* prove
-----------------------------------------
``QdrantClient(location=":memory:")`` is a pure-Python reimplementation. It warns
that payload indexes have no effect and that ``search_params`` is ignored, and it
brute-forces every query. So the conformance suite running against local mode
proves the **filter semantics** are right — it agrees hit for hit with the
reference store — and proves nothing at all about traversal or latency. The
claim that the filter is applied during graph traversal is only testable against
a real server, which is what the ``qdrant`` CI job runs. Saying otherwise would
be publishing a number the test did not measure.

Not built: Qdrant-side re-check. There is none, and there will not be one. The
index is a hint (ADR 0001); everything this module returns is an
``UncheckedHit`` and has to survive ``recheck()`` against the tuple store before
anyone sees it.
"""

from __future__ import annotations

import uuid
from typing import Any, Sequence

from sightline.store.base import StoreStats, UncheckedHit
from sightline.store.memory import first_matching_token, normalise, plan_bindings
from sightline.types import Chunk, FilterPlan, PlanStrategy

__all__ = ["QdrantVectorStore", "QDRANT_EXTRA_HINT", "POINT_ID_NAMESPACE"]

QDRANT_EXTRA_HINT = (
    "the Qdrant backend needs qdrant-client, which is an optional extra: "
    "pip install 'sightline[qdrant]'"
)

#: Qdrant point ids must be unsigned integers or UUIDs, and Sightline chunk ids
#: are opaque strings. A UUIDv5 keeps the mapping deterministic, so re-upserting
#: the same chunk id overwrites the same point instead of duplicating it — which
#: is exactly what the idempotency clause of the VectorStore contract requires.
POINT_ID_NAMESPACE = uuid.UUID("0f0d2a2e-2b3a-5f6c-9a1d-0c5e7d1a4b21")

try:  # pragma: no cover - exercised only by the optional-extra CI job
    from qdrant_client import QdrantClient, models
except ImportError:  # pragma: no cover
    QdrantClient = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]


def _require_qdrant() -> None:
    """Fail with the install command rather than an AttributeError three frames in."""
    if QdrantClient is None or models is None:
        raise ImportError(QDRANT_EXTRA_HINT)


def point_id_for(chunk_id: str) -> str:
    """Deterministic Qdrant point id for a Sightline chunk id."""
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


class QdrantVectorStore:
    """Vector search over Qdrant with the permission filter pushed into the index.

    Implements :class:`~sightline.store.base.VectorStore`. Honours every
    :class:`~sightline.types.PlanStrategy`:

    ``ENUMERATE``
        A ``MatchAny`` over ``object_ref`` (and ``object_id``, for plans that
        emit bare ids). Qdrant prunes on the keyword index.
    ``GRANT_TOKENS``
        A ``MatchAny`` over ``grant_tokens``. This is the path the ``payload_m``
        subgraphs exist for.
    ``UNFILTERED``
        No filter object at all. Only emitted for a principal the compiler has
        proved can see nearly everything; anything else reaching this branch
        would be a compiler bug, not an index bug.
    ``EXACT_SCAN``
        Filter as above, plus ``SearchParams(exact=True)``, which bypasses HNSW
        for a brute-force pass over the filtered set. This is the **only** path
        on this backend where a test may assert strict top-k equality with the
        oracle (FR-14); on every other path HNSW is approximate and the honest
        assertion is a recall floor.
    """

    name = "qdrant"

    def __init__(
        self,
        client: Any | None = None,
        *,
        collection: str = "sightline_chunks",
        dim: int | None = None,
        location: str | None = None,
        url: str | None = None,
        hnsw_m: int = 0,
        payload_m: int = 16,
        create: bool = True,
        **client_kwargs: Any,
    ) -> None:
        """Connect (or start an in-memory Qdrant) and ensure the collection exists.

        Args:
            client: An existing ``QdrantClient``. Supply this in tests.
            collection: Collection name.
            dim: Vector dimensionality. Required when ``create`` is true and the
                collection does not exist yet; otherwise inferred on first upsert.
            location: Passed to ``QdrantClient`` — ``":memory:"`` for the
                embedded test instance.
            url: Server URL, for a real deployment.
            hnsw_m: Global graph degree. **0 on purpose**: it disables the global
                graph so that only the per-payload-value subgraphs are built.
            payload_m: Degree of the per-payload-value subgraphs.
            create: Create the collection and payload indexes if missing.
            **client_kwargs: Forwarded to ``QdrantClient``.

        Raises:
            ImportError: If ``qdrant-client`` is not installed.
        """
        _require_qdrant()
        self.collection = collection
        self.hnsw_m = hnsw_m
        self.payload_m = payload_m
        self._dim = dim
        if client is not None:
            self.client = client
        elif url is not None:
            self.client = QdrantClient(url=url, **client_kwargs)
        else:
            self.client = QdrantClient(location=location or ":memory:", **client_kwargs)
        if create and dim is not None:
            self.ensure_collection(dim)

    # ------------------------------------------------------------ collection

    def ensure_collection(self, dim: int) -> None:
        """Create the collection and its payload indexes if they are missing.

        Idempotent: safe to call on every process start, which is how a deploy
        that adds a payload index gets it without a migration step.
        """
        _require_qdrant()
        self._dim = dim
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            # m=0 plus payload_m: no global graph, one subgraph per payload value.
            hnsw_config=models.HnswConfigDiff(m=self.hnsw_m, payload_m=self.payload_m),
        )
        self._create_keyword_index("grant_tokens", is_tenant=True)
        self._create_keyword_index("object_ref", is_tenant=False)
        self._create_keyword_index("object_id", is_tenant=False)

    def _create_keyword_index(self, field: str, *, is_tenant: bool) -> None:
        """Keyword payload index, with a graceful path on older client builds.

        ``KeywordIndexParams`` and ``is_tenant`` landed in qdrant-client 1.11. On
        anything older the index is still created, just without the tenant
        declaration — the ``payload_m`` subgraphs are what make the filter cheap,
        so losing the hint costs disk locality and nothing else.
        """
        try:
            schema: Any = models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=is_tenant or None,
            )
        except (AttributeError, TypeError):  # pragma: no cover - old client
            schema = models.PayloadSchemaType.KEYWORD
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name=field,
            field_schema=schema,
        )

    # ---------------------------------------------------------------- writes

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Insert or replace points. Idempotent on ``chunk.id`` via the UUIDv5 map.

        The payload carries ``grant_tokens`` and nothing else that bears on
        access. Per ADR 0002 there are no user ids in here, so a membership
        change rewrites zero points.
        """
        _require_qdrant()
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks and {len(vectors)} vectors")
        if not chunks:
            return
        prepared = [normalise(v, self._dim) for v in vectors]
        if self._dim is None:
            self.ensure_collection(int(prepared[0].shape[0]))
        elif not self.client.collection_exists(self.collection):
            self.ensure_collection(self._dim)

        points = []
        for chunk, vec in zip(chunks, prepared):
            points.append(
                models.PointStruct(
                    id=point_id_for(chunk.id),
                    vector=vec.tolist(),
                    payload={
                        "chunk_id": chunk.id,
                        "object_ref": str(chunk.object),
                        "object_id": chunk.object.id,
                        "text": chunk.text,
                        "grant_tokens": sorted(str(t) for t in chunk.grant_tokens),
                        "metadata": dict(chunk.metadata),
                    },
                )
            )
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    # ----------------------------------------------------------------- reads

    def build_filter(self, plan: FilterPlan) -> Any | None:
        """Compile a plan into a Qdrant payload filter.

        Returns ``None`` only for ``UNFILTERED``. Every binding condition becomes
        a top-level ``must`` entry, so multiple conditions combine with AND
        (combining them with OR is planted mutant M2). Inside a single condition
        the match is ``MatchAny``, which is OR over the values — any one grant
        suffices.

        Raises:
            ValueError: If the plan binds a condition whose value set is empty.
                Callers must check :meth:`_admits_nothing` first; an empty
                ``MatchAny`` is a client error in Qdrant and, worse, some
                versions have treated it as "no condition", which is planted
                mutant M6 arriving through the wire format.
        """
        _require_qdrant()
        by_ids, by_tokens = plan_bindings(plan)
        if not by_ids and not by_tokens:
            if plan.strategy is PlanStrategy.UNFILTERED:
                return None
            raise ValueError("plan binds no condition and is not UNFILTERED; admits nothing")

        must: list[Any] = []
        if by_tokens:
            if not plan.grant_tokens:
                raise ValueError("plan binds grant_tokens but holds none; admits nothing")
            must.append(
                models.FieldCondition(
                    key="grant_tokens",
                    match=models.MatchAny(any=sorted(str(t) for t in plan.grant_tokens)),
                )
            )
        if by_ids:
            if not plan.explicit_ids:
                raise ValueError("plan binds explicit_ids but holds none; admits nothing")
            refs = sorted(i for i in plan.explicit_ids if ":" in i)
            bare = sorted(i for i in plan.explicit_ids if ":" not in i)
            should: list[Any] = []
            if refs:
                should.append(
                    models.FieldCondition(key="object_ref", match=models.MatchAny(any=refs))
                )
            if bare:
                should.append(
                    models.FieldCondition(key="object_id", match=models.MatchAny(any=bare))
                )
            must.append(models.Filter(should=should))
        return models.Filter(must=must)

    @staticmethod
    def _admits_nothing(plan: FilterPlan) -> bool:
        """A plan that binds a condition it cannot satisfy admits zero chunks."""
        by_ids, by_tokens = plan_bindings(plan)
        if not by_ids and not by_tokens:
            return plan.strategy is not PlanStrategy.UNFILTERED
        return (by_tokens and not plan.grant_tokens) or (by_ids and not plan.explicit_ids)

    def search(
        self, query_vector: Sequence[float], plan: FilterPlan, k: int
    ) -> list[UncheckedHit]:
        """Top-k over the permitted subset, filtered inside the graph walk."""
        _require_qdrant()
        if k <= 0 or self._admits_nothing(plan):
            return []
        query = normalise(query_vector, self._dim).tolist()
        params = None
        if plan.strategy is PlanStrategy.EXACT_SCAN:
            params = models.SearchParams(exact=True)

        response = self.client.query_points(
            collection_name=self.collection,
            query=query,
            query_filter=self.build_filter(plan),
            limit=k,
            search_params=params,
            with_payload=True,
        )
        points = getattr(response, "points", response)

        hits: list[UncheckedHit] = []
        for point in points:
            payload = point.payload or {}
            tokens = payload.get("grant_tokens") or ()
            hits.append(
                UncheckedHit(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    score=float(point.score),
                    text=str(payload.get("text", "")),
                    object_ref=str(payload.get("object_ref", "")),
                    matched_token=first_matching_token(plan, tokens),
                )
            )
        return hits

    def count_matching(self, plan: FilterPlan) -> int:
        """Exact count of admitted points, for strategy selection."""
        _require_qdrant()
        if self._admits_nothing(plan):
            return 0
        result = self.client.count(
            collection_name=self.collection,
            count_filter=self.build_filter(plan),
            exact=True,
        )
        return int(result.count)

    def stats(self) -> StoreStats:
        _require_qdrant()
        try:
            info = self.client.get_collection(self.collection)
            n = int(getattr(info, "points_count", 0) or 0)
        except Exception:  # pragma: no cover - collection may not exist yet
            n = 0
        return StoreStats(
            n_vectors=n,
            resident_bytes=0,  # Qdrant owns the memory; asking it costs a round trip.
            disk_bytes=0,
            backend=self.name,
            detail={
                "collection": self.collection,
                "hnsw_m": str(self.hnsw_m),
                "payload_m": str(self.payload_m),
                "filter_stage": "during_traversal",
                "exact": "only under EXACT_SCAN",
                "dim": str(self._dim or 0),
            },
        )
