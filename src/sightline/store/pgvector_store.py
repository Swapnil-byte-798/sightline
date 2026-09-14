"""Postgres + pgvector backend, with row-level security as a second enforcement layer.

Every other backend has exactly one thing standing between a principal and a
document they may not see: the filter the application compiled. If that filter is
wrong, or absent, the index answers anyway. This backend does not have that
property. The table carries a Postgres ROW LEVEL SECURITY policy, so a query that
forgets its ``WHERE`` clause entirely still returns zero forbidden rows — the
database refuses them.

That is the whole reason this file exists. It is slower than Qdrant and it is not
the serving path. It is the layer that makes "we had a bug in the filter" a
non-event, and the test for it (FR-10) deliberately sends an unfiltered query
through :meth:`PgVectorStore.search_with_app_filter_removed` and asserts the
result is empty.

Threat model, stated honestly
-----------------------------
RLS here defends against **a missing or wrong application filter**. It does not
defend against a compromised application: the policy reads session settings that
the application sets, so anything that can set ``sightline.grant_tokens`` to a
real token set can set it to a wider one. Claiming otherwise would be a lie, and
a security control whose limits are undocumented is a control nobody can reason
about.

What it buys, precisely: the dangerous path stops being an *omission* and becomes
an explicit, greppable, loggable act. Forgetting a predicate is the single most
common way permission filters fail; forgetting a predicate here returns nothing.

The wildcard
------------
``UNFILTERED`` plans need a representation at the SQL layer, and the policy has
no "off" switch by design. So there is one reserved token, :data:`WILDCARD_TOKEN`,
which the policy treats as matching every row. It is a plaintext string with a
namespace prefix, while every real grant token is a keyed hash (FR-7), so a
derived token cannot collide with it. Setting it requires naming it, which is the
point: a grep for ``sightline:all`` finds every place the system decided someone
may see everything.

Session scope
-------------
The session settings are set with ``set_config(..., is_local => true)`` inside an
explicit transaction, so they are discarded at commit. On a pooled connection a
session-scoped GUC outlives the request that set it and the next principal
inherits it — that is a permission leak wearing a connection pool as a disguise,
and transaction scope is the only correct choice.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from sightline.store.base import StoreStats, UncheckedHit
from sightline.store.memory import first_matching_token, normalise, plan_bindings
from sightline.types import Chunk, FilterPlan, PlanStrategy

__all__ = [
    "APP_ROLE",
    "INGEST_ROLE",
    "POSTGRES_EXTRA_HINT",
    "SCHEMA_DDL",
    "WILDCARD_TOKEN",
    "PgVectorStore",
]

POSTGRES_EXTRA_HINT = (
    "the pgvector backend needs psycopg, which is an optional extra: "
    "pip install 'sightline[postgres]'"
)

#: The one token that matches every row. Only a ``UNFILTERED`` plan sets it.
WILDCARD_TOKEN = "sightline:all"

#: The role the query path connects as. SELECT only: the serving process cannot
#: write the permission payload it is being filtered by.
APP_ROLE = "sightline_app"

#: The role the indexer connects as. Writes rows, still subject to RLS on read.
INGEST_ROLE = "sightline_ingest"

try:  # pragma: no cover - exercised only by the optional-extra CI job
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]


#: Applied verbatim by CI so the policy in this docstring is the policy that runs.
#: ``{dim}`` is the only substitution; everything else is literal, because a
#: security policy assembled from f-strings at runtime is a policy nobody reviews.
SCHEMA_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sightline_chunk (
    chunk_id     text PRIMARY KEY,
    object_ref   text NOT NULL,
    object_id    text NOT NULL,
    body         text NOT NULL,
    grant_tokens text[] NOT NULL DEFAULT '{{}}',
    metadata     jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    embedding    vector({dim}) NOT NULL
);

CREATE INDEX IF NOT EXISTS sightline_chunk_tokens
    ON sightline_chunk USING gin (grant_tokens);
CREATE INDEX IF NOT EXISTS sightline_chunk_object
    ON sightline_chunk (object_ref);
CREATE INDEX IF NOT EXISTS sightline_chunk_embedding
    ON sightline_chunk USING hnsw (embedding vector_cosine_ops);

-- Reads the session settings the policy is built on. STABLE, and owned by the
-- schema owner rather than the app role, so the app cannot redefine it.
CREATE OR REPLACE FUNCTION sightline_session_array(setting text)
RETURNS text[]
LANGUAGE sql
STABLE
AS $$
    SELECT coalesce(
        string_to_array(nullif(current_setting(setting, true), ''), ','),
        '{{}}'::text[]
    );
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sightline_app') THEN
        CREATE ROLE sightline_app NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sightline_ingest') THEN
        CREATE ROLE sightline_ingest NOLOGIN;
    END IF;
END
$$;

-- The serving role reads. It does not get to write the payload it is filtered by.
GRANT SELECT ON sightline_chunk TO sightline_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON sightline_chunk TO sightline_ingest;
GRANT EXECUTE ON FUNCTION sightline_session_array(text) TO sightline_app, sightline_ingest;

ALTER TABLE sightline_chunk ENABLE ROW LEVEL SECURITY;
-- FORCE so the table owner is subject to the policy too. Without this, anyone
-- who happens to connect as the owner silently bypasses the whole control.
ALTER TABLE sightline_chunk FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS sightline_chunk_visible ON sightline_chunk;
CREATE POLICY sightline_chunk_visible ON sightline_chunk
FOR SELECT
USING (
    -- Grant-token condition. A session that set nothing gets the empty array,
    -- and `grant_tokens && '{{}}'` is false, so an undeclared session sees zero
    -- rows. Empty means nothing, never everything (planted mutant M6).
    (
        'sightline:all' = ANY (sightline_session_array('sightline.grant_tokens'))
        OR grant_tokens && sightline_session_array('sightline.grant_tokens')
    )
    -- Enumerate condition, ANDed with the above rather than ORed, because a plan
    -- carrying both conditions requires both (planted mutant M2). It applies
    -- only when the session declares it, so a GRANT_TOKENS plan is unaffected.
    AND (
        current_setting('sightline.enumerate', true) IS DISTINCT FROM 'on'
        OR object_ref = ANY (sightline_session_array('sightline.object_refs'))
        OR object_id  = ANY (sightline_session_array('sightline.object_refs'))
    )
);

-- The indexer sees everything: it has to, it is the thing writing the payload.
-- Postgres ORs permissive policies together, so this one also grants the ingest
-- role unrestricted SELECT. That is why the serving process connects as
-- sightline_app and why sightline_app is not a member of sightline_ingest. If
-- one deployment ever points the query path at the ingest credentials, the
-- second enforcement layer is gone and only the application filter remains.
DROP POLICY IF EXISTS sightline_chunk_write ON sightline_chunk;
CREATE POLICY sightline_chunk_write ON sightline_chunk
FOR ALL
TO sightline_ingest
USING (true)
WITH CHECK (true);
"""


def _require_psycopg() -> None:
    """Fail with the install command, not with a NoneType attribute error."""
    if psycopg is None:
        raise ImportError(POSTGRES_EXTRA_HINT)


def _vector_literal(values: Sequence[float]) -> str:
    """pgvector's text input format.

    Sent as text and cast with ``::vector`` so the query path does not require
    the ``pgvector`` Python adapter to be registered on every connection. One
    fewer piece of per-connection setup that can be forgotten.
    """
    return "[" + ",".join(f"{float(v):.9g}" for v in values) + "]"


class PgVectorStore:
    """pgvector search with the permission filter enforced twice, independently.

    Implements :class:`~sightline.store.base.VectorStore`. Every strategy maps to
    both an application predicate and a session setting the RLS policy reads:

    ``ENUMERATE``
        ``sightline.enumerate='on'`` plus the ref list. Both layers check it.
    ``GRANT_TOKENS``
        The token array, intersected with ``&&`` on both layers.
    ``UNFILTERED``
        :data:`WILDCARD_TOKEN`. The policy stays on; there is no bypass.
    ``EXACT_SCAN``
        Same predicates, plus ``SET LOCAL enable_indexscan = off`` so the planner
        does a sequential scan and the result is exact rather than HNSW-approximate.
        This is the only path on this backend where strict top-k equality with
        the oracle is a legitimate assertion (FR-14).
    """

    name = "pgvector"

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        connection: Any | None = None,
        dim: int | None = None,
        apply_schema: bool = False,
    ) -> None:
        """Open (or adopt) a connection.

        Args:
            conninfo: libpq connection string. Ignored when ``connection`` is given.
            connection: An existing ``psycopg.Connection``, which is how tests
                hand in a transaction they intend to roll back.
            dim: Vector dimensionality. Required to apply the schema.
            apply_schema: Run :data:`SCHEMA_DDL` on construction.

        Raises:
            ImportError: If ``psycopg`` is not installed.
            ValueError: If ``apply_schema`` is set without ``dim``.
        """
        _require_psycopg()
        self._dim = dim
        self._owns_connection = connection is None
        self.conn = connection if connection is not None else psycopg.connect(conninfo or "")
        if apply_schema:
            if dim is None:
                raise ValueError("apply_schema needs dim: the vector column is fixed-width")
            self.apply_schema(dim)

    def apply_schema(self, dim: int) -> None:
        """Create the table, indexes, roles and the RLS policy. Idempotent."""
        _require_psycopg()
        self._dim = dim
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA_DDL.format(dim=dim))
        self.conn.commit()

    def close(self) -> None:
        if self._owns_connection:
            self.conn.close()

    # ---------------------------------------------------------------- writes

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Insert or replace rows, keyed on ``chunk_id``. Requires the ingest role."""
        _require_psycopg()
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks and {len(vectors)} vectors")
        if not chunks:
            return
        rows = []
        for chunk, vec in zip(chunks, vectors):
            normalised = normalise(vec, self._dim)
            rows.append(
                (
                    chunk.id,
                    str(chunk.object),
                    chunk.object.id,
                    chunk.text,
                    sorted(str(t) for t in chunk.grant_tokens),
                    json.dumps(dict(chunk.metadata)),
                    _vector_literal(normalised.tolist()),
                )
            )
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO sightline_chunk
                    (chunk_id, object_ref, object_id, body, grant_tokens, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    object_ref   = EXCLUDED.object_ref,
                    object_id    = EXCLUDED.object_id,
                    body         = EXCLUDED.body,
                    grant_tokens = EXCLUDED.grant_tokens,
                    metadata     = EXCLUDED.metadata,
                    embedding    = EXCLUDED.embedding
                """,
                rows,
            )
        self.conn.commit()

    # ----------------------------------------------------------------- reads

    def session_settings(self, plan: FilterPlan) -> dict[str, str]:
        """The GUCs the RLS policy reads, compiled from a plan.

        Comma-joined because a GUC is a single text value; grant tokens are keyed
        hashes over a fixed alphabet and object refs are ``namespace:id``, so
        neither can contain a comma. If that ever stops being true this becomes a
        delimiter injection, which is why it is asserted rather than assumed.
        """
        by_ids, by_tokens = plan_bindings(plan)
        tokens: list[str] = []
        if plan.strategy is PlanStrategy.UNFILTERED:
            tokens = [WILDCARD_TOKEN]
        elif by_tokens:
            tokens = sorted(str(t) for t in plan.grant_tokens)
        settings = {
            "sightline.grant_tokens": ",".join(tokens),
            "sightline.enumerate": "on" if by_ids else "off",
            "sightline.object_refs": ",".join(sorted(plan.explicit_ids)) if by_ids else "",
        }
        for value in (*tokens, *(plan.explicit_ids if by_ids else ())):
            if "," in value:
                raise ValueError(f"plan value contains the GUC delimiter: {value!r}")
        return settings

    def _apply_settings(self, cur: Any, plan: FilterPlan) -> None:
        for key, value in self.session_settings(plan).items():
            # is_local => true: discarded at commit, so a pooled connection never
            # carries one principal's permissions into the next request.
            cur.execute("SELECT set_config(%s, %s, true)", (key, value))
        if plan.strategy is PlanStrategy.EXACT_SCAN:
            cur.execute("SET LOCAL enable_indexscan = off")

    def _app_predicate(self, plan: FilterPlan) -> tuple[str, list[Any]]:
        """The application-side filter. Redundant with RLS, and deliberately so.

        Two independent implementations of the same rule, in two languages, in
        two processes. They are checked against each other by the conformance
        suite, and a disagreement is a bug in whichever one is wrong — which is
        information you do not get from a single implementation.
        """
        by_ids, by_tokens = plan_bindings(plan)
        clauses: list[str] = []
        params: list[Any] = []
        if by_tokens:
            clauses.append("grant_tokens && %s::text[]")
            params.append(sorted(str(t) for t in plan.grant_tokens))
        if by_ids:
            clauses.append("(object_ref = ANY(%s::text[]) OR object_id = ANY(%s::text[]))")
            refs = sorted(plan.explicit_ids)
            params.extend([refs, refs])
        if not clauses:
            if plan.strategy is PlanStrategy.UNFILTERED:
                return "TRUE", []
            return "FALSE", []  # binds nothing and is not unfiltered: admit nothing.
        return " AND ".join(clauses), params

    def search(
        self, query_vector: Sequence[float], plan: FilterPlan, k: int
    ) -> list[UncheckedHit]:
        """Top-k over the permitted subset, filtered by the application *and* the database."""
        return self._search(query_vector, plan, k, app_filter=True)

    def search_with_app_filter_removed(
        self, query_vector: Sequence[float], plan: FilterPlan, k: int
    ) -> list[UncheckedHit]:
        """Run the query with the application predicate deleted. **Test hook only.**

        This is the FR-10 proof: it simulates the exact bug the second layer
        exists for — an engineer drops the ``WHERE`` clause — and the result must
        still contain zero rows the principal may not see, because the RLS policy
        refuses them. Never call this from anything that serves a user.
        """
        return self._search(query_vector, plan, k, app_filter=False)

    def _search(
        self,
        query_vector: Sequence[float],
        plan: FilterPlan,
        k: int,
        *,
        app_filter: bool,
    ) -> list[UncheckedHit]:
        _require_psycopg()
        if k <= 0:
            return []
        query = _vector_literal(normalise(query_vector, self._dim).tolist())
        where, params = self._app_predicate(plan) if app_filter else ("TRUE", [])
        sql = f"""
            SELECT chunk_id, object_ref, body, grant_tokens,
                   1 - (embedding <=> %s::vector) AS score
              FROM sightline_chunk
             WHERE {where}
             ORDER BY embedding <=> %s::vector
             LIMIT %s
        """
        with self.conn.transaction(), self.conn.cursor() as cur:
            self._apply_settings(cur, plan)
            cur.execute(sql, [query, *params, query, k])
            rows = cur.fetchall()
        return [
            UncheckedHit(
                chunk_id=row[0],
                score=float(row[4]),
                text=row[2],
                object_ref=row[1],
                matched_token=first_matching_token(plan, row[3] or ()),
            )
            for row in rows
        ]

    def count_matching(self, plan: FilterPlan) -> int:
        _require_psycopg()
        where, params = self._app_predicate(plan)
        with self.conn.transaction(), self.conn.cursor() as cur:
            self._apply_settings(cur, plan)
            cur.execute(f"SELECT count(*) FROM sightline_chunk WHERE {where}", params)
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def stats(self) -> StoreStats:
        _require_psycopg()
        # Transaction-scoped, like every other read: a transaction-local GUC set
        # outside a transaction is discarded before the next statement runs, and
        # the count would silently come back 0 under RLS.
        with self.conn.transaction(), self.conn.cursor() as cur:
            # Counting rows needs a session that can see them; the wildcard is
            # the honest way to say "this is an operational metric, not a read".
            cur.execute(
                "SELECT set_config('sightline.grant_tokens', %s, true)", (WILDCARD_TOKEN,)
            )
            cur.execute("SELECT count(*) FROM sightline_chunk")
            n = int((cur.fetchone() or [0])[0])
            cur.execute("SELECT pg_total_relation_size('sightline_chunk')")
            disk = int((cur.fetchone() or [0])[0])
        return StoreStats(
            n_vectors=n,
            resident_bytes=0,  # Postgres owns its buffers; do not pretend to know.
            disk_bytes=disk,
            backend=self.name,
            detail={
                "row_level_security": "forced",
                "policy": "sightline_chunk_visible",
                "app_role": APP_ROLE,
                "ingest_role": INGEST_ROLE,
                "enforcement_layers": "2",
                "exact": "only under EXACT_SCAN",
                "dim": str(self._dim or 0),
            },
        )
