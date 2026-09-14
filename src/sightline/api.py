"""The HTTP surface. A thin wrapper over :mod:`sightline.retrieve`, on purpose.

Every endpoint here does three things and no more: resolve the principal from a
verified signature, call one function in the library, and serialise the result.
Anything that looks like a decision — which strategy to use, whether a plan is
stale, whether a hit may be shown — happens below this layer, because a rule that
lives in a request handler is a rule that is not enforced when somebody calls the
library directly.

Four properties hold across the whole surface:

* **Identity comes only from the token signature** (SR-18). There is no
  ``principal`` field in any request body, no header that names the caller, and
  no query parameter that selects one. ``/v1/explain`` takes a ``principal``
  argument because it asks a question *about* somebody else's access, and it is
  therefore an administrative read gated by a ``check()`` against the admin
  relation.
* **Every response carries ``strategy`` and ``epoch``.** How you were authorised
  and under which policy version. An answer you cannot place in time is an answer
  you cannot audit.
* **Citations carry ``why_allowed``**, taken from the recheck derivation, so the
  UI can answer "why am I allowed to see this" without a second authorisation
  pass (FR-21).
* **No chunk text in any non-grounded response.** Refusals and errors carry a
  reason code and nothing else. Document content leaking through the error path
  is planted mutant M12, and it is the realistic way this kind of system leaks.

Administrative endpoints authorise through the same ``check()`` as everything
else (SR-19). There is no admin bypass, because a bypass is an untested code path
with maximum privilege.
"""

from __future__ import annotations

import json
import time
from typing import Annotated, Any, Iterator, Literal

from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from sightline import obs
from sightline.audit import AuditSink, JsonlAuditLog, MemoryAuditLog
from sightline.auth import TokenVerifier
from sightline.authz.check import check, explain
from sightline.authz.compile import PlanCompiler
from sightline.authz.recheck import LiveRechecker
from sightline.authz.tuples import MemoryTupleStore, SQLiteTupleStore, TupleStore, parse_tuples
from sightline.errors import (
    AuthorityUnavailable,
    MissingCredentials,
    PermissionDenied,
    SightlineError,
)
from sightline.generate import build_chain
from sightline.retrieve import PipelineEvent, QueryResult, Retriever, SearchResult
from sightline.settings import Settings, load_settings
from sightline.types import ObjectRef, PrincipalRef

__all__ = ["create_app", "build_retriever", "app"]

API_VERSION = "1"


# --------------------------------------------------------------------------
# Wire models
# --------------------------------------------------------------------------


class AskRequest(BaseModel):
    """A question. Note the absence of a principal field — that is the point."""

    model_config = ConfigDict(extra="forbid")

    q: str = Field(description="The question. Length-capped; over the cap refuses.")
    k: int | None = Field(
        default=None,
        description="Candidates to retrieve. Capped server-side; k is a lever on "
        "how many permission checks one request can demand.",
    )
    max_context_chars: int | None = Field(
        default=None,
        description="Hard cap on evidence characters sent to the model. Over "
        "budget refuses; it never drops checks to fit.",
    )


class CitationModel(BaseModel):
    chunk_id: str
    object: str = Field(description="Object reference, e.g. 'doc:1042'.")
    score: float
    why_allowed: str | None = Field(
        default=None,
        description="The derivation from the live recheck that admitted this "
        "chunk — not the index's advisory matched_token.",
    )


class DiagnosticsModel(BaseModel):
    candidates: int
    dropped_at_recheck: int = Field(
        description="Candidates the live permission check removed. The "
        "stale-index gap, per request."
    )
    plan_ms: float
    search_ms: float
    recheck_ms: float
    guard_ms: float
    synth_ms: float
    total_ms: float
    plan_cache: str
    provider: str
    degraded: bool = Field(
        description="True when no language model answered and the extractive "
        "arm produced the claims."
    )
    guardrails_fired: list[str]
    policy_moved: bool


class AnswerModel(BaseModel):
    """Grounded or refused. There is no third state."""

    text: str
    citations: list[CitationModel]
    refused: bool
    refusal_reason: str | None
    existence_protected: bool = Field(
        description="True when the system declined to say whether evidence even "
        "exists. 'No such document' and 'not for you' are the same answer here."
    )
    strategy: str = Field(description="PlanStrategy that executed this query.")
    epoch: int = Field(description="Policy epoch the answer was served under.")
    diagnostics: DiagnosticsModel
    request_id: str


class SearchHitModel(BaseModel):
    chunk_id: str
    score: float
    text: str
    object_ref: str
    why_allowed: str
    checked_at_epoch: int


class SearchResponse(BaseModel):
    hits: list[SearchHitModel]
    strategy: str
    epoch: int
    refused: bool
    refusal_reason: str | None
    existence_protected: bool
    diagnostics: DiagnosticsModel
    request_id: str


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object: str = Field(description="Object reference, e.g. 'doc:1042'.")
    relation: str = Field(default="viewer")
    principal: str | None = Field(
        default=None,
        description="Whose access to check. Omit for your own. Naming somebody "
        "else is an administrative read and is checked as one.",
    )


class CheckResponse(BaseModel):
    allowed: bool
    why: list[str] = Field(
        description="The derivation. For an allow, tuples that replay to the "
        "same answer one at a time; for a deny, the edges that were tried."
    )
    checked_tuples: int
    epoch: int
    principal: str
    object: str
    relation: str


class ExplainResponse(CheckResponse):
    depth_reached: int
    tree: dict[str, Any] = Field(description="The full userset expansion.")


class PlanResponse(BaseModel):
    """Counts, never tokens. A plan dump hands an attacker the filter."""

    strategy: str
    epoch: int
    estimated_cardinality: int
    n_grant_tokens: int
    n_explicit_ids: int


class TupleWriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    writes: list[str] = Field(default_factory=list, examples=[["doc:42#viewer@user:alice"]])
    deletes: list[str] = Field(default_factory=list)


class TupleWriteResponse(BaseModel):
    epoch: int = Field(description="Epoch after the write. Incremented in the "
                                   "same transaction as the tuples themselves.")
    applied: int


class TupleListResponse(BaseModel):
    tuples: list[str]
    epoch: int


class EpochResponse(BaseModel):
    epoch: int


class AuditRowModel(BaseModel):
    seq: int
    prev: str
    digest: str
    row: dict[str, Any]


class AuditResponse(BaseModel):
    rows: list[AuditRowModel]
    head: str
    chain: dict[str, Any] = Field(
        description="Result of recomputing the hash chain over the returned "
        "rows. ok=false means the log was edited or truncated."
    )
    total: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, str]
    epoch: int | None = None


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def _build_tuple_store(settings: Settings) -> TupleStore:
    if settings.store.tuple_store in ("", "memory", ":memory:"):
        return MemoryTupleStore()
    return SQLiteTupleStore(settings.store.tuple_store)


def _build_vector_store(settings: Settings) -> Any:
    from sightline.store import get_store

    name = settings.store.vector_backend
    kwargs: dict[str, Any] = {}
    if name in ("qdrant", "pgvector", "postgres", "pg") and settings.store.vector_url:
        kwargs["url"] = settings.store.vector_url
    return get_store(name, **kwargs)


def _build_audit(settings: Settings) -> AuditSink:
    if settings.audit.path:
        return JsonlAuditLog(settings.audit.path, fsync=settings.audit.fsync)
    return MemoryAuditLog()


def build_retriever(settings: Settings) -> Retriever:
    """Construct the pipeline from settings.

    The one place the concrete backends are chosen. Everything downstream takes
    them as constructor arguments, which is why the test suite can run the entire
    pipeline in memory with no services and no network.
    """
    from sightline.ingest.embed import load_embedder

    tuples = _build_tuple_store(settings)
    vectors = _build_vector_store(settings)
    embedder = load_embedder(settings.store.embedder, quiet=True)
    return Retriever(
        tuple_store=tuples,
        vector_store=vectors,
        embedder=embedder,
        compiler=PlanCompiler(tuples),
        rechecker=LiveRechecker(tuples),
        generator=build_chain(settings.generation),
        audit=_build_audit(settings),
        settings=settings,
    )


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise MissingCredentials("missing bearer token")
    return token.strip()


def _principal(request: Request) -> PrincipalRef:
    """Resolve the caller. The only source of identity in this application.

    Verified here even though the edge also verifies — see
    :mod:`sightline.auth` for why that is not redundant.
    """
    verifier: TokenVerifier = request.app.state.verifier
    principal, claims = verifier.principal(_bearer(request))
    request.state.claims = claims
    return principal


CallerDep = Annotated[PrincipalRef, Depends(_principal)]


def _require_admin(request: Request, caller: PrincipalRef) -> None:
    """Authorise an administrative operation through the ordinary check path.

    Not a role claim from the token, not an allowlist in configuration: a tuple,
    evaluated by the same evaluator that decides whether you may read a document.
    One authorisation path means one path to test, and SR-19 says the bypass that
    would be quicker here is exactly the untested code path with the most
    privilege.
    """
    settings: Settings = request.app.state.settings
    retriever: Retriever = request.app.state.retriever
    decision = check(
        retriever.tuple_store,
        ObjectRef.parse(settings.auth.admin_object),
        settings.auth.admin_relation,
        caller,
    )
    if not decision.allowed:
        raise PermissionDenied(
            f"{caller} lacks {settings.auth.admin_relation} on {settings.auth.admin_object}"
        )


# --------------------------------------------------------------------------
# Serialisation helpers
# --------------------------------------------------------------------------


def _diag_model(result: Any) -> DiagnosticsModel:
    return DiagnosticsModel(**result.diagnostics.to_dict())


def _answer_model(result: QueryResult) -> AnswerModel:
    answer = result.answer
    return AnswerModel(
        text=answer.text,
        citations=[
            CitationModel(
                chunk_id=c.chunk_id,
                object=str(c.object),
                score=c.score,
                why_allowed=c.why_allowed,
            )
            for c in answer.citations
        ],
        refused=answer.refused,
        refusal_reason=answer.refusal_reason.value if answer.refusal_reason else None,
        existence_protected=answer.existence_protected,
        strategy=answer.strategy.value if answer.strategy else "unknown",
        epoch=answer.epoch,
        diagnostics=_diag_model(result),
        request_id=result.request_id,
    )


def _search_model(result: SearchResult) -> SearchResponse:
    return SearchResponse(
        hits=[
            SearchHitModel(
                chunk_id=h.chunk_id,
                score=h.score,
                text=h.text,
                object_ref=h.object_ref,
                why_allowed=h.why_allowed,
                checked_at_epoch=h.checked_at_epoch,
            )
            for h in result.hits
        ],
        strategy=result.strategy.value,
        epoch=result.epoch,
        refused=result.refused,
        refusal_reason=result.refusal_reason.value if result.refusal_reason else None,
        existence_protected=result.existence_protected,
        diagnostics=_diag_model(result),
        request_id=result.request_id,
    )


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    """One server-sent event. ``event:`` names the stage, ``data:`` is JSON."""
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _stream(events: Iterator[PipelineEvent]) -> Iterator[bytes]:
    """Translate pipeline events into SSE frames.

    The stream carries stage transitions and then the finished answer. It never
    carries model tokens, because grounding and citation reconstruction run after
    generation completes, and text streamed before those checks is text that
    cannot be taken back.

    A failure after the response has started cannot become a status code — the
    200 is already on the wire — so it becomes an ``error`` event carrying the
    same content-free payload the exception handler would have returned. The
    stream then ends normally, because a client that sees ``done`` knows it has
    the whole story and a client that sees a truncated stream does not.
    """
    try:
        for event in events:
            if event.stage == "result":
                result: QueryResult = event.data["result"]
                yield _sse("answer", _answer_model(result).model_dump())
                yield _sse("done", {"request_id": result.request_id})
                return
            yield _sse(event.stage, event.data)
    except SightlineError as exc:
        yield _sse("error", exc.as_payload() | {"refused": True})
        yield _sse("done", {})


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    *,
    retriever: Retriever | None = None,
    verifier: TokenVerifier | None = None,
) -> FastAPI:
    """Build the ASGI app.

    Args:
        settings: Resolved configuration. Loaded from the environment if absent.
        retriever: Prebuilt pipeline. Tests pass one with in-memory backends;
            production lets :func:`build_retriever` wire it from settings.
        verifier: Prebuilt token verifier, for the same reason.

    Returns:
        A :class:`fastapi.FastAPI` with the dependencies on ``app.state``.
    """
    cfg = settings or load_settings()
    obs.configure(cfg.obs)

    app = FastAPI(
        title="Sightline",
        version="0.1.0",
        description=(
            "Internal AI search that can only see what you are allowed to see. "
            "Search returns candidates; the live permission database decides what "
            "is served. A stale index therefore loses results and never leaks them."
        ),
    )
    app.state.settings = cfg
    app.state.retriever = retriever or build_retriever(cfg)
    app.state.verifier = verifier or TokenVerifier(
        cfg.auth, allow_dev_tokens=cfg.allow_dev_tokens
    )
    app.state.started_at = time.time()

    _register_errors(app)
    _register_routes(app)
    return app


def _register_errors(app: FastAPI) -> None:
    """Map library exceptions to responses that say nothing they should not.

    The body is the exception's fixed public message plus a machine code. The
    operator-facing detail goes nowhere near it: an error payload is a disclosure
    surface, and the refusal-path contract test asserts no chunk text appears in
    any non-grounded response.
    """

    @app.exception_handler(SightlineError)
    async def _handle(request: Request, exc: SightlineError) -> JSONResponse:
        payload = exc.as_payload()
        # Refusal-shaped errors keep the Answer vocabulary so a client has one
        # branch for "you got nothing", not two.
        if exc.code in ("stale_policy", "budget_exceeded"):
            payload |= {"refused": True, "refusal_reason": exc.code}
        retriever: Retriever | None = getattr(request.app.state, "retriever", None)
        if retriever is not None:
            try:
                payload["epoch"] = retriever.tuple_store.epoch()
            except Exception:  # noqa: BLE001 - the epoch is a nicety here
                pass
        route = request.scope.get("route")
        obs.http_requests_total.inc(
            route=getattr(route, "path", "unknown"), status=f"{exc.status_code // 100}xx"
        )
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return JSONResponse(payload, status_code=exc.status_code, headers=headers)


def _register_routes(app: FastAPI) -> None:
    @app.post(
        "/v1/ask",
        summary="Ask a question, streamed as server-sent events",
        description=(
            "Streams pipeline stages (plan, search, recheck, guardrails, synth) "
            "and then the finished answer. Model tokens are deliberately not "
            "streamed: grounding and citation reconstruction run after generation, "
            "and a refusal you have already displayed half of is not a refusal."
        ),
        response_class=StreamingResponse,
    )
    async def ask_stream(body: AskRequest, request: Request, caller: CallerDep) -> Response:
        retriever: Retriever = request.app.state.retriever
        # Budget checks run here, before a single byte of the 200 is committed.
        # Inside the generator they would arrive as a truncated stream instead of
        # a 429, which is an error only a careful client would ever notice.
        await run_in_threadpool(
            retriever.validate_request,
            body.q,
            body.k,
            body.max_context_chars or retriever.settings.retrieval.default_max_context_chars,
        )
        events = retriever.events(
            body.q, caller, k=body.k, max_context_chars=body.max_context_chars
        )
        # The pipeline is synchronous — permission checks and vector search are
        # CPU and I/O bound in ordinary blocking code. Iterating it in a worker
        # thread keeps the event loop free instead of pretending it is async.
        return StreamingResponse(
            iterate_in_threadpool(_stream(events)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post(
        "/v1/query",
        response_model=AnswerModel,
        summary="Ask a question, one JSON response",
        description="The non-streaming form of /v1/ask. Identical pipeline.",
    )
    async def query(body: AskRequest, request: Request, caller: CallerDep) -> AnswerModel:
        retriever: Retriever = request.app.state.retriever
        result = await run_in_threadpool(
            retriever.ask, body.q, caller, k=body.k, max_context_chars=body.max_context_chars
        )
        obs.http_requests_total.inc(route="/v1/query", status="2xx")
        return _answer_model(result)

    @app.post(
        "/v1/search",
        response_model=SearchResponse,
        summary="Retrieve permitted passages, no generation",
        description=(
            "Same guarantees as /v1/query minus the model: every returned hit "
            "passed the live permission recheck, and the count that did not is in "
            "diagnostics.dropped_at_recheck."
        ),
    )
    async def search(body: AskRequest, request: Request, caller: CallerDep) -> SearchResponse:
        retriever: Retriever = request.app.state.retriever
        result = await run_in_threadpool(retriever.search, body.q, caller, k=body.k)
        obs.http_requests_total.inc(route="/v1/search", status="2xx")
        return _search_model(result)

    @app.post(
        "/v1/check",
        response_model=CheckResponse,
        summary="Explain one authorisation decision",
        description=(
            "The authoritative check, with its derivation. Checking your own "
            "access needs nothing; checking somebody else's is an administrative "
            "read, because the derivation names groups and naming groups to "
            "someone who cannot see the object is its own small disclosure."
        ),
    )
    async def check_endpoint(
        body: CheckRequest, request: Request, caller: CallerDep
    ) -> CheckResponse:
        retriever: Retriever = request.app.state.retriever
        subject = caller
        if body.principal:
            subject = PrincipalRef.parse(body.principal)
            if subject != caller:
                _require_admin(request, caller)
        decision = await run_in_threadpool(
            check, retriever.tuple_store, ObjectRef.parse(body.object), body.relation, subject
        )
        obs.http_requests_total.inc(route="/v1/check", status="2xx")
        return CheckResponse(
            allowed=decision.allowed,
            why=list(decision.why),
            checked_tuples=decision.checked_tuples,
            epoch=retriever.epoch(),
            principal=str(subject),
            object=body.object,
            relation=body.relation,
        )

    @app.get(
        "/v1/explain",
        response_model=ExplainResponse,
        summary="Full derivation tree for a decision (admin)",
    )
    async def explain_endpoint(
        request: Request,
        caller: CallerDep,
        object: Annotated[str, Query(examples=["doc:42"])],
        relation: str = "viewer",
        principal: str | None = None,
    ) -> ExplainResponse:
        _require_admin(request, caller)
        retriever: Retriever = request.app.state.retriever
        subject = PrincipalRef.parse(principal) if principal else caller
        result = await run_in_threadpool(
            explain, retriever.tuple_store, ObjectRef.parse(object), relation, subject
        )
        obs.http_requests_total.inc(route="/v1/explain", status="2xx")
        return ExplainResponse(
            allowed=result.allowed,
            why=list(result.why),
            checked_tuples=result.checked_tuples,
            depth_reached=result.depth_reached,
            tree=result.tree.to_dict(),
            epoch=retriever.epoch(),
            principal=str(subject),
            object=object,
            relation=relation,
        )

    @app.get(
        "/v1/plan",
        response_model=PlanResponse,
        summary="Your own compiled plan, as counts",
        description=(
            "Counts, never tokens (guardrail 12). Returning the compiled filter "
            "would hand the caller the exact predicate the index evaluates."
        ),
    )
    async def plan(request: Request, caller: CallerDep) -> PlanResponse:
        retriever: Retriever = request.app.state.retriever
        compiled, _cache, _live = await run_in_threadpool(retriever.plan_for, caller)
        obs.http_requests_total.inc(route="/v1/plan", status="2xx")
        return PlanResponse(
            strategy=compiled.strategy.value,
            epoch=compiled.epoch,
            estimated_cardinality=compiled.estimated_cardinality,
            n_grant_tokens=len(compiled.grant_tokens),
            n_explicit_ids=len(compiled.explicit_ids),
        )

    @app.get("/v1/policy/epoch", response_model=EpochResponse, summary="Current policy epoch")
    async def epoch(request: Request, caller: CallerDep) -> EpochResponse:
        retriever: Retriever = request.app.state.retriever
        return EpochResponse(epoch=retriever.epoch())

    @app.post(
        "/v1/tuples",
        response_model=TupleWriteResponse,
        summary="Write and delete permission tuples (admin)",
        description=(
            "Writes and deletes are applied with the epoch increment in the same "
            "transaction (FR-23). A deletion of a tuple that does not exist still "
            "increments the epoch: cheap to over-increment, unsafe to under-increment."
        ),
    )
    async def write_tuples(
        body: TupleWriteRequest, request: Request, caller: CallerDep
    ) -> TupleWriteResponse:
        _require_admin(request, caller)
        retriever: Retriever = request.app.state.retriever
        writes = parse_tuples(body.writes)
        deletes = parse_tuples(body.deletes)
        epoch_after, applied = await run_in_threadpool(
            retriever.tuple_store.apply, writes, deletes
        )
        obs.http_requests_total.inc(route="/v1/tuples", status="2xx")
        return TupleWriteResponse(epoch=epoch_after, applied=applied)

    @app.get(
        "/v1/tuples",
        response_model=TupleListResponse,
        summary="Read the tuples on one object (admin)",
    )
    async def read_tuples(
        request: Request,
        caller: CallerDep,
        object: Annotated[str, Query(examples=["doc:42"])],
        relation: str | None = None,
    ) -> TupleListResponse:
        _require_admin(request, caller)
        retriever: Retriever = request.app.state.retriever
        rows = await run_in_threadpool(
            retriever.tuple_store.read, ObjectRef.parse(object), relation
        )
        return TupleListResponse(tuples=[str(t) for t in rows], epoch=retriever.epoch())

    @app.get(
        "/v1/audit",
        response_model=AuditResponse,
        summary="Read the hash-chained audit log (admin)",
        description=(
            "Rows plus a recomputed chain verification. Query text is not stored, "
            "only a keyed hash. The pair to look at is retrieved_objects versus "
            "shown_objects: the difference is what the live check rejected."
        ),
    )
    async def audit(
        request: Request,
        caller: CallerDep,
        limit: int = 100,
        after_seq: int = -1,
    ) -> AuditResponse:
        _require_admin(request, caller)
        settings: Settings = request.app.state.settings
        retriever: Retriever = request.app.state.retriever
        capped = max(1, min(limit, settings.audit.max_page))
        rows = retriever.audit.read(limit=capped, after_seq=after_seq)
        verification = retriever.audit.verify()
        if not verification.ok:
            obs.audit_chain_failures_total.inc()
        return AuditResponse(
            rows=[
                AuditRowModel(seq=r.seq, prev=r.prev, digest=r.digest, row=r.row) for r in rows
            ],
            head=retriever.audit.head,
            chain=verification.to_dict(),
            total=len(retriever.audit),
        )

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        summary="Liveness",
        description="No auth, and no policy state in the body: a liveness probe "
        "that reports the epoch is an unauthenticated read of policy metadata.",
    )
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok", version=API_VERSION)

    @app.get(
        "/readyz",
        response_model=ReadyResponse,
        summary="Readiness",
        description=(
            "Tuple store reachable, epoch readable, index reachable. Not ready "
            "means not serving: every one of those failures is a refusal, and an "
            "instance that refuses everything should be out of rotation rather "
            "than degrading."
        ),
    )
    async def readyz(request: Request, response: Response) -> ReadyResponse:
        retriever: Retriever = request.app.state.retriever
        checks: dict[str, str] = {}
        epoch_value: int | None = None
        try:
            epoch_value = retriever.epoch()
            checks["tuple_store"] = "ok"
        except AuthorityUnavailable as exc:
            checks["tuple_store"] = f"unavailable: {exc.code}"
        try:
            retriever.vector_store.stats()
            checks["index"] = "ok"
        except Exception as exc:  # noqa: BLE001 - any failure is not-ready
            checks["index"] = f"unavailable: {type(exc).__name__}"
        ready = all(v == "ok" for v in checks.values())
        if not ready:
            response.status_code = 503
        return ReadyResponse(ready=ready, checks=checks, epoch=epoch_value)

    @app.get(
        "/metrics",
        summary="Prometheus metrics",
        description=(
            "Real client output when the 'obs' extra is installed, and the same "
            "in-process counters rendered in the text format when it is not. No "
            "principal appears as a label anywhere: strategy is a label, identity "
            "is a span attribute."
        ),
    )
    async def metrics() -> Response:
        body, content_type = obs.render_metrics()
        return Response(content=body, media_type=content_type)


def _default_app() -> FastAPI:
    """Lazily built module-level app, for ``uvicorn sightline.api:app``."""
    return create_app()


def __getattr__(name: str) -> Any:
    """Build the module-level ``app`` on first access (PEP 562).

    Importing this module must not construct a vector store, read the
    environment, or touch the filesystem — the test suite imports it to build its
    own app with in-memory backends. ``uvicorn sightline.api:app`` still works,
    because that is an attribute access and this is where it lands.
    """
    if name == "app":
        built = _default_app()
        globals()["app"] = built
        return built
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
