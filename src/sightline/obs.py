"""Traces and metrics, with the permission decision visible in both.

A trace that carries only span names and durations tells you where the time went
and nothing about why. The spans here carry the *decision*: which plan strategy
fired, how many grant tokens it compiled to, how many candidates the live recheck
threw away, and the policy epoch the whole thing ran under. ``recheck.dropped``
is the single most valuable number on the page — it is the stale-index gap, made
visible per request.

Three rules for attributes, and they are enforced here rather than remembered:

* **Counts, never contents.** ``authz.n_grant_tokens=37``, never the tokens. A
  plan dump hands an attacker the filter (guardrail 12).
* **Hash, never text.** ``query.hash``, never ``query.text``. An audit trail of
  everyone's questions is a liability created out of nothing (SR-12).
* **No chunk text in telemetry, ever.** Traces and logs land in tooling with a
  completely different access model to the document store. Copying a carefully
  permissioned paragraph into a system where all of engineering is an admin is
  the leak path the permission architecture does not cover, and it is a one-line
  mistake. Planted mutant M12 is exactly this.

Cardinality: **strategy is a label, principal is not.** 50,000 principals times 4
strategies is up to 200,000 active series, which at Prometheus's rough 1-3 KB of
scraper memory per series is most of a gigabyte for one metric. High-cardinality
identity belongs in a span or a log line. :func:`_check_labels` refuses the known
offenders at call time, because this mistake is always made in a hurry.

Optional dependencies: ``opentelemetry`` and ``prometheus_client`` live in the
``obs`` extra. Without them this module still works — spans become no-ops that
still validate their attributes, and the counters are kept in process so
``cross_tenant_leak_total`` is assertable in a test suite running on the four core
dependencies. The fallback exposition format is a genuine render of real
counters; it is not a Prometheus client and does not pretend to be one.

**Not built, deliberately:** tail sampling. Head sampling at 1% will miss a
once-a-day breach roughly two times in three. The right answer is to buffer spans
and keep 100% of the traces that refused, errored, blew the latency budget or
dropped anything at recheck. That needs a collector-side policy, it is written up
in ``docs/guide/07-guardrails-observability-eval.md``, and it is not in this file.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ALL_METRICS",
    "HAVE_OTEL",
    "HAVE_PROMETHEUS",
    "PROMETHEUS_CONTENT_TYPE",
    "Counter",
    "Gauge",
    "Histogram",
    "Metric",
    "SpanHandle",
    "audit_appends_total",
    "audit_chain_failures_total",
    "configure",
    "cross_tenant_leak_total",
    "current_span",
    "http_requests_total",
    "note_leak",
    "plan_cache_total",
    "provider_requests_total",
    "provider_tokens_total",
    "queries_total",
    "recheck_checked_total",
    "recheck_dropped_total",
    "record_generation",
    "record_plan",
    "record_recheck",
    "record_search",
    "refusals_total",
    "render_metrics",
    "span",
    "stage_seconds",
]

# --------------------------------------------------------------------------
# Optional dependencies
# --------------------------------------------------------------------------

try:  # pragma: no cover - exercised by the obs-extra CI job
    from opentelemetry import trace as _otel_trace

    HAVE_OTEL = True
except ImportError:  # pragma: no cover - the core-deps CI job takes this path
    _otel_trace = None  # type: ignore[assignment]
    HAVE_OTEL = False

try:  # pragma: no cover - exercised by the obs-extra CI job
    import prometheus_client as _prom

    HAVE_PROMETHEUS = True
except ImportError:  # pragma: no cover
    _prom = None  # type: ignore[assignment]
    HAVE_PROMETHEUS = False

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# --------------------------------------------------------------------------
# Cardinality guard
# --------------------------------------------------------------------------

#: Label names that would blow the series count or copy identity into a metrics
#: backend. The list is short because it only needs to contain the mistakes
#: people actually make at 3am.
_FORBIDDEN_LABELS = frozenset(
    {
        "principal",
        "principal_id",
        "user",
        "user_id",
        "subject",
        "query",
        "query_hash",
        "question",
        "chunk_id",
        "object",
        "object_ref",
        "text",
        "token",
        "grant_token",
    }
)


def _check_labels(names: Sequence[str]) -> None:
    bad = sorted(set(names) & _FORBIDDEN_LABELS)
    if bad:
        raise ValueError(
            f"refusing metric label(s) {bad}: unbounded cardinality and/or identity in a "
            "metrics backend. Strategy is a label; principal is a span attribute."
        )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Key:
    """A label tuple, hashable and ordered so exposition output is stable."""

    values: tuple[tuple[str, str], ...] = ()

    def render(self) -> str:
        if not self.values:
            return ""
        inner = ",".join(f'{k}="{_escape(v)}"' for k, v in self.values)
        return "{" + inner + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Metric:
    """Base for the three metric kinds. Keeps a local count in every case.

    The local count is not redundant with ``prometheus_client``. It is how a test
    running on the four core dependencies asserts that
    ``cross_tenant_leak_total`` is zero, and how ``/metrics`` says anything at all
    in a deployment without the ``obs`` extra. When the extra *is* installed the
    value is written to both, which costs one dictionary increment.
    """

    kind = "untyped"

    def __init__(self, name: str, documentation: str, labels: Sequence[str] = ()) -> None:
        _check_labels(labels)
        self.name = name
        self.documentation = documentation
        self.labels = tuple(labels)
        self._lock = threading.Lock()
        self._values: dict[_Key, float] = {}
        self._prom: Any = None
        ALL_METRICS.append(self)

    # -- helpers -----------------------------------------------------------
    def _key(self, labels: Mapping[str, str]) -> _Key:
        if set(labels) != set(self.labels):
            raise ValueError(
                f"{self.name} expects labels {sorted(self.labels)}, got {sorted(labels)}"
            )
        return _Key(tuple(sorted((k, str(v)) for k, v in labels.items())))

    def value(self, **labels: str) -> float:
        """Current value for one label set. Present so tests need no scraper."""
        with self._lock:
            return self._values.get(self._key(labels), 0.0)

    def total(self) -> float:
        """Sum across all label sets."""
        with self._lock:
            return sum(self._values.values())

    def samples(self) -> list[tuple[_Key, float]]:
        with self._lock:
            return sorted(self._values.items(), key=lambda kv: kv[0].values)

    def expose(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError


class Counter(Metric):
    """Monotonic count. Anything you would ever alert on is one of these."""

    kind = "counter"

    def __init__(self, name: str, documentation: str, labels: Sequence[str] = ()) -> None:
        super().__init__(name, documentation, labels)
        if HAVE_PROMETHEUS:  # pragma: no cover - obs-extra job
            self._prom = _prom.Counter(name, documentation, list(self.labels))

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise ValueError("a counter does not go down")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount
        if self._prom is not None:  # pragma: no cover - obs-extra job
            target = self._prom.labels(**labels) if self.labels else self._prom
            target.inc(amount)

    def expose(self) -> list[str]:
        out = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} counter"]
        samples = self.samples()
        if not samples and not self.labels:
            # An unlabelled counter publishes its zero. A labelled one publishes
            # nothing until a label set exists, because inventing one would
            # invent a series. cross_tenant_leak_total is unlabelled precisely so
            # that its zero is always visible on the dashboard.
            out.append(f"{self.name} 0")
        for key, val in samples:
            out.append(f"{self.name}{key.render()} {val:g}")
        return out


class Gauge(Metric):
    """A value that goes up and down. Used sparingly; gauges hide history."""

    kind = "gauge"

    def __init__(self, name: str, documentation: str, labels: Sequence[str] = ()) -> None:
        super().__init__(name, documentation, labels)
        if HAVE_PROMETHEUS:  # pragma: no cover - obs-extra job
            self._prom = _prom.Gauge(name, documentation, list(self.labels))

    def set(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = float(value)
        if self._prom is not None:  # pragma: no cover - obs-extra job
            target = self._prom.labels(**labels) if self.labels else self._prom
            target.set(value)

    def expose(self) -> list[str]:
        out = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} gauge"]
        for key, val in self.samples():
            out.append(f"{self.name}{key.render()} {val:g}")
        return out


#: Bucket boundaries in seconds, chosen around the budgets in ``docs/FRD.md``
#: section 5 rather than around round numbers. The quantile inside a bucket is
#: interpolation, not measurement, so the boundaries are dense exactly where the
#: decisions get made: 60 ms (end-to-end p50), 200 ms (p95), 600 ms (with
#: generation). A top bucket of "over one second" cannot tell a p99 of 1.1 s from
#: one of 40 s, so the last finite boundary is 5 s.
DEFAULT_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.06, 0.1, 0.2, 0.4, 0.6, 1.0, 2.5, 5.0)


class Histogram(Metric):
    """Bucketed observations.

    Buckets, not summaries, because percentiles do not average: two instances
    each reporting p99 = 200 ms do not combine to a p99 of 200 ms. Summing bucket
    counts across instances and computing the quantile from the sum is the only
    arithmetic that is actually valid, and it needs the buckets.
    """

    kind = "histogram"

    def __init__(
        self,
        name: str,
        documentation: str,
        labels: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        super().__init__(name, documentation, labels)
        self.buckets = tuple(buckets)
        self._counts: dict[_Key, list[int]] = {}
        self._sums: dict[_Key, float] = {}
        if HAVE_PROMETHEUS:  # pragma: no cover - obs-extra job
            self._prom = _prom.Histogram(
                name, documentation, list(self.labels), buckets=list(self.buckets)
            )

    def observe(self, seconds: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            for i, edge in enumerate(self.buckets):
                if seconds <= edge:
                    counts[i] += 1
            counts[-1] += 1
            self._sums[key] = self._sums.get(key, 0.0) + seconds
            self._values[key] = counts[-1]
        if self._prom is not None:  # pragma: no cover - obs-extra job
            target = self._prom.labels(**labels) if self.labels else self._prom
            target.observe(seconds)

    @contextmanager
    def time(self, **labels: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - started, **labels)

    def expose(self) -> list[str]:
        out = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} histogram"]
        with self._lock:
            snapshot = sorted(self._counts.items(), key=lambda kv: kv[0].values)
            sums = dict(self._sums)
        for key, counts in snapshot:
            base = list(key.values)
            for edge, count in zip(self.buckets, counts):
                labels = _Key(tuple(base + [("le", f"{edge:g}")]))
                out.append(f"{self.name}_bucket{labels.render()} {count}")
            inf = _Key(tuple(base + [("le", "+Inf")]))
            out.append(f"{self.name}_bucket{inf.render()} {counts[-1]}")
            out.append(f"{self.name}_sum{key.render()} {sums.get(key, 0.0):g}")
            out.append(f"{self.name}_count{key.render()} {counts[-1]}")
        return out


#: Every metric registers itself here at construction, so ``render_metrics`` does
#: not need a registry argument and a new metric cannot be forgotten.
ALL_METRICS: list[Metric] = []

# --------------------------------------------------------------------------
# The metrics themselves
# --------------------------------------------------------------------------

queries_total = Counter(
    "sightline_queries_total",
    "Queries served, by plan strategy and outcome. A refusal with a correct "
    "reason code is an outcome, not a failure: counting refusals against "
    "availability creates pressure to reduce them, and every way to reduce them "
    "makes the system leakier.",
    ("strategy", "outcome"),
)

refusals_total = Counter(
    "sightline_refusals_total",
    "Refusals by RefusalReason. Tracked separately from availability, because "
    "the aggregate mixes reasons with opposite meanings: NO_PERMITTED_EVIDENCE "
    "is somebody's permissions problem, UNGROUNDED is a retrieval quality "
    "problem, STALE_POLICY is a reindex lag bug.",
    ("reason",),
)

recheck_dropped_total = Counter(
    "sightline_recheck_dropped_total",
    "Candidates dropped by the live permission recheck. This going to exactly "
    "zero across all traffic for an hour is a ticket, not a celebration: it is "
    "the production signature of mutants M3 and M4, where recheck has quietly "
    "become a no-op.",
    ("strategy",),
)

recheck_checked_total = Counter(
    "sightline_recheck_checked_total",
    "Candidates presented to the recheck. Denominator for the drop rate.",
    ("strategy",),
)

cross_tenant_leak_total = Counter(
    "sightline_cross_tenant_leak_total",
    "Results served that the authoritative check() would have denied. This is "
    "the one alert that pages. There is no rate, no smoothing window and no "
    "threshold to tune: one is too many, because one means the compiled plan and "
    "the authority have diverged. It must stay at zero.",
)

stage_seconds = Histogram(
    "sightline_stage_seconds",
    "Per-stage latency: plan, search, recheck, guardrails, synth, total.",
    ("stage",),
)

provider_requests_total = Counter(
    "sightline_provider_requests_total",
    "Generation attempts by provider and outcome, including circuit-open "
    "attempts that were never made.",
    ("provider", "outcome"),
)

provider_tokens_total = Counter(
    "sightline_provider_tokens_total",
    "Tokens billed by provider and direction. The free tiers bind on tokens per "
    "day, not on request count, so this is the number the budget watches.",
    ("provider", "direction"),
)

plan_cache_total = Counter(
    "sightline_plan_cache_total",
    "Plan compiler cache hits and misses. The cache is keyed on principal AND "
    "epoch; keying on principal alone is planted mutant M8.",
    ("result",),
)

audit_appends_total = Counter(
    "sightline_audit_appends_total",
    "Audit rows appended, by outcome.",
    ("outcome",),
)

audit_chain_failures_total = Counter(
    "sightline_audit_chain_failures_total",
    "Hash-chain verifications that found a break. Non-zero means the log was "
    "edited or truncated.",
)

http_requests_total = Counter(
    "sightline_http_requests_total",
    "HTTP requests by route and status class. Route, not path: a path label "
    "carries ids and is an unbounded series.",
    ("route", "status"),
)

# --------------------------------------------------------------------------
# Tracing
# --------------------------------------------------------------------------

_configured = False
_tracer: Any = None


def configure(settings: Any = None) -> bool:
    """Wire up a tracer provider if the ``obs`` extra is installed.

    Idempotent, and safe to call when nothing is installed. Returns whether real
    tracing is active, so a caller can log the truth rather than assume.

    Args:
        settings: A :class:`~sightline.settings.ObsSettings`, or ``None`` for
            defaults. Duck-typed on purpose: this module is imported by tests
            that have no settings object.
    """
    global _configured, _tracer
    if _configured:
        return _tracer is not None
    _configured = True
    if not HAVE_OTEL:
        return False
    enabled = getattr(settings, "traces_enabled", True)
    if not enabled:
        return False
    service = getattr(settings, "service_name", "sightline")
    endpoint = getattr(settings, "otlp_endpoint", "")
    try:  # pragma: no cover - obs-extra job
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider(resource=Resource.create({"service.name": service}))
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        _otel_trace.set_tracer_provider(provider)
    except ImportError:  # pragma: no cover - API present, SDK absent
        # The API alone is enough to create spans; without the SDK they go
        # nowhere. Still worth doing: the attribute code runs, so a bug in it
        # cannot hide behind a missing exporter.
        pass
    _tracer = _otel_trace.get_tracer("sightline")
    return True


class SpanHandle:
    """One span, or a convincing no-op.

    Attribute names are validated on the way in whether or not a tracing backend
    exists, so the rule about chunk text is enforced in the core-dependencies CI
    job too — the job where nobody is looking at traces.
    """

    __slots__ = ("_span", "attributes", "name")

    #: Attribute keys that would put document content or a raw question into
    #: telemetry. Matching is on the suffix so ``chunk.text`` and ``doc.text``
    #: both fail.
    _FORBIDDEN_SUFFIXES = (".text", "_text", ".content", "_content", ".secret")

    def __init__(self, span: Any, name: str) -> None:
        self._span = span
        self.name = name
        self.attributes: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        low = key.lower()
        if low.endswith(self._FORBIDDEN_SUFFIXES) or low in ("query.text", "question"):
            raise ValueError(
                f"refusing span attribute {key!r}: telemetry never carries document text or "
                "raw question text (mutant M12, SR-12)"
            )
        self.attributes[key] = value
        if self._span is not None:  # pragma: no cover - obs-extra job
            self._span.set_attribute(key, value)

    def set_many(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            self.set(key, value)

    def record_exception(self, exc: BaseException) -> None:
        self.attributes["error"] = type(exc).__name__
        if self._span is not None:  # pragma: no cover - obs-extra job
            self._span.record_exception(exc)


_current: threading.local = threading.local()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[SpanHandle]:
    """Open a span. Works identically with and without OpenTelemetry installed."""
    otel_span = None
    cm = None
    if _tracer is not None:  # pragma: no cover - obs-extra job
        cm = _tracer.start_as_current_span(name)
        otel_span = cm.__enter__()
    handle = SpanHandle(otel_span, name)
    previous = getattr(_current, "handle", None)
    _current.handle = handle
    try:
        handle.set_many(attributes)
        yield handle
    except BaseException as exc:
        handle.record_exception(exc)
        raise
    finally:
        _current.handle = previous
        if cm is not None:  # pragma: no cover - obs-extra job
            cm.__exit__(None, None, None)


def current_span() -> SpanHandle | None:
    """The innermost span opened by :func:`span` on this thread, if any."""
    return getattr(_current, "handle", None)


# --------------------------------------------------------------------------
# The attributes that make a trace worth reading
# --------------------------------------------------------------------------


def record_plan(handle: SpanHandle, plan: Any, *, cache: str = "miss") -> None:
    """Put the authorisation decision on the span.

    Counts only: the number of grant tokens, never the tokens. Same rule as
    ``/v1/plan`` returning ``n_grant_tokens`` instead of the filter itself.
    """
    handle.set_many(
        {
            "authz.strategy": getattr(plan.strategy, "value", str(plan.strategy)),
            "authz.cache": cache,
            "authz.n_grant_tokens": len(plan.grant_tokens),
            "authz.n_explicit_ids": len(plan.explicit_ids),
            "authz.estimated_cardinality": plan.estimated_cardinality,
            "policy.epoch": plan.epoch,
        }
    )
    plan_cache_total.inc(result=cache)


def record_search(handle: SpanHandle, *, backend: str, k: int, returned: int) -> None:
    handle.set_many(
        {
            "search.backend": backend,
            "search.k": k,
            "search.candidates_returned": returned,
            "search.filter_pushed_down": True,
        }
    )


def record_recheck(handle: SpanHandle, *, strategy: str, presented: int, dropped: int,
                   epoch: int) -> None:
    """The three numbers an incident actually needs."""
    handle.set_many(
        {
            "recheck.in": presented,
            "recheck.dropped": dropped,
            "recheck.batched": True,
            "recheck.epoch": epoch,
        }
    )
    recheck_checked_total.inc(presented, strategy=strategy)
    if dropped:
        recheck_dropped_total.inc(dropped, strategy=strategy)


def record_generation(
    handle: SpanHandle,
    *,
    provider: str,
    chunks_in: int,
    claims_out: int,
    citations: int,
    ids_dropped: int,
    degraded: bool,
) -> None:
    handle.set_many(
        {
            "synth.provider": provider,
            "synth.chunks_in": chunks_in,
            "synth.claims_out": claims_out,
            "synth.citations_reconstructed": citations,
            "synth.ids_dropped": ids_dropped,
            "synth.degraded": degraded,
        }
    )


def note_leak(count: int = 1, *, detail: str = "") -> None:
    """Record that a served result would have been denied by ``check()``.

    Called by the oracle drift check. Kept here rather than in the oracle so that
    there is exactly one place in the codebase that can move this counter, and it
    is greppable. If this is ever non-zero in production, the compiled plan and
    the authority have diverged and the system is serving documents the database
    would refuse — the failure the product exists to prevent.
    """
    if count <= 0:
        return
    cross_tenant_leak_total.inc(count)
    handle = current_span()
    if handle is not None:
        handle.set("authz.leak_detected", True)
        if detail:
            handle.set("authz.leak_detail", detail)


# --------------------------------------------------------------------------
# Exposition
# --------------------------------------------------------------------------


def render_metrics() -> tuple[bytes, str]:
    """Return ``(body, content_type)`` for ``GET /metrics``.

    With ``prometheus_client`` installed this is the real client's output, which
    includes process and GC collectors. Without it, the in-process counters are
    rendered in the same text format. The fallback is honest about what it is:
    the same numbers, none of the runtime collectors.
    """
    if HAVE_PROMETHEUS:  # pragma: no cover - obs-extra job
        return _prom.generate_latest(), PROMETHEUS_CONTENT_TYPE
    lines: list[str] = [
        "# Sightline in-process metrics. prometheus_client is not installed; this "
        "is the same counter state rendered in the text format, with no process "
        'or GC collectors. Install the extra with: pip install "sightline[obs]"'
    ]
    for metric in ALL_METRICS:
        lines.extend(metric.expose())
    return ("\n".join(lines) + "\n").encode("utf-8"), PROMETHEUS_CONTENT_TYPE


def reset_metrics_for_tests() -> None:
    """Zero the in-process counters. Test helper, named so it reads as one."""
    for metric in ALL_METRICS:
        with metric._lock:
            metric._values.clear()
            if isinstance(metric, Histogram):
                metric._counts.clear()
                metric._sums.clear()
