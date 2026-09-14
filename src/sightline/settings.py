"""Configuration. Read once at startup, frozen afterwards.

Three rules shape this file.

**No setting may weaken the security argument.** SR-20: nothing here can disable
recheck, the epoch/staleness check, or the plan-versus-oracle agreement test. A
flag is a thing somebody sets at 2am during an incident and nobody unsets. The
absence is enforced rather than asserted in prose — :func:`Settings.__post_init__`
refuses to construct if a field name looks like such a switch, so adding one is a
failing test rather than a code review somebody might wave through.

**Settings are plain frozen dataclasses, not ``BaseSettings``.** ``pydantic-settings``
is a separate package and the core install is four dependencies (SR-8). Parsing a
dozen environment variables by hand costs thirty lines and keeps the promise.

**Secrets do not print.** API keys are wrapped in :class:`Secret`, whose ``repr``
is fixed text. Configuration objects end up in startup logs, exception context
and debugger frames, and a key that renders itself in all three is a key you have
published.

Every variable is ``SIGHTLINE_``-prefixed except the provider API keys, which use
the names the providers themselves document (``GROQ_API_KEY`` and friends), since
those are what a developer already has exported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Mapping

from sightline.errors import ConfigError

__all__ = [
    "Secret",
    "AuthSettings",
    "StoreSettings",
    "RetrievalSettings",
    "GenerationSettings",
    "AuditSettings",
    "ObsSettings",
    "Settings",
    "load_settings",
    "GROQ_DAILY_TOKEN_BUDGET",
]

#: Groq's free tier binds on **tokens per day**, not on request count. Budgeting
#: against requests is the mistake that gets you rate-limited at 11am with 90% of
#: the allowance unspent on a handful of long contexts. The number is approximate
#: and provider-published; it is here as a default, not as a promise.
GROQ_DAILY_TOKEN_BUDGET = 200_000


class Secret:
    """A string that refuses to render itself.

    ``str()`` and ``repr()`` both give ``'***'``. Reading the value is an explicit
    :meth:`reveal` call, which greps cleanly: every place a key leaves the process
    is one search away.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str = "") -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Secret):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return "Secret('***')" if self._value else "Secret(unset)"

    __str__ = __repr__


# --------------------------------------------------------------------------
# Environment helpers
# --------------------------------------------------------------------------

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _s(env: Mapping[str, str], key: str, default: str = "") -> str:
    return env.get(key, default).strip()


def _b(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    low = raw.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ConfigError(f"{key}={raw!r} is not a boolean")


def _i(env: Mapping[str, str], key: str, default: int, *, minimum: int | None = None) -> int:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not an integer") from exc
    if minimum is not None and val < minimum:
        raise ConfigError(f"{key}={val} is below the minimum {minimum}")
    return val


def _f(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not a number") from exc


def _list(env: Mapping[str, str], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuthSettings:
    """Token verification.

    ``jwks_url`` is the production path. ``dev_hmac_secret`` exists so the test
    suite and a laptop demo can mint tokens with no identity provider, and it is
    refused when ``environment`` is ``production`` — a development shortcut that
    survives into production is not a shortcut, it is a backdoor.
    """

    jwks_url: str = ""
    issuer: str = ""
    audience: str = ""
    #: Clock skew tolerance. Two machines whose clocks differ by a few seconds is
    #: normal; a token rejected for being three seconds early is an outage.
    leeway_seconds: int = 60
    #: How long a fetched key set is trusted before a background-free refetch.
    jwks_cache_seconds: int = 600
    #: Floor between JWKS refetches triggered by an unknown ``kid``. Without it,
    #: a stream of junk tokens is a free denial-of-service against the IdP.
    jwks_min_refetch_seconds: int = 30
    jwks_timeout_seconds: float = 3.0
    dev_hmac_secret: Secret = field(default_factory=Secret)
    #: Claim that carries the principal id. ``sub`` unless the IdP says otherwise.
    principal_claim: str = "sub"
    #: Namespace every authenticated subject lands in: ``user:<sub>``.
    principal_namespace: str = "user"
    #: The tuple consulted for administrative endpoints. Checked through the same
    #: ``check()`` path as everything else (SR-19); there is no admin bypass.
    admin_object: str = "system:policy"
    admin_relation: str = "admin"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "AuthSettings":
        return cls(
            jwks_url=_s(env, "SIGHTLINE_JWKS_URL"),
            issuer=_s(env, "SIGHTLINE_JWT_ISSUER"),
            audience=_s(env, "SIGHTLINE_JWT_AUDIENCE"),
            leeway_seconds=_i(env, "SIGHTLINE_JWT_LEEWAY", 60, minimum=0),
            jwks_cache_seconds=_i(env, "SIGHTLINE_JWKS_TTL", 600, minimum=1),
            jwks_min_refetch_seconds=_i(env, "SIGHTLINE_JWKS_MIN_REFETCH", 30, minimum=0),
            jwks_timeout_seconds=_f(env, "SIGHTLINE_JWKS_TIMEOUT", 3.0),
            dev_hmac_secret=Secret(_s(env, "SIGHTLINE_DEV_HMAC_SECRET")),
            principal_claim=_s(env, "SIGHTLINE_PRINCIPAL_CLAIM", "sub"),
            principal_namespace=_s(env, "SIGHTLINE_PRINCIPAL_NAMESPACE", "user"),
            admin_object=_s(env, "SIGHTLINE_ADMIN_OBJECT", "system:policy"),
            admin_relation=_s(env, "SIGHTLINE_ADMIN_RELATION", "admin"),
        )


@dataclass(frozen=True, slots=True)
class StoreSettings:
    """Which backends the serving process wires up."""

    #: One of ``sightline.store.STORE_NAMES``. ``postfilter`` is rejected here,
    #: not merely discouraged: it is the deliberately-wrong baseline arm (FR-12).
    vector_backend: str = "memory"
    vector_url: str = ""
    collection: str = "sightline"
    #: ``memory`` or a filesystem path for the SQLite-backed tuple store.
    tuple_store: str = "memory"
    #: ``auto`` prefers ONNX and falls back to the hash embedder, loudly.
    embedder: str = "auto"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "StoreSettings":
        return cls(
            vector_backend=_s(env, "SIGHTLINE_VECTOR_BACKEND", "memory"),
            vector_url=_s(env, "SIGHTLINE_VECTOR_URL"),
            collection=_s(env, "SIGHTLINE_COLLECTION", "sightline"),
            tuple_store=_s(env, "SIGHTLINE_TUPLE_STORE", "memory"),
            embedder=_s(env, "SIGHTLINE_EMBEDDER", "auto"),
        )


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """Caps, and the one timing knob that exists for a security reason.

    ``refusal_floor_ms`` is the constant-time floor under the two empty-handed
    refusals. ``NO_PERMITTED_EVIDENCE`` short-circuits before search and returns
    in about a millisecond; ``NO_EVIDENCE_AT_ALL`` has run a search and takes
    tens. The bodies are byte-identical (FR-20) and the timing is otherwise a
    clean oracle for "a document exists that you may not see". The floor narrows
    that channel; it does not close it, and the acceptance test asserts the
    difference is under the harness noise floor rather than asserting zero,
    because a zero assertion on a wall-clock test is a test that lies.
    """

    default_k: int = 10
    #: Hard cap on k. Every hit is one permission check, so k is a lever on how
    #: much work an unauthenticated-but-valid caller can demand (SR-7).
    max_k: int = 50
    default_max_context_chars: int = 8_000
    max_context_chars: int = 32_000
    max_question_chars: int = 2_000
    refusal_floor_ms: float = 60.0
    #: Recompile rather than refuse when the plan cache hands back a stale plan.
    #: Refusal is still what happens if the policy moves again mid-recompile.
    recompile_on_stale: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "RetrievalSettings":
        return cls(
            default_k=_i(env, "SIGHTLINE_DEFAULT_K", 10, minimum=1),
            max_k=_i(env, "SIGHTLINE_MAX_K", 50, minimum=1),
            default_max_context_chars=_i(env, "SIGHTLINE_CONTEXT_CHARS", 8_000, minimum=256),
            max_context_chars=_i(env, "SIGHTLINE_MAX_CONTEXT_CHARS", 32_000, minimum=256),
            max_question_chars=_i(env, "SIGHTLINE_MAX_QUESTION_CHARS", 2_000, minimum=16),
            refusal_floor_ms=_f(env, "SIGHTLINE_REFUSAL_FLOOR_MS", 60.0),
            recompile_on_stale=_b(env, "SIGHTLINE_RECOMPILE_ON_STALE", True),
        )


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """The provider chain, its budgets, and its breakers.

    The order is a failover chain, not a load balancer: each provider is tried in
    turn and the first success wins. ``extractive`` is always appended if absent,
    because the chain must degrade to quoting the permitted chunks rather than
    failing the request.
    """

    order: tuple[str, ...] = ("groq", "gemini", "github", "llama", "extractive")
    groq_api_key: Secret = field(default_factory=Secret)
    gemini_api_key: Secret = field(default_factory=Secret)
    github_token: Secret = field(default_factory=Secret)
    groq_model: str = "llama-3.1-8b-instant"
    gemini_model: str = "gemini-2.0-flash"
    github_model: str = "gpt-4o-mini"
    llama_url: str = "http://127.0.0.1:8080"
    llama_model: str = "local"
    timeout_seconds: float = 20.0
    max_output_tokens: int = 700
    #: Per-provider daily token budgets. Zero means unmetered (the local model).
    daily_token_budgets: tuple[tuple[str, int], ...] = (
        ("groq", GROQ_DAILY_TOKEN_BUDGET),
        ("gemini", 1_000_000),
        ("github", 200_000),
        ("llama", 0),
    )
    #: Consecutive failures before a provider's breaker opens.
    breaker_threshold: int = 3
    #: How long a breaker stays open before one trial request is allowed through.
    breaker_cooldown_seconds: float = 60.0
    #: Where token accounting survives a restart. Empty keeps it in memory, which
    #: means a crash-loop can burn the daily budget several times over.
    budget_state_path: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "GenerationSettings":
        budgets = {
            "groq": _i(env, "SIGHTLINE_GROQ_DAILY_TOKENS", GROQ_DAILY_TOKEN_BUDGET, minimum=0),
            "gemini": _i(env, "SIGHTLINE_GEMINI_DAILY_TOKENS", 1_000_000, minimum=0),
            "github": _i(env, "SIGHTLINE_GITHUB_DAILY_TOKENS", 200_000, minimum=0),
            "llama": 0,
        }
        return cls(
            order=_list(
                env,
                "SIGHTLINE_PROVIDER_ORDER",
                ("groq", "gemini", "github", "llama", "extractive"),
            ),
            groq_api_key=Secret(_s(env, "GROQ_API_KEY")),
            gemini_api_key=Secret(_s(env, "GEMINI_API_KEY") or _s(env, "GOOGLE_API_KEY")),
            github_token=Secret(_s(env, "GITHUB_TOKEN") or _s(env, "GITHUB_MODELS_TOKEN")),
            groq_model=_s(env, "SIGHTLINE_GROQ_MODEL", "llama-3.1-8b-instant"),
            gemini_model=_s(env, "SIGHTLINE_GEMINI_MODEL", "gemini-2.0-flash"),
            github_model=_s(env, "SIGHTLINE_GITHUB_MODEL", "gpt-4o-mini"),
            llama_url=_s(env, "SIGHTLINE_LLAMA_URL", "http://127.0.0.1:8080"),
            llama_model=_s(env, "SIGHTLINE_LLAMA_MODEL", "local"),
            timeout_seconds=_f(env, "SIGHTLINE_LLM_TIMEOUT", 20.0),
            max_output_tokens=_i(env, "SIGHTLINE_MAX_OUTPUT_TOKENS", 700, minimum=32),
            daily_token_budgets=tuple(sorted(budgets.items())),
            breaker_threshold=_i(env, "SIGHTLINE_BREAKER_THRESHOLD", 3, minimum=1),
            breaker_cooldown_seconds=_f(env, "SIGHTLINE_BREAKER_COOLDOWN", 60.0),
            budget_state_path=_s(env, "SIGHTLINE_BUDGET_STATE"),
        )

    def budget_for(self, provider: str) -> int:
        """Daily token allowance for one provider; ``0`` means unmetered."""
        return dict(self.daily_token_budgets).get(provider, 0)


@dataclass(frozen=True, slots=True)
class AuditSettings:
    """Where the hash-chained log goes, and whether a write failure is fatal."""

    path: str = ""
    #: Refuse the request if the audit write fails. Default on: an answer with no
    #: record of who saw what is the thing an incident review cannot reconstruct.
    required: bool = True
    #: Key for the query HMAC. A plain SHA-256 of a short question is trivially
    #: reversed from a candidate list, so the hash is keyed (SR-12). Losing the
    #: key makes historical hashes uncorrelatable, which is the intended
    #: direction of failure.
    hmac_key: Secret = field(default_factory=Secret)
    #: fsync every append. Slow, and the only version of "append-only" that
    #: survives a power cut mid-write.
    fsync: bool = False
    #: Maximum rows a single ``GET /v1/audit`` may return.
    max_page: int = 500

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "AuditSettings":
        return cls(
            path=_s(env, "SIGHTLINE_AUDIT_PATH"),
            required=_b(env, "SIGHTLINE_AUDIT_REQUIRED", True),
            hmac_key=Secret(_s(env, "SIGHTLINE_AUDIT_HMAC_KEY")),
            fsync=_b(env, "SIGHTLINE_AUDIT_FSYNC", False),
            max_page=_i(env, "SIGHTLINE_AUDIT_MAX_PAGE", 500, minimum=1),
        )


@dataclass(frozen=True, slots=True)
class ObsSettings:
    """Traces and metrics. Both degrade to working no-ops without the extra."""

    service_name: str = "sightline"
    traces_enabled: bool = True
    metrics_enabled: bool = True
    #: OTLP endpoint. Empty means spans are created and dropped, which still
    #: exercises the attribute code so a missing exporter cannot hide a bug.
    otlp_endpoint: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "ObsSettings":
        return cls(
            service_name=_s(env, "SIGHTLINE_SERVICE_NAME", "sightline"),
            traces_enabled=_b(env, "SIGHTLINE_TRACES", True),
            metrics_enabled=_b(env, "SIGHTLINE_METRICS", True),
            otlp_endpoint=_s(env, "OTEL_EXPORTER_OTLP_ENDPOINT"),
        )


# --------------------------------------------------------------------------
# The whole thing
# --------------------------------------------------------------------------

#: Field names that would amount to a switch on the security argument. Checked at
#: construction, so SR-20 is a failing import rather than a review comment.
_FORBIDDEN_FIELD_SUBSTRINGS = (
    "disable_recheck",
    "skip_recheck",
    "recheck_enabled",
    "disable_epoch",
    "skip_epoch",
    "ignore_stale",
    "allow_stale",
    "bypass_auth",
    "disable_authz",
    "unfiltered_search",
)


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the serving process needs, resolved once.

    Construct with :func:`load_settings`. Tests build one directly and override
    with :func:`dataclasses.replace`, which is why every section is frozen.
    """

    environment: str = "development"
    auth: AuthSettings = field(default_factory=AuthSettings)
    store: StoreSettings = field(default_factory=StoreSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    audit: AuditSettings = field(default_factory=AuditSettings)
    obs: ObsSettings = field(default_factory=ObsSettings)

    def __post_init__(self) -> None:
        self._assert_no_security_switches()
        if self.retrieval.default_k > self.retrieval.max_k:
            raise ConfigError(
                f"default_k {self.retrieval.default_k} exceeds max_k {self.retrieval.max_k}"
            )
        if self.retrieval.default_max_context_chars > self.retrieval.max_context_chars:
            raise ConfigError("default context budget exceeds the maximum context budget")
        if self.store.vector_backend in ("postfilter", "post_filter", "baseline"):
            raise ConfigError(
                "the post-filter store is the deliberately-wrong baseline arm (FR-12) and "
                "cannot back a serving process; use 'memory', 'qdrant' or 'pgvector'"
            )
        if self.is_production:
            if self.auth.dev_hmac_secret:
                raise ConfigError(
                    "SIGHTLINE_DEV_HMAC_SECRET is set in production; dev tokens are a "
                    "backdoor with a friendly name"
                )
            if not self.auth.jwks_url:
                raise ConfigError("production requires SIGHTLINE_JWKS_URL")
            if not self.auth.issuer or not self.auth.audience:
                raise ConfigError("production requires SIGHTLINE_JWT_ISSUER and _AUDIENCE")

    def _assert_no_security_switches(self) -> None:
        """Refuse to construct if a field name reads like a kill switch (SR-20).

        Walks the live sections rather than the annotations, because
        ``from __future__ import annotations`` makes every annotation a string
        and a check that silently inspects strings is a check that never fires.
        """
        names: list[str] = []
        for section in fields(self):
            names.append(section.name)
            value = getattr(self, section.name)
            if is_dataclass(value):
                names.extend(f.name for f in fields(value))
        for name in names:
            for bad in _FORBIDDEN_FIELD_SUBSTRINGS:
                if bad in name:
                    raise ConfigError(
                        f"setting {name!r} would disable part of the authorisation path; "
                        "SR-20 says no such setting exists"
                    )

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in ("production", "prod")

    @property
    def allow_dev_tokens(self) -> bool:
        """HMAC-signed local tokens, for the demo and the test suite only."""
        return bool(self.auth.dev_hmac_secret) and not self.is_production

    def describe(self) -> dict[str, object]:
        """A redacted summary, safe for a startup log line and for ``/readyz``.

        Counts and names, never keys. Same rule as the span attributes and
        ``/v1/plan``: the shape of the configuration is operationally useful, the
        contents of it are an attacker's shopping list.
        """
        return {
            "environment": self.environment,
            "vector_backend": self.store.vector_backend,
            "tuple_store": "memory" if self.store.tuple_store == "memory" else "sqlite",
            "embedder": self.store.embedder,
            "auth": (
                "jwks"
                if self.auth.jwks_url
                else ("dev-hmac" if self.allow_dev_tokens else "none")
            ),
            "provider_order": list(self.generation.order),
            "providers_configured": sorted(self.configured_providers()),
            "audit": "file" if self.audit.path else "memory",
            "max_k": self.retrieval.max_k,
        }

    def configured_providers(self) -> set[str]:
        """Providers with credentials present. ``extractive`` always qualifies."""
        gen = self.generation
        out = {"extractive", "llama"}
        if gen.groq_api_key:
            out.add("groq")
        if gen.gemini_api_key:
            out.add("gemini")
        if gen.github_token:
            out.add("github")
        return out

    def with_(self, **changes: object) -> "Settings":
        """``dataclasses.replace`` under a name that reads in a test."""
        return replace(self, **changes)  # type: ignore[arg-type]


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build :class:`Settings` from the environment.

    Args:
        env: Mapping to read. Defaults to ``os.environ``; tests pass a dict.

    Returns:
        A frozen, validated :class:`Settings`.

    Raises:
        ConfigError: On an unparseable value or a contradictory combination.
            Startup is the right place to fail: a configuration error found at
            request time is an outage with worse ergonomics.
    """
    src = os.environ if env is None else env
    return Settings(
        environment=_s(src, "SIGHTLINE_ENV", "development"),
        auth=AuthSettings.from_env(src),
        store=StoreSettings.from_env(src),
        retrieval=RetrievalSettings.from_env(src),
        generation=GenerationSettings.from_env(src),
        audit=AuditSettings.from_env(src),
        obs=ObsSettings.from_env(src),
    )
