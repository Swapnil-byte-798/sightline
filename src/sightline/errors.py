"""The exception hierarchy, and the rule about what an error is allowed to say.

Two ideas run through this file.

**Every failure in the serving path is a refusal, and refusal is the safe state.**
The tuple store is unreachable, the epoch cannot be read, the index is down, the
plan is stale — each of those raises, and each of them ends as a 4xx/5xx with no
documents attached. Nothing here degrades to a cached plan or a secondary path,
because "serve something rather than nothing" is the exact reasoning that leaks.

**An error payload is a disclosure surface.** Planted mutant M12 is "chunk text
appears in an error or refusal payload", and it is planted because that is the
realistic way document content escapes a permissioned system: not through the
answer, through the debugging. So every exception here carries two strings:
``public`` — what the HTTP client is told, fixed per class and content-free — and
``detail`` — what goes to the operator's log. The HTTP layer serialises ``public``
and never ``detail``. Putting evidence into ``detail`` is fine; putting it into
``public`` is a review-blocking change.

The status codes are chosen so a caller can retry correctly: 409 means "your
policy view moved, recompile and retry", 429 means "you asked for more than the
budget, ask for less", 503 means "the authority is unavailable, do not retry in
a tight loop".
"""

from __future__ import annotations

__all__ = [
    "AllProvidersFailed",
    "AuditChainBroken",
    "AuditUnavailable",
    "AuthError",
    "AuthorityUnavailable",
    "BudgetExceeded",
    "CircuitOpen",
    "ConfigError",
    "GenerationError",
    "IndexUnavailable",
    "InvalidToken",
    "JwksUnavailable",
    "MissingCredentials",
    "MissingExtra",
    "PermissionDenied",
    "ProviderBudgetExhausted",
    "ProviderError",
    "ProviderTimeout",
    "SightlineError",
    "StalePolicy",
    "TokenExpired",
    "UnknownSigningKey",
    "require_extra",
]


class SightlineError(Exception):
    """Base class. Carries an HTTP status, a machine code, and two messages.

    Args:
        detail: For operators. May name objects, principals and counts. Logged,
            never returned over HTTP.
        public: Overrides the class-level client-visible message. Callers should
            almost never pass this — the whole point of a fixed public string is
            that it cannot accidentally grow a document in it.
    """

    #: What the HTTP layer returns. Deliberately uninformative.
    public_message: str = "request failed"
    #: Stable identifier for clients that branch on the failure.
    code: str = "error"
    status_code: int = 500

    def __init__(self, detail: str = "", *, public: str | None = None) -> None:
        super().__init__(detail or self.public_message)
        self.detail = detail
        self.public = public or self.public_message

    def as_payload(self) -> dict[str, str]:
        """The JSON body. Two keys, both content-free, by construction."""
        return {"error": self.code, "message": self.public}


# --------------------------------------------------------------------------
# Configuration and optional dependencies
# --------------------------------------------------------------------------


class ConfigError(SightlineError):
    """A setting is missing or contradictory. Raised at startup, not per request."""

    public_message = "server misconfigured"
    code = "config_error"
    status_code = 500


class MissingExtra(ImportError, SightlineError):
    """An optional dependency is not installed, and the message says which.

    Inherits from :class:`ImportError` so ``except ImportError`` around a guarded
    import still catches it — several modules in this package rely on that, and a
    guard that only works when you remember to catch a bespoke type is not a
    guard.
    """

    public_message = "feature unavailable in this deployment"
    code = "missing_extra"
    status_code = 501

    def __init__(self, module: str, extra: str, *, purpose: str = "") -> None:
        self.module = module
        self.extra = extra
        tail = f" ({purpose})" if purpose else ""
        detail = (
            f"{module} is not installed{tail}. Install it with: "
            f'pip install "sightline[{extra}]"'
        )
        ImportError.__init__(self, detail)
        SightlineError.__init__(self, detail)


def require_extra(module: str, extra: str, *, purpose: str = "") -> object:
    """Import ``module`` or raise a :class:`MissingExtra` naming the extra.

    Every optional import in the serving path goes through here so the failure
    message is identical everywhere. A bare ``ModuleNotFoundError: No module
    named 'qdrant_client'`` makes the reader guess the extra's name; this does
    not.

    Args:
        module: Importable module name, e.g. ``"qdrant_client"``.
        extra: The extra that provides it, e.g. ``"qdrant"``.
        purpose: What breaks without it, quoted in the message.

    Returns:
        The imported module.

    Raises:
        MissingExtra: If the import fails.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtra(module, extra, purpose=purpose) from exc


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


class AuthError(SightlineError):
    """Anything that stops us knowing who is asking.

    All subclasses return the same public message on purpose. "Expired" versus
    "wrong issuer" versus "unknown key" is useful to an operator reading logs and
    useful to an attacker probing the edge, and only one of those two is our
    customer.
    """

    public_message = "authentication required"
    code = "unauthenticated"
    status_code = 401


class MissingCredentials(AuthError):
    code = "no_credentials"


class InvalidToken(AuthError):
    code = "invalid_token"


class TokenExpired(AuthError):
    code = "invalid_token"


class UnknownSigningKey(AuthError):
    code = "invalid_token"


class JwksUnavailable(AuthError):
    """The key set could not be fetched and nothing usable is cached.

    503, not 401: the token may well be valid and we simply cannot tell. Failing
    closed here is still the right answer, but the caller should retry rather
    than go and re-authenticate.
    """

    public_message = "identity provider unavailable"
    code = "jwks_unavailable"
    status_code = 503


class PermissionDenied(SightlineError):
    """The principal is known and is not allowed to do this.

    Used for administrative operations, where hiding the existence of the
    endpoint buys nothing. Document *content* denials never reach here: those are
    refusals with ``NO_PERMITTED_EVIDENCE`` and they are deliberately
    indistinguishable from "there is nothing to find" (FR-20).
    """

    public_message = "not permitted"
    code = "permission_denied"
    status_code = 403


# --------------------------------------------------------------------------
# The serving path
# --------------------------------------------------------------------------


class StalePolicy(SightlineError):
    """A plan compiled under an older epoch than the live policy.

    409 with a retry hint. The client is not wrong, its view of the world is one
    write behind, and the fix is to ask again.
    """

    public_message = "policy changed during the request; retry"
    code = "stale_policy"
    status_code = 409

    def __init__(self, detail: str = "", *, plan_epoch: int = 0, live_epoch: int = 0) -> None:
        super().__init__(detail or f"plan epoch {plan_epoch} < live epoch {live_epoch}")
        self.plan_epoch = plan_epoch
        self.live_epoch = live_epoch


class BudgetExceeded(SightlineError):
    """The request asked for more than a cap allows.

    The caps exist so that a large ``k`` cannot turn into an unbounded number of
    permission checks (SR-7). Note what this never does: it never trims the
    request down to fit and serves it anyway, because trimming would mean
    dropping checks, and dropping checks is the failure this system is about.
    """

    public_message = "request exceeds a configured budget"
    code = "budget_exceeded"
    status_code = 429


class AuthorityUnavailable(SightlineError):
    """The tuple store — the authority — cannot be reached.

    There is no cached-plan fallback. If the authority is down there is no
    authority, and serving from a cached plan is precisely the leak the product
    exists to prevent (FRD section 5, Availability).
    """

    public_message = "authorisation store unavailable"
    code = "authority_unavailable"
    status_code = 503


class IndexUnavailable(SightlineError):
    """The vector index cannot be reached. Distinct reason, same refusal."""

    public_message = "search index unavailable"
    code = "index_unavailable"
    status_code = 503


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


class GenerationError(SightlineError):
    """Base for the provider chain. Never fatal to a request on its own.

    The chain degrades to an extractive answer built from the rechecked chunks,
    so a provider outage costs answer quality and not availability. These types
    exist to drive failover and to be counted, not to be returned.
    """

    public_message = "generation unavailable"
    code = "generation_error"
    status_code = 503


class ProviderError(GenerationError):
    """One provider failed. Carries the provider name for the circuit breaker."""

    code = "provider_error"

    def __init__(self, provider: str, detail: str = "", *, retryable: bool = True) -> None:
        super().__init__(detail or f"{provider} failed")
        self.provider = provider
        self.retryable = retryable


class ProviderTimeout(ProviderError):
    code = "provider_timeout"


class ProviderBudgetExhausted(ProviderError):
    """This provider's daily token budget is spent.

    Not an error in any real sense — it is the free tier working as documented.
    It is an exception so that the chain's failover logic has one shape instead
    of two.
    """

    code = "provider_budget_exhausted"
    retryable = False


class CircuitOpen(ProviderError):
    """The breaker for this provider is open; the call was not attempted."""

    code = "circuit_open"


class AllProvidersFailed(GenerationError):
    """Every provider in the chain refused or failed.

    Reaching this is not an HTTP error: the caller gets an extractive answer or a
    structured refusal. It is raised only by the low-level chain so the caller
    can decide, and the caller always decides to degrade.
    """

    code = "all_providers_failed"

    def __init__(self, attempts: tuple[str, ...] = ()) -> None:
        super().__init__(f"all providers failed: {', '.join(attempts) or 'none configured'}")
        self.attempts = attempts


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


class AuditChainBroken(SightlineError):
    """Verification found a row whose digest does not commit to its predecessor.

    This means the log was edited, truncated in the middle, or written by two
    processes that disagreed about the head. It is reported, loudly, and it is
    not automatically repaired: a self-healing audit log is a log that erases the
    evidence of its own tampering.
    """

    public_message = "audit log integrity check failed"
    code = "audit_chain_broken"
    status_code = 500

    def __init__(self, seq: int, detail: str = "") -> None:
        super().__init__(detail or f"audit chain broken at seq {seq}")
        self.seq = seq


class AuditUnavailable(SightlineError):
    """The audit sink cannot be written.

    Whether this fails the request is a policy decision, made in
    :mod:`sightline.settings` (``audit_required``) rather than here. Default is
    to fail the request: an answer served with no record of who saw what is the
    thing a regulator asks about, and "we kept serving" is not an answer.
    """

    public_message = "audit log unavailable"
    code = "audit_unavailable"
    status_code = 503
