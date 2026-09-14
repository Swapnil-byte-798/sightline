"""Who is asking. Verified here, at the origin, on every request.

**Why verify again when the edge already did.** The gateway in front of this
service validates the same token. Doing it a second time is not redundancy
theatre, it is the difference between a trust boundary and a habit:

* The edge is one misconfiguration away from being bypassed — a service mesh
  route that skips it, a port forward during an incident, a second ingress added
  by a team that did not know. Anything on the internal network that can open a
  socket to this process is, without origin verification, an administrator.
* A header injected by the edge (``X-Authenticated-User``) is forgeable by
  anything that can reach the pod, and there is no way to tell a forged one from
  a real one. A signature is checkable by anyone; a header is a promise.
* The origin is where the permission decision is made, and the identity that
  decision is made against should not have travelled through a hop that could
  rewrite it.

Cost: one signature verification per request, a few hundred microseconds on the
reference machine, against a cached key. That is the cheapest security control in
the system.

**SR-18 — identity comes only from the signature.** There is no principal field
in any request body, no ``principal`` query parameter, and no header that names
the caller. A caller who can name their own principal is not a caller, they are
an attacker. The one apparent exception, ``/v1/explain?principal=...``, asks
about *somebody else's* access and is therefore an administrative read guarded by
a ``check()`` against the admin relation — the subject of the question is not the
identity of the asker.

**Group claims in the token are ignored for authorisation.** An IdP happily puts
``groups: ["legal", "finance"]`` in the token, and using it would be a plausible
mistake. It is wrong twice: the claim is a snapshot taken at login and is stale
for the whole token lifetime, which reopens exactly the revocation window this
product closes; and it makes the token's issuer a second authorisation authority
that nobody audits. Group membership is tuples, tuples are the database, the
database is the authority. Claims identify; they do not grant.

``python-jose`` is listed as a core dependency but is guarded anyway, because the
core-dependency CI job installs the four packages the README promises and a hard
import here would fail it. Without jose, HS256 development tokens still verify
through :mod:`hmac` in the standard library, and RS256/ES256 raise an error
naming what to install. Asymmetric verification without a crypto library is not
something to improvise.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import httpx

from sightline.errors import (
    InvalidToken,
    JwksUnavailable,
    MissingCredentials,
    TokenExpired,
    UnknownSigningKey,
)
from sightline.settings import AuthSettings
from sightline.types import PrincipalRef

__all__ = [
    "TokenClaims",
    "JwksCache",
    "TokenVerifier",
    "principal_from_claims",
    "mint_dev_token",
    "HAVE_JOSE",
    "SUPPORTED_ALGORITHMS",
]

try:  # pragma: no cover - depends on install shape
    from jose import jws as _jose_jws

    HAVE_JOSE = True
except ImportError:  # pragma: no cover - core-deps CI job
    _jose_jws = None  # type: ignore[assignment]
    HAVE_JOSE = False

#: Algorithms this service will verify. ``none`` is absent and that is the point:
#: the ``alg: none`` acceptance bug is old, famous, and still shipped every year.
#: Symmetric HS256 is accepted only for development tokens, and only when the
#: deployment is not production (:attr:`Settings.allow_dev_tokens`).
SUPPORTED_ALGORITHMS = frozenset({"RS256", "RS384", "RS512", "ES256", "ES384", "HS256"})

_SYMMETRIC = frozenset({"HS256"})


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenClaims:
    """The verified contents of a bearer token.

    ``raw`` keeps every claim for logging and for the explain endpoint, but note
    that nothing downstream reads a group or role claim out of it. See the module
    docstring for why that is deliberate.
    """

    subject: str
    issuer: str = ""
    audience: tuple[str, ...] = ()
    expires_at: float = 0.0
    issued_at: float = 0.0
    key_id: str = ""
    algorithm: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.expires_at - time.time())


def principal_from_claims(claims: TokenClaims, settings: AuthSettings) -> PrincipalRef:
    """Map a verified token to the principal the tuple store knows about.

    One mapping, one namespace, no configuration surface beyond the claim name
    and the namespace. Clever mappings — "if the subject contains an @, split it
    and use the domain as a tenant" — are how two different identities end up
    compiling to the same principal.

    Raises:
        InvalidToken: If the configured claim is missing or empty. A token
            without a subject is not an identity.
    """
    value = str(claims.raw.get(settings.principal_claim, "") or "").strip()
    if not value:
        raise InvalidToken(f"token has no {settings.principal_claim!r} claim")
    if ":" in value or "#" in value:
        # A subject that already looks like a ref would let the IdP choose its
        # own namespace, and the namespace is half of the authorisation key.
        raise InvalidToken("subject claim may not contain ':' or '#'")
    return PrincipalRef(settings.principal_namespace, value)


# --------------------------------------------------------------------------
# JWKS
# --------------------------------------------------------------------------


def _b64url(data: str) -> bytes:
    """Decode base64url without padding, strictly.

    Strict: a token segment with invalid characters is a malformed token, and
    ``validate=False`` would silently skip them and verify something the issuer
    never signed.
    """
    pad = "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(data + pad)
    except (binascii.Error, ValueError) as exc:
        raise InvalidToken("malformed base64url segment") from exc


class JwksCache:
    """Fetches and caches an issuer's public keys, and handles rotation.

    Rotation in practice: the issuer publishes the new key alongside the old one
    for a while, signs with the new one, then drops the old. A cache that only
    refreshes on a timer will reject every token signed with the new key until it
    expires. So there are two refresh triggers:

    1. The TTL, for the ordinary case.
    2. An unknown ``kid``, immediately — rate limited by
       ``jwks_min_refetch_seconds``. Without that floor, a stream of tokens with
       junk key ids is a free amplified denial of service pointed at the identity
       provider, delivered by us.

    A fetch failure with usable cached keys is not fatal: stale-but-valid public
    keys are exactly as trustworthy as fresh ones, since verification does not
    depend on the key set being current, only on the key being the issuer's.
    Failing with *no* cached keys is fatal, and it raises 503 rather than 401,
    because "we cannot tell" is not "you are not allowed".
    """

    def __init__(
        self,
        url: str,
        *,
        ttl_seconds: int = 600,
        min_refetch_seconds: int = 30,
        timeout: float = 3.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = url
        self.ttl = ttl_seconds
        self.min_refetch = min_refetch_seconds
        self.timeout = timeout
        self._client = client
        self._lock = threading.Lock()
        self._keys: dict[str, dict[str, Any]] = {}
        self._fetched_at = 0.0
        self._last_attempt = 0.0

    # -- fetching ----------------------------------------------------------
    def _fetch(self) -> dict[str, dict[str, Any]]:
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            response = client.get(self.url)
            response.raise_for_status()
            document = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise JwksUnavailable(f"JWKS fetch from {self.url} failed: {exc}") from exc
        finally:
            if self._client is None:
                client.close()
        keys = document.get("keys") if isinstance(document, dict) else None
        if not isinstance(keys, list) or not keys:
            raise JwksUnavailable(f"JWKS at {self.url} has no keys")
        out: dict[str, dict[str, Any]] = {}
        for key in keys:
            if isinstance(key, dict) and key.get("kid"):
                out[str(key["kid"])] = key
        if not out:
            raise JwksUnavailable(f"JWKS at {self.url} has no keys with a 'kid'")
        return out

    def refresh(self, *, force: bool = False) -> None:
        with self._lock:
            now = time.time()
            if not force and now - self._fetched_at < self.ttl:
                return
            if now - self._last_attempt < self.min_refetch and self._keys:
                return
            self._last_attempt = now
            try:
                self._keys = self._fetch()
                self._fetched_at = now
            except JwksUnavailable:
                if not self._keys:
                    raise
                # Keep serving on cached keys. A public key does not go stale in
                # any way that matters to a signature check.

    def key_for(self, kid: str) -> dict[str, Any]:
        """Return the JWK for ``kid``, refetching once if it is unknown."""
        with self._lock:
            key = self._keys.get(kid)
            expired = time.time() - self._fetched_at >= self.ttl
        if key is not None and not expired:
            return key
        self.refresh(force=key is None)
        with self._lock:
            key = self._keys.get(kid)
        if key is None:
            raise UnknownSigningKey(f"no key with kid {kid!r} in {self.url}")
        return key

    def __len__(self) -> int:
        with self._lock:
            return len(self._keys)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


class TokenVerifier:
    """Verifies a bearer token and returns its claims. Fail closed, always.

    Ordering is deliberate: signature first, then claims. Reading ``exp`` out of
    an unverified payload to "fail fast on expired tokens" means making decisions
    from attacker-controlled data, and the cost saved is a hash.
    """

    def __init__(
        self,
        settings: AuthSettings,
        *,
        allow_dev_tokens: bool = False,
        jwks: JwksCache | None = None,
        clock: Any = time.time,
    ) -> None:
        self.settings = settings
        self.allow_dev_tokens = allow_dev_tokens
        self.clock = clock
        if jwks is not None:
            self.jwks: JwksCache | None = jwks
        elif settings.jwks_url:
            self.jwks = JwksCache(
                settings.jwks_url,
                ttl_seconds=settings.jwks_cache_seconds,
                min_refetch_seconds=settings.jwks_min_refetch_seconds,
                timeout=settings.jwks_timeout_seconds,
            )
        else:
            self.jwks = None

    # -- public ------------------------------------------------------------
    def verify(self, token: str) -> TokenClaims:
        """Verify signature then claims.

        Raises:
            MissingCredentials: Empty token.
            InvalidToken: Malformed, wrong algorithm, bad signature, wrong
                issuer or audience.
            TokenExpired: Outside its validity window, allowing for skew.
            UnknownSigningKey: ``kid`` not present in the issuer's key set.
            JwksUnavailable: Keys could not be fetched and none are cached.
        """
        if not token or not token.strip():
            raise MissingCredentials("empty bearer token")
        parts = token.strip().split(".")
        if len(parts) != 3:
            raise InvalidToken("token is not three dot-separated segments")

        header = self._segment(parts[0], "header")
        alg = str(header.get("alg", ""))
        kid = str(header.get("kid", ""))
        if alg not in SUPPORTED_ALGORITHMS:
            raise InvalidToken(f"unsupported algorithm {alg!r}")

        if alg in _SYMMETRIC:
            self._verify_hmac(token, alg)
        else:
            self._verify_asymmetric(token, alg, kid)

        payload = self._segment(parts[1], "payload")
        return self._validate_claims(payload, alg=alg, kid=kid)

    # -- signature ---------------------------------------------------------
    def _segment(self, raw: str, what: str) -> dict[str, Any]:
        try:
            blob = json.loads(_b64url(raw))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InvalidToken(f"token {what} is not JSON") from exc
        if not isinstance(blob, dict):
            raise InvalidToken(f"token {what} is not an object")
        return blob

    def _verify_hmac(self, token: str, alg: str) -> None:
        """HS256 with the development secret, and only outside production.

        A symmetric algorithm means anyone who can verify can also mint. That is
        fine for a laptop demo and disqualifying for anything else, so the guard
        is on the deployment, not on a flag somebody can flip.
        """
        if not self.allow_dev_tokens:
            raise InvalidToken(
                "HS256 tokens are development-only and this deployment does not accept them"
            )
        secret = self.settings.dev_hmac_secret.reveal().encode("utf-8")
        if not secret:
            raise InvalidToken("no development signing secret configured")
        signing_input, _, signature = token.rpartition(".")
        expected = hmac.new(secret, signing_input.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url(signature)):
            raise InvalidToken("bad signature")

    def _verify_asymmetric(self, token: str, alg: str, kid: str) -> None:
        if self.jwks is None:
            raise InvalidToken("no JWKS configured; cannot verify an asymmetric signature")
        if not kid:
            raise InvalidToken("asymmetric token has no 'kid'; key rotation needs one")
        key = self.jwks.key_for(kid)
        if not HAVE_JOSE:  # pragma: no cover - core-deps CI job
            from sightline.errors import MissingExtra

            raise MissingExtra(
                "jose",
                "",
                purpose=f"verifying {alg} signatures; install python-jose[cryptography]",
            )
        try:  # pragma: no cover - requires the crypto stack
            _jose_jws.verify(token, key, algorithms=[alg])
        except Exception as exc:  # jose raises several unrelated types
            raise InvalidToken(f"signature verification failed: {type(exc).__name__}") from exc

    # -- claims ------------------------------------------------------------
    def _validate_claims(self, payload: Mapping[str, Any], *, alg: str, kid: str) -> TokenClaims:
        now = float(self.clock())
        leeway = float(self.settings.leeway_seconds)

        exp = payload.get("exp")
        if exp is None:
            raise InvalidToken("token has no 'exp'; a token that never expires is a password")
        if now > float(exp) + leeway:
            raise TokenExpired("token expired")

        nbf = payload.get("nbf")
        if nbf is not None and now < float(nbf) - leeway:
            raise InvalidToken("token not yet valid")

        iat = payload.get("iat")
        if iat is not None and float(iat) - leeway > now:
            # Clocks drift both ways. Beyond the allowance, a token issued in the
            # future is either a broken issuer or a replay with a rewritten clock.
            raise InvalidToken("token issued in the future")

        if self.settings.issuer and str(payload.get("iss", "")) != self.settings.issuer:
            raise InvalidToken("issuer mismatch")

        audience = _as_tuple(payload.get("aud"))
        if self.settings.audience and self.settings.audience not in audience:
            # An audience check is what stops a token minted for another service
            # in the same identity provider from working here.
            raise InvalidToken("audience mismatch")

        subject = str(payload.get(self.settings.principal_claim, "") or "")
        if not subject:
            raise InvalidToken(f"token has no {self.settings.principal_claim!r} claim")

        return TokenClaims(
            subject=subject,
            issuer=str(payload.get("iss", "")),
            audience=audience,
            expires_at=float(exp),
            issued_at=float(iat) if iat is not None else 0.0,
            key_id=kid,
            algorithm=alg,
            raw=dict(payload),
        )

    # -- convenience -------------------------------------------------------
    def principal(self, token: str) -> tuple[PrincipalRef, TokenClaims]:
        claims = self.verify(token)
        return principal_from_claims(claims, self.settings), claims


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(v) for v in value)
    return (str(value),)


# --------------------------------------------------------------------------
# Development tokens
# --------------------------------------------------------------------------


def mint_dev_token(
    subject: str,
    secret: str,
    *,
    issuer: str = "",
    audience: str = "",
    ttl_seconds: int = 3600,
    now: float | None = None,
) -> str:
    """Mint an HS256 token for tests and the laptop demo.

    Deliberately in the library rather than in ``tests/``: the contract tests and
    the demo script both need it, and a second copy of a token minter is a second
    thing to get wrong. It is harmless in a production build because
    :class:`TokenVerifier` refuses HS256 there — the restriction lives on the
    verifier, where it cannot be worked around by importing something else.
    """
    if not secret:
        raise ValueError("a development token needs a signing secret")
    issued = time.time() if now is None else now
    header = {"alg": "HS256", "typ": "JWT", "kid": "dev"}
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": int(issued),
        "exp": int(issued) + ttl_seconds,
    }
    if issuer:
        payload["iss"] = issuer
    if audience:
        payload["aud"] = audience

    def seg(obj: Mapping[str, Any]) -> str:
        raw = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    signing_input = f"{seg(header)}.{seg(payload)}"
    signature = hmac.new(
        secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{signing_input}.{base64.urlsafe_b64encode(signature).decode('ascii').rstrip('=')}"
