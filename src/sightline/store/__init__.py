"""Vector backends, and the one rule they all obey.

Every store in this package implements :class:`~sightline.store.base.VectorStore`
and returns :class:`~sightline.store.base.UncheckedHit`. None of them returns a
:class:`~sightline.store.base.Hit`, and none of them ever will: the index is a
hint, the tuple store is the authority, and ``authz.recheck.recheck()`` is the
only bridge (ADR 0001).

============  ===========================================================
Name          What it is for
============  ===========================================================
``memory``    Reference implementation. numpy only, exact, no services.
              CI runs against this.
``qdrant``    The serving path. Filter pushed into HNSW traversal.
``pgvector``  Second, independent enforcement layer: Postgres RLS refuses
              forbidden rows even with the application filter deleted.
``oracle``    Exact ground truth for every recall figure. Never served.
``postfilter`` The deliberately wrong baseline arm. Never a default, never
              in production, and the reason this project exists.
============  ===========================================================

Imports of the optional backends are deferred through a module ``__getattr__``,
so ``import sightline.store`` costs nothing and never fails on a machine without
``qdrant-client``, ``psycopg`` or ``faiss``. Touching one of those names raises
an ``ImportError`` naming the extra to install.
"""

from __future__ import annotations

from typing import Any

from sightline.store.base import Hit, Rechecker, StoreStats, UncheckedHit, VectorStore
from sightline.store.memory import (
    MemoryVectorStore,
    admits,
    first_matching_token,
    plan_bindings,
)

__all__ = [
    "Hit",
    "UncheckedHit",
    "StoreStats",
    "VectorStore",
    "Rechecker",
    "MemoryVectorStore",
    "PostFilterStore",
    "QdrantVectorStore",
    "PgVectorStore",
    "FaissOracleStore",
    "build_oracle",
    "admits",
    "plan_bindings",
    "first_matching_token",
    "get_store",
    "STORE_NAMES",
]

#: Canonical name -> ``(module, attribute)``. Resolved lazily by :func:`get_store`.
_BACKENDS: dict[str, tuple[str, str]] = {
    "memory": ("sightline.store.memory", "MemoryVectorStore"),
    "postfilter": ("sightline.store.postfilter", "PostFilterStore"),
    "qdrant": ("sightline.store.qdrant_store", "QdrantVectorStore"),
    "pgvector": ("sightline.store.pgvector_store", "PgVectorStore"),
    "oracle": ("sightline.store.faiss_oracle", "build_oracle"),
    "faiss": ("sightline.store.faiss_oracle", "FaissOracleStore"),
}

#: Aliases people actually type, kept separate so the canonical list stays short.
_ALIASES = {
    "postgres": "pgvector",
    "postgresql": "pgvector",
    "pg": "pgvector",
    "numpy": "memory",
    "reference": "memory",
    "baseline": "postfilter",
    "post_filter": "postfilter",
    "faiss-oracle": "oracle",
}

STORE_NAMES = tuple(sorted(_BACKENDS))


def _resolve(name: str) -> Any:
    import importlib

    module_name, attr = _BACKENDS[name]
    return getattr(importlib.import_module(module_name), attr)


def get_store(name: str, **kwargs: Any) -> VectorStore:
    """Construct a backend by name.

    Deliberately not a plugin registry with entry points. There are five
    backends, they are all in this package, and a factory you can read in one
    screen is worth more than an extension mechanism nobody will use.

    Args:
        name: One of :data:`STORE_NAMES`, or an alias (``postgres``, ``baseline``).
        **kwargs: Forwarded to the backend's constructor.

    Returns:
        A store implementing :class:`~sightline.store.base.VectorStore`.

    Raises:
        ValueError: If the name is not a known backend.
        ImportError: If the backend's optional extra is not installed. The
            message names the exact ``pip install`` to run.
    """
    key = _ALIASES.get(name, name)
    if key not in _BACKENDS:
        known = ", ".join(STORE_NAMES)
        raise ValueError(f"unknown store {name!r}; known backends: {known}")
    return _resolve(key)(**kwargs)


def __getattr__(attr: str) -> Any:
    """Defer optional-backend imports until someone actually names one.

    PEP 562. Without it, ``import sightline.store`` would pull in qdrant-client,
    psycopg and faiss, and the core-dependencies-only CI job — the one that
    proves the promise in the README — would fail at import.
    """
    lazy = {
        "PostFilterStore": ("sightline.store.postfilter", "PostFilterStore"),
        "QdrantVectorStore": ("sightline.store.qdrant_store", "QdrantVectorStore"),
        "PgVectorStore": ("sightline.store.pgvector_store", "PgVectorStore"),
        "FaissOracleStore": ("sightline.store.faiss_oracle", "FaissOracleStore"),
        "build_oracle": ("sightline.store.faiss_oracle", "build_oracle"),
    }
    if attr in lazy:
        import importlib

        module_name, target = lazy[attr]
        return getattr(importlib.import_module(module_name), target)
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
