"""Shared fixtures, and the two import tricks the suite needs.

The entire suite runs on ``pydantic``, ``fastapi``, ``numpy`` and ``httpx`` and
nothing else. No Qdrant, no Postgres, no ONNX, no network, no fixture files to
load. That is not minimalism for its own sake: a permission test with a setup
step is a permission test that gets skipped, and the tests in this directory are
the only thing standing between "the index is a hint" and a demo that leaks.

The two tricks:

1. ``src/`` goes on ``sys.path`` when ``sightline`` is not installed, so
   ``pytest`` works in a fresh clone with no build step.
2. This directory goes on ``sys.path`` too, because pytest 9 imports test
   modules through ``importlib`` and no longer adds it. Without this,
   ``from _world import ...`` fails and every test file has to re-declare the
   fixture corpus.

Modules that may legitimately be absent are reached through
:func:`pytest.importorskip`, so a missing module skips with a reason rather than
failing with an ImportError that reads like a real defect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _world import (  # noqa: E402  - must follow the sys.path insert above
    REPO_ROOT,
    SRC_ROOT,
    WORLD_EXPECTATIONS,
    WORLD_PRINCIPALS,
    WORLD_TUPLES,
    ensure_importable,
    group_chain_tuples,
    make_corpus,
)

ensure_importable()

__all__ = [
    "REPO_ROOT",
    "SRC_ROOT",
    "WORLD_EXPECTATIONS",
    "WORLD_PRINCIPALS",
    "WORLD_TUPLES",
    "make_corpus",
]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def world():
    """A :class:`MemoryTupleStore` holding :data:`WORLD_TUPLES`."""
    tuples = pytest.importorskip(
        "sightline.authz.tuples", reason="sightline.authz.tuples is not present yet"
    )
    return tuples.MemoryTupleStore(WORLD_TUPLES)


@pytest.fixture
def group_chain():
    """Factory returning a store with ``n`` nested groups. See ``_world``."""
    tuples = pytest.importorskip(
        "sightline.authz.tuples", reason="sightline.authz.tuples is not present yet"
    )
    return lambda n: tuples.MemoryTupleStore(group_chain_tuples(n))
