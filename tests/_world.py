"""The fixture world, and the corpus builder. Plain module, no pytest.

Kept out of ``conftest.py`` because pytest 9 imports conftest under its own
private module name, so ``from conftest import ...`` inside a test either fails
or imports a second copy of the file. A separate module imported through the
path ``conftest`` puts on ``sys.path`` has neither problem.

ONE SMALL ORGANISATION, REUSED EVERYWHERE
-----------------------------------------
Most of the suite runs against :data:`WORLD_TUPLES`, so a reader learns the
fixture once. Two entries look like typos and are not:

* ``group:legal-interns`` exists only so that a prefix match on subject ids
  (planted mutant M9) hands an intern the merger memo.
* ``user:dave`` holds nothing, anywhere. He is the fail-closed principal, and
  every "empty means nobody" assertion in the suite is written from his seat.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def ensure_importable() -> None:
    """Put ``src/`` on the path when ``sightline`` is not installed.

    Conditional, for the same reason ``eval/__init__.py`` is conditional: if
    somebody has run ``pip install -e .``, their installed copy is the one under
    test, and prepending unconditionally would mean the suite measured the
    working tree while CI shipped the wheel.
    """
    try:  # pragma: no cover - exercised by every import below it
        import sightline  # noqa: F401
    except ImportError:
        if SRC_ROOT.is_dir():
            sys.path.insert(0, str(SRC_ROOT))


ensure_importable()


WORLD_TUPLES: tuple[str, ...] = (
    # -- people into groups ------------------------------------------------
    "group:legal#member@user:alice",
    "group:legal-interns#member@user:mallory",
    "group:hr#member@user:carol",
    # -- groups into groups (nesting is a userset pointing at a userset) ----
    "group:partners#member@group:legal#member",
    "group:everyone#member@group:partners#member",
    # -- documents ---------------------------------------------------------
    "doc:merger#viewer@group:legal#member",
    "doc:handbook#viewer@group:everyone#member",
    "doc:bob-notes#viewer@user:bob",
    "doc:draft#owner@user:alice",
    # -- inheritance through a folder --------------------------------------
    "folder:hr#viewer@group:hr#member",
    "doc:payroll#parent@folder:hr",
)

#: Who can see what, according to a human reading the tuples above. Every
#: automated answer in the suite is checked against this table rather than
#: against another piece of the implementation — a test that derives its
#: expectation from the code proves only that the code is self-consistent,
#: which is also true of a broken one.
WORLD_EXPECTATIONS: tuple[tuple[str, str, bool], ...] = (
    ("user:alice", "doc:merger", True),
    ("user:alice", "doc:handbook", True),  # legal -> partners -> everyone
    ("user:alice", "doc:draft", True),  # owner implies viewer
    ("user:alice", "doc:payroll", False),
    ("user:alice", "doc:bob-notes", False),
    ("user:bob", "doc:bob-notes", True),
    ("user:bob", "doc:merger", False),
    ("user:carol", "doc:payroll", True),  # folder:hr -> doc:payroll
    ("user:carol", "doc:merger", False),
    ("user:mallory", "doc:merger", False),  # group:legal-interns is not group:legal
    ("user:mallory", "doc:handbook", False),
    ("user:dave", "doc:merger", False),
    ("user:dave", "doc:handbook", False),
    ("user:dave", "doc:payroll", False),
)

#: Principals used by the retrieval-side tests, most-permitted first.
WORLD_PRINCIPALS: tuple[str, ...] = (
    "user:alice",
    "user:bob",
    "user:carol",
    "user:mallory",
    "user:dave",
)


def group_chain_tuples(n: int) -> list[str]:
    """``doc:deep`` granted to ``group:g1``, nested ``n`` deep, ending in a user.

    Reaching ``user:deep-user`` costs exactly ``n`` subject hops, which is what
    makes the depth-boundary test an equality rather than an approximation.
    """
    rows = ["doc:deep#viewer@group:g1#member"]
    for i in range(1, n):
        rows.append(f"group:g{i}#member@group:g{i + 1}#member")
    rows.append(f"group:g{n}#member@user:deep-user")
    return rows


def make_corpus(
    n: int, groups: list[str] | tuple[str, ...], dim: int = 32, seed: int = 7
) -> tuple[list[Any], list[Any]]:
    """Round-robin one grant token per chunk: the friendliest possible ACL shape.

    Friendliest matters. If post-filtering still collapses when every chunk
    belongs to exactly one evenly-sized group, the collapse is not an artefact of
    a contrived permission graph.

    Args:
        n: How many chunks, one per document.
        groups: Grant-token strings to round-robin over.
        dim: Vector dimensionality.
        seed: Fixed, because a headline number from an unseeded run is a story.

    Returns:
        ``(chunks, vectors)`` ready for :meth:`VectorStore.upsert`.
    """
    import numpy as np

    from sightline.types import Chunk, GrantToken, ObjectRef

    rng = np.random.default_rng(seed)
    chunks: list[Any] = []
    vectors: list[Any] = []
    for i in range(n):
        chunks.append(
            Chunk(
                id=f"c{i}",
                object=ObjectRef("doc", str(i)),
                text=f"document {i}",
                grant_tokens=frozenset({GrantToken(groups[i % len(groups)])}),
            )
        )
        vectors.append(rng.normal(size=dim).astype("float32"))
    return chunks, vectors
