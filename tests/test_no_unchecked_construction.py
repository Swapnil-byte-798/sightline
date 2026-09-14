"""Enforce by test what the type system cannot.

``Hit`` means one thing: this result survived a re-check against live
permissions. Python will happily let any module construct one, ``Hit`` has no
runtime guard, and ``@dataclass(frozen=True)`` does not care where it is called
from. So the invariant is guarded here instead — **only ``authz/recheck.py`` may
build a ``Hit``** — and if this test fails, somebody has created a path to the
answer builder that skips authorisation.

The check is an AST walk rather than a grep. ``retrieve.py`` contains the string
``"expected Hit (the output of recheck())"`` inside an error message, which a
regex for ``Hit\\s*\\(`` matches and a parser does not. A security test with a
known false positive gets an exclusion, then another, and then it is off.

Two further things are asserted, because the first assertion alone can pass
vacuously:

* ``recheck.py`` really does construct one, so renaming the class does not
  silently satisfy the rule.
* ``UncheckedHit`` exposes no method at all, so there is no ``.promote()`` for a
  future contributor to reach for.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from sightline.store.base import Hit, UncheckedHit

SRC = Path(__file__).resolve().parents[1] / "src" / "sightline"

#: The one file allowed to construct a ``Hit``. Adding to this set is a security
#: decision, not a test fix, and it should be argued for in the pull request.
ALLOWED = {"authz/recheck.py"}

PROTECTED = "Hit"


def _constructions(path: Path) -> list[int]:
    """Line numbers where ``Hit(...)`` is *called* in ``path``.

    Attribute calls count too: ``base.Hit(...)`` and ``Hit(...)`` are the same
    construction wearing different imports.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute) else None
        )
        if name == PROTECTED:
            lines.append(node.lineno)
    return lines


def test_hit_is_constructed_only_inside_recheck():
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in ALLOWED:
            continue
        for lineno in _constructions(path):
            offenders.append(f"src/sightline/{rel}:{lineno}")
    assert not offenders, (
        "Hit() is constructed outside authz/recheck.py, which means there is a way "
        "to reach the answer builder without consulting live permissions:\n  "
        + "\n  ".join(offenders)
    )


def test_recheck_actually_constructs_one():
    """Guard against the vacuous pass.

    If ``Hit`` were renamed or the construction inlined elsewhere, the test above
    would go green while the property it protects had disappeared.
    """
    allowed_lines = [
        lineno for rel in ALLOWED for lineno in _constructions(SRC / rel)
    ]
    assert allowed_lines, (
        f"no Hit() construction found in {sorted(ALLOWED)}; either the class was "
        "renamed or the recheck path no longer builds one, and this test has been "
        "protecting nothing"
    )


def test_unchecked_hit_offers_no_way_up():
    """``UncheckedHit`` has no method that produces a ``Hit``.

    Deliberate: the dangerous path requires importing ``recheck``, which needs the
    tuple store, so writing the leak means writing an import that looks wrong in
    review.
    """
    public = {
        name
        for name in vars(UncheckedHit)
        if not name.startswith("_") and callable(getattr(UncheckedHit, name, None))
    }
    assert public == set(), f"UncheckedHit grew methods: {sorted(public)}"


def test_the_two_types_are_not_interchangeable():
    """A ``Hit`` carries two fields an ``UncheckedHit`` cannot supply.

    ``why_allowed`` and ``checked_at_epoch`` are the evidence of the check. They
    have no defaults, so a construction that skips the check does not merely
    violate a convention — it does not compile in a reviewer's head either.
    """
    hit_fields = set(Hit.__dataclass_fields__)
    unchecked_fields = set(UncheckedHit.__dataclass_fields__)
    assert {"why_allowed", "checked_at_epoch"} <= hit_fields - unchecked_fields
    for name in ("why_allowed", "checked_at_epoch"):
        field = Hit.__dataclass_fields__[name]
        assert field.default is dataclasses.MISSING, (
            f"Hit.{name} gained a default; the evidence of the permission check is "
            "now optional, which is how it stops being written"
        )


# NOT TESTED HERE, AND WHY
# ------------------------
# An earlier version of this file also flagged ``dataclasses.replace(hit, ...)``
# outside recheck.py, on the theory that rewriting a checked hit produces a
# result carrying the evidence of one authorisation and the content of another.
# It was removed: statically, ``replace(hit, ...)`` in ``postfilter.py`` is
# indistinguishable from the same call on a ``Hit``, because the check is a
# name heuristic and the variable is called ``hit`` in both places. A security
# test with a standing false positive gets an exclusion, then another, and then
# somebody turns it off. Catching that shape needs real type inference — run
# ``mypy`` for it — or a ``Hit`` that is not a dataclass, which would cost more
# than it buys.
