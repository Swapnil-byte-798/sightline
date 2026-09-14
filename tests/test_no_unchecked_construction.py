"""Enforce by test what the type system cannot.

``Hit`` means "this survived a re-check against live permissions". Python will
happily let any module construct one, so the invariant is guarded here instead:
only ``authz/recheck.py`` may build a ``Hit``. If this test fails, someone has
created a path to the model that skips authorisation.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "sightline"
ALLOWED = {"authz/recheck.py"}
# `Hit(` but not `UncheckedHit(` and not a bare annotation like `list[Hit]`.
PATTERN = re.compile(r"(?<![A-Za-z_])Hit\s*\(")
# Quoted spans, so a docstring or an error message mentioning "Hit(" is not a hit.
STRINGS = re.compile(r"'''.*?'''|\"\"\".*?\"\"\"|'[^'\n]*'|\"[^\"\n]*\"", re.DOTALL)


def _code_only(source: str) -> list[tuple[int, str]]:
    """Blank out string literals and comments, preserving line numbers."""
    blanked = STRINGS.sub(lambda m: re.sub(r"\S", " ", m.group(0)), source)
    out = []
    for lineno, line in enumerate(blanked.splitlines(), 1):
        out.append((lineno, line.split("#", 1)[0]))
    return out


def test_hit_is_constructed_only_by_recheck():
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel in ALLOWED:
            continue
        raw = path.read_text(encoding="utf-8").splitlines()
        for lineno, code in _code_only(path.read_text(encoding="utf-8")):
            if "UncheckedHit(" in code:
                continue
            if PATTERN.search(code):
                offenders.append(f"{rel}:{lineno}: {raw[lineno - 1].strip()[:90]}")
    assert not offenders, "Hit constructed outside recheck:\n" + "\n".join(offenders)
