"""The README publishes numbers it did not compute. This is the leash.

``eval/report.py`` rewrites the machine-managed blocks in ``README.md`` from a
real run, and ``make readme-check`` fails the build when a committed figure
drifts. That works only if the document and the generator agree on the block
names, the marker spelling, and which figures are promised — and that agreement
lives in two files maintained by two different people.

So it is asserted here, structurally, before anybody spends four minutes on an
eval to discover a typo. The most valuable assertion is the least obvious one:
**no placeholder may sit outside a managed block**, because a placeholder the
generator never reaches is a number this document promises forever and never
delivers.

Nothing here checks a *value*. Values are measurements and belong to CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
MAKEFILE = REPO_ROOT / "Makefile"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

PLACEHOLDER = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
#: Must match ``eval.report._BEGIN`` / ``_END`` exactly.
MARKER = re.compile(r"<!-- sightline:(begin|end):([a-z_-]+) -->")
CONTRACT_HEADING = "## For `eval/report.py`"

#: Targets CI invokes by name. A workflow step calling a target that does not
#: exist fails four minutes into a run instead of here.
REQUIRED_TARGETS = (
    "install",
    "test",
    "lint",
    "typecheck",
    "ingest",
    "eval",
    "headline",
    "serve",
    "docker-up",
    "readme-check",
    "oracle",
    "mutation",
)


@pytest.fixture(scope="module")
def readme() -> str:
    if not README.exists():  # pragma: no cover - the file is committed
        pytest.skip("README.md is not present")
    return README.read_text(encoding="utf-8")


def _declared(readme: str) -> set[str]:
    """Placeholder names from the contract section, and only that section.

    Bounded at the next heading so a backticked ``LICENSE`` three sections later
    is not mistaken for an undelivered measurement.
    """
    assert CONTRACT_HEADING in readme, "the README no longer states its contract"
    body = readme.split(CONTRACT_HEADING, 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"`([A-Z][A-Z0-9_]+)`", body))


def _managed_spans(readme: str) -> list[tuple[int, int, str]]:
    """``(start, end, name)`` for each managed block, in document order."""
    spans: list[tuple[int, int, str]] = []
    open_at: tuple[int, str] | None = None
    for match in MARKER.finditer(readme):
        kind, name = match.groups()
        if kind == "begin":
            assert open_at is None, f"nested marker block: {name} inside {open_at[1]}"
            open_at = (match.start(), name)
        else:
            assert open_at is not None and open_at[1] == name, f"unbalanced end: {name}"
            spans.append((open_at[0], match.end(), name))
            open_at = None
    assert open_at is None, f"unterminated marker block: {open_at}"
    return spans


# --------------------------------------------------------------------------
# The blocks
# --------------------------------------------------------------------------


def test_the_readme_carries_the_blocks_the_generator_writes(readme):
    """Missing blocks are not an error for ``report.py`` — it inserts them before
    ``## Licence``. But inserted blocks land wherever the tool decides, and the
    surrounding prose then explains a table that is three sections away."""
    report = pytest.importorskip("eval.report", reason="eval.report is not present yet")
    present = [name for _, _, name in _managed_spans(readme)]
    assert present == list(report.BLOCKS), (
        f"README blocks {present} do not match eval.report.BLOCKS {list(report.BLOCKS)}"
    )


def test_the_marker_spelling_matches_the_generator(readme):
    """A one-character difference here means every block is silently re-inserted
    on top of the one already in the document."""
    report = pytest.importorskip("eval.report", reason="eval.report is not present yet")
    for name in report.BLOCKS:
        assert report._BEGIN.format(name=name) in readme  # noqa: SLF001
        assert report._END.format(name=name) in readme  # noqa: SLF001


def test_marker_blocks_are_balanced_and_unique(readme):
    spans = _managed_spans(readme)
    names = [name for _, _, name in spans]
    assert len(names) == len(set(names)), f"duplicate marker block: {names}"
    assert names, "the README declares no generated blocks at all"


# --------------------------------------------------------------------------
# The placeholders
# --------------------------------------------------------------------------


def test_no_placeholder_escapes_the_managed_blocks(readme):
    """The assertion that matters. A placeholder in unmanaged prose is a promise
    nothing keeps: ``report.py`` only ever writes between its own markers."""
    spans = _managed_spans(readme)
    contract_at = readme.index(CONTRACT_HEADING)
    stray = [
        match.group(1)
        for match in PLACEHOLDER.finditer(readme)
        if match.start() > contract_at or not any(s <= match.start() < e for s, e, _ in spans)
    ]
    assert not stray, (
        "placeholders outside a machine-managed block, where nothing will ever "
        f"replace them: {sorted(set(stray))}"
    )


def test_every_placeholder_is_declared(readme):
    """A placeholder the generator has never heard of stays a placeholder."""
    used = set(PLACEHOLDER.findall(readme))
    undeclared = used - _declared(readme)
    assert not undeclared, f"used but not declared to eval/report.py: {sorted(undeclared)}"


def test_no_declared_placeholder_is_unused(readme):
    """The other direction: a declared name nothing uses is a number nobody reads."""
    unused = _declared(readme) - set(PLACEHOLDER.findall(readme))
    assert not unused, f"declared but never used: {sorted(unused)}"


def test_the_readme_says_the_numbers_are_generated(readme):
    """The honesty claim is load-bearing and must survive an edit.

    Delete this sentence and the placeholders become a stylistic quirk rather
    than a promise CI enforces.
    """
    assert "make readme-check" in readme
    assert "make headline" in readme
    assert "placeholder" in readme.lower()


# --------------------------------------------------------------------------
# The commands
# --------------------------------------------------------------------------


def test_ci_only_calls_make_targets_that_exist():
    if not (MAKEFILE.exists() and WORKFLOW.exists()):  # pragma: no cover
        pytest.skip("Makefile or workflow not present")
    targets = set(re.findall(r"^([a-zA-Z][\w-]*):", MAKEFILE.read_text(), re.M))
    called = set(re.findall(r"\bmake ([a-zA-Z][\w-]*)", WORKFLOW.read_text()))
    assert called <= targets, f"CI calls missing make targets: {sorted(called - targets)}"


def test_the_documented_targets_exist():
    if not MAKEFILE.exists():  # pragma: no cover
        pytest.skip("Makefile not present")
    targets = set(re.findall(r"^([a-zA-Z][\w-]*):", MAKEFILE.read_text(), re.M))
    missing = set(REQUIRED_TARGETS) - targets
    assert not missing, f"missing make targets: {sorted(missing)}"
