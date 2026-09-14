"""The headline metric: mutation kill rate.

Fifteen deliberate permission bugs, each one a mistake a competent engineer
could make on a Thursday afternoon. The harness copies the repository into a
scratch directory, applies exactly one mutation to the copy, runs the test suite
against it, and records whether the suite noticed. A mutant that causes a test
failure is **killed**. A mutant that passes every test **survives**, and a
survivor means the suite does not cover that behaviour whatever the coverage
percentage says.

THE FIFTEEN ARE PRE-REGISTERED
------------------------------
They are specified in ``docs/FRD.md`` §8.2, written down before the tests were,
and removing one is a review-blocking change. That ordering is the whole
methodology: a mutation suite whose author may add and drop mutants after seeing
the results measures the author's taste, not the tests. This module implements
that table and nothing else. Where a name appears in the source — and it does,
in about thirty docstrings — it refers to the same mutant as the row here.

WHY A COPY OF THE REPOSITORY, NOT MONKEYPATCHING
------------------------------------------------
Runtime patching is easier and it cannot express half of these. ``M3`` is caught
in part by a test that greps the source for constructions of ``Hit``; a
monkeypatched function body is invisible to it. ``M15`` is about which
transaction a statement sits inside, which is a property of the text. So each
mutation is a set of exact-string edits applied to a throwaway tree, verified to
apply exactly once and to still parse as Python. The working tree is never
touched, which also means this is safe to run while somebody else is editing it.

An edit whose anchor no longer matches is reported ``not_applicable`` with the
anchor quoted — never silently skipped, and never counted as a kill. A harness
that quietly drops the mutants it can no longer apply reports a rising kill rate
as the code rots away from it.

THE CANARY
----------
Before the fifteen, the harness applies :data:`CANARY` — ``check()`` returns
allow for everything — and requires the suite to fail. If the canary *survives*,
the overlay is not taking effect (a ``conftest.py`` that puts the real ``src``
back on the path would do it) and every subsequent "survived" would be a lie. In
that case the run aborts and reports nothing. Likewise a baseline run of the
unmutated copy must pass: mutants scored against an already-red suite are all
trivially "killed".

REPORT THE NUMBER YOU MEASURED
------------------------------
12/15 with the three survivors named and explained is a more useful artefact
than an unsubstantiated 15/15, and it is the one this harness will print if that
is what happened. ``docs/PRD.md`` sets the publishable floor at 13/15 with every
survivor named in the README. :mod:`eval.report` writes the survivors out in
full, because a kill rate without its survivor list is a percentage, not a
result.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import eval  # noqa: F401  - path bootstrap; see eval/__init__.py
from eval import REPO_ROOT

__all__ = [
    "CANARY",
    "MUTATIONS",
    "Edit",
    "MutantResult",
    "Mutation",
    "MutationReport",
    "SuiteRun",
    "main",
    "materialise",
    "render_markdown",
    "run_all",
    "run_mutation",
    "run_suite",
]

#: Directories never copied into the scratch tree. ``.venv`` and ``.git`` are
#: large and irrelevant; ``docs/build`` holds a generated PDF.
_IGNORE_DIRS = frozenset({".git", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache", "build"})

#: Default per-mutant wall-clock budget. Generous, because a mutant that causes
#: an infinite loop should be reported as a timeout rather than hang CI.
DEFAULT_TIMEOUT_SECONDS = 900.0


# --------------------------------------------------------------------------
# Edits
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Edit:
    """One exact-string replacement in one file.

    Exact strings rather than AST node matching, and the trade is deliberate. An
    AST rewrite survives reformatting; an exact string does not, and breaks
    loudly when the code it targets moves. Loud breakage is what this harness
    wants: a mutant that silently stops applying is a mutant that starts
    reporting a kill rate for a bug the code no longer contains.
    """

    path: str
    anchor: str
    replacement: str

    def apply(self, source: str) -> str:
        """Return ``source`` with the anchor replaced.

        Raises:
            LookupError: If the anchor does not appear exactly once. Zero means
                the target moved; more than one means the anchor is ambiguous
                and the mutation would land somewhere unintended.
        """
        count = source.count(self.anchor)
        if count != 1:
            raise LookupError(
                f"{self.path}: anchor appears {count} times, expected exactly 1: "
                f"{self.anchor.strip()[:90]!r}"
            )
        return source.replace(self.anchor, self.replacement, 1)


@dataclass(frozen=True, slots=True)
class Mutation:
    """One pre-registered permission bug.

    ``expected_killer`` is copied from the FRD table and is a claim about which
    test should catch this. It is not checked automatically — asserting that a
    specific test is the one that failed would make the harness brittle against
    ordinary test renames — but it is printed next to a survivor, because "which
    test was supposed to catch this" is the first question a survivor raises.
    """

    id: str
    title: str
    #: Why somebody writes this bug. Mutants that nobody would plausibly write
    #: measure nothing: killing them is free and surviving them is irrelevant.
    rationale: str
    expected_killer: str
    edits: tuple[Edit, ...]


# --------------------------------------------------------------------------
# The fifteen
# --------------------------------------------------------------------------
#
# Ordered as in docs/FRD.md §8.2. Every anchor below is a verbatim slice of the
# source it targets; `python -m eval.mutation verify` checks that all of them
# still apply, and CI runs that before it runs anything else.

MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        id="M1",
        title="FilterPlan.is_stale() always returns False",
        rationale=(
            "The epoch comparison shows up in a profile as a field read on every "
            "cached plan, and 'the recheck catches it anyway' is a sentence that "
            "gets said in review and sounds reasonable."
        ),
        expected_killer=(
            "Revocation test: tighten a document's ACL, query with a cached plan, "
            "expect the result to disappear"
        ),
        edits=(
            Edit(
                "src/sightline/types.py",
                "    def is_stale(self, current_epoch: int) -> bool:\n"
                "        return self.epoch < current_epoch",
                "    def is_stale(self, current_epoch: int) -> bool:\n"
                "        return False  # M1: the epoch staleness check is dropped",
            ),
        ),
    ),
    Mutation(
        id="M2",
        title="Group condition combination uses OR where it should use AND",
        rationale=(
            "Within one chunk's token set the test really is OR — any grant "
            "suffices. Carrying that intuition up one level, to the plan's "
            "conditions, flips an intersection into a union."
        ),
        expected_killer=(
            "Plan/oracle agreement property test on a principal whose access "
            "requires two conditions"
        ),
        edits=(
            Edit(
                "src/sightline/authz/compile.py",
                "    return bool(conditions) and all(conditions)",
                "    return bool(conditions) and any(conditions)  # M2: OR where AND was meant",
            ),
            Edit(
                "src/sightline/store/memory.py",
                "    if by_ids and not _id_condition_holds(plan, object_ref):\n"
                "        return False\n"
                "    if by_tokens and not (plan.grant_tokens & frozenset(grant_tokens)):\n"
                "        return False\n"
                "    return True",
                "    # M2: OR where AND was meant\n"
                "    id_ok = (not by_ids) or _id_condition_holds(plan, object_ref)\n"
                "    token_ok = (not by_tokens) or bool(\n"
                "        plan.grant_tokens & frozenset(grant_tokens)\n"
                "    )\n"
                "    return id_ok or token_ok",
            ),
            Edit(
                "src/sightline/store/memory.py",
                "        if by_ids:\n"
                "            rows = self._rows_for_ids(plan.explicit_ids)\n"
                "            if by_tokens and rows.size:\n"
                "                rows = rows[self._token_mask(rows, plan.grant_tokens)]\n"
                "            return rows",
                "        if by_ids:\n"
                "            rows = self._rows_for_ids(plan.explicit_ids)\n"
                "            if by_tokens:  # M2: OR where AND was meant\n"
                "                extra = np.nonzero(self._token_mask(None, plan.grant_tokens))[0]\n"
                "                rows = np.union1d(rows, extra.astype(np.int64, copy=False))\n"
                "            return rows",
            ),
        ),
    ),
    Mutation(
        id="M3",
        title="The pipeline skips recheck() and builds Hit straight from UncheckedHit",
        rationale=(
            "One graph walk per distinct object is the most expensive thing on "
            "the read path, and the index just told us what the tokens were. "
            "This is the optimisation the whole product exists to refuse."
        ),
        expected_killer="Source-grep test, plus the stale-index differential test",
        edits=(
            Edit(
                "src/sightline/retrieve.py",
                "        with obs.span(\"authz.recheck\"):\n"
                "            hits, dropped = self.rechecker.recheck(str(principal), unchecked)",
                "        with obs.span(\"authz.recheck\"):\n"
                "            # M3: the index is treated as the authority\n"
                "            hits = [\n"
                "                Hit(\n"
                "                    chunk_id=h.chunk_id,\n"
                "                    score=h.score,\n"
                "                    text=h.text,\n"
                "                    object_ref=h.object_ref,\n"
                "                    why_allowed=h.matched_token or \"index\",\n"
                "                    checked_at_epoch=plan.epoch,\n"
                "                )\n"
                "                for h in unchecked\n"
                "            ]\n"
                "            dropped = 0",
            ),
        ),
    ),
    Mutation(
        id="M4",
        title="Recheck trusts UncheckedHit.matched_token instead of re-deriving",
        rationale=(
            "The index already matched a token, and re-deriving it costs a graph "
            "walk. The token is advisory, written at ingest, and may name a group "
            "that has since been deleted — but nothing in the type says so."
        ),
        expected_killer=(
            "Forge a hit with a token the principal no longer holds; expect it dropped"
        ),
        edits=(
            Edit(
                "src/sightline/authz/recheck.py",
                "        if hit.object_ref not in decisions:\n"
                "            decisions[hit.object_ref] = _decide(\n"
                "                store, hit.object_ref, relation, principal, max_depth\n"
                "            )",
                "        if hit.object_ref not in decisions:\n"
                "            # M4: the index's advisory token is trusted\n"
                "            decisions[hit.object_ref] = (\n"
                "                Decision(True, (f\"matched_token:{hit.matched_token}\",), 0)\n"
                "                if hit.matched_token\n"
                "                else _decide(store, hit.object_ref, relation, principal, max_depth)\n"
                "            )",
            ),
        ),
    ),
    Mutation(
        id="M5",
        title="Off-by-one in userset expansion: the depth limit permits one extra level",
        rationale=(
            "`depth > limit` versus `depth >= limit` versus `depth > limit + 1` is "
            "the classic. Nobody notices, because the only observable difference "
            "is a graph exactly at the boundary, which no hand-written fixture has."
        ),
        expected_killer="Nested-group test at exactly the depth boundary, both sides",
        edits=(
            Edit(
                "src/sightline/authz/check.py",
                "        if depth > self._max_depth:\n"
                "            self._truncations += 1",
                "        if depth > self._max_depth + 1:  # M5: one level past the bound\n"
                "            self._truncations += 1",
            ),
        ),
    ),
    Mutation(
        id="M6",
        title="An empty grant-token set matches everything instead of nothing",
        rationale=(
            "Every filter library in the world treats an empty filter list as "
            "'no filter'. Carrying that convention into an authorisation filter "
            "turns a principal with no permissions into a principal with all of "
            "them. Fail-open versus fail-closed, in one falsy check."
        ),
        expected_killer=(
            "Query as a principal with zero permissions; expect zero candidates, "
            "not the whole corpus"
        ),
        edits=(
            Edit(
                "src/sightline/store/memory.py",
                "    by_ids = plan.strategy is PlanStrategy.ENUMERATE or bool(plan.explicit_ids)\n"
                "    by_tokens = plan.strategy is PlanStrategy.GRANT_TOKENS or bool(plan.grant_tokens)",
                "    # M6: an empty condition set is read as 'no filter'\n"
                "    by_ids = bool(plan.explicit_ids)\n"
                "    by_tokens = bool(plan.grant_tokens)",
            ),
            Edit(
                "src/sightline/store/memory.py",
                "        return plan.strategy is PlanStrategy.UNFILTERED\n",
                "        return True  # M6: no condition means no restriction\n",
            ),
            Edit(
                "src/sightline/store/memory.py",
                "            return empty  # no condition and not unfiltered: admit nothing.",
                "            return None  # M6: no condition and not unfiltered: admit everything.",
            ),
            Edit(
                "src/sightline/authz/compile.py",
                "        conditions.append(object_id in plan.explicit_ids)",
                "        # M6: an empty id set imposes no condition\n"
                "        conditions.append(not plan.explicit_ids or object_id in plan.explicit_ids)",
            ),
            Edit(
                "src/sightline/authz/compile.py",
                "        conditions.append(bool(plan.grant_tokens & chunk))",
                "        # M6: an empty token set imposes no condition\n"
                "        conditions.append(not plan.grant_tokens or bool(plan.grant_tokens & chunk))",
            ),
        ),
    ),
    Mutation(
        id="M7",
        title="The post-filter baseline pads results back up to k from the unfiltered pool",
        rationale=(
            "Post-filtering returns two results when the caller asked for ten, "
            "and that looks like a bug to whoever is on support. Padding makes "
            "the symptom go away by serving the forbidden documents."
        ),
        expected_killer="Differential leak test against the oracle",
        edits=(
            Edit(
                "src/sightline/store/postfilter.py",
                "        kept = survivors[:k]\n",
                "        kept = survivors[:k]\n"
                "        if len(kept) < k:  # M7: top up from the unfiltered pool\n"
                "            already = {h.chunk_id for h in kept}\n"
                "            for filler in candidates:\n"
                "                if len(kept) >= k:\n"
                "                    break\n"
                "                if filler.chunk_id not in already:\n"
                "                    kept.append(filler)\n",
            ),
        ),
    ),
    Mutation(
        id="M8",
        title="Plan cache keyed on principal only, ignoring the epoch",
        rationale=(
            "Keying on the epoch means a single permission write anywhere "
            "invalidates every entry, and somebody will measure that hit rate and "
            "call it a cache that does not work. Then they add a TTL, and a TTL "
            "looks short in a meeting and is five minutes of a revoked read."
        ),
        expected_killer="Write a tuple, requery within the cache TTL, expect the new policy",
        edits=(
            Edit(
                "src/sightline/authz/compile.py",
                "        return (str(principal), epoch, relation, namespace)",
                "        return (str(principal), 0, relation, namespace)  # M8: epoch not in the key",
            ),
            Edit(
                "src/sightline/authz/compile.py",
                "        return None if plan.is_stale(epoch) else plan",
                "        return plan  # M8: and no belt-and-braces staleness check either",
            ),
        ),
    ),
    Mutation(
        id="M9",
        title="Reverse tuple lookup matches subject ids by string prefix",
        rationale=(
            "Matching on the unparsed subject string is one line shorter and "
            "works on every fixture anybody writes by hand. Then somebody creates "
            "`group:legal-interns` and it inherits everything `group:legal` has."
        ),
        expected_killer="Adversarial naming fixture in the corpus generator",
        edits=(
            Edit(
                "src/sightline/authz/tuples.py",
                "            found = self._by_principal.get(self._pkey(principal), ())\n"
                "            if relation is None:\n"
                "                return list(found)",
                "            want = self._pkey(principal)  # M9: prefix match, not exact match\n"
                "            found = [\n"
                "                t\n"
                "                for key, group in self._by_principal.items()\n"
                "                if key[0] == want[0]\n"
                "                and key[2] == want[2]\n"
                "                and key[1].startswith(want[1])\n"
                "                for t in group\n"
                "            ]\n"
                "            if relation is None:\n"
                "                return list(found)",
            ),
        ),
    ),
    Mutation(
        id="M10",
        title="Recheck processes only the first batch of hits and passes the rest through",
        rationale=(
            "'Batch the recheck' is a real and correct latency instruction. "
            "Implementing it as 'check the first page' rather than 'de-duplicate "
            "by object' is the misreading, and it only leaks on large k."
        ),
        expected_killer=(
            "Query with k greater than the batch size, with a revoked document "
            "ranked late"
        ),
        edits=(
            Edit(
                "src/sightline/authz/recheck.py",
                "    for hit in hits:\n"
                "        if hit.object_ref not in decisions:",
                "    for _batch_index, hit in enumerate(hits):\n"
                "        if _batch_index >= 8:  # M10: only the first batch is checked\n"
                "            survivors.append(\n"
                "                Hit(\n"
                "                    chunk_id=hit.chunk_id,\n"
                "                    score=hit.score,\n"
                "                    text=hit.text,\n"
                "                    object_ref=hit.object_ref,\n"
                "                    why_allowed=\"batch_passthrough\",\n"
                "                    checked_at_epoch=epoch,\n"
                "                )\n"
                "            )\n"
                "            continue\n"
                "        if hit.object_ref not in decisions:",
            ),
        ),
    ),
    Mutation(
        id="M11",
        title="The 'you may not see it' refusal differs from the 'it does not exist' one",
        rationale=(
            "Two different situations, two different call sites, and a helpful "
            "error message at each. The distinction is a corpus enumerator: ask "
            "one question per guessed project name and read the refusal."
        ),
        expected_killer="Existence-protection byte-equality test",
        edits=(
            Edit(
                "src/sightline/retrieve.py",
                "                answer = self._empty_answer(plan)\n"
                "                seq = self._audit_row(\n"
                "                    rid, principal, question, plan, diag,\n"
                "                    internal_reason=RefusalReason.NO_PERMITTED_EVIDENCE,\n"
                "                )",
                "                # M11: a more helpful message on the permission path\n"
                "                answer = dataclasses.replace(\n"
                "                    self._empty_answer(plan),\n"
                "                    text=\"You do not have access to any documents matching \"\n"
                "                         \"this question.\",\n"
                "                    existence_protected=False,\n"
                "                )\n"
                "                seq = self._audit_row(\n"
                "                    rid, principal, question, plan, diag,\n"
                "                    internal_reason=RefusalReason.NO_PERMITTED_EVIDENCE,\n"
                "                )",
            ),
            Edit(
                "src/sightline/retrieve.py",
                "import re\nimport time\nimport uuid",
                "import dataclasses  # M11\nimport re\nimport time\nimport uuid",
            ),
        ),
    ),
    Mutation(
        id="M12",
        title="Chunk text is included in a refusal payload",
        rationale=(
            "An ungrounded refusal is the hardest failure to debug, so somebody "
            "attaches the evidence that failed to ground. It is the realistic way "
            "document content escapes: not through the answer, through the "
            "debugging."
        ),
        expected_killer=(
            "Refusal-path contract test asserting no chunk text in any "
            "non-grounded response"
        ),
        edits=(
            Edit(
                "src/sightline/retrieve.py",
                "                answer = Answer(\n"
                "                    text=_REFUSAL_UNGROUNDED,\n"
                "                    refused=True,\n"
                "                    refusal_reason=RefusalReason.UNGROUNDED,",
                "                answer = Answer(\n"
                "                    # M12: the evidence that failed to ground, for debugging\n"
                "                    text=_REFUSAL_UNGROUNDED\n"
                "                    + \" [evidence: \"\n"
                "                    + \" | \".join(h.text for h in hits)\n"
                "                    + \"]\",\n"
                "                    refused=True,\n"
                "                    refusal_reason=RefusalReason.UNGROUNDED,",
            ),
        ),
    ),
    Mutation(
        id="M13",
        title="A document's ACL change does not enqueue a reindex",
        rationale=(
            "Membership changes correctly cost zero vector writes (ADR 0002), and "
            "'permission change means no reindex' is a one-word generalisation of "
            "that away from being true. The index then serves the old, more "
            "permissive tokens until something else touches the document."
        ),
        expected_killer="Reindex-queue counter test for both event types",
        edits=(
            Edit(
                "src/sightline/ingest/pipeline.py",
                "            \"visible to anyone (ADR 0003). Delete its chunks rather than \"\n"
                "            \"reindexing them.\"\n"
                "        )\n",
                "            \"visible to anyone (ADR 0003). Delete its chunks rather than \"\n"
                "            \"reindexing them.\"\n"
                "        )\n"
                "    # M13: permission changes never require a vector rewrite\n"
                "    return 0\n",
            ),
        ),
    ),
    Mutation(
        id="M14",
        title="Citations are parsed out of the generated text, not reconstructed",
        rationale=(
            "The model already emitted the chunk ids it used. Trusting them is "
            "one dictionary lookup less, and every citation it produces looks "
            "correct right up until the model names an id it saw in a previous "
            "request."
        ),
        expected_killer=(
            "Synthesiser fixture that emits a citation to a forbidden chunk id; "
            "expect it dropped"
        ),
        edits=(
            Edit(
                "src/sightline/generate.py",
                "        kept = tuple(str(i) for i in ids if str(i) in allowed)",
                "        kept = tuple(str(i) for i in ids)  # M14: trust the ids the model named",
            ),
            Edit(
                "src/sightline/retrieve.py",
                "            hit = by_id.get(chunk_id)\n"
                "            if hit is None or chunk_id in seen:\n"
                "                continue",
                "            hit = by_id.get(chunk_id)\n"
                "            if chunk_id in seen:\n"
                "                continue\n"
                "            if hit is None:  # M14: cite what was said, not what survived\n"
                "                seen.add(chunk_id)\n"
                "                out.append(\n"
                "                    Citation(\n"
                "                        chunk_id=chunk_id,\n"
                "                        object=ObjectRef(\"doc\", chunk_id),\n"
                "                        score=0.0,\n"
                "                    )\n"
                "                )\n"
                "                continue",
            ),
        ),
    ),
    Mutation(
        id="M15",
        title="The epoch increment is moved outside the tuple write transaction",
        rationale=(
            "Two statements, one transaction, and the increment looks like "
            "bookkeeping rather than part of the write. Split them and a crash in "
            "between leaves a permission change that every cached plan is entitled "
            "to ignore."
        ),
        expected_killer="Induced-crash atomicity test between write and increment",
        edits=(
            Edit(
                "src/sightline/authz/tuples.py",
                "            self._conn.execute(\"UPDATE policy SET value = value + 1 WHERE key='epoch'\")\n"
                "            row2 = self._conn.execute(\"SELECT value FROM policy WHERE key='epoch'\").fetchone()\n"
                "            return int(row2[0]), applied",
                "            pass  # M15: the epoch increment has left this transaction\n"
                "        with self._lock, self._conn:\n"
                "            self._conn.execute(\"UPDATE policy SET value = value + 1 WHERE key='epoch'\")\n"
                "            row2 = self._conn.execute(\n"
                "                \"SELECT value FROM policy WHERE key='epoch'\"\n"
                "            ).fetchone()\n"
                "        return int(row2[0]), applied",
            ),
            Edit(
                "src/sightline/authz/tuples.py",
                "            for t in w:\n"
                "                applied += int(self._insert(t))\n"
                "            return self._bump(), applied",
                "            for t in w:\n"
                "                applied += int(self._insert(t))\n"
                "        # M15: the epoch increment has left the write's critical section\n"
                "        with self._lock:\n"
                "            return self._bump(), applied",
            ),
        ),
    ),
)

#: Not one of the fifteen. A mutation so total that any suite touching
#: authorisation at all must fail, used as a pre-flight to prove the scratch tree
#: is the tree being tested. If this survives, the harness reports nothing:
#: every "survived" below it would be an artefact of the copy not taking effect.
CANARY = Mutation(
    id="CANARY",
    title="check() allows everything",
    rationale="Pre-flight. Proves the mutated copy is the code under test.",
    expected_killer="Any test that asserts a denial",
    edits=(
        Edit(
            "src/sightline/authz/check.py",
            "    why = path if allowed else ev.notes\n"
            "    return Decision(allowed=allowed, why=why, checked_tuples=ev.checked)",
            "    allowed = True  # CANARY\n"
            "    why = path if allowed else ev.notes\n"
            "    return Decision(allowed=allowed, why=why, checked_tuples=ev.checked)",
        ),
    ),
)


# --------------------------------------------------------------------------
# Applying a mutation
# --------------------------------------------------------------------------


def _ignore(directory: str, names: list[str]) -> set[str]:
    return {n for n in names if n in _IGNORE_DIRS or n.endswith(".pyc")}


def materialise(mutation: Mutation, dest: Path, *, repo_root: Path = REPO_ROOT) -> Path:
    """Copy the repository to ``dest`` and apply one mutation to the copy.

    The whole repository, not just ``src/``. Overlaying a mutated package
    directory on ``PYTHONPATH`` loses to any ``conftest.py`` that puts the real
    source back, and that failure mode is invisible — every mutant survives and
    the suite looks terrible for the wrong reason. Copying the tree and running
    pytest with the copy as its working directory removes the question.

    Raises:
        LookupError: If any edit's anchor no longer applies.
        SyntaxError: If the mutated file does not parse. A mutation that breaks
            the parser measures nothing: every test fails at collection.
    """
    shutil.copytree(repo_root, dest, ignore=_ignore, dirs_exist_ok=True)
    for edit in mutation.edits:
        target = dest / edit.path
        source = target.read_text(encoding="utf-8")
        mutated = edit.apply(source)
        ast.parse(mutated, filename=str(target))
        target.write_text(mutated, encoding="utf-8")
    return dest


# --------------------------------------------------------------------------
# Running the suite
# --------------------------------------------------------------------------

#: Where the child pytest writes its machine-readable report, inside the scratch
#: tree so it disappears with it.
_JUNIT_NAME = ".sightline-mutation-report.xml"

# Fallback only. pytest's terminal summary is a presentation detail and it
# changes: pytest 9 with `-q` prints no "N passed" line at all on a green run,
# which silently made every count zero and every baseline look like a suite that
# collects nothing. That bug cost an afternoon, so the counts now come from
# --junitxml and this regex is the thing that runs when the XML is missing.
_COUNT_RE = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed|deselected)")
_FAIL_LINE_RE = re.compile(r"^(FAILED|ERROR) (\S+)", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class SuiteRun:
    """One pytest invocation, reduced to what the harness decides on."""

    returncode: int
    passed: int
    failed: int
    errors: int
    skipped: int
    duration_s: float
    timed_out: bool
    #: First failing node id, so a survivor's opposite is legible in the report.
    first_failure: str = ""
    tail: str = ""

    @property
    def collected_anything(self) -> bool:
        return (self.passed + self.failed + self.skipped) > 0

    @property
    def green(self) -> bool:
        return self.returncode == 0 and self.collected_anything

    def to_dict(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "duration_s": round(self.duration_s, 2),
            "timed_out": self.timed_out,
            "first_failure": self.first_failure,
        }


def _read_junit(path: Path) -> tuple[dict[str, int], str] | None:
    """Counts and the first failing node id from a JUnit XML report.

    ``None`` when the file is absent or unreadable — pytest writes it even for a
    collection error, so a missing file means the interpreter never got that
    far, and the caller falls back to scraping the terminal output.
    """
    if not path.is_file():
        return None
    try:
        root = ElementTree.parse(path).getroot()
    except (ElementTree.ParseError, OSError):
        return None

    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = failures = errors = skipped = 0
    for suite in suites:
        total += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))

    first = ""
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            where = case.get("file") or case.get("classname") or ""
            first = f"{where}::{case.get('name', '')}".lstrip(":")
            break

    counts = {
        "passed": max(0, total - failures - errors - skipped),
        "failed": failures,
        "error": errors,
        "skipped": skipped,
    }
    return counts, first


def run_suite(
    root: Path,
    *,
    python: str = sys.executable,
    pytest_args: Sequence[str] = (),
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    exitfirst: bool = True,
) -> SuiteRun:
    """Run the test suite rooted at ``root``. Never raises for a test failure.

    Counts come from ``--junitxml``, not from the terminal summary. Parsing the
    summary line is the obvious thing and it is wrong: it is a presentation
    detail that changes between pytest releases, and on the version installed
    here ``-q`` prints no count line at all when everything passes. A harness
    that reads "0 passed" off a green suite reports every mutant as unscorable,
    which is the most expensive kind of wrong — it looks like a result.

    ``-p no:cacheprovider`` because the scratch tree is deleted afterwards and a
    cache directory in it is noise. ``PYTHONDONTWRITEBYTECODE`` for the same
    reason, and because a stale ``__pycache__`` copied from the working tree
    would be the one loaded — which is exactly the class of bug this harness is
    built to avoid having.
    """
    junit = root / _JUNIT_NAME
    argv = [
        python, "-m", "pytest", "-q", "--no-header",
        "-p", "no:cacheprovider", f"--junitxml={junit}",
    ]
    if exitfirst:
        argv.append("-x")
    argv.extend(pytest_args)

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    # The mutated copy is a different checkout; a plan cache or model dir shared
    # with the working tree would carry state across runs.
    env.pop("PYTEST_ADDOPTS", None)

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        code = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        code = -1
        timed_out = True
    duration = time.perf_counter() - started

    parsed = _read_junit(junit)
    if parsed is None:
        counts = {name: 0 for name in ("passed", "failed", "error", "skipped")}
        for value, name in _COUNT_RE.findall(out):
            key = "error" if name.startswith("error") else name
            if key in counts:
                counts[key] = int(value)
        match = _FAIL_LINE_RE.search(out)
        first = match.group(2) if match else ""
    else:
        counts, first = parsed

    return SuiteRun(
        returncode=code,
        passed=counts["passed"],
        failed=counts["failed"],
        errors=counts["error"],
        skipped=counts["skipped"],
        duration_s=duration,
        timed_out=timed_out,
        first_failure=first,
        tail="\n".join(out.strip().splitlines()[-25:]),
    )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

#: ``killed`` and ``survived`` are the only two that count toward the rate.
STATUSES = ("killed", "survived", "not_applicable", "error")


@dataclass(frozen=True, slots=True)
class MutantResult:
    """What happened to one mutant."""

    id: str
    title: str
    status: str
    detail: str = ""
    run: SuiteRun | None = None
    expected_killer: str = ""

    @property
    def counts(self) -> bool:
        """Whether this result contributes to the published kill rate."""
        return self.status in ("killed", "survived")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "expected_killer": self.expected_killer,
            "run": self.run.to_dict() if self.run else None,
        }


@dataclass(frozen=True, slots=True)
class MutationReport:
    """The headline metric, plus everything needed to disbelieve it."""

    results: tuple[MutantResult, ...]
    baseline: SuiteRun | None = None
    canary: MutantResult | None = None
    aborted: str = ""
    started_at: float = field(default_factory=time.time)
    duration_s: float = 0.0

    @property
    def scored(self) -> tuple[MutantResult, ...]:
        return tuple(r for r in self.results if r.counts)

    @property
    def killed(self) -> tuple[MutantResult, ...]:
        return tuple(r for r in self.results if r.status == "killed")

    @property
    def survivors(self) -> tuple[MutantResult, ...]:
        return tuple(r for r in self.results if r.status == "survived")

    @property
    def unscorable(self) -> tuple[MutantResult, ...]:
        return tuple(r for r in self.results if not r.counts)

    @property
    def kill_rate(self) -> float:
        """Killed over *scored*, and 0.0 when nothing could be scored.

        The denominator is deliberately the mutants that actually ran, not
        fifteen. Dividing by fifteen when three could not be applied would report
        a low rate as a test-quality problem when it is a harness problem, and
        the two need different fixes. The unscorable ones are listed separately
        and loudly.
        """
        scored = self.scored
        if not scored:
            return 0.0
        return len(self.killed) / len(scored)

    @property
    def valid(self) -> bool:
        """Whether the number above means anything.

        False when the baseline was not green (every mutant is trivially
        killed), when the canary survived (the mutated copy is not what ran), or
        when any mutant could not be applied or errored.
        """
        return (
            not self.aborted
            and self.baseline is not None
            and self.baseline.green
            and self.canary is not None
            and self.canary.status == "killed"
            and not self.unscorable
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "killed": len(self.killed),
            "scored": len(self.scored),
            "total": len(self.results),
            "kill_rate": round(self.kill_rate, 4),
            "valid": self.valid,
            "aborted": self.aborted,
            "duration_s": round(self.duration_s, 2),
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "canary": self.canary.to_dict() if self.canary else None,
            "survivors": [r.id for r in self.survivors],
            "unscorable": [r.id for r in self.unscorable],
            "results": [r.to_dict() for r in self.results],
        }


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def _classify(run: SuiteRun, mutation: Mutation) -> MutantResult:
    """Turn one suite run into a verdict.

    A collection error is **not** a kill. A mutant that stops the suite from
    importing has been detected by the interpreter, not by a test, and counting
    that would let a harness inflate its number by writing mutants that do not
    compile in spirit.
    """
    if run.timed_out:
        return MutantResult(
            mutation.id, mutation.title, "error",
            detail=f"suite timed out after {run.duration_s:.0f}s",
            run=run, expected_killer=mutation.expected_killer,
        )
    if not run.collected_anything:
        return MutantResult(
            mutation.id, mutation.title, "error",
            detail=(
                "no tests ran under this mutant (collection error): the "
                "interpreter noticed it, which is not the same as the suite "
                "noticing it"
            ),
            run=run, expected_killer=mutation.expected_killer,
        )
    if run.returncode != 0:
        where = f" (first failure: {run.first_failure})" if run.first_failure else ""
        return MutantResult(
            mutation.id, mutation.title, "killed",
            detail=f"{run.failed} failed, {run.passed} passed{where}",
            run=run, expected_killer=mutation.expected_killer,
        )
    return MutantResult(
        mutation.id, mutation.title, "survived",
        detail=f"{run.passed} passed, {run.failed} failed — the suite did not notice",
        run=run, expected_killer=mutation.expected_killer,
    )


def run_mutation(
    mutation: Mutation,
    *,
    repo_root: Path = REPO_ROOT,
    python: str = sys.executable,
    pytest_args: Sequence[str] = (),
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    exitfirst: bool = True,
    keep: bool = False,
) -> MutantResult:
    """Apply one mutation to a scratch copy and run the suite against it."""
    workdir = Path(tempfile.mkdtemp(prefix=f"sightline-mut-{mutation.id.lower()}-"))
    try:
        try:
            materialise(mutation, workdir, repo_root=repo_root)
        except LookupError as exc:
            return MutantResult(
                mutation.id, mutation.title, "not_applicable",
                detail=str(exc), expected_killer=mutation.expected_killer,
            )
        except SyntaxError as exc:
            return MutantResult(
                mutation.id, mutation.title, "error",
                detail=f"mutated source does not parse: {exc}",
                expected_killer=mutation.expected_killer,
            )
        run = run_suite(
            workdir, python=python, pytest_args=pytest_args,
            timeout=timeout, exitfirst=exitfirst,
        )
        return _classify(run, mutation)
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)


def run_all(
    mutations: Sequence[Mutation] = MUTATIONS,
    *,
    repo_root: Path = REPO_ROOT,
    python: str = sys.executable,
    pytest_args: Sequence[str] = (),
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    exitfirst: bool = True,
    skip_preflight: bool = False,
    progress: Any = None,
) -> MutationReport:
    """Baseline, canary, then every mutant. Returns the report; asserts nothing.

    Args:
        mutations: Which mutants to run. Defaults to the pre-registered fifteen.
        repo_root: Tree to copy. Never modified.
        python: Interpreter for the child pytest.
        pytest_args: Extra arguments, e.g. a single test file while iterating.
        timeout: Per-run wall clock budget.
        exitfirst: Stop each run at the first failure. Faster, and the first
            failing node id is the evidence that matters for a kill.
        skip_preflight: Skip the baseline and canary. For development only; a
            report produced this way has ``valid=False`` and says why.
        progress: Called with one status line per mutant. ``None`` is silent.

    Returns:
        A :class:`MutationReport`. Check ``valid`` before believing ``kill_rate``.
    """
    started = time.perf_counter()

    def say(line: str) -> None:
        if progress is not None:
            progress(line)

    baseline: SuiteRun | None = None
    canary: MutantResult | None = None
    aborted = ""

    if not skip_preflight:
        say("baseline: running the unmutated suite")
        workdir = Path(tempfile.mkdtemp(prefix="sightline-mut-baseline-"))
        try:
            shutil.copytree(repo_root, workdir, ignore=_ignore, dirs_exist_ok=True)
            baseline = run_suite(
                workdir, python=python, pytest_args=pytest_args,
                timeout=timeout, exitfirst=False,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        say(
            f"baseline: {baseline.passed} passed, {baseline.failed} failed "
            f"({baseline.duration_s:.1f}s)"
        )
        if not baseline.collected_anything:
            aborted = (
                "the suite collected no tests, so every mutant would be scored "
                "against nothing. There is no kill rate to report."
            )
        elif not baseline.green:
            aborted = (
                f"the unmutated suite is not green ({baseline.failed} failed, "
                f"{baseline.errors} errors). Every mutant would be trivially "
                "'killed' by a failure that has nothing to do with it."
            )
        if aborted:
            return MutationReport(
                results=(), baseline=baseline, canary=None, aborted=aborted,
                duration_s=time.perf_counter() - started,
            )

        say("canary: check() allows everything; the suite must notice")
        canary = run_mutation(
            CANARY, repo_root=repo_root, python=python, pytest_args=pytest_args,
            timeout=timeout, exitfirst=exitfirst,
        )
        say(f"canary: {canary.status}")
        if canary.status != "killed":
            return MutationReport(
                results=(), baseline=baseline, canary=canary,
                aborted=(
                    "the canary survived. The scratch copy is not the code under "
                    "test — most likely a conftest that puts the working tree "
                    "back on sys.path. Every 'survived' below this would be an "
                    "artefact, so nothing is reported."
                ),
                duration_s=time.perf_counter() - started,
            )

    results: list[MutantResult] = []
    for mutation in mutations:
        say(f"{mutation.id}: {mutation.title}")
        result = run_mutation(
            mutation, repo_root=repo_root, python=python, pytest_args=pytest_args,
            timeout=timeout, exitfirst=exitfirst,
        )
        say(f"{mutation.id}: {result.status} — {result.detail}")
        results.append(result)

    return MutationReport(
        results=tuple(results), baseline=baseline, canary=canary,
        duration_s=time.perf_counter() - started,
    )


def verify_anchors(
    mutations: Sequence[Mutation] = (*MUTATIONS, CANARY), *, repo_root: Path = REPO_ROOT
) -> list[str]:
    """Check every edit still applies, without running any tests.

    Cheap enough to run as the first step of CI. A mutant whose anchor has moved
    is a mutant that is no longer testing anything, and finding that out in two
    seconds beats finding it out after a twenty-minute suite.
    """
    problems: list[str] = []
    for mutation in mutations:
        for edit in mutation.edits:
            target = repo_root / edit.path
            if not target.is_file():
                problems.append(f"{mutation.id}: {edit.path} does not exist")
                continue
            try:
                mutated = edit.apply(target.read_text(encoding="utf-8"))
                ast.parse(mutated, filename=str(target))
            except LookupError as exc:
                problems.append(f"{mutation.id}: {exc}")
            except SyntaxError as exc:
                problems.append(f"{mutation.id}: mutated source does not parse: {exc}")
    return problems


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_markdown(report: MutationReport) -> str:
    """The block :mod:`eval.report` writes into the README."""
    lines: list[str] = []
    if report.aborted:
        lines.append(f"**No kill rate to report.** {report.aborted}")
        lines.append("")
        return "\n".join(lines)

    killed = len(report.killed)
    scored = len(report.scored)
    lines.append(
        f"**{killed}/{scored} mutants killed** "
        f"({report.kill_rate:.0%}) in {report.duration_s:.0f}s."
    )
    if not report.valid:
        lines.append("")
        lines.append(
            "> This run is **not valid** as a published number: "
            + (
                "some mutants could not be applied or errored"
                if report.unscorable
                else "the pre-flight did not pass"
            )
            + ". See the table."
        )
    lines.append("")
    lines.append("| ID | Mutation | Result | Evidence |")
    lines.append("|---|---|---|---|")
    for result in report.results:
        mark = {
            "killed": "killed",
            "survived": "**SURVIVED**",
            "not_applicable": "not applicable",
            "error": "error",
        }[result.status]
        detail = result.detail.replace("|", "\\|")
        lines.append(f"| {result.id} | {result.title} | {mark} | {detail} |")
    lines.append("")
    if report.survivors:
        lines.append("**Survivors, named:**")
        lines.append("")
        for result in report.survivors:
            lines.append(
                f"* **{result.id} — {result.title}.** The suite was expected to "
                f"catch this with: {result.expected_killer}. It did not. "
                f"That behaviour is untested."
            )
        lines.append("")
    else:
        lines.append(
            "No survivors. That is the number the harness measured, not a target "
            "it was tuned to; the fifteen are specified in `docs/FRD.md` §8.2 and "
            "were written before the tests."
        )
        lines.append("")
    if report.baseline is not None:
        lines.append(
            f"Baseline (unmutated) suite: {report.baseline.passed} passed, "
            f"{report.baseline.skipped} skipped, {report.baseline.duration_s:.1f}s. "
            f"Canary (`check()` allows everything): {report.canary.status if report.canary else 'n/a'}."
        )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.mutation",
        description="Plant 15 permission bugs and measure how many the suite catches.",
    )
    parser.add_argument(
        "command", choices=("run", "list", "verify"), nargs="?", default="run"
    )
    parser.add_argument("--only", nargs="*", default=None, metavar="ID", help="e.g. --only M3 M4")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-exitfirst", action="store_true")
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="development only; the resulting report is marked invalid",
    )
    parser.add_argument("--pytest-arg", action="append", default=[], dest="pytest_args")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "list":
        for mutation in MUTATIONS:
            print(f"{mutation.id:>4}  {mutation.title}")
            print(f"      why: {mutation.rationale}")
            print(f"      killed by: {mutation.expected_killer}")
        return 0

    if args.command == "verify":
        problems = verify_anchors()
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            print(f"{len(problems)} mutation(s) no longer apply", file=sys.stderr)
            return 1
        print(f"all {len(MUTATIONS)} mutations (plus the canary) still apply")
        return 0

    selected = MUTATIONS
    if args.only:
        wanted = {mid.upper() for mid in args.only}
        selected = tuple(m for m in MUTATIONS if m.id.upper() in wanted)
        missing = wanted - {m.id.upper() for m in selected}
        if missing:
            parser.error(f"unknown mutant id(s): {', '.join(sorted(missing))}")

    report = run_all(
        selected,
        python=args.python,
        pytest_args=args.pytest_args,
        timeout=args.timeout,
        exitfirst=not args.no_exitfirst,
        skip_preflight=args.skip_preflight,
        progress=None if args.quiet or args.json else lambda line: print(line, file=sys.stderr),
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return 0 if report.valid and not report.survivors else 1


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
