"""Tamper with any row and ``verify_chain()`` must say so.

The audit log is a hash chain: every row commits to its predecessor's digest, so
changing one row invalidates every row after it. That does not stop an attacker
who can rewrite the whole file — nothing in a single process can — and the README
says so plainly. What it does is make *quiet* edits impossible: an operator who
changes one line to hide one query has to change every line since, and the head
digest still will not match whatever was published elsewhere.

The tests below are the full cross product of things a tamperer would try: edit
a field, edit a digest, reorder, delete from the middle, delete from the end,
and append a plausible row. Each of them has to be caught, and the report has to
name the row, because ``GET /v1/audit`` renders the break rather than raising.

Also asserted here: what the log does **not** contain. The question, the answer,
and the chunk text are absent by construction — the log records which policy
state served whom by which strategy, and not enough to reconstruct what anybody
asked.
"""

from __future__ import annotations

import dataclasses
import json
from itertools import pairwise

import pytest

audit = pytest.importorskip("sightline.audit", reason="sightline.audit is not present yet")

AuditRecord = audit.AuditRecord
AuditRow = audit.AuditRow
GENESIS_DIGEST = audit.GENESIS_DIGEST
MemoryAuditLog = audit.MemoryAuditLog
canonical_json = audit.canonical_json
chain_digest = audit.chain_digest
hash_query = audit.hash_query
redact = audit.redact
verify_chain = audit.verify_chain


def _record(i: int, **overrides) -> AuditRecord:
    base = dict(
        principal=f"user:u{i}",
        query_hash=hash_query(f"question {i}", key=b"test-key"),
        strategy="grant_tokens",
        epoch=i,
        n_candidates=10,
        n_dropped_at_recheck=i % 3,
        retrieved_objects=(f"doc:{i}",),
        shown_objects=(f"doc:{i}",),
        request_id=f"req{i}",
        ts=1_700_000_000.0 + i,
    )
    base.update(overrides)
    return AuditRecord(**base)


@pytest.fixture
def log():
    sink = MemoryAuditLog()
    for i in range(8):
        sink.append(_record(i))
    return sink


def _rows(log) -> list[AuditRow]:
    return log.read(limit=1000)


# --------------------------------------------------------------------------
# An untampered chain verifies
# --------------------------------------------------------------------------


def test_a_clean_chain_verifies(log):
    result = log.verify()
    assert result.ok is True
    assert result.checked == 8
    assert result.head == log.head
    assert result.broken_at is None


def test_the_first_row_commits_to_genesis(log):
    first = _rows(log)[0]
    assert first.seq == 0
    assert first.prev == GENESIS_DIGEST
    assert first.digest == chain_digest(GENESIS_DIGEST, first.row)


def test_every_row_commits_to_its_predecessor(log):
    rows = _rows(log)
    for previous, row in pairwise(rows):
        assert row.prev == previous.digest
        assert row.digest == chain_digest(previous.digest, row.row)


def test_a_truncated_tail_still_verifies(log):
    """Cutting the end produces a shorter valid chain, and that is not a break.

    Worth stating: the chain proves *internal consistency*, not completeness.
    Detecting a truncated tail needs the head digest published somewhere the
    operator does not control, which is an integration this repository does not
    have and does not pretend to.
    """
    assert verify_chain(_rows(log)[:5]).ok is True


# --------------------------------------------------------------------------
# Tampering
# --------------------------------------------------------------------------


def test_editing_a_field_breaks_the_chain(log):
    """The classic: quietly change who asked."""
    rows = _rows(log)
    rows[3] = dataclasses.replace(rows[3], row={**rows[3].row, "principal": "user:someone-else"})
    result = verify_chain(rows)
    assert result.ok is False
    assert result.broken_at == 3
    assert "digest" in result.reason


def test_editing_a_nested_value_breaks_the_chain(log):
    """Including the lists, which a shallow digest over scalars would miss."""
    rows = _rows(log)
    rows[2] = dataclasses.replace(
        rows[2], row={**rows[2].row, "shown_objects": ["doc:2", "doc:smuggled"]}
    )
    assert verify_chain(rows).broken_at == 2


def test_recomputing_the_digest_does_not_repair_it(log):
    """A tamperer who fixes the row's own digest still breaks the next link.

    This is the property that makes the chain worth having. Repairing one row
    requires repairing every row after it.
    """
    rows = _rows(log)
    edited = {**rows[3].row, "epoch": 9_999}
    rows[3] = dataclasses.replace(
        rows[3], row=edited, digest=chain_digest(rows[3].prev, edited)
    )
    result = verify_chain(rows)
    assert result.ok is False
    assert result.broken_at == 4
    assert "predecessor" in result.reason


def test_deleting_a_row_from_the_middle_is_caught(log):
    rows = _rows(log)
    del rows[4]
    result = verify_chain(rows)
    assert result.ok is False
    assert result.broken_at == 5
    assert "sequence" in result.reason


def test_reordering_rows_is_caught(log):
    rows = _rows(log)
    rows[2], rows[5] = rows[5], rows[2]
    assert verify_chain(rows).ok is False


def test_appending_a_forged_row_is_caught(log):
    """A row invented after the fact does not know the real head."""
    rows = _rows(log)
    forged_row = _record(99).to_row()
    rows.append(AuditRow(seq=8, prev=GENESIS_DIGEST, digest="0" * 64, row=forged_row))
    result = verify_chain(rows)
    assert result.ok is False
    assert result.broken_at == 8


@pytest.mark.parametrize("index", range(8))
def test_every_single_row_is_load_bearing(log, index):
    """Tamper with each row in turn. None of them is unprotected."""
    rows = _rows(log)
    rows[index] = dataclasses.replace(
        rows[index], row={**rows[index].row, "n_candidates": 4_242}
    )
    assert verify_chain(rows).ok is False


# --------------------------------------------------------------------------
# The digest itself
# --------------------------------------------------------------------------


def test_canonical_json_is_order_independent():
    """Determinism is the whole security property.

    Two processes that serialise the same row differently produce different
    digests and a chain that looks broken, so nothing here may depend on dict
    order, locale, or float formatting.
    """
    a = {"b": 1, "a": [1, 2], "c": {"z": 0, "y": 1}}
    b = {"c": {"y": 1, "z": 0}, "a": [1, 2], "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_escapes_non_ascii():
    """A row containing a non-ASCII document title must hash identically on a
    machine with a different default encoding."""
    blob = canonical_json({"title": "prêt-à-porter"})
    assert blob.decode("ascii")  # would raise if any raw UTF-8 got through


def test_the_separator_closes_the_concatenation_ambiguity():
    """``prev="ab"`` plus a row must not collide with ``prev="a"`` plus another.

    A one-byte delimiter for a length-extension-shaped ambiguity. It costs
    nothing and it is the kind of thing that is impossible to add later.
    """
    assert chain_digest("ab", {"x": "c"}) != chain_digest("a", {"x": "bc"})


def test_the_digest_covers_every_field_of_the_row():
    """Adding a field to ``AuditRecord`` without adding it to ``to_row`` would
    put unhashed data in the log. The row's key set is asserted so that change
    is a deliberate edit rather than a silent one."""
    row = _record(1).to_row()
    assert set(row) == {
        "ts",
        "route",
        "request_id",
        "principal",
        "query_hash",
        "strategy",
        "epoch",
        "n_candidates",
        "n_dropped_at_recheck",
        "retrieved_objects",
        "shown_objects",
        "guardrails_fired",
        "refusal_reason",
        "provider",
        "degraded",
    }


# --------------------------------------------------------------------------
# What the log must not contain
# --------------------------------------------------------------------------


def test_the_question_is_hashed_not_stored():
    record = _record(1, query_hash=hash_query("what is our parental leave policy", key=b"k"))
    blob = json.dumps(record.to_row())
    assert "parental" not in blob
    assert record.query_hash.startswith("k:")


def test_the_query_hash_is_keyed_so_it_cannot_be_dictionary_attacked():
    """An unkeyed hash of a plausible question is recoverable by anyone willing to
    hash a list of plausible questions."""
    assert hash_query("same question", key=b"key-a") != hash_query("same question", key=b"key-b")
    assert hash_query("same question", key=b"key-a") == hash_query("Same  Question", key=b"key-a")
    assert hash_query("q").startswith("u:"), "the unkeyed form must be marked as such"


def test_identifiers_that_are_really_email_addresses_are_redacted():
    """The Enron corpus produces principals shaped exactly like this."""
    row = _record(1, principal="user:kenneth.lay@enron.com").redacted().to_row()
    assert "enron.com" not in row["principal"]
    assert "[REDACTED:EMAIL]" in row["principal"]


def test_redaction_happens_before_the_digest():
    """Otherwise the digest commits to the unredacted value and the log's own
    integrity proof becomes a copy of what was supposed to be removed."""
    sink = MemoryAuditLog()
    row = sink.append(_record(1, principal="user:a.person@example.com"))
    assert "example.com" not in json.dumps(row.row)
    assert row.digest == chain_digest(GENESIS_DIGEST, row.row)
    assert sink.verify().ok is True


# --------------------------------------------------------------------------
# The durable sink
# --------------------------------------------------------------------------


def test_jsonl_sink_round_trips_and_verifies(tmp_path):
    path = tmp_path / "audit.jsonl"
    sink = audit.JsonlAuditLog(path)
    for i in range(5):
        sink.append(_record(i))
    assert sink.verify().ok is True
    assert len(sink.read(limit=100)) == 5

    # A second sink over the same file continues the chain rather than restarting.
    reopened = audit.JsonlAuditLog(path)
    assert reopened.head == sink.head
    reopened.append(_record(5))
    assert reopened.verify().ok is True
    assert len(reopened) == 6


def test_editing_the_file_on_disk_is_caught(tmp_path):
    """The realistic tamper: someone opens the log in an editor."""
    path = tmp_path / "audit.jsonl"
    sink = audit.JsonlAuditLog(path)
    for i in range(5):
        sink.append(_record(i))

    lines = path.read_text(encoding="utf-8").splitlines()
    blob = json.loads(lines[2])
    blob["row"]["principal"] = "user:nobody"
    lines[2] = json.dumps(blob, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = audit.JsonlAuditLog(path).verify()
    assert result.ok is False
    assert result.broken_at == 2
