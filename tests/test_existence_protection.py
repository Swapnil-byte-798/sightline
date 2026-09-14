"""\"No such document\" and \"not for you\" must be the same answer.

If a refusal tells you *why* it refused, the refusal is an enumerator. Ask one
question per guessed project name, read the wording, and you have mapped the
corpus without ever being allowed to read a byte of it. That is a real
exfiltration technique against a system whose access control is otherwise
perfect, and it is planted mutant M11.

So the two situations must be indistinguishable from outside:

* the document does not exist, and
* the document exists and the principal may not see it.

Indistinguishable does not mean undistinguished. The audit log records which one
happened, because the operator investigating a support ticket needs to know and
the person asking the question must not. That asymmetry — identical outside,
honest inside — is what the assertions below are shaped around.

Timing is a channel too. The permission-empty path returns in about a
millisecond and the searched-and-found-nothing path takes tens, so both are held
to a constant floor. The floor is asserted; the *difference* is not, because a
wall-clock equality assertion on shared CI hardware is a test that lies.
"""

from __future__ import annotations

import dataclasses
import time

import pytest

from sightline.store.memory import MemoryVectorStore
from sightline.types import Chunk, ObjectRef, PrincipalRef, RefusalReason

authz = pytest.importorskip("sightline.authz", reason="sightline.authz is not present yet")
retrieve = pytest.importorskip("sightline.retrieve", reason="sightline.retrieve is not present yet")
embed = pytest.importorskip(
    "sightline.ingest.embed", reason="sightline.ingest.embed is not present yet"
)
audit_mod = pytest.importorskip("sightline.audit", reason="sightline.audit is not present yet")
settings_mod = pytest.importorskip("sightline.settings")

DIM = 64
FLOOR_MS = 20.0

#: Indexed, and secret. carol may not see it; the question mentions it by name.
SECRET_TEXT = "Project Atlas: the merger with Northwind closes on the fourteenth."
#: A question about a document that does not exist anywhere, in any namespace.
NONEXISTENT_QUESTION = "what does the flibbertigibbet quarterly ledger say"
SECRET_QUESTION = "what does the Project Atlas merger memo say"


@pytest.fixture
def settings():
    base = settings_mod.Settings()
    return base.with_(
        retrieval=dataclasses.replace(base.retrieval, refusal_floor_ms=FLOOR_MS, default_k=5)
    )


@pytest.fixture
def retriever(world, settings):
    """A full pipeline over a corpus that holds exactly one secret document.

    carol's own permitted document (``doc:payroll``, through ``folder:hr``) is
    deliberately **not** indexed, so both of her questions end in a refusal. That
    is the only way to compare the two refusals against each other rather than
    against an answer.
    """
    store = MemoryVectorStore(dim=DIM)
    embedder = embed.HashEmbedder(dim=DIM, quiet=True)
    obj = ObjectRef("doc", "merger")
    chunk = Chunk(
        id="merger#0",
        object=obj,
        text=SECRET_TEXT,
        grant_tokens=authz.grant_tokens_for_object(world, obj),
    )
    store.upsert([chunk], [embedder.encode([SECRET_TEXT])[0]])
    return retrieve.Retriever(
        tuple_store=world,
        vector_store=store,
        embedder=embedder,
        audit=audit_mod.MemoryAuditLog(),
        settings=settings,
    )


def _public(answer) -> dict:
    """Everything a caller can see. Nothing that varies per request.

    ``request_id`` and the timings are excluded because they differ between any
    two requests; if they were the channel, every response would leak.
    """
    return {
        "text": answer.text,
        "citations": [dataclasses.asdict(c) for c in answer.citations],
        "refused": answer.refused,
        "refusal_reason": answer.refusal_reason,
        "existence_protected": answer.existence_protected,
        "strategy": answer.strategy,
        "epoch": answer.epoch,
    }


def _ask(retriever, question: str, who: str):
    started = time.perf_counter()
    result = retriever.ask(question, PrincipalRef.parse(who))
    return result, (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------
# The property
# --------------------------------------------------------------------------


def test_forbidden_and_nonexistent_are_byte_identical(retriever):
    """The headline of this file. One question names a real secret; the other
    names nothing at all. The two replies must be the same object."""
    secret, _ = _ask(retriever, SECRET_QUESTION, "user:carol")
    missing, _ = _ask(retriever, NONEXISTENT_QUESTION, "user:carol")

    assert secret.answer.refused is True
    assert _public(secret.answer) == _public(missing.answer)


def test_a_principal_with_no_permissions_gets_the_same_reply(retriever):
    """dave holds nothing and short-circuits before search; carol's query runs the
    whole pipeline. Two entirely different code paths, one identical answer."""
    dave, _ = _ask(retriever, SECRET_QUESTION, "user:dave")
    carol, _ = _ask(retriever, SECRET_QUESTION, "user:carol")
    assert _public(dave.answer) == _public(carol.answer)


def test_the_refusal_never_carries_the_evidence(retriever):
    """No chunk text, no document id, no group name (M12).

    The realistic way document content escapes is not the answer, it is the
    debugging: somebody attaches the evidence that failed to ground because an
    ungrounded refusal is hard to diagnose without it.
    """
    result, _ = _ask(retriever, SECRET_QUESTION, "user:carol")
    text = result.answer.text
    assert "Atlas" not in text and "Northwind" not in text
    assert "merger" not in text.lower()
    assert result.answer.citations == ()
    for word in ("legal", "group:", "doc:"):
        assert word not in text


def test_the_audit_log_knows_the_difference_even_though_the_caller_does_not(retriever):
    """Identical outside, honest inside.

    ``NO_PERMITTED_EVIDENCE`` means the plan admitted nothing or the live check
    dropped everything; ``NO_EVIDENCE_AT_ALL`` means the index had nothing to
    offer. The operator needs that distinction and the caller must not have it.
    """
    _ask(retriever, SECRET_QUESTION, "user:dave")
    _ask(retriever, SECRET_QUESTION, "user:carol")
    reasons = [row.row["refusal_reason"] for row in retriever.audit.read(limit=10)]
    assert reasons == [
        RefusalReason.NO_PERMITTED_EVIDENCE.value,
        RefusalReason.NO_EVIDENCE_AT_ALL.value,
    ], reasons


def test_search_refuses_the_same_way_as_ask(retriever):
    """``/v1/search`` returns no generated text, and therefore no new channel."""
    secret = retriever.search(SECRET_QUESTION, PrincipalRef("user", "carol"))
    missing = retriever.search(NONEXISTENT_QUESTION, PrincipalRef("user", "carol"))
    for result in (secret, missing):
        assert result.hits == ()
        assert result.refused is True
        assert result.existence_protected is True
    assert secret.refusal_reason == missing.refusal_reason
    assert secret.strategy == missing.strategy


def test_both_refusal_paths_are_held_to_the_floor(retriever):
    """The timing channel is narrowed, and the narrowing is asserted.

    Only the floor is checked. The difference between the two paths is a
    wall-clock measurement on hardware we do not control, and asserting it would
    make this file fail for reasons that have nothing to do with disclosure. The
    honest statement is in the README: the floor narrows the channel, it does not
    close it, and a patient attacker with thousands of timed probes still wins.
    """
    _, fast_ms = _ask(retriever, SECRET_QUESTION, "user:dave")
    _, slow_ms = _ask(retriever, SECRET_QUESTION, "user:carol")
    assert fast_ms >= FLOOR_MS * 0.9, f"the empty-plan path returned in {fast_ms:.1f} ms"
    assert slow_ms >= FLOOR_MS * 0.9


# --------------------------------------------------------------------------
# The same property one layer down
# --------------------------------------------------------------------------


def test_the_authority_itself_does_not_distinguish(world):
    """``check()`` answers "no" identically for absent and forbidden.

    Worth asserting separately: if the evaluator ever raised ``NotFound`` for an
    unknown object, every layer above it would have to remember to catch it, and
    one of them would not.
    """
    forbidden = authz.check(
        world, ObjectRef("doc", "merger"), "viewer", PrincipalRef("user", "carol")
    )
    absent = authz.check(
        world, ObjectRef("doc", "no-such-thing"), "viewer", PrincipalRef("user", "carol")
    )
    assert forbidden.allowed is absent.allowed is False


def test_the_plan_does_not_reveal_the_corpus(world):
    """A compiled plan names what you may see, never what exists.

    ``/v1/plan`` is a debugging endpoint, and the reason it can be exposed at all
    is that a plan for a principal with no access is empty rather than a list of
    everything they were denied.
    """
    plan = authz.compile_plan(world, PrincipalRef("user", "dave"))
    blob = repr(plan)
    assert "merger" not in blob and "handbook" not in blob
