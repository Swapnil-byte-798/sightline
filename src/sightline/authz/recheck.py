"""``recheck()`` — the last line of defence, and the shortest file in the repo.

The index is a hint. This is where the hint meets the authority. Every candidate
that came out of a vector store is re-checked against the live tuple store, and
only survivors become :class:`~sightline.store.base.Hit`, which is the only type
the answer builder accepts. There is no flag that turns this off, no fast path
that skips it, and no cache in front of it.

This file is deliberately boring. It is short enough to read in one sitting and
dull enough that a reviewer can be sure of it, because the alternative — a
clever recheck — is a recheck nobody audits.

Three things it does not do, each of them a named mutant:

* **It does not trust ``matched_token``** (M4). The token the index matched is
  advisory. It was written at ingest, it may name a group that has since been
  deleted, and re-deriving the answer from live tuples costs one graph walk.
* **It does not process a prefix of the hits** (M10). Every hit is checked.
  Batching here means "one pass that de-duplicates by object", not "the first
  page".
* **It does not widen anything.** A hit whose grant was removed since indexing
  is dropped, and the drop is counted and reported, because a silent drop is
  indistinguishable from a retrieval bug.
"""

from __future__ import annotations

from collections.abc import Sequence

from sightline.authz.check import MAX_USERSET_DEPTH, check
from sightline.authz.compile import DEFAULT_RELATION
from sightline.authz.tuples import TupleStore
from sightline.store.base import Hit, UncheckedHit
from sightline.types import Decision, ObjectRef, PrincipalRef, Relation

__all__ = ["LiveRechecker", "recheck"]


def recheck(
    store: TupleStore,
    principal_ref: str,
    hits: Sequence[UncheckedHit],
    *,
    relation: Relation = DEFAULT_RELATION,
    max_depth: int = MAX_USERSET_DEPTH,
) -> tuple[list[Hit], int]:
    """Turn index output into servable results. Returns ``(survivors, dropped)``.

    One authorisation check per distinct object, not per hit: a document that
    contributed eight chunks is one question, asked once. The de-duplication is
    the "batching" the latency budget refers to, and it is safe because the two
    chunks of one document have exactly the same answer — the permission is on
    the document.

    The epoch is read once, *before* any check, and stamped on every survivor. A
    write that lands mid-recheck therefore leaves hits labelled with the older
    epoch, which makes them look staler than they are. The caller's staleness
    logic then errs toward recompiling, which is the direction that costs
    latency instead of secrets.
    """
    epoch = store.epoch()
    principal = PrincipalRef.parse(principal_ref)

    decisions: dict[str, Decision | None] = {}
    survivors: list[Hit] = []
    dropped = 0

    for hit in hits:
        if hit.object_ref not in decisions:
            decisions[hit.object_ref] = _decide(
                store, hit.object_ref, relation, principal, max_depth
            )
        decision = decisions[hit.object_ref]
        if decision is None or not decision.allowed:
            # Includes the unparseable-object-ref case. An index entry we cannot
            # even name is not an entry we serve.
            dropped += 1
            continue
        survivors.append(
            Hit(
                chunk_id=hit.chunk_id,
                score=hit.score,
                text=hit.text,
                object_ref=hit.object_ref,
                why_allowed=" -> ".join(decision.why) if decision.why else str(decision.allowed),
                checked_at_epoch=epoch,
            )
        )
    return survivors, dropped


def _decide(
    store: TupleStore,
    object_ref: str,
    relation: Relation,
    principal: PrincipalRef,
    max_depth: int,
) -> Decision | None:
    """``None`` when the reference is malformed. Malformed means denied."""
    try:
        obj = ObjectRef.parse(object_ref)
    except ValueError:
        return None
    return check(store, obj, relation, principal, max_depth=max_depth)


class LiveRechecker:
    """:class:`~sightline.store.base.Rechecker` bound to a live tuple store.

    The query pipeline holds one of these. It exists so the pipeline depends on
    the protocol rather than on this module's import path — not so that recheck
    becomes swappable. There is no implementation of this protocol that does
    less work, and adding one would be a code review that ends in a no.
    """

    def __init__(
        self,
        store: TupleStore,
        *,
        relation: Relation = DEFAULT_RELATION,
        max_depth: int = MAX_USERSET_DEPTH,
    ) -> None:
        self.store = store
        self.relation = relation
        self.max_depth = max_depth

    def recheck(
        self, principal_ref: str, hits: Sequence[UncheckedHit]
    ) -> tuple[list[Hit], int]:
        return recheck(
            self.store,
            principal_ref,
            hits,
            relation=self.relation,
            max_depth=self.max_depth,
        )
