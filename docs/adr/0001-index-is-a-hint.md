# ADR 0001 — The index is a hint; the database is the authority

**Status:** accepted
**Date:** 2026-09-14

## Context

A vector index has to store permission information alongside each chunk, otherwise
it cannot filter during search. That copy is denormalised, which means it is always
slightly behind the real permission data. The gap is small — seconds to minutes —
but it is never zero.

Every system that treats the index as authoritative therefore has a window in which
someone whose access was just revoked still retrieves documents. Most products
accept this quietly. It is the single most likely way this system could leak.

## Decision

The index is advisory. Results coming out of it are re-checked against the live
permission database before they reach the user or the language model.

We enforce this with types rather than with discipline:

- `VectorStore.search()` returns `UncheckedHit`.
- The answer builder accepts only `Hit`.
- The only function that produces a `Hit` is `authz.recheck.recheck()`, which
  consults the live tuple store.

There is no method on `UncheckedHit` that yields a `Hit`. Getting one requires
importing the recheck module, which makes the dangerous path visible in review.

## Consequences

**Good.** A stale index *loses* results; it cannot *leak* them. A document whose
permissions were tightened since indexing is dropped at recheck time. This is the
security argument for the whole product and it is stated in one sentence.

**Bad.** A document whose permissions were *loosened* is missed until the next
reindex. That is an availability cost, and we are choosing it deliberately: being
briefly unhelpful is recoverable, being briefly unsafe is not.

**Cost.** One extra permission check per candidate result, on the hot path. It is
measured in `eval/selectivity.py` and reported rather than hidden.

## Alternatives rejected

- *Trust the index and reindex aggressively.* Shrinks the window without closing
  it, and costs far more compute.
- *Check permissions only at render time in the UI.* The document has already
  entered the model's context by then, so it can be paraphrased into the answer
  without ever being cited.
