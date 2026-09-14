# ADR 0003 — An empty permission set means deny everything

**Status:** accepted
**Date:** 2026-09-14

## Context

When a principal's compiled grant-token set comes back empty, there are two
possible readings: "this person can see nothing" or "no filter was specified, so
do not filter". The second reading is how permission systems leak, and it is an
easy accident — an empty list is falsy in most languages, and `if tokens:` quietly
becomes "skip the filter".

## Decision

Empty means **deny everything**. Always. A principal with no grant tokens gets
zero results, never all results.

This is asserted in `tests/test_fail_closed.py`, and it is one of the 15 bugs
planted deliberately by `eval/mutation.py` to check the test suite would catch it.

## Consequences

**Good.** The most dangerous single failure mode in the system is covered by a
test that runs on every commit, and by a mutation that proves the test works.

**Bad.** A misconfigured principal sees nothing and gets no explanation beyond a
refusal. The admin console exists partly to make that diagnosable.

## Note

We report false-allow and false-deny asymmetrically on purpose. A false allow is a
breach and its target is exactly zero, enforced as a blocking CI gate. A false
deny is a quality bug, reported as a rate with a target below 0.1%. Publishing
that asymmetry is part of the argument, not a footnote.
