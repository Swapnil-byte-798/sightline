# ADR 0002 — Tag chunks with groups, never with people

**Status:** accepted
**Date:** 2026-09-14

## Context

A query cannot carry a list of the 40,000 documents a person is allowed to see.
So the permission state has to be compressed into something a search index can
filter on cheaply.

The obvious approach is to stamp each chunk with the ids of everyone who can read
it. It works, and it is wrong: when one person joins a group of 500 documents,
500 vectors have to be rewritten. In a real company, group membership changes
constantly.

## Decision

Chunks are tagged with **grant tokens derived from groups**, never with user ids
and never with document id lists.

A principal is compiled into the set of grant tokens they hold, transitively
through nested groups. The index filters on token overlap.

## Consequences

**Good.** A person joining or leaving a group rewrites **zero** vectors — only
their compiled plan changes, and that is invalidated automatically by the policy
epoch. Only a change to a *document's own* permissions touches the index.

**Bad.** Someone belonging to very many groups has a large token set, which makes
the filter more expensive. We measure where that starts to hurt rather than
assuming a limit.

**Bad.** Grant tokens are coarser than per-document permissions, so a document
with a one-off exception needs its own token. We accept the extra token.

## Alternatives rejected

- *Stamp user ids on chunks.* Rewrite storms on every membership change.
- *Bloom filters over allowed document sets.* Considered and cut: set
  intersection is not a Bloom operation, and at realistic group counts the filter
  matches nearly everything, so it would have looked like it worked while doing
  nothing.
