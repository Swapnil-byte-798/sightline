# Sightline

**Internal AI search that can only see what you are allowed to see — and proves it.**

Companies buy an AI assistant over their own documents, then freeze the rollout.
Not because the answers are bad. Because they are good.

A shared folder someone set to "everyone in the company" in 2019 was harmless
while finding it meant knowing the URL. The moment a search engine indexes it,
that folder is one plain-English question away from every employee. So the
security team stops the rollout, and the company pays for a permissions audit
before it will switch anything on.

Sightline is the search service built the other way round. Permissions are not a
filter bolted on after retrieval. They are compiled into the search itself, so a
document you cannot open is never a candidate, never enters the model's context,
and can never be cited.

---

## The rule the whole system rests on

> **The index is a hint. The database is the authority.**

A vector index stores a copy of permission state, so it is always slightly stale.
That staleness window is where leaks live. Sightline closes it by making a leak
unrepresentable rather than unlikely:

- `VectorStore.search()` returns `UncheckedHit`.
- The answer builder accepts only `Hit`.
- The only thing that turns one into the other is `recheck()`, against the live
  permission store.

So a stale index **loses** results. It cannot **leak** them. A document whose
permissions were tightened is dropped at recheck; one whose permissions were
loosened is missed until reindex. That asymmetry is deliberate: being briefly
unhelpful is recoverable, being briefly unsafe is not.

See [ADR 0001](docs/adr/0001-index-is-a-hint.md).

## Why permission filtering breaks vector search

Search returns a fixed-size top-`k` ranked by similarity over whatever it
searched. Filter afterwards and you do not get the best `k` results the user may
see — you get whatever scraps of the global top `k` happen to be permitted.

`examples/recall_collapse.py` measures it. Same 4,000 chunks, two stores:

```
 visible   pre-filter  post-filter    recall
    2.5%         10.0          0.2     0.020
    5.0%         10.0          0.4     0.043
   10.0%         10.0          0.9     0.092
   50.0%         10.0          5.0     0.497
  100.0%         10.0         10.0     1.000
```

A user who can see 2.5% of the corpus asks for ten results and gets **0.2**. The
assistant then says it does not know, about a document in the user's own folder.

Nothing logs an error, which is why this survives in production for months. The
index answered inside its latency budget, the filter removed exactly what it was
meant to, the model wrote an honest sentence, and the status code was 200. **No
component's contract was violated.**

```bash
make demo
```

## Architecture

```
  tuples (object#relation@principal)        Zanzibar-style, groups nest
            |
            v
  +---------------------+  compile   +----------------------------+
  |  Check()            | ---------> |  FilterPlan                |
  |  authoritative,     |            |  grant tokens + strategy   |
  |  recursive, audited |            |  (enumerate / exact scan / |
  +---------------------+            |   tokens / unfiltered)     |
            ^                        +-------------+--------------+
            |                                      |
            | recheck (live)                       v
            |                          +-----------------------+
  +---------+-----------+              |  vector store         |
  |  Hit  <-- recheck --<--------------|  UncheckedHit         |
  |  (only these reach  |              |  Qdrant / pgvector+RLS|
  |   the model)        |              |  memory / postfilter  |
  +---------------------+              +-----------------------+
```

Chunks are tagged with grant tokens derived from **groups**, never user ids. A
person joining or leaving a group rewrites zero vectors ([ADR 0002](docs/adr/0002-grant-tokens-not-user-ids.md)).
An empty permission set means deny everything ([ADR 0003](docs/adr/0003-fail-closed.md)).

## Vector stores

| Backend | Role | Why |
|---|---|---|
| **Qdrant** | serving | Filter-aware graph traversal; the filter prunes *during* the walk |
| **pgvector + RLS** | second enforcement layer | Postgres refuses cross-principal reads even if application code is wrong |
| **FAISS** | oracle | Exact search, never served — it produces the ground truth others are measured against |
| **memory** | reference | Pure numpy; the whole test suite runs on it with no services |
| **postfilter** | baseline | Implements the bug on purpose, so the collapse can be measured |

All five sit behind one ~40-line protocol, so the comparison is apples to apples.

## Quickstart

```bash
make venv
make test     # 46 tests, no external services, no network
make demo     # the recall collapse
make serve    # API on :8000
```

## Documentation

| | |
|---|---|
| [PRD](docs/PRD.md) | problem, buyer, goals, non-goals, competitive landscape |
| [FRD](docs/FRD.md) | numbered requirements, permission model, API, security |
| [Field guide](https://github.com/Swapnil-byte-798/sightline-field-guide) | 105 pages, from first principles, with interview questions |
| [ADRs](docs/adr/) | the three decisions everything else follows from |

## Status — what is and is not built

Honest inventory, because the alternative is a README that lies.

**Working and tested.** Domain types; Zanzibar-style tuple store (memory and
SQLite); recursive `Check()` with cycle detection and derivation paths; the
permission compiler; `recheck()`; the differential oracle; five vector stores;
ingest (chunking, ONNX embedding, Enron connector); guardrails (direct and
indirect prompt injection, PII, structural grounding); hash-chained audit log;
JWT auth; OpenTelemetry and Prometheus wiring; the HTTP API; the selectivity
evaluation harness. 46 tests pass with only `pydantic`, `fastapi`, `numpy`
and `httpx` installed.

**Not built yet.**

- `eval/mutation.py` — the headline metric. Plant 15 deliberate permission bugs,
  measure how many the test suite catches. Publishing 12/15 with the three
  survivors named beats claiming 15/15 without the harness.
- `eval/attack.py` — cross-principal ground truth, existence probing, revocation
  races, timing enumeration, mapped to the OWASP LLM Top 10.
- `eval/report.py` — machine-write every number in this README from a fresh run
  and fail CI when a committed number drifts.
- Deployment. Nothing is running at a public URL yet.
- Load testing. No capacity numbers exist, so none are claimed.
- `Dockerfile` and `docker-compose.yml`.

**No numbers in this README are invented.** The recall table is the real output
of `examples/recall_collapse.py`, which runs in CI on every commit. When
`eval/report.py` lands, the rest will be generated the same way.

## Licence

MIT
