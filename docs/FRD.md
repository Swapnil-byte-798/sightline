# Sightline — Functional Requirements Document

Companion to `docs/PRD.md`. The PRD says why. This says what, precisely enough to
test. Where this document and the code disagree, the code is right and this
document is a bug.

Priority scale used throughout:

* **P0** — v1 does not ship without it. Most P0 items are the security argument.
* **P1** — v1 ships with it, but a defect here is a bug, not a breach.
* **P2** — planned, may slip past v1.

Terms are defined at first use. The short version, repeated because everything
below depends on it:

> **The index is a hint. The database is the authority.**
> Search returns `UncheckedHit`. The answer builder accepts only `Hit`. The only
> conversion is `recheck()`, which reads live permissions. Therefore a stale index
> loses results and never leaks them.

---

## 1. Permission model

### 1.1 The tuple

One permission fact is one tuple:

```
object#relation@principal
```

Read it as: *on this object, this relation is held by this principal.*

```
doc:42#viewer@user:alice                 alice may view doc 42
doc:42#viewer@group:legal#member         every member of group legal may view doc 42
folder:hr#parent@doc:42                  doc 42 sits inside folder hr
group:legal#member@group:paralegals#member   paralegals are members of legal
group:legal#member@user:bob              bob is a member of legal
```

The third form in that list is a **userset**: the principal is not a person, it is
"whoever holds relation `member` on `group:paralegals`". Usersets are the entire
mechanism for nested groups. There is no separate group table and no separate
nesting feature; nesting falls out of a userset pointing at another userset.

There is no `tenant_id` column and no `is_public` boolean. Both are shorthands that
stop being expressible the moment an organisation has nested groups, and both are
how permission bugs get shipped. A tenant is an object. "Public" is a tuple naming
a group that contains everyone.

### 1.2 Relations and rewrites

Each namespace declares its relations and how they derive from one another. v1
supports three rewrite forms and no more:

* **`this`** — direct tuples only.
* **`union`** — relation A is satisfied if any of its constituent relations are.
* **`tupleToUserset`** — inheritance through another relation. Example: `viewer`
  on a document is satisfied by `viewer` on its `parent` folder. This is how folder
  inheritance works.

Intersection and exclusion (`viewer AND NOT denied`) are **P2** and are not in v1.
They are named here because their absence is a design decision, not an oversight:
a half-implemented exclusion is a deny that silently does not deny.

### 1.3 Check semantics

`check(object, relation, principal) -> Decision`

* Returns `allowed=True` only if a derivation path exists from the tuple graph.
* Returns the derivation path in `Decision.why` as an ordered list of tuple
  strings. An access decision a human cannot audit is a liability.
* Default is deny. Absence of a tuple is denial, never "unknown", never "allow".
* Expansion is breadth-first with cycle detection over visited `(object, relation)`
  pairs. A cycle terminates that branch; it does not error and it does not grant.
* Expansion has a hard depth limit, `MAX_USERSET_DEPTH` (default 16). Reaching the
  limit **denies** and records `depth_limit_reached` in `why`. It never truncates
  toward allow. Truncating toward allow is planted mutant M5.

### 1.4 Revocation semantics — stated exactly

This is the part every competing product is vague about, so it is spelled out.

| Event | Effect on the authoritative store | Effect on the index | When the user stops seeing it | When the user starts seeing it |
|---|---|---|---|---|
| User removed from a group | Tuple deleted, epoch +1 | **Zero vector writes** | Next query. The compiled plan is stale by epoch and is recompiled; even if a stale plan were somehow served, recheck drops the hits | n/a |
| User added to a group | Tuple written, epoch +1 | **Zero vector writes** | n/a | Next query, after plan recompile |
| Document's own permissions tightened (a grant removed) | Tuple deleted, epoch +1 | Reindex of that document's chunks is queued | **Next query.** Recheck is authoritative and drops the chunk before the model sees it, regardless of index state | n/a |
| Document's own permissions loosened (a grant added) | Tuple written, epoch +1 | Reindex of that document's chunks is queued | n/a | When the reindex lands. Target ≤ 15 min p95. This is the deliberate availability cost |
| Group deleted | Tuples deleted, epoch +1 | Grant tokens derived from that group stop matching any plan; chunks keep dead tokens until their next reindex | Next query | n/a |
| Document deleted | Tuples deleted, epoch +1 | Chunk delete queued | Next query (recheck finds no grant) | n/a |

The asymmetry is the point. **Tightening is immediate. Loosening is eventual.**
Every staleness path in the system fails toward denial.

### 1.5 Grant tokens

A grant token is an opaque string stamped on a chunk at ingest, standing for one
group edge that grants access to that chunk's document.

* Derived as a keyed hash of the granting **userset** (e.g. `group:legal#member`)
  plus the relation. Keyed, so a token does not reveal a group name to anyone who
  can observe index payloads.
* **Never** derived from a user id. **Never** a list of document ids. Both make
  membership churn rewrite vectors, and expensive changes get batched into nightly
  jobs, which is how the staleness window gets wide.
* A chunk's token set is the set of group edges granting its document, including
  edges inherited from parent folders.
* At query time, the plan compiler produces the principal's token set. The index
  filter is a set-intersection test: does the chunk have **any** token the
  principal holds. Within a single chunk's tokens the test is OR (any grant
  suffices). Where a compiled plan carries multiple required conditions, they
  combine with AND. Confusing those two is planted mutant M2.
* An **empty** token set means the principal can see nothing, and must match zero
  chunks. Treating empty as "match everything" is planted mutant M6.
* The token that matched is advisory in the response. Recheck does not trust it.
  Trusting it is planted mutant M4.

### 1.6 Epoch

A single monotonic counter for the whole policy store.

* Incremented on **every** permission write, including writes that turn out to be
  no-ops. Cheap to over-increment, unsafe to under-increment.
* Stamped onto every `FilterPlan` at compile time.
* A plan whose epoch is below the live epoch is **stale**. A stale plan is never
  served: the system recompiles, or refuses with `STALE_POLICY`.
* Plan caches are keyed on `(principal, epoch)`. Keying on principal alone is
  planted mutant M8.
* Every `Answer` carries the epoch it was served under, so an incident can be
  reconstructed against the policy state at the time.

---

## 2. Functional requirements

### Authorisation core

**FR-1 — Tuple store**
*Priority: P0*
Store, delete, and query permission tuples. Query by object, by principal, and by
`(object, relation)`.
**Acceptance:** round-trip of all tuple forms including usersets; deleting a
non-existent tuple is a no-op that still increments the epoch; concurrent writes
produce strictly increasing epochs with no duplicates; an in-memory backend works
with core dependencies only.

**FR-2 — Check**
*Priority: P0*
`check(object, relation, principal)` returns an authoritative `Decision`.
**Acceptance:** direct grants allowed; grants via one, two, and three levels of
nested group allowed; folder inheritance via `tupleToUserset` allowed; unrelated
principal denied; cyclic group graph terminates and denies; depth beyond
`MAX_USERSET_DEPTH` denies and says so in `why`; `Decision.why` contains a
derivation path that a test can replay tuple by tuple; `checked_tuples` is
non-zero on any non-trivial check.

**FR-3 — Expand / explain**
*Priority: P0*
Return the full derivation tree for `(object, relation, principal)`, including the
failed branches when the answer is deny.
**Acceptance:** a denial explains which edges were tried; the explain output for an
allow, replayed as individual `check` calls, reproduces the allow; p95 ≤ 200 ms on
the reference corpus.

**FR-4 — Epoch counter**
*Priority: P0*
Monotonic counter incremented on every permission write.
**Acceptance:** never decreases; increments on no-op deletes; survives process
restart when backed by a durable store; `FilterPlan.is_stale()` is true for any
plan below the live epoch.

**FR-5 — Filter plan compiler**
*Priority: P0*
Compile one principal's permissions into a `FilterPlan` with a strategy chosen from
measured cardinality, not hardcoded.
**Acceptance:** small allowed set (< 512 objects) compiles to `ENUMERATE`; typical
set compiles to `GRANT_TOKENS`; a principal who can see ≥ 98% of the corpus
compiles to `UNFILTERED`; a very small set where exact scan beats approximate
search compiles to `EXACT_SCAN`; `estimated_cardinality` is within 20% of the true
count on the test corpus; the chosen strategy is recorded on every response.

**FR-6 — Plan / oracle agreement**
*Priority: P0*
For every principal and every object, the set a plan admits must equal the set
`check()` allows.
**Acceptance:** property-based test over randomly generated tuple graphs (nesting
depth 0–5, fan-out 1–20). A plan admitting an object that `check()` denies fails
the build and is classified as a **breach**, not a bug. A plan missing an object
`check()` allows is a recall failure and also fails, at a lower severity.

**FR-7 — Grant token derivation**
*Priority: P0*
Derive stable, opaque tokens from group edges.
**Acceptance:** identical for the same userset across runs with the same key;
different across different keys; contains no plaintext group name; changing group
**membership** produces no token change for any chunk.

### Retrieval

**FR-8 — Vector store interface**
*Priority: P0*
One `VectorStore` protocol; all backends implement it. `search()` returns
`UncheckedHit` and pushes the permission filter **into** the index.
**Acceptance:** every backend passes the same conformance suite; `upsert` is
idempotent on `chunk.id`; `count_matching(plan)` matches a brute-force count.

**FR-9 — Qdrant backend**
*Priority: P1 (P0 for the serving story)*
Serving path. Payload filter compiled from the plan.
**Acceptance:** import guarded — absent `qdrant-client` raises an error naming the
`qdrant` extra; filter is applied during graph traversal, not after; conformance
suite passes; skipped cleanly when the extra is not installed.

**FR-10 — pgvector backend with row-level security**
*Priority: P1*
Second, independent enforcement layer: even a bug in the application filter is
caught by a Postgres row-level security policy.
**Acceptance:** a test that deliberately sends an unfiltered query returns zero
forbidden rows because the database refuses them; import guarded, naming the
`postgres` extra.

**FR-11 — Exact oracle store**
*Priority: P0*
Brute-force index, **never served**, used to prove the others return the complete
correct answer under a filter.
**Acceptance:** with numpy only (no FAISS), produces exact top-k over the permitted
subset; used as ground truth for every recall figure; a test asserts it is not
reachable from the serving API.

**FR-12 — Post-filter baseline store**
*Priority: P0*
Retrieve top-k globally, then drop what the principal cannot see. Implemented
**deliberately** as the baseline arm, because demonstrating its recall collapse is
the point.
**Acceptance:** clearly marked as not-for-production in code and docs; produces the
recall-vs-permission-density curve in the evaluation; never padded back up to k
from the unfiltered pool (padding is planted mutant M7); an import-time guard
prevents it being wired into the serving app.

**FR-13 — Mandatory recheck**
*Priority: P0*
`recheck(principal_ref, hits) -> (list[Hit], int)` consults the live tuple store
and returns survivors plus the count dropped.
**Acceptance:** the only construction site of `Hit` in the package is inside
`recheck` — enforced by `tests/test_no_unchecked_construction.py`, which greps the
source; a hit whose grant was revoked after indexing is dropped; the drop count
appears in the response metadata and in metrics; recheck is a single batched call,
not one call per hit; there is no configuration flag that disables it.

**FR-14 — Honest recall guarantee**
*Priority: P0*
The system never asserts "filtered top-k equals exact top-k". HNSW is approximate;
that equality holds only when the query routes to brute force.
**Acceptance:** tests assert `recall@10 ≥ 0.95` against the exact oracle across
permission densities of 0.1%, 1%, 10%, 50%, 100%; the strict-equality assertion is
used only in the `EXACT_SCAN` path; a grep test fails the build if a strict
equality assertion appears in an approximate-path test.

### Ingest

**FR-15 — Chunking and stamping**
*Priority: P0*
Split documents into chunks and stamp each with its document's grant tokens.
**Acceptance:** chunk ids are stable across reruns on unchanged input; every chunk
carries at least one token or is rejected; a chunk with zero tokens is never
indexed as visible-to-all.

**FR-16 — Embedding via ONNX**
*Priority: P1*
Embed with `onnxruntime`. Never torch — the reference machine is macOS x86_64,
where PyTorch ships no wheels past 2.2.x.
**Acceptance:** import guarded, naming the `embed` extra; a deterministic hash
embedder is available with core dependencies only, so the full test suite runs
without any model download.

**FR-17 — Incremental reindex on ACL change**
*Priority: P1*
A document's permission change queues a reindex of that document's chunks only.
A membership change queues nothing.
**Acceptance:** measured vector writes for a membership change is exactly 0; for a
document ACL change, exactly the chunk count of that document; assertions are on
counters, not on log text.

### Answering

**FR-18 — Query pipeline**
*Priority: P0*
Compile plan → search → recheck → guardrails → synthesise → reconstruct citations.
**Acceptance:** the pipeline refuses if the plan is stale; the synthesiser receives
only rechecked text; citations are reconstructed from the rechecked set, never
parsed out of generated text (parsing them out is planted mutant M15).

**FR-19 — Grounded or refused**
*Priority: P0*
`Answer` is grounded or refused. There is no third state.
**Acceptance:** every refusal carries a `RefusalReason`; an answer with zero
citations and `refused=False` is impossible and a test asserts it.

**FR-20 — Existence protection**
*Priority: P0*
Do not distinguish "no such document" from "not for you". That distinction leaks
the corpus one question at a time.
**Acceptance:** for a principal denied everything, the refusal for a query matching
real documents is byte-identical to the refusal for a query matching nothing;
`existence_protected=True` is set; timing difference between the two cases is below
the noise floor of the test harness.

**FR-21 — Explain a served result**
*Priority: P1*
Every citation carries `why_allowed` so the UI can answer "why am I allowed to see
this" without a second authorisation pass.
**Acceptance:** `why_allowed` is populated from the recheck derivation, not from the
index's advisory `matched_token`.

### API

**FR-22 — HTTP API**
*Priority: P0*
FastAPI app exposing query, explain, tuple writes, and health.
**Acceptance:** principal identity comes from a verified token, never from a request
body field; every response carries `epoch` and `strategy`; contract tests cover
every endpoint.

**FR-23 — Admin tuple writes**
*Priority: P1*
Write and delete tuples over the API, with the epoch increment in the same
transaction as the write.
**Acceptance:** a crash between write and epoch increment is impossible — a test
forces the failure and asserts atomicity; writes require an admin relation, checked
through the same `check()` path as everything else.

### Evaluation

**FR-24 — Mutation harness**
*Priority: P0*
15 named permission mutants; the suite must kill them. See section 8.
**Acceptance:** `python -m eval.mutants run` applies each mutant, runs the suite,
records killed/survived; output is machine-readable; CI fails if the kill rate
differs from the number published in the README.

**FR-25 — Published numbers are generated**
*Priority: P0*
Every number in the README is produced by a script and checked against a fresh run.
**Acceptance:** CI fails the build when a committed figure drifts beyond its stated
tolerance. A drifting number is a failing build, not a stale document.

---

## 3. API surface

All endpoints are JSON over HTTP. The principal is taken from the verified bearer
token. There is deliberately no `principal` field in any request body: a caller who
can name their own principal is not a caller, they are an attacker.

```
POST /v1/query
  Request:  {"q": str, "k": int = 10, "max_context_chars": int = 8000}
  Response: {
    "text": str,
    "citations": [{"chunk_id": str, "object": str, "score": float,
                   "why_allowed": str | null}],
    "refused": bool,
    "refusal_reason": str | null,      # see RefusalReason
    "existence_protected": bool,
    "strategy": str,                   # PlanStrategy
    "epoch": int,
    "diagnostics": {"candidates": int, "dropped_at_recheck": int,
                    "plan_ms": float, "search_ms": float,
                    "recheck_ms": float, "synth_ms": float}
  }
  Errors: 401 no/invalid token; 409 stale policy (retry); 429 budget exceeded

POST /v1/search
  As /v1/query but returns rechecked hits with no generation. Same guarantees:
  every returned hit passed recheck.
  Request:  {"q": str, "k": int = 10}
  Response: {"hits": [{"chunk_id","score","text","object_ref","why_allowed",
                       "checked_at_epoch"}], "strategy": str, "epoch": int}

GET  /v1/explain?object=doc:42&relation=viewer&principal=user:alice
  Admin-only. Returns the derivation tree, allow or deny.
  Response: {"allowed": bool, "why": [str, ...], "checked_tuples": int,
             "epoch": int, "depth_reached": int}

GET  /v1/plan
  The caller's own compiled plan. Tokens are redacted to counts; a plan dump that
  prints raw tokens hands an attacker the filter.
  Response: {"strategy": str, "epoch": int, "estimated_cardinality": int,
             "n_grant_tokens": int, "n_explicit_ids": int}

POST /v1/tuples          Admin. {"writes": [str,...], "deletes": [str,...]}
                         -> {"epoch": int, "applied": int}
                         Write and epoch increment are one transaction.

GET  /v1/policy/epoch    -> {"epoch": int}

GET  /healthz            Liveness. No auth. No policy state in the body.
GET  /readyz             Readiness: tuple store reachable, index reachable,
                         epoch readable. Not ready means not serving.
GET  /metrics            Prometheus, when the `obs` extra is installed.
```

Python-level signatures that the HTTP layer is a thin wrapper over:

```python
def check(object: ObjectRef, relation: Relation, principal: PrincipalRef) -> Decision
def expand(object: ObjectRef, relation: Relation) -> UsersetTree
def compile_plan(principal: PrincipalRef, epoch: int) -> FilterPlan
def recheck(principal_ref: str, hits: Sequence[UncheckedHit]) -> tuple[list[Hit], int]
def answer(question: str, principal: PrincipalRef, k: int = 10) -> Answer
```

---

## 4. Data model

Types are defined in `src/sightline/types.py` and `src/sightline/store/base.py`.
Those files are the contract; this section describes the storage behind them.

**Tuples** — `(object_namespace, object_id, relation, principal_namespace,
principal_id, principal_relation NULL, created_at)`. Primary key is the whole
tuple. Indexed by `(object_namespace, object_id, relation)` for expand, and by
`(principal_namespace, principal_id, principal_relation)` for reverse lookup during
plan compilation. `principal_relation` non-null means the principal is a userset.

**Policy epoch** — a single row. Written in the same transaction as any tuple
write. Reading it is the cheapest operation in the system, because every query does
it.

**Namespace config** — relation definitions and rewrite rules per namespace. Versioned;
a namespace change also increments the epoch, because it changes what tuples mean.

**Chunks** — `(chunk_id, object_namespace, object_id, text, grant_tokens,
metadata, content_hash, indexed_at_epoch)`. `indexed_at_epoch` is how a reindex
queue knows what is behind.

**Vectors** — in the backend's own store, keyed by `chunk_id`, with `grant_tokens`
as a filterable payload field. The payload is the *only* permission data in the
index, and it is advisory.

**Reindex queue** — `(object_ref, reason, queued_at_epoch, state)`. `reason` is
`acl_change` or `content_change`. Membership changes never enqueue.

**Audit log** — append-only: `(ts, principal, query_hash, strategy, epoch,
n_candidates, n_dropped_at_recheck, refusal_reason, citation_object_refs)`. Query
text is hashed, not stored: an audit log of everyone's questions is a new liability
and we are not creating one.

---

## 5. Non-functional requirements

### Latency — reference machine is a 2015 dual-core Intel MacBook Pro, 8 GB RAM, no GPU

| Stage | p50 | p95 | Notes |
|---|---|---|---|
| Plan compile, cache hit | 0.2 ms | 1 ms | In-process, keyed on principal **and** epoch |
| Plan compile, cache miss | 15 ms | 60 ms | Dominated by reverse tuple lookup |
| Filtered search, 1M chunks | 25 ms | 90 ms | Qdrant, filter pushed into traversal |
| Recheck, k=10 | 8 ms | 25 ms | One batched store call |
| Guardrails | 3 ms | 10 ms | No model calls in the guardrail path |
| End to end, excluding generation | 60 ms | 200 ms | |
| End to end, including generation | — | 600 ms | Generation dominates and is not ours |
| Authorisation overhead vs unfiltered | — | ≤ 80 ms | The security layer's budget. Exceeding it invites someone to propose turning it off |
| `/v1/explain` | 60 ms | 200 ms | |

### Capacity

* 1M chunks, ~250k documents, on the reference machine with the Qdrant backend.
* 50k principals, 5k groups, nesting depth up to 16.
* 200k tuples in the in-memory store; unbounded with Postgres.
* Peak 20 queries/second on the reference machine. This is a laptop, not a claim
  about a cluster.

### Availability and degradation

* **Tuple store unreachable → refuse all queries.** Not degrade, not serve from
  cache. If the authority is down, there is no authority, and serving from a cached
  plan is exactly the leak the product exists to prevent.
* **Index unreachable → refuse** with a distinct reason. Nothing is served from a
  secondary path.
* **Epoch unreadable → refuse** with `STALE_POLICY`.
* Readiness probe fails on any of the above; an unready instance is removed from
  rotation rather than serving degraded.
* Target 99.9% for the query path, with the explicit note that every failure mode
  above is a *refusal*, and refusal is the safe state.

### Portability

* Python 3.11+. Core package and the **entire** test suite run with only pydantic,
  fastapi, numpy, httpx.
* `qdrant-client`, `psycopg`, `onnxruntime`, `faiss` are optional extras. Every
  optional import is guarded and raises an error naming the extra to install.
* No torch, anywhere, ever, for the reason in FR-16.

---

## 6. Security requirements, mapped to the OWASP Top 10 for LLM Applications

| OWASP item | How it shows up here | Requirement |
|---|---|---|
| **LLM01 Prompt injection** | A document in the corpus contains text instructing the assistant to ignore its rules or to fetch other documents | SR-1: scan retrieved chunk text for injection patterns before it enters context; refuse with `INJECTION_DETECTED`. SR-2: retrieved text is never treated as instructions — it is delimited and labelled as data in the prompt. SR-3: **injection cannot escalate permission**, because the plan is compiled before retrieval and recheck runs after it; there is no path from chunk text to the filter |
| **LLM02 Insecure output handling** | Generated text is rendered in a UI or passed to a downstream tool | SR-4: output is treated as untrusted text; no tool-calling from generated content in v1. SR-5: citations are reconstructed from the rechecked set, so a citation to an unreadable document is unrepresentable, not merely discouraged |
| **LLM03 Training data poisoning** | Not applicable: nothing is trained here | SR-6: state the non-applicability rather than claiming a control. Embeddings are computed, never fine-tuned |
| **LLM04 Model denial of service** | A query that returns many hits triggers many permission checks | SR-7: k is capped; recheck is batched; context is capped by `max_context_chars`; over-budget requests refuse with `BUDGET_EXCEEDED` rather than skipping any check |
| **LLM05 Supply chain** | Optional extras pull in large dependency trees | SR-8: core install is four dependencies; extras are pinned with upper bounds; CI runs the suite with core deps only to prove the guard rails work |
| **LLM06 Sensitive information disclosure** | **The central one.** The assistant answers correctly from a document the reader may not see | SR-9: mandatory recheck (FR-13). SR-10: existence protection (FR-20). SR-11: grant tokens are keyed hashes, so index payloads do not disclose group names. SR-12: the audit log stores a query hash, not query text |
| **LLM07 Insecure plugin design** | No plugins in v1 | SR-13: the synthesiser interface accepts text and returns text plus chunk ids. It has no network and no filesystem access |
| **LLM08 Excessive agency** | The system decides to widen its own search | SR-14: no query rewriting that expands the permitted set; no autonomous retrieval loops in v1. One query, one plan, one epoch |
| **LLM09 Overreliance** | A user trusts an answer built from thin evidence | SR-15: grounding check — an answer whose claims are not supported by the cited chunks refuses with `UNGROUNDED`. SR-16: refusal is a first-class outcome, not a fallback |
| **LLM10 Model theft** | Not applicable | SR-17: no proprietary model is hosted |

Additional, not on the OWASP list but load-bearing:

* **SR-18** — principal identity comes only from a verified token signature. No
  request field, header, or query parameter may name the principal.
* **SR-19** — admin operations are authorised through the same `check()` path as
  everything else. There is no admin bypass, because a bypass is an untested code
  path with maximum privilege.
* **SR-20** — no configuration flag, environment variable, or feature toggle can
  disable recheck, the epoch check, or the plan/oracle agreement test. A flag is a
  thing someone will set at 2am during an incident.

---

## 7. Guardrails — the full list

Each runs on the **rechecked** set. None of them is a substitute for the
authorisation layer; they are defence in depth behind it.

1. **Stale plan guard** — refuse if `plan.epoch < live_epoch`. Reason:
   `STALE_POLICY`.
2. **Recheck guard** — the type system. `Hit` is unconstructable outside
   `recheck()`, enforced by a source-grepping test.
3. **Empty-permitted-set guard** — if the principal's plan admits nothing, refuse
   with `NO_PERMITTED_EVIDENCE` before any search runs.
4. **Existence protection** — `NO_PERMITTED_EVIDENCE` and `NO_EVIDENCE_AT_ALL`
   produce identical user-visible output and comparable timing. The distinction
   exists only in the audit log.
5. **Injection scanner** — pattern and heuristic scan over retrieved chunk text for
   instruction-shaped content. Refuse with `INJECTION_DETECTED`. No model call, so
   it cannot itself be prompt-injected.
6. **Instruction isolation** — retrieved text is placed in the prompt inside
   delimiters and labelled as untrusted data.
7. **Grounding check** — every sentence of the answer must be supported by a cited
   chunk. Unsupported answers refuse with `UNGROUNDED`.
8. **Citation reconstruction** — citations are built by mapping returned chunk ids
   back to the rechecked set. Ids not in that set are dropped silently. The
   synthesiser never emits free-text citations.
9. **Context budget** — hard cap on characters and on k. Over budget refuses with
   `BUDGET_EXCEEDED`; it never silently drops the recheck to fit.
10. **Depth limit** — userset expansion beyond `MAX_USERSET_DEPTH` denies.
11. **Cycle guard** — cycles in the group graph terminate the branch and deny.
12. **Token opacity** — grant tokens are never returned raw by any endpoint;
    `/v1/plan` returns counts.
13. **Oracle drift check** — a background job samples served queries and replays
    them against the exact oracle. Any disagreement pages, because it means the
    plan and `check()` have diverged.
14. **No-flag guard** — a test asserts that no setting disables guardrails 1, 2,
    or 13.

---

## 8. Test strategy

### 8.1 Layers

* **Unit** — tuple parsing, token derivation, plan compilation, strategy selection.
* **Property-based** (hypothesis) — randomly generated tuple graphs; assert
  plan/oracle agreement (FR-6) across nesting depths 0–5 and fan-out 1–20. This is
  where the subtle expansion bugs die.
* **Conformance** — one suite, run against every `VectorStore` backend, including
  the ones behind optional extras (skipped cleanly when absent).
* **Differential** — serving path vs exact oracle. Leak count must be exactly 0.
  Recall must clear the floor.
* **Source-level** — grep tests: `Hit` is constructed only inside `recheck`; no
  strict top-k equality assertion appears in an approximate-path test; no torch
  import anywhere.
* **Contract** — every HTTP endpoint, including the error and refusal paths.
* **Performance** — latency budgets from section 5, asserted with generous
  tolerance on the reference machine, reported rather than asserted in CI.

The entire suite runs with core dependencies only: pydantic, fastapi, numpy, httpx.
A test that requires an extra is skipped, and CI has a job that runs with **only**
the core dependencies installed to prove it.

### 8.2 The mutation-kill-rate gate

This is the headline metric for the whole project.

We plant 15 deliberate permission bugs. Each is a real mistake a competent engineer
could make. The harness applies one mutant at a time, runs the suite, and records
whether the suite failed. A mutant that causes a failure is **killed**. A mutant
that passes all tests **survives**, and a survivor means the test suite does not
cover that behaviour, whatever the coverage percentage says.

| ID | Mutation | Expected to be killed by |
|---|---|---|
| **M1** | `FilterPlan.is_stale()` always returns `False` — the epoch check is dropped | Revocation test: tighten a document's ACL, query with a cached plan, expect the result to disappear |
| **M2** | Group condition combination uses OR where it should use AND | Plan/oracle agreement property test on a principal whose access requires two conditions |
| **M3** | The pipeline skips `recheck()` and constructs `Hit` directly from `UncheckedHit` | Source-grep test, plus the stale-index differential test |
| **M4** | Recheck trusts `UncheckedHit.matched_token` instead of re-deriving from live tuples | Forge a hit with a token the principal no longer holds; expect it dropped |
| **M5** | Off-by-one in userset expansion: the depth limit permits one extra level, or truncation resolves toward allow | Nested-group test at exactly the depth boundary, both sides |
| **M6** | An empty grant-token set is treated as "match everything" rather than "match nothing" | Query as a principal with zero permissions; expect zero candidates, not the whole corpus |
| **M7** | Post-filter path pads results back up to k from the unfiltered pool | Differential leak test against the oracle |
| **M8** | Plan cache keyed on principal only, ignoring epoch | Write a tuple, requery within the cache TTL, expect the new policy |
| **M9** | Reverse tuple lookup during plan compilation uses string prefix matching, so `group:legal` also matches `group:legal-interns` | Adversarial naming fixture in the corpus generator |
| **M10** | Recheck processes only the first batch of hits and passes the remainder through | Query with k greater than the batch size, with a revoked document ranked late |
| **M11** | Refusal for "you may not see it" differs from "it does not exist" in text or in status code | Existence-protection byte-equality test |
| **M12** | Chunk text is included in an error or refusal payload | Refusal-path contract test asserting no chunk text in any non-grounded response |
| **M13** | A document's ACL change fails to enqueue a reindex, while membership changes do enqueue | Reindex-queue counter test for both event types |
| **M14** | Citations are parsed out of the generated text rather than reconstructed from the rechecked set | Synthesiser fixture that emits a citation to a forbidden chunk id; expect it dropped |
| **M15** | The epoch increment is moved outside the tuple write transaction | Induced-crash atomicity test between write and increment |

**Target: 15/15 for v1. Publishable floor: 13/15, with every survivor named in the
README and explained.**

A suite that catches 12 of 15 and says so is honest and interesting. A suite that
claims 15 of 15 without a runnable harness is a sentence in a slide deck. Deleting
an inconvenient mutant is the failure mode of this whole approach, so mutants are
specified here, in this document, before the tests are written, and removing one is
a review-blocking change.

**CI gate:** `eval/mutants` runs on every push. The build fails if the measured
kill rate is lower than the last committed value, or if the README's published
number does not match a fresh run. The number in the README is generated, not
typed.

### 8.3 Other published numbers, also gated

* Recall at 10 vs the exact oracle, filtered, at permission densities 0.1%, 1%,
  10%, 50%, 100%. Floor 0.95. Drift tolerance ±0.02.
* Recall at 10 for the post-filter baseline at the same densities. This one is
  expected to collapse; the README publishes the collapse because it is the
  argument for pushing the filter into the index.
* Leak count against the oracle over 10^6 randomised query/principal pairs.
  Expected exactly 0. Any other value fails the build outright.
* Vector writes per membership change. Expected exactly 0.
* Latency table from section 5, reported per run on the reference machine.

### 8.4 What is deliberately not tested

Named so that their absence is not mistaken for coverage.

* No load testing beyond 20 queries/second. The reference machine is a laptop and
  a throughput number from it would be meaningless.
* No connector permission-mapping tests, because there are no connectors in v1.
* No adversarial machine-learning evaluation of the injection scanner. It is a
  pattern matcher, it is described as a pattern matcher, and claiming a detection
  rate for it would be dishonest.
* No fuzzing of the HTTP layer beyond schema validation. FastAPI and pydantic are
  doing that job and re-testing them is theatre.
