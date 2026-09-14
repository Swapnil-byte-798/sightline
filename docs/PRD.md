# Sightline — Product Requirements Document

Status: draft, v1 scope frozen.
Owner: project author.
Last updated: alongside the code in this repository. If this document and the code
disagree, the code is right and this document is a bug.

---

## 0. Terms, defined once

This document tries to use ordinary words. Where it cannot, the term is defined
here the first time it appears.

* **RAG** — retrieval-augmented generation. The assistant searches your documents,
  puts the best matching passages into the model's prompt, and the model answers
  from those passages.
* **Chunk** — one passage of a document, a few hundred words. Search happens over
  chunks, not whole files.
* **Vector index** — the search structure that finds chunks by meaning rather than
  by keyword. It holds its own copy of each chunk's metadata.
* **Tuple** — one permission fact, written `object#relation@principal`. Example:
  `doc:42#viewer@group:legal#member` means "every member of the legal group may
  view document 42". This is the only way permission is written down in Sightline.
* **Principal** — whoever is asking. A person, or a service account.
* **Userset** — a principal that is itself a group-and-relation, like
  `group:legal#member`. Usersets are how nested groups work.
* **Grant token** — a short opaque string stamped onto a chunk at indexing time
  that stands for "whoever satisfies this group edge may see this chunk". The
  index filters on these.
* **Epoch** — a counter that goes up by one every time anyone writes a permission.
  It is how the system knows a cached permission answer is out of date.
* **Recheck** — asking the authoritative permission database, after search, whether
  the searcher may really see each result.
* **Mutation testing** — deliberately breaking your own code in a known way and
  checking that some test fails. If no test fails, the test suite did not cover
  that behaviour, whatever the coverage percentage said.

---

## 1. The problem

An organisation buys an AI assistant that searches its own documents. Somebody
runs a pilot. Within about two weeks, someone in the pilot group types a question
like "what is the plan for the Denver site" or "what are this year's comp bands"
and gets a clear, correct, well-written answer built out of documents they were
never supposed to open.

Nobody hacked anything. The documents were sitting in a SharePoint folder that
somebody set to "everyone in the organisation" in 2019 to unblock a deadline, or
in a Google Drive folder shared by link, or in a Confluence space with inherited
permissions nobody has read since the space was created. The files were always
technically reachable. What changed is that finding them used to require knowing
they existed and guessing the right words. Now a language model reads everything
the searcher is nominally entitled to and writes them a summary.

This is the core dynamic and it is worth stating plainly: **AI search does not
create new access. It removes the obscurity that was doing the actual security
work.** Every large document estate has a long tail of over-shared files that
survived only because nobody could find them.

The result is predictable. The rollout gets paused. It sits paused for months,
because nobody can say what the assistant can see, and "we think it's fine" is not
an answer a CISO can sign. The pilot is not cancelled — cancelling admits the
spend was wasted — so it stays in limbo, and the vendor's renewal conversation
goes badly.

### The second problem, which arrives with the first

Once the rollout is paused, someone asks: can you prove it is safe now? And the
honest answer from most stacks is no, because permission enforcement lives in the
search index, and the search index is a copy. Copies go stale. If the enforcement
point is a stale copy, then somewhere there is a window — minutes, hours, or until
the next nightly reindex — in which a person who lost access still gets results.
Nobody can tell you how wide that window is, because nobody measured it.

---

## 2. Who has this problem

The buyer is one of two people, and usually both have to agree.

**The CISO or Head of Security Engineering.** Owns the veto. Their problem is not
that they want AI search; it is that the business wants it and they are the one who
has to sign. They need an artifact they can hand to audit that says what the system
can see and why. They are being asked to approve a system whose failure mode is
"correct answer, wrong reader", which is the failure mode their existing DLP
tooling is worst at catching, because nothing was exfiltrated — it was retrieved
by a legitimate user through a legitimate interface.

**The Head of Knowledge Management, Head of Internal Productivity, or whoever owns
the intranet.** Owns the budget and the pain. They bought Glean or turned on
Microsoft 365 Copilot, they have a per-seat contract running, and adoption is
frozen at the pilot cohort. Their problem is that they are paying for something
that is not deployed, and every month of delay makes the purchase look worse.

The third party in the room is **the IT admin or platform engineer** who will
actually operate the thing. They do not care about the security argument in the
abstract. They care that they can answer, at a terminal, in under a minute: why
did this user see this document.

### Company shape where this bites hardest

* 1,000 to 50,000 employees. Below that, people know what is in the shared drive.
  Above that, the buying cycle is long enough that this document is not the
  bottleneck.
* Regulated or contractually constrained: financial services, healthcare,
  legal, defence supply chain, any company with customer data-handling clauses.
* Document estate spread across at least three systems with three different
  permission models.
* Has already had one incident, or one close call, in a pilot. That incident is
  the actual trigger for the purchase.

---

## 3. Jobs to be done

| Who | Job | What they do today | What "done" looks like |
|---|---|---|---|
| CISO | Approve an AI search rollout without taking personal risk for a leak | Delays; demands a manual review of shared-folder permissions that never finishes | A written enforcement argument plus a test result they can cite in an audit response |
| CISO | Answer "what could the assistant have shown last Tuesday" after an incident | Nothing useful; logs show queries, not the permission state at the time | A per-query record: strategy used, epoch, number of results dropped at recheck |
| Head of Knowledge | Unfreeze the rollout and get past the pilot cohort | Runs a permission clean-up project that takes two quarters | Turn the assistant on for everyone, with the guarantee that mis-shared files are still a data-hygiene problem, not an AI problem |
| Platform engineer | Explain one specific access decision | Reads three permission UIs and guesses | One API call returns the derivation path: which tuple, through which group, at which epoch |
| Platform engineer | Change group membership without a reindex | Waits for the nightly job | Membership change costs zero vector writes. Only a document's own permissions changing touches the index |
| Compliance / audit | Test the control, not the claim | Reviews vendor documentation | A reproducible test suite with a published pass rate, run in CI, that fails the build if the published number drifts |

---

## 4. Why existing products do not solve this

Being precise here matters, because every vendor in this space says the words
"respects your existing permissions", and most of them do — for some definition
that does not survive contact with a nested group and a revocation.

**They enforce at the index.** The dominant design stores an access-control list
on each document in the search index and filters on it. This is fast and it is
mostly right. The failure is structural: the index is a denormalised copy of a
permission state that lives somewhere else, so it lags. The lag is small on a good
day and unbounded on a bad one (a failed sync job, a rate-limited connector, a
weekend). During the lag, enforcement is wrong in the direction that leaks.

**They filter after retrieval.** A common shortcut: search the whole corpus for
the top 50 results, then drop the ones this user cannot see, then answer from what
is left. This is safe against leaks and terrible at its job. If a user can see 2%
of the corpus, most of their 50 slots are consumed by documents that get thrown
away, and the answer is built from whatever scraps survive. Recall collapses
quietly, and quiet recall collapse reads to users as "the AI is dumb", not as "the
filter is wrong". We implement this deliberately as our comparison baseline,
because showing the size of that collapse is the clearest way to explain why the
filter has to be pushed into the index.

**They tag chunks with user identities or document id lists.** Both are operational
traps. Tag with user ids and every hire, departure, and team transfer rewrites
vectors. Tag with "allowed document ids" and the filter becomes a 40,000-element
list that no index can execute efficiently. Both designs make permission changes
expensive, and expensive changes get batched into a nightly job, which is exactly
how the staleness window gets wide.

**They stop at governance reporting.** A large category of tool tells you that a
folder is over-shared. That is genuinely useful and it is not the same product.
Knowing that `\\HR\Comp 2024` is open to the whole company does not stop the
assistant citing it at 9am tomorrow; somebody still has to go and fix the folder,
and there are eleven thousand folders.

**They are a policy layer in front of the model.** Another category intercepts the
prompt and the response and applies a rule about what the assistant should not
say. That is a guardrail, not an access control. It is enforced by a classifier on
generated text, which means it is probabilistic, it is bypassable by rephrasing,
and it operates after the forbidden content has already been placed into the
model's context. If the content reached the context, it reached a third-party API
and a log.

**None of them publish a number.** Not one ships a test result that an auditor can
re-run. "Permission-aware" is an adjective. This project's position is that the
only useful version of this claim is a harness that deliberately breaks the
permission logic in specific ways and reports how many of those breaks the test
suite catches.

---

## 5. What Sightline is

Sightline compiles permissions into the search index, so a document you cannot
open is never a candidate, never enters the model's context, and cannot be cited.

Three commitments hold it together.

**1. The index is a hint. The database is the authority.**
Search returns objects of type `UncheckedHit`. The answer builder accepts only
`Hit`. The only way to turn one into the other is `recheck()`, which asks the live
permission database. The two types are different types on purpose: skipping the
check is not a discipline problem, it is a code that does not run. The consequence
is the security argument of the entire product — **a stale index loses results, it
never leaks them.** A document whose permissions were tightened since indexing
gets dropped at recheck. A document whose permissions were loosened is missed
until the next reindex. That second case is an availability cost and we chose it
knowingly.

**2. Chunks are tagged with grant tokens derived from groups.** Never with user
ids, never with document id lists. A person joining or leaving a group rewrites
zero vectors, because the token stands for the group edge, not for the people
currently behind it. Only a document's own permissions changing touches the index.
This is what makes the staleness window narrow enough to be honest about:
membership churn, which is most of the churn, does not touch the index at all.

**3. Every cached permission answer carries an epoch.** The epoch goes up on every
permission write. A compiled plan built under an older epoch than the live policy
is stale and is not served. The system would rather recompile or refuse than serve
an answer it cannot date.

---

## 6. Goals

**G1 — No leak through retrieval.** No chunk the asking principal cannot see under
a live authorisation check ever reaches the model's context or a citation. This is
binary; it is the product.

**G2 — Keep recall while filtering.** Filtering must happen inside the index, not
after it, so that a user who can see a small slice of the corpus still gets good
answers about that slice. Target: with permission filtering on, recall at 10
against a brute-force exact oracle stays at or above 0.95 across all tested
permission densities.

**G3 — Prove it, do not assert it.** Ship a mutation harness that plants 15
specific permission bugs and reports how many the test suite catches. Publish the
number. Fail CI when the published number drifts from a fresh run.

**G4 — Explain any decision.** Every result carries why it was allowed. Every
refusal carries why it was refused, in a machine-readable form.

**G5 — Permission changes are cheap.** Membership change: zero vector writes.
Document ACL change: bounded reindex of that document's chunks only.

**G6 — Runs on a laptop.** The whole system, including the test suite, runs on a
2015 dual-core Intel MacBook Pro with 8 GB of RAM and no GPU. No torch. Heavy
backends are optional extras. If an evaluation cannot be reproduced by a reader on
commodity hardware, it is marketing.

---

## 7. Non-goals

Naming these matters more than the goals do, because an unstated non-goal reads as
a missing feature.

* **Not a permission fixer.** Sightline will not clean up your over-shared
  SharePoint folders. It will stop the assistant from surfacing them, and it will
  tell you which ones it had to stop. Remediation is somebody else's product.
* **Not a data loss prevention tool.** It does not scan outbound email, it does not
  classify documents by sensitivity, it does not watermark.
* **Not a model provider.** Sightline is the retrieval and authorisation layer.
  Bring your own generator. The synthesiser interface is deliberately narrow.
* **Not an identity provider.** It consumes identity and group membership from your
  IdP. It does not manage users or issue credentials.
* **Not attribute-based or time-based policy in v1.** No "viewable only from a
  corporate IP", no "expires Friday". The tuple model can express these later; the
  compiler cannot yet, and half-implementing it is worse than not having it.
* **Not multi-tenant SaaS in v1.** Single-organisation deployment. There is
  deliberately no `tenant_id` column anywhere, because a tenant is just an object
  in the tuple graph and a special-case column is how the isolation bug gets
  written.
* **Not a UI product.** There is an admin console for explaining decisions. There
  is no end-user chat interface; that is your existing assistant's job.
* **Not write operations.** Sightline reads documents. It never modifies the source
  systems.
* **Not real-time sub-second permission propagation across connectors.** The
  connector polls. We are honest about the lag, and the architecture makes the lag
  lose results rather than leak them.

---

## 8. Success metrics

The headline number is first because it is the one the README publishes and CI
gates on.

### Headline: mutation kill rate

We plant 15 deliberate permission bugs in the codebase — drop the epoch check, use
OR instead of AND when intersecting group requirements, skip the recheck, trust
the token stamped on the index, go one level too far in expanding nested groups,
and so on. Each one is a real bug that a competent engineer could write on a bad
afternoon. The test suite must fail for each.

* **Target for v1: 15/15.**
* **Publishable floor: 13/15**, with the survivors named in the README and an
  explanation of why they survive.

A suite that catches 12 of 15 and says so is honest and useful. A suite that claims
15 of 15 without a harness that anyone can run is a sentence in a slide deck. The
harness is in `eval/mutants/`, it runs in CI, and the build fails if the number in
the README does not match a fresh run.

### Supporting metrics

| Metric | Target | Why this number |
|---|---|---|
| Leak rate under the authorisation oracle | 0 in 10^6 randomised query/principal pairs | Any non-zero value here is a breach, not a regression. The oracle is a brute-force exact check over all tuples |
| Recall at 10 vs exact oracle, filtered | ≥ 0.95 at every permission density from 0.1% to 100% of corpus visible | The approximate index is allowed to miss a little. It is not allowed to fall off a cliff when the filter gets selective |
| Recall at 10, post-filter baseline, at 1% visibility | Expected ≤ 0.30 | This is the number we are trying to make look bad. If the baseline does not collapse, the whole argument is wrong and we should know |
| p95 end-to-end query latency | ≤ 600 ms on 1M chunks, on the reference laptop | Slower than this and users route around the assistant |
| p95 added latency from authorisation | ≤ 80 ms over an unfiltered query | The security layer has a budget. If it exceeds the budget, someone will propose turning it off |
| Vector writes per group membership change | 0 | The architectural claim, stated as a measurement |
| Staleness window, document ACL tightened → not retrievable | ≤ 1 query (immediate, via recheck) | Tightening is enforced at recheck, so it takes effect on the next query, not the next reindex |
| Staleness window, document ACL loosened → retrievable | ≤ 15 min p95 | The deliberately chosen availability cost. Stated so nobody is surprised by it |
| Explain-decision API response | ≤ 200 ms p95, always includes the full derivation path | An unexplainable decision is a liability |
| Cold install to first answer | ≤ 10 min, core dependencies only | pydantic, fastapi, numpy, httpx. If a reviewer cannot get it running over lunch, they will not review it |

### Adoption metrics, if this became a real product

* Pilot cohort expanded beyond the frozen group within 30 days of install.
* Security sign-off obtained without a permission clean-up project as a
  precondition.
* Number of "why did I see this" support tickets that are closed with a link to an
  explain-decision output rather than a manual investigation.

---

## 9. Scope

### v1 — in scope

* Zanzibar-style tuple store with nested groups via usersets.
* `Check()` — the authoritative, slow, correct permission answer.
* `Expand()` — the derivation path behind a decision, for the explain API.
* Filter plan compiler: turns one principal's permissions into something an index
  can execute, choosing between four strategies based on measured cardinality
  (enumerate ids, filter on grant tokens, skip filtering, or exact scan).
* Epoch counter, incremented on every permission write; stale plans are refused.
* Grant token derivation and stamping at ingest.
* Three vector backends behind one interface: Qdrant (serving), pgvector with
  Postgres row-level security (independent second enforcement layer), brute-force
  FAISS (the oracle — never served, used to prove the others return the complete
  correct answer under a filter).
* Post-filter baseline store, implemented deliberately as the comparison arm.
* Mandatory recheck between search and answer, enforced by the type system.
* Guardrail set: grounding check, injection detection in retrieved text, existence
  protection, citation reconstruction from the permitted set only, budget limits.
* Query and explain APIs.
* Mutation harness with the 15 planted bugs, wired into CI.
* Evaluation suite producing the published recall and latency numbers, with CI
  failing on drift from the committed figures.

### v1 — explicitly deferred

* Live connectors to SharePoint, Google Drive, Confluence, Slack. v1 ingests from a
  local corpus and a documented tuple-import format. Connectors are where the
  calendar goes to die and they are not where the interesting problem is.
* Attribute-based and time-bounded policy.
* Multi-tenancy.
* Write-back to source systems.
* An end-user chat interface.
* Incremental index compaction and tiered storage.
* Caching layer for compiled plans across processes (v1 caches in-process only,
  keyed on principal **and** epoch).

### Later, in rough order

1. One real connector, end to end, with its permission model mapped to tuples and
   the mapping tested. SharePoint first, because it is where the problem lives.
2. Plan cache shared across processes, still epoch-keyed.
3. Attribute and time conditions in the tuple language.
4. Multi-tenant deployment, with the tenant expressed as an object in the graph.
5. Continuous authorisation drift monitoring: run the oracle against the serving
   path on a sample of live traffic and alert on any disagreement.

---

## 10. Competitive landscape

| Product | What it actually is | Where it is strong | Why it does not close this gap |
|---|---|---|---|
| **Glean** | Enterprise search plus an assistant, with connectors to everything | Best-in-class connector coverage and ranking. Genuinely good product | Permissions are mirrored into their index and enforced there. Mirrors lag. They do not publish a staleness bound or a leak test. The over-shared-folder problem is treated as the customer's data hygiene issue, which is correct as an argument and useless as an answer to a paused rollout |
| **Microsoft 365 Copilot + Purview** | Assistant over the Microsoft graph, plus a labelling and governance suite | Enforcement is close to the source of truth inside Microsoft's own estate. Purview's sensitivity labels are real controls | Two problems. First, it is Microsoft-only; the Confluence and Google Drive halves of the estate are outside. Second, Purview requires the documents to be labelled, and labelling a legacy estate is the two-quarter project people are trying to avoid. Semantic Index inherits the same over-sharing exposure — this is precisely the pattern that has frozen the most rollouts |
| **Varonis** | Data security posture management: finds over-shared and stale-permission data | Excellent at the discovery half. Will tell you exactly which folders are open to everyone | Reporting, not enforcement. It hands you a remediation backlog. Nothing in it sits between the assistant and the index at query time |
| **Knostic** | Need-to-know policy layer for enterprise AI assistants | Correctly identifies the exact problem in this document and articulates it well | Operates as a policy and inference layer around the assistant rather than compiling permissions into retrieval. Enforcement on generated output is after the fact: the content already entered the model's context. Closest competitor in framing, different in mechanism |
| **Onyx (formerly Danswer)** | Open-source RAG over company documents, with connectors | Open, self-hostable, easy to stand up, honest about being open source | Permission filtering is index-side ACL matching. No mandatory post-search recheck against a live authority, no epoch, no published leak test. It is the closest thing to this project's shape, and the gap is the enforcement guarantee, not the features |
| **Roll your own on pgvector or Qdrant** | What most teams actually do | Full control, no vendor | The team writes the filter once, correctly, and then somebody adds a feature. There is no oracle to catch the regression. This is the audience for the open-source version of Sightline |

### How Sightline differs, in one paragraph

Everyone filters. The difference is where the authority lives and whether anyone
checked. Sightline makes the index non-authoritative by construction — the type
returned by search cannot be shown to a user without a live database check — and
then it plants fifteen permission bugs in its own source and publishes how many of
them its tests catch. Nobody else does the second part. The first part is a design
choice other people could adopt; the second part is the thing that makes the first
part believable.

---

## 11. Risks

| Risk | Likelihood | Impact | What we do about it |
|---|---|---|---|
| The mandatory recheck adds too much latency, and someone proposes making it optional "just for the fast path" | High | Fatal — it is the whole product | Recheck has a measured latency budget (80 ms p95). Batch it. Cache the tuple store's hot subgraph. Never make it configurable: there is no flag to turn it off, because a flag is a thing someone will set |
| Recheck becomes a denial-of-service vector: a query returning 500 hits triggers 500 permission checks | Medium | High | Recheck is batched into one store call, and k is capped. Load-shedding refuses with `BUDGET_EXCEEDED` rather than skipping the check |
| Grant token cardinality explodes: a document with many distinct group grants produces so many tokens the index filter becomes slow | Medium | Medium | Cardinality is measured per plan, and the plan compiler falls back to `ENUMERATE` or `EXACT_SCAN` when tokens are a bad fit. This is why there are four strategies and not one |
| Grant tokens leak information by themselves — a token is derived from a group, so an attacker who can observe tokens learns group names | Low | Medium | Tokens are opaque and derived by keyed hash. They are advisory in the response and never trusted at recheck |
| The approximate index (HNSW) misses permitted results, and someone reads a recall miss as a security feature | Medium | Medium | We never assert "filtered top-k equals exact top-k". That equality only holds when the query routes to brute force. We publish a recall floor against the exact oracle instead, and the floor is a test |
| Nested group expansion has a cycle, or is deep enough to be slow | Medium | Medium | Expansion has an explicit depth limit and cycle detection. Both are tested. The depth limit is a refusal, not a silent truncation — silent truncation in the permissive direction is mutant #5 |
| Loosened permissions take up to 15 minutes to become retrievable, and users file it as a bug | High | Low | It is a documented, deliberate trade-off, surfaced in the docs and in the response metadata. The alternative — trusting the index — is the thing we are selling against |
| Connectors are not in v1, so nobody can try it on their real data | High | High commercially, low technically | Accepted. v1 ships a documented tuple-import format and a synthetic corpus generator. The argument is the architecture and the harness, not the integration surface |
| Microsoft closes the gap inside its own estate and the addressable problem shrinks to hybrid estates | Medium | High | Hybrid estates are the majority and are getting more hybrid, not less. Also accepted: this is a portfolio project first |
| Mutation harness becomes theatre — mutants are written to be easy to catch | Medium | High | Mutants are specified in the FRD before the tests are written, each one is a bug pattern seen in real systems, and any survivor is published rather than removed. Deleting an inconvenient mutant is the failure mode and it is a review-blocking change |
| The published numbers drift from reality as the code changes | High | High — it is the author's whole point | CI re-runs the evaluation and fails the build if a committed number differs from a fresh run beyond tolerance. The README's numbers are generated, not typed |
