## Relationship-based access control, the Zanzibar model

### The short version

Most systems store permissions as a label on a row: a tenant column, a role name, a public flag. That works until permission stops being a property of the thing and becomes a property of a chain — this document belongs to that folder, that folder is shared with that team, and you are in a team inside that team. The Zanzibar model stores permissions as plain facts about relationships between two named things, and answers "may this person see this document?" by walking the chain. It is slower and more general, and the generality is the point.

### The intuition

Start with the thing everyone builds first.

You have a documents table. You add a `tenant_id` column so customers cannot see each other's files, and an `is_public` flag so some files are visible to the whole company. Two columns, one `WHERE` clause, done. For about eighteen months this is the correct engineering decision and anyone who tells you otherwise is over-designing.

Then a real organisation shows up. Say it has 2,400 employees, 40,000 documents, and 310 groups. The requests arrive in this order:

1. "Legal should see the contracts folder." Fine — add a `group_id` column, or a join table.
2. "EU Legal is part of Legal, so they should see it too." Now membership is a graph, not a column. Your `WHERE group_id = ?` needs to become "group_id is in the set of groups reachable from this person", and that set is computed by walking edges.
3. "The Q3 subfolder inherits from the contracts folder, except one file that belongs to the deal team." Now permission is inherited down a second graph, the containment graph, and one leaf overrides it.
4. "Alice is on secondment to Legal for three months." One row, and somebody has to be sure exactly which 6,000 documents that row now exposes.

Each step is reasonable. Together they mean the answer to "may Alice read doc 42?" cannot be read off doc 42. It has to be derived. And the flag from step one is now actively dangerous: `is_public = true` was set on a SharePoint folder in 2019 by someone who meant "the people on my floor who know the URL", and nothing in your schema records the difference.

Here is the reframing that makes the rest of this chapter obvious. Stop storing **attributes of documents**. Start storing **facts about pairs of things**, one fact per row:

```
doc:42       is viewed by   group:legal#member
group:legal  has member     group:legal-eu#member
group:legal-eu has member   user:alice
```

Three facts. No flag, no tenant column, no role table. "May Alice read doc 42?" is now a question about whether a path exists from `user:alice` to `doc:42`. Nested groups are not a feature you implement; they are what happens when the middle row's subject happens to be another group. Nobody had to plan for them.

This is relationship-based access control (ReBAC), and the specific spelling below comes from Google's Zanzibar paper.

### How it actually works

**The tuple.** One permission fact is written:

```
object # relation @ principal

doc:42#viewer@user:alice                 a person, directly
doc:42#viewer@group:legal#member         every member of group:legal
doc:42#parent@folder:contracts           doc:42 lives in that folder
group:legal#member@group:legal-eu#member every EU Legal member is a Legal member
```

The `object` is a namespaced identifier (`doc:42`, `folder:hr`). The `relation` is a name (`viewer`, `member`, `parent`). The `principal` — Zanzibar calls it the user — is either a concrete subject (`user:alice`) or a **userset**: an object plus a relation, `group:legal#member`, meaning "whoever holds `member` on `group:legal`, whoever that turns out to be".

The userset is the whole idea. A tuple whose subject is a userset is a pointer into another set that will be resolved at check time. Because the second row above has a userset on *both* sides, groups nest for free. There is no group table, no `depth` column, no closure table to maintain.

**The namespace config.** Tuples alone cannot express "an editor is also a viewer" or "documents inherit from their folder", because those are rules, not facts. Each namespace declares, per relation, a *rewrite*: a small expression saying how the relation is satisfied. Sightline supports three forms and one combinator, which is the Zanzibar core minus intersection and exclusion:

| Form | Meaning | Example |
|---|---|---|
| `this` | Direct tuples only. The base case. | `owner = this` |
| `computed_userset(r)` | Another relation **on the same object** implies this one. | `editor = union(this, computed(owner))` |
| `tuple_to_userset(t, r)` | For every object reached by relation `t`, whoever holds `r` *there* holds this relation here. | `viewer <- ttu(parent, viewer)` |
| `union(...)` | Satisfied if any child is. | see above |

So the document namespace reads:

```
doc.owner  = this
doc.editor = union(this, computed(owner))
doc.viewer = union(this, computed(editor), ttu(parent, viewer))
doc.parent = this
```

Four lines. "Owners are editors, editors are viewers, and viewers of my folder are viewers of me."

**Check(), recursively.** The evaluator takes an object, a relation and a principal, and asks whether a derivation exists. It is depth-first with memoisation, and it looks like this for `check(doc:42, viewer, user:alice)`:

```
doc:42 #viewer  -> union(this, computed(editor), ttu(parent, viewer))
  |
  +- this: read tuples on (doc:42, viewer)
  |     doc:42#viewer@group:legal#member      subject is a userset ->
  |       check(group:legal, member, alice)                     depth+1
  |         this: read (group:legal, member)
  |           group:legal#member@group:legal-eu#member  -> recurse depth+2
  |             this: read (group:legal-eu, member)
  |               group:legal-eu#member@user:alice      -> MATCH, allow
  |
  +- computed(editor): check(doc:42, editor, alice)   same object, no depth charge
  |
  +- ttu(parent, viewer): for each F with doc:42#parent@F
                            check(F, viewer, alice)             depth+1
```

An allow returns the path that produced it, in order:

```
doc:42#viewer@group:legal#member
group:legal#member@group:legal-eu#member
group:legal-eu#member@user:alice
```

Three lines an administrator can read, and three lines a test can replay one edge at a time. Absence of any such path is a denial. Not "unknown", not "allow while we figure it out" — default deny is the base case of the recursion, not a policy bolted on top.

**Cycles, which are not hypothetical.** Two administrators, eighteen months apart, write `group:leads#member@group:seniors#member` and `group:seniors#member@group:leads#member`. Neither is wrong on its own. Together they are a loop, and a naive evaluator recurses until the stack dies. The fix is not clever: keep the set of `(object, relation)` pairs currently on the evaluation stack, and when a branch revisits one, **terminate that branch and return deny**. Not an exception — an exception turns an odd group graph into a 500 on the search endpoint. Not a grant — if the only route to the principal goes through a loop, there is no derivation.

**A depth bound, resolving toward deny.** Even without cycles, group graphs can be deep, and a request must have a worst case. Expansion stops at a fixed depth and the stop **denies**. Truncating toward allow is the bug that only appears at the one customer with a twenty-level hierarchy, which is to say at the customer who will notice.

**Auditability is a requirement, not a nicety.** A permission system that returns a bare boolean cannot be debugged, cannot be tested at the edge level, and cannot answer the only question an administrator ever asks, which is "why not?". So a denial carries its failed branches too — which tuples were read, where a cycle was hit, where the depth limit stopped the walk.

**This is a known pattern, honestly.** Google published Zanzibar in 2019; it backs Drive, Calendar, YouTube and more, at a scale measured in trillions of tuples. The open implementations are real and mature: SpiceDB from AuthZed, OpenFGA (originated at Auth0/Okta, now a Cloud Native Computing Foundation project), Permify, and Ory Keto. Sightline reimplements a deliberate subset of the model rather than importing one of these, because the point of the project is to show the evaluator's behaviour under adversarial tests — you cannot plant a bug in a dependency and measure whether your test suite catches it. In production you would run SpiceDB or OpenFGA and spend your effort on the retrieval side.

### Where it goes wrong

**The new-enemy problem.** Alice removes Bob from a group, then moves a sensitive document into that group's folder. If any part of the system evaluates against a snapshot older than the removal, Bob reads the document. Zanzibar's answer is *zookies*: consistency tokens carried with the content, so a check can demand a policy view at least as fresh as the data. Most homegrown implementations skip this entirely and are quietly wrong under concurrency. Any caching layer you put in front of a check re-creates this problem exactly.

**Fan-out.** `check()` on a document shared with a 200,000-member group, where the principal is not a member, has to fail *fast*, and the naive evaluator reads 200,000 tuples to find out. Zanzibar's answer is a separate flattened index (Leopard) for deeply nested set membership. A small system's answer is a bound on work plus the observation that most checks resolve on the first direct tuple.

**Migration is the real project.** Converting an existing access-control list estate into tuples is where breaches come from. A 2% error rate in a schema migration is a rounding error; a 2% error rate in a permission migration is 800 documents visible to the wrong people, and nothing errors.

**Deny rules are a trap.** Intersection and exclusion ("viewer AND NOT suspended") look like obvious features and are the hardest part of the model, because a deny that does not deny is undetectable from the outside. A half-implemented exclusion is worse than no exclusion, since people will rely on it.

**Performance panic leads to denormalisation.** Someone times `check()`, finds 40 ms at p99, and proposes caching decisions by user id. That cache is now the authority, and it is an authority with no revocation story. If you must precompute — and for search you must — precompute something that a change to the graph invalidates automatically. That is the next chapter.

### In this project

`src/sightline/types.py` defines the three nouns and nothing else: `ObjectRef`, `PrincipalRef` (which carries an optional `relation`, so it doubles as a userset), and `Tuple_`. There is no `tenant_id` field anywhere in the codebase and no `is_public` boolean; the module docstring says so explicitly, because the absence is the design.

`src/sightline/authz/tuples.py` is storage only. It keeps two indexes with genuinely different access patterns: a **forward** index keyed on `(object, relation)` for `check()`, and a **reverse** index keyed on the principal for the compiler, which asks "what does this subject hold?" without knowing the relation in advance. The reverse index compares parsed principals rather than string prefixes — matching on a prefix is planted mutant **M9**, where `group:legal` swallows `group:legal-interns` and an intern reads the merger memo. Both an in-memory backend and a SQLite backend implement one `TupleStore` protocol; the SQLite one increments the policy epoch inside the same transaction as the tuple write, and moving it outside is planted mutant **M15**.

`src/sightline/authz/check.py` is the authority, and it is a separate file on purpose — an index that decides things is an index nobody audits. `MAX_USERSET_DEPTH = 16`, split into `MAX_OBJECT_DEPTH = 8` and `MAX_SUBJECT_DEPTH = 8`, with a module-level `assert` that the two halves still sum to the whole, so a careless edit fails at import rather than at 3am. The split exists because the index stamps tokens by walking the object side while the compiler walks the subject side; if both could spend the full budget, a token match could imply a 16-deep path that `check()` itself would refuse, and the compiler would be *more* permissive than the authority. Truncating toward allow instead of deny is planted mutant **M5**.

Cycles return a denial and record `cycle: doc:42#viewer` in the diagnostics. Memoisation is per call and thrown away afterwards, because a cache that outlives the call is a cache that can serve a revoked answer. The memo is also depth-aware: a complete deny is reusable anywhere, but a complete allow is only reusable if its derivation still fits the remaining budget, otherwise a shallow success smuggles a too-deep path into a context that should have refused.

`Decision.why` carries the derivation, and `explain()` returns it alongside the full expansion tree for `/v1/explain`. That endpoint is administrator-only, because naming the groups on a document to someone who cannot see the document is its own small disclosure.

### Interview questions

**1. What is a relation tuple, and why is that shape useful?**

It is one fact, written `object#relation@principal` — for example `doc:42#viewer@user:alice`. The useful part is that the principal can itself be a userset, an object plus a relation like `group:legal#member`, which means "whoever holds member on group:legal". Because both sides of a tuple can be a userset, group nesting falls out of the data model rather than being a feature somebody implements. It also means permission is stored as edges rather than as attributes of a row, so a document's visibility is derived rather than read off the document.

**2. Why does role-based access control with a tenant column stop working?**

It works fine while permission is a property of the document. It stops the moment permission is a property of a path — this document is in that folder, that folder is shared with a team, and you are in a sub-team of that team. To answer that with columns you need a recursive join over a membership table plus a second recursive join over containment, and then you need to express the result as a filter that your search index can evaluate, which it cannot. The `is_public` flag is the worse half: it encodes an assumption about obscurity that stops being true the instant something indexes the corpus.

**3. How does `Check()` actually evaluate?**

It is a depth-first search over the tuple graph, guided by the namespace's rewrite rules. For a given object and relation it expands the rewrite — direct tuples, other relations on the same object that imply it, and inheritance from parent objects — and for any tuple whose subject is a userset it recurses into that userset. It short-circuits on the first derivation, returns the ordered list of tuples that produced it, and returns deny if no derivation exists. Absence of a tuple is a denial, which means default-deny is the base case of the recursion rather than a rule layered on top.

**4. Group graphs contain cycles. What do you do?**

Keep the set of `(object, relation)` pairs currently on the stack, and terminate a branch that revisits one. It returns deny for that branch and records a note saying a cycle was hit. Two things it must not do: raise, because then a badly administered group graph becomes a 500 on the search path, and grant, because if the only route to the principal is a loop then no derivation exists. Cycles are not exotic — two administrators writing sensible tuples eighteen months apart is all it takes.

**5. Why does a permission decision need to carry its derivation?**

Because a boolean is undebuggable. The question an administrator asks is never "is Alice allowed", it is "why is Alice allowed" or, far more often, "why is Alice not allowed", and neither is answerable from a `false`. Carrying the ordered list of tuples means an auditor can replay it edge by edge, a test can assert on the exact path rather than the outcome, and the admin console can render it. I also make the *denial* carry its failed branches, including depth-limit and cycle markers, because those are the two causes that look identical to "no permission" from outside.

**6. You cap expansion depth. Which way does the cap resolve, and why does it matter?**

Toward deny, always. A depth limit that truncates toward allow is a bug that never shows up in testing, because test fixtures have shallow group graphs, and shows up at the one customer with a twenty-level hierarchy inherited from an acquisition. The cost of denying is a false negative: somebody who should see a document does not, complains, and you find out. That asymmetry — a false deny is a support ticket, a false allow is an incident — is the reason every ambiguous case in a permission system should resolve the same direction.

**7. Zanzibar is a known design. Why build one rather than adopt SpiceDB or OpenFGA?**

For production I would adopt one. They handle the parts that are genuinely hard at scale — the consistency tokens for the new-enemy problem, the flattened index for deeply nested sets, the operational surface. I built a subset here because the project's headline metric is a mutation kill rate: I plant deliberate permission bugs and measure how many the test suite catches, and you cannot plant a bug inside a dependency. The subset is honest about what it omits — no intersection, no exclusion, no zookies — and each omission is written down as a decision rather than discovered as a gap.

**8. Describe the new-enemy problem and how you would defend against it.**

Alice removes Bob from a group and then moves a sensitive document into that group's folder. If any component evaluates against a view of the policy older than the removal, Bob reads the document, and every component has been individually correct. Zanzibar solves it with zookies, consistency tokens carried with the content so a check can require a policy snapshot at least as new as the data. In this system I take a cheaper route that fits the architecture: a monotonic policy epoch bumped in the same transaction as every write, stamped onto anything precomputed, and a final recheck against the live store that has no cache in front of it at all. That does not give me Zanzibar's guarantees across a distributed cluster — it gives me the property that a stale artefact can only lose results, never leak them.

---

## The permission compiler

### The short version

Checking one document against the permission graph is cheap. Checking 40,000 of them, once per search, is not, and you cannot put 40,000 document identifiers into a query either. So before searching, the system compiles a person into a small handful of opaque tokens plus a decision about *how* to filter, chosen from how much of the corpus that person can see. The compiled plan is a cache of the permission graph, and the design goes to some length to make sure it can never be more permissive than the graph it came from.

### The intuition

Think about a conference with 40,000 sessions and a badge desk.

The naive design: when you arrive, the desk looks up every session you are entitled to attend and prints the list on your badge. It is correct. It is also a 300-page badge, it has to be reprinted whenever any session changes, and the door staff have to search it.

The design that actually works: the desk prints three coloured stripes on your badge — speaker, sponsor, workshop track B. Each door has a small sign saying which stripes it accepts. Nobody enumerates anything. A person's badge has three stripes whether they can attend four sessions or four thousand.

Now the crucial detail, the one that is easy to miss. The stripes are **roles, not people**. The door sign says "workshop track B", not "Alice, Bo, Chen, Dee". When a new person joins track B, you print one badge. You do not visit 400 doors and rewrite the signs.

That is the entire grant-token scheme, and here is the arithmetic that makes it non-negotiable. Suppose `group:legal` grants access to 8,000 documents, averaging 20 chunks each — 160,000 vectors in the index. If chunks were tagged with the user identifiers of everyone allowed to read them, adding one person to Legal would rewrite 160,000 index payloads. Removing them would rewrite 160,000 more. With group-derived tokens, both are **zero vector writes**. The only thing that changes is that person's compiled plan, and that is recomputed on their next query.

The other half of the compiler is choosing *how* to filter. Two people, same system:

* A contractor can see 40 documents out of 40,000. The best filter is the list of 40 identifiers. It is tiny, exact, and the search can be an exact scan of those 40 vectors — faster and more accurate than any approximate index.
* The chief financial officer can see 39,900 of 40,000. Sending 39,900 identifiers is absurd. Filtering by token overlap during the search is right.

Same code path, same corpus, different plans, and the difference is three orders of magnitude of cardinality. Picking one strategy for everyone means being wrong for most people.

### How it actually works

Compilation has three steps: walk the graph backwards, count, and choose.

**Step 1: the reverse walk.** `check()` runs from the object toward the person. The compiler runs the other way: from the person, up through everything they hold, transitively. It is breadth-first over the reverse tuple index, so hop counts are minimal and the depth bound means what it says:

```
user:alice
  hop 1: group:legal-eu#member          (alice is a member)
  hop 2: group:legal#member             (legal-eu is a member of legal)
  hop 2: folder:contracts#viewer        (legal is a viewer of that folder)
  hop 3: folder:q3#viewer               (inherited)
```

The result is a **closure**: every userset the principal holds, with its hop count. It also closes over implication in reverse — if `viewer` is defined as a union containing `computed(owner)`, then holding `owner` on an object means holding `viewer` on it, and without that step an owner's plan would miss their own documents.

**Step 2: derive tokens.** Each userset in the closure becomes one grant token:

```
token = "gt_" + blake2b(f"{subject}|{relation}", digest_size=16, key=K).hexdigest()

group:legal#member  + viewer  ->  gt_9f2c...  (32 hex characters)
```

Three properties matter. It is **keyed**, so somebody who can read the vector collection does not learn that a group called `legal-acquisitions` exists. The **relation is part of the input**, so a token granting `viewer` cannot be replayed as a token granting `editor` if a second filtered relation is ever added. And it is **stable** for a given key, so the ingest side and the query side derive the same string without coordinating.

Ingest does the mirror-image walk: it expands a document's `viewer` relation over the object side and stops at the group edge, stamping one token per granting userset. The two halves meet in the middle at the userset, and neither side ever writes a person into the index.

Sizes: a typical employee in twelve groups, nesting three deep, closes over roughly thirty usersets — thirty tokens, about 1 KB in a query payload. The identifier list for the same person could be 40,000 strings.

**Step 3: count, then choose.** The compiler walks forward from the closure to find which document identifiers it actually reaches, stopping at a hard cap of 50,000, at which point the answer is "lots" — which is all the strategy chooser needs, and the walk stops costing query latency. Then:

| Strategy | Condition | What the index does | Why |
|---|---|---|---|
| `ENUMERATE` | `n == 0`, or `n <= 512` | `WHERE id IN (...)` | Small enough to send. Exact, no approximation anywhere. |
| `EXACT_SCAN` | `512 < n <= 4096` | Fetch those vectors, brute-force cosine | Below a few thousand, a full scan beats a graph walk *and* is exact. |
| `GRANT_TOKENS` | `n > 4096`, or the walk was truncated | Payload filter on token overlap, evaluated during traversal | The identifier list is too big to send; the token set is not. |
| `UNFILTERED` | `n >= corpus_size` | No filter | Nothing to exclude, so do not pay for the filter. |

Note the zero case. Nobody-can-see-anything compiles to `ENUMERATE` over an empty identifier list, not to `UNFILTERED` and not to "no filter". Empty means nobody, always. Treating an empty token set as "no predicate" is one of the classic fail-open bugs and it is planted here as a named mutant.

**One definition of what a plan admits.** Every backend — a Qdrant payload filter, a SQL `WHERE`, a numpy mask — must agree with a single reference function:

```python
if plan.strategy is UNFILTERED:            return True
if strategy in (ENUMERATE, EXACT_SCAN):    condition = object_id in plan.explicit_ids
if strategy is GRANT_TOKENS:               condition = bool(plan.grant_tokens & chunk_tokens)
return bool(conditions) and all(conditions)
```

Two mistakes live in those four lines. **Within** one chunk's token set the test is OR — any single matching grant suffices. **Across** a plan's conditions it is AND. Swapping those is a mutant. And `bool(conditions)` is what stops a plan with no conditions at all from falling through to allow.

**Why the thresholds must be measured.** 512 and 4096 have the right shape and, until the selectivity benchmark runs on the reference machine, the wrong values. The crossover depends on things the compiler cannot see: the Hierarchical Navigable Small World (HNSW) graph parameters `m` and `ef_search`, vector dimensionality, whether the payload index is a keyword index or a bitmap, and how the filter interacts with graph traversal. What is *not* negotiable is the ordering — a filter that is cheap to express beats a filter that is cheap to evaluate, but only while the identifier list stays small enough to send. Anyone who ships a hardcoded threshold and calls it tuned is guessing in public.

**Caching, keyed on the epoch.** Compiling is a graph walk, and users ask several questions in a row, so plans are cached. The key is `(principal, policy epoch, relation, namespace)`, and the epoch is what makes the cache safe: every policy write bumps a monotonic counter, so every existing key becomes unreachable at once and stale entries age out of the least-recently-used list. **No invalidation logic exists, because none is needed.** Keying on the principal alone with a time-to-live is the mutant, and it is also how this bug happens in the wild — the time-to-live looks short in a meeting, and it is five minutes of a revoked user reading documents.

One ordering detail: the epoch is read *before* the walk, not after. Reading it afterwards would let a write that landed mid-walk produce a plan stamped with the new epoch while containing the old policy — fresh-looking and wrong. Stamping the older epoch can only cost a recompile.

### Where it goes wrong

**The compiler out-permitting the authority.** This is the failure that matters. The plan is a projection of `check()`, and if it admits one document `check()` would deny, that is a breach, not a bug — and "the recheck will catch it" is exactly the reasoning the whole product exists to refuse. The defence is a differential oracle: compile a plan, run `check()` over every document, and assert the plan's admitted set is a subset of the authority's.

**Token explosion.** Someone in 200 groups gets 200 tokens, and a payload filter with 200 terms is not free. The honest mitigations are capping, and noticing that a principal with that many grants is usually close to `UNFILTERED` anyway.

**Truncation costs recall, quietly.** Grants requiring more nesting than the subject-side depth budget are visible to `check()` and invisible to the index. That is a false *deny* — a quality bug, not a breach — but it looks exactly like "the document does not exist", so it must be measured rather than waited for.

**Key rotation forces a full reindex.** Tokens are keyed hashes; change the key and every stamped token is wrong. That cost belongs in a runbook, discovered in writing rather than at rotation time.

**"98% is basically everything."** A specification that says a principal seeing 98% of the corpus compiles to `UNFILTERED` is proposing a plan that admits 2% of the corpus it should not. At 40,000 documents that is 800 documents, and they are the 800 nobody wrote a tuple for.

**The cardinality estimate is not free.** A reverse walk over a large graph on the query path is real latency. Capping it and caching the result is what makes it affordable, which is why the cap and the cache are part of the design rather than optimisations added later.

### In this project

`src/sightline/authz/compile.py` holds all of it. `principal_closure()` is the breadth-first reverse walk, bounded by `MAX_SUBJECT_DEPTH = 8`; `reachable_objects()` is the forward walk with `MAX_ENUMERATION = 50_000`; `compile_plan()` puts them together and returns a `FilterPlan` from `src/sightline/types.py`, carrying the strategy, the tokens, the explicit identifiers, the epoch, and the estimated cardinality.

Thresholds are module constants — `ENUMERATE_MAX = 512`, `EXACT_SCAN_MAX = 4096`, `UNFILTERED_MIN_RATIO = 1.0` — with a comment above them stating plainly that they are not measurements and stay wrong until `eval/selectivity.py` runs on the reference machine. `UNFILTERED_MIN_RATIO` is 1.0 rather than 0.98, in explicit disagreement with `docs/FRD.md` FR-5, and it additionally requires the caller to supply the corpus size, because the tuple store only knows about documents somebody wrote a tuple for and the dangerous document is precisely the one nobody did.

`derive_grant_token()` uses keyed blake2b with a 16-byte digest. The development key is hardcoded and published in the source, on the grounds that a secret with a default is not a secret and pretending otherwise is worse than saying so; deployments set `SIGHTLINE_GRANT_KEY`.

A plan only carries the identifiers its own strategy reads — a 40,000-element list nobody consults is payload, latency, and a disclosure waiting for a debug log. `plan_admits()` is the single definition every backend's filter is conformance-tested against, and it names its two mutants in the docstring: **M2** (combining a plan's conditions with OR) and **M6** (treating an empty token set as no filter).

`PlanCache` is a locked least-recently-used map keyed on `(principal, epoch, relation, namespace)`, with a belt-and-braces staleness check on read; keying without the epoch is mutant **M8**. `PlanCompiler` binds a store, a cache and a corpus size, and is the object the HTTP layer injects, so the epoch is read in exactly one place.

The design decision behind all of it is written up as `docs/adr/0002-grant-tokens-not-user-ids.md`: tag chunks with groups, never with people. One consequence is worth stating precisely, because it sounds like an exception and is not. Tokens for direct per-user grants, `doc:42#viewer@user:alice`, *are* derived from `user:alice`. The ban is on expanding group *membership* into user identifiers, because membership churns. A direct grant tuple is a change to that document's own access list, so the vector write it causes is one the design already pays for.

And the compiled plan is never the last word. `src/sightline/authz/recheck.py` re-derives every surviving candidate against the live tuple store before anything reaches the model, with no cache in front of it and no flag to disable it. The compiler exists to make the candidate set small and cheap. It does not exist to be trusted.

### Interview questions

**1. Why compile permissions at all? Why not check each result?**

You do check each result — that is the final recheck and it is not optional. The compiler solves a different problem, which is that the search itself needs to know what to look at. If you search the whole corpus and check afterwards, a person who can see 1% of the corpus gets, in expectation, 0.1 permitted results in a top-10, so the model receives nothing and answers "I don't know". The compiler produces a filter the index can apply *during* the search so the candidates that come back are mostly permitted. It is a recall device first and a performance device second.

**2. What is a grant token and why is it derived from groups?**

It is an opaque string derived from one granting userset plus one relation — a keyed hash of something like `group:legal#member|viewer`. Chunks are stamped with the tokens of whatever grants access to their document; a person's compiled plan carries the tokens of the usersets they hold; the filter is set intersection. It is derived from groups rather than people because membership churns constantly and document access lists do not. If I tagged chunks with user identifiers, adding one person to a group with 8,000 documents at 20 chunks each would rewrite 160,000 index payloads. With group tokens it rewrites zero.

**3. Walk me through the four strategies and when each fires.**

Cardinality decides. Zero or up to a few hundred permitted documents: enumerate the identifiers directly in the query — small, exact, no approximation. A few hundred to a few thousand: exact scan of that subset, because brute-force cosine over 4,000 vectors beats an approximate graph walk and is exactly correct. Above that: filter on grant tokens, evaluated inside the index during traversal, because the identifier list is now too big to send but the token set is still tiny. And if the person can see the entire corpus, skip the filter. The zero case is the one people get wrong — it compiles to an enumeration of nothing, never to "no filter".

**4. Where did 512 and 4096 come from?**

Right now, from nowhere defensible — they have the right shape and unverified values, and the source says so in a comment. The real crossover depends on the HNSW graph parameters, the vector dimensionality, the payload index type, and how the filter interacts with traversal, none of which the compiler can see. The benchmark that sets them sweeps selectivity from 0.1% to 100% and measures latency and recall for each strategy at each point, and the thresholds are where the curves actually cross on the reference hardware. I would rather ship a number labelled "not measured" than one labelled "tuned" that nobody measured.

**5. How do you invalidate a compiled plan?**

I do not. The cache key contains a monotonic policy epoch that every write bumps inside the same transaction as the tuple change, so after any policy write every existing key is unreachable and the old entries age out of the LRU. Invalidation that requires remembering to call it is invalidation that does not happen, and a time-to-live is worse — it is a fixed window during which a revoked user still has access, and it always looks short enough in the meeting where someone proposes it. One subtlety: the epoch is read before the graph walk, not after, so a write landing mid-walk produces a plan stamped with the older epoch, which costs a recompile instead of serving stale policy as fresh.

**6. What stops the compiler from being more permissive than `Check()`?**

Two things, one structural and one empirical. Structurally, the depth budget is split: the object-side expansion at ingest and the subject-side closure at compile time each get half of the total, and there is an assertion that the halves still sum. Without that split, a token stamped after eight object hops could match a plan built after eight subject hops, implying a sixteen-hop path — a path `check()` would refuse — and the index would out-permit the authority. Empirically, a differential oracle compiles a plan and compares its admitted set against `check()` over every document; a plan admitting something `check()` denies is reported as a breach, and a plan missing something `check()` allows is reported as a recall bug. Those are different severities on purpose.

**7. Your plan only admits documents the person can see, but the recheck still runs. Is that not redundant work?**

No, and the fact that it looks redundant is why it gets removed by well-meaning people. The plan is derived from a snapshot of the graph and the index was stamped at some earlier time; both can be stale, and staleness in a filter is invisible. The recheck is the only component that reads live state per candidate, and it costs one graph walk per surviving hit — tens of candidates, not tens of thousands, precisely because the filter did its job. The division of labour is the point: the filter is allowed to be approximate and fast, the recheck is allowed to be slow and is not allowed to be wrong. If I ever had to drop one, I would drop the filter and eat the recall loss.

**8. A specification you inherit says a user who can see 98% of the corpus should skip filtering. Do you implement it?**

No, and I would push back in writing. That rule says a plan may admit 2% of the corpus it should not, which at 40,000 documents is 800 documents, and by the oracle's own definition that is a false allow. The argument for it is that the recheck catches them anyway — which is true and is exactly the reasoning the product exists to refuse, because it makes the last line of defence the first. There is a second problem: to know you are at 98% you need the corpus size, and the tuple store only knows about documents somebody wrote a tuple for. The dangerous document is the one nobody did, so the denominator is wrong in the unsafe direction. I set the threshold at 100% and require the caller to pass the corpus size explicitly.
