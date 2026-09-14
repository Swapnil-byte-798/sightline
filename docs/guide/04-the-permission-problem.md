# Part 4 — The permission problem

---

## Why permission filtering breaks vector search

### The short version

Search finds the passages closest in meaning to a question, across the whole corpus. Then
someone points out that most employees are not allowed to read most of the corpus, so you
drop the results they cannot see. What is left is often nothing. The assistant then says
"I do not know" about a document sitting in the asker's own folder, no error is logged
anywhere, and everyone concludes the AI is not very good.

### The intuition

A company has one million passages of text indexed — call it 250,000 documents chopped
into chunks of a few hundred words each. A salesperson named Dana works on the Denver
account. She can read the company handbook, her own team's folder, and the shared space
for the deals she is on. Add it up and she is allowed to see about 20,000 of those one
million chunks. Two per cent.

Dana asks: "what discount can I offer Denver on renewal?"

The retriever embeds her question and returns the ten closest chunks in the whole index.
Those ten are genuinely the best ten. Seven come from the pricing committee's folder.
Two come from a finance model. One comes from the shared deal space Dana is actually on.

Then the permission filter runs, and nine of the ten disappear.

Dana gets one passage. It is a meeting note that mentions Denver but says nothing about
discounts. The model, behaving correctly, writes: "I could not find information about
renewal discounts for the Denver account."

Now change one number. A contractor on a six-week engagement can see 0.1% of the corpus —
1,000 chunks. For him, the expected number of permitted results in a global top ten is
0.01. He gets nothing, for almost every question he asks, including questions whose answer
is in the folder he was given access to on his first day.

Work the arithmetic out, because it is worse than it feels. If the passages Dana may see
were scattered at random through the ranking, then each of the ten slots is permitted with
probability 0.02:

```
expected permitted results in the top 10 = 10 x 0.02 = 0.2
probability of getting at least one      = 1 - 0.98^10 = 18.3%
probability of getting all ten           = 0.02^10     = about 1 in 10^17
```

So four queries in five return literally nothing. Not a bad answer — no answer, from a
system that is working exactly as designed.

And here is the part that makes this a product problem rather than a bug report: **nothing
in that sequence is an error.** The vector index returned results in 25 milliseconds. The
permission filter did its job perfectly and leaked nothing. The model was honest about
what it did not know. Every component reported success. There is no stack trace, no alert,
no 500. The failure is invisible from the inside, and the only signal is that people stop
using the tool.

### How it actually works

Some definitions, each used for the rest of the guide.

A **corpus** `C` of `N` chunks. A **principal** `p` — the person or service account
asking. The **permitted set** `A_p`, the subset of `C` that `p` is allowed to read. And
the number that governs everything in this chapter:

> **Selectivity** (also called permission density) is `s = |A_p| / N` — the share of the
> corpus one principal may see.

Dana's selectivity is 0.02. The contractor's is 0.001. An administrator running the demo
has `s = 1.0`.

**What the correct answer is.** Given a query vector `q`, the right result is the top `k`
chunks *of the permitted set*, ranked by similarity:

```
ideal(q, p, k) = top_k( A_p , similarity to q )
```

That is the definition of correct, and it is worth writing down because the wrong thing
looks so much like it.

**What post-filtering computes** is a different set:

```
returned(q, p, k) = top_k( C , similarity to q )  ∩  A_p     (then truncated to k)
```

Search everything, then drop. These two expressions agree when `s = 1.0`, which is
precisely the configuration every demo runs under, and diverge as `s` falls.

**Recall at k** is the measure of the gap: of the `k` results that should have come back,
how many did. `recall@k = |returned ∩ ideal| / k`.

Under the assumption that permitted chunks are scattered uniformly through the global
ranking, the expected size of `returned` is `k x s`, so:

```
recall@k  ≈  min(1, s)
```

Post-filter recall is approximately the selectivity itself. At `s = 0.02`, recall@10 is
about 0.02. This is not a degradation; it is a collapse.

**Overfetching** is the obvious patch: retrieve `f x k` candidates instead of `k`, hoping
`k` survive. Expected survivors are `f x k x s`, so to expect ten survivors you need
`f ≥ 1/s`:

| Selectivity `s` | Who that is | Overfetch `f` needed to expect 10 | Candidates to score |
|---|---|---|---|
| 1.0 | Admin running the demo | 1 | 10 |
| 0.50 | Handbook-level content | 2 | 20 |
| 0.10 | A whole engineering org | 10 | 100 |
| 0.02 | Dana | 50 | 500 |
| 0.001 | Six-week contractor | 1,000 | 10,000 |

"Expect ten" is not "get ten". The survivor count is `Binomial(f x k, s)`, so at `f = 1/s`
the distribution is centred on ten and lands below it roughly half the time. Reliability
needs two to three times more again, and the latency cost of overfetch is linear while the
recall gain is not.

**Why low selectivity is normal, not an edge case.** The instinct is that `s = 0.02` is a
pathological test fixture. It is the median employee. Consider a 20,000-person company
with 250,000 documents:

| Content class | Share of corpus | Who can read it |
|---|---|---|
| Company-wide (handbook, policies, all-hands notes) | 3–5% | Everyone |
| One department's shared space | 2–4% | That department, maybe 1/20 of staff |
| One project or deal space | under 0.1% each | Five to fifty people |
| Executive, legal, people-team, finance | 10–15% | A few dozen people |
| Everything else — other departments, other projects | the remaining 70%+ | Not you |

An individual contributor reaches maybe 2% to 8%. A new joiner is lower. A contractor is
lower again. The people with high selectivity are a handful of executives and the IT
administrator who set the system up and ran the pilot.

That last sentence is the trap. The pilot is run by the highest-selectivity principals in
the building, at the one point on the curve where post-filtering is exactly correct.

**One honest complication.** The random-scattering assumption is a simplification, and it
cuts both ways. Permitted chunks are not independent of relevance: Dana asks about Denver,
and the Denver deal space is hers, so her permitted documents are over-represented among
the relevant ones compared with a uniform sample. Real recall is therefore better than
`min(1, s)` for questions about a principal's own work.

It is not better for the questions people actually find valuable — "what is our policy
on", "has anyone done this before", "what happened with that customer last year" — which
are exactly the cross-cutting questions where the answer lives outside your own folder but
inside something you are nonetheless entitled to read. And the size of the correlation is
a property of the customer's organisation, not of the algorithm, so it cannot be assumed
during design. It has to be measured on a corpus with realistic group structure.

### Where it goes wrong

**Confusing "did not leak" with "correct".** A post-filtered store never returns a
forbidden chunk. It is a safe store. It is also a useless one at low selectivity. A store
that returns zero results for every query is perfectly safe too. Safety is a floor, not a
success criterion, and teams that measure only leaks will ship this and call it done.

**Measuring recall averaged over principals.** The mean is dominated by the
high-selectivity accounts, who are a small minority of users and a large majority of test
fixtures. Recall has to be reported *as a function of selectivity*, broken into buckets,
because the shape is the finding.

**Padding results back up to `k`.** When the result list comes back short, the tempting
fix is to top it up from the unfiltered candidate pool. This converts a silent recall
failure into an actual leak. In Sightline it is a planted mutation, M7, and there is no
code path in the baseline store that can do it.

**Raising `k` until it works.** Overfetch moves the collapse to the right on the curve and
never removes it, at linear cost in latency and in tokens if the candidates are reranked.

**Counting selectivity in documents when search ranks chunks.** A principal permitted on
500 documents might hold 1,500 chunks or 150,000, depending on document length. Every
threshold in a query planner is a threshold on chunks.

**Assuming the problem is the model.** The observable symptom is a language model saying
"I do not know". Every instinct points at the model: better prompt, bigger context window,
a different vendor. None of it helps, because the evidence never reached the prompt. A
retrieval failure wearing a generation costume is the single most expensive
misdiagnosis in this field.

### In this project

Sightline implements the wrong approach on purpose, as the baseline arm of the evaluation.
`/Users/cook/sightline/src/sightline/store/postfilter.py` retrieves the global top-`k`
ignoring permissions and drops afterwards. Its docstring is blunt about what it is for,
and the module carries three guards so it cannot drift into the serving path: a
`NOT_FOR_PRODUCTION` flag that a test asserts on, an `assert_not_serving()` call that
fires at import time if `uvicorn` is in `sys.modules`, and an override environment
variable, `SIGHTLINE_ALLOW_POSTFILTER`, that exists so that turning the guard off is a
greppable, deliberate act.

The class records a `PostFilterTrace` on every search with four fields — `requested_k`,
`retrieved`, `returned`, `dropped` — plus a `shortfall` property. Those counters, plotted
against selectivity, are the headline chart of the project. Its `overfetch` constructor
argument exists for one reason: so the evaluation can show what the usual production patch
does and does not buy.

Selectivity is a first-class quantity in the type system, not a benchmark parameter.
`FilterPlan` in `/Users/cook/sightline/src/sightline/types.py` carries
`estimated_cardinality`, and the execution strategy is chosen from it. Functional
requirement FR-14 gates `recall@10 ≥ 0.95` against an exact oracle at permission densities
of 0.1%, 1%, 10%, 50% and 100% — five densities, because density is the axis along which
this fails — with the post-filter arm's curve published next to it, collapse included.

One design decision makes it possible to discuss any of this honestly. Because
`VectorStore.search()` returns `UncheckedHit` and only `recheck()` against the live
permission store produces a `Hit`
(`/Users/cook/sightline/src/sightline/store/base.py`, and architecture decision record
0001), the security property does not depend on the retrieval strategy being right. A
stale or badly-filtered index loses results; it cannot leak them. So everything in this
chapter is an availability argument, not a security argument, and it can be measured and
published without anyone's instinct to hide a bad number kicking in.

### Interview questions

**1. Explain in one minute why adding permissions to a RAG system is harder than adding a
`WHERE` clause.**

Retrieval-augmented generation — search your documents, put the best passages in the
model's prompt — depends on the best passages actually being retrieved. A `WHERE` clause on
a database query narrows the rows, and the rows you get back are exactly the rows that
match. A vector search does not work that way: it returns a fixed-size top-`k` ranked by
similarity over whatever set it searched. If you search everything and then remove the
forbidden results, you do not get the top ten of what the user may see, you get whatever
scraps of the global top ten happen to be permitted. At 2% selectivity that is zero or one
result. The filter is correct and the answer is still wrong.

**2. What is selectivity and why does it matter more than corpus size?**

Selectivity is the fraction of the corpus one principal is allowed to see. Corpus size
tells you how long a search takes; selectivity tells you whether the search finds
anything. For a post-filtered system, recall at `k` is roughly the selectivity itself, so
a user at 2% gets about 2% of the results they should. It matters more than corpus size
because the cost of size is linear and fixable with an index, while the cost of low
selectivity is a collapse that no amount of indexing addresses. It is also the number
nobody measures, because the person running the pilot is usually at or near 100%.

**3. A user says the assistant cannot find a document that is in their own folder. Walk me
through your diagnosis.**

First I check whether the chunk is in the index at all, because a missing ingest looks
identical from the outside. If it is indexed, I run the query against an exact oracle over
that user's permitted set and see where the chunk ranks — if it ranks first there and is
absent from production, that is a retrieval-strategy problem, not an embedding problem.
Then I look at the recorded strategy and the drop counts on that response: a large
`dropped` count with a small `returned` count is post-filter collapse, and a zero
`returned` with a zero `dropped` is an empty permitted set, which is a permission-compiler
bug instead. The thing I try hard not to do first is change the prompt or the embedding
model, because both are unfalsifiable fixes that occasionally appear to work.

**4. Why does nothing log an error when this happens?**

Because nothing failed. The index returned results inside its latency budget, the filter
removed exactly the chunks it was meant to remove, the model wrote an honest sentence
about not knowing the answer, and the HTTP status was 200. There is no component in the
chain whose contract was violated. That is why you have to manufacture the signal
yourself: record how many candidates were retrieved, how many survived filtering, and what
share of `k` you actually returned, then alert on the shortfall. If you do not instrument
the gap between "asked for ten" and "returned one", the system will quietly be bad for
months, and what you will hear is that people find it unhelpful.

**5. Someone proposes fixing it by retrieving 1,000 candidates instead of 10. Respond.**

It helps, it is bounded, and it is the right first move if you need something this week.
The arithmetic is that expected survivors are `f x k x s`, so overfetch by `1/s` and you
expect `k` back — at 2% that is a factor of fifty, and that is expectation, not guarantee,
because the survivor count is binomial and lands short about half the time at exactly
`1/s`. The cost is linear in the overfetch factor and the benefit saturates, so at 0.1%
selectivity you are scoring 10,000 candidates to reliably keep ten, and any reranker
downstream now costs a hundred times more. It also fails silently in exactly the same way
for the users with the least access, who are the ones most likely to be contractors or new
joiners. So I would take it as a stopgap and not as the design.

**6. How do you measure recall here without an index to measure against?**

You need an oracle that is exact by construction, which means brute force: score every
vector in the permitted set and take the true top `k`. It is slow, which is fine, because
it never serves traffic. Sightline has one as a separate store with a test asserting it is
unreachable from the serving application. Measuring approximate retrieval against another
approximate index means measuring two unknowns at once, and the errors do not cancel —
they conveniently agree, which is worse.

**7. Is low selectivity a real-world condition or a benchmark artefact?**

It is the median case and I would push back on anyone who calls it an edge case. In a
20,000-person company the corpus divides into a few per cent that is genuinely
company-wide, a department's space that maybe one in twenty people can read, and a long
tail of project spaces with a few dozen readers each. An individual contributor reaches
somewhere between 2% and 8% of chunks; a contractor is an order of magnitude below that.
The people above 50% are executives and the administrator who configured the pilot. Every
system I have seen that ships post-filtering was validated by the administrator.

**8. Where does the "permitted documents are also the relevant ones" argument break down?**

It is a real effect and it makes the naive arithmetic pessimistic — Dana asks about Denver
and the Denver folder is hers, so her permitted chunks are over-represented among the
relevant ones. I would not design around it for two reasons. First, the correlation is
strongest for questions about your own work, which are the questions you least need help
with, and weakest for the cross-cutting ones — precedent, policy, what happened last year
— that are the reason anyone buys the product. Second, the strength of the correlation is
a property of how a particular customer organises their permissions, so it varies between
deployments and cannot be validated once. The way to handle it is to build the evaluation
corpus with realistic clustered group structure rather than a uniform random permitted
subset, report the curve you actually measure, and treat the uniform model as the
worst-case bound it is.

---

## Pre-filter, post-filter, and the third option

### The short version

There are three ways to make a vector search respect permissions. Filter after searching —
fast, and it returns almost nothing. Work out what the person can see and compare their
question against all of it — always correct, and the cost grows with how much they can
see. Or keep the search structure and check permissions during the walk, admitting only
allowed results — fast and usually correct, and its accuracy depends on how selective the
filter is. Which one wins is a measurable curve, not an opinion, and measuring it is the
point of this project.

### The intuition

Back to the library, where you hold keys to some rooms and not others.

**Post-filter.** Ask the front desk for the ten books nearest your question. They hand you
a list. You walk to each one, find nine doors locked, and go home with one book. Fast,
polite, useless.

**Pre-filter, done by brute force.** Get the list of rooms you can enter. Walk every shelf
in every one of them and measure each book against your question. You will find the ten
best books you are allowed to read — exactly those, with no approximation. The walk takes
as long as your key ring is large. With keys to 200 shelves this takes a minute. With keys
to 8,000 shelves it takes all afternoon.

**Filter-aware traversal.** Use the signposts, but change one rule. At each step you look
at the neighbouring books the signposts suggest. You may *walk through* a locked room to
get somewhere — the signs in it still tell you which way to go — but you may only put a
book in your bag if it is in a room you can enter. You end up near the right shelves fast,
and you come home with books you can read.

The third one is the production answer, and the reason it needs explaining is the
difference between "walk through but do not take" and "take then put back". Post-filtering
is the second. It sounds like the same operation and it is not, and the reason is worth
spelling out in terms of how the search actually moves.

**Why post-filtering collapses, in graph terms.** An approximate nearest neighbour index —
most commonly HNSW, hierarchical navigable small world, a layered graph where every vector
is linked to a few dozen near neighbours — searches by greedy descent. You enter somewhere
near the top, hop to whichever neighbour is closer to the query, repeat until no neighbour
improves, drop a layer, repeat. You carry a candidate list of size `ef_search`, typically
64 to 256, and you have a fixed budget of nodes you will visit — a few thousand out of a
million.

That budget is spent on a *trajectory*, and the trajectory is chosen entirely by
similarity. So the walk heads straight into the densest region of documents that match the
question. For Dana, that region is the pricing committee's folder, because that is where
the answer genuinely lives. The walk arrives there, fills its candidate list with pricing
committee chunks, stops improving, and returns.

```
            query
              |
              v
   ,---------------------------.
   |  x  x  x  x  x  x  x  x  x |   region the greedy walk explores
   |  x  x  x  x  x  x  x  x  x |   and fills its ef_search budget with
   |  x  x  x  x  x  x  x  x  x |
   `---------------------------'        o        o
                                            o  o        <- permitted chunks,
                                          o      o         slightly further out,
                                             o             never visited at all

   x = chunk this reader may NOT see      o = chunk this reader MAY see
```

Post-filtering then deletes every `x`. There were no `o`s in the candidate list, because
nothing in the search ever knew the `o`s were the only nodes that counted. The search was
not "the right search plus a filter". It was a search whose entire path was chosen by
documents the user cannot see.

Filter-aware traversal changes one line of that walk: a node is still *traversed* — its
edges are still used, because the graph's connectivity is what makes the search fast — but
it is only *admitted to the candidate list* if the filter accepts it. The walk keeps
moving through `x` territory and only collects `o`s. That is the whole idea, and it is why
"prune during" is a different algorithm from "prune after", not an optimisation of it.

### How it actually works

Take the reference numbers from the rest of the guide: `N = 1,000,000` chunks, `D = 384`
dimensions, float32, so 1,536 bytes per vector, on a machine that sustains roughly 10 GB
per second of memory bandwidth on one thread.

**Strategy 1 — post-filter.**

```
candidates = ann_search(q, f*k)        # ignores permissions
results    = [c for c in candidates if permitted(c)][:k]
```

Cost: one graph walk, `O(log N)` hops, 25 ms at a million chunks. Recall at `k`: about
`min(1, f x s)`. Never leaks. Collapses.

**Strategy 2 — pre-filter by exact scan over the permitted set.**

```
A = compile_permitted_set(principal)   # the expensive part
scores = A_vectors @ q                 # |A| x D multiply-adds
results = top_k(scores)
```

Recall is 1.0 by construction — this is the definition of the right answer, not an
approximation of it. Cost is linear in `|A|`:

| `|A|` | Selectivity | Bytes read | Sequential-read time | Realistic with scattered access |
|---|---|---|---|---|
| 1,000 | 0.1% | 1.5 MB | 0.2 ms | under 1 ms |
| 20,000 | 2% | 30.7 MB | 3 ms | 5–15 ms |
| 100,000 | 10% | 154 MB | 15 ms | 30–60 ms |
| 1,000,000 | 100% | 1.54 GB | 150 ms | 150 ms+ |

The "scattered access" column is the one people forget. The permitted vectors are not
contiguous in memory. You are issuing random reads of 1,536 bytes, each one a handful of
cache lines, and the hardware prefetcher cannot help you. Sequential bandwidth is an
optimistic bound; assume two to five times worse until you have measured it on your own
layout.

There is also a second cost: working out `|A|` and which vectors are in it. Expressing the
permitted set as a list of document identifiers works until the list has 40,000 entries,
at which point `WHERE doc_id IN (...)` is a query plan disaster. That is the reason grant
tokens exist — a small set of opaque strings standing for group edges, so the filter
condition is a set intersection of a dozen items rather than an enumeration of tens of
thousands.

**Strategy 3 — filter-aware graph traversal.**

Keep the graph over all `N` vectors. During the walk, evaluate the filter on each visited
node, use every node's edges, admit only matching nodes to the result heap. Qdrant does
this with payload filters applied inside traversal; pgvector's equivalent is iterative
index scanning.

The cost model is the interesting part. To collect `ef_search` *permitted* candidates you
must visit roughly `ef_search / s` nodes, because that is how many you have to look at
before enough of them match:

```
nodes visited  ≈  min( N ,  c x ef_search / s )
```

At `s = 0.5` with `ef_search = 128`, that is a few hundred nodes. At `s = 0.02` it is
around 6,400. At `s = 0.001` the formula says 128,000, which is an eighth of the corpus
visited in random order — far more expensive than scanning the permitted 1,000 vectors
sequentially. **Filtered graph search costs roughly `1/s`; exact scan costs roughly `s`.
Two curves moving in opposite directions cross somewhere.** That crossing point is the
whole design question.

Worse, recall degrades in the same direction the cost rises. The graph's edges encode
proximity among *all* vectors. Take a 0.1% subset and the permitted sub-graph may be
barely connected or genuinely disconnected: greedy descent lands on a permitted node whose
every neighbour is forbidden, has nowhere to improve to, and terminates early with a short
and badly-ranked list. This is why serious implementations add a fallback to exact scan
below some cardinality — not as a performance tweak, but because below that point the
graph has stopped being a useful structure.

| | Post-filter | Exact scan over `A_p` | Filter-aware traversal |
|---|---|---|---|
| Returns | Permitted members of the global top-`k` | True top-`k` of `A_p` | Approximate top-`k` of `A_p` |
| Recall@k | `≈ min(1, f x s)` | 1.0 by construction | measured; high at moderate `s`, degrades as `s` falls |
| Cost | one graph walk | `O(|A_p| x D)` | `O(ef_search / s)` node visits |
| Gets better as | `s → 1` | `s → 0` | `s → 1` |
| Leak risk | none | none | none |
| Honest use | a baseline to measure against | small permitted sets | the general case |

**The crossover is measured, not argued.** Where exact scan overtakes filtered traversal
depends on the embedding dimension, the graph's fan-out, the size of the payload attached
to each vector, the machine's cache behaviour, and — the one nobody controls — how
clustered the permitted set happens to be in embedding space, which is a fact about the
customer's organisational structure. No amount of reasoning from first principles produces
that number. You produce it by sweeping selectivity across a grid, measuring latency and
recall for each arm against an exact oracle, and reading the intersection off the chart.

That is why the strategy in Sightline is selected at query time from an estimated
cardinality rather than hardcoded, and why the chart is the project's headline result.

### Where it goes wrong

**The cardinality estimate is wrong.** The whole planner rests on knowing roughly how many
chunks the principal may see, before searching. If the estimate is too high you route to
the graph when a scan would have been faster and exactly correct — a latency cost.
If it is far too low, you route a 200,000-chunk brute-force scan into a 200 millisecond
budget and blow it. The estimate must be validated against a real count, not trusted.

**Counting objects instead of chunks.** A plan admitting 500 documents could be 1,500
chunks or 150,000. A threshold expressed in documents is not a threshold at all.

**The engine silently falls back.** Several vector databases quietly switch to a full scan
when a filter is very selective. The results stay correct and the 99th-percentile latency
multiplies with no error and no log line. If you cannot see which strategy actually ran
for a given query, you cannot diagnose this, and you will spend a week blaming the network.

**Tuning `ef_search` upward to fix filtered recall.** It helps when the sub-graph is
connected and the beam is the constraint. It does nothing when the permitted sub-graph is
disconnected, because a wider beam over a region with no path to the answer is still no
path to the answer. Meanwhile latency rises roughly linearly. Raising `ef_search` is the
right reflex for uniform recall shortfalls and the wrong one for low-density collapse, and
telling them apart requires breaking recall down by selectivity bucket first.

**Treating an empty permitted set as "no filter".** If the compiled condition is empty,
one natural reading is "nothing to filter on, match everything". The other is "this
principal may see nothing, match nothing". The first is a total corpus disclosure to
anyone with no permissions at all. It is a one-character mistake and it is planted as
mutation M6 for exactly that reason.

**Padding a short result list from the unfiltered pool.** Planted mutation M7. It turns
the recall failure of chapter seven into a genuine leak.

**Benchmarking with a uniformly random permitted subset.** It is easy to generate and it
is the worst case for graph connectivity, because a random 2% of a graph is a dust cloud.
Real permitted sets are clustered — a department's documents are about the same handful of
topics, so they sit near each other in embedding space and the permitted sub-graph is
better connected than the random model predicts. Reporting only the random-subset curve
understates filtered traversal; reporting only a clustered curve overstates it. Report
both, and say which is which.

**Caching search results per user to avoid the cost.** Now the cache is a copy of
permission state, with an invalidation problem on every permission write. This is the
staleness window from architecture decision record 0001, reintroduced at a different layer
by someone optimising latency.

### In this project

The contract is stated on the interface itself, in
`/Users/cook/sightline/src/sightline/store/base.py`: `search()` must return "top-k over
the subset `plan` admits. Push the filter INTO the index." The docstring on the
`VectorStore` protocol says outright that the top-`k` over the permitted subset and the
global top-`k` with forbidden entries removed are different sets, and that the size of
that difference is the project's headline chart.

The choice between strategies is an enum in
`/Users/cook/sightline/src/sightline/types.py`, carried on every `FilterPlan` and recorded
on every `Answer`:

| `PlanStrategy` | Chosen when | What executes |
|---|---|---|
| `ENUMERATE` | fewer than 512 permitted objects | the identifiers go into the filter directly |
| `EXACT_SCAN` | permitted set small enough that brute force beats the graph | full scan of `A_p`, recall 1.0 |
| `GRANT_TOKENS` | the typical case | set-intersection filter applied during traversal |
| `UNFILTERED` | principal can see at least 98% of the corpus | no filter at all |

Functional requirement FR-5 demands the choice come from measured cardinality rather than
a hardcoded constant, and that `estimated_cardinality` land within 20% of the true count on
the test corpus. `VectorStore.count_matching(plan)` exists so the estimate can be checked
against a brute-force count in the conformance suite.

The admission test itself is `admits()` in
`/Users/cook/sightline/src/sightline/store/memory.py`. Its empty-condition branch returns
`True` only for `UNFILTERED` and `False` for everything else, with a comment saying why:
losing results is recoverable, leaking them is not. That is mutation M6's kill site.

Four backends sit behind the one protocol, and each has a job:

* `qdrant_store.py` — the serving path. FR-9's acceptance criterion is explicit that the
  filter is applied during graph traversal, not after.
* `pgvector_store.py` — a second, independent enforcement layer. FR-10 requires a test
  that sends a deliberately unfiltered query and gets zero forbidden rows back, because
  Postgres row-level security refuses them regardless of what the application asked for.
* the exact oracle — brute force, never served, the ground truth for every recall number
  in the guide. A test asserts it is unreachable from the serving application.
* `postfilter.py` — the baseline arm, guarded so it cannot serve traffic, instrumented
  with `PostFilterTrace` so its collapse can be plotted rather than described.

The latency budget these strategies compete inside, from the reference machine — a 2015
dual-core laptop with no graphics card — is 25 milliseconds at the median and 90 at the
95th percentile for filtered search over a million chunks, within a 200 millisecond
end-to-end budget excluding generation, and an authorisation overhead ceiling of 80
milliseconds. That ceiling exists for a political reason as much as a technical one: a
security layer that costs more than that is a security layer someone will propose turning
off.

### Interview questions

**1. Define pre-filtering and post-filtering in a vector search context.**

Post-filtering searches the entire index first, gets the global top-`k`, and then discards
the results the user is not allowed to see, so it returns at most `k` and usually far
fewer. Pre-filtering restricts the candidate set to what the user may see and searches
only that — in its simplest honest form, brute-force scoring every permitted vector, which
gives the exact right answer at a cost proportional to the size of the permitted set. The
distinction that matters is that post-filtering answers a question nobody asked, namely
"which of the globally best passages are you allowed to read", when the question was
"which passages that you are allowed to read are best".

**2. Why does post-filtering collapse rather than degrade gracefully?**

Because the search budget is spent on a trajectory, and the trajectory is chosen by
similarity over the whole corpus. Greedy descent through the graph heads into the region
where the best global matches are, fills its candidate list there, and stops. If that
region belongs to a department the user is not in, every single candidate is discarded and
the permitted chunks — which may be only slightly further away — were never visited,
because nothing in the walk knew they were the only ones that counted. So it is not that a
few results are lost at the margin. The entire search happened in the wrong neighbourhood.

**3. What is filter-aware traversal actually doing differently?**

It separates two things that post-filtering conflates: whether you may walk through a node,
and whether you may take it. Non-matching nodes are still traversed, because their edges
are the connectivity that makes graph search fast, but only matching nodes are admitted to
the result heap. So the walk keeps the full graph's navigability while the answers come
from the permitted sub-graph. The cost is that you evaluate the filter on every visited
node, which is why the filter has to be cheap — in our case a set intersection over a
dozen grant tokens, not a membership test against a 40,000-element identifier list.

**4. When would you deliberately choose brute force over an approximate index?**

When the permitted set is small, which for a permissioned system is most of the time. At
384 dimensions and float32, 20,000 vectors is 30 megabytes, which is single-digit
milliseconds sequentially and maybe ten to fifteen with scattered access — comfortably
inside a 200 millisecond budget. And you get recall of exactly 1.0, which is not an
approximation of the right answer but the definition of it. Below some cardinality the
graph is also actively worse: the permitted sub-graph is sparsely connected, greedy
descent dead-ends, and you pay more to get less. So the low-selectivity end of the curve
is where brute force wins on both axes at once, which is unusual and worth exploiting.

**5. How do you find the crossover point between exact scan and filtered graph search?**

You measure it, because it depends on dimension, graph fan-out, payload size, cache
behaviour and how clustered the permitted set is in embedding space, and that last one is a
fact about the customer's org chart. Concretely: sweep selectivity across a grid — we use
0.1%, 1%, 10%, 50% and 100% — and for each point record latency at the median and 95th
percentile and recall at ten against a brute-force oracle, for both arms. Filtered
traversal's cost rises roughly as `1/s` as selectivity falls while scan's falls roughly as
`s`, so the curves cross, and you read the threshold off the chart. Then you put that
number in the planner, record which strategy actually ran on every response, and re-measure
when the corpus or the hardware changes, because the number is a property of a deployment
rather than of the algorithm.

**6. Your planner estimated 400 permitted chunks and routed to exact scan, but the true
number was 90,000. What happens, and how do you find out?**

Nothing incorrect happens — the results are still exactly right, because a brute-force scan
over a larger set is still a brute-force scan. What happens is that one query reads about
138 megabytes instead of 600 kilobytes and blows the latency budget by an order of
magnitude, and it will do so consistently for that principal. I find it because the chosen
strategy and the estimated cardinality are both recorded on the response and emitted as
OpenTelemetry span attributes, so a slow-query trace shows `EXACT_SCAN` with an estimate
that does not match the work done. The underlying fix is in the estimator, which is why
there is a requirement that it land within 20% of a true count and a conformance test that
compares it against `count_matching()`. The guard I would add on top is a hard ceiling: if
the scan exceeds the estimate by more than some factor, abandon it and fall back, rather
than quietly spending the budget.

**7. Is there a right answer to "pre-filter or post-filter"?**

Not as a general question, and anyone who answers it without asking about selectivity is
answering a different question. At high selectivity post-filtering is nearly free and
nearly correct, and building anything more elaborate is wasted work — above about 98%
we skip filtering altogether. At low selectivity post-filtering is not a trade-off, it is
a failure, and the only defensible options are exact scan or filter-aware traversal. So
the right answer is that the decision is per query, not per system, which is why the
strategy is a compiled property of a plan with the cardinality estimate attached, and why
it is recorded on the response. The genuinely contested part is the boundary between exact
scan and filtered traversal in the middle of the range, and that one is empirical.

**8. You have a vector database that claims to support filtered search. How do you verify
it does what you think?**

Three tests, and I would not trust a vendor benchmark for any of them. First, correctness:
a differential test against a brute-force oracle over the same permitted set, across
randomised principal and query pairs, asserting zero forbidden results — ours runs a
million pairs and any non-zero count fails the build outright. Second, whether the filter
is really applied during traversal or after: sweep selectivity and watch the recall curve,
because a post-filter implementation wearing a pre-filter interface shows a collapse that
tracks selectivity almost exactly, and that signature is unmistakable. Third, whether it
silently falls back to a full scan under selective filters: watch the 95th-percentile
latency across the same sweep and look for the step change. The general principle is that
a filtered-search claim is a claim about recall and latency jointly, and a vendor who
publishes one without the other has told you which one is bad.

**9. What would you have to change to support permissions that vary by field rather than
by document?**

The grant token model assumes the unit of authorisation is the object a chunk belongs to,
so field-level or paragraph-level permissions break the assumption that a chunk's token set
is derivable from its document. The mechanical answer is to stamp tokens per chunk from a
finer-grained source, which the index does not care about at all — the filter is still a
set intersection. The real cost lands in two other places: recheck has to be able to
authorise at that granularity, which means the tuple store needs objects at that
granularity and the permission graph gets much larger, and reindexing churn goes up because
a permission edit now touches specific chunks rather than a whole document's worth. I would
want to know how many distinct field-level policies actually exist before agreeing, because
if the answer is a few dozen recurring patterns, modelling them as synthetic parent objects
keeps the tuple count manageable, and if the answer is per-paragraph and unique, the honest
response is that the permission store, not the vector index, is the thing that has to be
redesigned.
