# Part 2 — Finding things in a pile of vectors

---

## Nearest-neighbour search and why exact is sometimes right

### The short version

Once every chunk of text is a list of numbers, "find me the relevant passages" becomes
"find me the closest lists of numbers". The honest way to do that is to compare the
question against every chunk and keep the best few. Everyone assumes that is too slow to
consider, so they reach for a clever index straight away. For corpora of the size most
internal tools actually have, the honest way is fast enough, and it is exactly right,
which turns out to matter more than it sounds.

### The intuition

Imagine a library where every book has been given a position on a map. Books about
employment contracts sit near each other. Books about server capacity sit somewhere else
entirely. You walk in with a question, the question also gets a position on the map, and
you want the ten books nearest that spot.

There are two ways to find them.

The first: walk the whole library, measure your distance to every book, keep a running
list of the ten closest. This always gives the right answer. It takes time proportional to
the size of the library.

The second: build a system of signposts in advance — "for anything about contracts, head
this way" — then follow signposts from wherever you are until you stop making progress.
This is much faster and usually right. Sometimes the signposts lead you into a corner of
the library that is close, but not closest, and you miss a book you should have found.

The instinct is that the first way is far too slow to consider. Put real numbers on it.

Sightline's development corpus is 70,000 chunks of text. Each chunk is a vector — an
ordered list of numbers — of 384 dimensions, stored as float32, meaning each number takes
4 bytes. The model that produces them is a small sentence embedder run through ONNX
Runtime.

A comparison between the question and one chunk, when both vectors are normalised to unit
length, is a **dot product**: multiply the two lists element by element and add up the
results. That is 384 multiply-add operations.

Across the whole corpus:

```
70,000 chunks x 384 dimensions = 26,880,000 multiply-adds per query
70,000 chunks x 384 dims x 4 bytes = 107,520,000 bytes = 107.5 MB read per query
```

Twenty-seven million multiply-adds sounds enormous. It is not. Sightline's reference
machine is a 2015 dual-core Intel MacBook Pro with no GPU. One of its cores, using AVX2
vector instructions, does roughly 16 multiply-adds per clock cycle at about 2.7 GHz —
call it 43 billion per second. The arithmetic therefore takes about 0.6 milliseconds.

The arithmetic is not the problem. The **memory bus** is. To do that arithmetic the CPU
must pull 107.5 MB out of RAM, and that machine sustains maybe 10 GB per second on one
thread. 107.5 MB at 10 GB/s is about 11 milliseconds.

So: an exhaustive, exactly-correct search over 70,000 chunks costs roughly **10 to 15
milliseconds** on a ten-year-old laptop. For comparison, Sightline's whole non-generation
latency budget is 200 ms at the 95th percentile. Brute force fits inside it with room to
spare.

### How it actually works

**The operation.** Given a query vector `q` and a set of `N` stored vectors, return the
`k` whose similarity to `q` is highest. For unit-normalised vectors, cosine similarity and
dot product are the same number, and both are monotonically related to Euclidean distance,
so the ranking is identical whichever you compute. Sightline normalises at ingest and uses
dot product, because it is the cheapest of the three.

**The cost model.** Exhaustive search — "flat" search, in the literature — is:

| Quantity | Formula | At N=70,000, D=384 |
|---|---|---|
| Multiply-adds per query | `N x D` | 26.9 million |
| Bytes read per query | `N x D x 4` | 107.5 MB |
| Top-k selection | heap of size k over N | ~70,000 compares, negligible |
| Memory held | `N x D x 4` | 107.5 MB |
| Build time | none | zero |
| Recall | 1.0, by construction | 1.0 |

The top-k step barely registers. Maintaining a 10-element min-heap over 70,000 scores
costs one comparison per candidate, and the overwhelming majority fail that comparison
immediately and cost nothing else.

**How it scales.** Everything is linear in `N`:

| Corpus | Memory | Bytes/query | Est. latency, one core |
|---|---|---|---|
| 70,000 chunks | 107 MB | 107 MB | ~11 ms |
| 250,000 chunks | 384 MB | 384 MB | ~40 ms |
| 1,000,000 chunks | 1.54 GB | 1.54 GB | ~155 ms |
| 10,000,000 chunks | 15.4 GB | 15.4 GB | ~1.6 s |

Sightline's stated capacity target is one million chunks on that laptop, with a 90 ms
p95 for filtered search. Brute force misses that by a wide margin, and at 8 GB of total
RAM the vectors alone are a fifth of the machine. That is the point where an approximate
index stops being premature optimisation and starts being the only option. Below it, it
is a cost with no matching benefit.

**When exact is the correct engineering choice.** Four situations, and they are not rare:

1. **Small corpus.** Under roughly 100,000 vectors on commodity hardware, a flat scan is
   inside most latency budgets. Every millisecond you save with an index is bought with
   build time, a tuning surface, and an answer that can now be wrong.
2. **A highly selective filter.** This is the one that matters most to Sightline. If a
   user is permitted to see 400 chunks, scanning those 400 costs `400 x 384 = 153,600`
   multiply-adds over 614 KB — small enough to sit in the CPU's L2 cache, done in
   microseconds. An approximate graph index, by contrast, evaluates a roughly constant
   few thousand distances per query regardless of corpus size, so on a set that small it
   does *more* work than the exact scan and still returns an approximate answer.
3. **You need ground truth.** You cannot measure how good an approximate index is without
   something exact to measure it against.
4. **Recall is contractual.** If missing a result is a correctness failure rather than a
   quality regression, approximation is not available to you.

**The triangle.** Every vector search design trades three things off against each other,
and you can hold two.

| Pick | What you give up | Typical shape |
|---|---|---|
| Recall + low latency | Memory | Graph index with vectors held uncompressed in RAM |
| Recall + low memory | Latency | Exhaustive scan, or a disk-backed index |
| Low latency + low memory | Recall | Compressed vectors, aggressive approximation |

"Recall" here has a precise meaning. **Recall@k** is the fraction of the true top-k that
your system actually returned:

```
recall@k = |returned_top_k  intersect  true_top_k| / k
```

If the exact answer for a query is chunks {3, 9, 14, 21, 30, 41, 55, 62, 77, 88} and your
index returns eight of those plus two others, recall@10 is 0.8. Note what this does not
measure: whether the two it missed mattered. A result set can have recall@10 of 0.7 and be
perfectly serviceable, or recall@10 of 0.95 and be missing the one document that contained
the answer.

### Where it goes wrong

**People skip the measurement.** The most common failure is reaching for an approximate
index at 20,000 vectors because that is what the tutorial did, then spending a week tuning
parameters to recover recall that a flat scan gave away free. Measure the flat scan first.
It is four lines of NumPy and it tells you whether you have a problem.

**The bottleneck gets misdiagnosed.** Engineers optimise the arithmetic — SIMD, better
BLAS, more threads — when the scan is memory-bound. Going from 0.6 ms of compute to 0.3 ms
of compute does nothing when 11 ms are spent waiting on RAM. The lever that actually moves
a flat scan is reducing bytes read: fewer dimensions, or smaller ones (int8 instead of
float32 is a 4x reduction in traffic).

**Latency and throughput get confused.** A flat scan at 11 ms per query is 90 queries per
second per core, and a second core roughly doubles that if you parallelise across queries
rather than within one. For an internal tool used by 500 people, peak load is a handful of
queries per second. The single-query latency is the thing to care about; the throughput
number is almost never the constraint, and people size their architecture off it anyway.

**Exhaustive search does not scale gracefully.** It degrades linearly and predictably,
which is a virtue, but there is no knob. When you outgrow it you do not tune it, you
replace it, and the replacement has different correctness properties. Plan the crossover
before you hit it.

**"Exact" is only exact about distance.** A flat scan returns the true nearest neighbours
*of the embedding*. If the embedding model is a poor fit for your domain, exact search
returns the exactly wrong answers with great confidence. Exactness is a property of the
retrieval step, not of the result's usefulness.

### In this project

Sightline runs **three** vector backends against one interface, deliberately, because each
one exists to answer a different question.

The contract is a `VectorStore` protocol in `src/sightline/store/base.py`. Every backend
implements it; every backend passes the same conformance suite; `search()` returns
`UncheckedHit` and never `Hit`, because only `recheck()` against the live permission store
may produce a `Hit`.

| Backend | Role | Search | Served? |
|---|---|---|---|
| Qdrant | Production serving path | HNSW graph, filter pushed into traversal | Yes |
| pgvector + Postgres row-level security | Second, independent enforcement layer | HNSW or exact, plus a database policy that refuses forbidden rows | Yes |
| Exact oracle (NumPy, optionally FAISS) | Ground truth | Exhaustive | **Never** |
| Post-filter baseline | Measuring the failure it causes | Search globally, discard afterwards | **Never** |

The exact oracle is the reason the published numbers mean anything. Sightline's
requirement FR-14 is called "honest recall guarantee", and what it forbids is interesting:
the system is not allowed to assert that filtered approximate search equals exact search.
A test that makes that assertion on an approximate path fails the build, enforced by a
grep over the test sources. The strict-equality assertion is permitted in exactly one
place — the `EXACT_SCAN` path — because there it is true.

That path is a first-class execution strategy, not a fallback. `PlanStrategy.EXACT_SCAN`
in `src/sightline/types.py` is chosen by the permission compiler when a principal's
estimated cardinality — the number of documents they can see — is small enough that
scanning that set beats traversing a graph. Concretely: a graph index at one million
chunks evaluates on the order of 2,000 to 6,000 distances per query. Below roughly that
many permitted chunks, the exact scan is both faster and correct. The threshold is
measured on the reference corpus and recorded in the response, so "why was this query
slow" and "why is this answer exactly right" are both answerable after the fact.

The oracle also does something stronger than measuring recall. The evaluation harness runs
10^6 randomised query/principal pairs through the serving path and the oracle and compares
the permitted sets. The expected leak count is **exactly zero**, and any other value fails
the build outright. Missing a result the oracle found is a recall failure and fails at
lower severity. Returning a result the oracle's permitted set excludes is a breach.

### Interview questions

**1. Why is a flat vector scan memory-bound rather than compute-bound?**

Because the ratio of work to data is terrible. For each 4-byte float you load, you do one
multiply and one add — that is about two operations per four bytes. A modern core can do
tens of billions of floating-point operations a second but can only pull maybe 10 to 20
gigabytes a second off RAM on one thread, and each vector is touched once and then thrown
away, so caching does not help. On our reference laptop, 70,000 chunks at 384 dimensions
is 27 million multiply-adds, about 0.6 ms of arithmetic, but 107 MB of traffic, about
11 ms of waiting. That is why the fix is fewer or smaller bytes — int8 quantisation,
fewer dimensions — not faster arithmetic.

**2. When would you deliberately not use an approximate index?**

Four cases. Small corpus, where a flat scan already fits the latency budget and an index
buys you nothing but a tuning surface and a new way to be wrong. Highly selective filters,
where the permitted set is a few hundred vectors and scanning them exactly is faster than
traversing a graph. When you need ground truth to evaluate something else. And when recall
is a correctness requirement rather than a quality metric — if a missed result is a bug
rather than a slightly worse answer, you cannot use an approximate method. In Sightline
the first two both apply, which is why exact scan is a named execution strategy rather
than a fallback.

**3. Define recall@10 and explain what it does not tell you.**

Recall@10 is the size of the intersection between the ten results you returned and the
true top ten, divided by ten. If you return eight of the correct ten, that is 0.8. What it
does not tell you is whether the two you missed mattered. Recall is unweighted — missing
the top-ranked result and missing the tenth count the same. It also assumes the exact
top-k is the right answer, which is only true if the embedding model is good for your
domain. I treat recall as a regression guard against the index, not as a measure of answer
quality; for that I need an end-to-end evaluation with judged answers.

**4. Your corpus is 70,000 chunks and someone proposes HNSW. What do you say?**

I ask what the measured flat-scan latency is, because at that size it is around 10 to
15 milliseconds on modest hardware and that is usually inside budget. If it is, HNSW costs
build time, a set of parameters somebody has to own, and a recall number that is now below
one — and buys latency we did not need. The argument that would change my mind is growth:
if we are at 70,000 now and expect a million within the year, building the index interface
early is reasonable, because the crossover is genuinely around a few hundred thousand
vectors. But I would still keep the flat path as the evaluation oracle rather than
deleting it.

**5. How do the economics change when the query has a selective permission filter?**

They invert. A graph index does roughly a constant number of distance evaluations per
query — a few thousand — largely independent of corpus size. An exhaustive scan does one
per candidate. So on the full corpus the graph wins by orders of magnitude, but on a
permitted set of, say, 400 chunks, the scan does 400 evaluations over 614 KB that fits in
cache, and the graph does more work than that and returns an approximate answer. Worse,
selective filters are precisely where graph traversal degrades, because the permitted
nodes may not be well connected to each other in the graph. That is why our plan compiler
picks the strategy from estimated cardinality instead of hardcoding one.

**6. You have an exact oracle that is never served. Justify the cost of maintaining it.**

Without it, every number we publish is unfalsifiable. Recall figures need ground truth by
definition. More importantly, it is how we detect leaks: we run a million randomised
query-and-principal pairs through both the serving path and the oracle and compare the
permitted sets, and the expected leak count is exactly zero — anything else fails the
build. That test is only possible because something exhaustive computes the true permitted
answer. The cost is one NumPy file and a test that never runs in production, and we have a
test asserting the oracle is not reachable from the serving API so it cannot drift into
the hot path.

**7. Our latency spec says 90 ms p95 for filtered search at one million chunks. Walk me
through whether brute force could hit that, and what you would do if it could not.**

It cannot, and the arithmetic says so before I write code. A million 384-dimensional
float32 vectors is 1.54 GB. At roughly 10 GB/s of single-thread memory bandwidth that is
about 155 ms of pure memory traffic per query, and we have a budget of 90 for the whole
search stage. Two cores get us to maybe 80 ms with perfect scaling, which is not a margin
I would ship on an 8 GB laptop that also has to hold the vectors. The options in order of
what I would try: push the permission filter into the index so most queries scan far less
than a million, which we do anyway; quantise to int8 for a 4x cut in traffic, at a recall
cost I would measure; then a graph index. What I would not do is parallelise harder and
call it solved, because the ceiling is the memory bus and more threads share the same bus.

**8. Is there a single right answer to "exact or approximate"?**

No, and I would be suspicious of anyone who says otherwise. It is a function of corpus
size, hardware, latency budget, how selective your filters are, and how expensive a missed
result is. The framing I find useful is the triangle — recall, latency, memory, pick two —
and then asking which of the three the product actually cares about. Sightline cares most
about recall, because a permission-filtered search that silently drops results is
indistinguishable from a permission bug to the person using it. So we spend memory and
accept a graph index only where the corpus forces it, and we keep an exact path for the
cases where it is both faster and correct.

---

## HNSW, the index everyone uses

### The short version

HNSW — Hierarchical Navigable Small World — is the data structure behind almost every
vector database you have heard of. It arranges your vectors into a graph with a few
long-range shortcuts at the top and dense local connections at the bottom, then finds
neighbours by following edges downhill towards the query. It is fast and it usually gives
the right answer. Understanding the three knobs it exposes, and understanding that
"usually" is not "always", is most of what you need to run one in production.

### The intuition

Think about how you would find a specific house in a country you do not know.

You would not drive down every street. You would take a plane to the right city, then a
train to the right district, then a taxi to the right street, then walk. Each stage moves
you a smaller distance with more precision. The plane network has few nodes and enormous
hops. The pavement network has every address on it and hops of a few metres.

HNSW is that, built out of your vectors.

The bottom layer contains every vector, each connected to a few dozen of its near
neighbours — the pavement. Above it sits a layer containing a small random sample of those
vectors, maybe one in sixteen, connected to *their* nearest neighbours, which are now
further apart — the road network. Above that, a sample of the sample, with hops further
still. At the very top, a handful of nodes with continental reach.

Searching works like the journey. Start at a single entry point in the top layer. Look at
its neighbours. Move to whichever is closest to your query. Repeat until no neighbour is
closer — you have arrived in the right region at this scale. Drop down a layer, where your
current node has more, shorter edges, and do it again. By the bottom layer you are already
in the right neighbourhood and you are refining, not travelling.

```
Layer 3      (A)- - - - - - - - - - - - - -(B)          few nodes, huge hops
              |                              |
Layer 2      (A)- - - -(C)- - - -(D)- - - -(B)          sample of the sample
              |         |         |          |
Layer 1      (A)-(E)-(C)-(F)-(D)-(G)-(H)-(B)            ~1 in 16 of everything
              |   |   |   |   |   |   |   |
Layer 0      (A)(E)(I)(C)(J)(F)(D)(K)(G)(H)(B)(L)...    every vector, dense local links

query  --> enter at A, greedy-descend, refine at layer 0, return best k
```

If you have seen a skip list — the sorted list with express lanes that let you skip ahead
before dropping down to the exact position — this is that idea, generalised from one
dimension to hundreds. In a skip list "closer" means a smaller number. Here it means a
larger dot product.

Why does this work? The number of layers grows like the logarithm of the number of
vectors. With one million vectors and sixteen neighbours per node, you get about five
layers. You travel through each of them in a handful of hops. So instead of a million
distance calculations, you do a few thousand — and crucially, that count barely grows as
the corpus does.

### How it actually works

**Construction.** Vectors are inserted one at a time. Each new vector is assigned a
maximum layer drawn from an exponentially decaying distribution:

```
level = floor( -ln(uniform(0,1)) * mL )      where  mL = 1 / ln(M)
```

With `M = 16`, `mL` is about 0.361, and the probability of a node reaching layer 1 or
above is `1/M` = 6.25%. Layer 2 is 0.39%, and so on. This is what makes each layer a
roughly uniform 1-in-M sample of the one below, with no explicit sampling step.

The vector is then inserted into every layer from its assigned level down to 0. At each
layer the algorithm searches for the `ef_construction` nearest already-inserted nodes,
picks `M` of them as neighbours using a heuristic that prefers *diverse* directions over
merely closest candidates, and adds bidirectional edges. If a neighbour now exceeds its
edge budget, its worst edge is pruned.

The diversity heuristic matters more than it sounds. Connecting each node to its M closest
neighbours produces tight clusters with no bridges between them, and a greedy search gets
trapped. Preferring a slightly further neighbour that lies in a direction not already
covered is what keeps the graph navigable.

**Search.** Given query `q` and result count `k`:

1. Start at the fixed entry point on the top layer.
2. For each layer above 0: greedily move to the neighbour closest to `q`, repeat until no
   neighbour improves, then descend using that node as the entry point.
3. At layer 0: run a best-first search maintaining a candidate list of size `ef_search`,
   expanding the most promising unexplored node, stopping when the nearest unexplored
   candidate is worse than the worst result held.
4. Return the top `k` from that list.

**The three parameters.**

| Parameter | When it applies | Raising it improves | Raising it costs | Changeable after build? |
|---|---|---|---|---|
| `M` | Build | Recall, graph connectivity | Memory (linear), build time, per-hop work | No — full rebuild |
| `ef_construction` | Build | Graph quality, therefore recall at any `ef_search` | Build time only | No — full rebuild |
| `ef_search` | Query | Recall | Query latency (roughly linear) | **Yes, per query** |

`M` is the maximum number of bidirectional edges per node on layers above 0. Layer 0 gets
`M0 = 2M`, because the bottom layer does the precision work and needs denser connectivity.
Typical values are 12 to 48. Higher `M` helps most when the data is high-dimensional or
clustered; past about 48 the returns vanish and the memory cost does not.

`ef_construction` is the size of the candidate list used while inserting. It does not
appear at query time and costs no query memory. It buys a better graph. Typical values are
100 to 500. This is the parameter people under-set, because its cost lands during an
offline build where nobody is watching the clock, and its benefit lands in every query
forever.

`ef_search` is the size of the candidate list at query time and must be at least `k`. It
is the operational knob: you can raise it for one query without touching the index. A
common production shape is `ef_search = 64` for interactive search and 200+ for an
evaluation or an export where latency does not matter.

**Why the result is approximate.** Greedy descent finds a *local* optimum of the graph,
not a global one. If the true nearest neighbour sits in a region the graph does not link
to from where you entered, you will not find it. Larger `ef_search` widens the beam and
makes that less likely; it never makes it impossible. So recall@k — the fraction of the
true top-k you returned — is below 1.0, and the honest way to state HNSW's behaviour is a
curve, not a number: recall as a function of `ef_search`, measured against an exact oracle
on your actual data. A typical curve looks like 0.85 at `ef_search=16`, 0.95 at 64, 0.99
at 200, with latency rising roughly in step.

**Memory arithmetic.** Two components: the vectors, stored uncompressed inline, and the
graph edges.

Vectors, at 70,000 chunks and 384 dimensions in float32:

```
70,000 x 384 x 4 = 107,520,000 bytes = 107.5 MB
```

Graph edges, with `M = 16` so `M0 = 32`, and 4-byte integer node identifiers:

```
layer 0:      32 links x 4 bytes                  = 128 bytes per node
upper layers: 16 x 4 x (1/(M-1)) = 64/15          ~  4.3 bytes per node (expected)
total                                             ~ 132 bytes per node
70,000 x 132                                      ~ 9.3 MB
```

So the graph is under 9% of the vector data. This is the number people get wrong in both
directions — they either assume the graph is negligible (it is not, at high `M`) or that it
dominates (it does not, at typical `M`). Doubling `M` to 32 takes the graph to about 18 MB,
still small beside 107 MB of vectors.

Scaling the same shape to Sightline's one-million-chunk target: 1.54 GB of vectors plus
about 132 MB of graph, roughly 1.7 GB before payloads. On an 8 GB machine that is a real
but manageable commitment.

**Comparison with the alternatives.**

| Method | Idea | Memory at N=70,000, D=384 | Query cost | Recall | Notes |
|---|---|---|---|---|---|
| Flat | Compare with everything | 107 MB | O(N·D) — 27M ops | 1.0 exactly | No build, no tuning |
| HNSW | Navigable graph, greedy descent | 117 MB | ~2–6k distance evals | ~0.95–0.99, tunable | Fast, memory-hungry, build is slow |
| IVF-Flat | k-means into `nlist` cells, scan `nprobe` of them | 108 MB | `nprobe/nlist` of the corpus | Tunable via `nprobe` | Cheap build, poor on cell-boundary queries |
| IVF-PQ | IVF plus compressed vectors | ~3.5 MB + centroids | Table lookups, not float ops | Noticeably lower | Huge compression, approximate *scores* |

**IVF** — inverted file index — runs k-means over the corpus to produce `nlist` cells, each
with a centroid, and assigns every vector to its nearest cell. At query time you compare
the query to the `nlist` centroids, pick the closest `nprobe` cells, and exhaustively scan
only those. At `N = 70,000` a reasonable `nlist` is 1,024, giving about 68 vectors per
cell; with `nprobe = 16` you scan roughly 1,100 vectors instead of 70,000. Builds far
faster than HNSW and uses almost no extra memory. Its weakness is geometric: a query that
lands near a cell boundary has its true neighbours split across cells it did not probe.

**Product quantisation (PQ)** is a compression scheme, not an index, and composes with the
others. Split each 384-dimensional vector into `m = 48` sub-vectors of 8 dimensions. Run
k-means on each sub-space to get 256 centroids, so each sub-vector is stored as a single
byte naming its centroid. A vector goes from 1,536 bytes to 48 — a 32x reduction, taking
70,000 chunks from 107 MB to 3.4 MB. Distances are computed by precomputing a small lookup
table per query and summing 48 table entries per candidate instead of doing 384
multiply-adds. The cost is that the score itself is now approximate, which means you can no
longer trust the ranking near the top and you cannot use the index as ground truth for
anything. **Scalar quantisation** to int8 is the gentler version: 4x smaller, a recall cost
usually in the low single digits of a percent, and far easier to reason about.

### Where it goes wrong

**Filtered search is where HNSW hurts.** The graph was built over the whole corpus. Apply a
filter that admits 1% of nodes and the permitted sub-graph may be barely connected —
greedy descent walks into regions where every neighbour is forbidden, burns its candidate
budget, and returns few results or poor ones. Recall collapses exactly where the filter is
most selective. Production systems handle this by falling back to exact search below a
cardinality threshold: Qdrant's `full_scan_threshold` does precisely this. If you take one
practical thing from this chapter, take that the recall number quoted in the benchmark is
an *unfiltered* number and tells you little about your filtered workload.

**Deletion is awkward.** HNSW has no clean delete. Implementations mark nodes as deleted
and skip them during search, which means the graph keeps edges pointing at tombstones,
traversal gets slower, and quality drifts. Heavy churn requires periodic rebuilds. Anyone
planning frequent deletions should price the rebuild in from the start.

**Builds are slow and non-incremental in the ways that matter.** Insert cost grows with
`ef_construction` and `M`, and building a million-vector index is minutes to hours, not
seconds. Changing `M` or `ef_construction` means rebuilding everything. Only `ef_search`
is free to change, which is why it is worth deciding the other two deliberately rather
than accepting defaults.

**The entry point is a single point of failure for quality.** All searches begin at the
same top-layer node. Adversarial or merely unusual data distributions can make that entry
point a poor starting place for a whole region of the query space, and the symptom is a
subset of queries with quietly bad recall while the average looks fine. Measure the recall
*distribution*, not the mean.

**Benchmarks are not your data.** Published recall-versus-latency curves are measured on
SIFT or GloVe with uniform queries and no filters. Your embedding model, dimensionality,
clustering and filter selectivity are all different. The only curve that means anything is
the one you measure on your corpus against an exact oracle.

**High `M` is not free.** People raise `M` when recall disappoints, because it is the
parameter that sounds most like "quality". It raises memory linearly, slows the build, and
adds work per hop. Raising `ef_search` first is cheaper, reversible, and often enough.

### In this project

Sightline uses HNSW on the serving path through Qdrant, and treats its approximate nature
as something to be measured rather than assumed away.

The design decision that shapes everything is recorded in
`docs/adr/0001-index-is-a-hint.md`: **the index is a hint, the database is the authority.**
Vector search returns `UncheckedHit`. Only `recheck()`, which reads the live permission
store, converts one into a `Hit`, and `Hit` has exactly one construction site in the whole
package — enforced by `tests/test_no_unchecked_construction.py`, which greps the source.
The consequence for this chapter is precise: because HNSW is approximate and its payloads
can be stale, a wrong index can only *lose* results. It can never produce one that reaches
the model unchecked.

The permission filter is pushed **into** the traversal, not applied after it. Qdrant
evaluates grant tokens — opaque keyed hashes derived from group edges, stored as a
filterable payload on each chunk — while walking the graph, so the candidate list fills
with permitted nodes rather than being drained by a post-filter. The post-filtering
approach is still implemented, in the baseline store, because demonstrating its recall
collapse across permission densities is the argument for the design. That store is marked
not-for-production, has an import-time guard preventing it being wired into the serving
app, and is never padded back up to `k` from the unfiltered pool — padding is planted
mutant M7 in the mutation suite.

The graph-degradation problem is handled by the execution strategy, not by hoping. The
compiler in the permission layer produces a `FilterPlan` whose `strategy` is one of
`ENUMERATE`, `GRANT_TOKENS`, `EXACT_SCAN` or `UNFILTERED`, chosen from
`estimated_cardinality`, which requirement FR-5 holds to within 20% of the true count on
the test corpus. When the permitted set is small enough that graph traversal would be both
slower and less reliable, the query routes to exact scan. The chosen strategy is recorded
on every `Answer` and appears as an OpenTelemetry span attribute, so a bad recall number
can always be traced to the path that produced it.

Recall is gated in continuous integration, not claimed. Requirement FR-14 sets a floor of
`recall@10 >= 0.95` against the exact oracle at permission densities of 0.1%, 1%, 10%, 50%
and 100%, with a drift tolerance of ±0.02 — five densities because the density is the axis
along which filtered HNSW actually fails. A strict-equality assertion is permitted only on
the `EXACT_SCAN` path; a grep test fails the build if one appears in an approximate-path
test. The corresponding curve for the post-filter baseline is published alongside it,
collapse and all.

### Interview questions

**1. Explain HNSW to someone who knows what a skip list is.**

A skip list is a sorted list with express lanes: a top level with few elements and big
jumps, lower levels with more elements and smaller jumps, and you descend as you home in.
HNSW is the same idea in many dimensions, where "closer" means a higher dot product rather
than a smaller number. The bottom layer holds every vector connected to a few dozen near
neighbours; each layer above is roughly a 1-in-M random sample of the one below, so its
edges span greater distances. You enter at the top, greedily walk to whichever neighbour is
closer to the query until none is, drop a layer, and repeat. Layer count grows like the log
of corpus size, so you visit a few thousand nodes instead of a million.

**2. What do M, ef_construction and ef_search each trade?**

`M` is edges per node — layer 0 gets 2M. Raising it improves connectivity and recall, and
costs memory linearly, build time, and work per hop. `ef_construction` is the candidate
list size during insertion; it buys graph quality and costs build time only, with zero
query-time cost, which makes it the parameter people wrongly under-set. `ef_search` is the
candidate list size at query time, must be at least `k`, and trades recall against latency
roughly linearly. The operational difference that matters: `ef_search` is changeable per
query, the other two require a full rebuild. So I set `M` and `ef_construction` once,
deliberately, and tune `ef_search` against a measured recall curve.

**3. Why is HNSW approximate at all, given the graph contains every vector?**

Because the search is greedy and local. You follow edges downhill and stop at a local
optimum of the graph, and there is no guarantee the global nearest neighbour is reachable
from where you entered via improving steps. If the true best result sits in a region the
graph does not bridge to from your path, you miss it, and no amount of layer structure
rules that out. `ef_search` widens the beam so you explore more before committing, which
makes misses rarer but never impossible. That is why the correct way to report HNSW's
behaviour is a recall curve against an exact oracle on your own data, not a single number.

**4. Work out the memory for a million chunks at 384 dimensions with M=16.**

Vectors first, because they dominate: a million times 384 dimensions times 4 bytes for
float32 is 1.536 GB. Graph edges: layer 0 holds 2M = 32 links at 4 bytes each, so 128 bytes
per node, and the upper layers add an expected `M x 4 / (M-1)`, about 4.3 bytes, because
each layer is a 1-in-M sample of the one below — call it 132 bytes per node, so about
132 MB. Total roughly 1.67 GB before payloads. So the graph is under 9% of the footprint,
and if I need to cut memory the lever is the vectors — int8 scalar quantisation takes 1.54
GB to 384 MB — not the graph.

**5. Compare HNSW to IVF and product quantisation. When would you pick each?**

HNSW when latency matters most and memory is available: best recall-per-millisecond, but
vectors sit uncompressed in RAM and builds are slow. IVF when build time and memory matter
and you can accept boundary effects: k-means into cells, scan `nprobe` of them, which at
70,000 vectors with `nlist=1024` and `nprobe=16` means scanning about 1,100 vectors instead
of 70,000, and it rebuilds cheaply. Product quantisation is orthogonal — it compresses each
vector from 1,536 bytes to about 48 by splitting into sub-vectors and storing centroid ids,
so it composes with either. I would reach for IVF-PQ when the corpus does not fit in
memory, and accept that the scores are now approximate so the top of the ranking cannot be
trusted without a rerank against full vectors.

**6. Why does HNSW recall degrade under a selective filter, and what do you do about it?**

Because the graph was built over the whole corpus, so the edges encode proximity among all
vectors, not among the permitted ones. Filter down to 1% and the permitted sub-graph may be
sparsely connected or disconnected — greedy descent keeps landing on nodes whose neighbours
are all forbidden, burns the candidate budget on dead ends, and returns a short or poor
list. It fails worst precisely where the filter is most selective, which is the opposite of
what you want. The mitigations are pushing the filter into traversal rather than applying
it afterwards, raising `ef_search` when the plan is selective, and falling back to exact
search below a cardinality threshold. In Sightline the fallback is a named strategy chosen
from estimated cardinality, and we publish the recall curve across five permission
densities specifically because the density is the axis where this breaks.

**7. Your filtered recall@10 is 0.90 and the requirement is 0.95. Diagnose it.**

First I check whether it is uniform or concentrated: the mean hides a distribution, and a
common shape is fine recall on most queries with a subset collapsing because their
permitted sets are small or poorly connected in the graph. So I break the number down by
permission density before touching a parameter. If the low-density buckets are the
problem, that is graph degradation and the answer is routing those to exact scan, not
tuning. If it is uniform, I raise `ef_search` first because it is free and reversible, and
watch the latency budget. Only if the recall-versus-`ef_search` curve flattens below 0.95
do I consider raising `M` or `ef_construction`, which means a rebuild. And throughout, I
compare against the exact oracle, because a filtered recall number measured against
anything else is measuring two unknowns at once.

**8. Someone proposes turning off recheck for HNSW results because the index payload
already has the grant tokens. Respond.**

I would not, and not primarily for latency reasons. The index payload is a copy of the
permission state as of the last reindex, and our staleness window for a loosened ACL is up
to 15 minutes. That asymmetry is the whole design: because search returns `UncheckedHit`
and only `recheck()` against the live store produces a `Hit`, a stale index loses results
and cannot leak them. Trusting the payload inverts that — now a permission removed five
minutes ago is still served, and the failure is silent. It is also why there is no
configuration flag to disable recheck: a flag is a thing someone eventually sets during an
incident. If the cost is the concern, the answer is to batch recheck into one store call
and cache the hot subgraph, which is what we do within an 80 ms authorisation budget.

**9. Is there a right answer to "HNSW or IVF"?**

Not in general, and the honest answer is that it depends on which resource you are actually
short of. HNSW gives the best recall per millisecond and is the right default when memory
is available and queries are interactive. IVF wins when you rebuild often, when memory is
tight, or when you want a structure simple enough to reason about under filters — and it
degrades more gracefully in some filtered cases because you can probe more cells. The
deciding questions I would ask are: how often does the corpus change, does it fit in
memory, how selective are the filters, and is anyone going to own the tuning. For Sightline
the filters are extremely selective and the corpus is modest, so the interesting decision
was not HNSW versus IVF at all — it was how much traffic to route away from the approximate
index entirely.
