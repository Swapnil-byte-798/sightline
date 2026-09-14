## What a vector database gives you that a library does not

### The short version

FAISS is a library: you hand it a block of numbers in memory and it finds the nearest ones very fast. Qdrant, pgvector, Weaviate and the rest are databases: they also store the numbers, survive a restart, accept writes from several processes at once, attach data to each vector that you can filter on, and tell you what happened when something breaks. The search algorithm inside them is often the same algorithm. Almost everything you pay for is the part that is not the algorithm.

### The intuition

Nobody is confused about this in the relational world.

A B-tree is a data structure. There are good B-tree implementations you can import into a program, and if you need an ordered map on disk, you use one. Nobody says "we evaluated Postgres and decided to use a B-tree instead". The question would not parse. Postgres *contains* a B-tree. What you are buying from Postgres is the write-ahead log, the transactions, the connection handling, the query planner, the backups, the replication, and the fact that someone has thought about what happens when the power goes out mid-write.

Vector search has exactly the same shape, and people get it wrong constantly, because the library is famous and the gap is invisible until you are in production.

Here is the gap, concretely. Say you have 1 million text chunks, each embedded into a 384-dimensional vector — an embedding, a list of 384 numbers that positions a piece of text by meaning. In float32, each vector is 384 x 4 = 1,536 bytes. A million of them is about 1.5 GB.

You build a FAISS index. It takes a few minutes. Searches come back in under 10 milliseconds. It is excellent. Then reality arrives, in this order:

1. **The process restarts.** The index was in memory. It is gone. You call `faiss.write_index()` before shutdown next time — but the crash did not give you a shutdown, so you rebuild from scratch, which takes minutes with the service down.
2. **A second worker starts.** It loads its own copy. Now you are using 3 GB, and the two copies cannot see each other's writes.
3. **Someone deletes a document.** Under a General Data Protection Regulation (GDPR) deletion request, you have to actually remove it. FAISS's graph index has no supported removal — there is no way to un-link a node from a Hierarchical Navigable Small World (HNSW) graph, the layered "shortcut network" that makes search fast. Your options are a tombstone list you filter against afterwards, or a full rebuild.
4. **You need to search only what this user may see.** FAISS returns integer ids and distances. It does not store your metadata, so the permission check happens after the search, on the results it gave you. That is post-filtering, and it is where the recall collapse lives.

None of these are FAISS bugs. FAISS is doing precisely what it says: approximate nearest neighbour search, as fast as the hardware allows. The four problems above are database problems, and FAISS is not a database.

### How it actually works

Here is the honest list of what a vector database adds, and what each one costs.

| Capability | Library (FAISS) | Database (Qdrant, pgvector, ...) |
|---|---|---|
| Persistence | `write_index` / `read_index` — a whole-file snapshot, taken when you remember to | Write-ahead log: an acknowledged write survives a crash, and recovery replays the tail |
| Concurrency | Concurrent reads on a frozen index are fine; adding while searching is undefined behaviour | Multi-version concurrency: readers see a consistent snapshot while writers commit |
| Deletes and updates | Not supported on graph indexes; tombstone or rebuild | Tombstone plus background compaction, handled for you |
| Filtering | Vectors and ids only; you keep metadata in a side table and filter afterwards | Payload stored next to the vector, predicates evaluated during traversal |
| Memory ceiling | Index must fit in RAM | Memory-mapped on-disk storage and quantisation, so the index can exceed RAM |
| Replication | None | Shards and replicas, with a stated consistency model |
| Operations | Whatever you write yourself | Health checks, metrics, snapshots, authentication, resource limits, rolling upgrades |
| Failure contract | The process dies and you find out from your users | Documented behaviour per failure, and a status endpoint that says so |

Two of those deserve real detail, because they are where interviews go.

**Filtering, and why it has to be inside the index.** You want the 10 nearest chunks *that this person may read*. There are three ways to arrange that.

```
POST-FILTER          search all 1M  -> top 10 -> drop forbidden -> 0..10 left
                     fast, silent, wrong: shortfall grows as permissions narrow

PRE-FILTER           compute allowed id set -> exact scan of that set -> top 10
                     always correct, cost is linear in the allowed set size

FILTER-AWARE         walk the HNSW graph, skipping nodes that fail the predicate
TRAVERSAL            k results from the permitted subset, at graph-search cost
```

Post-filtering is the one a library forces on you, because the library never saw your predicate. Do the arithmetic on it. A principal who may read 1% of a 1M-chunk corpus asks for k = 10. The global top 10 contains, in expectation, 10 x 0.01 = 0.1 permitted chunks. So the usual outcome is zero evidence, while roughly 10,000 relevant permitted chunks sit unretrieved. The user sees "I don't know" and concludes the assistant is useless. Nothing in the logs says otherwise, because no error occurred.

The standard patch is overfetch: ask for 10,000 candidates and hope 10 survive. It moves the collapse to the right. It does not remove it, and the latency cost is linear in the overfetch factor while the recall gain is not.

Filter-aware traversal is a database feature, not an algorithm feature, because it requires the predicate and the graph to live in the same process with the same view of the data. Qdrant does it by keeping payload indexes alongside the graph and estimating the predicate's cardinality — roughly how many points match — before deciding whether to walk the graph or scan the matching set exactly. pgvector 0.8 added iterative index scans for the same reason: keep pulling from the index until enough rows pass the `WHERE` clause, instead of filtering a fixed candidate batch and returning short.

**Persistence and the consistency model.** The subtle one is not "does it save to disk" but "when is a write visible to a search". Qdrant's upsert API takes a `wait` parameter, and the default is not to block until the point is searchable. That is a reasonable default for throughput and a reliable source of flaky tests and "my document is not there yet" bug reports. Every distributed vector store has a knob like this. Knowing it exists is most of the battle.

**Memory, with numbers.** At 1M chunks x 384 dimensions:

| Store | Bytes per vector | Per million | Why |
|---|---|---|---|
| FAISS `IndexFlat` | 1,536 | ~1.5 GB | Raw matrix, no graph, exact search |
| Qdrant HNSW, m=16 | ~1,700-1,900 | ~1.8 GB | Vector plus ~128 bytes of graph links plus payload |
| Qdrant, int8 quantised | ~500 in RAM | ~0.5 GB | Quantised vectors resident, originals on disk for rescoring |
| pgvector HNSW | ~3,200 | ~3.2 GB | The index stores its own copy of the vector, on top of the table row |

That pgvector row surprises people. An HNSW index in pgvector holds the full vector inside the index tuple, so you pay for the vector twice. `halfvec` (16-bit floats) halves it. It is not a defect; it is the cost of living inside a general-purpose storage engine.

### Where it goes wrong

**Calling FAISS your vector database.** This is the common interview stumble, and the follow-up questions are always the same three: what happens when the pod restarts, how do you delete one document, and how do two replicas agree. If the answer is "we rebuild nightly", that is a legitimate architecture for an offline system and a disqualifying one for a live index.

**Reaching for a database at 50,000 vectors.** At that size the whole matrix is 77 MB. A numpy dot product scans it in single-digit milliseconds and is exact. Adding a stateful service buys you nothing and costs you an on-call rotation.

**Assuming top-k with a filter means k results.** Many engines return fewer, silently, and the number depends on the selectivity of your predicate. Test this deliberately, at 1%, 0.1% and 0.01% permitted, or you will not find out until a customer with an unusual permission shape does.

**Benchmark numbers that do not include filters.** The famous throughput charts are unfiltered. Filtered recall is a different, harder, much less flattering measurement, and it is the one that matters if anything about your query depends on who is asking.

**"We already run Postgres, so pgvector is free."** The extension is free. The HNSW build time, the index bloat, the autovacuum behaviour on a heavily updated vector table, and the fact that a large index competes with your transactional working set for shared buffers are not free.

### In this project

Sightline treats the backend as a replaceable component behind one interface, `VectorStore`, defined in `src/sightline/store/base.py`. Every backend implements the same protocol and passes the same conformance suite, so a disagreement between two of them is a test failure rather than a production incident.

Four implementations exist, with different jobs:

* `src/sightline/store/memory.py` — numpy only, exact cosine similarity, no service to run. The entire test suite runs against it, so the suite has no setup step. It is also the definition of correct behaviour: when Qdrant disagrees with this file, this file wins. Grant tokens are stored as a packed 64-bit bitset per chunk, so the permission filter is a vectorised bitwise AND rather than 70,000 Python set intersections. At 70k chunks x 384 dimensions that is about 103 MB of vectors and under 1 MB of permission bits.
* `src/sightline/store/qdrant_store.py` — the serving path, with the filter pushed into graph traversal.
* the pgvector backend — a second, independent enforcement layer, covered in the next chapter.
* the brute-force FAISS oracle — never served, used to prove the others return the complete correct answer under a filter.
* `src/sightline/store/postfilter.py` — the wrong way, implemented on purpose as the baseline arm of the evaluation. It raises on import inside a serving process, detected by `uvicorn` being in `sys.modules`, and can be overridden only by setting `SIGHTLINE_ALLOW_POSTFILTER=1`, which makes the override a greppable, deliberate act.

Two design decisions are worth naming. `VectorStore` has **no delete method**: a reindex rewrites, and that is stated rather than implied, because half-supported deletion on a graph index is how stale vectors survive. And `upsert` is idempotent on `chunk.id` — the Qdrant backend derives the point id as a version-5 UUID from the chunk id, so re-ingesting the same chunk overwrites one point instead of creating a duplicate that the filter will later have to deal with twice.

There is one more thing the architecture buys. Because of the central rule — the index is a hint, the database is the authority — the vector store is never trusted with a security decision. `search()` returns `UncheckedHit`, and only `recheck()` against the live permission store turns those into `Hit`. That is why a library-grade component like FAISS can sit in the system at all: the oracle has no persistence story, no concurrency story and no delete story, and none of that matters, because it never serves a user.

### Interview questions

**1. What is the difference between FAISS and a vector database?**

FAISS is a library for approximate nearest neighbour search — you give it a matrix of vectors in memory and it returns ids and distances, very fast. A vector database wraps an algorithm like that in the things a service needs: durable storage with a write-ahead log, concurrent readers and writers, metadata stored next to each vector so you can filter, deletes and compaction, replication, and an operational surface with metrics and health checks. The algorithms are often literally the same code; Milvus's engine started as a FAISS fork. So the comparison is not "which searches better", it is "how much of the surrounding system do I want to write myself".

**2. Why can you not filter FAISS results after the search?**

You can, and it is correct in the sense that you never return a forbidden document — but it destroys recall, silently. If a user may see 1% of the corpus and you ask for the global top 10, you expect 0.1 permitted results. So the model gets no evidence and answers "I don't know", while thousands of relevant permitted chunks were never retrieved. Overfetching moves the cliff, it does not remove it, and the latency is linear in the overfetch factor. The failure mode is the dangerous kind: nothing errors, nothing is logged, and it looks like the model is bad rather than the retrieval.

**3. What actually happens when you delete a document from an HNSW index?**

In most implementations, nothing immediately. The node stays in the graph and gets marked deleted, because you cannot cleanly remove a node from a navigable small world graph without breaking the connectivity that makes search work. Results are filtered against the tombstone set at query time, which means your effective k shrinks, and a background process eventually rebuilds or merges segments to reclaim the space. A database does that compaction for you and tells you when it is behind. With a bare library you are writing that yourself, and getting it wrong shows up as deleted documents reappearing after a restart.

**4. How do you decide whether you need a vector database at all?**

Size, write rate, and whether more than one process needs the same view. Under a few hundred thousand vectors, a numpy matrix scan is exact, simple to reason about, and fast enough — 100k by 384 dimensions is about 150 MB and a single matrix multiply. I start reaching for a database when the index stops fitting comfortably in one process's memory, when writes are continuous rather than batch, or when I need filtering that has to happen during the search rather than after it. That last one is the trigger that has nothing to do with scale, and it is the one that applies to permissioned retrieval at any size.

**5. What is the consistency model question you always ask about a vector store?**

When is an acknowledged write visible to a search. Qdrant's upsert does not block on indexing by default, so a write can return 200 and not be findable yet; that is fine for bulk ingest and terrible if a user uploaded a document a second ago and is asking about it. The related question is what happens to in-flight searches during compaction or a segment merge. I ask these because they determine whether "reindex after a permission change" is a second, a minute, or unbounded — and the length of that window is exactly the exposure window of a system that trusts its index.

**6. Your index no longer fits in RAM. What are your options, in order?**

First, quantise: int8 scalar quantisation cuts a 384-dimensional float32 vector from 1,536 bytes to 384, and you keep the originals on disk to rescore the shortlist, so recall loss is small and measurable. Second, memory-map the vectors and let the page cache manage residency, which trades tail latency for footprint. Third, shrink the embedding itself — many models support Matryoshka-style truncation to 256 or 128 dimensions with modest quality loss. Fourth, shard across nodes, which I treat as last because it introduces a distributed system where there was not one. The order matters: the first two are configuration, the last is architecture.

**7. Someone proposes replacing your database-backed index with an in-memory FAISS index rebuilt nightly, for cost reasons. Make the case both ways.**

For: it is genuinely cheaper and simpler, there is no stateful service to operate, rebuild is deterministic, and for a corpus that changes slowly it can be entirely defensible. Against: your worst-case staleness becomes 24 hours, deletion becomes a next-day operation, and a crash means a cold rebuild with the service down. The deciding question is not cost, it is what a stale entry does. In Sightline a stale index can only lose results, never leak them, because everything is rechecked against the live permission store before it reaches the model — so nightly rebuild degrades helpfulness, not safety. In a system that enforces permissions *in* the index, the same proposal is a 24-hour leak window, and I would refuse it.

**8. You inherit a system where the vector store is the source of truth for who can see what. What do you change first, and what do you not touch?**

First I stop the bleeding without a rewrite: add a post-search recheck against the real permission system, so nothing reaches the model that has not been confirmed against the authority, and measure how often the recheck disagrees with the index — that number is the size of the problem and it makes the argument for me. I do not touch the index-side filter, because it is doing useful work: it keeps the candidate set small and cheap, and removing it would hurt recall while the recheck carries correctness. Then I make the dangerous path hard to write again, by changing the types so that search returns something that cannot be shown to a user. The order is deliberate — correctness first with a measurable number, then structure, and performance work last, because a fast wrong answer is the thing I was hired to remove.

---

## Choosing one, with the reasoning

### The short version

There are eight plausible vector stores and most comparisons rank them on speed, which is the criterion that matters least here. For a system where the answer depends on who is asking, the questions are: does filtered search stay accurate, can I isolate tenants, what does a vector cost in memory, how much operational work is it, does it cost money to run, and can I see inside it. Sightline uses three of them at once — Qdrant to serve, pgvector as a second independent check, FAISS as the exact answer to compare against. That is not indecision; each one has a different job, and the fact that they disagree is the point.

### The intuition

Think about how a careful lab weighs something.

There is the scale you use all day: fast, good enough, sitting on the bench. There is a second scale from a different manufacturer, used occasionally, because two instruments that share a design also share a systematic error — if both are calibrated the same way and both drift the same way, agreement between them proves nothing. And there is a certified reference mass in a box, which is slow and inconvenient and never used for actual work, but which tells you whether either scale is lying.

Nobody calls that lab indecisive. The three instruments answer three different questions: what is it, is my instrument wrong, and what is true.

Sightline's three backends map onto those exactly:

| Instrument | Sightline | Question it answers |
|---|---|---|
| Bench scale | Qdrant | What do I serve, fast, with the filter applied during search |
| Second scale, different maker | pgvector with row-level security | Would a bug in my filter code be caught by something that does not share my code |
| Certified reference mass | FAISS exact search | What is the complete correct answer, so recall has a denominator |

The mistake people make is picking one store and then having no way to know if it is wrong. Approximate search cannot grade its own homework. "Recall@10 is 0.94" is a ratio, and the denominator has to come from somewhere that is exact.

### How it actually works

Here are the eight candidates against the criteria that matter for this project. "Inspect internals" means: can I read the source, see why a query chose a plan, and reason about the failure. That criterion is unusual in a buying decision and essential in a portfolio project, because the project's claim is about correctness, and a claim about correctness you cannot inspect is a brand.

| Store | What it is | Filtered search quality | Multi-tenancy | Memory per 384-d vector | Operational burden | Cost at zero budget | Inspect internals |
|---|---|---|---|---|---|---|---|
| **Qdrant** | Rust vector database, single binary | Filter evaluated during HNSW traversal; cardinality estimation picks graph walk vs exact scan | First class: tenant-keyed payload index, per-value subgraphs via `payload_m` | ~1.8 KB; ~0.5 KB with int8 quantisation | Low — one container, no external dependencies | Free, Apache 2.0, runs on a laptop | Yes, open source and the filtering design is documented |
| **pgvector** | Postgres extension | Good since 0.8 iterative index scans; before that, filtered queries returned short | Via row-level security and ordinary `WHERE` clauses — the strongest isolation on the list | ~3.2 KB (index stores a second copy); ~1.7 KB with `halfvec` | Low if you already run Postgres, real if you do not | Free | Yes, plus `EXPLAIN ANALYZE`, which no dedicated vector store matches |
| **Weaviate** | Go vector database with modules | Good; inverted index plus filtered traversal, falls back to flat search at low cardinality | Genuinely first class: tenants are isolated shards, with hot/cold offloading | ~2 KB, more with the object store | Medium — more concepts, more configuration | Free, open source | Yes |
| **Milvus** | Distributed vector database | Good, many index types, partition keys for filter pushdown | Partition keys and collections; designed for it | ~1.8 KB plus cluster overhead | High — full deployment wants etcd, object storage and a message queue | Free, but the cluster is not free to run | Yes, though the storage/compute split is a lot of surface |
| **Chroma** | Embedded, developer-first | Metadata pre-filter then search; fine at small scale, weakest here | Collections only; you build isolation yourself | ~1.6 KB, in process | Very low | Free | Yes, and small enough to read |
| **LanceDB** | Embedded, columnar format on disk/object storage | Improving; disk-first design, filter pushdown into the columnar scan | DIY | ~100 bytes resident with IVF-PQ, rest on disk | Low, no server | Free | Yes, though the format is newer and moving |
| **FAISS** | Library, not a database | None built in — you filter before or after | Not applicable | 1,536 bytes exactly, flat index | None to run, all of it to write | Free | Yes, and it is the reference implementation of most of these algorithms |
| **Pinecone** | Managed service | Good, filters are a first-class part of the query | Namespaces, designed for it | Not exposed | Lowest — it is somebody else's problem | Fails: no free self-hosted tier | No, closed |

A few of those cells carry the decision.

**Qdrant's filtering design.** Sightline's collection is created with `hnsw_config.m = 0` and `hnsw_config.payload_m = 16`. That disables the global HNSW graph entirely and builds a separate subgraph per indexed payload value. A query carrying a grant-token filter then traverses only the subgraphs for tokens it holds. This is the property the product needs: filtered search that stays fast as selectivity increases, instead of degrading into a scan with a predicate bolted on.

The cost is stated plainly. With a multi-valued field like `grant_tokens`, a point joins one subgraph per token it carries, so build time and index memory scale with the average number of tokens per chunk. That is affordable only because of the grant-token design — tokens come from groups, so the count per chunk is small and stable, and it does not grow when people join or leave those groups. A design where chunks were tagged with user ids would make this configuration unusable.

**pgvector's real advantage is not vector search.** It is row-level security: a policy attached to the table that the database enforces on every query, including one that forgot its `WHERE` clause. That is a different kind of guarantee from an application filter. The application filter is code Sightline wrote; a bug in the plan compiler produces a too-wide filter and a second copy of the same code would agree with it enthusiastically. Row-level security is enforced by Postgres, below Sightline's code, on a query Sightline may have built wrong.

**FAISS's advantage is that it is boring.** A flat index is a matrix and a dot product. There is no graph, no approximation, no tuning parameter, nothing to be subtly misconfigured. It gives the true top-k over whatever subset you hand it, at 1,536 bytes per vector and linear time. For an oracle those are the right properties: exactness and predictability, with speed irrelevant because it runs offline.

### Where it goes wrong

**Choosing on published throughput.** The well-known benchmark charts are unfiltered, single-tenant, and run on hardware nobody's budget resembles. Filtered recall under a narrow predicate is a different measurement, rarely published, and the one that predicts whether your permissioned assistant works.

**"Postgres is one less system."** True, and it is not free. An HNSW index in pgvector stores its own copy of every vector, so a million 384-dimensional vectors costs around 3.2 GB across table and index rather than 1.5 GB. Index builds are slow and hold resources. A heavily updated vector table gives autovacuum real work. If your Postgres is already carrying transactional load, the vector index is now competing for the same shared buffers.

**Picking the distributed system first.** Milvus is a serious piece of engineering and it will run rings around everything here at a hundred million vectors. At one million it is several stateful services to operate for a workload that fits in 2 GB. Choosing the scale ceiling you will not reach for two years costs you all of those two years in operational time.

**Managed services you cannot inspect.** For a business, a managed store is often the right call. For a system whose entire claim is "the permission decision is correct and here is the evidence", a component whose filtering behaviour you cannot read is a hole in the argument.

**Running three backends and letting them drift.** This is the genuine risk of Sightline's own choice, and it deserves to be named rather than glossed. Three implementations of "what does this filter plan admit" means three chances to get it subtly different, and a conformance suite that passes on the easy cases and misses the interesting one.

### In this project

The pairing is the trade-off story, so here it is stated directly. Each backend earns its place by doing something the others cannot.

| Backend | Role | Why not one of the others |
|---|---|---|
| Qdrant | Serving | Only candidate on the list that pushes a multi-valued keyword filter into graph traversal and lets you build per-value subgraphs. That single feature is the product |
| pgvector + row-level security | Second, independent enforcement layer | Enforcement happens below the application. A Sightline bug cannot talk Postgres out of a policy. Independence is the point; a second Qdrant would share every assumption |
| FAISS flat | Exact oracle, never served | Recall needs a denominator. Approximate search cannot supply its own. Exact, boring, and offline |
| numpy `MemoryVectorStore` | Reference implementation and test backend | The whole suite runs with no services. When a backend disagrees with it, it is the backend that is wrong |
| Post-filter store | Baseline arm of the evaluation | Producing its recall collapse under measurement is the reason the project exists |

Three deliberate consequences follow.

**The drift risk is answered by having one definition.** Plan semantics — what a `FilterPlan` admits — live in exactly one place, `plan_bindings`, `admits` and `first_matching_token` in `src/sightline/store/memory.py`. Every other backend imports them rather than re-deriving the rules. Two of the fifteen planted permission bugs (M2, an `OR` where an `AND` was meant; M6, an empty token set matching everything) are exactly the mistakes you make once per backend when each backend writes its own rules.

**Independence is tested, not asserted.** The acceptance test for the pgvector backend deliberately issues an unfiltered query and asserts it returns zero forbidden rows, because the database refuses them. That test fails if row-level security is misconfigured, which is the only way to know that the second layer is actually a second layer.

**The oracle is unreachable from the serving path.** A test asserts it. An exact brute-force index that could be served would eventually be served, under load, by someone with good intentions, and its properties are wrong for that.

Optional dependencies keep all of this honest: `qdrant-client`, `psycopg` and `faiss` are extras, every optional import is guarded, and the error names the extra to install. The core package and the entire test suite run with numpy, pydantic, fastapi and httpx. A reviewer can clone the repository and run everything, including the mutation harness, without starting a single service — which is the only reason anyone will actually check the claims.

### Interview questions

**1. How would you pick a vector database?**

I would start from the query shape, not the benchmark. If every query carries a filter — a tenant, a permission, a date range — then filtered recall under a narrow predicate is the first criterion, and it eliminates most options quickly. Then scale and memory budget, then operational burden, which I weight heavily because a store that needs three supporting services is a store somebody has to be paged for. Cost and inspectability come last for a company and first for a project whose whole claim is correctness. I would be suspicious of any decision made from a throughput chart, because those are unfiltered and single-tenant and my workload is neither.

**2. Why Qdrant rather than pgvector for serving?**

The filter. Qdrant lets me disable the global graph and build a subgraph per payload value, so a query with a grant-token filter walks only the subgraphs for the tokens the user holds. That keeps filtered search fast precisely when selectivity is high, which is the case that matters — the user who can see 1% of the corpus. pgvector has improved a lot, and since 0.8 its iterative index scans stop filtered queries returning short, but it is still filter-then-check rather than a filter-shaped index. The other half of the answer is memory: pgvector's HNSW index keeps a second copy of every vector, so a million vectors is about 3.2 GB against Qdrant's 1.8, or under a gigabyte quantised, and my reference machine is an 8 GB laptop.

**3. Then why keep pgvector at all?**

Because it enforces permissions somewhere my code cannot reach. Postgres row-level security is a policy on the table, applied by the database to every query, including one my plan compiler built wrong. A second copy of my own filtering logic would be no help at all — it would share every assumption and agree with every bug. The test that makes this worth having sends a deliberately unfiltered query and asserts it comes back with zero forbidden rows. That is defence in depth in the real sense: two layers that fail for different reasons.

**4. Why does a project need an exact index if it never serves it?**

Because recall is a fraction and somebody has to supply the denominator. If I say filtered recall@10 is 0.94, that means the approximate filtered search found 94% of what exact search over the same permitted subset would have found — and the only way to know the second number is to compute it exactly. FAISS flat is the right tool because it is boring: a matrix, a dot product, no approximation and no tuning knobs to get wrong. It runs offline, so its cost is irrelevant, and a test asserts the serving API cannot reach it, because an index that is exact and available will eventually be served by someone trying to fix a recall complaint.

**5. Is running three vector stores a sign of indecision?**

They have three different jobs: serve, cross-check, and measure. I would not call it indecision to run Postgres in production, a test database in continuous integration, and a spreadsheet of expected results — they are not competing for the same slot. The real cost is drift: three implementations of "what does this filter admit" is three chances to diverge. I handle that by defining plan semantics in exactly one module and having every backend import it, and by running one conformance suite against all of them. If I could not afford that discipline, I would drop to two, and the one I would drop is pgvector — it is the layer that is most valuable and least load-bearing.

**6. What is the strongest argument against your choice?**

That Postgres alone would have been enough and much simpler. One system, real transactions, row-level security, `EXPLAIN ANALYZE`, and a filtering story that is adequate since 0.8 — and the permission tuples are already in a database anyway, so a join is available where I currently make a second round trip. I would have paid for it in memory and in filtered-search latency at high selectivity, and I think that would have bitten at a million chunks on this hardware. But if someone told me to ship in a week with one service, Postgres is the answer, and I would want the recall numbers measured before I believed either of us.

**7. A vendor shows you a benchmark where their store beats Qdrant on recall@10 at higher throughput. What do you ask?**

Whether the queries had filters, and what the selectivity was. Unfiltered recall is not the workload; I care about recall when the predicate admits 1% of the corpus and when it admits 0.01%. Then I ask what recall is measured against — an exact search over the same filtered subset, or the engine's own results at a higher `ef`, because the second is not recall, it is self-consistency. Then dataset size relative to memory, because everything looks good when the index fits in RAM with room to spare. If the filtered numbers exist and hold up, I will take them seriously; usually the reason they are not in the chart is that they are not flattering.

**8. Suppose Qdrant had not existed. Redesign the serving path.**

I would go to Postgres and change the shape of the problem rather than look for another approximate index. Partition the vector table by grant token, so the permission filter becomes partition pruning and the index scan happens only inside partitions the user's tokens touch — that recovers most of what the subgraph design was buying, at the cost of a partition count tied to token cardinality and a painful rebuild when the token scheme changes. For a user with many tokens I would fall back to the exact scan strategy the plan compiler already has, because their permitted set is large enough that a scan is fine. The honest caveat is that this trades a well-tested engine feature for a schema I maintain myself, so I would want the FAISS oracle in place before I wrote a line of it. That is the general lesson: the oracle is what makes it safe to change the serving path at all.
