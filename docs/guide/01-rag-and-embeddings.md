## What retrieval-augmented generation actually is

### The short version

A language model has read a great deal of the public internet and none of your company's files. If you ask it about your own invoices, it has two options: say it does not know, or make something up. Retrieval-augmented generation (RAG) closes that gap by searching your documents first and pasting the best matching passages into the model's prompt, so the model answers from text it can see rather than from memory. Almost everything that goes wrong with these systems goes wrong in the search step, not the writing step.

### The intuition

Imagine hiring a consultant who is articulate, fast, widely read, and has never been inside your building. On day one you ask: "What is our refund policy for enterprise customers who cancel mid-term?"

She has never seen your contracts. But she has read ten thousand other companies' refund policies, so she can produce a fluent, plausible, professional-sounding answer in fifteen seconds. It will use the right vocabulary. It will have the right shape. It will be wrong in the specific ways that matter, because it is a description of the average company, not yours. This is what people mean when they say a model "hallucinates": it is not lying, it is filling a gap with the most likely-looking thing.

Now change one detail. Before she answers, an assistant runs to the filing room, pulls the four pages most relevant to the question, and puts them on her desk. She reads them and answers from them. Same consultant, same fluency, completely different reliability.

That errand is retrieval. RAG is the errand plus the consultant.

The errand has a budget, and the budget is the whole design problem. Say the corpus is one million passages. The model's prompt can hold perhaps 8,000 tokens of evidence — about 10 to 20 passages. So the retriever must go from 1,000,000 candidates down to 10, in roughly 100 milliseconds, and the right passage must be inside those 10. If it is not, nothing later in the pipeline can recover it. The model cannot reason its way to a document it was never shown. It will instead do what the unassisted consultant did: write something plausible from the passages it did get.

This is why the slogan for this chapter is: **RAG is a retrieval problem wearing a generation costume.** The visible output is prose, so the failures look like writing failures. Nearly all of them are search failures.

### How it actually works

A RAG system has two halves that run at different times.

**Ingest** runs offline, once per document version:

```
document ──► chunk ──► embed ──► index
  (PDF,      (~512     (vector    (searchable
   wiki,      tokens,   per        structure +
   ticket)    64 token  chunk)     metadata)
              overlap)
```

**Query** runs online, per question, inside a latency budget:

```
question ──► embed ──► search top-k ──► [filter] ──► rerank ──► build prompt ──► model ──► answer + citations
             ~5 ms      ~25-90 ms                    ~20 ms                     ~1-3 s
```

Stage by stage, with the numbers that matter.

**Chunking.** Whole documents are the wrong unit. A 60-page handbook is one embedding of nothing in particular, and it will not fit in a prompt. So documents are split into passages — in this project, roughly 512 tokens (a token is about 0.75 English words) with a 64-token overlap so a sentence straddling a boundary survives in at least one chunk. Chunk identifiers must be stable: re-ingesting an unchanged document must produce the same chunk ids, or every reindex churns the whole index.

**Embedding.** Each chunk becomes a vector — a fixed-length list of numbers that stands for its meaning. Chapter 2 is entirely about this. For now: same model for chunks and for questions, always.

**Indexing.** The vectors go into a structure that can find near neighbours without comparing against all one million. In practice that is HNSW (Hierarchical Navigable Small World), a layered graph you walk greedily toward the query. It is *approximate*: it can miss a true neighbour. That is a deliberate trade — exact search over a million vectors costs roughly 1.5 GB of arithmetic per query; HNSW answers in single-digit milliseconds by looking at maybe 0.1% of the data.

**Search.** The question is embedded with the same model and the index returns the top *k* nearest chunks, typically k = 10 to 50.

**Filtering.** For a system with permissions, this is where the entire product lives, and where the naive version is wrong. Chapter 5 covers why filtering after search destroys recall.

**Reranking.** An optional second pass: a slower, more accurate model scores each of the 50 candidates against the question directly and keeps the best 10. Cross-encoders (models that read query and passage together rather than comparing two independent vectors) are markedly better at this and roughly 100× slower per pair, which is why they only ever see a shortlist.

**Prompt construction.** The surviving chunks are concatenated, labelled, and delimited, with an instruction to answer only from them and to cite chunk ids.

**Generation and citation.** The model writes the answer. Citations should be reconstructed from the retrieved set by id, not parsed out of the generated text — a model asked to cite will happily invent a plausible reference.

The metric that governs all of this is **recall@k**: of the chunks that genuinely answer the question, what fraction appear in the top *k*? If recall@10 is 0.6, then four questions in ten are being answered from incomplete evidence, and no amount of prompt engineering fixes it. Precision matters less than people expect, because a strong model tolerates two irrelevant passages among ten. A missing passage it cannot tolerate at all.

### Where it goes wrong

Every stage has a characteristic failure, and they all surface as "the AI gave a bad answer".

| Stage | Failure | What the user sees |
|---|---|---|
| Chunking | The answer spans a boundary; half of it is in one chunk | Confidently incomplete answer |
| Chunking | Chunks too large; one topic's signal is diluted by five others | Relevant document never ranks |
| Embedding | Query and documents embedded with different models or versions | Retrieval is near-random, latency looks fine |
| Indexing | Approximate search misses true neighbours under a selective filter | Quiet recall collapse |
| Search | k too small for a question needing evidence from six documents | Partial answer, stated with full confidence |
| Filtering | Permission filter applied after top-k | Recall falls off a cliff for exactly the people with least access |
| Prompting | Retrieved text treated as instructions, not data | Indirect prompt injection |
| Generation | Model answers from prior knowledge when evidence is thin | Fluent, plausible, unsupported |

Four things people get wrong in practice, bluntly.

**They debug the wrong half.** When answers are bad, the instinct is to rewrite the system prompt or switch to a bigger model. Before touching either, look at what was actually retrieved. Most of the time the right passage was not in the context at all, and every hour spent on wording is wasted.

**They have no ground truth.** "It seems better" is not a measurement. You need a set of questions with known correct chunk ids, and an exact-search oracle to define what the top-k should have been. Without that, you cannot distinguish a retrieval regression from a model change.

**They assume more context solves it.** Longer context windows help less than they should. Models attend unevenly across a long prompt — evidence in the middle of 50 passages is measurably less likely to be used than the same evidence at position 2. Retrieving 100 chunks instead of 10 mostly buys noise and latency.

**They let the model fill gaps silently.** If retrieval returns nothing useful, the correct behaviour is refusal. Most systems instead produce a confident answer from parametric memory, which is the single worst failure mode: indistinguishable from a good answer at a glance.

### In this project

Sightline is a RAG system where one extra stage is the point of the whole thing: the permission decision.

The query pipeline is specified in `docs/FRD.md` as FR-18 and runs in this fixed order: compile plan → search → recheck → guardrails → synthesise → reconstruct citations. Two design decisions hold it together.

First, the permission filter is compiled *before* retrieval and rechecked *after* it, so no text retrieved from the corpus can influence which documents are reachable. That closes the escalation path for indirect prompt injection structurally rather than by pattern matching.

Second, the stage boundary is enforced by types, not by discipline. `VectorStore.search()` in `/Users/cook/sightline/src/sightline/store/base.py` returns `UncheckedHit`. The answer builder accepts only `Hit`. The only function that converts one to the other is `recheck()`, which reads the live permission database. There is deliberately no method on `UncheckedHit` that yields a `Hit`, so the unsafe path requires an import that looks wrong in code review.

The output type reflects the refusal discipline. `Answer` in `/Users/cook/sightline/src/sightline/types.py` is grounded or refused, with no third state, and `RefusalReason` enumerates the honest failures: `NO_PERMITTED_EVIDENCE`, `NO_EVIDENCE_AT_ALL`, `UNGROUNDED`, `INJECTION_DETECTED`, `BUDGET_EXCEEDED`, `STALE_POLICY`. Citations are reconstructed from the rechecked set by chunk id; parsing them out of generated text is planted mutant M15 in the mutation-testing suite, which exists precisely because that shortcut is tempting and silent.

### Interview questions

**1. What problem does RAG solve that fine-tuning does not?**

Fine-tuning changes how a model behaves; retrieval changes what it knows right now. If I fine-tune on our documents, I get a model that has absorbed them into its weights, with no way to cite a source, no way to revoke a document, and a retraining cycle every time a policy changes. Retrieval keeps the corpus outside the model, so a document deleted this morning is gone from answers this afternoon, and every claim can point at a chunk id. I would fine-tune for format and tone — making the model answer in our house style — and retrieve for facts. They solve different problems and people conflate them constantly.

**2. Walk me through a RAG pipeline end to end.**

Offline: split documents into chunks of roughly 512 tokens with a small overlap, embed each chunk with a sentence model, and put the vectors in an index alongside metadata. Online: embed the question with the same model, pull the top 10 to 50 nearest chunks, apply any filters, optionally rerank the shortlist with a cross-encoder, paste the survivors into the prompt as delimited data, and have the model answer with citations. The part I would emphasise is that the online path has a hard latency budget — a few hundred milliseconds before the model even starts — so every stage is a speed-versus-recall trade.

**3. Answers are vague and generic. How do you find out why?**

First I look at what was retrieved, not at the prompt. I take the failing questions, dump the top-k chunks, and check by hand whether the answer was present in the context. That splits the problem in two. If the evidence was there and the model ignored it, it is a prompting or grounding problem. If it was not there, it is a retrieval problem and I go further up: was it a chunking issue, an embedding mismatch, a k that is too small, or a filter dropping it. In my experience it is retrieval about four times out of five, and teams burn weeks on prompt wording first.

**4. Why is recall the metric you care about, rather than precision?**

Because the two errors are not symmetric downstream. If I retrieve ten chunks and three are irrelevant, a decent model ignores them — I pay in tokens and a little distraction. If the one chunk that contains the answer is missing, there is no recovery anywhere later in the pipeline; the model will produce something fluent and wrong, and it will look exactly like a good answer. So I track recall@k against an exact-search oracle and treat precision as a cost-control metric. The caveat is that precision does start to matter once you push k high, because attention over long contexts is uneven.

**5. How do you decide chunk size?**

There is no single right answer, and I would say so in an interview. The trade is that small chunks give a clean, focused embedding but split reasoning across boundaries, and large chunks keep context together but dilute the signal so the chunk stops ranking for anything specific. I start around 512 tokens with about a 64-token overlap, then measure recall@10 on a labelled question set at two or three sizes. I would also let the document type drive it — a policy document with numbered clauses wants to be split on clauses, a chat transcript wants to be split on conversational turns. The measurement matters more than the default.

**6. Where does prompt injection enter a RAG system, and what actually stops it?**

The dangerous one is indirect: a document in the corpus contains text telling the assistant to ignore its instructions or fetch other files, and that text arrives in the prompt as trusted context. Delimiting retrieved text and labelling it as data helps, and pattern scanning catches the obvious cases, but neither is a guarantee, because you are asking a model to distinguish data from instruction in the same channel. The real defence is structural: make sure nothing the model reads can change what it is allowed to reach. In Sightline the permission plan is compiled before retrieval and rechecked after it, so there is no code path from chunk text to the filter. A successful injection can make the answer bad. It cannot make it leak.

**7. Your index is a few minutes stale. Argue for and against treating it as authoritative.**

For: it is the only copy the search engine can filter on during traversal, so consulting it is free, and reindexing aggressively shrinks the staleness window to seconds. Against: it never shrinks to zero, and the window is exactly when a revoked user still retrieves. I take the position that a denormalised copy of permission state is a cache, and you never let a cache be the authority on access. So search returns unchecked results, and a recheck against the live database converts them into servable results. The cost is one extra batched permission call on the hot path, and the benefit is that a stale index loses results instead of leaking them. I would rather be briefly unhelpful than briefly unsafe, because only one of those is recoverable.

**8. If you had to remove one stage from a RAG pipeline to halve latency, which goes?**

Reranking, and I would measure the damage before believing it. It is the most expensive stage per unit of recall improvement — a cross-encoder scores query and passage together, so cost grows linearly with shortlist size, and cutting it typically costs a few points of ranking quality rather than recall of the answer-bearing chunk. What I would not remove is the recheck, because it is not a quality stage, it is a correctness stage, and a system with no configuration flag to disable it is easier to defend than one where somebody can turn it off under load. The honest caveat: if the corpus has lots of near-duplicate boilerplate, reranking is doing more work than it appears to, and removing it will show up as answers built from the wrong version of a template.

---

## Embeddings: turning meaning into numbers

### The short version

An embedding turns a piece of text into a list of numbers — a point in space — arranged so that text with similar meaning lands in a similar direction. That lets a computer find "what the company owes a supplier" when the question asked about "outstanding payables", with no shared words. Comparing two pieces of text becomes arithmetic, which is fast enough to do a million times per query. Keyword search did not go away, and for identifiers, product codes and rare terms it still wins.

### The intuition

Think of a map of a city where things are placed by what they are for, not by street address. All the bakeries end up in one region, all the hardware shops in another, all the bicycle repair places in a third. You do not need to know a shop's name to find it. You point at a region and ask what is nearby.

An embedding model draws that map for language. It reads a passage and outputs coordinates. Passages about invoices land near each other. Passages about kayaking land somewhere else entirely. Nobody hand-placed them; the model learned the layout from enormous amounts of text, largely by being trained to put sentences that appeared in similar contexts near each other.

Here is a toy version with real arithmetic. Suppose there are only three dimensions and we can name them: how much the text is about *money*, how much about *paperwork*, how much about *water sports*.

```
invoice = [0.90, 0.40, 0.00]
bill    = [0.80, 0.50, 0.00]
kayak   = [0.00, 0.10, 0.99]
```

Similarity is the cosine of the angle between two vectors: the dot product divided by the two lengths.

```
cos(invoice, bill)  = (0.90·0.80 + 0.40·0.50 + 0) / (0.985 × 0.943) = 0.92 / 0.929 = 0.99
cos(invoice, kayak) = (0 + 0.04 + 0)             / (0.985 × 0.995) = 0.04 / 0.980 = 0.04
```

"Invoice" and "bill" share no letters and score 0.99. That is the whole trick, and it is why this beats keyword matching for paraphrase.

Now the trap that explains normalisation. Take a chunk that says "invoice invoice invoice" — three times the length, same meaning. Un-normalised it is `[2.70, 1.20, 0.00]`, and its raw dot product with `bill` is 2.76 instead of 0.92. Three times bigger, purely because the text is longer. If you rank by dot product without normalising, long documents beat short ones regardless of relevance. Dividing by the length throws away magnitude and keeps direction, which is the part that carries meaning.

### How it actually works

An embedding model is a small transformer. Text goes in as tokens; the model produces one vector per token; a pooling step — usually the mean of the token vectors, masked so padding does not count — collapses those into a single fixed-length vector for the whole passage. That vector is then normalised to unit length.

Once every vector has length 1, cosine similarity and dot product are the same operation:

```
cos(a, b) = (a · b) / (‖a‖ ‖b‖)     and if ‖a‖ = ‖b‖ = 1, cos(a, b) = a · b
```

This matters for engineering reasons, not aesthetic ones. Dot product is one fused multiply-add per dimension with no division and no square roots, so normalising once at ingest removes work from every one of a million comparisons at query time.

Squared Euclidean distance is the same ranking, for unit vectors:

```
‖a − b‖² = ‖a‖² + ‖b‖² − 2(a · b) = 2 − 2 cos(a, b)
```

So cosine 0.99 is distance 0.14, and cosine 0.04 is distance 1.39. Sorting by cosine descending and by Euclidean distance ascending gives identical orders — but only when both vectors are normalised. Mixing normalised and un-normalised vectors in one index is a real and common bug, and it does not throw an error, it quietly degrades ranking.

**Dimensionality and what it costs.** Real models use hundreds or thousands of dimensions, none of them individually meaningful. `all-MiniLM-L6-v2` — six transformer layers, about 22 million parameters — outputs 384 dimensions with a 256-token input limit. The storage arithmetic:

| Representation | Bytes per vector (384-d) | 1M chunks | Relative recall |
|---|---|---|---|
| float32 | 1,536 | 1.54 GB | baseline |
| float16 | 768 | 0.77 GB | ~unchanged |
| int8 scalar quantisation | 384 | 0.38 GB | small loss, usually recoverable by rescoring |
| binary (1 bit/dim) | 48 | 0.05 GB | large loss alone; usable as a first-pass filter |

Add the index structure on top. HNSW with `m = 16` keeps up to 32 neighbour links per node at the base layer, so at 4 bytes per identifier that is roughly 128–200 bytes per vector of graph. A one-million-chunk float32 index is therefore about 1.7 GB resident, which is the number that decides whether this fits on one machine.

Bigger models — 768 or 1,024 dimensions — retrieve better, at double or triple the memory and proportionally more comparison cost. The choice is a straight trade and should be measured on your own corpus, not taken from a leaderboard.

**Why lexical search is still competitive.** BM25 (Best Matching 25) is a keyword scoring function from the 1990s and it remains hard to beat on a large class of queries:

```
score(D, Q) = Σ  IDF(qi) ·        f(qi, D) · (k1 + 1)
              qi          ─────────────────────────────────────
                          f(qi, D) + k1 · (1 − b + b · |D|/avgdl)

IDF(qi) = ln(1 + (N − n(qi) + 0.5) / (n(qi) + 0.5))     k1 ≈ 1.2, b ≈ 0.75
```

Read it in three pieces: a term counts for more if it appears often in this document (`f`), for much more if it is rare across the corpus (`IDF`), and the whole thing is damped so a long document does not win by repetition (`b · |D|/avgdl`).

The `IDF` term is why BM25 wins on rare strings. In a corpus of 1,000,000 chunks, a term appearing in 12 of them scores `ln(1 + 999988.5/12.5) ≈ 11.3`. The word "the", in 900,000 chunks, scores `ln(1 + 100000.5/900000.5) ≈ 0.11` — a hundredfold difference, derived from the data with no training.

Where each one wins:

| Query type | Dense embeddings | BM25 |
|---|---|---|
| "how do we handle refunds mid-contract" | strong — paraphrase | weak if wording differs |
| "form 27B stroke 6" | weak — identifiers carry no learned meaning | strong |
| "Kubernetes CVE-2024-3177" | weak on the identifier | strong |
| Jargon absent from training data | weak | strong |
| Non-English or code-switched text | depends entirely on the model | consistent |
| Short keyword queries | mediocre | strong |
| Long natural-language questions | strong | mediocre |

The practical answer is both. Run the two retrievers and fuse the rankings with Reciprocal Rank Fusion: `score(d) = Σ 1 / (60 + rank_i(d))`, summed over retrievers. It needs no score calibration between systems, which is its main virtue — raw BM25 scores and cosine scores are not on comparable scales and averaging them directly is meaningless.

**The hashing "embedder" is not semantic, and this is worth being precise about.** A hash embedder takes each token, hashes it to a bucket in a 384-dimensional array, increments that bucket, and normalises. It is deterministic, needs no model, runs instantly, and produces vectors of exactly the right shape. Every function signature is satisfied. The pipeline runs end to end and the tests go green.

It has no idea that "invoice" and "bill" are related. They hash to different buckets, so their cosine similarity is 0. Not low — zero, unless a collision makes it accidentally non-zero, which is worse. What you have built is a bag-of-words feature hash with no inverse document frequency weighting and random collisions: strictly worse than BM25, wearing an embedding's clothes.

I shipped this once. The tell is obvious in hindsight and takes thirty seconds to check: embed two paraphrases with no shared tokens and look at the number. A real sentence model gives something in the 0.6 to 0.9 range. A hash embedder gives 0. The reason it survives review is that a hash embedder is genuinely the right tool for one job — making the test suite run deterministically with no model download — and the mistake is letting the test fixture reach the serving path.

### Where it goes wrong

**Similarity is not relevance.** Cosine similarity measures "these look like they are about the same thing". A question and its answer are frequently not about the same thing in that sense. "When does the Denver lease expire?" and a table row reading "Denver, 2027-03-31" are semantically distant. This is the asymmetry problem, and it is why some models have separate query and document encoders or require a prefix such as `query:` on one side. Using such a model without its prefix silently costs several points of recall.

**Cosine scores have no absolute meaning.** Many models never produce similarity below about 0.3 for any pair of English sentences, because the vectors occupy a narrow cone rather than the whole sphere. So a hard threshold like "accept anything above 0.7" is a corpus-specific constant, not a truth about the model, and it will be wrong on the next corpus. Rank, then use a threshold calibrated against labelled data if you need one at all.

**Truncation is silent.** MiniLM takes 256 tokens. Feed it 900 and the tail is discarded, with no warning and no error. The chunk still gets a vector; the vector represents the first third of it.

**Changing the model means rebuilding everything.** Vectors from two models are not comparable, even at the same dimensionality. Upgrading is a full re-embed of the corpus, so the model identifier and version belong in stored metadata, and a mismatch between query-time and index-time models should be a startup failure rather than a quiet ranking collapse.

**Domain shift.** A model trained on web text does not know your internal acronyms. If "MDR" means monthly deal review in your company and medical device regulation on the internet, the model uses the internet's meaning. This is another argument for hybrid retrieval: BM25 has no opinions about what your acronyms mean.

**Embeddings are not a security boundary.** Two facts here. First, an embedding is a lossy but real representation of its source text — inversion attacks can reconstruct a meaningful fraction of the original from the vector alone, so a vector store holding confidential text is itself confidential. Second, and more important for this project, nothing about "how similar is this vector to the query" tells you whether the asker is allowed to read it. Permission is a separate axis and must be enforced by separate machinery.

### In this project

Sightline treats the embedder as a replaceable component behind an interface, with three deliberate choices.

The serving embedder runs under `onnxruntime`, not PyTorch — `docs/FRD.md` FR-16 records why: the reference machine is macOS x86_64, where PyTorch ships no wheels past 2.2.x. The `embed` extra in `/Users/cook/sightline/pyproject.toml` pins `onnxruntime` and `tokenizers`, and the import is guarded so a missing extra raises an error naming it rather than failing at first query.

The deterministic hash embedder exists on purpose, and its purpose is narrow: the full test suite must run with core dependencies only and no model download. That is a plumbing fixture, not a retriever. Given the mistake described above, it carries an explicit non-semantic marker and the serving application refuses to start with it — the honest version of the lesson is that the fixture was fine and the missing guard was the defect.

Vectors are normalised at ingest, so every backend compares by dot product. That matters most for the exact oracle in the store package: it computes exact top-k over the permitted subset with numpy alone (FR-11) and is the ground truth behind every recall figure in `eval/`, including the recall@10 ≥ 0.95 floor asserted across permission densities of 0.1%, 1%, 10%, 50% and 100%.

Finally, the split that the rest of the guide depends on: nothing about permission is encoded in the vector. Permission lives in the grant tokens stamped on the `Chunk` dataclass in `/Users/cook/sightline/src/sightline/types.py`, derived from group edges as keyed hashes, and in the live tuple store that `recheck()` consults. The vector answers "is this relevant". The tokens and the database answer "may this person see it". Mixing those two questions into one score is how leaks get built.

### Interview questions

**1. Explain an embedding to someone non-technical.**

It is a way of giving every piece of text a position on a map, where the position is decided by meaning rather than by spelling. Text about the same topic ends up in the same neighbourhood, so a computer can find related documents by looking at what is nearby, even when the words are completely different. The coordinates themselves are not readable — there are a few hundred of them and none corresponds to anything you could name. What matters is relative position, not the numbers.

**2. Why normalise vectors before storing them?**

Two reasons. The semantic one: magnitude mostly encodes length and repetition, not meaning, so an un-normalised dot product systematically favours longer chunks. The engineering one: once everything is unit length, cosine similarity collapses to a plain dot product, which is one fused multiply-add per dimension with no division and no square root. I normalise once at ingest and save that work on every comparison at query time. It also means cosine ranking and Euclidean ranking agree, so I can switch distance metrics between backends without the results changing order.

**3. What does a 384-dimension float32 vector cost, and how does that scale?**

384 floats at 4 bytes each is 1,536 bytes per vector, so a million chunks is about 1.5 GB of raw vectors. HNSW adds roughly 128 to 200 bytes per vector for the neighbour graph, so call it 1.7 GB resident — that is the number that decides single-machine versus sharded. If I need it smaller, int8 scalar quantisation takes it to 384 bytes per vector, about a quarter, and I recover most of the ranking loss by rescoring the top few hundred candidates against full-precision vectors. I would measure that recall cost rather than assume it.

**4. When does BM25 beat a dense retriever?**

Whenever the query contains a rare exact string: part numbers, error codes, ticket ids, internal acronyms, people's surnames. An embedding model has no learned meaning for "form 27B/6", so the vector is near arbitrary, while BM25's inverse document frequency term makes a token appearing in 12 chunks out of a million score about a hundred times higher than a common word — derived from the corpus with no training at all. It also beats dense retrieval on any vocabulary the embedding model never saw. That is why I run both and fuse with reciprocal rank fusion rather than picking a side: RRF needs no score calibration, which matters because BM25 scores and cosine scores are not on a comparable scale.

**5. You have a hashing embedder that produces 384-d vectors and all the tests pass. What is wrong?**

It is not semantic. It hashes tokens into buckets and counts them, so two paraphrases with no shared words have cosine similarity of exactly zero — it is a bag-of-words feature hash without inverse document frequency weighting, which makes it strictly worse than BM25 while having the shape of an embedding. I know this because I shipped it once. The check takes thirty seconds: embed "invoice due" and "bill payable" and look at the number. A real sentence model gives something around 0.7; a hash gives 0. The fixture itself is legitimate for making tests run without a model download — the defect was that nothing stopped it reaching the serving path, so now it declares itself non-semantic and the app refuses to start with it.

**6. A stakeholder wants a "confidence score" from cosine similarity. What do you tell them?**

That the raw number is not calibrated and I should not hand it over as-is. Many sentence models never produce similarity below about 0.3 for any two English sentences, because the vectors sit in a narrow cone rather than spread over the sphere, so 0.55 does not mean "moderately confident", it means "nothing in particular". A threshold that works on one corpus will be wrong on the next. If they need a gate, I would build it from labelled data — pick a cut-off that hits a target precision on a held-out set — and re-derive it whenever the model or corpus changes. Failing that, I would use rank position, or a cross-encoder score, which at least correlates with relevance.

**7. How would you decide between a 384-dimension and a 1,024-dimension model?**

There is no universally right answer and it depends on the corpus, so I would run the comparison rather than argue it. I build a labelled question set with known answer chunks, index the corpus both ways, and compare recall@10 against an exact-search oracle, alongside memory footprint and p95 query latency. 1,024 dimensions costs 2.7× the memory and proportionally more comparison work; if it buys two points of recall on my data, that is usually not worth it, and if it buys fifteen, it clearly is. I would also check truncation limits, because a model with a 512-token window that matches my chunk size can beat a nominally stronger model that silently discards half of every chunk.

**8. Your retrieval quality dropped overnight with no code deploy. How do you diagnose it?**

First I check whether the query-time and index-time embedding models still match, because that is the failure that produces near-random retrieval with perfectly healthy latency and no errors. I store the model identifier and version alongside the vectors precisely so this is a one-query check rather than an investigation, and a mismatch should fail at startup. If they match, I look at whether the corpus changed — a bulk ingest of boilerplate can flood the neighbourhood around common queries and push real answers out of the top 10 without anything breaking. Then I rerun the labelled evaluation set against the exact oracle to separate "the index is missing things" from "the ranking got worse", because those have completely different fixes. The thing I would not do first is change the prompt, since nothing in the prompt can explain an overnight change with no deploy.
