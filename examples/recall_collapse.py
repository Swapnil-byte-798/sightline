#!/usr/bin/env python3
"""Why permission filtering breaks vector search, in 60 lines.

This is the whole argument for Sightline, reduced to something you can run.

Two stores hold the same 4,000 chunks. One searches only what the principal is
allowed to see (pre-filter). The other does what nearly every retrieval-augmented
generation tutorial ships: search everything, then drop the results the principal
cannot see (post-filter).

    python examples/recall_collapse.py

The post-filter column is not a bug in the implementation. It is the correct
behaviour of that design, and it is invisible in production because a missing
result logs nothing at all.
"""

from __future__ import annotations

import numpy as np

from sightline.store.memory import MemoryVectorStore
from sightline.store.postfilter import PostFilterStore
from sightline.types import Chunk, FilterPlan, GrantToken, ObjectRef, PlanStrategy, PrincipalRef

DIM, N_CHUNKS, N_GROUPS, TRIALS, K = 64, 4_000, 40, 40, 10


def build_corpus(rng: np.random.Generator) -> tuple[list[Chunk], list[np.ndarray]]:
    """One group per chunk, round-robin. Deliberately the friendliest possible case."""
    groups = [f"g{i}" for i in range(N_GROUPS)]
    chunks, vectors = [], []
    for i in range(N_CHUNKS):
        chunks.append(
            Chunk(
                id=f"c{i}",
                object=ObjectRef("doc", str(i)),
                text=f"document {i}",
                grant_tokens=frozenset({GrantToken(groups[i % N_GROUPS])}),
            )
        )
        vectors.append(rng.normal(size=DIM).astype("float32"))
    return chunks, vectors


def main() -> None:
    rng = np.random.default_rng(42)
    chunks, vectors = build_corpus(rng)
    groups = [f"g{i}" for i in range(N_GROUPS)]

    print(f"corpus: {N_CHUNKS} chunks, {N_GROUPS} groups, {DIM} dimensions, k={K}\n")
    print(f"{'visible':>8} {'pre-filter':>12} {'post-filter':>12} {'recall':>9}")
    print("-" * 45)

    for n in (1, 2, 4, 20, 40):
        plan = FilterPlan(
            principal=PrincipalRef("user", "dana"),
            strategy=PlanStrategy.GRANT_TOKENS,
            grant_tokens=frozenset(GrantToken(g) for g in groups[:n]),
            epoch=1,
        )
        pre, post = MemoryVectorStore(dim=DIM), PostFilterStore(dim=DIM)
        pre.upsert(chunks, vectors)
        post.upsert(chunks, vectors)

        n_pre = n_post = 0
        for _ in range(TRIALS):
            q = rng.normal(size=DIM).astype("float32")
            n_pre += len(pre.search(q, plan, K))
            n_post += len(post.search(q, plan, K))

        recall = n_post / n_pre if n_pre else 0.0
        print(
            f"{100 * n / N_GROUPS:7.1f}% {n_pre / TRIALS:11.1f} "
            f"{n_post / TRIALS:12.1f} {recall:9.3f}"
        )

    print(
        "\npre-filter   search only what the principal may see\n"
        "post-filter  search everything, then drop what they may not see\n\n"
        "At 2.5% visibility a request for 10 results returns 0.2. The assistant then\n"
        "says it does not know, about a document sitting in the asker's own folder."
    )


if __name__ == "__main__":
    main()
