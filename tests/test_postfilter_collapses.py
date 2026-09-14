"""The baseline arm must actually exhibit the bug we claim to have measured.

``PostFilterStore`` implements the wrong design on purpose: retrieve the global
top-k, then drop what the principal may not see. The README publishes how badly
that performs as permission density falls, and a bug you rely on has to be tested
like any other behaviour — if the baseline ever stopped losing recall, the
headline chart would be measuring nothing and nobody would notice, because the
chart would still render.

Recall here is measured against ground truth computed **in this file** with
numpy, not against another store. Comparing two implementations proves they
agree; comparing one against arithmetic proves it is right.

One caution the whole project depends on: ``MemoryVectorStore`` is an exact brute
scan, which is the only situation where "filtered top-k equals exact top-k" may
be asserted as an equality. The approximate backends get a recall *floor*
(``eval.selectivity.RECALL_FLOOR``) and never an equality, because HNSW is
approximate and a hard equality there is a test that eventually lies.
"""

from __future__ import annotations

import numpy as np
import pytest
from _world import make_corpus

from sightline.store.memory import MemoryVectorStore
from sightline.types import FilterPlan, GrantToken, PlanStrategy, PrincipalRef

postfilter = pytest.importorskip(
    "sightline.store.postfilter", reason="the baseline arm is not present yet"
)
PostFilterStore = postfilter.PostFilterStore

DIM, K, TRIALS = 32, 10, 25
N_DOCS = 1_000
GROUPS = [f"g{i}" for i in range(20)]  # one group = 5% of the corpus


@pytest.fixture(scope="module")
def corpus():
    return make_corpus(N_DOCS, GROUPS, dim=DIM)


@pytest.fixture(scope="module")
def stores(corpus):
    chunks, vectors = corpus
    pre = MemoryVectorStore(dim=DIM)
    post = PostFilterStore(dim=DIM)
    over = PostFilterStore(dim=DIM, overfetch=10, name="postfilter_x10")
    for store in (pre, post, over):
        store.upsert(chunks, vectors)
    return pre, post, over


def _plan(n_groups: int) -> FilterPlan:
    return FilterPlan(
        principal=PrincipalRef("user", "dana"),
        strategy=PlanStrategy.GRANT_TOKENS,
        grant_tokens=frozenset(GrantToken(g) for g in GROUPS[:n_groups]),
        epoch=1,
    )


def _queries(n: int = TRIALS, seed: int = 11) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(size=DIM).astype("float32") for _ in range(n)]


def _truth(corpus, query: np.ndarray, n_groups: int, k: int = K) -> list[str]:
    """Exact top-k over the permitted subset, computed from first principles."""
    chunks, vectors = corpus
    allowed = set(GROUPS[:n_groups])
    rows = [i for i, c in enumerate(chunks) if set(map(str, c.grant_tokens)) & allowed]
    if not rows:
        return []
    mat = np.asarray([vectors[i] for i in rows], dtype=np.float64)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    q = np.asarray(query, dtype=np.float64)
    q /= np.linalg.norm(q)
    scores = mat @ q
    order = np.argsort(-scores)[:k]
    return [chunks[rows[int(i)]].id for i in order]


def _measure(store, corpus, n_groups: int) -> tuple[float, float]:
    """Mean recall@k and mean number of results returned, over the query set."""
    plan = _plan(n_groups)
    recalls: list[float] = []
    returned: list[int] = []
    for query in _queries():
        hits = store.search(query, plan, K)
        truth = _truth(corpus, query, n_groups)
        returned.append(len(hits))
        if truth:
            found = {h.chunk_id for h in hits} & set(truth)
            recalls.append(len(found) / len(truth))
    return float(np.mean(recalls)), float(np.mean(returned))


# --------------------------------------------------------------------------
# The correct arm, so the collapse has something to collapse from
# --------------------------------------------------------------------------


def test_prefiltered_search_is_exactly_right(stores, corpus):
    """Filter-then-score returns the true top-k of the permitted subset.

    Asserted as an equality only because this store is an exact brute scan. The
    same assertion against an approximate backend would be wrong even when the
    backend is behaving.
    """
    pre, _, _ = stores
    recall, returned = _measure(pre, corpus, n_groups=1)
    assert recall == 1.0
    assert returned == float(K)


def test_prefilter_fills_k_at_every_density(stores, corpus):
    pre, _, _ = stores
    for n in (1, 2, 5, 10, 20):
        _, returned = _measure(pre, corpus, n_groups=n)
        assert returned == float(K)


# --------------------------------------------------------------------------
# The collapse
# --------------------------------------------------------------------------


def test_postfilter_collapses_at_low_visibility(stores, corpus):
    """One group in twenty is 5% visibility. The baseline returns far under k.

    This is the number the README leads with, and the reason "just filter
    afterwards" is not an implementation detail: at 5% visibility the global top
    ten contains about half a permitted document, so the user asks for ten
    results and gets zero or one.
    """
    _, post, _ = stores
    recall, returned = _measure(post, corpus, n_groups=1)
    assert returned < K / 4, f"expected a collapse, got {returned:.1f} of {K} results"
    assert recall < 0.25, f"expected recall to collapse, got {recall:.3f}"


def test_postfilter_recovers_at_full_visibility(stores, corpus):
    """The two arms agree when the principal can see everything.

    Without this, a post-filter that was simply broken would also pass the test
    above and we would be publishing a chart of our own defect.
    """
    pre, post, _ = stores
    assert _measure(post, corpus, 20) == _measure(pre, corpus, 20)


def test_collapse_is_monotonic_in_visibility(stores, corpus):
    """More visibility must never return fewer results."""
    _, post, _ = stores
    counts = [_measure(post, corpus, n)[1] for n in (1, 2, 5, 10, 20)]
    assert counts == sorted(counts), counts


def test_overfetching_ten_times_does_not_fix_it(stores, corpus):
    """The usual production patch buys a constant factor against a 1/density problem.

    It helps — that is why people ship it and believe the problem is solved — and
    it still does not reach the permitted top-k. Measuring what the patch is
    worth is more useful than asserting it is worthless.
    """
    pre, post, over = stores
    plain, _ = _measure(post, corpus, 1)
    patched, patched_returned = _measure(over, corpus, 1)
    exact, _ = _measure(pre, corpus, 1)
    assert patched > plain, "overfetching should help, or the arm is misconfigured"
    assert patched < exact, "overfetching must not reach the permitted top-k"
    assert patched_returned < K


def test_the_shortfall_is_reported_not_inferred(stores, corpus):
    """The baseline reports its own trace, so the chart is not reverse-engineered."""
    _, post, _ = stores
    hits, trace = post.search_with_trace(_queries(1)[0], _plan(1), K)
    assert trace.requested_k == K
    assert trace.returned == len(hits)
    assert trace.shortfall == K - len(hits) > 0
    assert trace.retrieved == K, "the baseline retrieved globally, as it is supposed to"


def test_the_baseline_never_returns_a_forbidden_chunk(stores, corpus):
    """Wrong about recall, and only about recall.

    The baseline is a *correct* store in the sense that it never shows a chunk the
    plan forbids; it is an incorrect one in the sense that matters. Keeping those
    two apart is what makes the evaluation attributable.
    """
    chunks, _ = corpus
    _, post, _ = stores
    permitted_ids = {c.id for c in chunks if str(GROUPS[0]) in set(map(str, c.grant_tokens))}
    for query in _queries(10):
        for hit in post.search(query, _plan(1), K):
            assert hit.chunk_id in permitted_ids


def test_the_baseline_refuses_to_load_in_a_serving_process():
    """The wrong arm is fenced off from production by an import-time check.

    A module you must not serve is a comment in most repositories. Here it is a
    guard that reads ``sys.modules`` for a server runtime and raises.
    """
    assert postfilter.NOT_FOR_PRODUCTION is True
    with pytest.MonkeyPatch.context() as mp:
        import sys

        mp.setitem(sys.modules, "uvicorn", object())
        mp.delenv(postfilter.ALLOW_ENV_VAR, raising=False)
        with pytest.raises(RuntimeError, match="baseline"):
            postfilter.assert_not_serving()
