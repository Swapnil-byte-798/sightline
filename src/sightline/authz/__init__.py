"""Authorisation: the part of Sightline that is allowed to say no.

Read the modules in this order; each one only makes sense after the previous:

1. :mod:`~sightline.authz.tuples` — storage. Zanzibar-style tuples, namespace
   configs with usersets, and the monotonic policy epoch. Two backends, one
   Protocol, and two indexes because the forward and reverse questions have
   genuinely different shapes.
2. :mod:`~sightline.authz.check` — the authority. A bounded, cycle-safe,
   memoised graph walk that returns a decision *and its derivation*. Slow by
   design; correct by requirement.
3. :mod:`~sightline.authz.compile` — the permission compiler. One principal
   becomes a small set of grant tokens plus a strategy, so a query can carry
   authorisation without carrying 40,000 document ids.
4. :mod:`~sightline.authz.recheck` — the guarantee. ``UncheckedHit`` in, ``Hit``
   out, live tuples consulted, no flag to disable it.
5. :mod:`~sightline.authz.oracle` — the proof. Plan versus authority over a
   stratified sample, reporting false allows and false denies separately
   because one is a breach and the other is a missing search result.

The invariant that ties them together: **the index is a hint, the database is
the authority**. Steps 3 and 5 exist because step 2 is too slow to run per
candidate; step 4 exists because step 3 is a copy, and copies go stale.
"""

from sightline.authz.check import (
    DEPTH_LIMIT_NOTE,
    MAX_OBJECT_DEPTH,
    MAX_SUBJECT_DEPTH,
    MAX_USERSET_DEPTH,
    Explanation,
    UsersetTree,
    check,
    expand,
    expand_leaves,
    explain,
    implying_relations,
    tuple_to_userset_rules,
)
from sightline.authz.compile import (
    ENUMERATE_MAX,
    EXACT_SCAN_MAX,
    MAX_ENUMERATION,
    UNFILTERED_MIN_RATIO,
    Closure,
    PlanCache,
    PlanCompiler,
    compile_plan,
    derive_grant_token,
    grant_token_key,
    grant_tokens_for_object,
    plan_admits,
    principal_closure,
    reachable_objects,
)
from sightline.authz.oracle import (
    FALSE_ALLOW_TARGET,
    FALSE_DENY_TARGET,
    OraclePair,
    OracleResult,
    Stratum,
    run_oracle,
)
from sightline.authz.recheck import LiveRechecker, recheck
from sightline.authz.tuples import (
    DEFAULT_NAMESPACES,
    ComputedUserset,
    MemoryTupleStore,
    NamespaceConfig,
    Rewrite,
    SQLiteTupleStore,
    This,
    TupleStore,
    TupleStoreStats,
    TupleToUserset,
    Union,
    parse_tuples,
)

__all__ = [
    # tuples
    "This",
    "ComputedUserset",
    "TupleToUserset",
    "Union",
    "Rewrite",
    "NamespaceConfig",
    "DEFAULT_NAMESPACES",
    "TupleStore",
    "TupleStoreStats",
    "MemoryTupleStore",
    "SQLiteTupleStore",
    "parse_tuples",
    # check
    "check",
    "expand",
    "expand_leaves",
    "explain",
    "implying_relations",
    "tuple_to_userset_rules",
    "UsersetTree",
    "Explanation",
    "MAX_USERSET_DEPTH",
    "MAX_OBJECT_DEPTH",
    "MAX_SUBJECT_DEPTH",
    "DEPTH_LIMIT_NOTE",
    # compile
    "compile_plan",
    "PlanCompiler",
    "PlanCache",
    "plan_admits",
    "derive_grant_token",
    "grant_token_key",
    "grant_tokens_for_object",
    "Closure",
    "principal_closure",
    "reachable_objects",
    "ENUMERATE_MAX",
    "EXACT_SCAN_MAX",
    "UNFILTERED_MIN_RATIO",
    "MAX_ENUMERATION",
    # recheck
    "recheck",
    "LiveRechecker",
    # oracle
    "run_oracle",
    "OracleResult",
    "OraclePair",
    "Stratum",
    "FALSE_ALLOW_TARGET",
    "FALSE_DENY_TARGET",
]
