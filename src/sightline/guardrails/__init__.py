"""The five controls that sit *behind* the permission layer, never instead of it.

None of these is access control. Access control is
:mod:`sightline.authz`, it happens before retrieval and again after it, and
every guardrail in this package runs on the **rechecked** set — hits that
already passed :func:`sightline.authz.recheck.recheck`. If a guardrail here
looks like it is deciding who may see something, it has been miswired.

``docs/FRD.md`` section 7 lists fourteen numbered guards. Here is where each one
actually lives, because "we have guardrails" is not a claim anyone can check:

=====  ==========================  ======================================
Guard  Name                        Implemented in
=====  ==========================  ======================================
1      Stale plan guard            ``authz.compile`` / the query pipeline
2      Recheck guard               ``store.base`` types + ``authz.recheck``
3      Empty-permitted-set guard   the query pipeline
4      Existence protection        :mod:`.grounding`
5      Injection scanner           :mod:`.injection`
6      Instruction isolation       :mod:`.injection` (``context_block``)
7      Grounding check             :mod:`.grounding` (``support_score``)
8      Citation reconstruction     :mod:`.grounding` (``citations_for``)
9      Context budget              :mod:`.budget`
10     Depth limit                 ``authz.check``
11     Cycle guard                 ``authz.check``
12     Token opacity               the HTTP layer (``/v1/plan``)
13     Oracle drift check          ``authz.oracle``
14     No-flag guard               a test, not code
=====  ==========================  ======================================

:mod:`.pii` is not on that list. It is hygiene rather than a guard: it removes
credentials and card numbers at ingest, and it keeps document text out of logs,
traces and error payloads at egress. Redaction is not authorisation and this
package is careful never to let it be used as one.

THE HONEST SUMMARY
------------------
The scanner is a pattern matcher. The grounding check is lexical overlap. The
PII detector is regexes with checksums. Each is a speed bump, each is described
as one, and no detection rate is published for any of them anywhere in this
repo. What actually stops an injected document from leaking the board deck is
that the board deck was never retrieved, because the filter was compiled from
group tuples before the query was read and there is no code path from chunk text
back to that filter. These modules are defence in depth behind that property,
not a substitute for it.

Ordering, when wiring a pipeline:

1. Budget check (:func:`sightline.guardrails.budget.decide`) — before any
   provider call, on the complete rechecked set.
2. Injection scan (:func:`sightline.guardrails.injection.scan_retrieved`) — on
   the same rechecked set; refuse on an indirect HIGH finding.
3. Context assembly (:func:`sightline.guardrails.injection.context_block`).
4. Grounding (:func:`sightline.guardrails.grounding.build_answer`) — on the
   synthesiser's claims, reconstructing citations from the rechecked set.
5. Egress redaction (:func:`sightline.guardrails.pii.redact_for_log`) on
   anything about to be logged or traced.
"""

from __future__ import annotations

from sightline.guardrails import budget, grounding, injection, pii
from sightline.guardrails.budget import (
    Budget,
    BudgetDecision,
    Rung,
    TokenLedger,
    clamp_k,
    estimate_tokens,
    extractive_answer,
    refuse_over_budget,
)
from sightline.guardrails.grounding import (
    Claim,
    ClaimVerdict,
    GroundingResult,
    build_answer,
    citations_for,
    existence_floor,
    refuse,
    refuse_no_evidence,
    support_score,
)
from sightline.guardrails.injection import (
    InjectionVerdict,
    Severity,
    Source,
    context_block,
    sanitise,
    scan_chunk,
    scan_query,
    scan_retrieved,
)
from sightline.guardrails.pii import (
    PiiKind,
    Redaction,
    redact_for_egress,
    redact_for_ingest,
    redact_for_log,
)

__all__ = [
    # submodules, for the call sites where the qualified name reads better
    "injection",
    "pii",
    "grounding",
    "budget",
    # injection
    "Severity",
    "Source",
    "InjectionVerdict",
    "scan_query",
    "scan_chunk",
    "scan_retrieved",
    "sanitise",
    "context_block",
    # pii
    "PiiKind",
    "Redaction",
    "redact_for_ingest",
    "redact_for_egress",
    "redact_for_log",
    # grounding
    "Claim",
    "ClaimVerdict",
    "GroundingResult",
    "support_score",
    "citations_for",
    "build_answer",
    "refuse",
    "refuse_no_evidence",
    "existence_floor",
    # budget
    "Budget",
    "BudgetDecision",
    "Rung",
    "TokenLedger",
    "clamp_k",
    "estimate_tokens",
    "extractive_answer",
    "refuse_over_budget",
]
