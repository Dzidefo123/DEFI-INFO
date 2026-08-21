"""The investigation branch: planner, specialist agents, risk, verification, report.

This is the §2 pipeline as LangGraph nodes. The topology, the state contract and
the report were built and tested before any specialist existed, so each agent
landed into a shape already known to work rather than defining it on the way
past. All of them are real now: research (§8), blockchain (§9), security (§10),
the risk engine (§11) and verification (§13).

**Nothing here reports an all-clear it did not earn.** An agent returning "no
incidents found" when it searched nothing would be the single most dangerous line
of code in this repository: indistinguishable downstream from a real negative
finding, and strictly more confident than the evidence permits. Every path that
comes up empty records a LIMITATION instead, and the vocabulary built in earlier
phases makes that the path of least resistance — `UNKNOWN` severity is already
not `NORMAL`, an unsupported claim already scores zero, and a report with no
verified findings already refuses to conclude. `_not_implemented` remains as the
shape a future agent starts from.

Two rules about who may assert what, both load-bearing:

- The **blockchain agent produces no claims**. Whether a reading is abnormal is
  decided by the risk engine from arithmetic, so no claim about behaviour rests
  on the collecting code's opinion of it.
- The **risk engine's own computation is evidence**. A claim quoting a baseline
  and a z-score cites figures that exist in no raw reading, and without a node
  for the computation those numbers have no provenance — which verification
  correctly flags.

State crossing: nodes hold `src.evidence.models` objects in memory and write
plain dicts into channels. See `AgentState` for why that boundary is not
negotiable.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from src.evidence.confidence import assess
from src.evidence.models import (
    AgentName,
    Claim,
    ClaimKind,
    Evidence,
    EvidenceKind,
    EvidenceLink,
    SourceRef,
    SourceTier,
    Stance,
    VerificationStatus,
)
from src.graph.state import AgentState
from src.intelligence.plan import InvestigationPlan
from src.obs.metrics import timed
from src.reports.intelligence_report import (
    coverage_gaps,
    render_report,
    report_payload,
)

# --- the state boundary -------------------------------------------------


def dump(model: BaseModel) -> dict[str, Any]:
    """Model -> plain dict for a state channel. `mode="json"` so datetimes become
    ISO strings and enums their values, leaving nothing msgpack has to guess at."""
    return model.model_dump(mode="json")


def parse_claims(state: AgentState) -> list[Claim]:
    return [Claim.model_validate(c) for c in state.get("claims") or []]


def parse_evidence(state: AgentState) -> list[Evidence]:
    return [Evidence.model_validate(e) for e in state.get("evidence") or []]


def parse_plan(state: AgentState) -> InvestigationPlan:
    return InvestigationPlan.model_validate(state["investigation_plan"])


# --- planner ------------------------------------------------------------


@timed
def plan(state: AgentState) -> dict:
    """Record what will be attempted, and clear the previous investigation.

    The reset is explicit — `accumulate` treats `None` as "start empty" — because
    these channels persist in the checkpoint and would otherwise let one
    investigation inherit the last one's evidence. `guard` does the same for the
    CX channels; this is the same discipline one layer in.
    """
    from src.intelligence.plan import build_plan

    built = build_plan(
        question=state.get("original_question", state["question"]),
        query_type=state["query_type"],
        protocols=state.get("protocols") or [],
    )
    return {
        "investigation_plan": dump(built),
        "evidence": None,
        "claims": None,
        "errors": None,
        "risk_signals": [],
        "verification": {},
        "evidence_graph": {},
        "final_report": {},
        "research_results": {},
        "blockchain_results": {},
        "security_results": {},
    }


# --- specialist agents (stubs) -----------------------------------------


def _not_implemented(agent: AgentName, capability: str) -> dict:
    """A specialist that cannot run yet, saying so in the one place it matters.

    Returns no claims. A stub that invented a reassuring one would be
    indistinguishable, downstream, from a real finding — and every consumer of
    `claims` is built to trust that a claim means somebody looked.
    """
    return {
        "errors": [f"{agent.value}: not implemented — {capability}"],
        "claims": [],
        "evidence": [],
    }


@timed
def research_agent(state: AgentState) -> dict:
    """§8. Decomposes the question, retrieves across angles, extracts evidence.

    The one place the CX pipeline and the investigation path share machinery:
    both go through `hybrid_search`, so the measured retrieval quality is the
    same on both. What differs is that this returns linked claims rather than
    prose.
    """
    from src.agents import research

    plan_ = parse_plan(state)
    result = research.investigate(
        question=plan_.question,
        protocols=plan_.protocols,
    )
    return {
        "claims": [dump(c) for c in result.claims],
        "evidence": [dump(e) for e in result.evidence],
        "errors": list(result.limitations),
        "research_results": {
            "status": "ok",
            "sub_queries": [dump(q) for q in result.sub_queries],
            "evidence_count": len(result.evidence),
            "claim_count": len(result.claims),
        },
    }


@timed
def blockchain_agent(state: AgentState) -> dict:
    """§9. Collects current on-chain readings and pairs them with their history.

    Produces evidence, never claims. Whether a reading is abnormal is decided by
    `risk_engine` from the arithmetic, so no claim about behaviour rests on this
    agent's opinion of it.
    """
    from src.agents import blockchain

    plan_ = parse_plan(state)
    result = blockchain.investigate(
        protocols=plan_.protocols,
        subjects=_markets_in_scope(state),
    )
    return {
        "claims": [],
        "evidence": [dump(e) for e in result.evidence],
        "errors": list(result.limitations),
        "blockchain_results": {
            "status": "ok",
            # The contract `risk_engine` reads. Keyed by the metric name that is
            # actually scored, which for a cumulative counter is its derived rate.
            "metrics": {
                s.metric: {
                    "current": s.current,
                    "history": list(s.history),
                    "protocol": s.protocol,
                    "subject": s.subject,
                }
                for s in result.series
            },
            "readings": len(result.evidence),
            "scoreable": sum(1 for s in result.series if s.scoreable),
        },
    }


def _markets_in_scope(state: AgentState) -> tuple[str, ...]:
    """Which markets to read, when the protocol's metrics are per-market.

    Uses the ticker the router already extracted. Falls back to nothing rather
    than to a default market: collecting BTC for a question that never mentioned
    it would file readings under an investigation they have no bearing on.
    """
    coin = state.get("coin")
    return (coin,) if coin else ()


@timed
def security_agent(state: AgentState) -> dict:
    """§10. Searches the curated incident registry and the protocol's own
    security documentation, keeping the four classifications strictly apart."""
    from src.agents import security
    from src.security.incidents import counts_by_classification

    plan_ = parse_plan(state)
    result = security.investigate(protocols=plan_.protocols)
    counts = counts_by_classification(result.records)
    return {
        "claims": [dump(c) for c in result.claims],
        "evidence": [dump(e) for e in result.evidence],
        "errors": list(result.limitations),
        "security_results": {
            "status": "ok",
            # Reported per category, never as one total. A single "3 findings"
            # is exactly the merge §10 forbids.
            "by_classification": {c.value: n for c, n in counts.items()},
            "established": sum(1 for r in result.records if r.is_established),
            "documentation_sections": sum(
                1 for e in result.evidence if e.payload.get("security_documentation")
            ),
        },
    }


AGENT_NODES: dict[str, str] = {
    AgentName.RESEARCH.value: "research_agent",
    AgentName.BLOCKCHAIN.value: "blockchain_agent",
    AgentName.SECURITY.value: "security_agent",
}


# --- risk engine --------------------------------------------------------


@timed
def risk_engine(state: AgentState) -> dict:
    """§11. Scores collected metrics against their own history.

    Runs unconditionally as a node and no-ops when the plan does not call for it,
    so the graph keeps one static shape instead of a conditional edge out of every
    agent. The engine itself is built and tested (`src.risk`); what it lacks is
    data, because the feature store arrives with the blockchain agent. A metric
    with no history is `UNKNOWN`, never `NORMAL` — the distinction that stops a
    blind spot reading as a clean bill of health.
    """
    plan_ = parse_plan(state)
    if not plan_.risk_engine:
        return {"risk_signals": []}

    metrics = (state.get("blockchain_results") or {}).get("metrics") or {}
    if not metrics:
        return {
            "risk_signals": [],
            "errors": [
                "risk_engine: no metrics were collected, so nothing could be "
                "scored against a baseline"
            ],
        }

    from src.risk.signals import assess_metric, explain

    evidence = parse_evidence(state)
    signals, claims, derived = [], [], []

    for name, series in metrics.items():
        signal = assess_metric(
            metric=name,
            current_value=series["current"],
            history=series.get("history") or [],
            protocol=series.get("protocol"),
        )
        signals.append(dump(signal))

        # Anomalies become claims HERE, not in the collecting agent. Every claim
        # about abnormal behaviour therefore traces to a computation someone can
        # redo by hand, rather than to an agent's impression of a number — which
        # is §11.2's rule ("the LLM explains the signal; it does not calculate
        # it") applied to who is allowed to assert one.
        if not signal.anomaly:
            continue
        supporting = _evidence_for_metric(evidence, name, series.get("subject", ""))
        if not supporting:
            # A signal that cannot be tied back to a stored reading. Dropped
            # rather than asserted: `link_claims` refuses uncited claims from the
            # Research Agent, and a claim from the risk engine gets no exemption.
            continue

        # The computation itself is evidence. A claim like "gas is 55 sigma above
        # a baseline median of 2.3 over 10 observations" quotes figures that exist
        # nowhere in the raw reading — the baseline, the z-score and the fence are
        # derived from history that is not itself in the evidence pool. Without
        # this node those numbers have no provenance, and verification correctly
        # flags them as stated-but-unevidenced.
        computed = _signal_evidence(signal, series)
        derived.append(dump(computed))

        claims.append(
            dump(
                Claim(
                    text=explain(signal),
                    agent=AgentName.RISK_ENGINE,
                    protocols=(series["protocol"],) if series.get("protocol") else (),
                    # A signal asserts what a metric currently IS, against its own
                    # history. Documentation cannot speak to that at all.
                    kind=ClaimKind.STATE,
                    links=tuple(
                        EvidenceLink(evidence_id=e.evidence_id, stance=Stance.SUPPORTS)
                        for e in (*supporting, computed)
                    ),
                )
            )
        )

    return {"risk_signals": signals, "claims": claims, "evidence": derived}


def _signal_evidence(signal, series: dict) -> Evidence:
    """A computed risk signal, as an evidence node.

    Tier is PRIMARY: this is arithmetic over readings the system holds, and it is
    reproducible from them by anyone. Its `observed_at` is the reading's, not the
    moment of computation — the signal is about when the metric was that value.
    """
    from src.risk.signals import explain

    base = signal.baseline
    return Evidence(
        kind=EvidenceKind.STATISTICAL_SIGNAL,
        source=SourceRef(
            tier=SourceTier.PRIMARY,
            uri="internal://risk-engine",
            protocol=series.get("protocol"),
            title="Statistical risk engine",
            locator=signal.signal_id,
        ),
        agent=AgentName.RISK_ENGINE,
        summary=explain(signal),
        payload={
            "metric": signal.metric,
            "subject": series.get("subject", ""),
            "current_value": signal.current_value,
            "baseline_median": base.median,
            "baseline_mean": base.mean,
            "observations": base.n,
            "z": signal.z,
            "modified_z": signal.modified_z,
            "severity": signal.severity.value,
        },
        observed_at=signal.observed_at,
    )


def _evidence_for_metric(
    evidence: list[Evidence], scored_name: str, subject: str
) -> list[Evidence]:
    """The stored readings a signal was computed from.

    `scored_name` may be a derived name: a cumulative counter is scored as
    `<metric>_rate` while the evidence recorded for it is the counter itself. So
    the match strips that suffix rather than comparing names directly.
    """
    base = scored_name[: -len("_rate")] if scored_name.endswith("_rate") else scored_name
    return [
        e
        for e in evidence
        if e.payload.get("metric") == base and e.payload.get("subject", "") == subject
    ]


# --- verification -------------------------------------------------------


@timed
def verify_claims(state: AgentState) -> dict:
    """§13. Challenges every claim the specialists were prepared to make.

    Semantic entailment is left off by default: it is one model call per
    surviving claim, and the deterministic checks already reject most of what
    would fail. When it is off, `verify_all` says so as a limitation rather than
    letting the report imply the claims were checked for relevance.
    """
    from src.agents import verification

    claims = parse_claims(state)
    evidence = parse_evidence(state)
    result = verification.verify_all(claims, evidence, entailer=_entailer())

    return {
        "verification": {
            "verdicts": [
                {
                    "claim_id": v.claim_id,
                    "status": v.status.value,
                    "confidence": v.confidence,
                    "checks_failed": [c.check for c in v.failures],
                    "summary": v.summary(),
                }
                for v in result.verdicts
            ],
            "claims_examined": len(claims),
            "by_status": result.by_status,
            "unsupported": sum(1 for c in claims if c.is_unsupported()),
        },
        "errors": list(result.limitations),
    }


def _entailer():
    """The semantic check, when it is switched on.

    Off unless `settings.verify_entailment` is set. It costs one model call per
    claim that survived the free checks, which is the right thing to pay for on
    an investigation someone is acting on and the wrong thing to pay on every
    turn by default. The cost of leaving it off is disclosed in the report, not
    hidden.
    """
    from src.agents import verification
    from src.config import settings

    return verification.llm_entails if settings.verify_entailment else None


# --- report -------------------------------------------------------------


@timed
def evidence_graph(state: AgentState) -> dict:
    """§14. Assemble the investigation record as a traversable graph.

    A stage of its own rather than something the report builds inline, because
    what it produces is queryable state: "why did you conclude that" should be
    answerable after the fact, from a checkpoint, without re-running anything.
    """
    from src.evidence import graph as eg

    plan_ = parse_plan(state)
    claims = apply_verdicts(parse_claims(state), state.get("verification") or {})
    built = eg.build(
        claims=claims,
        evidence=parse_evidence(state),
        risk_signals=state.get("risk_signals") or [],
        protocols=plan_.protocols,
        question=plan_.question,
    )

    shared = eg.independence(built)
    groups = eg.independent_claim_groups(built)

    errors: list[str] = []
    if shared:
        # The analysis the flat record cannot do. Reported as a limitation
        # because it bounds what the findings are entitled to claim.
        errors.append(
            f"evidence_graph: {len(claims)} claim(s) rest on {len(groups)} "
            f"independent line(s) of evidence — "
            + "; ".join(
                f"{len(s.claim_ids)} claims share the source {s.label!r}"
                for s in shared[:3]
            )
        )

    return {
        "evidence_graph": eg.to_state(built)
        | {
            "shared_sources": [s.model_dump(mode="json") for s in shared],
            "independent_groups": [list(g) for g in groups],
            "mermaid": eg.to_mermaid(built),
        },
        "errors": errors,
    }


def apply_verdicts(claims: list[Claim], verification: dict[str, Any]) -> list[Claim]:
    """Stamp each claim with the verdict verification reached about it.

    Verification cannot write this back onto the `claims` channel itself: that
    channel uses the `accumulate` reducer, so a write appends rather than
    replaces and the claims would double. Keeping the record append-only is the
    right shape anyway — `claims` is what was asserted, `verification` is a
    separate layer of judgement about it — but the two must be recombined before
    anything reads a claim's standing. Without this the report saw every claim as
    UNVERIFIED and reported a well-evidenced investigation as producing nothing.
    """
    by_id = {v["claim_id"]: v["status"] for v in verification.get("verdicts") or []}
    return [
        claim.model_copy(
            update={"verification": VerificationStatus(by_id[claim.claim_id])}
        )
        if claim.claim_id in by_id
        else claim
        for claim in claims
    ]


@timed
def report(state: AgentState) -> dict:
    """§15. Renders the investigation record as a structured artifact.

    Deterministic: the report is assembled from state, not written by a model.
    That keeps it reproducible and free, and — more importantly — makes it
    impossible for the prose to claim more than the record contains, which is the
    failure the whole architecture is arranged against.
    """
    verification = state.get("verification") or {}
    plan_ = parse_plan(state)
    answer = render_report(
        plan=plan_,
        claims=apply_verdicts(parse_claims(state), verification),
        evidence=parse_evidence(state),
        risk_signals=state.get("risk_signals") or [],
        verification=verification,
        limitations=state.get("errors") or [],
        gaps=coverage_gaps(
            plan_,
            state.get("risk_signals") or [],
            state.get("security_results") or {},
        ),
        graph=state.get("evidence_graph") or {},
    )
    question = state.get("original_question", state["question"])
    return {
        "answer": answer,
        # Prose and data, rendered from the same record so they cannot disagree.
        # The structured half is what the evaluation harness will read.
        "final_report": {
            "markdown": answer,
            "structured": report_payload(
                plan=plan_,
                claims=apply_verdicts(parse_claims(state), verification),
                evidence=parse_evidence(state),
                risk_signals=state.get("risk_signals") or [],
                verification=verification,
                limitations=state.get("errors") or [],
                gaps=coverage_gaps(
                    plan_,
                    state.get("risk_signals") or [],
                    state.get("security_results") or {},
                ),
                graph=state.get("evidence_graph") or {},
            ),
        },
        "citations": [],
        "messages": [HumanMessage(question), AIMessage(answer)],
    }
