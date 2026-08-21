"""How much machinery a question needs — §5.1's classification axis.

The Intelligence Manager's first job is to decide whether a request is a support
question or an investigation. This module holds that decision's vocabulary and
its deterministic consequences: which specialists a classification implies, and
which classifications the safety layer is allowed to override.

**Depth is a separate axis from safety, and safety wins.** `intent`
(docs / live_data / account_action / out_of_scope) already decides whether the
agent may act at all. `QueryType` decides only how deeply it investigates when it
may. Keeping them apart is what stops "run a full investigation into why my
wallet was drained" from routing around the account_action escalation: the depth
axis cannot promote a request past a terminal safety branch, because
`effective_query_type` clamps it. That clamp is code, not a prompt instruction,
because a prompt instruction is exactly the thing a crafted question is trying to
talk its way around.

**The default is CX, and that is a load-bearing default.** §21: a simple user
question should stay simple. Investigation machinery is slower, costlier, and
makes more claims that must then be verified — none of which makes a
documentation lookup more correct. Classification errors toward CX cost a
follow-up question; errors toward FULL_INVESTIGATION cost a multi-agent run on
something the docs already answered.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from src.evidence.models import AgentName


class QueryType(str, Enum):
    """§5.1. What kind of work a request implies."""

    CX = "cx"                                  # ordinary support: docs or one live lookup
    RESEARCH = "research"                      # documented context from several angles
    BLOCKCHAIN_ANALYSIS = "blockchain_analysis"  # is on-chain behaviour unusual?
    SECURITY_ANALYSIS = "security_analysis"    # vulnerabilities, incidents, audits
    RISK_ASSESSMENT = "risk_assessment"        # exposure spanning behaviour + posture
    FULL_INVESTIGATION = "full_investigation"  # open question, no single source answers


class Requirements(BaseModel):
    """What a classification commits the system to running.

    Derived from the pipelines drawn in §5.1, kept as data so the planner in the
    next phase reads a table instead of re-deriving a policy in prose.
    """

    model_config = ConfigDict(frozen=True)

    agents: tuple[AgentName, ...] = ()
    risk_engine: bool = False
    verification: bool = False

    @property
    def is_investigation(self) -> bool:
        return bool(self.agents)


# The §5.1 execution paths, as a table.
#
# RESEARCH and SECURITY_ANALYSIS do not run the risk engine: it scores numeric
# series against their own history, and neither of those produces one. Handing it
# a document count would manufacture a z-score out of something that never varied
# for a measurable reason.
#
# Every investigation runs verification. CX does not — not because its answers
# need less checking, but because it already has its own `verify` node inside the
# self-corrective RAG loop, and that stage is grounding-checking a generated
# answer rather than adjudicating claims against evidence.
REQUIREMENTS: dict[QueryType, Requirements] = {
    QueryType.CX: Requirements(),
    QueryType.RESEARCH: Requirements(
        agents=(AgentName.RESEARCH,), verification=True
    ),
    QueryType.BLOCKCHAIN_ANALYSIS: Requirements(
        agents=(AgentName.BLOCKCHAIN,), risk_engine=True, verification=True
    ),
    QueryType.SECURITY_ANALYSIS: Requirements(
        agents=(AgentName.SECURITY,), verification=True
    ),
    QueryType.RISK_ASSESSMENT: Requirements(
        agents=(AgentName.BLOCKCHAIN, AgentName.SECURITY),
        risk_engine=True,
        verification=True,
    ),
    QueryType.FULL_INVESTIGATION: Requirements(
        agents=(AgentName.RESEARCH, AgentName.BLOCKCHAIN, AgentName.SECURITY),
        risk_engine=True,
        verification=True,
    ),
}

# Intents that terminate before any investigation could run. `account_action`
# goes to a human because it touches funds; `out_of_scope` is declined. Neither
# outcome changes because a question was phrased as an investigation, so the
# depth axis is discarded on these branches rather than honoured.
_TERMINAL_INTENTS = frozenset({"account_action", "out_of_scope"})


def requirements_for(query_type: QueryType) -> Requirements:
    return REQUIREMENTS[query_type]


def is_investigation(query_type: QueryType) -> bool:
    """True when the classification calls for specialist agents."""
    return requirements_for(query_type).is_investigation


def effective_query_type(intent: str, query_type: QueryType) -> QueryType:
    """The classification after the safety axis has had its say.

    A terminal intent collapses any depth to CX. The point is not that such a
    request *is* a support question — it is that no investigation runs on it, and
    saying so in one place beats every downstream consumer remembering to check
    `intent` before honouring `query_type`.

    "Investigate whether my drained wallet was an exploit" is the case this
    exists for. It reads as FULL_INVESTIGATION and is an account_action, and the
    account_action must win: the deterministic guardrail already escalates
    compromise reports, and the depth axis must not be able to re-route past it.
    """
    if intent in _TERMINAL_INTENTS:
        return QueryType.CX
    return query_type


def required_agents(intent: str, query_type: QueryType) -> tuple[AgentName, ...]:
    """Which specialists this turn should run, after safety clamping."""
    return requirements_for(effective_query_type(intent, query_type)).agents
