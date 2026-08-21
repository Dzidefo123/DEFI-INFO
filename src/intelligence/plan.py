"""The investigation plan — what the Intelligence Manager decided to run.

§5 lists seven questions the Manager must answer. Four of them are settled by the
classification alone, deterministically, and this module settles them: which
agents are required, whether the risk engine applies, whether verification
applies, and what the investigation is scoped to. §3.3's rule — use AI for
ambiguity, deterministic systems for facts — puts all four on this side of the
line. The genuinely ambiguous ones (is the evidence sufficient yet, is another
pass needed) belong to later phases and to a model.

Planning is therefore free: no model call, no tokens, and the same question
always produces the same plan, which is what makes an investigation reproducible
enough to argue with.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.evidence.models import AgentName
from src.intelligence.query_types import QueryType, requirements_for


class InvestigationPlan(BaseModel):
    """A record of what was going to be attempted, written before it is.

    Kept in state so a report can say what the system SET OUT to do, not only
    what it managed. The difference between "security was investigated and found
    nothing" and "security was never investigated" is the whole point of §13, and
    it is unrecoverable after the fact unless the intent was recorded up front.
    """

    model_config = ConfigDict(frozen=True)

    question: str
    query_type: str                       # QueryType value; plain str, see AgentState
    agents: tuple[str, ...] = ()          # AgentName values
    protocols: tuple[str, ...] = ()
    risk_engine: bool = False
    verification: bool = False
    notes: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def is_scoped_to_protocols(self) -> bool:
        return bool(self.protocols)

    def runs(self, agent: AgentName) -> bool:
        return agent.value in self.agents


# Investigating without a protocol in scope is possible but weak: on-chain
# metrics and security history are per-protocol, so an unscoped investigation can
# only really do research. Recorded as a note rather than blocked, so the report
# can state the limitation instead of the system silently doing less.
_UNSCOPED_NOTE = (
    "No protocol was identified, so on-chain and security findings cannot be "
    "scoped to one; only documentary research applies."
)


def build_plan(
    question: str, query_type: str, protocols: list[str] | None = None
) -> InvestigationPlan:
    """Derive the plan from the classification. Pure, and free.

    `query_type` is expected to have already passed through
    `effective_query_type`, so a terminal intent never reaches here as an
    investigation.
    """
    reqs = requirements_for(query_type)
    protocols = tuple(protocols or ())

    notes: list[str] = []
    if not protocols and reqs.agents != (AgentName.RESEARCH,):
        notes.append(_UNSCOPED_NOTE)

    return InvestigationPlan(
        question=question,
        query_type=QueryType(query_type).value,
        agents=tuple(a.value for a in reqs.agents),
        protocols=protocols,
        risk_engine=reqs.risk_engine,
        verification=reqs.verification,
        notes=tuple(notes),
    )
