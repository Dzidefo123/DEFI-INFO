from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.graph import investigation, nodes
from src.graph.state import AgentState
from src.intelligence.plan import InvestigationPlan
from src.intelligence.query_types import QueryType, is_investigation

# Guardrail action -> terminal node. Deliberately a total mapping: a new
# guardrail action must name its destination here or KeyError at wiring time,
# rather than silently falling through to the router.
_GUARD_EXIT = {
    "refuse_secret": "guard_reply",
    "refuse_scope": "guard_reply",
    "refuse_injection": "guard_reply",
    "escalate": "escalate",
}


def _after_guard(state: AgentState) -> str:
    action = state.get("guardrail_action")
    if action is None:
        return "route"
    return _GUARD_EXIT[action]


_CX_EXIT = {
    "docs": "retrieve",
    "live_data": "live_data",
    "account_action": "escalate",
    "out_of_scope": "refuse",
}


def _after_route(state: AgentState) -> str:
    """Depth first, then intent.

    Consulting `query_type` alone is safe here only because `route` already ran
    it through `effective_query_type`: a terminal intent cannot arrive carrying
    an investigation classification, so there is no ordering hazard where an
    account_action gets investigated instead of escalated. That clamp is what
    lets this function stay one line rather than a matrix of both axes.

    Defaults to CX when the classification is absent. `guard` seeds it on every
    path, so absence means something upstream is broken — and the safe response
    to that is the cheap path, not a crashed turn. Failing closed here would take
    down a working support agent over a missing label.
    """
    if is_investigation(state.get("query_type") or QueryType.CX.value):
        return "plan"
    return _CX_EXIT[state["intent"]]


def _after_plan(state: AgentState) -> list[str]:
    """Fan out to exactly the specialists the plan named.

    Returns a LIST, which is how LangGraph schedules parallel branches — the §2
    diagram's three agents running side by side rather than in sequence. They
    write disjoint result channels and append to the shared ones through
    `accumulate`, so concurrent writes neither race nor overwrite.
    """
    plan = InvestigationPlan.model_validate(state["investigation_plan"])
    return [investigation.AGENT_NODES[a] for a in plan.agents]


def _after_grade(state: AgentState) -> str:
    if state["docs"]:
        return "generate"
    # Nothing survived grading. Rewrite once; if that also fails, the KB
    # genuinely doesn't cover this and a human should take it.
    if state.get("attempts", 0) < nodes.MAX_ATTEMPTS:
        return "rewrite"
    return "escalate"


def _after_verify(state: AgentState) -> str:
    if state["grounded"]:
        return "finalize"
    if state.get("attempts", 0) < nodes.MAX_ATTEMPTS:
        return "rewrite"
    return "escalate"


def _guard_reply(state: AgentState) -> dict:
    """No-op terminal: `guard` already wrote the canned answer into state."""
    return {}


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)

    for name in (
        "guard", "route", "retrieve", "grade", "rewrite",
        "generate", "verify", "finalize",
        "live_data", "escalate", "refuse",
    ):
        g.add_node(name, getattr(nodes, name))
    g.add_node("guard_reply", _guard_reply)

    # Deterministic guardrails run before any model sees the message.
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", _after_guard, ["route", "guard_reply", "escalate"])

    for name in ("plan", "research_agent", "blockchain_agent", "security_agent",
                 "risk_engine", "verify_claims", "evidence_graph", "report"):
        g.add_node(name, getattr(investigation, name))

    # The explicit path maps are not decoration: they let LangGraph validate
    # every branch target at compile time and render the real topology,
    # instead of resolving returned node names at runtime.
    g.add_conditional_edges(
        "route", _after_route, ["retrieve", "live_data", "escalate", "refuse", "plan"]
    )

    # Investigation branch (§2). Agents fan out in parallel and fan back in on
    # `risk_engine`, which waits for every branch that was actually scheduled.
    #
    # `risk_engine` and `verify_claims` are unconditional NODES that no-op when
    # the plan does not call for them, rather than conditional edges out of each
    # agent. One static shape is far easier to read in a rendered graph, and it
    # keeps the "which stages ran" decision in the plan — one place — instead of
    # spread across three identical branch functions.
    g.add_conditional_edges(
        "plan", _after_plan,
        ["research_agent", "blockchain_agent", "security_agent"],
    )
    for agent in ("research_agent", "blockchain_agent", "security_agent"):
        g.add_edge(agent, "risk_engine")
    g.add_edge("risk_engine", "verify_claims")
    # The graph is assembled AFTER verification, so its claim nodes carry the
    # verdicts rather than a status the report would have to patch in later.
    g.add_edge("verify_claims", "evidence_graph")
    g.add_edge("evidence_graph", "report")

    # Self-corrective RAG loop: retrieve -> grade -> generate -> verify,
    # falling back to rewrite-and-retry, then to a human.
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _after_grade, ["generate", "rewrite", "escalate"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", _after_verify, ["finalize", "rewrite", "escalate"])

    for terminal in ("finalize", "live_data", "escalate", "refuse", "guard_reply",
                     "report"):
        g.add_edge(terminal, END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
