from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.graph import nodes
from src.graph.state import AgentState

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


def _after_route(state: AgentState) -> str:
    return {
        "docs": "retrieve",
        "live_data": "live_data",
        "account_action": "escalate",
        "out_of_scope": "refuse",
    }[state["intent"]]


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

    # The explicit path maps are not decoration: they let LangGraph validate
    # every branch target at compile time and render the real topology,
    # instead of resolving returned node names at runtime.
    g.add_conditional_edges(
        "route", _after_route, ["retrieve", "live_data", "escalate", "refuse"]
    )

    # Self-corrective RAG loop: retrieve -> grade -> generate -> verify,
    # falling back to rewrite-and-retry, then to a human.
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _after_grade, ["generate", "rewrite", "escalate"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", _after_verify, ["finalize", "rewrite", "escalate"])

    for terminal in ("finalize", "live_data", "escalate", "refuse", "guard_reply"):
        g.add_edge(terminal, END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
