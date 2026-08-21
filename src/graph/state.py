from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages

Intent = Literal["docs", "live_data", "account_action", "out_of_scope"]


def accumulate(left: list | None, right: list | None) -> list:
    """Append-with-reset reducer for channels several agents write concurrently.

    Specialist agents fan out in parallel and each appends to the shared
    `evidence` / `claims` / `errors` channels, so those need a reducer or the
    last writer silently wins. Plain `operator.add` would do that — except it
    also accumulates across TURNS, since channels persist in the checkpoint.
    An investigation would then inherit the previous investigation's evidence and
    report claims it never gathered.

    That is not hypothetical: `escalation_reason` was a sticky channel for the
    same reason, and every turn after an escalation reported itself escalated.
    Here the stakes are higher, so the reducer takes `None` as an explicit reset
    and the planner clears every investigation channel at the start of a turn.
    """
    if right is None:
        return []
    return (left or []) + right

# What the deterministic guardrail layer decided, before any model ran.
GuardrailAction = Literal[
    "refuse_secret", "escalate", "refuse_scope", "refuse_injection"
]


class AgentState(TypedDict, total=False):
    """Shared state passed between nodes.

    `messages` uses the add_messages reducer so conversational turns append;
    every other key is last-write-wins within a turn.
    """

    messages: Annotated[list, add_messages]

    question: str
    original_question: str  # preserved across rewrite, for tracing and citations
    intent: Intent
    protocols: list[str]    # whitelisted protocol keys the query concerns; [] = all
    coin: str | None

    # How much machinery the question needs (§5.1). A separate axis from
    # `intent`, which decides whether the agent may act at all. Already clamped
    # by `effective_query_type`, so a terminal intent reads CX here regardless of
    # what the router proposed.
    #
    # A plain `str`, holding a `QueryType` VALUE — never the enum member itself.
    # LangGraph checkpoints state through msgpack, which flags any class it does
    # not know ("Deserializing unregistered type ... will be blocked in a future
    # version") even when, as here, the class subclasses `str`. That is the same
    # failure that already cost this project once, when a metrics object lived in
    # a state channel and every persisted multi-turn conversation carried it.
    # `Intent` above is a Literal of bare strings for the same reason. `QueryType`
    # is a str-Enum, so a bare value compares, hashes, and indexes the
    # requirements table identically — nothing downstream needs to convert.
    query_type: str

    # Guardrails
    guardrail_rule: str | None
    guardrail_action: GuardrailAction | None

    docs: list[Document]
    answer: str
    citations: list[str]

    grounded: bool
    attempts: int
    escalation_reason: str | None

    # --- investigation channels (§7) ------------------------------------
    #
    # Everything here is PLAIN dicts and lists, never Pydantic models. Under
    # `LANGGRAPH_STRICT_MSGPACK` a model in a channel is refused outright, and
    # without it the model is silently rehydrated as a `dict` — so a node reading
    # `state["evidence"][0].summary` works on the turn that wrote it and raises
    # AttributeError on the next one. `src.evidence.models` stays the in-memory
    # vocabulary; `dump`/`parse` in `src.graph.investigation` cross the boundary.

    investigation_plan: dict[str, Any]

    # Per-agent raw output, each written by exactly one node, so no reducer.
    research_results: dict[str, Any]
    blockchain_results: dict[str, Any]
    security_results: dict[str, Any]

    # Shared, written concurrently by fanned-out agents. See `accumulate`.
    evidence: Annotated[list[dict[str, Any]], accumulate]
    claims: Annotated[list[dict[str, Any]], accumulate]
    errors: Annotated[list[str], accumulate]

    risk_signals: list[dict[str, Any]]
    verification: dict[str, Any]
    # §14's graph, flattened to node/edge lists. Kept in state rather than built
    # at render time so "why did you conclude that" stays answerable from a
    # checkpoint, without re-running the investigation.
    evidence_graph: dict[str, Any]
    final_report: dict[str, Any]
