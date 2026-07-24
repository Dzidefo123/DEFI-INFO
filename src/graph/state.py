from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages

Intent = Literal["docs", "live_data", "account_action", "out_of_scope"]

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

    # Guardrails
    guardrail_rule: str | None
    guardrail_action: GuardrailAction | None

    docs: list[Document]
    answer: str
    citations: list[str]

    grounded: bool
    attempts: int
    escalation_reason: str | None
