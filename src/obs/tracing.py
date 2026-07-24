"""LangSmith tracing setup.

LangSmith is named explicitly in the Coinbase JD, and it earns the slot: a
self-corrective RAG loop is genuinely hard to debug from logs. When an answer
escalates, the question is *which* stage gave up — did retrieval return nothing,
did the grader reject good chunks, or did the verifier reject a fine answer? A
trace shows the whole tree with per-node inputs and outputs; a log line shows
"escalated: ungrounded".

Tracing is opt-in via env and degrades to a no-op, so the agent never fails
because an observability backend is down.
"""

from __future__ import annotations

import os


def init_tracing() -> bool:
    """Enable LangSmith if configured. Returns whether tracing is on."""
    if os.getenv("LANGSMITH_TRACING", "").lower() not in ("1", "true", "yes"):
        return False
    if not os.getenv("LANGSMITH_API_KEY"):
        return False

    # LangChain reads these; setting them here keeps the wiring in one place.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"])
    os.environ.setdefault(
        "LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "hyperliquid-cx-agent")
    )
    return True
