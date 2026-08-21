"""LangSmith tracing setup.

Both graph paths are hard to debug from logs, for the same underlying reason:
failure is a property of the route taken, not of any single node. When a CX
answer escalates, the question is *which* stage gave up — did retrieval return
nothing, did the grader reject good chunks, or did the verifier reject a fine
answer? A log line says "escalated: ungrounded"; a trace shows the whole tree
with per-node inputs and outputs.

The investigation path makes this sharper. Agents fan out in parallel, so their
log lines interleave and stop being readable as a sequence, and a thin report is
usually one agent returning nothing rather than the synthesis failing. A trace
keeps each branch separate and attributes the gap to the branch that caused it.

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
        "LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "defi-info")
    )
    return True
