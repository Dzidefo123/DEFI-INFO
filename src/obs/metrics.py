"""Per-stage latency and token/cost accounting.

"Cost per conversation" and "latency per stage" are the two numbers that decide
whether a support agent ships. A self-corrective RAG loop makes 3+ LLM calls per
answer (route, grade x N chunks, generate, verify) — the grader is usually the
surprise, because it scales with chunk count, not question count.

Timing is captured by wrapping each node; token usage is captured by a callback
handler that attributes every LLM call to whichever node is on the stack.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from langchain_core.callbacks import BaseCallbackHandler

# USD per 1M tokens. Keep in sync with the model actually configured.
PRICING = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def rate_for(model: str) -> tuple[float, float]:
    """Price for a model id, matched on prefix.

    The API returns a dated id for some models and a bare one for others —
    `claude-haiku-4-5-20251001` against `claude-opus-4-8`. An exact-match table
    prices the dated ones at zero, which is the worst possible way to be wrong
    about cost: a report reading $0.00 because a key did not match is
    indistinguishable from a report reading $0.00 because nothing was spent.

    Unknown models still return (0, 0), but the caller records the name so the
    gap is visible rather than absorbed into the total.
    """
    if model in PRICING:
        return PRICING[model]
    for key, rate in PRICING.items():
        if model.startswith(key):
            return rate
    return (0.0, 0.0)


_current_node: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_node", default="unknown"
)

# The active turn's Report travels here, NOT through the graph's state channels.
# It used to be an `AgentState` key, which meant LangGraph serialized this
# per-turn telemetry object into the conversation checkpoint and re-hydrated it
# every turn — a msgpack deprecation warning today, and a hard error the moment
# strict serialization ships, which would break exactly the persisted, multi-turn
# conversations the app is built around. A contextvar is turn-scoped and never
# checkpointed. Propagation into nodes (incl. under a checkpointer) is verified.
_report: contextvars.ContextVar["Report | None"] = contextvars.ContextVar(
    "report", default=None
)


@dataclass
class NodeStats:
    calls: int = 0
    ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class Report:
    nodes: dict[str, NodeStats] = field(default_factory=dict)
    # Models seen that no PRICING entry covers. Their tokens are counted, their
    # cost is not, and a total computed over them understates the true spend —
    # so the names are kept rather than the shortfall being silently absorbed.
    unpriced: set[str] = field(default_factory=set)

    def stat(self, name: str) -> NodeStats:
        return self.nodes.setdefault(name, NodeStats())

    @property
    def total_ms(self) -> float:
        return sum(s.ms for s in self.nodes.values())

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.nodes.values())

    @property
    def total_tokens(self) -> tuple[int, int]:
        return (
            sum(s.input_tokens for s in self.nodes.values()),
            sum(s.output_tokens for s in self.nodes.values()),
        )

    @property
    def total_calls(self) -> int:
        """Total model calls this turn — the number the grader inflates."""
        return sum(s.calls for s in self.nodes.values())


@contextlib.contextmanager
def report_scope(report: "Report") -> Iterator["Report"]:
    """Bind `report` as the active turn's telemetry sink, for its duration.

    Wrap a single `graph.invoke(...)` in this. Nodes record wall time here and
    the callback records tokens here, all without the report ever entering
    graph state. See `_report` for why that matters.
    """
    token = _report.set(report)
    try:
        yield report
    finally:
        _report.reset(token)


class UsageCollector(BaseCallbackHandler):
    """Attributes each LLM call's tokens and cost to the node that made it."""

    def __init__(self, report: Report) -> None:
        self.report = report

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        # Count model calls HERE, not in `timed`. `timed` wraps a node, which
        # runs exactly once — but a node can make many model calls: `grade`
        # invokes the model once per chunk. Counting node runs reported grade as
        # `calls=1` while it was really N, hiding the per-chunk cost this whole
        # table exists to expose. Counted on start so a call with unparseable
        # usage still registers as a call.
        self.report.stat(_current_node.get()).calls += 1

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage, model = _extract_usage(response)
        if not usage:
            return
        self._record(usage, model)

    def _record(self, usage: dict, model: str) -> None:
        """Attribute one call's tokens and cost. Separate from `on_llm_end` so
        the pricing rule can be tested without constructing an LLMResult."""
        stat = self.report.stat(_current_node.get())
        stat.input_tokens += usage.get("input_tokens", 0)
        stat.output_tokens += usage.get("output_tokens", 0)

        rate_in, rate_out = rate_for(model)
        if model and (rate_in, rate_out) == (0.0, 0.0):
            self.report.unpriced.add(model)
        stat.cost_usd += (
            usage.get("input_tokens", 0) / 1e6 * rate_in
            + usage.get("output_tokens", 0) / 1e6 * rate_out
        )


def _extract_usage(response: Any) -> tuple[dict, str]:
    """Dig usage out of an LLMResult — shape varies across provider versions."""
    out = getattr(response, "llm_output", None) or {}
    model = out.get("model_name") or out.get("model") or ""
    if usage := out.get("usage"):
        return dict(usage), model

    for gen_list in getattr(response, "generations", []) or []:
        for gen in gen_list:
            msg = getattr(gen, "message", None)
            if msg is None:
                continue
            if meta := getattr(msg, "usage_metadata", None):
                model = model or (getattr(msg, "response_metadata", {}) or {}).get("model", "")
                return dict(meta), model
    return {}, model


def timed(fn):
    """Wrap a graph node: record wall time and scope token attribution to it.

    The report comes from the `report_scope` contextvar, not from state, so an
    un-instrumented run (eval, a direct node call in a test) simply finds None
    and no-ops. Node-run counting lives in `UsageCollector`, not here — this only
    owns wall time and the current-node scope.
    """

    @functools.wraps(fn)
    def wrapper(state, *args, **kwargs):
        report = _report.get()
        if report is None:
            return fn(state, *args, **kwargs)

        token = _current_node.set(fn.__name__)
        start = time.perf_counter()
        try:
            return fn(state, *args, **kwargs)
        finally:
            report.stat(fn.__name__).ms += (time.perf_counter() - start) * 1000
            _current_node.reset(token)

    return wrapper


def format_report(report: Report) -> str:
    rows = ["stage         calls    ms    in_tok  out_tok    usd"]
    for name, s in sorted(report.nodes.items(), key=lambda kv: -kv[1].ms):
        rows.append(
            f"{name:<13} {s.calls:>5} {s.ms:>6.0f} {s.input_tokens:>8} "
            f"{s.output_tokens:>8} {s.cost_usd:>6.4f}"
        )
    tin, tout = report.total_tokens
    rows.append(
        f"{'TOTAL':<13} {report.total_calls:>5} {report.total_ms:>6.0f} {tin:>8} "
        f"{tout:>8} {report.total_cost_usd:>6.4f}"
    )
    return "\n".join(rows)
