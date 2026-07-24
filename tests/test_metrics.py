"""Instrumentation invariants.

Two regressions this pins down:

  1. The per-turn Report must never travel through graph state — it did, which
     made LangGraph serialize telemetry into the conversation checkpoint. Here we
     assert the state schema carries no report channel and that timing flows
     through the contextvar instead.
  2. `calls` must count model calls, not node executions. `grade` invokes the
     model once per chunk; counting node runs reported it as 1, hiding exactly
     the per-chunk cost the table exists to show.
"""

from src.graph.state import AgentState
from src.obs.metrics import (
    Report,
    UsageCollector,
    _current_node,
    format_report,
    report_scope,
    timed,
)


class _FakeLLMResult:
    """Minimal stand-in with usage_metadata, shaped like a chat generation."""

    def __init__(self, model, tin, tout):
        meta = {"input_tokens": tin, "output_tokens": tout}
        msg = type("Msg", (), {"usage_metadata": meta, "response_metadata": {"model": model}})()
        gen = type("Gen", (), {"message": msg})()
        self.generations = [[gen]]
        self.llm_output = {}


# --- the checkpoint bug: report is not a state channel -------------------


def test_report_is_not_a_state_channel():
    assert "report" not in AgentState.__annotations__


def test_timed_reads_the_contextvar_not_state():
    report = Report()

    @timed
    def node(state):
        return {"ok": True}

    # No scope bound: instrumentation no-ops, node still runs.
    node({})
    assert report.nodes == {}

    # Bound: wall time is recorded, under the function's name, from the
    # contextvar — `state` carries nothing.
    with report_scope(report):
        node({})
    assert "node" in report.nodes
    assert report.nodes["node"].ms >= 0
    assert report.nodes["node"].calls == 0  # no model call happened


# --- the calls bug: count model calls, not node runs ---------------------


def test_calls_counts_model_calls_attributed_to_current_node():
    report = Report()
    collector = UsageCollector(report)

    # Simulate `grade` looping the model once per chunk: node runs once,
    # three model calls inside it.
    tok = _current_node.set("grade")
    for _ in range(3):
        collector.on_llm_start()
        collector.on_llm_end(_FakeLLMResult("claude-opus-4-8", 100, 20))
    _current_node.reset(tok)

    assert report.nodes["grade"].calls == 3          # not 1
    assert report.nodes["grade"].input_tokens == 300
    assert report.nodes["grade"].output_tokens == 60
    assert report.nodes["grade"].cost_usd > 0


def test_local_node_records_time_but_zero_calls():
    """A retrieval/guard node makes no model call — it should read calls=0,
    which is the honest signal that the stage costs time, not tokens."""
    report = Report()

    @timed
    def retrieve(state):
        return {}

    with report_scope(report):
        retrieve({})

    assert report.nodes["retrieve"].calls == 0
    assert report.nodes["retrieve"].cost_usd == 0.0


def test_total_calls_and_report_row():
    report = Report()
    report.stat("route").calls = 1
    report.stat("grade").calls = 5
    report.stat("retrieve").calls = 0
    assert report.total_calls == 6
    # The TOTAL row surfaces it rather than leaving the column blank.
    assert "TOTAL" in format_report(report)


def test_report_scope_restores_previous_binding():
    from src.obs import metrics

    assert metrics._report.get() is None
    outer = Report()
    with report_scope(outer):
        assert metrics._report.get() is outer
        inner = Report()
        with report_scope(inner):
            assert metrics._report.get() is inner
        assert metrics._report.get() is outer
    assert metrics._report.get() is None
