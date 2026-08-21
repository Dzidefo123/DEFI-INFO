"""The investigation plan, rendered as a checklist that fills in during execution.

What changes here is affect, not information. "No security findings" printed
under a `## Security Findings` heading reads as breakage, because it appears as
an absence in a slot that expected a value. The same fact as an unticked step in
a list of intended steps reads as scope.

A test suite reporting skipped tests is not thought to be broken, and for the
same reason: the skip is presented as a decision rather than as a hole.
"""

import pytest

from src.evidence.models import (
    AgentName,
    Evidence,
    EvidenceKind,
    SourceRef,
    SourceTier,
    utcnow,
)
from src.intelligence.plan import build_plan
from src.reports.intelligence_report import DONE, EMPTY, SKIPPED, _stage_rows, render_report

NOW = utcnow()


def _ev(agent, i=0):
    return Evidence(
        kind=EvidenceKind.CHAIN_STATE,
        source=SourceRef(tier=SourceTier.CHAIN, uri=f"https://rpc/{agent.value}/{i}"),
        agent=agent,
        summary=f"reading {i}",
        observed_at=NOW,
        collected_at=NOW,
    )


def _rows(plan, **kw):
    kw.setdefault("claims", [])
    kw.setdefault("evidence", [])
    kw.setdefault("risk_signals", [])
    kw.setdefault("verification", {})
    return {label: (state, detail) for label, state, detail in _stage_rows(plan=plan, **kw)}


def _full():
    return build_plan("Is the wrapper solvent?", "full_investigation", ["hyperevm"])


# --- the three states ---------------------------------------------------


def test_a_stage_that_produced_something_is_ticked():
    rows = _rows(_full(), evidence=[_ev(AgentName.BLOCKCHAIN, i) for i in range(3)])
    assert rows["On-chain readings"] == (DONE, "3 readings")


def test_a_stage_that_ran_and_found_nothing_is_unticked_not_absent():
    """The distinction the whole architecture exists for, at the level of the
    page: this stage was attempted and came back empty."""
    state, detail = _rows(_full())["Security review"]
    assert state == EMPTY
    assert "ran" in detail


def test_a_stage_outside_scope_is_marked_skipped_not_failed():
    """A research-only question never intended to read the chain. Showing that
    as an empty result would invent a gap the plan never had."""
    plan = build_plan("How does funding work?", "research", ["hyperliquid"])
    state, detail = _rows(plan)["On-chain readings"]
    assert state == SKIPPED
    assert "not in scope" in detail


def test_the_three_states_are_visually_distinct():
    assert len({DONE, EMPTY, SKIPPED}) == 3


# --- every planned stage appears ----------------------------------------


def test_every_stage_is_listed_whatever_the_plan():
    for query_type in ("research", "blockchain_analysis", "full_investigation"):
        plan = build_plan("q", query_type, ["hyperevm"])
        rows = _rows(plan)
        assert set(rows) == {
            "Documentation research",
            "On-chain readings",
            "Security review",
            "Anomaly scoring",
            "Claim verification",
        }, f"{query_type} dropped a stage"


def test_scoring_reports_only_metrics_that_actually_scored():
    """An `unknown` signal is a metric that could not be judged. Counting it as
    scored would put a tick beside a stage that established nothing."""
    plan = _full()
    rows = _rows(plan, risk_signals=[{"severity": "unknown"}, {"severity": "unknown"}])
    assert rows["Anomaly scoring"][0] == EMPTY
    rows = _rows(plan, risk_signals=[{"severity": "normal"}, {"severity": "unknown"}])
    assert rows["Anomaly scoring"] == (DONE, "1 metric scored against a baseline")


def test_verification_that_ran_with_nothing_to_check_is_not_a_tick():
    plan = _full()
    assert _rows(plan, verification={"claims_examined": 0})["Claim verification"][0] == EMPTY
    assert _rows(plan, verification={"claims_examined": 4})["Claim verification"] == (
        DONE, "4 claims examined"
    )


@pytest.mark.parametrize("n,expected", [(1, "1 reading"), (2, "2 readings")])
def test_counts_are_pluralised(n, expected):
    rows = _rows(_full(), evidence=[_ev(AgentName.BLOCKCHAIN, i) for i in range(n)])
    assert rows["On-chain readings"] == (DONE, expected)


# --- it reaches the page ------------------------------------------------


def test_the_checklist_appears_in_the_rendered_report():
    text = render_report(
        plan=_full(), claims=[], evidence=[], risk_signals=[],
        verification={}, limitations=[],
    )
    assert "**Coverage**" in text
    scope = text[text.index("## Investigation Scope"):text.index("## Protocol")]
    assert "- [ ] Security review" in scope
    assert scope.index("**Coverage**") > scope.index("**Question:**")


def test_the_report_stays_deterministic():
    """Same record in, same bytes out — the checklist must not introduce timing
    or set-ordering into a pure function."""
    kw = dict(plan=_full(), claims=[], evidence=[_ev(AgentName.BLOCKCHAIN)],
              risk_signals=[], verification={}, limitations=[])
    assert render_report(**kw) == render_report(**kw)
