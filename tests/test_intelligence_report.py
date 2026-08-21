"""The §15 report artifact.

The property under test throughout: the report may never read as more complete
than the record behind it. An empty investigation must be distinguishable, at a
glance, from a thorough one that found nothing — because those two call for
opposite responses and only one of them is reassuring.
"""

from datetime import datetime, timezone

import pytest

from src.evidence.models import (
    AgentName,
    Claim,
    Evidence,
    EvidenceKind,
    EvidenceLink,
    SourceRef,
    SourceTier,
    Stance,
    VerificationStatus,
    utcnow,
)
from src.intelligence.plan import build_plan
from src.reports.intelligence_report import render_report
from src.risk.signals import assess_metric

# Fixtures are stamped relative to the real clock, not a fixed date. Confidence
# decays with evidence age against wall-clock time, so a hardcoded "now" means
# these fixtures age one day per day: they encode day-zero freshness and then
# drift out of it. That is how this file broke the morning after it was written.
# Offsets below (NOW - timedelta(...)) still express deliberate staleness.
NOW = utcnow()
QUIET = [2.3, 2.1, 2.5, 2.4, 2.2, 2.0, 2.6, 2.3, 2.4, 2.2]


def _ev(uri, summary="an observation", tier=SourceTier.PRIMARY):
    return Evidence(
        kind=EvidenceKind.ON_CHAIN_METRIC,
        source=SourceRef(tier=tier, uri=uri, protocol="ethena"),
        agent=AgentName.BLOCKCHAIN,
        summary=summary,
        observed_at=NOW,
        collected_at=NOW,
    )


def _claim(text, evs, stances=None, verification=VerificationStatus.VERIFIED):
    stances = stances or [Stance.SUPPORTS] * len(evs)
    return Claim(
        text=text,
        agent=AgentName.BLOCKCHAIN,
        protocols=("ethena",),
        created_at=NOW,
        verification=verification,
        links=tuple(
            EvidenceLink(evidence_id=e.evidence_id, stance=s)
            for e, s in zip(evs, stances)
        ),
    )


def _plan(query_type="full_investigation", protocols=("ethena",)):
    return build_plan("Is Ethena showing risk?", query_type, list(protocols))


def _render(**kw):
    kw.setdefault("plan", _plan())
    kw.setdefault("claims", [])
    kw.setdefault("evidence", [])
    kw.setdefault("risk_signals", [])
    kw.setdefault("verification", {})
    kw.setdefault("limitations", [])
    return render_report(**kw)


# --- the empty investigation -------------------------------------------


def test_an_empty_investigation_does_not_read_as_an_all_clear():
    """The single most important line in the report. 'No findings' and 'no
    problems' are opposite conclusions and the prose must not blur them."""
    text = _render(limitations=["security_agent: not implemented"])
    assert "not a clean bill of health" in text
    assert "no findings" in text.lower()


def test_absent_statistics_say_why_rather_than_showing_normal():
    """An empty statistics table would read as 'we checked, all normal'."""
    text = _render()
    assert "No metrics were scored" in text
    assert "no statistical claim can be made in either direction" in text


def test_absent_contradictions_say_none_were_searched_for():
    text = _render()
    assert "empty rather than clear" in text


def test_limitations_are_never_silently_dropped():
    text = _render(limitations=["security_agent: threat feeds not connected"])
    assert "threat feeds not connected" in text


def test_verification_with_nothing_to_examine_is_recorded_as_a_limitation():
    text = _render(verification={"claims_examined": 0, "verdicts": []})
    assert "no claims to examine" in text


def test_an_unscoped_investigation_reports_that_limitation():
    """`build_plan` notes it; the report must surface it."""
    text = _render(plan=_plan(protocols=()))
    assert "No protocol was identified" in text
    assert "not identified" in text


# --- a populated investigation -----------------------------------------


def test_a_populated_report_names_its_findings_and_evidence():
    evs = [_ev(f"https://chain/{i}", f"observation {i}") for i in range(3)]
    claims = [_claim("Outflows were abnormally high.", evs)]
    text = _render(claims=claims, evidence=evs)

    assert "Outflows were abnormally high." in text
    assert "Evidence (3)" in text
    assert "https://chain/0" in text
    assert "not a clean bill of health" not in text


def test_findings_are_ordered_strongest_first():
    """A reader who stops after one heading should have read the best-supported
    thing the investigation found."""
    strong_evs = [_ev(f"https://chain/s{i}", f"s{i}") for i in range(4)]
    weak_ev = [_ev("https://forum/w", "w", tier=SourceTier.UNVERIFIED)]
    claims = [
        _claim("Weak claim.", weak_ev, verification=VerificationStatus.PARTIALLY_VERIFIED),
        _claim("Strong claim.", strong_evs),
    ]
    text = _render(claims=claims, evidence=strong_evs + weak_ev)
    assert text.index("Strong claim.") < text.index("Weak claim.")


def test_each_finding_shows_what_limits_its_confidence():
    """§3.1: a bare number cannot distinguish 'excellent but stale' from 'fresh
    but anonymous', and those call for different responses."""
    ev = [_ev("https://forum/x", "rumour", tier=SourceTier.UNVERIFIED)]
    text = _render(claims=[_claim("Something happened.", ev)], evidence=ev)
    assert "limited by source reliability" in text


def test_the_statistics_table_renders_scored_metrics():
    signal = assess_metric("liquidity_outflow", 12.5, QUIET, protocol="ethena")
    text = _render(risk_signals=[signal.model_dump(mode="json")])
    assert "| liquidity_outflow" in text
    assert "High anomaly" in text or "Critical anomaly" in text


def test_an_unassessable_metric_renders_as_not_assessed_not_normal():
    signal = assess_metric("tvl", 5.0, [1.0, 2.0])  # too little history
    text = _render(risk_signals=[signal.model_dump(mode="json")])
    assert "Not assessed" in text
    assert "| Normal |" not in text


def test_a_metric_with_no_history_shows_no_baseline_not_its_own_value():
    """`assess_metric` fills the empty baseline with the current value. Printing
    that shows a metric sitting exactly on its baseline — the most reassuring row
    in the table, produced by the case where nothing was measured at all."""
    signal = assess_metric("tvl", 5.0, [1.0, 2.0])
    text = _render(risk_signals=[signal.model_dump(mode="json")])
    assert "no history" in text
    assert "| 5 | 5 |" not in text


def test_contradicting_evidence_gets_its_own_section():
    """§15 gives contradictions a heading precisely so they cannot be buried."""
    support = _ev("https://chain/a", "outflow spiked")
    against = _ev("https://chain/b", "inflows matched it")
    claim = _claim(
        "Funds left the protocol.",
        [support, against],
        stances=[Stance.SUPPORTS, Stance.CONTRADICTS],
        verification=VerificationStatus.PARTIALLY_VERIFIED,
    )
    text = _render(claims=[claim], evidence=[support, against])
    section = text.split("## Contradictory Evidence")[1]
    assert "inflows matched it" in section


def test_a_dangling_evidence_link_is_shown_not_hidden():
    """An agent claiming support from evidence nobody can produce is a defect in
    that agent, and hiding it makes the finding look better than it is."""
    real = _ev("https://chain/a")
    claim = Claim(
        text="Something happened.",
        agent=AgentName.BLOCKCHAIN,
        created_at=NOW,
        verification=VerificationStatus.PARTIALLY_VERIFIED,
        links=(
            EvidenceLink(evidence_id=real.evidence_id, stance=Stance.SUPPORTS),
            EvidenceLink(evidence_id="ev_ghost", stance=Stance.SUPPORTS),
        ),
    )
    text = _render(claims=[claim], evidence=[real])
    assert "missing evidence" in text and "ev_ghost" in text


def test_an_entirely_unverified_set_of_claims_still_refuses_to_conclude():
    """Claims exist, but none passed verification — the assessment must not
    promote them into findings."""
    evs = [_ev("https://chain/a")]
    claims = [_claim("Unchecked.", evs, verification=VerificationStatus.UNVERIFIED)]
    text = _render(claims=claims, evidence=evs)
    assert "not a clean bill of health" in text


# --- structure and determinism -----------------------------------------


REQUIRED_HEADINGS = [
    "# Intelligence Assessment",
    "## Executive Summary",
    "## Investigation Scope",
    "## Protocol / Entity",
    "## Key Findings",
    "## Statistical Findings",
    "## Contradictory Evidence",
    "## Limitations",
    "## Final Assessment",
]


@pytest.mark.parametrize("heading", REQUIRED_HEADINGS)
def test_every_section_of_the_spec_is_present(heading):
    assert heading in _render()


def test_rendering_is_deterministic():
    """Same record in, same bytes out — so a report can be diffed across runs and
    costs nothing to regenerate."""
    evs = [_ev("https://chain/a")]
    args = dict(claims=[_claim("A claim.", evs)], evidence=evs)
    assert _render(**args) == _render(**args)


def test_the_report_states_which_agents_were_dispatched():
    """Distinguishing 'security found nothing' from 'security never ran' is
    unrecoverable after the fact unless the intent was recorded up front."""
    text = _render(plan=_plan("blockchain_analysis"))
    assert "blockchain agent" in text
    assert "security agent" not in text


def test_the_report_discloses_that_confidence_is_uncalibrated():
    assert "uncalibrated" in _render()


# --- coverage gaps ------------------------------------------------------


def test_a_partial_investigation_leads_with_what_it_did_not_cover():
    """Confidence is a property of a claim, not of an investigation. An
    investigation that measured no on-chain data and searched an empty registry
    can still verify a claim about documentation at 0.98 — and a reader skimming
    the first line would take that as an answer about risk."""
    evs = [_ev(f"https://docs/{i}") for i in range(3)]
    text = _render(
        claims=[_claim("Documentation exists.", evs)],
        evidence=evs,
        gaps=["no on-chain metric could be scored against a baseline"],
    )
    summary = text.split("## Investigation Scope")[0]
    assert "Partial investigation" in summary
    assert "not an answer about the parts that were not" in summary
    assert summary.index("Partial investigation") < summary.index("confidence")


def test_a_complete_investigation_does_not_cry_partial():
    evs = [_ev("https://docs/a")]
    text = _render(claims=[_claim("A claim.", evs)], evidence=evs, gaps=[])
    assert "Partial investigation" not in text


def test_gaps_are_derived_from_the_plan_not_from_what_happened():
    """'The risk engine found nothing unusual' and 'the risk engine had nothing
    to look at' are different sentences, and only the plan knows which applies."""
    from src.reports.intelligence_report import coverage_gaps

    scored = [{"severity": "normal"}]
    unscored = [{"severity": "unknown"}]

    planned = _plan("blockchain_analysis")
    assert coverage_gaps(planned, scored, {}) == []
    assert "scored against a baseline" in coverage_gaps(planned, unscored, {})[0]

    # A research-only plan never asked for metrics, so their absence is no gap.
    assert coverage_gaps(_plan("research"), [], {}) == []


def test_an_empty_security_registry_counts_as_a_gap():
    from src.reports.intelligence_report import coverage_gaps

    plan = _plan("security_analysis")
    empty = {"by_classification": {"confirmed_incident": 0, "unverified_claim": 0}}
    found = {"by_classification": {"confirmed_incident": 1}}

    assert "no security findings were on file" in coverage_gaps(plan, [], empty)[0]
    assert coverage_gaps(plan, [], found) == []
