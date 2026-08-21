"""D2's report refinements: the summary/assessment split, security findings, data
freshness, bounded evidence lists, and the structured form.

The §15 structure shipped in B2 and the provenance section in D1. What is tested
here is the difference between a report that contains the right headings and one
a reader can actually rely on.
"""

from datetime import datetime, timedelta, timezone

import json

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
from src.reports.intelligence_report import (
    MAX_EVIDENCE_SHOWN,
    render_report,
    report_payload,
)

# Fixtures are stamped relative to the real clock, not a fixed date. Confidence
# decays with evidence age against wall-clock time, so a hardcoded "now" means
# these fixtures age one day per day: they encode day-zero freshness and then
# drift out of it. That is how this file broke the morning after it was written.
# Offsets below (NOW - timedelta(...)) still express deliberate staleness.
NOW = utcnow()



def _day(dt):
    """The date as the report prints it. Derived from the fixture rather than
    written out, so the assertion still checks the right date tomorrow."""
    return dt.strftime("%Y-%m-%d")

def _ev(uri, summary="an observation", tier=SourceTier.PRIMARY, observed_at=NOW):
    return Evidence(
        kind=EvidenceKind.ON_CHAIN_METRIC,
        source=SourceRef(tier=tier, uri=uri, protocol="ethena"),
        agent=AgentName.BLOCKCHAIN,
        summary=summary,
        observed_at=observed_at,
        collected_at=NOW,
    )


def _sec_ev(classification, title="Oracle deviation", status="mitigated"):
    return Evidence(
        kind=EvidenceKind.INCIDENT_REPORT,
        source=SourceRef(
            tier=SourceTier.OFFICIAL,
            uri="https://example.org/report",
            protocol="ethena",
            title=title,
        ),
        agent=AgentName.SECURITY,
        summary="[" + classification + "] " + title,
        payload={"classification": classification, "status": status},
        observed_at=NOW,
        collected_at=NOW,
    )


def _claim(text, evs, verification=VerificationStatus.VERIFIED):
    return Claim(
        text=text,
        agent=AgentName.BLOCKCHAIN,
        protocols=("ethena",),
        created_at=NOW,
        verification=verification,
        links=tuple(
            EvidenceLink(evidence_id=e.evidence_id, stance=Stance.SUPPORTS) for e in evs
        ),
    )


def _plan(query_type="full_investigation"):
    return build_plan("Is Ethena showing risk?", query_type, ["ethena"])


def _render(**kw):
    kw.setdefault("plan", _plan())
    kw.setdefault("claims", [])
    kw.setdefault("evidence", [])
    kw.setdefault("risk_signals", [])
    kw.setdefault("verification", {})
    kw.setdefault("limitations", [])
    return render_report(**kw)


def _payload(**kw):
    kw.setdefault("plan", _plan())
    kw.setdefault("claims", [])
    kw.setdefault("evidence", [])
    kw.setdefault("risk_signals", [])
    kw.setdefault("verification", {})
    kw.setdefault("limitations", [])
    return report_payload(**kw)


def _sections(text):
    return {
        part.split("\n", 1)[0].strip(): part.split("\n", 1)[1]
        for part in text.split("\n## ")[1:]
    }


# --- the summary and the assessment do different jobs -------------------


def test_the_summary_and_the_final_assessment_are_not_the_same_text():
    """They had identical wording, which wasted one of the two sections §15 gives
    separate headings. The summary says what was found; the assessment says what
    the investigation is entitled to conclude."""
    evs = [_ev("https://chain/" + str(i)) for i in range(3)]
    parts = _sections(_render(claims=[_claim("A finding.", evs)], evidence=evs))
    assert parts["Executive Summary"].strip() != parts["Final Assessment"].strip()


def test_the_final_assessment_states_what_may_be_relied_on():
    evs = [_ev("https://chain/" + str(i)) for i in range(4)]
    parts = _sections(_render(claims=[_claim("A finding.", evs)], evidence=evs))
    assert "can be relied on" in parts["Final Assessment"]


def test_a_weakly_supported_finding_is_called_a_lead_not_a_conclusion():
    ev = [_ev("https://forum/x", "rumour", tier=SourceTier.UNVERIFIED)]
    claim = _claim("Something.", ev, VerificationStatus.PARTIALLY_VERIFIED)
    parts = _sections(_render(claims=[claim], evidence=ev))
    assert "not a conclusion to act on" in parts["Final Assessment"]


def test_the_final_assessment_names_what_stays_open():
    evs = [_ev("https://chain/" + str(i)) for i in range(3)]
    parts = _sections(
        _render(
            claims=[_claim("A finding.", evs)],
            evidence=evs,
            gaps=["no on-chain metric could be scored against a baseline"],
        )
    )
    assert "bounded by what was searched" in parts["Final Assessment"]
    assert "Nothing here speaks to those" in parts["Final Assessment"]


def test_the_final_assessment_counts_claims_that_did_not_survive():
    evs = [_ev("https://chain/a")]
    good = _claim("Verified.", evs)
    bad = _claim("Rejected.", evs, VerificationStatus.INSUFFICIENT_EVIDENCE)
    parts = _sections(_render(claims=[good, bad], evidence=evs))
    assert "1 further claim(s) did not survive" in parts["Final Assessment"]


def test_nothing_established_says_the_questions_remain_open():
    evs = [_ev("https://chain/a")]
    claim = _claim("x", evs, VerificationStatus.INSUFFICIENT_EVIDENCE)
    parts = _sections(_render(claims=[claim], evidence=evs))
    assert "remain open" in parts["Final Assessment"]
    assert "in either direction" in parts["Final Assessment"]


# --- security findings ---------------------------------------------------


def test_security_findings_get_their_own_section():
    """§15 lists it, and §10's four categories had nowhere to appear separately."""
    parts = _sections(_render(evidence=[_sec_ev("confirmed_incident")]))
    section = parts["Security Findings"]
    assert "Confirmed incident" in section
    assert "Oracle deviation" in section
    assert "mitigated" in section


def test_established_and_unestablished_findings_are_separated():
    """A rumour printed beside a confirmed incident, in the same typeface, reads
    as a second incident. That is the merge §10 forbids."""
    evs = [
        _sec_ev("confirmed_incident", "Real thing"),
        _sec_ev("unverified_claim", "Alleged thing"),
    ]
    section = _sections(_render(evidence=evs))["Security Findings"]
    assert section.index("Real thing") < section.index("context only")
    assert section.index("context only") < section.index("Alleged thing")
    assert "does not make any finding above more" in section


def test_an_empty_security_section_says_nothing_was_recorded():
    section = _sections(_render())["Security Findings"]
    assert "not that nothing has happened" in section


def test_each_classification_is_labelled_separately():
    evs = [
        _sec_ev("confirmed_incident", "A"),
        _sec_ev("known_vulnerability", "B"),
        _sec_ev("suspicious_signal", "C"),
    ]
    section = _sections(_render(evidence=evs))["Security Findings"]
    for label in ("Confirmed incident", "Known vulnerability", "Suspicious signal"):
        assert "**" + label + "**" in section


# --- data freshness ------------------------------------------------------


def test_the_report_states_when_its_evidence_was_true():
    """An artifact about on-chain data with no as-of window is dangerous: the
    numbers look current because the document is."""
    evs = [_ev("https://chain/a"), _ev("https://chain/b", observed_at=NOW - timedelta(days=30))]
    text = _render(evidence=evs)
    assert _day(NOW - timedelta(days=30)) in text and _day(NOW) in text


def test_a_single_day_of_measurements_reads_as_one_date():
    text = _render(evidence=[_ev("https://a")])
    assert "Measurements are as of **" + _day(NOW) + "**" in text


def test_documentation_is_dated_by_retrieval_and_says_so():
    """A docs page has no authored date the system can see, so its timestamp is a
    fact about the crawl, not the content. Printing it beside measurements would
    date a year-old page to today."""
    doc = Evidence(
        kind=EvidenceKind.DOCUMENT,
        source=SourceRef(tier=SourceTier.PRIMARY, uri="https://docs/a"),
        agent=AgentName.RESEARCH,
        summary="a docs chunk",
        collected_at=NOW,
    )
    text = _render(evidence=[doc])
    assert "retrieved on **" + _day(NOW) + "**" in text
    assert "own age is not known" in text
    assert "Measurements are as of" not in text


def test_measurements_and_documentation_are_dated_separately():
    doc = Evidence(
        kind=EvidenceKind.DOCUMENT,
        source=SourceRef(tier=SourceTier.PRIMARY, uri="https://docs/a"),
        agent=AgentName.RESEARCH,
        summary="a docs chunk",
        collected_at=NOW,
    )
    text = _render(evidence=[_ev("https://chain/a"), doc])
    assert "Measurements are as of" in text
    assert "documentation excerpt(s) were retrieved" in text


def test_freshness_uses_the_truth_time_not_the_fetch_time():
    """Re-reading stale data must not present as fresh."""
    stale = Evidence(
        kind=EvidenceKind.ON_CHAIN_METRIC,
        source=SourceRef(tier=SourceTier.PRIMARY, uri="https://chain/a"),
        agent=AgentName.BLOCKCHAIN,
        summary="an old reading, fetched just now",
        observed_at=NOW - timedelta(days=90),
        collected_at=NOW,
    )
    assert _day(NOW - timedelta(days=90)) in _render(evidence=[stale])


def test_no_evidence_means_no_data_window():
    assert "no data window" in _render()


# --- bounded evidence lists ----------------------------------------------


def test_a_heavily_cited_finding_summarises_the_tail():
    """Twelve bullets bury the next finding; the count is already stated and the
    evidence graph holds all of them."""
    evs = [_ev("https://chain/" + str(i), "observation " + str(i)) for i in range(12)]
    text = _render(claims=[_claim("A finding.", evs)], evidence=evs)
    assert "Evidence (12)" in text
    assert "and " + str(12 - MAX_EVIDENCE_SHOWN) + " more" in text
    assert "observation 11" not in text


def test_a_short_evidence_list_is_shown_in_full():
    evs = [_ev("https://chain/" + str(i), "observation " + str(i)) for i in range(3)]
    text = _render(claims=[_claim("A finding.", evs)], evidence=evs)
    assert "more, listed" not in text
    assert "observation 2" in text


# --- the structured form -------------------------------------------------


def test_the_structured_form_carries_the_findings_as_data():
    """Markdown can be read but not queried, diffed by field, or scored — and the
    evaluation harness needs all three."""
    evs = [_ev("https://chain/" + str(i)) for i in range(3)]
    data = _payload(claims=[_claim("A finding.", evs)], evidence=evs)

    finding = data["findings"][0]
    assert finding["text"] == "A finding."
    assert finding["verification"] == "verified"
    assert 0 < finding["confidence"] <= 1
    assert finding["confidence_band"] in ("high", "moderate", "low", "very low")
    assert len(finding["supporting_sources"]) == 3
    assert finding["limiting_factor"]["name"]


def test_the_structured_findings_are_ordered_like_the_prose():
    strong = [_ev("https://chain/s" + str(i)) for i in range(4)]
    weak = [_ev("https://forum/w", tier=SourceTier.UNVERIFIED)]
    claims = [
        _claim("Weak.", weak, VerificationStatus.PARTIALLY_VERIFIED),
        _claim("Strong.", strong),
    ]
    data = _payload(claims=claims, evidence=strong + weak)
    assert [f["text"] for f in data["findings"]] == ["Strong.", "Weak."]


def test_the_structured_form_keeps_security_counts_per_category():
    evs = [
        _sec_ev("confirmed_incident"),
        _sec_ev("unverified_claim"),
        _sec_ev("unverified_claim", "B"),
    ]
    counts = _payload(evidence=evs)["security"]["by_classification"]
    assert counts["confirmed_incident"] == 1
    assert counts["unverified_claim"] == 2
    assert counts["suspicious_signal"] == 0


def test_the_structured_form_reports_independent_lines_not_just_claims():
    evs = [_ev("https://chain/a")]
    data = _payload(
        claims=[_claim("x", evs)],
        evidence=evs,
        graph={"independent_groups": [["a", "b"]], "shared_sources": [{"label": "p"}]},
    )
    assert data["provenance"]["independent_lines"] == 1
    assert data["provenance"]["shared_sources"] == 1


def test_the_structured_form_records_the_evidence_window():
    data = _payload(evidence=[_ev("https://a")])
    assert data["evidence_window"]["from"] == data["evidence_window"]["to"]
    assert data["evidence_window"]["from"].startswith(_day(NOW))


def test_the_structured_form_declares_its_calibration():
    assert "uncalibrated" in _payload()["calibration"]


def test_the_structured_form_is_json_serialisable():
    """It travels in a checkpointed state channel."""
    evs = [_ev("https://chain/a")]
    json.dumps(_payload(claims=[_claim("x", evs)], evidence=evs))


def test_prose_and_data_cannot_disagree_about_the_findings():
    """Rendered from the same record, so a reader and a script see the same
    investigation."""
    evs = [_ev("https://chain/" + str(i)) for i in range(3)]
    claims = [_claim("A finding.", evs)]
    text = _render(claims=claims, evidence=evs)
    data = _payload(claims=claims, evidence=evs)
    assert data["findings"][0]["text"] in text
    assert "{:.2f}".format(data["findings"][0]["confidence"]) in text


def test_the_structured_form_records_what_was_not_covered():
    data = _payload(gaps=["no security findings were on file to review"])
    assert data["coverage_gaps"] == ["no security findings were on file to review"]


def test_the_structured_form_records_which_agents_ran():
    """Distinguishing 'searched and found nothing' from 'never searched' has to
    survive into the machine-readable form too."""
    data = _payload(plan=_plan("blockchain_analysis"))
    assert data["agents_dispatched"] == ["blockchain_agent"]
