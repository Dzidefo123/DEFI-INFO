"""§13's Verification Agent — the component whose success looks like removing output.

Every check but one is deterministic, so almost all of this runs for free. The
entailment check is injected.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.verification import (
    CONTRADICTION_THRESHOLD,
    MIN_TEMPORAL_RELEVANCE,
    STRONG_CLAIM_MIN_SOURCES,
    check_contradiction,
    check_numeric_consistency,
    check_overclaiming,
    check_source_quality,
    check_support,
    check_temporal_relevance,
    parse_numbers,
    verify,
    verify_all,
)
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
)

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


def _ev(uri, summary="an observation", tier=SourceTier.PRIMARY,
        kind=EvidenceKind.DOCUMENT, observed_at=NOW, text=""):
    return Evidence(
        kind=kind,
        source=SourceRef(tier=tier, uri=uri, protocol="ethena"),
        agent=AgentName.BLOCKCHAIN,
        summary=summary,
        payload={"text": text} if text else {},
        observed_at=observed_at,
        collected_at=NOW,
    )


def _claim(text, evs, stances=None, relevances=None):
    stances = stances or [Stance.SUPPORTS] * len(evs)
    relevances = relevances or [1.0] * len(evs)
    return Claim(
        text=text,
        agent=AgentName.RESEARCH,
        protocols=("ethena",),
        created_at=NOW,
        links=tuple(
            EvidenceLink(evidence_id=e.evidence_id, stance=s, relevance=r)
            for e, s, r in zip(evs, stances, relevances)
        ),
    )


def _yes(claim, evidence):
    return True, "the excerpts state this"


def _no(claim, evidence):
    return False, "the excerpts are on topic but do not state this"


# --- support -------------------------------------------------------------


def test_a_claim_with_no_evidence_fails_blocking():
    result = check_support(_claim("Something.", []), [])
    assert not result.passed and result.blocks


def test_support_counts_independent_sources_not_chunks():
    """Otherwise finer chunking makes a claim look better evidenced."""
    same_page = [_ev("https://docs/a", f"chunk {i}") for i in range(4)]
    result = check_support(_claim("x", same_page), same_page)
    assert "1 independent source" in result.detail


# --- contradiction -------------------------------------------------------


def test_undisputed_evidence_passes():
    evs = [_ev("https://a"), _ev("https://b")]
    claim = _claim("x", evs)
    assert check_contradiction(claim, {e.evidence_id: e for e in evs}).passed


def test_contradicting_evidence_outweighing_support_blocks():
    evs = [_ev("https://a"), _ev("https://b"), _ev("https://c")]
    claim = _claim("x", evs, [Stance.SUPPORTS, Stance.CONTRADICTS, Stance.CONTRADICTS])
    result = check_contradiction(claim, {e.evidence_id: e for e in evs})
    assert not result.passed and result.blocks


def test_contradiction_is_relevance_weighted_not_a_headcount():
    """A marginal objection should not cancel a directly relevant finding."""
    evs = [_ev("https://a"), _ev("https://b")]
    claim = _claim("x", evs, [Stance.SUPPORTS, Stance.CONTRADICTS], [1.0, 0.1])
    assert check_contradiction(claim, {e.evidence_id: e for e in evs}).passed


def test_an_even_split_is_not_a_contradiction():
    """Exactly at the threshold, support has not been outweighed."""
    evs = [_ev("https://a"), _ev("https://b")]
    claim = _claim("x", evs, [Stance.SUPPORTS, Stance.CONTRADICTS])
    result = check_contradiction(claim, {e.evidence_id: e for e in evs})
    assert result.passed
    assert f"{CONTRADICTION_THRESHOLD:.2f}" in result.detail


# --- source quality ------------------------------------------------------


def test_an_entirely_unverified_claim_fails_source_quality():
    evs = [_ev("https://forum/a", tier=SourceTier.UNVERIFIED)]
    assert not check_source_quality(_claim("x", evs), evs).passed


def test_one_good_source_among_weak_ones_passes():
    """Consistent with how confidence treats reliability: the best provenance
    behind a claim, not the average — so citing a corroborating forum post
    alongside the docs does not penalise thoroughness."""
    evs = [
        _ev("https://forum/a", tier=SourceTier.UNVERIFIED),
        _ev("https://docs/b", tier=SourceTier.PRIMARY),
    ]
    assert check_source_quality(_claim("x", evs), evs).passed


# --- temporal relevance --------------------------------------------------


def test_current_evidence_passes():
    evs = [_ev("https://a", observed_at=NOW)]
    assert check_temporal_relevance(_claim("x", evs), evs, NOW).passed


def test_a_stale_market_reading_fails():
    """Market data has a six-hour half-life. A day-old funding rate cannot carry
    a present-tense claim."""
    old = [_ev("https://a", kind=EvidenceKind.MARKET_DATA,
               observed_at=NOW - timedelta(days=2))]
    result = check_temporal_relevance(_claim("x", old), old, NOW)
    assert not result.passed
    assert "stale" in result.detail


def test_a_year_old_documentation_page_still_passes():
    """The same age, a different kind. One global freshness window would either
    discard durable documentation or treat live metrics as durable."""
    old = [_ev("https://a", kind=EvidenceKind.DOCUMENT,
               observed_at=NOW - timedelta(days=300))]
    assert check_temporal_relevance(_claim("x", old), old, NOW).passed


def test_one_current_source_is_enough():
    evs = [
        _ev("https://a", kind=EvidenceKind.MARKET_DATA, observed_at=NOW - timedelta(days=5)),
        _ev("https://b", kind=EvidenceKind.MARKET_DATA, observed_at=NOW),
    ]
    assert check_temporal_relevance(_claim("x", evs), evs, NOW).passed


# --- numeric consistency -------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("outflow was 12.5", [12.5]),
        ("$12.5M left the pool", [12_500_000.0]),
        ("12,500,000 tokens", [12_500_000.0]),
        ("a 4% fee", [4.0]),
        ("grew 3x", [3.0]),
        ("no figures here", []),
        ("between 5 and 10", [5.0, 10.0]),
    ],
)
def test_numbers_are_parsed_and_normalised(text, expected):
    """'$12.5M' and '12,500,000' must compare equal, so neither the claim nor the
    evidence has to be written a particular way."""
    assert parse_numbers(text) == expected


def test_a_claim_with_no_figures_passes_trivially():
    evs = [_ev("https://a")]
    assert check_numeric_consistency(_claim("USDe is overcollateralised.", evs), evs).passed


def test_a_figure_present_in_the_evidence_passes():
    evs = [_ev("https://a", "Daily outflow reached $12.5M")]
    claim = _claim("Outflows reached $12.5M yesterday.", evs)
    assert check_numeric_consistency(claim, evs).passed


def test_a_fabricated_figure_is_caught():
    """The cheapest defence against the most expensive error a DeFi answer can
    make. Retrieval cannot catch it and grading cannot — the chunk is relevant,
    the sentence is fluent, and the number is invented."""
    evs = [_ev("https://a", "Daily outflow reached $12.5M")]
    claim = _claim("Outflows reached $87.3M yesterday.", evs)
    result = check_numeric_consistency(claim, evs)
    assert not result.passed
    assert "appear nowhere in the evidence" in result.detail


def test_matching_tolerates_rounding():
    evs = [_ev("https://a", "the rate was 4.01%")]
    assert check_numeric_consistency(_claim("The rate was 4%.", evs), evs).passed


def test_figures_are_matched_across_units():
    evs = [_ev("https://a", "12,500,000 USDe was redeemed")]
    assert check_numeric_consistency(_claim("$12.5M was redeemed.", evs), evs).passed


def test_a_figure_in_the_evidence_payload_counts():
    """On-chain evidence carries its value in the payload, not the prose."""
    evs = [
        Evidence(
            kind=EvidenceKind.ON_CHAIN_METRIC,
            source=SourceRef(tier=SourceTier.PRIMARY, uri="https://chain/a"),
            agent=AgentName.BLOCKCHAIN,
            summary="gas average",
            payload={"value": 12.5},
            observed_at=NOW,
        )
    ]
    assert check_numeric_consistency(_claim("Gas averaged 12.5 gwei.", evs), evs).passed


def test_a_fabricated_figure_only_weakens_it_does_not_block():
    """A claim may legitimately restate a figure in units the evidence does not
    use, so this is reported rather than fatal."""
    evs = [_ev("https://a", "outflow reached 12.5")]
    result = check_numeric_consistency(_claim("Outflow reached 999.", evs), evs)
    assert not result.passed and not result.blocks


# --- overclaiming --------------------------------------------------------


def test_an_observation_passes():
    evs = [_ev("https://a")]
    result = check_overclaiming(_claim("Outflows were unusually high.", evs), evs)
    assert result.passed
    assert "stated as an observation" in result.detail


@pytest.mark.parametrize(
    "text",
    [
        "The outflows were caused by a security exploit.",
        "The depeg happened because of an oracle failure.",
        "The loss was due to a contract bug.",
        "The upgrade resulted in a liquidity drop.",
    ],
)
def test_a_causal_claim_on_one_source_is_overclaiming(text):
    """§13.2's example, as a rule. Abnormal withdrawals were observed; that they
    were CAUSED BY an exploit is a different and much larger claim."""
    evs = [_ev("https://a")]
    result = check_overclaiming(_claim(text, evs), evs)
    assert not result.passed
    assert "more than an observation" in result.detail


@pytest.mark.parametrize(
    "text",
    [
        "The protocol is completely safe.",
        "Users can never lose funds.",
        "Redemption is guaranteed.",
        "This proves the peg holds.",
    ],
)
def test_an_absolute_claim_on_one_source_is_overclaiming(text):
    evs = [_ev("https://a")]
    assert not check_overclaiming(_claim(text, evs), evs).passed


def test_a_causal_claim_with_enough_sources_passes():
    evs = [_ev(f"https://{i}") for i in range(STRONG_CLAIM_MIN_SOURCES)]
    claim = _claim("The drop was caused by a scheduled unlock.", evs)
    assert check_overclaiming(claim, evs).passed


# --- status assignment ---------------------------------------------------


def test_a_well_evidenced_observation_verifies():
    evs = [_ev(f"https://{i}") for i in range(3)]
    assert verify(_claim("Outflows rose.", evs), evs, now=NOW).status is (
        VerificationStatus.VERIFIED
    )


def test_an_unsupported_claim_is_insufficient_evidence_not_contradicted():
    """'We could not establish this' and 'we have reason to believe otherwise'
    are different findings, and the weaker must not be reported as the stronger."""
    assert verify(_claim("Something.", []), [], now=NOW).status is (
        VerificationStatus.INSUFFICIENT_EVIDENCE
    )


def test_outweighed_support_is_contradicted():
    evs = [_ev("https://a"), _ev("https://b"), _ev("https://c")]
    claim = _claim("x", evs, [Stance.SUPPORTS, Stance.CONTRADICTS, Stance.CONTRADICTS])
    assert verify(claim, evs, now=NOW).status is VerificationStatus.CONTRADICTED


def test_a_weakening_failure_partially_verifies():
    evs = [_ev("https://a", "outflow reached 12.5")]
    claim = _claim("Outflow reached 999.", evs)  # numeric mismatch only
    verdict = verify(claim, evs, now=NOW)
    assert verdict.status is VerificationStatus.PARTIALLY_VERIFIED
    assert "numeric_consistency" in [c.check for c in verdict.failures]


def test_the_docs_worked_example_end_to_end():
    """§13.2 both ways. Abnormal withdrawals: four sources, verified. Caused by
    an exploit: one unverified source, two contradictions, and causal language."""
    kind = EvidenceKind.ON_CHAIN_METRIC
    supporting = [_ev(f"https://chain/{i}", kind=kind) for i in range(4)]
    observed = verify(_claim("Withdrawals were abnormally high.", supporting),
                      supporting, now=NOW)

    rumour = _ev("https://forum/x", tier=SourceTier.UNVERIFIED, kind=kind)
    against = [_ev(f"https://chain/c{i}", kind=kind) for i in range(2)]
    pool = [rumour, *against]
    caused = verify(
        _claim("The withdrawals were caused by a security exploit.", pool,
               [Stance.SUPPORTS, Stance.CONTRADICTS, Stance.CONTRADICTS]),
        pool, now=NOW,
    )

    assert observed.status is VerificationStatus.VERIFIED
    assert caused.status is VerificationStatus.CONTRADICTED
    assert observed.confidence > caused.confidence


def test_a_verdict_explains_itself():
    evs = [_ev("https://forum/a", tier=SourceTier.UNVERIFIED)]
    verdict = verify(_claim("The peg broke because of an oracle failure.", evs), evs, now=NOW)
    failed = {c.check for c in verdict.failures}
    assert {"source_quality", "overclaiming"} <= failed
    assert "overclaiming" in verdict.summary()


# --- entailment ----------------------------------------------------------


def test_entailment_can_reject_a_claim_every_rule_accepts():
    """The judgement no rule can make. Retrieval returns what is topically close,
    and an agent can link a chunk about the right protocol and the right feature
    that says nothing about the assertion."""
    evs = [_ev(f"https://docs/{i}") for i in range(3)]
    claim = _claim("USDe is overcollateralised.", evs)

    assert verify(claim, evs, now=NOW).status is VerificationStatus.VERIFIED
    rejected = verify(claim, evs, entailer=_no, now=NOW)
    assert rejected.status is VerificationStatus.INSUFFICIENT_EVIDENCE


def test_entailment_is_not_consulted_on_an_already_rejected_claim():
    """The expensive check is never spent on a claim the free ones already
    rejected."""
    calls = []

    def counting(claim, evidence):
        calls.append(claim)
        return True, "yes"

    verify(_claim("Something.", []), [], entailer=counting, now=NOW)
    assert calls == []


def test_a_passing_entailer_leaves_the_verdict_verified():
    evs = [_ev(f"https://docs/{i}") for i in range(2)]
    assert verify(_claim("x", evs), evs, entailer=_yes, now=NOW).status is (
        VerificationStatus.VERIFIED
    )


# --- the agent -----------------------------------------------------------


def test_no_claims_is_reported_not_passed_over():
    out = verify_all([], [])
    assert out.verdicts == ()
    assert "nothing to verify" in out.limitations[0]


def test_skipping_entailment_is_disclosed():
    """Leaving the check off is a cost decision; hiding that it was left off
    would let the report imply the claims were checked for relevance."""
    evs = [_ev("https://a")]
    out = verify_all([_claim("x", evs)], evs)
    assert any("not as actually being about the claim" in l for l in out.limitations)


def test_running_entailment_removes_that_disclosure():
    evs = [_ev("https://a")]
    out = verify_all([_claim("x", evs)], evs, entailer=_yes)
    assert not any("semantic entailment was not checked" in l for l in out.limitations)


def test_rejected_claims_are_summarised_with_reasons():
    evs = [_ev("https://forum/a", tier=SourceTier.UNVERIFIED)]
    out = verify_all([_claim("Redemption is guaranteed.", evs)], evs)
    note = next(l for l in out.limitations if "did not\nfully verify" in l
                or "did not fully verify" in l)
    assert "overclaiming" in note


def test_statuses_are_counted_per_category():
    evs = [_ev(f"https://docs/{i}") for i in range(2)]
    good = _claim("Outflows rose.", evs)
    bad = _claim("Something.", [])
    out = verify_all([good, bad], evs)
    assert out.by_status["verified"] == 1
    assert out.by_status["insufficient_evidence"] == 1


def test_verification_adds_no_new_information():
    """§13: its role is not to generate. A verdict may only reference claims and
    evidence it was given."""
    evs = [_ev("https://a", "an observation")]
    claim = _claim("Outflows rose.", evs)
    verdict = verify(claim, evs, now=NOW)
    assert verdict.claim_text == claim.text
    assert verdict.claim_id == claim.claim_id
