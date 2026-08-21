"""The §16 confidence model.

Every number here is computed from evidence links, source tiers, the clock, and
the verification verdict — never asserted by a model. These tests pin the
behaviours that make that worth doing.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.evidence.confidence import (
    HALF_LIFE_DAYS,
    reliability_of,
    MIN_TEMPORAL_RELEVANCE,
    TIER_WEIGHT,
    VERIFICATION_WEIGHT,
    ConfidenceBreakdown,
    assess,
    temporal_relevance,
)
from src.evidence.models import (
    AgentName,
    Claim,
    ClaimKind,
    Evidence,
    EvidenceKind,
    EvidenceLink,
    SourceRef,
    SourceTier,
    Stance,
    VerificationStatus,
    utcnow,
)

# Fixtures are stamped relative to the real clock, not a fixed date. Confidence
# decays with evidence age against wall-clock time, so a hardcoded "now" means
# these fixtures age one day per day: they encode day-zero freshness and then
# drift out of it. That is how this file broke the morning after it was written.
# Offsets below (NOW - timedelta(...)) still express deliberate staleness.
NOW = utcnow()


def _ev(uri, tier=SourceTier.PRIMARY, kind=EvidenceKind.DOCUMENT, observed_at=NOW, **kw):
    return Evidence(
        kind=kind,
        source=SourceRef(tier=tier, uri=uri),
        agent=AgentName.RESEARCH,
        summary=kw.pop("summary", f"observation from {uri}"),
        observed_at=observed_at,
        collected_at=NOW,
        **kw,
    )


def _claim(evs, stances=None, relevances=None, verification=VerificationStatus.UNVERIFIED):
    stances = stances or [Stance.SUPPORTS] * len(evs)
    relevances = relevances or [1.0] * len(evs)
    return Claim(
        text="Protocol X uses overcollateralized lending.",
        agent=AgentName.RESEARCH,
        created_at=NOW,
        verification=verification,
        links=tuple(
            EvidenceLink(evidence_id=e.evidence_id, stance=s, relevance=r)
            for e, s, r in zip(evs, stances, relevances)
        ),
    )


# --- the geometric-mean decision ---------------------------------------


def _bd(quality=1.0, agreement=1.0, reliability=1.0, temporal=1.0, verification=1.0):
    return ConfidenceBreakdown(
        evidence_quality=quality,
        evidence_agreement=agreement,
        source_reliability=reliability,
        temporal_relevance=temporal,
        verification_score=verification,
    )


def test_uniform_evidence_factors_report_themselves():
    """The reason for the geometric mean. A bare product of four 0.9s gives 0.66,
    so a well-evidenced claim would report as a coin flip and the scale would
    stop meaning anything."""
    assert _bd(0.9, 0.9, 0.9, 0.9).evidence_strength == pytest.approx(0.9, abs=1e-6)


def test_the_evidence_factors_combine_conjunctively():
    """The property the multiplicative form was chosen FOR: one bad factor drags
    the whole score down, and excellence elsewhere cannot buy it back."""
    assert _bd().evidence_strength == 1.0
    assert _bd(temporal=0.1).evidence_strength < 0.7


def test_any_zero_evidence_factor_zeroes_the_score():
    assert _bd(agreement=0.0).score == 0.0


def test_verification_gates_the_score_rather_than_averaging_into_it():
    """The flaw this model shipped with, pinned. Inside a five-factor geometric
    mean, a term of 0.05 cannot pull the result below 0.05^(1/5) ~= 0.55 — so a
    REFUTED claim would have reported as moderately confident and outscored an
    honest 'insufficient evidence'. A verification stage that strong evidence can
    outvote is not a verification stage."""
    perfectly_evidenced = _bd(verification=0.0)
    assert perfectly_evidenced.evidence_strength == 1.0
    assert perfectly_evidenced.score == 0.0


def test_the_score_is_strength_times_the_gate():
    b = _bd(0.8, 0.8, 0.8, 0.8, verification=0.7)
    assert b.score == pytest.approx(0.8 * 0.7, abs=1e-3)


def test_weakest_factor_names_what_is_holding_the_score_down():
    b = _bd(quality=0.9, agreement=0.95, reliability=0.3, temporal=0.8)
    assert b.weakest_factor() == ("source_reliability", 0.3)


def test_weakest_factor_can_name_verification():
    """A report saying 'nothing has checked this yet' is more actionable than one
    saying 'confidence 0.44'."""
    assert _bd(verification=0.5).weakest_factor() == ("verification_score", 0.5)


@pytest.mark.parametrize(
    "evidence,verification,expected",
    [
        (1.0, 1.0, "high"),
        (0.9, 0.7, "moderate"),   # strong evidence, only partially verified
        (0.8, 0.5, "low"),        # strong evidence, never examined
        (0.9, 0.25, "very low"),  # strong evidence, judged insufficient
        (0.9, 0.0, "very low"),   # strong evidence, refuted
    ],
)
def test_labels_band_the_score(evidence, verification, expected):
    assert _bd(evidence, evidence, evidence, evidence, verification).label == expected


# --- unsupported claims -------------------------------------------------


def test_an_unsupported_claim_scores_zero():
    """Not 'low confidence' — not a finding at all."""
    claim = Claim(text="Something happened.", agent=AgentName.RESEARCH, created_at=NOW)
    assert assess(claim, [], now=NOW).score == 0.0


def test_neutral_evidence_alone_scores_zero():
    ev = _ev("https://docs.example/a")
    claim = _claim([ev], stances=[Stance.NEUTRAL])
    assert assess(claim, [ev], now=NOW).score == 0.0


def test_a_link_to_evidence_nobody_can_produce_cannot_raise_the_score():
    """An agent claiming support from evidence absent from the pool is the exact
    failure this architecture exists to catch, so a dangling link is ignored
    rather than trusted."""
    real = _ev("https://docs.example/a")
    claim = Claim(
        text="Protocol X uses overcollateralized lending.",
        agent=AgentName.RESEARCH,
        created_at=NOW,
        links=(
            EvidenceLink(evidence_id=real.evidence_id, stance=Stance.SUPPORTS),
            EvidenceLink(evidence_id="ev_fabricated", stance=Stance.SUPPORTS),
        ),
    )
    with_ghost = assess(claim, [real], now=NOW)
    honest = assess(_claim([real]), [real], now=NOW)
    assert with_ghost.supporting_count == 1
    assert with_ghost.score == honest.score


# --- evidence quality counts independent sources -----------------------


def test_quality_rises_with_independent_sources():
    evs = [_ev(f"https://docs.example/{i}") for i in range(4)]
    scores = [
        assess(_claim(evs[:n]), evs[:n], now=NOW).evidence_quality for n in (1, 2, 3, 4)
    ]
    assert scores == sorted(scores)
    assert scores[0] < scores[-1] < 1.0  # saturating, never certain


def test_one_page_cited_five_times_is_one_source():
    """Corroboration means independent observation. Without this, an agent that
    chunks a document finely looks better-evidenced than one that does not."""
    same_page = [
        _ev("https://docs.example/funding", summary=f"chunk {i}") for i in range(5)
    ]
    five_pages = [_ev(f"https://docs.example/p{i}") for i in range(5)]

    chunked = assess(_claim(same_page), same_page, now=NOW)
    distinct = assess(_claim(five_pages), five_pages, now=NOW)

    assert chunked.distinct_sources == 1
    assert distinct.distinct_sources == 5
    assert chunked.evidence_quality < distinct.evidence_quality


def test_a_single_directly_relevant_primary_source_is_not_punished():
    """A documentation question answered by the protocol's own docs page is the
    ordinary, correct case — it must not score as weak."""
    ev = _ev("https://docs.ethena.fi/how-usde-works")
    b = assess(_claim([ev], verification=VerificationStatus.VERIFIED), [ev], now=NOW)
    assert b.evidence_quality >= 0.55
    assert b.label in ("high", "moderate")


def test_low_relevance_support_counts_for_less():
    ev = _ev("https://docs.example/a")
    strong = assess(_claim([ev], relevances=[1.0]), [ev], now=NOW)
    weak = assess(_claim([ev], relevances=[0.2]), [ev], now=NOW)
    assert weak.evidence_quality < strong.evidence_quality


# --- agreement ----------------------------------------------------------


def test_undisputed_evidence_agrees_completely():
    evs = [_ev(f"https://docs.example/{i}") for i in range(3)]
    assert assess(_claim(evs), evs, now=NOW).evidence_agreement == 1.0


def test_contradiction_lowers_agreement():
    evs = [_ev("https://docs.example/a"), _ev("https://docs.example/b")]
    claim = _claim(evs, stances=[Stance.SUPPORTS, Stance.CONTRADICTS])
    b = assess(claim, evs, now=NOW)
    assert b.evidence_agreement == pytest.approx(0.5)
    assert b.contradicting_count == 1


def test_agreement_is_relevance_weighted_not_a_headcount():
    """A marginally relevant objection should not cancel a directly relevant
    finding one-for-one."""
    evs = [_ev("https://docs.example/a"), _ev("https://docs.example/b")]
    claim = _claim(
        evs, stances=[Stance.SUPPORTS, Stance.CONTRADICTS], relevances=[1.0, 0.1]
    )
    assert assess(claim, evs, now=NOW).evidence_agreement > 0.85


# --- reliability is the best provenance, not the average ---------------


def test_reliability_takes_the_best_source_not_the_mean():
    """Averaging would mean that citing a corroborating forum post ALONGSIDE the
    protocol's own documentation lowers confidence — penalising thoroughness and
    pushing agents toward citing less."""
    primary = _ev("https://docs.ethena.fi/x", tier=SourceTier.PRIMARY)
    forum = _ev("https://forum.example/y", tier=SourceTier.UNVERIFIED)

    alone = assess(_claim([primary]), [primary], now=NOW)
    corroborated = assess(_claim([primary, forum]), [primary, forum], now=NOW)

    top = reliability_of(ClaimKind.UNSPECIFIED, SourceTier.PRIMARY)
    assert corroborated.source_reliability == alone.source_reliability == top
    assert corroborated.score >= alone.score


def test_weaker_provenance_scores_lower():
    tiers = [
        SourceTier.CHAIN,
        SourceTier.PRIMARY,
        SourceTier.OFFICIAL,
        SourceTier.COMMUNITY,
        SourceTier.UNVERIFIED,
    ]
    scores = []
    for tier in tiers:
        ev = _ev("https://example/x", tier=tier)
        scores.append(assess(_claim([ev]), [ev], now=NOW).source_reliability)
    assert scores == sorted(scores, reverse=True)
    assert scores == [TIER_WEIGHT[t] for t in tiers]


# --- temporal decay -----------------------------------------------------


def test_evidence_observed_now_is_fully_relevant():
    assert temporal_relevance(_ev("https://x/a", observed_at=NOW), now=NOW) == 1.0


def test_one_half_life_halves_relevance():
    kind = EvidenceKind.ON_CHAIN_METRIC
    age = timedelta(days=HALF_LIFE_DAYS[kind])
    ev = _ev("https://x/a", kind=kind, observed_at=NOW - age)
    assert temporal_relevance(ev, now=NOW) == pytest.approx(0.5, abs=1e-6)


def test_a_funding_rate_goes_stale_far_faster_than_a_documented_mechanic():
    """The reason half-lives are per-kind. One global freshness window would
    either treat live metrics as durable or discard documentation that never
    expired."""
    day_old = timedelta(days=1)
    market = _ev("https://api/x", kind=EvidenceKind.MARKET_DATA, observed_at=NOW - day_old)
    doc = _ev("https://docs/x", kind=EvidenceKind.DOCUMENT, observed_at=NOW - day_old)
    assert temporal_relevance(market, now=NOW) < 0.1
    assert temporal_relevance(doc, now=NOW) > 0.99


def test_very_old_evidence_keeps_a_floor():
    """A 2022 exploit report is still the reason a protocol has the mitigation it
    has. Old evidence loses weight; it does not become evidence of nothing."""
    ancient = _ev(
        "https://x/a",
        kind=EvidenceKind.MARKET_DATA,
        observed_at=NOW - timedelta(days=4000),
    )
    assert temporal_relevance(ancient, now=NOW) == MIN_TEMPORAL_RELEVANCE


def test_future_timestamps_do_not_exceed_full_relevance():
    """Clock skew between a chain and this host must not manufacture confidence."""
    skewed = _ev("https://x/a", observed_at=NOW + timedelta(hours=6))
    assert temporal_relevance(skewed, now=NOW) == 1.0


def test_freshness_is_averaged_across_supporting_evidence():
    """An investigation resting mostly on stale readings should say so even when
    one item is current."""
    kind = EvidenceKind.ON_CHAIN_METRIC
    fresh = _ev("https://x/a", kind=kind, observed_at=NOW)
    stale = [
        _ev(f"https://x/b{i}", kind=kind, observed_at=NOW - timedelta(days=10))
        for i in range(3)
    ]
    evs = [fresh, *stale]
    assert assess(_claim(evs), evs, now=NOW).temporal_relevance < 0.4


# --- verification -------------------------------------------------------


def test_verification_statuses_are_ordered_by_credit():
    order = [
        VerificationStatus.VERIFIED,
        VerificationStatus.PARTIALLY_VERIFIED,
        VerificationStatus.UNVERIFIED,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
        VerificationStatus.CONTRADICTED,
    ]
    weights = [VERIFICATION_WEIGHT[s] for s in order]
    assert weights == sorted(weights, reverse=True)


def test_an_unexamined_claim_does_not_score_like_a_verified_one():
    """UNVERIFIED sits at a neutral 0.5, not 1.0: a claim the Verification Agent
    has not looked at has not earned the benefit of the doubt."""
    ev = _ev("https://docs.example/a")
    unverified = assess(_claim([ev]), [ev], now=NOW)
    verified = assess(
        _claim([ev], verification=VerificationStatus.VERIFIED), [ev], now=NOW
    )
    assert unverified.score < verified.score


def test_a_contradicted_claim_collapses_even_with_perfect_evidence():
    evs = [_ev(f"https://docs.example/{i}") for i in range(5)]
    claim = _claim(evs, verification=VerificationStatus.CONTRADICTED)
    assert assess(claim, evs, now=NOW).label == "very low"


# --- the worked example from the architecture doc ----------------------


def test_abnormal_withdrawals_scenario_scores_high_and_the_cause_does_not():
    """§13.2: the system may detect abnormal activity without claiming to know
    why. Four corroborating sources and a VERIFIED verdict versus one source,
    two contradictions and INSUFFICIENT_EVIDENCE must land far apart."""
    kind = EvidenceKind.ON_CHAIN_METRIC
    supporting = [_ev(f"https://chain/{i}", kind=kind) for i in range(4)]
    observed = _claim(supporting, verification=VerificationStatus.VERIFIED)

    one = _ev("https://forum/rumor", tier=SourceTier.UNVERIFIED, kind=kind)
    against = [_ev(f"https://chain/c{i}", kind=kind) for i in range(2)]
    caused_by_exploit = _claim(
        [one, *against],
        stances=[Stance.SUPPORTS, Stance.CONTRADICTS, Stance.CONTRADICTS],
        verification=VerificationStatus.INSUFFICIENT_EVIDENCE,
    )

    strong = assess(observed, supporting, now=NOW)
    weak = assess(caused_by_exploit, [one, *against], now=NOW)

    assert strong.label == "high"
    assert weak.label == "very low"
    assert strong.score > weak.score * 3
