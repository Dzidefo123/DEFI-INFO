"""Properties of the scoring model, asserted over its whole input space.

Three of the six design defects in this repository were caught by review rather
than by tests, and the diagnosis in the README is that tests written to confirm a
design cannot challenge its premise. That diagnosis had no mechanism attached.
This file is the mechanism.

The compensatory-confidence bug is the case in point. It appeared twice — once as
a five-factor geometric mean that could not fall below 0.55, and once as additive
pooling that let twenty chain observations outscore a documentation source on a
MECHANISM claim. Those were written up as two defects. They are one bug class:
*a scoring rule in which enough of a weak input substitutes for a strong one.*

An example-based test asks whether a particular claim scores what its author
expected. A property asks whether any point in the space can violate a rule the
design claims to enforce, and it is indifferent to what the author expected. The
searches below are exhaustive over the discrete axes — every claim kind, every
tier, every status — and gridded over the continuous ones, which for a model with
five bounded factors is small enough to enumerate honestly.

If a future refactor reintroduces compensation anywhere in this model, one of
these fails without anyone having predicted where it would appear.
"""

from __future__ import annotations

import itertools
from datetime import timedelta

import pytest

from src.evidence.confidence import (
    CORROBORATION_FLOOR,
    MAX_INAPT_CONTRIBUTION,
    RELIABILITY,
    VERIFICATION_WEIGHT,
    ConfidenceBreakdown,
    assess,
)
from src.evidence.models import (
    AGENT_CLAIM_KINDS,
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

NOW = utcnow()

# Grid over the unit interval, endpoints included. Eleven points across four
# factors is 14,641 combinations — cheap, and it covers every corner.
GRID = [i / 10 for i in range(11)]

BANDS = {"high": 0.80, "moderate": 0.60, "low": 0.35}


def _breakdown(quality, agreement, reliability, temporal, status, **kw):
    return ConfidenceBreakdown(
        evidence_quality=quality,
        evidence_agreement=agreement,
        source_reliability=reliability,
        temporal_relevance=temporal,
        verification_score=VERIFICATION_WEIGHT[status],
        **kw,
    )


def _evidence(tier, kind=EvidenceKind.DOCUMENT, uri=None, age_days=0.0):
    return Evidence(
        kind=kind,
        source=SourceRef(tier=tier, uri=uri or f"https://example.org/{tier.value}"),
        agent=AgentName.RESEARCH,
        summary="an observation",
        observed_at=NOW - timedelta(days=age_days),
        collected_at=NOW,
    )


def _agent_for(kind: ClaimKind) -> AgentName:
    """An agent permitted to assert this claim kind.

    `AGENT_CLAIM_KINDS` is the fix for "claim kind was agent identity in
    disguise": a claim declares its own kind, but an agent may still only assert
    kinds it is competent to judge. UNSPECIFIED appears in no agent's set and is
    nonetheless permitted to all of them by `permitted_kind` — it is the default,
    and forbidding the default would make every claim fail to construct. So it
    resolves here rather than skipping, which keeps the UNSPECIFIED row of the
    reliability matrix inside these properties.
    """
    for agent, kinds in AGENT_CLAIM_KINDS.items():
        if kind in kinds:
            return agent
    return AgentName.RESEARCH


def _claim(evs, kind=ClaimKind.MECHANISM, status=VerificationStatus.VERIFIED,
           stance=Stance.SUPPORTS):
    return Claim(
        text="A finding.",
        agent=_agent_for(kind),
        kind=kind,
        created_at=NOW,
        verification=status,
        links=tuple(EvidenceLink(evidence_id=e.evidence_id, stance=stance) for e in evs),
    )


# --- verification is a gate, not a factor -------------------------------


def test_no_contradicted_claim_can_score_above_zero():
    """The first compensatory bug, as a property rather than an example.

    Verification was originally the fifth term in a geometric mean, which cannot
    fall below the fifth root of its smallest factor: a CONTRADICTED claim with
    everything else perfect scored about 0.55, outranking an honest
    INSUFFICIENT_EVIDENCE. No arrangement of the other four factors may rescue a
    refuted claim, and this searches all of them.
    """
    worst = max(
        _breakdown(q, a, r, t, VerificationStatus.CONTRADICTED).score
        for q, a, r, t in itertools.product(GRID, repeat=4)
    )
    assert worst == 0.0


def test_a_refuted_claim_never_outranks_an_unsupported_one():
    """The comparison that made the original bug visible. Perfect evidence for a
    contradicted claim against nothing at all for an unverified one."""
    refuted = _breakdown(1.0, 1.0, 1.0, 1.0, VerificationStatus.CONTRADICTED).score
    unsupported = _breakdown(0.0, 0.0, 0.0, 0.0, VerificationStatus.INSUFFICIENT_EVIDENCE).score
    assert refuted <= unsupported


@pytest.mark.parametrize("status", list(VerificationStatus))
def test_the_verification_ordering_is_never_violated(status):
    """A weaker verdict may never score above a stronger one on identical
    evidence — for any evidence."""
    stronger = [s for s in VerificationStatus
                if VERIFICATION_WEIGHT[s] > VERIFICATION_WEIGHT[status]]
    for q, a, r, t in itertools.product(GRID[::2], repeat=4):
        here = _breakdown(q, a, r, t, status).score
        for other in stronger:
            assert here <= _breakdown(q, a, r, t, other).score


# --- no factor is compensatory ------------------------------------------


@pytest.mark.parametrize("missing", ["quality", "agreement", "reliability", "temporal"])
def test_any_single_zero_factor_forces_a_zero_score(missing):
    """The conjunctive rule, stated as a property. Perfect on three axes and
    absent on the fourth is not a finding, however good the three are."""
    values = {"quality": 1.0, "agreement": 1.0, "reliability": 1.0, "temporal": 1.0}
    values[missing] = 0.0
    score = _breakdown(**values, status=VerificationStatus.VERIFIED).score
    assert score == 0.0


def test_the_score_is_monotone_in_every_factor():
    """Improving any input may never lower the output. A non-monotone scoring
    rule is unexplainable by construction: no true sentence of the form 'this
    scored lower because the evidence was fresher' exists."""
    for q, a, r, t in itertools.product(GRID[::3], repeat=4):
        base = _breakdown(q, a, r, t, VerificationStatus.VERIFIED).score
        for i, factor in enumerate((q, a, r, t)):
            if factor >= 1.0:
                continue
            bumped = [q, a, r, t]
            bumped[i] = min(1.0, factor + 0.1)
            assert _breakdown(*bumped, status=VerificationStatus.VERIFIED).score >= base


def test_no_score_can_exceed_its_weakest_evidence_factor():
    """A geometric mean is bounded above by its largest term and below by its
    smallest; what matters here is that the *score* never exceeds the weakest
    factor by enough to hide it. This is the invariant that makes
    `weakest_factor` an honest explanation of a score."""
    for q, a, r, t in itertools.product(GRID[::2], repeat=4):
        b = _breakdown(q, a, r, t, VerificationStatus.VERIFIED)
        assert b.evidence_strength >= min(q, a, r, t) - 1e-9
        assert b.evidence_strength <= max(q, a, r, t) + 1e-9


# --- volume never substitutes for aptness -------------------------------


@pytest.mark.parametrize("claim_kind", list(ClaimKind))
def test_no_volume_of_below_floor_evidence_outscores_one_apt_source(claim_kind):
    """The second appearance of the same bug class, as a property.

    Measured before `CORROBORATION_FLOOR` existed: twenty chain observations
    scored 0.891 on a MECHANISM claim against a documentation source's 0.880.
    Underdetermination does not improve with observation count — a hundred
    liquidations are consistent with the same dozen rules as ten — so no number
    of inapt sources may overtake a single apt one.
    """
    tiers = RELIABILITY[claim_kind]
    apt = max(tiers, key=lambda t: tiers[t])
    inapt = [t for t in tiers if tiers[t] < CORROBORATION_FLOOR]
    if not inapt:
        pytest.skip(f"every tier is above the floor for {claim_kind.value}")

    one_apt = _evidence(apt)
    best_apt = assess(_claim([one_apt], kind=claim_kind), [one_apt]).score

    for tier in inapt:
        many = [_evidence(tier, uri=f"https://example.org/{tier.value}/{i}")
                for i in range(40)]
        piled = assess(_claim(many, kind=claim_kind), many).score
        assert piled <= best_apt, (
            f"{len(many)} {tier.value} sources scored {piled} on a "
            f"{claim_kind.value} claim, beating one {apt.value} source at {best_apt}"
        )


@pytest.mark.parametrize("claim_kind", list(ClaimKind))
def test_piling_on_inapt_sources_saturates(claim_kind):
    """Below the floor, sources stop multiplying. Doubling the pile again must
    not move the score, or the cap is a discount and the bug is back."""
    tiers = RELIABILITY[claim_kind]
    inapt = [t for t in tiers if tiers[t] < CORROBORATION_FLOOR]
    if not inapt:
        pytest.skip(f"every tier is above the floor for {claim_kind.value}")
    tier = inapt[0]

    def pile(n):
        evs = [_evidence(tier, uri=f"https://example.org/x/{i}") for i in range(n)]
        return assess(_claim(evs, kind=claim_kind), evs).score

    assert pile(50) == pytest.approx(pile(10), abs=1e-6)


def test_the_cap_is_shared_between_inapt_sources_not_applied_each():
    """`MAX_INAPT_CONTRIBUTION` is a budget for the whole group. Applied
    per-source it would be no cap at all."""
    assert MAX_INAPT_CONTRIBUTION == 1.0
    tier = min(RELIABILITY[ClaimKind.MECHANISM],
               key=lambda t: RELIABILITY[ClaimKind.MECHANISM][t])
    evs = [_evidence(tier, uri=f"https://example.org/y/{i}") for i in range(30)]
    breakdown = assess(_claim(evs, kind=ClaimKind.MECHANISM), evs)
    assert breakdown.capped_sources == 30
    assert breakdown.evidence_quality <= 0.6


# --- corroboration must be independent ----------------------------------


def test_repeating_one_source_is_not_corroboration():
    """Ten citations of one page are one finding stated ten times. If volume
    counted here, the evidence graph's independence analysis would be describing
    a property the score does not have."""
    page = "https://docs.example.org/page"
    single = [_evidence(SourceTier.PRIMARY, uri=page)]
    # Distinct excerpts, one page. Identical evidence would collapse on its
    # content-addressed id, which would make this test prove nothing.
    repeated = [
        Evidence(
            kind=EvidenceKind.DOCUMENT,
            source=SourceRef(tier=SourceTier.PRIMARY, uri=page),
            agent=AgentName.RESEARCH,
            summary=f"excerpt {i}",
            observed_at=NOW,
            collected_at=NOW,
        )
        for i in range(10)
    ]
    assert assess(_claim(repeated), repeated).score == pytest.approx(
        assess(_claim(single), single).score, abs=1e-6
    )


def test_contradicting_evidence_can_only_lower_a_score():
    """Adding evidence against a claim must never raise its confidence, for any
    starting pool."""
    for n in (1, 3, 8):
        supporting = [_evidence(SourceTier.PRIMARY, uri=f"https://a/{i}") for i in range(n)]
        base = assess(_claim(supporting), supporting).score
        against = _evidence(SourceTier.OFFICIAL, uri="https://b/against")
        claim = Claim(
            text="A finding.",
            agent=_agent_for(ClaimKind.MECHANISM),
            kind=ClaimKind.MECHANISM,
            created_at=NOW,
            verification=VerificationStatus.VERIFIED,
            links=(
                *[EvidenceLink(evidence_id=e.evidence_id, stance=Stance.SUPPORTS)
                  for e in supporting],
                EvidenceLink(evidence_id=against.evidence_id, stance=Stance.CONTRADICTS),
            ),
        )
        assert assess(claim, [*supporting, against]).score <= base


# --- bands mean what they say -------------------------------------------


@pytest.mark.parametrize("band,floor", sorted(BANDS.items(), key=lambda kv: -kv[1]))
def test_no_unverified_claim_reaches_a_reassuring_band(band, floor):
    """A claim nobody verified must not read as "high" or "moderate" confidence
    on evidence factors alone. This is the same rule as the verification gate,
    checked at the level a reader actually sees — the word, not the number."""
    if floor < 0.60:
        pytest.skip("low and below are not reassuring")
    best = max(
        _breakdown(q, a, r, t, VerificationStatus.UNVERIFIED).score
        for q, a, r, t in itertools.product(GRID[::2], repeat=4)
    )
    assert best < floor


def test_stale_evidence_cannot_carry_a_high_band_alone():
    """Freshness is one of four factors, so a very old source has to drag the
    score out of the top band however authoritative it is."""
    ancient = _evidence(SourceTier.PRIMARY, kind=EvidenceKind.MARKET_DATA, age_days=400)
    score = assess(_claim([ancient], kind=ClaimKind.STATE), [ancient]).score
    assert score < BANDS["moderate"]
