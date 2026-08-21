"""The confidence model of §16, made deterministic and inspectable.

    CONFIDENCE = Evidence Quality
               × Evidence Agreement
               × Source Reliability
               × Temporal Relevance
               × Verification Score

Three departures from the formula as written, all deliberate.

**The evidence factors are combined by geometric mean.** A bare product
collapses: four genuinely good factors of 0.9 multiply to 0.66, so a
well-evidenced claim reports as a coin flip and the scale stops meaning
anything. The geometric mean is that product renormalized onto the factors' own
scale — it keeps the property the multiplicative form was chosen FOR
(conjunctive: any one factor near zero drags the result down, and no amount of
excellence elsewhere buys it back) without the systematic deflation. `0.9`
across the board now reports `0.9`.

**Verification is a gate, not a fifth factor.** It multiplies the combined
evidence strength instead of joining the mean. This is not a detail — putting it
inside the mean makes it *compensatory*, and a compensatory verification stage
is a broken one. Concretely: with five factors, a term of 0.05 cannot pull the
result below 0.05^(1/5) ≈ 0.55, so a claim the Verification Agent had actively
refuted would still report as moderately confident, outscoring an honest
"insufficient evidence". §13 exists to stop unsupported conclusions reaching a
user; an aggregator that lets strong evidence outvote a refutation cannot do
that. Hence `CONTRADICTED` carries weight 0.0 — a refuted claim is not a weak
finding, it is not a finding, the same category as one with no support at all.

The split also reads better in a report, which is the §3.1 requirement:
"evidence strength 0.88, verification: partially verified → confidence 0.62"
says where the number came from in a way one blended figure never can.

**Nothing here is scored by a language model.** Every factor is computed from
the evidence links, the source tiers, the clock, and the verification verdict.
A model that rates its own evidence rates it well, and a confidence number built
on that measures the model's self-regard rather than the strength of the case.

The constants below are a transparent starting point, not calibrated truth.
They are named, gathered in one place, and reported alongside the score so a
reader can disagree with them specifically. Calibrating them against outcomes is
future work and should be labelled as such wherever a score is shown.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from src.evidence.models import (
    Claim,
    ClaimKind,
    Evidence,
    EvidenceKind,
    SourceTier,
    VerificationStatus,
    utcnow,
)

# --- calibration constants ---------------------------------------------

# Provenance weights, by what the claim asserts.
#
# A single ranking cannot be right, because the ordering inverts between the two
# most authoritative sources this system has. Documentation records what a
# protocol COMMITS TO; chain state records what it IS DOING.
#
#   "Reserves are $87.3M"                 -> the chain settles it. Documentation
#                                            describes intent, and intent can be
#                                            stale or aspirational.
#   "Liquidation uses a 3-minute TWAP"    -> the documentation settles it. You
#                                            cannot read a rule off a sequence of
#                                            transactions; chain state shows
#                                            behaviour consistent with many rules.
#
# So reliability is a matrix. Each row is a claim kind; each cell is what a
# source of that tier is worth for that kind of assertion.
#
# These are a transparent starting point, not calibrated truth — the same caveat
# that applies to every constant in this module.
RELIABILITY: Mapping[ClaimKind, Mapping[SourceTier, float]] = {
    # Current values. The chain is the thing itself.
    ClaimKind.STATE: {
        SourceTier.CHAIN: 1.00,
        SourceTier.OFFICIAL: 0.70,
        SourceTier.PRIMARY: 0.55,   # docs state intent, not the present value
        SourceTier.COMMUNITY: 0.45,
        SourceTier.UNVERIFIED: 0.20,
    },
    # How a system works by design. The protocol's own specification governs.
    ClaimKind.MECHANISM: {
        SourceTier.PRIMARY: 1.00,
        SourceTier.OFFICIAL: 0.85,
        SourceTier.CHAIN: 0.65,     # behaviour is consistent with many rules
        SourceTier.COMMUNITY: 0.50,
        SourceTier.UNVERIFIED: 0.25,
    },
    # Something that happened. Chain records are close to decisive; an
    # accountable publisher is close behind, and documentation rarely speaks to
    # a specific event at all.
    ClaimKind.EVENT: {
        SourceTier.CHAIN: 0.95,
        SourceTier.OFFICIAL: 0.90,
        SourceTier.PRIMARY: 0.65,
        SourceTier.COMMUNITY: 0.55,
        SourceTier.UNVERIFIED: 0.25,
    },
    # Undeclared. A flat ranking that assumes nothing about the assertion, and
    # deliberately never awards a full 1.00 — a claim whose kind nobody stated
    # has not earned the top of any column.
    ClaimKind.UNSPECIFIED: {
        SourceTier.CHAIN: 0.90,
        SourceTier.PRIMARY: 0.90,
        SourceTier.OFFICIAL: 0.80,
        SourceTier.COMMUNITY: 0.50,
        SourceTier.UNVERIFIED: 0.25,
    },
}

# Kept for callers that need a tier ranking without a claim in hand — the
# verification agent's source-quality gate, which asks only whether anything
# better than anonymous is behind a claim.
TIER_WEIGHT: Mapping[SourceTier, float] = RELIABILITY[ClaimKind.UNSPECIFIED]


def reliability_of(kind: ClaimKind, tier: SourceTier) -> float:
    """What a source of `tier` is worth for a claim of `kind`."""
    return RELIABILITY[kind][tier]

# Gate weights, applied to combined evidence strength.
#
# UNVERIFIED sits at a neutral 0.5 rather than 1.0: a claim the Verification
# Agent has not examined has not earned the benefit of the doubt, and must not
# score like one that passed.
#
# CONTRADICTED is 0.0, not a small number. The evidence pointing the other way
# is not a weak yes; it is a no, and it belongs in the same category as a claim
# with no support at all.
VERIFICATION_WEIGHT: Mapping[VerificationStatus, float] = {
    VerificationStatus.VERIFIED: 1.00,
    VerificationStatus.PARTIALLY_VERIFIED: 0.70,
    VerificationStatus.UNVERIFIED: 0.50,
    VerificationStatus.INSUFFICIENT_EVIDENCE: 0.25,
    VerificationStatus.CONTRADICTED: 0.00,
}

# How fast each kind of evidence goes stale, in days. These differ by orders of
# magnitude and that is the point: a documented mechanic is roughly as true next
# month as today, while a funding rate is worthless by tomorrow. One global
# freshness window would either treat live metrics as durable or throw away
# documentation that never expired.
HALF_LIFE_DAYS: Mapping[EvidenceKind, float] = {
    EvidenceKind.DOCUMENT: 365.0,
    EvidenceKind.AUDIT_FINDING: 365.0,
    EvidenceKind.SECURITY_ADVISORY: 180.0,
    EvidenceKind.INCIDENT_REPORT: 180.0,
    EvidenceKind.STATISTICAL_SIGNAL: 7.0,
    # Contract state is a fact about ONE BLOCK. A reserve balance read twelve
    # hours ago is a 1.00-reliability source vouching for a moment well past, and
    # `ON_CHAIN_METRIC`'s day-scale life would leave it at 0.71 — systematically
    # overconfident about the freshest-rotting thing in the system.
    EvidenceKind.CHAIN_STATE: 0.1667,   # 4 hours
    EvidenceKind.ON_CHAIN_METRIC: 1.0,
    EvidenceKind.MARKET_DATA: 0.25,
}

# Half-saturation point for evidence quality, in relevance-weighted independent
# sources. At this value quality is 0.5. Set below 1 so that a single directly
# relevant primary source — one authoritative docs page answering a
# documentation question — scores respectably rather than being punished for
# being sufficient.
QUALITY_HALF_SATURATION = 0.667

# Reliability below which a source is the wrong INSTRUMENT for a claim of that
# kind, and its observations stop corroborating each other.
#
# This is a cap, not a discount, and the distinction is the whole point.
# Underdetermination does not improve with observation count: a hundred
# liquidations are consistent with the same dozen rules as ten, so no number of
# chain observations establishes what the rule IS. A low weight alone does not
# express that — measured before this existed, twenty chain observations scored
# 0.891 against a documentation source's 0.880 on a MECHANISM claim, because the
# weight capped reliability while `evidence_quality` accumulated around it. That
# is the compensatory-confidence failure from §5, relocated into the matrix.
#
# So sources below the floor contribute at most one source's worth of
# corroboration BETWEEN THEM, however many there are. They still count — they are
# evidence — they just stop multiplying.
CORROBORATION_FLOOR = 0.70
MAX_INAPT_CONTRIBUTION = 1.0

# Floor for temporal relevance. Old evidence loses weight; it does not become
# evidence of nothing. A 2022 exploit report is still the reason a protocol has
# the mitigation it has.
MIN_TEMPORAL_RELEVANCE = 0.05


# --- the breakdown ------------------------------------------------------


class ConfidenceBreakdown(BaseModel):
    """A confidence score that shows its work.

    The five factors are retained, not just the product, because §3.1 requires
    the system to answer "how confident is it, and why". A single number cannot
    distinguish "excellent evidence that is six months stale" from "fresh
    evidence from an anonymous source", and those call for different responses.
    """

    model_config = ConfigDict(frozen=True)

    evidence_quality: float = Field(ge=0.0, le=1.0)
    evidence_agreement: float = Field(ge=0.0, le=1.0)
    source_reliability: float = Field(ge=0.0, le=1.0)
    temporal_relevance: float = Field(ge=0.0, le=1.0)
    verification_score: float = Field(ge=0.0, le=1.0)

    supporting_count: int = 0
    contradicting_count: int = 0
    distinct_sources: int = 0
    # Distinct sources that were the wrong instrument for this claim's kind and
    # therefore stopped corroborating one another. Reported so a low quality
    # score can be explained rather than merely observed.
    capped_sources: int = 0

    @property
    def evidence_factors(self) -> dict[str, float]:
        """The four factors describing the evidence itself, before verification."""
        return {
            "evidence_quality": self.evidence_quality,
            "evidence_agreement": self.evidence_agreement,
            "source_reliability": self.source_reliability,
            "temporal_relevance": self.temporal_relevance,
        }

    @property
    def factors(self) -> dict[str, float]:
        """Everything that went into the score, for reports and traversal."""
        return self.evidence_factors | {"verification_score": self.verification_score}

    @property
    def evidence_strength(self) -> float:
        """Geometric mean of the four evidence factors. Zero if any is zero.

        The short-circuit is not an optimization — it is the conjunctive rule
        stated plainly. A claim with no supporting evidence is not 'somewhat
        confident'; it is not a finding.
        """
        values = list(self.evidence_factors.values())
        if any(v <= 0.0 for v in values):
            return 0.0
        return math.exp(sum(math.log(v) for v in values) / len(values))

    @property
    def score(self) -> float:
        """Evidence strength, gated by the verification verdict.

        See the module docstring for why verification multiplies rather than
        joining the mean: inside the mean it would be compensatory, and a
        refuted claim would still outscore an honest "insufficient evidence".
        """
        return round(self.evidence_strength * self.verification_score, 4)

    @property
    def label(self) -> str:
        """Coarse band for prose. Reports should print the number too."""
        s = self.score
        if s >= 0.80:
            return "high"
        if s >= 0.60:
            return "moderate"
        if s >= 0.35:
            return "low"
        return "very low"

    def weakest_factor(self) -> tuple[str, float]:
        """Which factor is holding the score down — the one a report should name."""
        return min(self.factors.items(), key=lambda kv: kv[1])


# --- factor computations ------------------------------------------------


def temporal_relevance(evidence: Evidence, now=None) -> float:
    """Exponential decay from the evidence's truth time, by kind.

    Half-life, not a cliff: freshness degrades continuously, so nothing changes
    category the instant a threshold is crossed.
    """
    now = now or utcnow()
    half_life = HALF_LIFE_DAYS.get(evidence.kind, 30.0)
    age_days = max((now - evidence.as_of).total_seconds() / 86400.0, 0.0)
    decayed = 0.5 ** (age_days / half_life)
    return max(decayed, MIN_TEMPORAL_RELEVANCE)


def _weighted(links, by_id: Mapping[str, Evidence]) -> float:
    """Relevance-weighted count of links whose evidence we actually hold."""
    return sum(link.relevance for link in links if link.evidence_id in by_id)


def assess(
    claim: Claim, evidence: Iterable[Evidence], now=None
) -> ConfidenceBreakdown:
    """Score `claim` against the evidence pool it draws on.

    Links naming evidence absent from the pool are ignored rather than trusted.
    An agent asserting support from evidence nobody can produce is describing
    exactly the failure this system exists to catch, so a dangling link must not
    be able to raise a score.
    """
    now = now or utcnow()
    by_id = {e.evidence_id: e for e in evidence}

    supporting = [l for l in claim.supporting() if l.evidence_id in by_id]
    contradicting = [l for l in claim.contradicting() if l.evidence_id in by_id]

    # No support: not a weak finding, not a finding. Every factor that depends on
    # supporting evidence is zero, so the conjunctive rule returns zero.
    if not supporting:
        return ConfidenceBreakdown(
            evidence_quality=0.0,
            evidence_agreement=0.0,
            source_reliability=0.0,
            temporal_relevance=0.0,
            verification_score=VERIFICATION_WEIGHT[claim.verification],
            supporting_count=0,
            contradicting_count=len(contradicting),
            distinct_sources=0,
        )

    # Quality counts DISTINCT sources, so one page cited five times is one
    # source. Corroboration means independent observation; without this, an agent
    # that chunks a document finely looks better-evidenced than one that does not.
    #
    # Sources are then split by whether they are the right INSTRUMENT for this
    # kind of claim, and the wrong ones are capped rather than merely discounted.
    # See `CORROBORATION_FLOOR` — this is the fix for a measured failure where
    # twenty chain observations outscored a documentation source on a claim about
    # mechanism, because a low reliability weight capped one factor while quality
    # accumulated around it.
    apt: dict[str, float] = {}
    inapt: dict[str, float] = {}
    for link in supporting:
        item = by_id[link.evidence_id]
        bucket = (
            apt
            if reliability_of(claim.kind, item.source.tier) >= CORROBORATION_FLOOR
            else inapt
        )
        bucket[item.source.uri] = max(bucket.get(item.source.uri, 0.0), link.relevance)

    n_eff = sum(apt.values()) + min(sum(inapt.values()), MAX_INAPT_CONTRIBUTION)
    quality = n_eff / (n_eff + QUALITY_HALF_SATURATION)

    # Agreement: how one-sided the evidence is. All support -> 1.0.
    s_weight = _weighted(supporting, by_id)
    c_weight = _weighted(contradicting, by_id)
    agreement = s_weight / (s_weight + c_weight) if (s_weight + c_weight) else 0.0

    # Reliability is the BEST provenance backing the claim, not the average.
    # Averaging would mean that citing a corroborating forum post alongside the
    # protocol's own documentation LOWERS confidence — penalising thoroughness,
    # and pushing agents toward citing less.
    #
    # "Best" is resolved against what the claim ASSERTS, not against a fixed
    # ranking of sources: a chain read is the strongest support for a claim about
    # current state and among the weakest for a claim about mechanism. See
    # `RELIABILITY`.
    reliability = max(
        reliability_of(claim.kind, by_id[l.evidence_id].source.tier)
        for l in supporting
    )

    # Freshness is relevance-weighted across supporting evidence: an investigation
    # resting mostly on stale readings should say so even if one item is current.
    freshness = sum(
        temporal_relevance(by_id[l.evidence_id], now) * l.relevance for l in supporting
    ) / (s_weight or 1.0)

    return ConfidenceBreakdown(
        evidence_quality=round(quality, 4),
        evidence_agreement=round(agreement, 4),
        source_reliability=round(reliability, 4),
        temporal_relevance=round(min(freshness, 1.0), 4),
        verification_score=VERIFICATION_WEIGHT[claim.verification],
        supporting_count=len(supporting),
        contradicting_count=len(contradicting),
        distinct_sources=len(apt) + len(inapt),
        capped_sources=len(inapt) if len(inapt) > 1 else 0,
    )
