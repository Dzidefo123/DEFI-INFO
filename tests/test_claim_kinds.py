"""Claim-kind-aware source reliability.

A single ranking over sources cannot be right, because the ordering genuinely
inverts between the two most authoritative sources this system has:

    Documentation records what a protocol COMMITS TO.
    Chain state records what it IS DOING.

These tests pin the inversion, the reason it matters, and the guards that stop it
being gamed.
"""

from datetime import datetime, timezone

import pytest

from pydantic import ValidationError

from src.evidence.confidence import RELIABILITY, assess, reliability_of
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
)

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


def _ev(uri, tier, kind=EvidenceKind.DOCUMENT, summary="an observation"):
    return Evidence(
        kind=kind,
        source=SourceRef(tier=tier, uri=uri),
        agent=AgentName.RESEARCH,
        summary=summary,
        observed_at=NOW,
        collected_at=NOW,
    )


def _claim(evs, kind=ClaimKind.UNSPECIFIED, text="A claim.",
           verification=VerificationStatus.VERIFIED):
    return Claim(
        text=text,
        agent=AgentName.RESEARCH,
        created_at=NOW,
        kind=kind,
        verification=verification,
        links=tuple(
            EvidenceLink(evidence_id=e.evidence_id, stance=Stance.SUPPORTS) for e in evs
        ),
    )


# --- the matrix is total and ordered ------------------------------------


@pytest.mark.parametrize("kind", list(ClaimKind))
def test_every_claim_kind_ranks_every_tier(kind):
    """A missing cell would KeyError at scoring time, on a claim someone is
    waiting on."""
    for tier in SourceTier:
        assert 0.0 < reliability_of(kind, tier) <= 1.0


@pytest.mark.parametrize("kind", list(ClaimKind))
def test_an_anonymous_source_is_worst_for_every_kind(kind):
    """The one ordering that does NOT invert."""
    weights = RELIABILITY[kind]
    assert weights[SourceTier.UNVERIFIED] == min(weights.values())


# --- the inversion -------------------------------------------------------


def test_the_chain_outranks_documentation_for_a_claim_about_state():
    """Documentation describes intent, and intent can be stale or aspirational."""
    assert reliability_of(ClaimKind.STATE, SourceTier.CHAIN) > reliability_of(
        ClaimKind.STATE, SourceTier.PRIMARY
    )


def test_documentation_outranks_the_chain_for_a_claim_about_mechanism():
    """You cannot read a rule off a sequence of transactions — observed behaviour
    is consistent with many rules."""
    assert reliability_of(ClaimKind.MECHANISM, SourceTier.PRIMARY) > reliability_of(
        ClaimKind.MECHANISM, SourceTier.CHAIN
    )


def test_the_ordering_genuinely_inverts_rather_than_merely_narrowing():
    """The point of a matrix rather than a list: the same two sources swap places
    depending on what is being asserted."""
    chain_state = reliability_of(ClaimKind.STATE, SourceTier.CHAIN)
    docs_state = reliability_of(ClaimKind.STATE, SourceTier.PRIMARY)
    chain_mech = reliability_of(ClaimKind.MECHANISM, SourceTier.CHAIN)
    docs_mech = reliability_of(ClaimKind.MECHANISM, SourceTier.PRIMARY)

    assert chain_state > docs_state
    assert docs_mech > chain_mech


def test_an_undeclared_claim_treats_them_alike_and_tops_out_below_one():
    """A claim whose kind nobody stated has not earned the top of any column."""
    chain = reliability_of(ClaimKind.UNSPECIFIED, SourceTier.CHAIN)
    docs = reliability_of(ClaimKind.UNSPECIFIED, SourceTier.PRIMARY)
    assert chain == docs
    assert max(RELIABILITY[ClaimKind.UNSPECIFIED].values()) < 1.0


def test_an_accountable_publisher_leads_on_events():
    """Auditors and CVE publishers speak to what happened; a protocol's own docs
    rarely mention a specific incident at all."""
    assert reliability_of(ClaimKind.EVENT, SourceTier.OFFICIAL) > reliability_of(
        ClaimKind.EVENT, SourceTier.PRIMARY
    )


# --- what it does to a score --------------------------------------------


def test_the_same_evidence_scores_differently_by_what_is_claimed():
    """The whole point, end to end."""
    chain = _ev("https://rpc/x", SourceTier.CHAIN, EvidenceKind.ON_CHAIN_METRIC)

    as_state = assess(_claim([chain], ClaimKind.STATE), [chain], now=NOW)
    as_mechanism = assess(_claim([chain], ClaimKind.MECHANISM), [chain], now=NOW)

    assert as_state.source_reliability > as_mechanism.source_reliability
    assert as_state.score > as_mechanism.score


def test_a_reserves_claim_is_settled_by_the_chain_not_the_docs():
    """The case that motivated the matrix. "Holds exactly $87,300,000 in reserves"
    is a claim about current state: reading it off a documentation page is weak
    evidence, and reading it off the chain is the strongest evidence there is."""
    from_docs = _ev("https://docs.x/reserves", SourceTier.PRIMARY)
    from_chain = _ev("https://rpc/x", SourceTier.CHAIN, EvidenceKind.ON_CHAIN_METRIC)

    documented = assess(_claim([from_docs], ClaimKind.STATE), [from_docs], now=NOW)
    measured = assess(_claim([from_chain], ClaimKind.STATE), [from_chain], now=NOW)

    assert measured.source_reliability == 1.0
    assert documented.source_reliability < 0.6
    assert measured.score > documented.score


def test_a_mechanism_claim_is_settled_by_the_docs_not_the_chain():
    """And the mirror image, which is what makes this a matrix rather than a
    promotion of chain reads."""
    from_docs = _ev("https://docs.x/liquidation", SourceTier.PRIMARY)
    from_chain = _ev("https://rpc/x", SourceTier.CHAIN, EvidenceKind.ON_CHAIN_METRIC)

    documented = assess(_claim([from_docs], ClaimKind.MECHANISM), [from_docs], now=NOW)
    measured = assess(_claim([from_chain], ClaimKind.MECHANISM), [from_chain], now=NOW)

    assert documented.source_reliability == 1.0
    assert documented.score > measured.score


def test_reliability_still_takes_the_best_source_for_the_kind():
    """Corroboration must never lower a score — but "best" is now resolved
    against what the claim asserts, not a fixed ranking."""
    chain = _ev("https://rpc/x", SourceTier.CHAIN, EvidenceKind.ON_CHAIN_METRIC)
    forum = _ev("https://forum/y", SourceTier.UNVERIFIED)

    alone = assess(_claim([chain], ClaimKind.STATE), [chain], now=NOW)
    both = assess(_claim([chain, forum], ClaimKind.STATE), [chain, forum], now=NOW)

    assert both.source_reliability == alone.source_reliability == 1.0


# --- identity and the guard against gaming ------------------------------


def test_claim_kind_is_not_part_of_claim_identity():
    """Two agents asserting the same sentence about the same protocol are making
    one claim. Disagreeing about how to weigh it must not split the evidence
    pooled behind it — the same reasoning that keeps `agent` out of the id."""
    ev = [_ev("https://x/a", SourceTier.PRIMARY)]
    as_state = _claim(ev, ClaimKind.STATE, text="Reserves are $10M.")
    as_mechanism = _claim(ev, ClaimKind.MECHANISM, text="Reserves are $10M.")
    assert as_state.claim_id == as_mechanism.claim_id


def test_an_undeclared_claim_defaults_rather_than_guessing():
    """Inferring the kind from wording would put a language model in charge of
    how heavily its own evidence counts."""
    ev = [_ev("https://x/a", SourceTier.PRIMARY)]
    assert Claim(text="x", agent=AgentName.RESEARCH).kind is ClaimKind.UNSPECIFIED


def test_declaring_a_kind_cannot_manufacture_certainty_from_a_weak_source():
    """The obvious way to game a matrix is to pick the row that flatters your
    evidence. It does not help: an anonymous source is the weakest cell in every
    row, so no declaration rescues it."""
    forum = _ev("https://forum/y", SourceTier.UNVERIFIED)
    scores = [
        assess(_claim([forum], kind), [forum], now=NOW).source_reliability
        for kind in AGENT_CLAIM_KINDS[AgentName.RESEARCH] | {ClaimKind.UNSPECIFIED}
    ]
    assert max(scores) <= 0.25


# --- kind is declared per claim, bounded by competence -------------------


def test_an_agent_can_emit_more_than_one_kind():
    """The fix for kind being agent identity in disguise. A per-agent constant
    would be wrong in a predictable direction: a docs page stating "reserves are
    $87.3M" is a claim about STATE, and locking research to MECHANISM would score
    that documentation at 1.00 in the row where documentation is weakest."""
    assert len(AGENT_CLAIM_KINDS[AgentName.RESEARCH]) > 1
    assert ClaimKind.STATE in AGENT_CLAIM_KINDS[AgentName.RESEARCH]
    assert ClaimKind.MECHANISM in AGENT_CLAIM_KINDS[AgentName.RESEARCH]


def test_a_documented_value_is_scored_as_state_not_mechanism():
    """The concrete case. Same page, same sentence — declaring what it actually
    asserts drops reliability from 1.00 to 0.55, which is correct: a figure read
    off documentation is weak evidence for that figure today."""
    page = _ev("https://docs.x/reserves", SourceTier.PRIMARY)
    as_mechanism = assess(_claim([page], ClaimKind.MECHANISM), [page], now=NOW)
    as_state = assess(_claim([page], ClaimKind.STATE), [page], now=NOW)

    assert as_mechanism.source_reliability == 1.00
    assert as_state.source_reliability == 0.55


def test_an_agent_cannot_assert_outside_its_competence():
    """What stops the declaration being a free parameter: the thing being scored
    does not get to pick its own row."""
    ev = [_ev("https://docs/a", SourceTier.PRIMARY)]
    with pytest.raises(ValidationError, match="may not assert a event claim"):
        _claim(ev, ClaimKind.EVENT)


def test_declining_to_declare_is_always_allowed():
    """It carries its own penalty in the matrix; it is not a competence claim."""
    ev = [_ev("https://docs/a", SourceTier.PRIMARY)]
    assert _claim(ev, ClaimKind.UNSPECIFIED).kind is ClaimKind.UNSPECIFIED


def test_the_research_synthesiser_may_declare_state():
    from src.agents.research import ProposedClaim, link_claims

    evidence = [_ev("https://docs/a", SourceTier.PRIMARY)]
    claims, _ = link_claims(
        [ProposedClaim(text="Reserves are $87.3M.", excerpts=[1], kind=ClaimKind.STATE)],
        evidence,
    )
    assert claims[0].kind is ClaimKind.STATE


def test_an_out_of_competence_declaration_falls_back_rather_than_failing():
    """The claim itself may still be sound; one bad label should not take down an
    investigation."""
    from src.agents.research import ProposedClaim, link_claims

    evidence = [_ev("https://docs/a", SourceTier.PRIMARY)]
    claims, _ = link_claims(
        [ProposedClaim(text="An incident happened.", excerpts=[1], kind=ClaimKind.EVENT)],
        evidence,
    )
    assert claims[0].kind is ClaimKind.MECHANISM


# --- the corroboration floor --------------------------------------------


def test_chain_observations_cannot_outvote_documentation_on_mechanism():
    """The measured failure this floor exists for.

    Before it, twenty chain observations scored 0.891 against a documentation
    source's 0.880 on a MECHANISM claim — the low reliability weight capped one
    factor while `evidence_quality` accumulated around it. Underdetermination does
    not improve with observation count: a hundred liquidations are consistent
    with the same dozen rules as ten.
    """
    docs = [_ev("https://docs.x/liq", SourceTier.PRIMARY)]
    documented = assess(_claim(docs, ClaimKind.MECHANISM), docs, now=NOW)

    for n in (5, 20, 100):
        chain = [
            _ev(f"https://rpc/tx{i}", SourceTier.CHAIN, EvidenceKind.ON_CHAIN_METRIC)
            for i in range(n)
        ]
        observed = assess(_claim(chain, ClaimKind.MECHANISM), chain, now=NOW)
        assert observed.score < documented.score, f"{n} observations outvoted the docs"


def test_the_wrong_instrument_is_capped_not_discounted():
    """A discount still accumulates; a cap does not. Ten inappropriate sources
    contribute no more corroboration than one."""
    def chain_quality(n):
        evs = [
            _ev(f"https://rpc/{i}", SourceTier.CHAIN, EvidenceKind.ON_CHAIN_METRIC)
            for i in range(n)
        ]
        return assess(_claim(evs, ClaimKind.MECHANISM), evs, now=NOW).evidence_quality

    assert chain_quality(1) == chain_quality(10) == chain_quality(50)


def test_the_right_instrument_still_corroborates_normally():
    """The floor must not break ordinary corroboration."""
    def docs_quality(n):
        evs = [_ev(f"https://docs/{i}", SourceTier.PRIMARY) for i in range(n)]
        return assess(_claim(evs, ClaimKind.MECHANISM), evs, now=NOW).evidence_quality

    assert docs_quality(1) < docs_quality(3) < docs_quality(10)


def test_documentation_stops_corroborating_a_claim_about_state():
    """The mirror case, and the one that shows this is not special-casing the
    chain: ten docs pages stating a value are still one class of evidence for
    what the value is right now."""
    def quality(n):
        evs = [_ev(f"https://docs/{i}", SourceTier.PRIMARY) for i in range(n)]
        return assess(_claim(evs, ClaimKind.STATE), evs, now=NOW).evidence_quality

    assert quality(1) == quality(10)


def test_capped_sources_are_reported_so_low_quality_can_be_explained():
    evs = [
        _ev(f"https://rpc/{i}", SourceTier.CHAIN, EvidenceKind.ON_CHAIN_METRIC)
        for i in range(6)
    ]
    b = assess(_claim(evs, ClaimKind.MECHANISM), evs, now=NOW)
    assert b.distinct_sources == 6
    assert b.capped_sources == 6


# --- the agents declare what they assert --------------------------------


def test_research_claims_are_declared_as_mechanism():
    from src.agents.research import ProposedClaim, link_claims

    evidence = [_ev("https://docs/a", SourceTier.PRIMARY)]
    claims, _ = link_claims([ProposedClaim(text="A rule.", excerpts=[1])], evidence)
    assert claims[0].kind is ClaimKind.MECHANISM


def test_security_incident_claims_are_declared_as_events():
    from src.agents.security import claims_from_records, evidence_from_record
    from src.security.incidents import Classification, IncidentRecord

    record = IncidentRecord(
        id="i1", protocol="ethena", classification=Classification.CONFIRMED_INCIDENT,
        title="An incident", summary="It happened.",
        source_uri="https://example.org/r", source_tier=SourceTier.OFFICIAL,
    )
    evidence = [evidence_from_record(record)]
    assert claims_from_records([record], evidence)[0].kind is ClaimKind.EVENT
