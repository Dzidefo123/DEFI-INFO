"""Evidence and claim invariants. Pure data — no model, no network, no clock drift."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

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
    normalize_claim_text,
)

T0 = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


def _source(uri="https://docs.ethena.fi/how-usde-works", **kw):
    kw.setdefault("tier", SourceTier.PRIMARY)
    kw.setdefault("protocol", "ethena")
    return SourceRef(uri=uri, **kw)


def _evidence(summary="USDe is backed by delta-neutral positions.", **kw):
    kw.setdefault("kind", EvidenceKind.DOCUMENT)
    kw.setdefault("source", _source())
    kw.setdefault("agent", AgentName.RESEARCH)
    kw.setdefault("collected_at", T0)
    return Evidence(summary=summary, **kw)


def _claim(text="USDe is backed by delta-neutral positions.", **kw):
    kw.setdefault("agent", AgentName.RESEARCH)
    kw.setdefault("created_at", T0)
    return Claim(text=text, **kw)


# --- content-addressed identity ----------------------------------------


def test_identical_evidence_has_one_id():
    assert _evidence().evidence_id == _evidence().evidence_id


def test_the_same_fact_found_by_two_agents_is_one_evidence_node():
    """Authorship is metadata. If it were identity, running more agents would
    inflate the independent-source count that confidence is built on."""
    research = _evidence(agent=AgentName.RESEARCH)
    security = _evidence(agent=AgentName.SECURITY)
    assert research.evidence_id == security.evidence_id
    assert research.agent is not security.agent


def test_recollecting_unchanged_evidence_does_not_mint_a_new_id():
    """`collected_at` is excluded from the digest: re-reading the same fact
    tomorrow must not look like a second, corroborating observation."""
    today = _evidence(collected_at=T0)
    tomorrow = _evidence(collected_at=T0 + timedelta(days=1))
    assert today.evidence_id == tomorrow.evidence_id


def test_different_observation_times_are_different_evidence():
    """Unlike collection time, the time a fact was TRUE is part of what it is:
    TVL at block N and at block N+1000 are two observations, not one."""
    a = _evidence(kind=EvidenceKind.ON_CHAIN_METRIC, observed_at=T0)
    b = _evidence(kind=EvidenceKind.ON_CHAIN_METRIC, observed_at=T0 + timedelta(hours=1))
    assert a.evidence_id != b.evidence_id


def test_payload_is_part_of_identity():
    a = _evidence(payload={"tvl_usd": 1_000_000})
    b = _evidence(payload={"tvl_usd": 2_000_000})
    assert a.evidence_id != b.evidence_id


def test_payload_key_order_does_not_change_the_id():
    """Dict ordering is an implementation detail of whichever agent built it."""
    a = _evidence(payload={"tvl": 1, "z": 2})
    b = _evidence(payload={"z": 2, "tvl": 1})
    assert a.evidence_id == b.evidence_id


def test_ids_are_prefixed_for_readability_in_reports():
    assert _evidence().evidence_id.startswith("ev_")
    assert _claim().claim_id.startswith("claim_")


# --- claim identity -----------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("Funding is charged hourly.", "funding is charged hourly"),
        ("Funding  is\ncharged hourly", "Funding is charged hourly."),
        ("  Funding is charged hourly.  ", "FUNDING IS CHARGED HOURLY"),
    ],
)
def test_presentational_differences_collapse_to_one_claim(a, b):
    assert _claim(a).claim_id == _claim(b).claim_id


def test_the_same_sentence_about_different_protocols_is_different_claims():
    """The collision this corpus was chosen to expose: 'funding' means one thing
    on a perps venue and another on a synthetic-dollar protocol. One claim id for
    both would pool their evidence and let each protocol's docs 'support' the
    other's mechanics."""
    hl = _claim("Funding is charged hourly.", protocols=("hyperliquid",))
    eth = _claim("Funding is charged hourly.", protocols=("ethena",))
    assert hl.claim_id != eth.claim_id


def test_protocol_order_does_not_change_claim_identity():
    a = _claim(protocols=("ethena", "hyperliquid"))
    b = _claim(protocols=("hyperliquid", "ethena"))
    assert a.claim_id == b.claim_id


def test_normalize_is_the_documented_identity_form():
    assert normalize_claim_text("  The  Fee is\n0.03%. ") == "the fee is 0.03%"


# --- stance lives on the edge ------------------------------------------


def test_one_evidence_node_can_support_one_claim_and_contradict_another():
    """Evidence has no intrinsic polarity — only a relationship to a claim does.
    A TVL drop supports 'activity declined' and contradicts 'the protocol grew'."""
    ev = _evidence(kind=EvidenceKind.ON_CHAIN_METRIC, payload={"tvl_change_7d": -0.42})
    declined = _claim("Activity declined.", links=(
        EvidenceLink(evidence_id=ev.evidence_id, stance=Stance.SUPPORTS),
    ))
    grew = _claim("The protocol grew.", links=(
        EvidenceLink(evidence_id=ev.evidence_id, stance=Stance.CONTRADICTS),
    ))
    assert declined.supporting()[0].evidence_id == ev.evidence_id
    assert grew.contradicting()[0].evidence_id == ev.evidence_id


def test_supporting_and_contradicting_partition_by_stance():
    links = (
        EvidenceLink(evidence_id="ev_a", stance=Stance.SUPPORTS),
        EvidenceLink(evidence_id="ev_b", stance=Stance.CONTRADICTS),
        EvidenceLink(evidence_id="ev_c", stance=Stance.NEUTRAL),
    )
    claim = _claim(links=links)
    assert [l.evidence_id for l in claim.supporting()] == ["ev_a"]
    assert [l.evidence_id for l in claim.contradicting()] == ["ev_b"]


def test_a_claim_with_only_neutral_evidence_is_unsupported():
    """Context that settles nothing is not support. This is the predicate that
    keeps a well-written but unevidenced finding away from a user."""
    claim = _claim(links=(EvidenceLink(evidence_id="ev_a", stance=Stance.NEUTRAL),))
    assert claim.is_unsupported()


def test_a_claim_with_no_links_is_unsupported():
    assert _claim().is_unsupported()


def test_linking_the_same_evidence_twice_is_rejected():
    """Otherwise one source votes twice, which is precisely what
    content-addressed ids exist to prevent."""
    with pytest.raises(ValidationError, match="more than once"):
        _claim(links=(
            EvidenceLink(evidence_id="ev_a", stance=Stance.SUPPORTS),
            EvidenceLink(evidence_id="ev_a", stance=Stance.SUPPORTS, relevance=0.5),
        ))


def test_relevance_is_bounded():
    with pytest.raises(ValidationError):
        EvidenceLink(evidence_id="ev_a", stance=Stance.SUPPORTS, relevance=1.5)


# --- timestamps ---------------------------------------------------------


def test_naive_timestamps_are_rejected():
    """A naive datetime silently means 'some timezone', which turns every age
    calculation into an error that never raises."""
    with pytest.raises(ValidationError, match="timezone-aware"):
        _evidence(observed_at=datetime(2026, 8, 20, 12, 0))


def test_naive_collected_at_is_rejected():
    with pytest.raises(ValidationError, match="timezone-aware"):
        Evidence(
            kind=EvidenceKind.DOCUMENT,
            source=_source(),
            agent=AgentName.RESEARCH,
            summary="x",
            collected_at=datetime(2026, 8, 20),
        )


def test_as_of_prefers_the_truth_time_over_the_lookup_time():
    """Aging an on-chain reading from when we queried the explorer would make
    stale data look fresh exactly when it is being re-read."""
    ev = _evidence(observed_at=T0 - timedelta(days=30), collected_at=T0)
    assert ev.as_of == T0 - timedelta(days=30)


def test_as_of_falls_back_to_collection_when_truth_time_is_unknown():
    assert _evidence(collected_at=T0).as_of == T0


# --- provenance ---------------------------------------------------------


def test_evidence_without_a_uri_is_rejected():
    with pytest.raises(ValidationError, match="unattributable"):
        SourceRef(tier=SourceTier.PRIMARY, uri="   ")


def test_empty_summary_is_rejected():
    with pytest.raises(ValidationError):
        _evidence(summary="   ")


def test_empty_claim_text_is_rejected():
    with pytest.raises(ValidationError):
        _claim("   ")


def test_models_are_frozen_so_evidence_cannot_be_edited_after_the_fact():
    """An investigation record that can be rewritten in place is not a record."""
    ev = _evidence()
    with pytest.raises(ValidationError):
        ev.summary = "something else"


def test_claims_start_unverified():
    """Nothing is verified until the Verification Agent says so."""
    assert _claim().verification is VerificationStatus.UNVERIFIED
