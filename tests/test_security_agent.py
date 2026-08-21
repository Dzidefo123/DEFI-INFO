"""§10's Security Agent and its incident registry.

The property under test throughout: the four classifications stay apart, and
only established ones can support a conclusion. Blurring them has asymmetric
cost — reading a rumour as a confirmed incident defames a protocol and panics a
user; reading a confirmed incident as a rumour leaves someone exposed. There is
no safe direction, so the separation lives in the schema and is asserted here.
"""

import json
from datetime import datetime, timezone

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from src.agents.security import (
    SECURITY_ANGLES,
    claims_from_records,
    evidence_from_document,
    evidence_from_record,
    investigate,
)
from src.evidence.models import EvidenceKind, SourceTier, Stance
from src.security.incidents import (
    REGISTRY,
    Classification,
    IncidentRecord,
    RegistryError,
    Status,
    counts_by_classification,
    for_protocols,
    load,
)

T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)

ESTABLISHED = [Classification.CONFIRMED_INCIDENT, Classification.KNOWN_VULNERABILITY]
UNESTABLISHED = [Classification.SUSPICIOUS_SIGNAL, Classification.UNVERIFIED_CLAIM]


def _record(classification=Classification.CONFIRMED_INCIDENT, rid="inc-1", **kw):
    return IncidentRecord(
        id=rid,
        protocol=kw.pop("protocol", "ethena"),
        classification=classification,
        title=kw.pop("title", "Oracle deviation"),
        summary=kw.pop("summary", "A price feed deviated for 20 minutes."),
        source_uri=kw.pop("source_uri", "https://example.org/report"),
        source_tier=kw.pop("source_tier", SourceTier.OFFICIAL),
        **kw,
    )


def _doc(source="https://docs.ethena.fi/risks", protocol="ethena"):
    return Document(
        page_content="Custodial risk is managed via off-exchange settlement.",
        metadata={
            "source": source,
            "protocol": protocol,
            "doc_id": f"{protocol}:{source}#0",
            "title": "Ethena",
            "heading": "Risks > Custodial Risk",
        },
    )


def _write(tmp_path, records):
    path = tmp_path / "registry.jsonl"
    path.write_text(
        "\n".join(json.dumps(json.loads(r.model_dump_json())) for r in records),
        encoding="utf-8",
    )
    return path


# --- the shipped registry ------------------------------------------------


def test_the_shipped_registry_is_empty_and_loads_cleanly():
    """It ships empty on purpose. An entry labelled CONFIRMED_INCIDENT is trusted
    by everything downstream, so filling it from recollection would put
    defamatory content behind that label."""
    assert load() == ()


def test_the_shipped_registry_documents_how_to_add_an_entry():
    text = REGISTRY.read_text(encoding="utf-8")
    for term in ("confirmed_incident", "source_tier", "INDEPENDENT", "citation"):
        assert term in text


# --- the four categories -------------------------------------------------


def test_all_four_categories_exist_and_are_distinct():
    assert len(set(Classification)) == 4


@pytest.mark.parametrize("classification", ESTABLISHED)
def test_established_findings_support_a_conclusion(classification):
    record = _record(classification)
    assert record.is_established
    assert record.stance is Stance.SUPPORTS


@pytest.mark.parametrize("classification", UNESTABLISHED)
def test_unestablished_findings_are_neutral_and_never_support(classification):
    """§13.2's worked example, as a rule: a forum post attributing an outflow to
    an exploit does not make the exploit more likely — it makes the rumour
    visible. Attaching it as support would let 'something looks odd' accumulate
    into 'something is wrong' by repetition alone."""
    record = _record(classification)
    assert not record.is_established
    assert record.stance is Stance.NEUTRAL


def test_a_confirmed_incident_is_an_incident_report_not_an_advisory():
    assert _record(Classification.CONFIRMED_INCIDENT).evidence_kind is (
        EvidenceKind.INCIDENT_REPORT
    )


def test_a_known_vulnerability_is_an_advisory():
    assert _record(Classification.KNOWN_VULNERABILITY).evidence_kind is (
        EvidenceKind.SECURITY_ADVISORY
    )


@pytest.mark.parametrize("classification", list(Classification))
def test_classification_and_source_tier_are_independent(classification):
    """Collapsing 'who said it' into 'how established is it' is the specific
    mistake §10 exists to prevent. A confirmed incident reported by a community
    researcher is confirmed AND community-tier."""
    for tier in SourceTier:
        record = _record(classification, source_tier=tier)
        assert record.classification is classification
        assert record.source_tier is tier
        assert record.is_established == (classification in ESTABLISHED)


def test_counts_keep_the_categories_separate():
    records = (
        _record(Classification.CONFIRMED_INCIDENT, "a"),
        _record(Classification.UNVERIFIED_CLAIM, "b"),
        _record(Classification.UNVERIFIED_CLAIM, "c"),
    )
    counts = counts_by_classification(records)
    assert counts[Classification.CONFIRMED_INCIDENT] == 1
    assert counts[Classification.UNVERIFIED_CLAIM] == 2
    assert counts[Classification.SUSPICIOUS_SIGNAL] == 0


# --- registry integrity --------------------------------------------------


def test_a_finding_about_a_non_whitelisted_protocol_is_refused():
    """It could not be surfaced, scoped or corrected, and would be filed under a
    name nothing else in the system recognises."""
    with pytest.raises(ValidationError, match="non-whitelisted"):
        _record(protocol="aave")


def test_a_finding_without_a_source_is_refused():
    with pytest.raises(ValidationError, match="rumour"):
        _record(source_uri="   ")


def test_a_malformed_line_fails_the_whole_load(tmp_path):
    """Deliberately not lenient. Skipping a bad line means a typo silently
    removes a confirmed incident from every future investigation, and that
    silence is indistinguishable from a protocol having no incidents."""
    path = tmp_path / "registry.jsonl"
    path.write_text('{"id": "broken"}\n', encoding="utf-8")
    with pytest.raises(RegistryError, match="line 1"):
        load(path)


def test_duplicate_ids_are_refused(tmp_path):
    path = _write(tmp_path, [_record(rid="dup"), _record(rid="dup")])
    with pytest.raises(RegistryError, match="duplicate id"):
        load(path)


def test_comments_and_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "registry.jsonl"
    path.write_text(
        "// a comment\n\n" + _record().model_dump_json() + "\n", encoding="utf-8"
    )
    assert len(load(path)) == 1


def test_a_missing_registry_is_empty_not_an_error(tmp_path):
    assert load(tmp_path / "nope.jsonl") == ()


def test_findings_are_filtered_by_protocol(tmp_path):
    path = _write(
        tmp_path,
        [_record(rid="e", protocol="ethena"), _record(rid="h", protocol="hyperliquid")],
    )
    assert [r.id for r in for_protocols(["ethena"], path)] == ["e"]


# --- evidence from records -----------------------------------------------


def test_registry_evidence_carries_its_classification_in_the_payload():
    """So nothing downstream has to parse prose to discover whether a finding was
    confirmed or merely alleged."""
    ev = evidence_from_record(_record(Classification.UNVERIFIED_CLAIM))
    assert ev.payload["classification"] == "unverified_claim"
    assert ev.summary.startswith("[unverified_claim]")


def test_registry_evidence_keeps_the_records_own_source_tier():
    ev = evidence_from_record(_record(source_tier=SourceTier.COMMUNITY))
    assert ev.source.tier is SourceTier.COMMUNITY


def test_evidence_is_dated_by_when_the_event_happened():
    ev = evidence_from_record(_record(occurred_at=T0))
    assert ev.observed_at == T0


# --- claims from records -------------------------------------------------


def _claims_for(records):
    evidence = [evidence_from_record(r) for r in records]
    return claims_from_records(records, evidence), evidence


def test_an_established_finding_becomes_a_claim():
    claims, _ = _claims_for([_record(Classification.CONFIRMED_INCIDENT)])
    assert len(claims) == 1
    assert "Ethena" in claims[0].text
    assert "confirmed incident" in claims[0].text


@pytest.mark.parametrize("classification", UNESTABLISHED)
def test_an_unestablished_finding_produces_no_claim_of_its_own(classification):
    claims, _ = _claims_for([_record(classification)])
    assert claims == []


def test_a_rumour_attaches_as_context_without_supporting_anything():
    """It appears in the record so a report can show it, and it can never raise
    the confidence of the finding it sits beside."""
    records = [
        _record(Classification.CONFIRMED_INCIDENT, "real"),
        _record(Classification.UNVERIFIED_CLAIM, "rumour"),
    ]
    claims, evidence = _claims_for(records)
    claim = claims[0]

    rumour_id = next(e.evidence_id for e in evidence if e.source.locator == "rumour")
    linked = {l.evidence_id: l.stance for l in claim.links}

    assert linked[rumour_id] is Stance.NEUTRAL
    assert len(claim.supporting()) == 1


def test_a_rumour_cannot_raise_a_confidence_score():
    """The arithmetic version of the same rule."""
    from src.evidence.confidence import assess

    real = [_record(Classification.CONFIRMED_INCIDENT, "real")]
    with_rumours = real + [
        _record(Classification.UNVERIFIED_CLAIM, f"r{i}") for i in range(4)
    ]

    alone_claims, alone_ev = _claims_for(real)
    noisy_claims, noisy_ev = _claims_for(with_rumours)

    assert assess(noisy_claims[0], noisy_ev).score == assess(alone_claims[0], alone_ev).score


# --- documentation stream ------------------------------------------------


def test_a_security_docs_chunk_is_a_document_not_an_advisory():
    """A protocol's risk page is the protocol talking about itself. Calling it an
    advisory would give a well-written disclosure the weight of a published
    vulnerability notice."""
    ev = evidence_from_document(_doc())
    assert ev.kind is EvidenceKind.DOCUMENT
    assert ev.source.tier is SourceTier.PRIMARY
    assert ev.payload["security_documentation"] is True


def _fake_retriever(docs):
    calls = []

    def retrieve(query, protocols):
        calls.append(query)
        return list(docs)

    retrieve.calls = calls
    return retrieve


def test_every_security_angle_is_searched():
    retriever = _fake_retriever([])
    investigate(["ethena"], retriever=retriever, registry_path="nope.jsonl")
    assert retriever.calls == list(SECURITY_ANGLES)


def test_documented_risk_is_reported_as_documentation_not_as_a_finding():
    """Treating 'Ethena documents custodial risk' as a security finding would make
    every well-documented protocol a suspect and every undocumented one look
    clean — precisely inverted."""
    out = investigate(
        ["ethena"], retriever=_fake_retriever([_doc()]), registry_path="nope.jsonl"
    )
    text = out.claims[0].text
    assert "not evidence of an incident" in text
    assert "unpatched vulnerability" in text


def test_the_same_page_found_by_several_angles_is_one_piece_of_evidence():
    out = investigate(
        ["ethena"], retriever=_fake_retriever([_doc()]), registry_path="nope.jsonl"
    )
    assert len(out.evidence) == 1


# --- the agent's silences ------------------------------------------------


def test_an_empty_registry_says_so_explicitly():
    """The most dangerous silence in the system: an empty security section reads
    exactly like a clean one."""
    out = investigate(
        ["ethena"], retriever=_fake_retriever([_doc()]), registry_path="nope.jsonl"
    )
    note = next(l for l in out.limitations if "no security findings" in l)
    assert "NOT that nothing has happened" in note
    assert "No external threat-intelligence feed is connected" in note


def test_no_protocol_means_nothing_was_searched():
    out = investigate([])
    assert out.evidence == () and out.claims == ()
    assert "nothing was searched" in out.limitations[0]


def test_retrieving_no_documentation_is_reported():
    out = investigate(
        ["ethena"], retriever=_fake_retriever([]), registry_path="nope.jsonl"
    )
    assert any("could not\nbe reviewed" in l or "could not be reviewed" in l
               for l in out.limitations)


def test_findings_on_file_are_summarised_per_category(tmp_path):
    path = _write(
        tmp_path,
        [
            _record(Classification.CONFIRMED_INCIDENT, "a"),
            _record(Classification.UNVERIFIED_CLAIM, "b"),
        ],
    )
    out = investigate(
        ["ethena"], retriever=_fake_retriever([]), registry_path=path
    )
    note = next(l for l in out.limitations if "finding(s) on file" in l)
    assert "1 confirmed incident" in note
    assert "1 unverified claim" in note
    assert "do not support any conclusion" in note
