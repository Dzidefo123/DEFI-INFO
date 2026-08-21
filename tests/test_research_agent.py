"""§8's Research Agent.

The two model-dependent steps (decompose, synthesize) are injected, so the whole
pipeline runs here without an API key. Everything between them is deterministic
and is where the safety properties live: a model proposes claims, but code
decides which of them are supportable.
"""

import pytest
from langchain_core.documents import Document

from src.agents.research import (
    MAX_CONTEXT_CHUNKS,
    MAX_SUB_QUERIES,
    ProposedClaim,
    SubQuery,
    evidence_from_document,
    gather_evidence,
    investigate,
    link_claims,
)
from src.evidence.models import AgentName, SourceTier


def _doc(source, text="chunk body", protocol="ethena", **meta):
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "protocol": protocol,
            "doc_id": f"{protocol}:{source}#0",
            "title": meta.pop("title", "Ethena Docs"),
            "heading": meta.pop("heading", "How USDe Works"),
            **meta,
        },
    )


def _fixed_retriever(docs_by_query=None, default=()):
    """A retriever that returns canned documents, recording what it was asked."""
    calls = []

    def retrieve(query, protocols):
        calls.append((query, tuple(protocols or ())))
        return list((docs_by_query or {}).get(query, default))

    retrieve.calls = calls
    return retrieve


# --- evidence extraction (deterministic) --------------------------------


def test_a_chunk_becomes_evidence_with_full_provenance():
    ev = evidence_from_document(_doc("https://docs.ethena.fi/how-usde-works"))
    assert ev.kind.value == "document"
    assert ev.source.uri == "https://docs.ethena.fi/how-usde-works"
    assert ev.source.protocol == "ethena"
    assert ev.source.locator == "ethena:https://docs.ethena.fi/how-usde-works#0"
    assert ev.agent is AgentName.RESEARCH


def test_documentation_evidence_is_primary_by_rule_not_by_judgement():
    """The corpus IS the whitelist: a chunk exists only because `protocols.py`
    authorised crawling the page. A protocol's own docs are the most
    authoritative statement of what that protocol does, and that is decided by
    the registry, not by a model's impression of the page."""
    assert evidence_from_document(_doc("https://docs.ethena.fi/x")).source.tier is (
        SourceTier.PRIMARY
    )


def test_the_summary_is_the_breadcrumb_so_a_report_can_name_the_section():
    ev = evidence_from_document(_doc("https://x/a", title="Ethena", heading="Risks"))
    assert ev.summary == "Ethena > Risks"


def test_the_breadcrumb_does_not_stutter_when_the_h1_is_the_page_title():
    """`chunk.py` builds `heading` from the h1/h2/h3 chain, and on these sites the
    h1 usually IS the page title. The naive join produced 'How USDe Works > How
    USDe Works > How USDe Maintains its Peg' on the real corpus — a stuttering
    label reads as a bug in the evidence, not in the formatting."""
    ev = evidence_from_document(
        _doc(
            "https://x/a",
            title="How USDe Works",
            heading="How USDe Works > How USDe Maintains its Peg",
        )
    )
    assert ev.summary == "How USDe Works > How USDe Maintains its Peg"


def test_a_deeper_section_path_is_preserved_intact():
    ev = evidence_from_document(
        _doc("https://x/a", title="Ethena", heading="Risks > Funding Risk")
    )
    assert ev.summary == "Ethena > Risks > Funding Risk"


def test_a_chunk_with_no_breadcrumb_falls_back_to_its_url():
    ev = evidence_from_document(_doc("https://x/a", title="", heading=""))
    assert ev.summary == "https://x/a"


def test_the_rerank_score_is_carried_into_the_payload():
    ev = evidence_from_document(_doc("https://x/a", rerank_score=4.2))
    assert ev.payload["rerank_score"] == 4.2


def test_the_chunk_text_is_preserved_for_synthesis():
    ev = evidence_from_document(_doc("https://x/a", text="funding is charged hourly"))
    assert ev.payload["text"] == "funding is charged hourly"


# --- pooling across sub-queries -----------------------------------------


def test_every_sub_query_is_searched():
    retriever = _fixed_retriever()
    subs = [SubQuery(text="q1", angle="documentation"), SubQuery(text="q2", angle="governance")]
    gather_evidence(subs, ["ethena"], retriever)
    assert [c[0] for c in retriever.calls] == ["q1", "q2"]


def test_the_protocol_filter_reaches_every_sub_query():
    """Decomposition must not be a way to lose the protocol scope — the filter is
    what guarantees 0% cross-protocol leakage."""
    retriever = _fixed_retriever()
    gather_evidence([SubQuery(text="q", angle="documentation")], ["ethena"], retriever)
    assert retriever.calls[0][1] == ("ethena",)


def test_a_chunk_found_by_several_angles_is_one_piece_of_evidence():
    """Content-addressed ids make this automatic, and it matters beyond
    tidiness: confidence counts DISTINCT sources, so without deduplication,
    asking the same question four ways would look like four-fold corroboration
    of whatever all four ways happened to find."""
    shared = _doc("https://docs.ethena.fi/shared")
    retriever = _fixed_retriever(
        {"q1": [shared, _doc("https://a")], "q2": [shared, _doc("https://b")]}
    )
    pooled = gather_evidence(
        [SubQuery(text="q1", angle="documentation"), SubQuery(text="q2", angle="historical")],
        ["ethena"],
        retriever,
    )
    assert len(pooled) == 3
    assert len({e.evidence_id for e in pooled}) == 3


def test_pooling_nothing_yields_nothing():
    retriever = _fixed_retriever()
    assert gather_evidence([SubQuery(text="q", angle="documentation")], [], retriever) == []


# --- citation checking: the load-bearing sanitisation -------------------


def _evidence(n):
    return [evidence_from_document(_doc(f"https://docs.ethena.fi/{i}")) for i in range(n)]


def test_a_claim_links_the_excerpts_it_cited():
    evidence = _evidence(3)
    claims, dropped = link_claims(
        [ProposedClaim(text="USDe is delta-neutral.", excerpts=[1, 3])], evidence
    )
    assert dropped == []
    linked = {l.evidence_id for l in claims[0].links}
    assert linked == {evidence[0].evidence_id, evidence[2].evidence_id}


def test_an_out_of_range_citation_is_dropped():
    """The model cites by number; a number outside the range it was handed refers
    to nothing. Same sanitisation the router's protocol keys already get."""
    evidence = _evidence(2)
    claims, _ = link_claims([ProposedClaim(text="A claim.", excerpts=[1, 99])], evidence)
    assert len(claims[0].links) == 1


def test_a_zero_or_negative_citation_is_dropped():
    """Excerpt numbers are 1-based; 0 would silently index the last item."""
    evidence = _evidence(3)
    claims, _ = link_claims([ProposedClaim(text="A claim.", excerpts=[0, -1, 2])], evidence)
    assert len(claims[0].links) == 1
    assert claims[0].links[0].evidence_id == evidence[1].evidence_id


def test_a_repeated_citation_links_once():
    """`Claim` rejects duplicate links outright, so without this a model citing
    the same excerpt twice would take down the whole investigation rather than
    just that claim."""
    evidence = _evidence(2)
    claims, dropped = link_claims(
        [ProposedClaim(text="A claim.", excerpts=[1, 1, 1])], evidence
    )
    assert dropped == []
    assert len(claims[0].links) == 1


def test_a_claim_citing_nothing_is_discarded_not_kept_weakly():
    """It would score zero anyway — but a zero-confidence claim still sits in the
    record inviting someone to read it as a weak finding. It is not a weak
    finding; nothing supports it."""
    claims, dropped = link_claims(
        [ProposedClaim(text="Something unsupported.", excerpts=[])], _evidence(2)
    )
    assert claims == []
    assert len(dropped) == 1
    assert "unsupported claim" in dropped[0]


def test_a_claim_whose_every_citation_is_invalid_is_discarded():
    claims, dropped = link_claims(
        [ProposedClaim(text="Fabricated.", excerpts=[42, 43])], _evidence(2)
    )
    assert claims == []
    assert "Fabricated." in dropped[0]


def test_discarding_one_claim_does_not_discard_the_others():
    claims, dropped = link_claims(
        [
            ProposedClaim(text="Supported.", excerpts=[1]),
            ProposedClaim(text="Unsupported.", excerpts=[]),
        ],
        _evidence(2),
    )
    assert [c.text for c in claims] == ["Supported."]
    assert len(dropped) == 1


def test_claims_carry_the_protocol_scope():
    """Claim identity includes protocols — 'funding is hourly' about Ethena is a
    different claim than about Hyperliquid."""
    claims, _ = link_claims(
        [ProposedClaim(text="A claim.", excerpts=[1])], _evidence(1), ["ethena"]
    )
    assert claims[0].protocols == ("ethena",)


def test_every_link_from_research_supports_rather_than_contradicts():
    """This agent reads documentation; documentation does not argue with the
    question. Contradiction detection is the Verification Agent's job."""
    claims, _ = link_claims([ProposedClaim(text="A claim.", excerpts=[1])], _evidence(1))
    assert claims[0].supporting() and not claims[0].contradicting()


# --- the full agent (models injected) -----------------------------------


def _run(docs, proposed, subs=None, protocols=("ethena",)):
    subs = subs or [SubQuery(text="q1", angle="documentation")]
    return investigate(
        question="How does USDe stay pegged?",
        protocols=protocols,
        decomposer=lambda q, p: list(subs),
        synthesizer=lambda q, d: list(proposed),
        retriever=_fixed_retriever(default=docs),
    )


def test_the_agent_produces_linked_claims_and_evidence():
    out = _run(
        [_doc("https://docs.ethena.fi/peg")],
        [ProposedClaim(text="USDe holds its peg by arbitrage.", excerpts=[1])],
    )
    assert len(out.claims) == 1
    assert len(out.evidence) == 1
    assert out.claims[0].links[0].evidence_id == out.evidence[0].evidence_id
    assert out.limitations == ()


def test_retrieval_finding_nothing_is_reported_as_a_search_failure():
    """Not as an answer. 'The corpus was not searched successfully' and 'the
    answer is no' are opposite conclusions."""
    out = _run([], [])
    assert out.claims == ()
    assert "not that the answer is no" in out.limitations[0]


def test_a_model_that_proposes_nothing_yields_no_claims_and_says_so():
    out = _run([_doc("https://docs.ethena.fi/peg")], [])
    assert out.claims == ()
    assert any("does not support a finding" in l for l in out.limitations)


def test_an_empty_decomposition_falls_back_to_the_original_question():
    """Never silently degrade to doing no research at all."""
    out = investigate(
        question="How does USDe stay pegged?",
        protocols=["ethena"],
        decomposer=lambda q, p: [],
        synthesizer=lambda q, d: [ProposedClaim(text="A claim.", excerpts=[1])],
        retriever=_fixed_retriever(default=[_doc("https://docs.ethena.fi/peg")]),
    )
    assert out.sub_queries[0].text == "How does USDe stay pegged?"
    assert any("only one angle was searched" in l for l in out.limitations)


def test_synthesis_sees_the_best_ranked_excerpts_first():
    """Ordered by the cross-encoder score retrieval already computed, so the
    model reads the best excerpts rather than whichever sub-query ran first."""
    seen = {}

    def capture(question, docs):
        seen["sources"] = [d.metadata["source"] for d in docs]
        return []

    investigate(
        question="q",
        protocols=["ethena"],
        decomposer=lambda q, p: [SubQuery(text="q1", angle="documentation")],
        synthesizer=capture,
        retriever=_fixed_retriever(
            default=[
                _doc("https://low", rerank_score=-2.0),
                _doc("https://high", rerank_score=8.0),
                _doc("https://mid", rerank_score=1.0),
            ]
        ),
    )
    assert seen["sources"] == ["https://high", "https://mid", "https://low"]


def test_the_context_handed_to_synthesis_is_bounded():
    """Bounds the one call whose cost scales with retrieval breadth."""
    seen = {}
    many = [_doc(f"https://docs.ethena.fi/{i}", rerank_score=float(i)) for i in range(40)]

    def capture(question, docs):
        seen["n"] = len(docs)
        return []

    investigate(
        question="q",
        protocols=["ethena"],
        decomposer=lambda q, p: [SubQuery(text="q1", angle="documentation")],
        synthesizer=capture,
        retriever=_fixed_retriever(default=many),
    )
    assert seen["n"] == MAX_CONTEXT_CHUNKS


def test_the_record_keeps_all_retrieved_evidence_not_just_what_synthesis_saw():
    """The report should show everything retrieval found, even the excerpts that
    did not fit in the synthesis window."""
    many = [_doc(f"https://docs.ethena.fi/{i}", rerank_score=float(i)) for i in range(20)]
    out = _run(many, [])
    assert len(out.evidence) == 20 > MAX_CONTEXT_CHUNKS


def test_decomposition_is_capped():
    assert MAX_SUB_QUERIES == 4  # one per §8.2 angle


def test_a_fabricating_model_cannot_manufacture_a_finding():
    """The end-to-end version of the citation check: a model that invents claims
    out of excerpts it never saw produces an investigation with no findings and
    an explicit limitation, not a confident wrong answer."""
    out = _run(
        [_doc("https://docs.ethena.fi/peg")],
        [
            ProposedClaim(text="Ethena is insolvent.", excerpts=[7]),
            ProposedClaim(text="A backdoor exists.", excerpts=[]),
        ],
    )
    assert out.claims == ()
    assert len(out.limitations) == 3  # two discards + "no claim survived"
