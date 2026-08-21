"""§14's evidence graph.

The graph has to earn its place over the flat record it is built from. It does so
in two ways, and both are tested here: it answers "why did you conclude that" by
walking, and it reveals when findings are not independent — which a list of
claims and citations structurally cannot show.
"""

from datetime import datetime, timezone

import pytest

from src.evidence.graph import (
    EdgeType,
    EvidenceGraph,
    Node,
    NodeType,
    build,
    explain,
    from_state,
    independence,
    independent_claim_groups,
    to_mermaid,
    to_state,
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


def _ev(uri, summary="an observation", protocol="ethena",
        kind=EvidenceKind.DOCUMENT, agent=AgentName.RESEARCH, **payload):
    return Evidence(
        kind=kind,
        source=SourceRef(tier=SourceTier.PRIMARY, uri=uri, protocol=protocol),
        agent=agent,
        summary=summary,
        payload=payload,
        observed_at=NOW,
        collected_at=NOW,
    )


def _claim(text, evs, stances=None, agent=AgentName.RESEARCH,
           verification=VerificationStatus.VERIFIED):
    stances = stances or [Stance.SUPPORTS] * len(evs)
    return Claim(
        text=text,
        agent=agent,
        protocols=("ethena",),
        created_at=NOW,
        verification=verification,
        links=tuple(
            EvidenceLink(evidence_id=e.evidence_id, stance=s)
            for e, s in zip(evs, stances)
        ),
    )


# --- construction --------------------------------------------------------


def test_an_empty_investigation_still_has_a_root():
    graph = build([], [], question="is it safe?")
    assert graph.of_type(NodeType.INVESTIGATION)[0].label == "is it safe?"


def test_claims_and_evidence_become_nodes():
    evs = [_ev("https://docs/a"), _ev("https://docs/b")]
    graph = build([_claim("A finding.", evs)], evs)
    assert len(graph.of_type(NodeType.CLAIM)) == 1
    assert len(graph.of_type(NodeType.EVIDENCE)) == 2


def test_supporting_and_contradicting_links_become_different_edges():
    """§14.2 gives them separate relationships, and collapsing them would make
    evidence against a claim look like evidence for it."""
    evs = [_ev("https://a"), _ev("https://b")]
    claim = _claim("x", evs, [Stance.SUPPORTS, Stance.CONTRADICTS])
    graph = build([claim], evs)

    assert len(graph.out_edges(claim.claim_id, EdgeType.SUPPORTED_BY)) == 1
    assert len(graph.out_edges(claim.claim_id, EdgeType.CONTRADICTED_BY)) == 1


def test_neutral_links_are_not_edges_of_either_kind():
    """Context attaches to a claim without arguing either way — the security
    agent's rumours. It must not appear as support."""
    ev = _ev("https://forum/x")
    claim = Claim(
        text="x", agent=AgentName.SECURITY, created_at=NOW,
        links=(EvidenceLink(evidence_id=ev.evidence_id, stance=Stance.NEUTRAL),),
    )
    graph = build([claim], [ev])
    assert graph.out_edges(claim.claim_id, EdgeType.SUPPORTED_BY) == []
    assert graph.out_edges(claim.claim_id, EdgeType.CONTRADICTED_BY) == []


def test_evidence_from_one_page_converges_on_one_source_node():
    """The whole reason the graph exists: this convergence is what turns 'six
    citations' into 'one page, cited six times'."""
    chunks = [_ev("https://docs/funding", f"chunk {i}") for i in range(4)]
    graph = build([_claim("x", chunks)], chunks)

    assert len(graph.of_type(NodeType.EVIDENCE)) == 4
    assert len(graph.of_type(NodeType.SOURCE)) == 1


def test_agents_are_recorded_for_both_claims_and_evidence():
    ev = _ev("https://a", agent=AgentName.BLOCKCHAIN)
    graph = build([_claim("x", [ev], agent=AgentName.RISK_ENGINE)], [ev])
    labels = {n.label for n in graph.of_type(NodeType.AGENT)}
    assert labels == {"blockchain_agent", "risk_engine"}


def test_claim_nodes_carry_their_verdict():
    ev = _ev("https://a")
    graph = build(
        [_claim("x", [ev], verification=VerificationStatus.PARTIALLY_VERIFIED)], [ev]
    )
    assert graph.of_type(NodeType.CLAIM)[0].attrs["verification"] == "partially_verified"


def test_a_link_to_absent_evidence_does_not_create_a_dangling_edge():
    """The flat record already reports that as a defect; the graph should not
    also crash on it."""
    claim = Claim(
        text="x", agent=AgentName.RESEARCH, created_at=NOW,
        links=(EvidenceLink(evidence_id="ev_ghost", stance=Stance.SUPPORTS),),
    )
    graph = build([claim], [])
    assert graph.out_edges(claim.claim_id, EdgeType.SUPPORTED_BY) == []


def test_adding_the_same_node_twice_keeps_one():
    graph = EvidenceGraph()
    graph.add_node(Node(id="a", type=NodeType.SOURCE, label="first"))
    graph.add_node(Node(id="a", type=NodeType.SOURCE, label="second"))
    assert len(graph) == 1
    assert graph.nodes["a"].label == "first"


def test_duplicate_edges_are_ignored():
    graph = EvidenceGraph()
    graph.add_node(Node(id="a", type=NodeType.CLAIM, label="a"))
    graph.add_node(Node(id="b", type=NodeType.EVIDENCE, label="b"))
    graph.add_edge("a", "b", EdgeType.SUPPORTED_BY)
    graph.add_edge("a", "b", EdgeType.SUPPORTED_BY)
    assert len(graph.edges) == 1


def test_risk_signals_attach_to_their_protocol_and_their_computation():
    signal_ev = _ev(
        "internal://risk-engine", "gas is critical",
        kind=EvidenceKind.STATISTICAL_SIGNAL, agent=AgentName.RISK_ENGINE,
        metric="gas_average",
    )
    graph = build(
        [], [signal_ev],
        risk_signals=[{"metric": "gas_average", "protocol": "ethena",
                       "severity": "critical", "z": 5.6}],
    )
    node = graph.of_type(NodeType.RISK_SIGNAL)[0]
    assert node.attrs["severity"] == "critical"
    assert graph.in_edges(node.id, EdgeType.HAS_RISK_SIGNAL)
    assert graph.out_edges(node.id, EdgeType.DERIVED_FROM)


# --- §14.3: why did you conclude that? ----------------------------------


def test_explaining_a_claim_reaches_its_evidence_and_its_sources():
    evs = [_ev("https://docs/a"), _ev("https://docs/b")]
    claim = _claim("A finding.", evs)
    graph = build([claim], evs)

    steps = explain(graph, claim.claim_id)
    types = [s.node.type for s in steps]

    assert types[0] is NodeType.CLAIM
    assert types.count(NodeType.EVIDENCE) == 2
    assert types.count(NodeType.SOURCE) == 2


def test_the_path_reads_outward_from_the_thing_asked_about():
    """Breadth-first: the claim, then what supports it, then where that came
    from — the order §14.3 draws and the order a reader needs."""
    evs = [_ev("https://docs/a")]
    claim = _claim("A finding.", evs)
    steps = explain(build([claim], evs), claim.claim_id)
    assert [s.depth for s in steps] == sorted(s.depth for s in steps)


def test_explanation_records_which_relationship_each_step_came_through():
    evs = [_ev("https://docs/a")]
    claim = _claim("x", evs)
    steps = explain(build([claim], evs), claim.claim_id)
    assert steps[1].via == "supported_by"
    assert steps[2].via == "from_source"


def test_explanation_does_not_wander_sideways_into_agents_or_protocols():
    """`generated_by` and `related_to` are real relationships that lead sideways.
    Following them would make "why do you believe this" answer with who said it
    rather than what it rests on."""
    evs = [_ev("https://docs/a")]
    claim = _claim("x", evs)
    steps = explain(build([claim], evs), claim.claim_id)
    kinds = {s.node.type for s in steps}
    assert NodeType.AGENT not in kinds
    assert NodeType.PROTOCOL not in kinds


def test_explaining_the_investigation_reaches_every_claim():
    evs = [_ev("https://a"), _ev("https://b")]
    claims = [_claim("One.", [evs[0]]), _claim("Two.", [evs[1]])]
    graph = build(claims, evs, question="q")
    steps = explain(graph, "investigation:current")
    assert sum(1 for s in steps if s.node.type is NodeType.CLAIM) == 2


def test_explaining_an_unknown_node_is_empty_not_an_error():
    assert explain(build([], []), "nope") == []


def test_traversal_terminates_on_a_cycle():
    graph = EvidenceGraph()
    for name in ("a", "b"):
        graph.add_node(Node(id=name, type=NodeType.CLAIM, label=name))
    graph.add_edge("a", "b", EdgeType.SUPPORTED_BY)
    graph.add_edge("b", "a", EdgeType.SUPPORTED_BY)
    assert len(explain(graph, "a")) == 2


def test_depth_is_bounded():
    evs = [_ev("https://a")]
    claim = _claim("x", evs)
    graph = build([claim], evs, question="q")
    assert all(s.depth <= 1 for s in explain(graph, claim.claim_id, max_depth=1))


# --- independence: the analysis a flat record cannot do -----------------


def test_claims_resting_on_the_same_page_are_flagged():
    """Two claims each carrying citations look like two corroborated findings.
    If every citation resolves to one page, they are one finding stated twice."""
    chunks = [_ev("https://docs/funding", f"chunk {i}") for i in range(4)]
    claims = [_claim("Funding is hourly.", chunks[:2]),
              _claim("Funding is capped.", chunks[2:])]
    graph = build(claims, chunks)

    shared = independence(graph)
    assert len(shared) == 1
    assert len(shared[0].claim_ids) == 2


def test_claims_on_genuinely_different_sources_are_not_flagged():
    a, b = _ev("https://docs/one"), _ev("https://docs/two")
    graph = build([_claim("One.", [a]), _claim("Two.", [b])], [a, b])
    assert independence(graph) == []


def test_independent_groups_count_lines_of_evidence_not_claims():
    """The honest number for a report: how many separate things were established,
    not how many sentences were written."""
    shared_page = [_ev("https://docs/x", f"c{i}") for i in range(2)]
    separate = _ev("https://docs/y")
    claims = [
        _claim("One.", [shared_page[0]]),
        _claim("Two.", [shared_page[1]]),
        _claim("Three.", [separate]),
    ]
    groups = independent_claim_groups(build(claims, [*shared_page, separate]))

    assert len(claims) == 3
    assert len(groups) == 2  # two claims collapse into one line of evidence


def test_a_chain_of_shared_sources_merges_into_one_group():
    """A shares a source with B, B with C: all three are one line of evidence
    even though A and C share nothing directly."""
    p, q = _ev("https://docs/p"), _ev("https://docs/q")
    p2, q2 = _ev("https://docs/p", "other chunk"), _ev("https://docs/q", "other chunk")
    claims = [_claim("A.", [p]), _claim("B.", [p2, q]), _claim("C.", [q2])]
    groups = independent_claim_groups(build(claims, [p, q, p2, q2]))
    assert len(groups) == 1


def test_contradicting_evidence_does_not_create_shared_support():
    """Only support counts toward corroboration."""
    ev = _ev("https://docs/x")
    a = _claim("A.", [ev])
    b = _claim("B.", [ev], [Stance.CONTRADICTS])
    assert independence(build([a, b], [ev])) == []


# --- visualisation and state --------------------------------------------


def test_mermaid_renders_nodes_and_edges():
    evs = [_ev("https://docs/a")]
    diagram = to_mermaid(build([_claim("A finding.", evs)], evs))
    assert diagram.startswith("flowchart LR")
    assert "supported_by" in diagram
    assert "A finding." in diagram


def test_mermaid_labels_are_escaped():
    """Quotes and brackets break Mermaid's parser."""
    ev = _ev("https://a", 'a "quoted" [bracketed] label')
    diagram = to_mermaid(build([_claim("x", [ev])], [ev]))
    body = diagram.split("\n", 1)[1]
    assert '"quoted"' not in body
    assert "[bracketed]" not in body


def test_mermaid_can_be_filtered_to_a_subset():
    evs = [_ev("https://docs/a")]
    diagram = to_mermaid(
        build([_claim("x", evs)], evs), include={NodeType.CLAIM, NodeType.EVIDENCE}
    )
    assert "agent" not in diagram


def test_the_graph_round_trips_through_plain_state():
    """State channels are checkpointed through msgpack; see AgentState."""
    evs = [_ev("https://docs/a"), _ev("https://docs/b")]
    original = build([_claim("A finding.", evs)], evs, protocols=("ethena",))

    payload = to_state(original)
    assert all(isinstance(n, dict) for n in payload["nodes"])

    restored = from_state(payload)
    assert len(restored) == len(original)
    assert len(restored.edges) == len(original.edges)


def test_a_restored_graph_still_explains():
    evs = [_ev("https://docs/a")]
    claim = _claim("A finding.", evs)
    restored = from_state(to_state(build([claim], evs)))
    assert len(explain(restored, claim.claim_id)) == 3
