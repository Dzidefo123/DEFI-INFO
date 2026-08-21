"""§14. The evidence graph — what makes a conclusion traceable.

The flat investigation record already holds every claim and every piece of
evidence. So the graph has to earn its place, and it does so in two ways that a
list genuinely cannot.

**It answers "why did you conclude that?" by walking.** §14.3's question is a
traversal: a conclusion leads to the claims under it, each claim to the evidence
supporting it, each piece of evidence to the source it came from. `explain()`
returns that path, and because every id is content-addressed the path is stable
across runs.

**It shows when findings are not independent.** This is the part a list hides.
Two claims each carrying three citations look like two corroborated findings —
until you notice all six citations resolve to the same documentation page. In the
flat record they are six links; in the graph they converge on one source node,
and `independence()` reports it. That distinction changes what a report is
entitled to say, and nothing else in the system can see it.

Built deterministically from the record, per investigation, and thrown away with
it. It is a lens over state, not a second store that can drift out of agreement
with the first. Persisting it across investigations — §14's "memory and reasoning
structure" — is a later step and a different problem, because it introduces
exactly the staleness the rest of this architecture is arranged to avoid.

No graph library: the graphs are tens to low hundreds of nodes and the traversals
are specific ones, not general algorithms. Adjacency dicts are the whole
implementation.
"""

from __future__ import annotations

from collections import defaultdict, deque
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.evidence.models import Claim, Evidence, Stance


class NodeType(str, Enum):
    """The §14.1 entities this system actually produces today.

    Deliberately shorter than the spec's list. A node type nothing constructs is
    a promise the graph cannot keep, and an empty category in a provenance view
    reads as "we checked and found none" rather than "this was never populated" —
    the same failure the rest of the system is arranged against.
    """

    CLAIM = "claim"
    EVIDENCE = "evidence"
    SOURCE = "source"          # a document, endpoint or feed; several evidence nodes may share one
    PROTOCOL = "protocol"
    AGENT = "agent"
    RISK_SIGNAL = "risk_signal"
    INVESTIGATION = "investigation"


class EdgeType(str, Enum):
    """§14.2's relationships, plus the ones provenance needs."""

    SUPPORTED_BY = "supported_by"        # claim   -> evidence
    CONTRADICTED_BY = "contradicted_by"  # claim   -> evidence
    GENERATED_BY = "generated_by"        # claim   -> agent
    RELATED_TO = "related_to"            # claim   -> protocol
    FROM_SOURCE = "from_source"          # evidence-> source
    COLLECTED_BY = "collected_by"        # evidence-> agent
    DOCUMENTS = "documents"              # source  -> protocol
    HAS_RISK_SIGNAL = "has_risk_signal"  # protocol-> risk signal
    DERIVED_FROM = "derived_from"        # risk signal -> evidence
    INVESTIGATED = "investigated"        # investigation -> claim


class Node(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: NodeType
    label: str
    attrs: dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    type: EdgeType

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.type.value)


class EvidenceGraph(BaseModel):
    """Nodes, edges, and the traversals a report needs."""

    model_config = ConfigDict(frozen=False)

    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)

    # --- construction ---------------------------------------------------

    def add_node(self, node: Node) -> Node:
        """Insert, or return the existing node with this id.

        First write wins. Ids are content-addressed, so a second node with the
        same id describes the same thing — and re-adding it must not create a
        duplicate, or the convergence that `independence` depends on would never
        happen.
        """
        return self.nodes.setdefault(node.id, node)

    def add_edge(self, source: str, target: str, type: EdgeType) -> None:
        """Connect two nodes. Silently ignores duplicates and dangling ends.

        Dangling is ignored rather than raised because a claim may legitimately
        link evidence that is not in this investigation's pool; the flat record
        already reports that as a defect, and the graph should not also crash on
        it.
        """
        if source not in self.nodes or target not in self.nodes:
            return
        edge = Edge(source=source, target=target, type=type)
        if edge.key not in {e.key for e in self.edges}:
            self.edges.append(edge)

    # --- queries --------------------------------------------------------

    def out_edges(self, node_id: str, type: EdgeType | None = None) -> list[Edge]:
        return [
            e for e in self.edges
            if e.source == node_id and (type is None or e.type is type)
        ]

    def in_edges(self, node_id: str, type: EdgeType | None = None) -> list[Edge]:
        return [
            e for e in self.edges
            if e.target == node_id and (type is None or e.type is type)
        ]

    def neighbours(self, node_id: str, type: EdgeType | None = None) -> list[Node]:
        return [self.nodes[e.target] for e in self.out_edges(node_id, type)]

    def of_type(self, type: NodeType) -> list[Node]:
        return [n for n in self.nodes.values() if n.type is type]

    def __len__(self) -> int:
        return len(self.nodes)


# --- construction from an investigation record --------------------------


def _agent_id(name: str) -> str:
    return f"agent:{name}"


def _protocol_id(key: str) -> str:
    return f"protocol:{key}"


def _source_id(uri: str) -> str:
    return f"source:{uri}"


def build(
    claims: list[Claim],
    evidence: list[Evidence],
    risk_signals: list[dict] | None = None,
    protocols: tuple[str, ...] = (),
    question: str = "",
) -> EvidenceGraph:
    """Assemble the graph from one investigation's record. Pure and free."""
    graph = EvidenceGraph()
    signals = risk_signals or []

    root = Node(
        id="investigation:current",
        type=NodeType.INVESTIGATION,
        label=question or "investigation",
    )
    graph.add_node(root)

    for key in protocols:
        graph.add_node(Node(id=_protocol_id(key), type=NodeType.PROTOCOL, label=key))

    # Evidence first, so claims can attach to nodes that already exist.
    for item in evidence:
        graph.add_node(
            Node(
                id=item.evidence_id,
                type=NodeType.EVIDENCE,
                label=item.summary,
                attrs={
                    "kind": item.kind.value,
                    "tier": item.source.tier.value,
                    "observed_at": item.as_of.isoformat(),
                },
            )
        )
        # Several evidence nodes converging on one source is the whole point:
        # it is what turns "six citations" into "one page, cited six times".
        source = graph.add_node(
            Node(
                id=_source_id(item.source.uri),
                type=NodeType.SOURCE,
                label=item.source.title or item.source.uri,
                attrs={"uri": item.source.uri, "tier": item.source.tier.value},
            )
        )
        graph.add_edge(item.evidence_id, source.id, EdgeType.FROM_SOURCE)

        agent = graph.add_node(
            Node(id=_agent_id(item.agent.value), type=NodeType.AGENT, label=item.agent.value)
        )
        graph.add_edge(item.evidence_id, agent.id, EdgeType.COLLECTED_BY)

        if item.source.protocol:
            proto = graph.add_node(
                Node(
                    id=_protocol_id(item.source.protocol),
                    type=NodeType.PROTOCOL,
                    label=item.source.protocol,
                )
            )
            graph.add_edge(source.id, proto.id, EdgeType.DOCUMENTS)

    for claim in claims:
        graph.add_node(
            Node(
                id=claim.claim_id,
                type=NodeType.CLAIM,
                label=claim.text,
                attrs={"verification": claim.verification.value},
            )
        )
        graph.add_edge(root.id, claim.claim_id, EdgeType.INVESTIGATED)

        agent = graph.add_node(
            Node(id=_agent_id(claim.agent.value), type=NodeType.AGENT, label=claim.agent.value)
        )
        graph.add_edge(claim.claim_id, agent.id, EdgeType.GENERATED_BY)

        for key in claim.protocols:
            proto = graph.add_node(
                Node(id=_protocol_id(key), type=NodeType.PROTOCOL, label=key)
            )
            graph.add_edge(claim.claim_id, proto.id, EdgeType.RELATED_TO)

        for link in claim.links:
            if link.stance is Stance.SUPPORTS:
                graph.add_edge(claim.claim_id, link.evidence_id, EdgeType.SUPPORTED_BY)
            elif link.stance is Stance.CONTRADICTS:
                graph.add_edge(claim.claim_id, link.evidence_id, EdgeType.CONTRADICTED_BY)

    for signal in signals:
        node_id = f"risk:{signal.get('metric')}:{signal.get('protocol') or ''}"
        graph.add_node(
            Node(
                id=node_id,
                type=NodeType.RISK_SIGNAL,
                label=f"{signal.get('metric')} — {signal.get('severity')}",
                attrs={"severity": signal.get("severity"), "z": signal.get("z")},
            )
        )
        if signal.get("protocol"):
            proto = graph.add_node(
                Node(
                    id=_protocol_id(signal["protocol"]),
                    type=NodeType.PROTOCOL,
                    label=signal["protocol"],
                )
            )
            graph.add_edge(proto.id, node_id, EdgeType.HAS_RISK_SIGNAL)
        # Tie the signal to the statistical-evidence node that records it.
        for item in evidence:
            if item.payload.get("metric") == signal.get("metric") and (
                item.kind.value == "statistical_signal"
            ):
                graph.add_edge(node_id, item.evidence_id, EdgeType.DERIVED_FROM)

    return graph


# --- §14.3: why did you conclude that? ---------------------------------


class ProvenanceStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    depth: int
    node: Node
    via: str | None = None


def explain(graph: EvidenceGraph, node_id: str, max_depth: int = 4) -> list[ProvenanceStep]:
    """Walk from a conclusion down to the sources under it.

    Breadth-first so the path reads outward from the thing asked about — the
    claim, then what supports it, then where that came from — which is the order
    §14.3 draws and the order a reader needs.

    Only provenance edges are followed. `related_to` and `generated_by` are real
    relationships but they lead sideways to a protocol or an agent, not down
    toward evidence, and including them would make the answer to "why do you
    believe this" list who said it rather than what it rests on.
    """
    if node_id not in graph.nodes:
        return []

    downward = {
        EdgeType.SUPPORTED_BY,
        EdgeType.CONTRADICTED_BY,
        EdgeType.FROM_SOURCE,
        EdgeType.DERIVED_FROM,
        EdgeType.INVESTIGATED,
        EdgeType.HAS_RISK_SIGNAL,
    }

    steps = [ProvenanceStep(depth=0, node=graph.nodes[node_id])]
    seen = {node_id}
    queue: deque[tuple[str, int]] = deque([(node_id, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for edge in graph.out_edges(current):
            if edge.type not in downward or edge.target in seen:
                continue
            seen.add(edge.target)
            steps.append(
                ProvenanceStep(
                    depth=depth + 1,
                    node=graph.nodes[edge.target],
                    via=edge.type.value,
                )
            )
            queue.append((edge.target, depth + 1))
    return steps


# --- independence -------------------------------------------------------


class SharedSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    label: str
    claim_ids: tuple[str, ...]


def independence(graph: EvidenceGraph) -> list[SharedSource]:
    """Sources that more than one claim rests on.

    The analysis a flat record cannot do. Two claims each carrying three
    citations look like two corroborated findings; if all six citations resolve
    to one page, they are one finding stated twice. A report that lists them as
    separate findings is overstating what was established, and only the
    convergence in this graph makes that visible.
    """
    by_source: dict[str, set[str]] = defaultdict(set)

    for claim in graph.of_type(NodeType.CLAIM):
        for edge in graph.out_edges(claim.id, EdgeType.SUPPORTED_BY):
            for source_edge in graph.out_edges(edge.target, EdgeType.FROM_SOURCE):
                by_source[source_edge.target].add(claim.id)

    return [
        SharedSource(
            source_id=source_id,
            label=graph.nodes[source_id].label,
            claim_ids=tuple(sorted(claim_ids)),
        )
        for source_id, claim_ids in sorted(by_source.items())
        if len(claim_ids) > 1
    ]


def independent_claim_groups(graph: EvidenceGraph) -> list[tuple[str, ...]]:
    """Claims partitioned into groups that share no source.

    Each group is one line of evidence. A report's honest count of independent
    findings is the number of groups, not the number of claims.
    """
    claims = [c.id for c in graph.of_type(NodeType.CLAIM)]
    sources: dict[str, set[str]] = {}
    for claim_id in claims:
        found: set[str] = set()
        for edge in graph.out_edges(claim_id, EdgeType.SUPPORTED_BY):
            for source_edge in graph.out_edges(edge.target, EdgeType.FROM_SOURCE):
                found.add(source_edge.target)
        sources[claim_id] = found

    # Union-find by shared source, iterated to a fixed point.
    groups: list[set[str]] = []
    for claim_id in claims:
        merged = {claim_id}
        remaining = []
        for group in groups:
            if any(sources[claim_id] & sources[other] for other in group):
                merged |= group
            else:
                remaining.append(group)
        remaining.append(merged)
        groups = remaining

    return [tuple(sorted(g)) for g in groups]


# --- visualisation ------------------------------------------------------

_SHAPE = {
    NodeType.CLAIM: ("[", "]"),
    NodeType.EVIDENCE: ("(", ")"),
    NodeType.SOURCE: ("[(", ")]"),
    NodeType.PROTOCOL: ("{{", "}}"),
    NodeType.AGENT: ("([", "])"),
    NodeType.RISK_SIGNAL: (">", "]"),
    NodeType.INVESTIGATION: ("[/", "/]"),
}


def _safe(text: str, limit: int = 48) -> str:
    """Mermaid labels cannot contain quotes, brackets or newlines."""
    cleaned = " ".join(str(text).split())
    cleaned = cleaned.replace('"', "'").replace("[", "(").replace("]", ")")
    return cleaned[: limit - 1] + "…" if len(cleaned) > limit else cleaned


def to_mermaid(graph: EvidenceGraph, include: set[NodeType] | None = None) -> str:
    """Render as a Mermaid flowchart.

    §7's roadmap asks for graph visualisation before a graph database, and
    Mermaid is text: it costs nothing, diffs cleanly, and renders wherever the
    report is read.
    """
    include = include or set(NodeType)
    ids: dict[str, str] = {}
    lines = ["flowchart LR"]

    for i, node in enumerate(graph.nodes.values()):
        if node.type not in include:
            continue
        alias = f"n{i}"
        ids[node.id] = alias
        open_, close = _SHAPE[node.type]
        lines.append(f'    {alias}{open_}"{_safe(node.label)}"{close}')

    for edge in graph.edges:
        if edge.source in ids and edge.target in ids:
            lines.append(
                f"    {ids[edge.source]} -->|{edge.type.value}| {ids[edge.target]}"
            )
    return "\n".join(lines)


# --- state crossing -----------------------------------------------------


def to_state(graph: EvidenceGraph) -> dict[str, Any]:
    """Plain dicts for a checkpointed channel. See `AgentState`."""
    return {
        "nodes": [n.model_dump(mode="json") for n in graph.nodes.values()],
        "edges": [e.model_dump(mode="json") for e in graph.edges],
    }


def from_state(payload: dict[str, Any]) -> EvidenceGraph:
    graph = EvidenceGraph()
    for raw in payload.get("nodes") or []:
        graph.add_node(Node.model_validate(raw))
    for raw in payload.get("edges") or []:
        edge = Edge.model_validate(raw)
        graph.add_edge(edge.source, edge.target, edge.type)
    return graph
