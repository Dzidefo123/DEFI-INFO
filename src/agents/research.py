"""§8. The Research Agent: the existing retrieval pipeline, turned into evidence.

The CX pipeline answers a question with prose and citations. This agent answers
it with *claims linked to evidence*, which is a different output even though the
retrieval underneath is the same — and it is deliberately the same. Protocol
routing, BM25, dense search, RRF and the cross-encoder are the measured parts of
this system (recall@5 = 0.969); rebuilding them for the investigation path would
mean maintaining two retrievers and measuring one.

What is new is the shape at each end:

**In**: one question becomes several, each attacking a different angle. A single
query has one vocabulary and reaches whatever that vocabulary reaches; §8.2's
decomposition is how an investigation reaches documentation the original phrasing
would have missed.

**Out**: chunks become `Evidence`, and a model proposes `Claim`s that cite them
by number. The model chooses what to assert and what it rests on; it does not get
to invent the resting place. Every citation is resolved against the evidence
actually retrieved, and a claim left with nothing is discarded rather than
reported weakly — see `link_claims`.

**Cost**: two model calls (decompose, synthesize) regardless of how many
sub-queries run, because retrieval itself is local and free. That is the whole
reason decomposition is affordable here.

Every model-dependent step is injected, so the agent runs end-to-end in tests
without an API key.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from src.agents import prompts
from src.config import settings
from src.evidence.models import (
    DEFAULT_CLAIM_KIND,
    AgentName,
    Claim,
    ClaimKind,
    Evidence,
    EvidenceKind,
    EvidenceLink,
    SourceRef,
    SourceTier,
    Stance,
    permitted_kind,
)
from src.protocols import english_list, get_protocol, is_known
from src.retrieval.retriever import hybrid_search

# One query per §8.2 angle. More would not add angles, only near-duplicates of
# the ones already asked — and every extra query widens the evidence pool that
# verification then has to work through.
MAX_SUB_QUERIES = 4

# Ceiling on excerpts handed to synthesis. Bounds the one call whose cost scales
# with retrieval breadth; beyond this the marginal chunk is near-duplicate anyway
# after fusion has deduplicated by doc_id.
MAX_CONTEXT_CHUNKS = 12

ANGLES = ("documentation", "architecture", "governance", "historical")


# --- structured model outputs -------------------------------------------


class SubQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    angle: str = Field(description=f"one of {', '.join(ANGLES)}")


class Decomposition(BaseModel):
    queries: list[SubQuery] = Field(default_factory=list)


class ProposedClaim(BaseModel):
    """A claim as the model offers it: text plus the excerpt numbers it rests on.

    Deliberately NOT a `Claim`. The model proposes; `link_claims` decides what
    becomes a claim, after checking that every cited excerpt exists. Letting a
    model construct the evidence graph directly would let it cite evidence it
    invented, which is the failure the graph exists to make visible.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    excerpts: list[int] = Field(
        default_factory=list, description="1-based excerpt numbers supporting this claim"
    )
    # Declared per claim rather than fixed per agent. A documentation page can
    # describe how something works OR state a current value, and those weigh
    # differently — a stated value read off documentation is weak evidence for
    # that value, however authoritative the page. Bounded by what the agent may
    # assert; see `AGENT_CLAIM_KINDS`.
    kind: ClaimKind = Field(
        default=ClaimKind.MECHANISM,
        description=(
            "mechanism = how it works by design; state = a current value the "
            "excerpts report. Default mechanism."
        ),
    )


class Synthesis(BaseModel):
    claims: list[ProposedClaim] = Field(default_factory=list)


class ResearchOutput(BaseModel):
    """§8.3's structured result."""

    model_config = ConfigDict(frozen=True)

    agent: str = AgentName.RESEARCH.value
    sub_queries: tuple[SubQuery, ...] = ()
    claims: tuple[Claim, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    limitations: tuple[str, ...] = ()


# --- models -------------------------------------------------------------


@lru_cache(maxsize=None)
def _llm(model_id: str, max_tokens: int = 2048):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model_id, max_tokens=max_tokens, api_key=settings.anthropic_api_key or None
    )


Decomposer = Callable[[str, Sequence[str]], list[SubQuery]]
Synthesizer = Callable[[str, Sequence[Document]], list[ProposedClaim]]
Retriever = Callable[[str, Sequence[str] | None], list[Document]]


def llm_decompose(question: str, protocols: Sequence[str]) -> list[SubQuery]:
    """Split the question across §8.2's angles. One structured-output call."""
    named = [get_protocol(k).name for k in protocols if is_known(k)]
    scope = (
        f"The investigation concerns {english_list(named)}. Phrase queries in "
        f"that protocol's terms."
        if named
        else "No specific protocol was identified; keep queries general."
    )
    model = _llm(settings.router_model_id, max_tokens=512).with_structured_output(
        Decomposition
    )
    result: Decomposition = model.invoke(
        [
            (
                "system",
                prompts.DECOMPOSE.format(max_queries=MAX_SUB_QUERIES, scope=scope),
            ),
            ("human", question),
        ]
    )
    return result.queries[:MAX_SUB_QUERIES]


def llm_synthesize(question: str, docs: Sequence[Document]) -> list[ProposedClaim]:
    """Propose claims over the retrieved excerpts. One structured-output call."""
    context = "\n\n---\n\n".join(
        f"[{i}] {d.metadata.get('source', '?')}\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )
    model = _llm(settings.model_id).with_structured_output(Synthesis)
    result: Synthesis = model.invoke(
        [("system", prompts.SYNTHESIZE.format(question=question, context=context))]
    )
    return result.claims


def _default_retriever(query: str, protocols: Sequence[str] | None) -> list[Document]:
    return hybrid_search(query, protocols=protocols or None)


# --- deterministic core -------------------------------------------------


def _breadcrumb(title: str | None, heading: str | None) -> str:
    """Page title plus section path, without repeating a segment.

    `chunk.py` builds `heading` from the h1/h2/h3 chain, and on these docs sites
    the h1 is usually the page title — so the naive join produces "How USDe Works
    > How USDe Works > How USDe Maintains its Peg". Consecutive duplicates are
    collapsed because this string is what a report prints to identify a source,
    and a stuttering label reads as a bug in the evidence rather than in the
    formatting.
    """
    parts: list[str] = []
    for segment in [title, *(heading or "").split(" > ")]:
        segment = (segment or "").strip()
        if segment and (not parts or parts[-1] != segment):
            parts.append(segment)
    return " > ".join(parts)


def evidence_from_document(doc: Document) -> Evidence:
    """One retrieved chunk becomes one piece of evidence.

    Tier is PRIMARY without inspection because the corpus IS the whitelist: a
    chunk exists only if `protocols.py` authorised the crawl of the page it came
    from, and a protocol's own documentation is the most authoritative statement
    of what that protocol does. That is provenance decided by rule, per §3.3 —
    not a model's opinion of how trustworthy a page looked.
    """
    meta = doc.metadata
    breadcrumb = _breadcrumb(meta.get("title"), meta.get("heading"))
    payload = {"text": doc.page_content}
    if (score := meta.get("rerank_score")) is not None:
        payload["rerank_score"] = score

    return Evidence(
        kind=EvidenceKind.DOCUMENT,
        source=SourceRef(
            tier=SourceTier.PRIMARY,
            uri=meta["source"],
            protocol=meta.get("protocol"),
            title=meta.get("title"),
            locator=meta.get("doc_id"),
        ),
        agent=AgentName.RESEARCH,
        summary=breadcrumb or meta["source"],
        payload=payload,
    )


def gather_evidence(
    sub_queries: Iterable[SubQuery],
    protocols: Sequence[str] | None,
    retriever: Retriever = _default_retriever,
) -> list[Evidence]:
    """Run every sub-query and pool the results, deduplicated.

    Deduplication is by `evidence_id`, which is content-addressed — so a chunk
    reached by three different angles is one piece of evidence, not three. That
    matters beyond tidiness: `confidence.assess` counts distinct sources, and
    without this, asking the same question four ways would look like four-fold
    corroboration of whatever all four ways happened to find.
    """
    pooled: dict[str, Evidence] = {}
    for sub in sub_queries:
        for doc in retriever(sub.text, protocols):
            item = evidence_from_document(doc)
            pooled.setdefault(item.evidence_id, item)
    return list(pooled.values())


def link_claims(
    proposed: Sequence[ProposedClaim],
    ordered_evidence: Sequence[Evidence],
    protocols: Sequence[str] = (),
) -> tuple[list[Claim], list[str]]:
    """Resolve proposed claims against the evidence actually retrieved.

    Returns the claims that survived, and a limitation line for each that did
    not. Three things are enforced here, all deterministically:

    1. An excerpt number outside the range handed to the model is dropped. It
       refers to nothing, and the same sanitisation already guards the router's
       protocol keys.
    2. A claim citing the same excerpt twice links it once. `Claim` rejects
       duplicate links outright, so without this a repeated citation would take
       down the whole investigation rather than the one claim.
    3. A claim left with no valid citation is DISCARDED, not kept and scored low.
       It would score zero anyway, but a zero-confidence claim still appears in
       the record and invites someone to read it as a weak finding. It is not a
       weak finding; nothing supports it.
    """
    kept: list[Claim] = []
    dropped: list[str] = []
    default = DEFAULT_CLAIM_KIND[AgentName.RESEARCH]

    for claim in proposed:
        # A kind outside the agent's competence falls back to the default rather
        # than failing the investigation — the claim itself may still be sound.
        kind = claim.kind if permitted_kind(AgentName.RESEARCH, claim.kind) else default
        links, seen = [], set()
        for number in claim.excerpts:
            index = number - 1
            if not 0 <= index < len(ordered_evidence):
                continue
            evidence_id = ordered_evidence[index].evidence_id
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            links.append(EvidenceLink(evidence_id=evidence_id, stance=Stance.SUPPORTS))

        if not links:
            dropped.append(
                f"research_agent: discarded an unsupported claim — "
                f"{claim.text!r} cited no retrievable excerpt"
            )
            continue

        kept.append(
            Claim(
                text=claim.text,
                agent=AgentName.RESEARCH,
                protocols=tuple(protocols),
                links=tuple(links),
                # Declared by the synthesiser, bounded by competence. A model
                # cannot use this to flatter its evidence: within what the
                # research agent may assert, MECHANISM is already the strongest
                # row for documentation, so any other declaration LOWERS the
                # score. Declaring STATE for "reserves are $87.3M" is therefore
                # both correct and against the model's interest.
                kind=kind,
            )
        )
    return kept, dropped


# --- the agent ----------------------------------------------------------


def investigate(
    question: str,
    protocols: Sequence[str] = (),
    decomposer: Decomposer | None = None,
    synthesizer: Synthesizer | None = None,
    retriever: Retriever | None = None,
) -> ResearchOutput:
    """Decompose -> retrieve -> extract evidence -> propose claims -> link them.

    The model-dependent steps are parameters so the pipeline can be exercised
    end-to-end without an API key. That is not only a testing convenience: it
    keeps the boundary between "a model decided this" and "code decided this"
    visible in the signature.

    Defaults resolve HERE rather than in the signature. A default argument binds
    once, at import, so `decomposer=llm_decompose` would capture the function
    object and make the module attribute unpatchable — a caller that swapped
    `research.llm_decompose` would still reach the network, which is exactly the
    failure that would go unnoticed in a test suite meant to run offline.
    """
    decomposer = decomposer or llm_decompose
    synthesizer = synthesizer or llm_synthesize
    retriever = retriever or _default_retriever

    limitations: list[str] = []

    sub_queries = decomposer(question, protocols)
    if not sub_queries:
        # Never silently fall back to no research at all.
        sub_queries = [SubQuery(text=question, angle="documentation")]
        limitations.append(
            "research_agent: decomposition produced no sub-queries; fell back to "
            "the original question, so only one angle was searched"
        )

    evidence = gather_evidence(sub_queries, protocols, retriever)
    if not evidence:
        return ResearchOutput(
            sub_queries=tuple(sub_queries),
            limitations=tuple(
                limitations
                + [
                    "research_agent: retrieval returned nothing for any sub-query, "
                    "so no documentary evidence was found — this means the corpus "
                    "was not searched successfully, not that the answer is no"
                ]
            ),
        )

    # Rank by the cross-encoder score retrieval already computed, so the excerpts
    # the model sees are the best ones rather than whichever sub-query ran first.
    ordered = sorted(
        evidence,
        key=lambda e: e.payload.get("rerank_score", float("-inf")),
        reverse=True,
    )[:MAX_CONTEXT_CHUNKS]

    docs = [
        Document(page_content=e.payload["text"], metadata={"source": e.source.uri})
        for e in ordered
    ]
    proposed = synthesizer(question, docs)
    claims, dropped = link_claims(proposed, ordered, protocols)
    limitations.extend(dropped)

    if not claims:
        limitations.append(
            "research_agent: no claim survived citation checking, so the "
            "documentation retrieved does not support a finding"
        )

    return ResearchOutput(
        sub_queries=tuple(sub_queries),
        claims=tuple(claims),
        # Report the full pooled evidence, not just what synthesis saw, so the
        # record shows everything retrieval found.
        evidence=tuple(evidence),
        limitations=tuple(limitations),
    )
