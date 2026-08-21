"""§10. The Security Intelligence Agent.

Two streams, deliberately not blended.

**The incident registry** answers "is anything known to have happened, or to be
wrong". Entries are curated, cited, and carry §10's four-way classification. Only
established categories may support a finding; see `security.incidents`.

**The protocol's own security documentation** answers "what does this protocol
say about its own risks" — audits, bug bounties, custody arrangements, risk
disclosures. This is already in the corpus, so it costs one local retrieval and
no network.

The second stream is the one to be careful with. A risk-disclosure page is a
protocol describing a risk it designed around; it is **not** evidence that
anything went wrong. Treating "Ethena documents custodial risk" as a security
finding would turn every well-documented protocol into a suspect and every
undocumented one into a clean bill of health — precisely inverted. So documented
risks produce evidence and a factual claim about what is documented, and never a
claim that a vulnerability exists.

**Silence here is the most dangerous silence in the system.** An empty security
section reads exactly like a clean one, and this agent will be empty for a while:
the registry ships with no entries. Every path out therefore states what was
searched, and says that nothing on file is not the same as nothing having
happened.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict

from src.evidence.models import (
    AgentName,
    Claim,
    ClaimKind,
    Evidence,
    EvidenceLink,
    SourceRef,
    SourceTier,
    Stance,
)
from src.evidence.models import EvidenceKind
from src.protocols import english_list, get_protocol, is_known
from src.security.incidents import (
    Classification,
    IncidentRecord,
    counts_by_classification,
    for_protocols,
)

# Angles for the documentation sweep. Fixed rather than model-generated: these
# are the standing questions of a security review, they do not vary with the
# phrasing of the request, and a fixed list costs nothing and cannot drift.
SECURITY_ANGLES = (
    "security audits and audit findings",
    "known risks and risk disclosures",
    "custody, collateral and asset safekeeping",
    "bug bounty and vulnerability disclosure process",
)

DOCS_PER_ANGLE = 3


class SecurityOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent: str = AgentName.SECURITY.value
    evidence: tuple[Evidence, ...] = ()
    claims: tuple[Claim, ...] = ()
    records: tuple[IncidentRecord, ...] = ()
    limitations: tuple[str, ...] = ()


Retriever = Callable[[str, Sequence[str] | None], list[Document]]


def _default_retriever(query: str, protocols: Sequence[str] | None) -> list[Document]:
    from src.retrieval.retriever import hybrid_search

    return hybrid_search(query, k=DOCS_PER_ANGLE, protocols=protocols or None)


# --- registry stream ----------------------------------------------------


def evidence_from_record(record: IncidentRecord) -> Evidence:
    """A registry entry becomes evidence, carrying its classification.

    The classification travels in the payload rather than being flattened into
    the summary, so nothing downstream has to parse prose to discover whether a
    finding was confirmed or merely alleged.
    """
    return Evidence(
        kind=record.evidence_kind,
        source=SourceRef(
            tier=record.source_tier,
            uri=record.source_uri,
            protocol=record.protocol,
            title=record.title,
            locator=record.id,
        ),
        agent=AgentName.SECURITY,
        summary=f"[{record.classification.value}] {record.title}: {record.summary}",
        payload={
            "classification": record.classification.value,
            "status": record.status.value,
            "incident_id": record.id,
            "references": list(record.references),
        },
        observed_at=record.as_of,
    )


def claims_from_records(
    records: Sequence[IncidentRecord], evidence: Sequence[Evidence]
) -> list[Claim]:
    """One claim per established finding, linked to the entry behind it.

    Suspicious signals and unverified claims produce NO claim of their own. They
    are still attached as neutral context to the claims that do exist, so a
    report can show them without their presence implying support.
    """
    by_id = {e.source.locator: e for e in evidence}
    context = tuple(
        EvidenceLink(evidence_id=by_id[r.id].evidence_id, stance=Stance.NEUTRAL)
        for r in records
        if not r.is_established and r.id in by_id
    )

    claims: list[Claim] = []
    for record in records:
        if not record.is_established or record.id not in by_id:
            continue
        primary = EvidenceLink(
            evidence_id=by_id[record.id].evidence_id, stance=record.stance
        )
        claims.append(
            Claim(
                text=(
                    f"{get_protocol(record.protocol).name}: "
                    f"{record.title} ({record.classification.value.replace('_', ' ')}, "
                    f"status {record.status.value})"
                ),
                agent=AgentName.SECURITY,
                protocols=(record.protocol,),
                links=(primary, *context),
                # A security finding asserts that something happened.
                kind=ClaimKind.EVENT,
            )
        )
    return claims


# --- documentation stream ----------------------------------------------


def evidence_from_document(doc: Document) -> Evidence:
    """A retrieved security-documentation chunk.

    Kind is DOCUMENT, not SECURITY_ADVISORY. A protocol's risk page is the
    protocol talking about itself; calling it an advisory would give a
    well-written disclosure the weight of a published vulnerability notice.
    """
    meta = doc.metadata
    from src.agents.research import _breadcrumb

    return Evidence(
        kind=EvidenceKind.DOCUMENT,
        source=SourceRef(
            tier=SourceTier.PRIMARY,
            uri=meta["source"],
            protocol=meta.get("protocol"),
            title=meta.get("title"),
            locator=meta.get("doc_id"),
        ),
        agent=AgentName.SECURITY,
        summary=_breadcrumb(meta.get("title"), meta.get("heading")) or meta["source"],
        payload={"text": doc.page_content, "security_documentation": True},
    )


def _documented_risk_claim(
    protocols: Sequence[str], evidence: Sequence[Evidence]
) -> Claim | None:
    """A single factual claim about what the protocol documents.

    Carefully worded, and the wording is the point: it asserts that risks are
    *documented*, not that risks were *found*. Those are different statements and
    only one of them is supported by having read a docs page.
    """
    if not evidence:
        return None
    named = [get_protocol(k).name for k in protocols if is_known(k)]
    who = english_list(named) if named else "the protocols in scope"
    return Claim(
        # Phrased to avoid number agreement across one or several protocols, and
        # more importantly to keep the subject "documentation" rather than "risk".
        text=(
            f"Security documentation for {who} covers {len(evidence)} relevant "
            f"sections, describing disclosed risks and stated controls. This is "
            f"what the protocol publishes about itself; it is not evidence of an "
            f"incident or of an unpatched vulnerability."
        ),
        agent=AgentName.SECURITY,
        protocols=tuple(protocols),
        # A claim about what the documentation COVERS is a claim about the
        # protocol's design and disclosures, not about an event.
        kind=ClaimKind.MECHANISM,
        links=tuple(
            EvidenceLink(evidence_id=e.evidence_id, stance=Stance.SUPPORTS)
            for e in evidence
        ),
    )


# --- the agent ----------------------------------------------------------


def _registry_note(
    protocols: Sequence[str], records: Sequence[IncidentRecord]
) -> str | None:
    """State what the registry search covered, especially when it found nothing."""
    named = english_list([get_protocol(k).name for k in protocols if is_known(k)])
    if not records:
        return (
            f"security_agent: no security findings are on file for {named or 'the '
            'protocols in scope'}. The incident registry is curated and currently "
            f"empty, so this means nothing has been recorded — NOT that nothing "
            f"has happened. No external threat-intelligence feed is connected."
        )
    counts = counts_by_classification(tuple(records))
    breakdown = ", ".join(
        f"{n} {c.value.replace('_', ' ')}" for c, n in counts.items() if n
    )
    unestablished = sum(
        n for c, n in counts.items() if c not in {Classification.CONFIRMED_INCIDENT,
                                                  Classification.KNOWN_VULNERABILITY}
    )
    note = f"security_agent: {len(records)} finding(s) on file — {breakdown}."
    if unestablished:
        note += (
            f" {unestablished} of these are not established and are shown as "
            f"context only; they do not support any conclusion."
        )
    return note


def investigate(
    protocols: Sequence[str],
    retriever: Retriever | None = None,
    registry_path=None,
) -> SecurityOutput:
    """Search the registry and the protocol's own security documentation.

    `retriever` resolves in the body, not as a default argument — see
    `research.investigate` for the bug that pattern caused twice.
    """
    retriever = retriever or _default_retriever

    if not protocols:
        return SecurityOutput(
            limitations=(
                "security_agent: no protocol was identified, and security history "
                "is per-protocol — nothing was searched",
            )
        )

    limitations: list[str] = []

    # Stream 1: curated findings.
    records = for_protocols(list(protocols), registry_path)
    registry_evidence = [evidence_from_record(r) for r in records]
    claims = claims_from_records(records, registry_evidence)
    if note := _registry_note(protocols, records):
        limitations.append(note)

    # Stream 2: the protocol's own security documentation.
    pooled: dict[str, Evidence] = {}
    for angle in SECURITY_ANGLES:
        for doc in retriever(angle, protocols):
            item = evidence_from_document(doc)
            pooled.setdefault(item.evidence_id, item)
    docs_evidence = list(pooled.values())

    if not docs_evidence:
        limitations.append(
            "security_agent: no security documentation was retrieved for the "
            "protocols in scope, so their disclosed risks and controls could not "
            "be reviewed"
        )
    elif (documented := _documented_risk_claim(protocols, docs_evidence)) is not None:
        claims.append(documented)

    return SecurityOutput(
        evidence=tuple(registry_evidence + docs_evidence),
        claims=tuple(claims),
        records=records,
        limitations=tuple(limitations),
    )
