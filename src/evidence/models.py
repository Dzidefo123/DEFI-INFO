"""Evidence, claims, and the links between them.

This is the vocabulary every specialist agent writes into graph state. It exists
so agents communicate in structures rather than prose: the architecture's whole
premise is that "I found something suspicious, liquidity seems to have dropped"
is not a finding, and `{"metric": "liquidity_outflow", "value": 12_500_000,
"baseline": 2_300_000, "z_score": 5.67}` is.

Three decisions here shape everything downstream.

**Identity is content, not authorship.** `evidence_id` and `claim_id` are hashes
of the semantically identifying fields, so the same fact observed twice — by two
agents, in two investigations, a week apart — is one node, not two. Without that,
"how many independent sources support this claim?" counts duplicates and every
confidence score inflates with the number of agents you happen to run. Which
agent produced a piece of evidence is recorded, but it is metadata; it is
deliberately NOT part of the id.

**Stance lives on the edge, not on the evidence.** A TVL drop supports "activity
declined" and contradicts "the protocol is growing". Evidence has no intrinsic
polarity — only a relationship to a specific claim does. Modelling stance as a
field on `Evidence` would force a copy per claim and quietly break dedupe.

**Reliability is assigned by rule, not by a model.** `SourceTier` is derived from
where the evidence came from, deterministically. A language model asked to rate
its own sources will rate them well, and confidence built on that is circular.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


# --- enumerations -------------------------------------------------------


class EvidenceKind(str, Enum):
    """What sort of observation this is. Drives how it may be used."""

    DOCUMENT = "document"                  # a retrieved documentation chunk
    ON_CHAIN_METRIC = "on_chain_metric"    # a measured quantity read from a chain
    CHAIN_STATE = "chain_state"            # contract state at a block: supply, balance, ratio
    MARKET_DATA = "market_data"            # a venue-reported market quantity
    STATISTICAL_SIGNAL = "statistical_signal"  # output of the risk engine
    SECURITY_ADVISORY = "security_advisory"    # a published vulnerability notice
    INCIDENT_REPORT = "incident_report"    # a report of something that happened
    AUDIT_FINDING = "audit_finding"        # a finding from a smart-contract audit


class SourceTier(str, Enum):
    """How much weight a source's provenance earns, by rule.

    Declared in descending order of authority — `verification.check_source_quality`
    relies on that ordering — but the ordering is not absolute. See `ClaimKind`:
    a chain read outranks documentation for a claim about current state and is
    outranked by it for a claim about mechanism.

    The tier is a property of WHERE something came from, never of how convincing
    it reads.
    """

    CHAIN = "chain"            # read directly from chain state
    PRIMARY = "primary"        # the protocol's own whitelisted documentation
    OFFICIAL = "official"      # a named, accountable publisher: audit firm, CVE, advisory
    COMMUNITY = "community"    # attributable but unaccountable: forums, dashboards
    UNVERIFIED = "unverified"  # anonymous or unattributable


class ClaimKind(str, Enum):
    """What sort of assertion a claim makes, which decides what can support it.

    A single reliability ranking over sources cannot be right, because the
    ordering genuinely inverts:

        Documentation records what a protocol COMMITS TO.
        Chain state records what it IS DOING.

    For "reserves are $87.3M" the chain is decisive and the documentation is
    nearly irrelevant — docs describe intent, and intent can be stale or
    aspirational. For "liquidation uses a 3-minute TWAP oracle" the documentation
    is authoritative and chain state says almost nothing: you cannot read a rule
    off a sequence of transactions. Neither strictly outranks the other, so the
    weighting is a matrix rather than a list. See `confidence.RELIABILITY`.

    The kind is DECLARED by the agent that makes the claim rather than inferred
    from its text. An agent reading a documentation page knows it is asserting a
    mechanism; the risk engine knows it is asserting a measured value. Inferring
    it later from wording would put a language model in charge of how heavily its
    own evidence counts.
    """

    STATE = "state"            # a current, measurable value
    MECHANISM = "mechanism"    # how something works, by design
    EVENT = "event"            # something that happened, at a time
    UNSPECIFIED = "unspecified"  # not declared; falls back to a flat ranking


class Stance(str, Enum):
    """A piece of evidence's relationship to one specific claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"  # relevant context that settles nothing either way


class VerificationStatus(str, Enum):
    """The Verification Agent's verdict. `UNVERIFIED` means it has not run.

    `INSUFFICIENT_EVIDENCE` and `CONTRADICTED` are distinct on purpose: the first
    says we do not know, the second says we have reason to think otherwise. A
    system that collapses them reports ignorance as disagreement, or worse,
    disagreement as ignorance.
    """

    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTED = "contradicted"


class AgentName(str, Enum):
    """Every producer of evidence or claims, so attribution cannot be free text."""

    RESEARCH = "research_agent"
    BLOCKCHAIN = "blockchain_agent"
    SECURITY = "security_agent"
    RISK_ENGINE = "risk_engine"
    VERIFICATION = "verification_agent"


# What each agent is competent to assert, and what it assumes when nothing is
# declared.
#
# A per-agent CONSTANT would make `kind` agent identity in disguise — and it
# would be wrong in a specific, predictable direction. A documentation page that
# states "reserves are $87.3M" is a claim about STATE, and a research agent
# locked to MECHANISM would score that documentation at 1.00 in the one row where
# documentation is weakest. So kind is declared per claim; this table bounds what
# a declaration may be, and supplies the default when there is none.
#
# The bound is what stops a declaration from being a free parameter. An agent can
# only claim within its competence — the research agent may say a docs page
# describes a mechanism or states a value, but not that an incident occurred.
AGENT_CLAIM_KINDS: dict[AgentName, frozenset[ClaimKind]] = {
    # Documentation describes how a protocol works, and sometimes states a
    # current value. Both are within reach of a docs page; an incident is not.
    AgentName.RESEARCH: frozenset({ClaimKind.MECHANISM, ClaimKind.STATE}),
    # Reads a curated incident registry and the protocol's own security
    # documentation — genuinely two different kinds of assertion.
    AgentName.SECURITY: frozenset({ClaimKind.EVENT, ClaimKind.MECHANISM}),
    # Produces measurements only; it makes no claims at all today.
    AgentName.BLOCKCHAIN: frozenset({ClaimKind.STATE}),
    # Scores a measured value against its own history. An anomaly is a statement
    # about the present value; a detected regime change is an event.
    AgentName.RISK_ENGINE: frozenset({ClaimKind.STATE, ClaimKind.EVENT}),
    # Adjudicates other agents' claims; it originates none.
    AgentName.VERIFICATION: frozenset(),
}

DEFAULT_CLAIM_KIND: dict[AgentName, ClaimKind] = {
    AgentName.RESEARCH: ClaimKind.MECHANISM,
    AgentName.SECURITY: ClaimKind.EVENT,
    AgentName.BLOCKCHAIN: ClaimKind.STATE,
    AgentName.RISK_ENGINE: ClaimKind.STATE,
    AgentName.VERIFICATION: ClaimKind.UNSPECIFIED,
}


def permitted_kind(agent: AgentName, kind: ClaimKind) -> bool:
    """Whether `agent` may assert a claim of this kind.

    `UNSPECIFIED` is always allowed: declining to declare is not a claim about
    competence, and it already carries its own penalty in the reliability matrix.
    """
    return kind is ClaimKind.UNSPECIFIED or kind in AGENT_CLAIM_KINDS.get(
        agent, frozenset()
    )


# --- helpers ------------------------------------------------------------


def _digest(prefix: str, *parts: Any) -> str:
    """A short, stable, content-addressed id.

    Deterministic across processes and runs — no salt, no randomness, no clock —
    because the point is that the same content yields the same id in tomorrow's
    investigation as in today's. Parts are JSON-encoded with sorted keys so dict
    ordering cannot change an id.
    """
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()}"


_WS = re.compile(r"\s+")


def normalize_claim_text(text: str) -> str:
    """Fold claim text to its identity form.

    Casing, internal whitespace, and a trailing full stop are presentation, not
    content. Two agents phrasing the same finding with and without a period must
    produce one claim, or the evidence for it is split across two nodes and both
    look under-supported.
    """
    return _WS.sub(" ", text).strip().rstrip(".").lower()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _require_aware(value: datetime | None, field: str) -> datetime | None:
    """Reject naive datetimes.

    Temporal relevance and "is this evidence stale?" are load-bearing here, and a
    naive timestamp silently means "some timezone" — which turns an age
    calculation into an off-by-hours error that never raises.
    """
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware; got a naive datetime")
    return value


# --- source -------------------------------------------------------------


class SourceRef(BaseModel):
    """Where a piece of evidence came from, precisely enough to go back to it."""

    model_config = ConfigDict(frozen=True)

    tier: SourceTier
    uri: str                          # doc URL, API endpoint, chain explorer link
    protocol: str | None = None       # whitelisted protocol key, when it has one
    title: str | None = None
    locator: str | None = None        # chunk_id, block number, tx hash, section anchor

    @field_validator("uri")
    @classmethod
    def _uri_present(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("a SourceRef needs a uri; unattributable evidence is not evidence")
        return v.strip()


# --- evidence -----------------------------------------------------------


class Evidence(BaseModel):
    """One observation, from one source, at one time.

    `observed_at` is when the fact was true; `collected_at` is when we looked.
    They differ in the case that matters most: a block's timestamp is the truth
    of an on-chain metric, while the moment we queried the explorer says nothing
    about the data's age. Collapsing them into one field makes stale data look
    fresh precisely when it is being re-read.
    """

    model_config = ConfigDict(frozen=True)

    kind: EvidenceKind
    source: SourceRef
    agent: AgentName
    summary: str                              # human-readable, for the report
    payload: dict[str, Any] = Field(default_factory=dict)  # structured, for computation
    observed_at: datetime | None = None
    collected_at: datetime = Field(default_factory=utcnow)

    @field_validator("observed_at")
    @classmethod
    def _observed_aware(cls, v: datetime | None) -> datetime | None:
        return _require_aware(v, "observed_at")

    @field_validator("collected_at")
    @classmethod
    def _collected_aware(cls, v: datetime) -> datetime:
        return _require_aware(v, "collected_at")

    @field_validator("summary")
    @classmethod
    def _summary_present(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("evidence needs a summary")
        return v.strip()

    @computed_field
    @property
    def evidence_id(self) -> str:
        """Content-addressed. See the module docstring for why `agent` is excluded.

        `collected_at` is excluded for the same reason: re-reading an unchanged
        fact must not mint a second node, or repeated collection inflates the
        independent-source count that confidence is built on.

        A computed FIELD rather than a plain property, so it survives
        `model_dump`. Evidence travels through graph state as plain dicts, and a
        serialized record that cannot state its own identity forces every reader
        to rebuild the model just to resolve a link — including readers outside
        Python, like the evidence graph or anything rendering the record.
        Recomputing it on load is safe precisely because it is derived from
        content: it cannot drift from what it identifies.
        """
        return _digest(
            "ev",
            self.kind.value,
            self.source.uri,
            self.source.locator,
            self.summary,
            self.payload,
            self.observed_at,
        )

    @property
    def as_of(self) -> datetime:
        """The timestamp to age this evidence from — the truth time if known."""
        return self.observed_at or self.collected_at


# --- claims -------------------------------------------------------------


class EvidenceLink(BaseModel):
    """The edge between a claim and a piece of evidence: §14.2's SUPPORTED_BY /
    CONTRADICTED_BY, carrying the stance the evidence takes toward THIS claim."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    stance: Stance
    # How directly the evidence bears on the claim, independent of which way it
    # points. Evidence can be highly reliable and barely relevant; keeping the
    # two apart is what lets verification say "the source is good but it is not
    # about this".
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    note: str | None = None


class Claim(BaseModel):
    """An assertion an agent is prepared to be held to, plus its evidence.

    A claim carries no confidence of its own. Confidence is computed from the
    claim's evidence and verification state by `src.evidence.confidence`, so it
    cannot be asserted by whatever produced the claim — an agent scoring its own
    output is the failure this architecture is built to avoid.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    agent: AgentName
    protocols: tuple[str, ...] = ()
    links: tuple[EvidenceLink, ...] = ()
    verification: VerificationStatus = VerificationStatus.UNVERIFIED
    created_at: datetime = Field(default_factory=utcnow)

    # What sort of assertion this is, declared by whichever agent made it. It
    # decides which sources can properly support the claim — see `ClaimKind`.
    # Deliberately NOT part of `claim_id`: two agents asserting the same sentence
    # about the same protocol are making one claim, and disagreeing about how to
    # weigh it must not split the evidence pooled behind it. Same reasoning as
    # `agent`.
    kind: ClaimKind = ClaimKind.UNSPECIFIED

    @field_validator("text")
    @classmethod
    def _text_present(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("a claim needs text")
        return v.strip()

    @field_validator("created_at")
    @classmethod
    def _created_aware(cls, v: datetime) -> datetime:
        return _require_aware(v, "created_at")

    @model_validator(mode="after")
    def _kind_is_within_competence(self) -> "Claim":
        """An agent may not assert a kind of claim it is not equipped to make.

        This is what stops the declared kind from being a free parameter. Without
        it, whatever produces a claim could choose the row of the reliability
        matrix that flatters its evidence — and the whole point of deriving
        reliability from provenance is that the thing being scored does not get
        to pick its own score.
        """
        if not permitted_kind(self.agent, self.kind):
            allowed = sorted(k.value for k in AGENT_CLAIM_KINDS.get(self.agent, ()))
            raise ValueError(
                f"{self.agent.value} may not assert a {self.kind.value} claim; "
                f"it can assert: {', '.join(allowed) or 'none'}"
            )
        return self

    @model_validator(mode="after")
    def _links_are_unique(self) -> "Claim":
        """One evidence node may not be linked twice to the same claim.

        Duplicate links are how an agent accidentally votes twice with one
        source, which is exactly what content-addressed ids are meant to prevent.
        """
        seen = [link.evidence_id for link in self.links]
        if len(seen) != len(set(seen)):
            dupes = sorted({e for e in seen if seen.count(e) > 1})
            raise ValueError(f"claim links the same evidence more than once: {dupes}")
        return self

    @computed_field
    @property
    def claim_id(self) -> str:
        """Content-addressed on normalized text + protocols.

        Protocols are part of identity because "funding is charged hourly" is a
        different claim about Hyperliquid than about Ethena — the collision this
        corpus was specifically chosen to expose.
        """
        return _digest("claim", normalize_claim_text(self.text), sorted(self.protocols))

    def supporting(self) -> tuple[EvidenceLink, ...]:
        return tuple(l for l in self.links if l.stance is Stance.SUPPORTS)

    def contradicting(self) -> tuple[EvidenceLink, ...]:
        return tuple(l for l in self.links if l.stance is Stance.CONTRADICTS)

    def is_unsupported(self) -> bool:
        """No evidence points toward this claim.

        The single most important predicate in the system: an unsupported claim
        must never reach a user as a finding, regardless of how well it reads.
        """
        return not self.supporting()
