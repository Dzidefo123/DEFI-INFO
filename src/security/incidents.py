"""§10. Security findings, and the four categories that must never be merged.

    Confirmed Incident   — it happened, and a named source stands behind that
    Known Vulnerability  — a weakness is documented, whether or not it was used
    Suspicious Signal    — something looks off; nobody has established what
    Unverified Claim     — somebody said so

§10 is emphatic that these stay apart, and the reason is asymmetric cost. Reading
an unverified claim as a confirmed incident defames a protocol and panics a user
into moving funds. Reading a confirmed incident as an unverified claim leaves
someone exposed to a known-exploited flaw. There is no safe direction to blur in,
so the distinction is carried in the schema rather than in prose.

**The rule that enforces it: only established categories may SUPPORT a claim.**
A suspicious signal and an unverified claim attach with `NEUTRAL` stance — they
appear in the record, a report can mention them, and they can never raise the
confidence of a security finding. That is exactly §13.2's worked example: a forum
post attributing an outflow to an exploit does not make the exploit more likely,
it makes the rumour visible.

**Classification and source tier are independent axes.** A confirmed incident
reported by a community researcher is still confirmed, and still community-tier.
An unverified rumour on an official-looking site is still unverified. Collapsing
"who said it" into "how established is it" is the specific mistake this module
exists to prevent.

---

**Why the registry ships empty.**

This file loads a whitelist of security findings, the same way `protocols.py`
loads a whitelist of protocols: an entry exists because a person put it there on
purpose, with a source anyone can check.

It ships with no entries. Populating it from a model's recollection of DeFi
history would be the worst possible use of this system — a fabricated or
misremembered incident attached to a real protocol is defamatory, and it would
arrive wearing the CONFIRMED_INCIDENT label that everything downstream is built
to trust. Every entry needs a citation an operator has verified.

Until then the agent reports that nothing is on file, and says plainly that this
is not the same as nothing having happened.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.evidence.models import EvidenceKind, SourceTier, Stance
from src.protocols import is_known

REGISTRY = Path(__file__).parent / "registry.jsonl"


class Classification(str, Enum):
    """§10's four categories. Ordered most to least established."""

    CONFIRMED_INCIDENT = "confirmed_incident"
    KNOWN_VULNERABILITY = "known_vulnerability"
    SUSPICIOUS_SIGNAL = "suspicious_signal"
    UNVERIFIED_CLAIM = "unverified_claim"


class Status(str, Enum):
    """Whether the finding still bears on the protocol's current risk."""

    ACTIVE = "active"          # unmitigated as far as the source says
    MITIGATED = "mitigated"    # addressed, but the history is still relevant
    RESOLVED = "resolved"      # closed out
    UNKNOWN = "unknown"


# Which categories may support a claim. The whole point of §10's separation.
#
# A suspicious signal is a prompt to look, not a finding: attaching it as support
# would let "something looks odd" accumulate into "something is wrong" purely by
# repetition. An unverified claim is weaker still.
SUPPORTING = frozenset({Classification.CONFIRMED_INCIDENT, Classification.KNOWN_VULNERABILITY})

STANCE: dict[Classification, Stance] = {
    Classification.CONFIRMED_INCIDENT: Stance.SUPPORTS,
    Classification.KNOWN_VULNERABILITY: Stance.SUPPORTS,
    Classification.SUSPICIOUS_SIGNAL: Stance.NEUTRAL,
    Classification.UNVERIFIED_CLAIM: Stance.NEUTRAL,
}

EVIDENCE_KIND: dict[Classification, EvidenceKind] = {
    Classification.CONFIRMED_INCIDENT: EvidenceKind.INCIDENT_REPORT,
    Classification.KNOWN_VULNERABILITY: EvidenceKind.SECURITY_ADVISORY,
    Classification.SUSPICIOUS_SIGNAL: EvidenceKind.SECURITY_ADVISORY,
    Classification.UNVERIFIED_CLAIM: EvidenceKind.SECURITY_ADVISORY,
}


class IncidentRecord(BaseModel):
    """One security finding about one whitelisted protocol."""

    model_config = ConfigDict(frozen=True)

    id: str
    protocol: str
    classification: Classification
    title: str
    summary: str
    source_uri: str
    source_tier: SourceTier
    status: Status = Status.UNKNOWN
    occurred_at: datetime | None = None
    disclosed_at: datetime | None = None
    references: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("protocol")
    @classmethod
    def _protocol_is_whitelisted(cls, v: str) -> str:
        """A finding about a protocol the system does not cover cannot be
        surfaced, scoped, or corrected — and would be filed under a name nothing
        else in the system recognises."""
        if not is_known(v):
            raise ValueError(f"incident references non-whitelisted protocol {v!r}")
        return v

    @field_validator("source_uri")
    @classmethod
    def _has_a_source(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("a security finding without a checkable source is a rumour")
        return v.strip()

    @property
    def stance(self) -> Stance:
        return STANCE[self.classification]

    @property
    def evidence_kind(self) -> EvidenceKind:
        return EVIDENCE_KIND[self.classification]

    @property
    def is_established(self) -> bool:
        """Whether this may support a finding, as opposed to merely appearing."""
        return self.classification in SUPPORTING

    @property
    def as_of(self) -> datetime | None:
        """When this became true, preferring the event over its disclosure."""
        return self.occurred_at or self.disclosed_at


class RegistryError(ValueError):
    """A malformed registry. Raised rather than skipped — see `load`."""


def load(path: Path | None = None) -> tuple[IncidentRecord, ...]:
    """Read the registry. A malformed entry fails the whole load.

    Deliberately not lenient. Skipping a bad line would mean a typo silently
    removes a confirmed incident from every future investigation, and the
    resulting silence is indistinguishable from a protocol having no incidents.
    Loud beats lossy here.
    """
    target = Path(path or REGISTRY)
    if not target.exists():
        return ()

    records: list[IncidentRecord] = []
    seen: set[str] = set()
    with target.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                record = IncidentRecord.model_validate_json(line)
            except Exception as exc:
                raise RegistryError(f"{target.name} line {number}: {exc}") from exc
            if record.id in seen:
                raise RegistryError(f"{target.name} line {number}: duplicate id {record.id!r}")
            seen.add(record.id)
            records.append(record)
    return tuple(records)


def for_protocols(
    protocols: list[str] | tuple[str, ...], path: Path | None = None
) -> tuple[IncidentRecord, ...]:
    """Findings on file for any of `protocols`. Empty tuple when none are."""
    wanted = set(protocols)
    return tuple(r for r in load(path) if r.protocol in wanted)


def counts_by_classification(
    records: tuple[IncidentRecord, ...],
) -> dict[Classification, int]:
    """A breakdown that keeps the categories visibly separate in any summary."""
    return {c: sum(1 for r in records if r.classification is c) for c in Classification}
