"""§9. The Blockchain Intelligence Agent.

Its job is not to call an API. §9 is explicit that the question is whether
behaviour has *deviated* — and a deviation is a comparison, so this agent's real
output is a current reading paired with the history to judge it against.

That pairing is the whole design:

    take one fresh reading  ->  fetch the series BEFORE it  ->  hand both onward

The agent does not decide whether anything is abnormal. It gathers; the risk
engine scores. Keeping that boundary is what stops "unusual" from meaning
whatever the collecting code happened to think, and it is the same split as §11.2's
"the LLM explains the signal, it does not calculate it" applied one layer earlier.

**This agent makes no claims.** It produces evidence — measurements with times and
sources. The claims about abnormality are produced by the risk engine from the
arithmetic, so that every such claim traces to a computation rather than to an
assertion.

**What it will honestly report today.** A newly-created feature store has no
history, so nothing can be scored and the agent says exactly that, per metric.
That is not a placeholder state to be embarrassed about — it is the correct
output, and a version that produced findings from one reading would be lying.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from src.blockchain import store
from src.blockchain.collectors import Collection, collect
from src.blockchain.features import get_spec, prepare_for_scoring
from src.blockchain.store import Observation
from src.config import settings
from src.evidence.models import (
    AgentName,
    Evidence,
    EvidenceKind,
    SourceRef,
    SourceTier,
)
from src.protocols import get_protocol, is_known
from src.risk.statistics import MIN_BASELINE_N

# Where each protocol's readings come from, for evidence provenance.
#
# CHAIN tier, distinct from the PRIMARY tier documentation carries. The two are
# not ranked against each other globally — see `ClaimKind` — but they are
# different KINDS of authority, and collapsing them would mean a claim about
# current reserves scored a docs page and a chain read identically.
def _source_uri(protocol: str) -> str:
    """Where a reading came from, resolved at call time.

    For chains this is the configured RPC provider rather than a hardcoded
    endpoint, so the citation in a report names the host that actually answered.
    The previous version pointed at an explorer path that no longer exists —
    provenance that outlives the source it names is worse than none.
    """
    from src.blockchain.rpc import providers_for

    if providers := providers_for(protocol):
        return providers[0].url
    if protocol == "hyperliquid":
        return f"{settings.hyperliquid_api}/info"
    return protocol


class MetricSeries(BaseModel):
    """One metric ready for scoring: the reading, and the baseline before it."""

    model_config = ConfigDict(frozen=True)

    metric: str
    protocol: str
    subject: str = ""
    current: float
    history: tuple[float, ...] = ()
    observed_at: datetime

    @property
    def scoreable(self) -> bool:
        return len(self.history) >= MIN_BASELINE_N


class BlockchainOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent: str = AgentName.BLOCKCHAIN.value
    evidence: tuple[Evidence, ...] = ()
    series: tuple[MetricSeries, ...] = ()
    limitations: tuple[str, ...] = ()


def evidence_from_observation(observation: Observation) -> Evidence:
    """One reading becomes one piece of on-chain evidence.

    `observed_at` is the time the reading describes, not the time we asked — the
    distinction `Evidence` was built to keep, and the one that decides whether a
    six-hour-old funding rate is treated as current.
    """
    spec = get_spec(observation.protocol, observation.metric)
    where = f" on {observation.subject}" if observation.subject else ""
    return Evidence(
        # From the metric spec, not a blanket constant: contract state is a fact
        # about one block and goes stale in hours, while a gas price describes a
        # period. One decay for both would make the shortest-lived readings the
        # most overconfident.
        kind=EvidenceKind(spec.evidence_kind),
        source=SourceRef(
            tier=SourceTier.CHAIN,
            uri=_source_uri(observation.protocol),
            protocol=observation.protocol,
            title=get_protocol(observation.protocol).name
            if is_known(observation.protocol)
            else None,
            locator=f"{observation.metric}{where}",
        ),
        agent=AgentName.BLOCKCHAIN,
        summary=(
            f"{spec.description}{where}: {observation.value:,.6g} {spec.unit} "
            f"({observation.protocol})"
        ),
        payload={
            "metric": observation.metric,
            "subject": observation.subject,
            "value": observation.value,
            "unit": spec.unit,
            "kind": spec.kind.value,
        },
        observed_at=observation.observed_at,
        collected_at=observation.collected_at,
    )


def _series_for(observation: Observation, path=None) -> tuple[MetricSeries | None, str | None]:
    """Pair a reading with the history that precedes it, or explain the gap."""
    spec = get_spec(observation.protocol, observation.metric)
    where = f" on {observation.subject}" if observation.subject else ""
    label = f"{observation.protocol}.{spec.scored_as}{where}"

    raw_history = store.prior_history(
        observation.protocol,
        observation.metric,
        observation.subject,
        before=observation.observed_at,
        path=path,
    )

    prepared = prepare_for_scoring(spec, raw_history, observation.value)
    if prepared is None:
        return None, (
            f"{label}: only one reading of a cumulative counter exists, so no "
            f"rate of change can be derived yet"
        )

    current, history = prepared
    series = MetricSeries(
        metric=spec.scored_as,
        protocol=observation.protocol,
        subject=observation.subject,
        current=current,
        history=tuple(history),
        observed_at=observation.observed_at,
    )
    if not series.scoreable:
        return series, (
            f"{label}: {len(history)} prior reading(s) on record, "
            f"{MIN_BASELINE_N} needed before behaviour can be compared against a "
            f"baseline — not enough history to say whether this is unusual"
        )
    return series, None


def investigate(
    protocols: Sequence[str],
    subjects: Sequence[str] = (),
    collector=None,
    path=None,
) -> BlockchainOutput:
    """Collect current readings for each protocol and pair them with their history.

    `collector` is injected so the pipeline runs in tests without touching a
    network, mirroring how the Research Agent's model calls are injected.

    Resolved in the body, not as a default argument. A default binds once at
    import, so `collector=collect` would capture the function object and make the
    module attribute unpatchable — a test that swapped it would still reach an
    explorer. The Research Agent shipped with exactly that bug in C1.
    """
    collector = collector or collect

    evidence: list[Evidence] = []
    series: list[MetricSeries] = []
    limitations: list[str] = []

    if not protocols:
        return BlockchainOutput(
            limitations=(
                "blockchain_agent: no protocol was identified, and on-chain "
                "metrics are per-protocol — nothing could be measured",
            )
        )

    for key in protocols:
        results: list[Collection] = []
        if subjects and is_known(key) and _is_per_market(key):
            results.extend(collector(key, subject) for subject in subjects)
        else:
            results.append(collector(key))

        for result in results:
            limitations.extend(result.errors)
            if result.observations:
                store.record(result.observations, path=path)
            for observation in result.observations:
                evidence.append(evidence_from_observation(observation))
                one, gap = _series_for(observation, path=path)
                if one is not None:
                    series.append(one)
                if gap:
                    limitations.append(f"blockchain_agent: {gap}")

    return BlockchainOutput(
        evidence=tuple(evidence),
        series=tuple(series),
        limitations=tuple(limitations),
    )


def _is_per_market(protocol_key: str) -> bool:
    from src.blockchain.collectors import needs_subject

    return needs_subject(protocol_key)
