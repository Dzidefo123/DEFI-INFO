"""Turning the existing read-only tools into a stream of stored readings.

Nothing here talks to a chain directly — `src.tools.hyperliquid` and
`src.tools.hyperevm` already do that, and they are the audited, rate-limit-aware,
whitelist-respecting path. A collector's whole job is to call one of them, pull
out the fields the registry knows about, validate them, and stamp them with a
time.

**A protocol with no live tool collects nothing, and says so.** Ethena is in the
whitelist and has no on-chain reader wired up. The temptation is to skip it
silently; the consequence of doing that is an investigation into Ethena that
reports no anomalies, which reads exactly like an investigation that found none.
`collect` returns the reason instead.

**Failures are reported, never substituted.** If the explorer is unreachable, the
collector returns no observations and an error. It does not carry forward the
last known value — a repeated stale reading would shrink the variance of the
baseline it joins, making the series look calmer than it is precisely while we
have stopped being able to see it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.blockchain.features import (
    InvalidObservation,
    MetricSpec,
    SubjectKind,
    specs_for,
    validate,
)
from src.blockchain.store import Observation
from src.protocols import get_protocol, is_known


@dataclass
class Collection:
    """What one collection attempt produced."""

    observations: list[Observation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.observations)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _build(
    spec: MetricSpec,
    raw: object,
    subject: str,
    observed_at: datetime,
    collected_at: datetime,
) -> Observation:
    return Observation(
        protocol=spec.protocol,
        metric=spec.key,
        subject=subject,
        value=validate(spec, raw),
        observed_at=observed_at,
        collected_at=collected_at,
    )


def _harvest(
    protocol: str,
    readings: dict[str, object],
    subject: str,
    observed_at: datetime,
    collected_at: datetime,
    subject_kind=None,
) -> Collection:
    """Map a tool's response onto the registered metrics for its protocol.

    Driven by the registry rather than by the response: a field the tool returns
    that nothing registered is ignored, and a registered metric the tool did not
    return is recorded as an error. The alternative — storing whatever came back
    — means the feature store's schema is whatever the upstream API happened to
    send last, which is not a schema.
    """
    out = Collection()
    for spec in specs_for(protocol):
        # A chain-level run should not be told it is missing every contract
        # metric, nor a contract run every chain metric. Absence only means
        # something within the set actually being collected.
        if subject_kind is not None and spec.subject_kind is not subject_kind:
            continue
        if spec.key not in readings:
            out.errors.append(f"{protocol}.{spec.key}: not present in the response")
            continue
        try:
            out.observations.append(
                _build(spec, readings[spec.key], subject, observed_at, collected_at)
            )
        except InvalidObservation as exc:
            out.errors.append(f"{protocol}.{spec.key}: rejected — {exc}")
    return out


# --- per-protocol collectors -------------------------------------------


def collect_hyperliquid(coin: str) -> Collection:
    """One perp market's current readings."""
    from src.tools.hyperliquid import MarketDataError, market_snapshot

    now = _now()
    try:
        snap = market_snapshot(coin)
    except MarketDataError as exc:
        return Collection(errors=[f"hyperliquid: market data unavailable — {exc}"])

    return _harvest("hyperliquid", snap, snap["coin"], now, now)


# Scan parameters, sized from measurement rather than from the documentation.
#
# Observed on 2026-08-20 over 90 contiguous blocks: 0.99s per block, gas limits
# of exactly 3,000,000 and 30,000,000, and ONE big block in ninety. So big blocks
# arrive roughly every 60-90 blocks, and a fixed short window would collect the
# small series reliably while almost never sampling the big one.
#
# The scan therefore walks backward from the head and stops as soon as it has
# enough of both, rather than always paying for the worst case. Expected cost is
# around 45 reads; the ceiling keeps a run inside the public endpoint's
# 100-per-minute budget even when a big block is slow to appear.
MAX_SCAN_BLOCKS = 95
MIN_SMALL_BLOCKS = 20
MIN_BIG_BLOCKS = 1


def collect_hyperevm(_subject: str = "") -> Collection:
    """Chain-level readings for HyperEVM, over JSON-RPC.

    Reads a CONTIGUOUS window of blocks rather than one, for two reasons. It
    separates the two block populations — see `features.classify_block` — and it
    averages within each, so the reading describes the window rather than
    whichever block happened to be latest.
    """
    from src.blockchain import rpc
    from src.blockchain.features import classify_block

    now = _now()
    out = Collection()
    readings: dict[str, object] = {}

    # Gas price first: one call, and if the provider is unusable we learn it
    # before spending thirty more.
    try:
        wei, _ = rpc.gas_price_wei("hyperevm")
        readings["gas_price"] = wei / 1e9
    except (rpc.RpcUnavailable, rpc.RpcShapeError) as exc:
        return Collection(errors=[f"hyperevm.gas_price: not collected — {exc}"])

    try:
        latest, _ = rpc.block_number("hyperevm")
        readings["latest_block"] = latest
    except (rpc.RpcUnavailable, rpc.RpcShapeError) as exc:
        out.errors.append(f"hyperevm.latest_block: not collected — {exc}")
        return _harvest_partial(
            "hyperevm", readings, now, out,
            {"latest_block", "block_tx_count_small", "block_tx_count_big"},
        )

    by_kind: dict[str, list[int]] = {"small": [], "big": []}
    unclassified = 0
    missed = 0
    scanned = 0

    for number in range(latest, latest - MAX_SCAN_BLOCKS, -1):
        if (
            len(by_kind["small"]) >= MIN_SMALL_BLOCKS
            and len(by_kind["big"]) >= MIN_BIG_BLOCKS
        ):
            break
        scanned += 1
        try:
            block, _ = rpc.get_block("hyperevm", number)
        except (rpc.RpcUnavailable, rpc.RpcShapeError):
            # One unreadable block narrows the sample; it does not invalidate it.
            missed += 1
            continue
        kind = classify_block(block.gas_limit)
        if kind is None:
            unclassified += 1
            continue
        by_kind[kind].append(block.transaction_count)

    explained: set[str] = set()
    for kind, counts in by_kind.items():
        metric = f"block_tx_count_{kind}"
        if counts:
            readings[metric] = sum(counts) / len(counts)
        else:
            # Reported, never zero. "No big block appeared in the scan" and "big
            # blocks carried no transactions" are opposite facts, and only one of
            # them is reassuring.
            explained.add(metric)
            out.errors.append(
                f"hyperevm.{metric}: not collected — no {kind} block appeared in "
                f"{scanned} block(s) scanned back from the head"
            )

    if unclassified:
        out.errors.append(
            f"hyperevm: {unclassified} block(s) had a gas limit matching neither "
            f"known block type and were not filed into either series"
        )
    if missed:
        out.errors.append(f"hyperevm: {missed} block(s) could not be read")

    chain_level = _harvest_partial("hyperevm", readings, now, out, explained)
    contract_level = collect_contracts("hyperevm", now)
    return Collection(
        observations=chain_level.observations + contract_level.observations,
        errors=chain_level.errors + contract_level.errors,
    )


def collect_contracts(protocol: str, observed_at=None) -> Collection:
    """Read state from every whitelisted contract for this protocol.

    Each contract's identity is confirmed against the chain before any of its
    readings are kept. A registry entry that has stopped matching — a mistyped
    address, a redeployment, a proxy pointed elsewhere — produces a real number
    from a real chain read at the highest reliability tier, about the wrong
    thing. Verification is what makes that loud instead of plausible.
    """
    from src.blockchain import contracts

    now = observed_at or _now()
    out = Collection()

    for spec in contracts.for_protocol(protocol):
        try:
            contracts.verify(spec)
        except contracts.ContractReadError as exc:
            out.errors.append(f"{protocol}.{spec.symbol}: not collected — {exc}")
            continue

        readings: dict[str, object] = {}
        try:
            if spec.kind is contracts.ContractKind.WRAPPED_NATIVE:
                # Ratio, supply and backing come from one block-guarded read, so
                # the three series describe the same moment.
                ratio, supply, backing = contracts.backing_ratio(spec)
                readings["wrapper_backing_ratio"] = ratio
                readings["token_total_supply"] = supply
                readings["wrapper_native_backing"] = backing
            else:
                readings["token_total_supply"] = contracts.total_supply(spec)
        except contracts.ContractReadError as exc:
            out.errors.append(f"{protocol}.{spec.symbol}: not collected — {exc}")
            continue

        harvested = _harvest(
            protocol, readings, spec.subject, now, now, SubjectKind.CONTRACT
        )
        out.observations.extend(harvested.observations)
        # `_harvest` reports every registered metric this contract did not
        # produce, which for a contract registry is most of them — a plain token
        # has no backing ratio. Only genuine rejections are worth surfacing.
        out.errors.extend(e for e in harvested.errors if "not present" not in e)

    return out


def _harvest_partial(
    protocol: str,
    readings: dict[str, object],
    now,
    collected: Collection,
    explained: set[str] | None = None,
) -> Collection:
    """Map whatever was read onto the registry, keeping errors already recorded.

    A registered metric this run did not produce is reported by `_harvest` as
    absent — the registry defines what should exist, so a silent omission becomes
    a stated gap. `explained` suppresses that second notice for metrics the
    collector has already accounted for, so one gap is reported once with the
    specific reason rather than twice with a generic one.
    """
    explained = explained or set()
    harvested = _harvest(protocol, readings, "", now, now, SubjectKind.CHAIN)
    extra = [
        e for e in harvested.errors
        if not any(e.startswith(f"{protocol}.{m}:") for m in explained)
    ]
    return Collection(
        observations=harvested.observations, errors=collected.errors + extra
    )


# Keyed by `Protocol.live_tool`, mirroring the live-data registry in
# `graph/nodes.py`. Onboarding a protocol's collector is a registry entry.
COLLECTORS: dict[str, Callable[..., Collection]] = {
    "hyperliquid": collect_hyperliquid,
    "hyperevm": collect_hyperevm,
}

# Markets collected on the scheduled run. Deliberately short: §9 says start with
# a limited number of metrics rather than trying to cover every chain at once,
# and every extra market is another series that must be kept current to stay
# useful.
DEFAULT_MARKETS = ("BTC", "ETH", "HYPE")


def needs_subject(protocol_key: str) -> bool:
    """True when this protocol's metrics describe a market rather than a chain."""
    from src.blockchain.features import SubjectKind

    return any(
        s.subject_kind is SubjectKind.MARKET for s in specs_for(protocol_key)
    )


def collect(protocol_key: str, subject: str = "") -> Collection:
    """Collect for one whitelisted protocol, or explain why nothing was collected."""
    if not is_known(protocol_key):
        return Collection(errors=[f"{protocol_key}: not a whitelisted protocol"])

    protocol = get_protocol(protocol_key)
    collector = COLLECTORS.get(protocol.live_tool or "")
    if collector is None:
        return Collection(
            errors=[
                f"{protocol.name}: no on-chain data source is wired up, so no "
                f"metrics exist to baseline — the absence of an anomaly here "
                f"means nothing was measured"
            ]
        )

    # Per-market metrics are meaningless without a market. Saying so beats
    # defaulting to some arbitrary ticker, which would file readings about BTC
    # under a question that never mentioned it.
    if needs_subject(protocol_key) and not subject:
        return Collection(
            errors=[
                f"{protocol.name}: metrics are per-market and no market was "
                f"named, so nothing was collected"
            ]
        )
    return collector(subject)
