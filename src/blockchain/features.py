"""What we measure on-chain, and how each measurement must be handled.

§9.2 lists candidate features. The registry below is deliberately shorter than
that list: it contains only metrics the existing read-only tools can actually
produce today. A registry entry is a promise that a number can be collected, and
a metric nobody collects would sit in the feature store forever as a gap that
looks like a value nobody has looked at recently.

The distinction this module exists to enforce is **gauge versus cumulative**.

A gauge is a reading: gas price, open interest, today's transaction count. It
goes up and down, and asking "is this unusual against its own history" is
meaningful.

A cumulative counter only ever increases: total transactions since genesis, total
addresses ever seen. Scoring one directly does not produce noisy alerts — it
produces **near-total blindness**, which is worse, because it looks like
everything is fine.

A counter sits at its all-time high by construction, so it is always roughly the
same distance above its own trailing mean. Measured on 30 days of ~1000 tx/day:

    next day            raw counter          as a rate
    normal   (+1000)    z=+1.76  normal      z=  -0.12  normal
    3x surge (+3000)    z=+1.98  normal      z= +34.63  critical
    10x      (+10000)   z=+2.77  elevated    z=+156.24  critical
    stalls   (+5)       z=+1.64  normal      z= -17.40  critical

The raw z-score barely moves across scenarios that differ by three orders of
magnitude, and it is pinned just under the 2σ line, so it reads NORMAL almost
whatever happens. The last row is the one that matters: **a chain that stops
entirely scores as normal**, because the counter is still near its peak — it
simply stopped climbing. That is a catastrophic miss dressed as an all-clear, and
it is exactly the outcome this system is built to prevent.

Cumulative metrics are therefore differenced into rates before scoring, and the
rate is what carries a baseline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, ConfigDict

from src.risk.invariants import Bound, Invariant


class MetricKind(str, Enum):
    GAUGE = "gauge"            # a reading; score it directly
    CUMULATIVE = "cumulative"  # a monotonic counter; difference it first


class SubjectKind(str, Enum):
    """Whether a metric describes a whole chain or one market within it."""

    CHAIN = "chain"        # subject is empty
    MARKET = "market"      # subject is a ticker, e.g. "ETH"
    # Subject is a contract symbol. Supplied by the contract registry rather
    # than by the caller, which is why `needs_subject` keys on MARKET alone.
    CONTRACT = "contract"


class MetricSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    protocol: str
    kind: MetricKind
    subject_kind: SubjectKind
    unit: str
    description: str
    # Which evidence kind a reading of this metric becomes, and therefore how
    # fast it goes stale. Contract state is a fact about one block; a gas price
    # describes a period. Defaulting them to the same decay would make the
    # shortest-lived readings the most overconfident.
    evidence_kind: str = "on_chain_metric"
    # A property this metric must satisfy regardless of its history. Declared on
    # the spec rather than passed at the call site so that every consumer of a
    # metric gets the same check — an invariant enforced in one code path and
    # forgotten in another is worse than none, because the passing path implies
    # the property was verified.
    invariant: Invariant | None = None

    @property
    def is_cumulative(self) -> bool:
        return self.kind is MetricKind.CUMULATIVE

    @property
    def scored_as(self) -> str:
        """The metric name the risk engine actually scores.

        A cumulative counter is scored through its derived rate, and the name says
        so — a report that showed a z-score beside "total_transactions" would be
        claiming something about the counter that was in fact measured about its
        rate of change.
        """
        return f"{self.key}_rate" if self.is_cumulative else self.key


_SPECS: tuple[MetricSpec, ...] = (
    # --- Hyperliquid: per-market readings from the public /info endpoint ---
    MetricSpec(
        key="mark_price",
        protocol="hyperliquid",
        kind=MetricKind.GAUGE,
        subject_kind=SubjectKind.MARKET,
        unit="usd",
        description="Perp mark price",
    ),
    MetricSpec(
        key="funding_hourly_pct",
        protocol="hyperliquid",
        kind=MetricKind.GAUGE,
        subject_kind=SubjectKind.MARKET,
        unit="percent_per_hour",
        description="Hourly funding rate",
    ),
    MetricSpec(
        key="open_interest",
        protocol="hyperliquid",
        kind=MetricKind.GAUGE,
        subject_kind=SubjectKind.MARKET,
        unit="contracts",
        description="Open interest",
    ),
    MetricSpec(
        key="day_volume_usd",
        protocol="hyperliquid",
        kind=MetricKind.GAUGE,
        subject_kind=SubjectKind.MARKET,
        unit="usd",
        description="Rolling 24h notional volume",
    ),
    # --- HyperEVM: chain-level readings over JSON-RPC ---
    #
    # The cumulative counters that used to live here (total transactions, total
    # addresses) are gone with the explorer that served them, and are not missed:
    # a counter sits at its all-time high by construction, so a chain that stops
    # entirely still reads normal on one. See the module docstring.
    MetricSpec(
        key="gas_price",
        protocol="hyperevm",
        kind=MetricKind.GAUGE,
        subject_kind=SubjectKind.CHAIN,
        unit="gwei",
        description="Gas price",
    ),
    # Block height is a counter, and here that is the point rather than a
    # problem. Scored raw it would be blind like any other; differenced, its rate
    # is blocks produced per collection interval — which is chain liveness. A
    # halt or a slowdown shows up immediately and unambiguously, and it is the
    # one thing a counter measures better than a gauge could.
    MetricSpec(
        key="latest_block",
        protocol="hyperevm",
        kind=MetricKind.CUMULATIVE,
        subject_kind=SubjectKind.CHAIN,
        unit="count",
        description="Block height",
    ),
    # Throughput is split by block type, and the split is not cosmetic.
    #
    # HyperEVM produces two interleaved kinds of block: small ones roughly every
    # second with a low gas limit, and big ones roughly every minute with a much
    # higher one. Sampled together, a transactions-per-block series is a MIXTURE
    # OF TWO POPULATIONS — its variance is dominated by which kind of block the
    # sample happened to land on rather than by chain activity.
    #
    # That is worse than noisy. The severity thresholds were calibrated against
    # unimodal data, and the robust median/MAD score handles skew and outlier
    # contamination but does nothing about bimodality: the series would carry
    # large, stable variance and simply never fire. It is the same class of error
    # as scoring a cumulative counter — assuming a number means one thing when
    # the process generating it makes it mean two.
    MetricSpec(
        key="block_tx_count_small",
        protocol="hyperevm",
        kind=MetricKind.GAUGE,
        subject_kind=SubjectKind.CHAIN,
        unit="count",
        description="Transactions per small block",
    ),
    MetricSpec(
        key="block_tx_count_big",
        protocol="hyperevm",
        kind=MetricKind.GAUGE,
        subject_kind=SubjectKind.CHAIN,
        unit="count",
        description="Transactions per big block",
    ),
    # --- contract state, read via eth_call against the whitelisted registry ---
    #
    # Subject is the contract's symbol, so each contract gets its own series.
    MetricSpec(
        key="token_total_supply",
        protocol="hyperevm",
        kind=MetricKind.GAUGE,
        subject_kind=SubjectKind.CONTRACT,
        unit="tokens",
        description="Circulating supply",
        evidence_kind="chain_state",
    ),
    MetricSpec(
        key="wrapper_native_backing",
        protocol="hyperevm",
        kind=MetricKind.GAUGE,
        subject_kind=SubjectKind.CONTRACT,
        unit="native",
        description="Native coin locked in the wrapper",
        evidence_kind="chain_state",
    ),
    # The invariant, and the reason this is worth collecting at all. A wrapper
    # should hold exactly one unit of native coin per wrapped token, so this
    # series sits at 1.0 and any sustained deviation is under- or
    # over-collateralisation — a real finding rather than a statistic about
    # activity. Measured 1.000000 on 2026-08-21.
    #
    # It is declared rather than learned because a flat series is unscoreable by
    # construction: nine identical readings give a baseline with zero spread, so
    # every z-score is undefined and the metric reports UNKNOWN however far it
    # later moves. The statistical path is blind here precisely because the
    # protocol is working.
    MetricSpec(
        key="wrapper_backing_ratio",
        protocol="hyperevm",
        kind=MetricKind.GAUGE,
        subject_kind=SubjectKind.CONTRACT,
        unit="ratio",
        description="Native backing per wrapped token",
        evidence_kind="chain_state",
        invariant=Invariant(
            target=1.0,
            # AT_LEAST, not EQUALS, because the two directions are not the same
            # event. Below 1.0 the wrapper has issued tokens it cannot redeem,
            # which is insolvency. Above 1.0 someone has sent native coin to the
            # contract without minting — a donation or a mistake, and no holder is
            # worse off. Scoring both as "deviation from 1.0" would raise a
            # solvency alarm over a stray transfer.
            bound=Bound.AT_LEAST,
            rationale=(
                "A wrapper mints one token per unit of native coin deposited and "
                "burns one per unit withdrawn, so its native balance must cover "
                "its total supply. A shortfall means tokens exist that cannot be "
                "redeemed."
            ),
        ),
    ),
)

# Gas limit above which a block is the big kind. A midpoint rather than an exact
# match on either published limit, so a parameter tweak reclassifies cleanly
# instead of making every block unrecognised.
BIG_BLOCK_GAS_LIMIT = 10_000_000

# Outside this range a block matches neither known kind. Such a block is reported
# as unclassifiable rather than forced into a series, because a misfiled reading
# corrupts a baseline permanently while a missing one only leaves a gap.
PLAUSIBLE_GAS_LIMITS = (100_000, 200_000_000)


def classify_block(gas_limit: int) -> str | None:
    """"small" | "big", or None when the limit matches neither known kind."""
    low, high = PLAUSIBLE_GAS_LIMITS
    if not low <= gas_limit <= high:
        return None
    return "big" if gas_limit >= BIG_BLOCK_GAS_LIMIT else "small"

_BY_KEY: dict[tuple[str, str], MetricSpec] = {(s.protocol, s.key): s for s in _SPECS}


def all_specs() -> tuple[MetricSpec, ...]:
    return _SPECS


def specs_for(protocol: str) -> tuple[MetricSpec, ...]:
    """Every metric collectable for one protocol. Empty when none are."""
    return tuple(s for s in _SPECS if s.protocol == protocol)


def get_spec(protocol: str, key: str) -> MetricSpec:
    try:
        return _BY_KEY[(protocol, key)]
    except KeyError:
        raise KeyError(f"no metric {key!r} registered for protocol {protocol!r}") from None


def is_registered(protocol: str, key: str) -> bool:
    return (protocol, key) in _BY_KEY


# --- validation ---------------------------------------------------------


class InvalidObservation(ValueError):
    """A reading that must not enter the feature store."""


def validate(spec: MetricSpec, value: object) -> float:
    """Coerce and sanity-check one reading before it is persisted.

    §9's architecture puts validation between collection and feature
    engineering, and the reason is that a bad reading is worse than a missing
    one: a missing reading widens the gap in a baseline, while a bad one silently
    moves the baseline and suppresses the very anomaly it should have raised. A
    single spurious zero in a volume series drags the mean down and inflates σ.
    """
    if value is None:
        raise InvalidObservation(f"{spec.key}: value is missing")
    if isinstance(value, bool):
        # bool is an int subclass; a True that reached here means a collector
        # mapped the wrong field, and 1.0 would look like a plausible reading.
        raise InvalidObservation(f"{spec.key}: value is a boolean, not a measurement")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InvalidObservation(f"{spec.key}: {value!r} is not numeric") from None

    if math.isnan(number) or math.isinf(number):
        raise InvalidObservation(f"{spec.key}: value is {number}")
    # Every registered metric is a count, a price, a volume or a size. None of
    # them can be negative; funding, which genuinely can, is a percentage and is
    # excluded here.
    if number < 0 and spec.unit != "percent_per_hour":
        raise InvalidObservation(f"{spec.key}: negative value {number} for unit {spec.unit}")
    return number


# --- feature engineering ------------------------------------------------


def rate_series(values: Sequence[float]) -> list[float]:
    """Turn a cumulative counter into per-interval increments.

    Returns one fewer value than it was given — a rate needs two readings. Drops
    negative increments rather than recording them: a counter that went backwards
    means the source reset, re-indexed, or served a stale response, and a large
    negative "rate" would be scored as a dramatic anomaly caused entirely by the
    data pipeline.
    """
    return [
        later - earlier
        for earlier, later in zip(values, values[1:])
        if later >= earlier
    ]


def prepare_for_scoring(
    spec: MetricSpec, history: Sequence[float], current: float
) -> tuple[float, list[float]] | None:
    """Produce the (current, baseline) pair the risk engine should score.

    For a gauge this is the reading and its history, unchanged. For a cumulative
    counter it is the latest increment scored against previous increments.

    Returns `None` when a cumulative metric has too few readings to difference at
    all — one observation of a counter says nothing about its rate, and inventing
    a first rate from a single point would be a number with no measurement behind
    it.
    """
    if not spec.is_cumulative:
        return current, list(history)

    if not history:
        return None
    increments = rate_series([*history, current])
    if not increments:
        return None
    return increments[-1], increments[:-1]
