"""Metric registry, validation, and the gauge/cumulative distinction.

The distinction is not bookkeeping. A z-score on a monotonic counter reports a
CRITICAL anomaly every day of a completely healthy chain, for the sole reason
that time passed — so which metrics get differenced is a correctness question,
and these tests are where it is settled.
"""

import pytest

from src.blockchain.features import (
    InvalidObservation,
    MetricKind,
    SubjectKind,
    all_specs,
    get_spec,
    is_registered,
    prepare_for_scoring,
    rate_series,
    specs_for,
    validate,
)

GAUGE = get_spec("hyperevm", "gas_price")
FUNDING = get_spec("hyperliquid", "funding_hourly_pct")

# Block height is the one registered cumulative metric, and differencing it is
# the point: its rate is blocks produced per interval, i.e. chain liveness.
COUNTER = get_spec("hyperevm", "latest_block")


# --- the registry -------------------------------------------------------


def test_every_registered_metric_belongs_to_a_whitelisted_protocol():
    """A metric for a protocol the registry does not know could never be
    collected, and would sit in the store as a gap that looks like a value."""
    from src.protocols import is_known

    assert all(is_known(s.protocol) for s in all_specs())


def test_a_protocol_with_no_on_chain_reader_registers_no_metrics():
    """Ethena is whitelisted and has no live tool. An empty list here is what
    makes the agent able to say 'nothing was measured' rather than 'nothing was
    found'."""
    assert specs_for("ethena") == ()


def test_lookups_of_unknown_metrics_fail_loudly():
    with pytest.raises(KeyError, match="no metric"):
        get_spec("hyperevm", "invented_metric")
    assert not is_registered("hyperevm", "invented_metric")


def test_metrics_are_namespaced_by_protocol():
    """Two chains can both have a metric called `gas_price` without collision."""
    assert is_registered("hyperevm", "gas_price")
    assert not is_registered("hyperliquid", "gas_price")


def test_market_metrics_are_distinguished_from_chain_metrics():
    assert FUNDING.subject_kind is SubjectKind.MARKET
    assert GAUGE.subject_kind is SubjectKind.CHAIN


# --- how a metric is scored ---------------------------------------------


def test_a_gauge_is_scored_under_its_own_name():
    assert GAUGE.kind is MetricKind.GAUGE
    assert GAUGE.scored_as == "gas_price"


def test_a_counter_is_scored_under_its_rate():
    """A report showing a z-score beside a counter's name would be claiming
    something about the counter that was measured about its rate of change."""
    assert COUNTER.is_cumulative
    assert COUNTER.scored_as == "latest_block_rate"


# --- validation ---------------------------------------------------------


def test_a_good_reading_passes_through_as_a_float():
    assert validate(GAUGE, "12") == 12.0


@pytest.mark.parametrize("bad", [None, "n/a", "", object()])
def test_non_numeric_readings_are_rejected(bad):
    with pytest.raises(InvalidObservation):
        validate(GAUGE, bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_are_rejected(bad):
    """Either would poison every statistic computed from the series it joins."""
    with pytest.raises(InvalidObservation):
        validate(GAUGE, bad)


def test_a_boolean_is_rejected_rather_than_read_as_one():
    """`bool` is an `int` subclass, so a collector that mapped the wrong field
    would store 1.0 — a completely plausible-looking gas price."""
    with pytest.raises(InvalidObservation, match="boolean"):
        validate(GAUGE, True)


def test_negative_counts_are_rejected():
    with pytest.raises(InvalidObservation, match="negative"):
        validate(GAUGE, -5)


def test_funding_may_be_negative_because_it_genuinely_is():
    """Shorts pay longs when funding is negative. Rejecting it would discard the
    readings that matter most."""
    assert validate(FUNDING, -0.0031) == pytest.approx(-0.0031)


def test_zero_is_a_valid_reading():
    assert validate(GAUGE, 0) == 0.0


# --- differencing -------------------------------------------------------


def test_a_counter_becomes_its_increments():
    assert rate_series([100, 110, 125, 130]) == [10, 15, 5]


def test_differencing_yields_one_fewer_value():
    assert len(rate_series([1, 2, 3, 4, 5])) == 4


def test_a_single_reading_yields_no_rate():
    assert rate_series([100]) == []


def test_a_counter_that_went_backwards_is_dropped_not_recorded():
    """A counter decreasing means the source reset, re-indexed, or served a stale
    response. A large negative 'rate' would score as a dramatic anomaly caused
    entirely by the data pipeline."""
    assert rate_series([100, 110, 40, 55]) == [10, 15]


# --- what actually gets handed to the risk engine -----------------------


def test_a_gauge_is_handed_over_unchanged():
    current, history = prepare_for_scoring(GAUGE, [1.0, 2.0, 3.0], 4.0)
    assert current == 4.0
    assert history == [1.0, 2.0, 3.0]


def test_a_counter_is_handed_over_as_increments():
    current, history = prepare_for_scoring(COUNTER, [100, 110, 125], 130)
    assert current == 5      # latest increment
    assert history == [10, 15]  # previous increments


def test_a_counter_with_one_prior_reading_gives_a_rate_and_no_baseline():
    current, history = prepare_for_scoring(COUNTER, [100], 130)
    assert current == 30
    assert history == []


def test_a_counter_with_no_history_cannot_be_scored_at_all():
    """One observation of a counter says nothing about its rate, and inventing a
    first rate from a single point is a number with no measurement behind it."""
    assert prepare_for_scoring(COUNTER, [], 100) is None


def _counter_history(seed=11, days=30, per_day=1000.0, noise=60.0):
    """A realistic monotonic counter: ~`per_day` a day with a little variation."""
    import random

    rng = random.Random(seed)
    total, series = 100_000.0, []
    for _ in range(days):
        total += per_day + rng.gauss(0, noise)
        series.append(total)
    return series


def test_a_raw_counter_is_blind_to_a_chain_that_has_stopped():
    """The failure this whole distinction exists to stop, and it is a MISS rather
    than a false alarm.

    A counter sits at its all-time high by construction, so it is always about
    the same distance above its own trailing mean. When a chain stalls the
    counter does not fall — it simply stops climbing, and still reads normal.
    A catastrophic outcome dressed as an all-clear.
    """
    from src.risk.signals import Severity, assess_metric

    counter = _counter_history()
    stalled = counter[-1] + 5  # essentially no activity for a whole day

    raw = assess_metric("counter", stalled, counter)
    assert raw.severity is Severity.NORMAL, "the trap: a stalled chain looks fine"

    current, history = prepare_for_scoring(COUNTER, counter, stalled)
    corrected = assess_metric("counter_rate", current, history)
    assert corrected.severity is Severity.CRITICAL
    assert corrected.z < 0  # and it correctly reads as a collapse, not a surge


def test_a_raw_counter_also_misses_a_tripling_of_activity():
    from src.risk.signals import Severity, assess_metric

    counter = _counter_history()
    surge = counter[-1] + 3000

    assert assess_metric("counter", surge, counter).severity is Severity.NORMAL

    current, history = prepare_for_scoring(COUNTER, counter, surge)
    assert assess_metric("rate", current, history).severity is Severity.CRITICAL


def test_a_raw_counters_score_barely_moves_across_wildly_different_days():
    """Why it is blind: the score carries almost no information about behaviour.
    Three scenarios spanning three orders of magnitude land within ~1 sigma."""
    from src.risk.signals import assess_metric

    counter = _counter_history()
    scores = [
        abs(assess_metric("c", counter[-1] + delta, counter).z)
        for delta in (5, 1000, 10_000)
    ]
    assert max(scores) - min(scores) < 1.5


def test_the_rate_correctly_reports_an_ordinary_day_as_ordinary():
    """The correction must not simply make everything critical."""
    from src.risk.signals import Severity, assess_metric

    counter = _counter_history()
    current, history = prepare_for_scoring(COUNTER, counter, counter[-1] + 1000)
    assert assess_metric("rate", current, history).severity is Severity.NORMAL


# --- the dual-block split ------------------------------------------------


def test_the_two_block_kinds_are_separated_at_the_measured_limits():
    """Measured on 2026-08-20: gas limits are exactly 3,000,000 and 30,000,000.
    The threshold is a midpoint so a parameter tweak reclassifies cleanly rather
    than making every block unrecognisable."""
    from src.blockchain.features import classify_block

    assert classify_block(3_000_000) == "small"
    assert classify_block(30_000_000) == "big"


def test_a_gas_limit_matching_neither_kind_is_refused():
    """A misfiled reading corrupts a baseline permanently; a missing one only
    leaves a gap."""
    from src.blockchain.features import classify_block

    assert classify_block(1) is None
    assert classify_block(5_000_000_000) is None


def test_both_block_series_are_registered_separately():
    """Sampled together they are a mixture of two populations, and the severity
    thresholds were calibrated against unimodal data."""
    keys = {s.key for s in specs_for("hyperevm")}
    assert {"block_tx_count_small", "block_tx_count_big"} <= keys
