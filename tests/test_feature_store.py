"""The historical feature store.

Every test runs against a temp database. This is the piece the risk engine has
been waiting for: anomaly detection is a comparison against a baseline, and a
baseline is history.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.blockchain.store import (
    Observation,
    coverage,
    prior_history,
    record,
    series_length,
)

T0 = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "features.sqlite"


def _obs(value, at, metric="gas_price", protocol="hyperevm", subject=""):
    return Observation(
        protocol=protocol,
        metric=metric,
        subject=subject,
        value=value,
        observed_at=at,
        collected_at=at,
    )


def _hourly(values, start=T0, **kw):
    return [_obs(v, start + timedelta(hours=i), **kw) for i, v in enumerate(values)]


# --- writing ------------------------------------------------------------


def test_readings_are_stored_and_counted(db):
    assert record(_hourly([1.0, 2.0, 3.0]), path=db) == 3
    assert series_length("hyperevm", "gas_price", path=db) == 3


def test_recording_the_same_observation_twice_stores_it_once(db):
    """Collection will be scheduled, and schedules overlap and retry. A duplicate
    is not harmless: it doubles that value's weight in the baseline it forms part
    of."""
    batch = _hourly([1.0, 2.0])
    assert record(batch, path=db) == 2
    assert record(batch, path=db) == 0
    assert series_length("hyperevm", "gas_price", path=db) == 2


def test_a_conflicting_later_reading_does_not_overwrite_history(db):
    """A source reporting a different value for a moment already recorded is an
    inconsistency. Quietly rewriting history to match the newest answer would
    erase the evidence of it."""
    record([_obs(10.0, T0)], path=db)
    record([_obs(999.0, T0)], path=db)
    assert prior_history("hyperevm", "gas_price", before=T0 + timedelta(hours=1), path=db) == [10.0]


def test_recording_nothing_is_a_no_op(db):
    assert record([], path=db) == 0


def test_naive_timestamps_are_refused(db):
    """The column is TEXT and ordering is lexical, so a naive value would break
    series order in a way no test of the statistics would catch."""
    with pytest.raises(ValueError, match="timezone-aware"):
        record([_obs(1.0, datetime(2026, 8, 20, 12, 0))], path=db)


def test_timestamps_are_normalised_so_lexical_order_is_chronological(db):
    """A mixture of offsets would silently sort wrongly."""
    other_zone = timezone(timedelta(hours=9))
    record([_obs(1.0, T0), _obs(2.0, (T0 + timedelta(hours=1)).astimezone(other_zone))], path=db)
    assert prior_history("hyperevm", "gas_price", before=T0 + timedelta(days=1), path=db) == [1.0, 2.0]


# --- reading ------------------------------------------------------------


def test_history_comes_back_oldest_first(db):
    """The statistics expect a series in time order."""
    record(_hourly([5.0, 6.0, 7.0]), path=db)
    assert prior_history("hyperevm", "gas_price", before=T0 + timedelta(days=1), path=db) == [5.0, 6.0, 7.0]


def test_prior_history_excludes_the_observation_being_tested(db):
    """The API enforces the risk engine's core rule. An outlier included in its
    own baseline inflates the standard deviation and suppresses its own score, so
    the query that callers reach for cannot include the current reading."""
    record(_hourly([1.0, 2.0, 3.0]), path=db)
    latest = T0 + timedelta(hours=2)
    assert prior_history("hyperevm", "gas_price", before=latest, path=db) == [1.0, 2.0]


def test_series_are_isolated_by_protocol(db):
    record(_hourly([1.0, 2.0]), path=db)
    record(_hourly([9.0], protocol="hyperliquid", metric="mark_price", subject="ETH"), path=db)
    assert prior_history("hyperevm", "gas_price", before=T0 + timedelta(days=1), path=db) == [1.0, 2.0]


def test_series_are_isolated_by_subject(db):
    """Two markets share a metric name and must not share a baseline — scoring
    ETH's price against BTC's history would be meaningless."""
    record(_hourly([3000.0, 3100.0], metric="mark_price", protocol="hyperliquid", subject="ETH"), path=db)
    record(_hourly([60000.0], metric="mark_price", protocol="hyperliquid", subject="BTC"), path=db)
    later = T0 + timedelta(days=1)
    assert prior_history("hyperliquid", "mark_price", "ETH", before=later, path=db) == [3000.0, 3100.0]
    assert prior_history("hyperliquid", "mark_price", "BTC", before=later, path=db) == [60000.0]


def test_history_is_capped_and_keeps_the_most_recent(db):
    """A metric that changed regime six months ago should not still be compared
    against its old behaviour."""
    record(_hourly([float(i) for i in range(50)]), path=db)
    recent = prior_history("hyperevm", "gas_price", before=T0 + timedelta(days=9), limit=10, path=db)
    assert recent == [float(i) for i in range(40, 50)]


def test_an_empty_store_has_no_history(db):
    assert prior_history("hyperevm", "gas_price", path=db) == []
    assert series_length("hyperevm", "gas_price", path=db) == 0


# --- coverage -----------------------------------------------------------


def test_coverage_reports_one_row_per_series(db):
    record(_hourly([1.0, 2.0]), path=db)
    record(_hourly([9.0], metric="mark_price", protocol="hyperliquid", subject="ETH"), path=db)

    rows = coverage(path=db)
    assert len(rows) == 2
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["gas_price"]["observations"] == 2
    assert by_metric["mark_price"]["subject"] == "ETH"


def test_coverage_of_an_empty_store_is_empty(db):
    assert coverage(path=db) == []


# --- the integration that matters --------------------------------------


def test_a_stored_series_is_directly_scoreable_by_the_risk_engine(db):
    """The whole point of C2: history in, baseline out, anomaly detected."""
    from src.risk.signals import Severity, assess_metric

    quiet = [2.3, 2.1, 2.5, 2.4, 2.2, 2.0, 2.6, 2.3, 2.4, 2.2]
    record(_hourly(quiet), path=db)

    spike_at = T0 + timedelta(hours=len(quiet))
    record([_obs(12.5, spike_at)], path=db)

    history = prior_history("hyperevm", "gas_price", before=spike_at, path=db)
    signal = assess_metric("gas_price", 12.5, history)

    assert len(history) == len(quiet)
    assert signal.severity is Severity.CRITICAL
    assert signal.anomaly


def test_too_little_stored_history_yields_unknown_not_normal(db):
    """A store that has only just started collecting must not report calm."""
    from src.risk.signals import Severity, assess_metric

    record(_hourly([2.0, 2.1, 2.2]), path=db)
    history = prior_history("hyperevm", "gas_price", before=T0 + timedelta(days=1), path=db)

    signal = assess_metric("gas_price", 99.0, history)
    assert signal.severity is Severity.UNKNOWN
    assert not signal.anomaly
