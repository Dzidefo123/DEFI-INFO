"""§9's Blockchain Agent and its collectors.

The collector is injected, so nothing here touches a network. What is exercised
for real: validation, storage, baseline pairing, and the honest reporting of
gaps — which is most of what this agent does.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.blockchain import (
    MetricSeries,
    evidence_from_observation,
    investigate,
)
from src.blockchain.collectors import Collection, _harvest, collect, needs_subject
from src.blockchain.features import SubjectKind
from src.blockchain.store import Observation, record, series_length
from src.evidence.models import AgentName, EvidenceKind, SourceTier
from src.risk.statistics import MIN_BASELINE_N

T0 = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "features.sqlite"


def _obs(metric, value, at=T0, protocol="hyperevm", subject=""):
    return Observation(
        protocol=protocol, metric=metric, subject=subject,
        value=value, observed_at=at, collected_at=at,
    )


def _collector(collection):
    return lambda key, subject="": collection


# --- mapping a tool response onto the registry -------------------------


def test_harvest_takes_only_the_registered_metrics():
    """Driven by the registry, not the response. Storing whatever came back means
    the store's schema is whatever the upstream API last sent, which is not a
    schema."""
    result = _harvest(
        "hyperevm",
        {
            "gas_price": 1.2,
            "latest_block": 90_000,
            "block_tx_count_small": 6.0,
            "block_tx_count_big": 40.0,
            "gas_used": 1.0,            # real field, not a registered metric
        },
        "", T0, T0, SubjectKind.CHAIN,
    )
    assert {o.metric for o in result.observations} == {
        "gas_price", "latest_block", "block_tx_count_small", "block_tx_count_big"
    }
    # Scoped to chain-level metrics, so the contract series are not reported as
    # missing from a run that was never collecting them.
    assert result.errors == []


def test_a_missing_registered_metric_is_reported_not_skipped():
    result = _harvest("hyperevm", {"gas_price": 1.2}, "", T0, T0)
    assert len(result.observations) == 1
    assert any("block_tx_count_big" in e and "not present" in e for e in result.errors)


def test_an_invalid_reading_is_rejected_and_the_rest_still_land():
    """A bad reading is worse than a missing one: it silently moves the baseline
    and suppresses the anomaly it should have raised."""
    result = _harvest(
        "hyperevm",
        {"gas_price": -5, "latest_block": 90_000,
         "block_tx_count_small": 6.0, "block_tx_count_big": 40.0},
        "", T0, T0,
    )
    assert {o.metric for o in result.observations} == {
        "latest_block", "block_tx_count_small", "block_tx_count_big"
    }
    assert any("rejected" in e for e in result.errors)


# --- which protocols can be collected at all ---------------------------


def test_a_protocol_with_no_on_chain_reader_says_so_rather_than_returning_nothing():
    """Ethena is whitelisted with no live tool. Silently skipping it produces an
    investigation that reports no anomalies — indistinguishable from one that
    found none."""
    result = collect("ethena")
    assert result.observations == []
    assert "no on-chain data source" in result.errors[0]
    assert "nothing was measured" in result.errors[0]


def test_an_unknown_protocol_is_refused():
    assert "not a whitelisted protocol" in collect("aave").errors[0]


def test_per_market_metrics_without_a_market_are_refused():
    """Defaulting to some ticker would file readings about BTC under a question
    that never mentioned it."""
    result = collect("hyperliquid")
    assert result.observations == []
    assert "no market was named" in result.errors[0]


def test_the_registry_decides_which_protocols_need_a_market():
    assert needs_subject("hyperliquid")
    assert not needs_subject("hyperevm")
    assert not needs_subject("ethena")


# --- evidence -----------------------------------------------------------


def test_a_reading_becomes_on_chain_evidence():
    ev = evidence_from_observation(_obs("gas_price", 1.25))
    assert ev.kind is EvidenceKind.ON_CHAIN_METRIC
    assert ev.agent is AgentName.BLOCKCHAIN
    assert ev.source.tier is SourceTier.CHAIN
    assert ev.source.protocol == "hyperevm"
    assert "1.25" in ev.summary and "gwei" in ev.summary


def test_evidence_records_when_the_reading_was_TRUE_not_when_we_asked():
    """The distinction `Evidence` was built for, and what decides whether a
    six-hour-old funding rate is treated as current."""
    observed = T0 - timedelta(hours=6)
    ev = evidence_from_observation(
        Observation(protocol="hyperevm", metric="gas_price", value=1.0,
                    observed_at=observed, collected_at=T0)
    )
    assert ev.as_of == observed


def test_market_evidence_names_its_market():
    ev = evidence_from_observation(
        _obs("mark_price", 3000.0, protocol="hyperliquid", subject="ETH")
    )
    assert "ETH" in ev.summary
    assert ev.source.locator == "mark_price on ETH"


def test_the_same_reading_twice_is_one_piece_of_evidence():
    a = evidence_from_observation(_obs("gas_price", 1.25))
    b = evidence_from_observation(_obs("gas_price", 1.25))
    assert a.evidence_id == b.evidence_id


# --- the agent ----------------------------------------------------------


def test_no_protocol_in_scope_means_nothing_could_be_measured():
    out = investigate(protocols=[])
    assert out.evidence == () and out.series == ()
    assert "no protocol was identified" in out.limitations[0]


def test_the_agent_stores_what_it_collects(db):
    out = investigate(
        protocols=["hyperevm"],
        collector=_collector(Collection(observations=[_obs("gas_price", 1.2)])),
        path=db,
    )
    assert len(out.evidence) == 1
    assert series_length("hyperevm", "gas_price", path=db) == 1


def test_a_first_ever_reading_reports_that_it_cannot_be_judged(db):
    """The correct output of a newly-created feature store. A version that
    produced findings from one reading would be lying."""
    out = investigate(
        protocols=["hyperevm"],
        collector=_collector(Collection(observations=[_obs("gas_price", 1.2)])),
        path=db,
    )
    assert len(out.series) == 1
    assert not out.series[0].scoreable
    assert any("not enough history" in l for l in out.limitations)
    assert any(str(MIN_BASELINE_N) in l for l in out.limitations)


def test_once_enough_history_exists_the_series_becomes_scoreable(db):
    prior = [
        _obs("gas_price", 1.0 + i * 0.05, at=T0 + timedelta(hours=i))
        for i in range(MIN_BASELINE_N)
    ]
    record(prior, path=db)

    latest = _obs("gas_price", 9.9, at=T0 + timedelta(hours=MIN_BASELINE_N))
    out = investigate(
        protocols=["hyperevm"],
        collector=_collector(Collection(observations=[latest])),
        path=db,
    )
    series = out.series[0]
    assert series.scoreable
    assert len(series.history) == MIN_BASELINE_N
    assert series.current == 9.9
    assert not any("not enough history" in l for l in out.limitations)


def test_the_baseline_never_contains_the_reading_being_judged(db):
    """The rule the whole store API is shaped around."""
    record([_obs("gas_price", 1.0, at=T0)], path=db)
    latest = _obs("gas_price", 99.0, at=T0 + timedelta(hours=1))
    out = investigate(
        protocols=["hyperevm"],
        collector=_collector(Collection(observations=[latest])),
        path=db,
    )
    assert 99.0 not in out.series[0].history


def test_collector_errors_become_limitations(db):
    out = investigate(
        protocols=["hyperevm"],
        collector=_collector(Collection(errors=["hyperevm: explorer unreachable"])),
        path=db,
    )
    assert out.evidence == ()
    assert "explorer unreachable" in out.limitations[0]


def test_a_cumulative_metric_is_reported_under_its_rate(db):
    prior = [
        _obs("latest_block", 100_000 + i * 1000, at=T0 + timedelta(hours=i))
        for i in range(3)
    ]
    record(prior, path=db)
    latest = _obs("latest_block", 103_000, at=T0 + timedelta(hours=3))

    out = investigate(
        protocols=["hyperevm"],
        collector=_collector(Collection(observations=[latest])),
        path=db,
    )
    series = out.series[0]
    assert series.metric == "latest_block_rate"
    assert series.current == 1000  # the increment, not the counter


def test_a_first_ever_cumulative_reading_yields_no_rate_at_all(db):
    """One observation of a counter says nothing about its rate."""
    out = investigate(
        protocols=["hyperevm"],
        collector=_collector(Collection(observations=[_obs("latest_block", 100_000)])),
        path=db,
    )
    assert out.series == ()
    assert any("no rate of change can be derived" in l for l in out.limitations)
    assert len(out.evidence) == 1  # the reading is still recorded as evidence


def test_the_agent_makes_no_claims(db):
    """It gathers; the risk engine scores. Keeping that boundary is what stops
    'unusual' from meaning whatever the collecting code thought."""
    out = investigate(
        protocols=["hyperevm"],
        collector=_collector(Collection(observations=[_obs("gas_price", 1.2)])),
        path=db,
    )
    assert not hasattr(out, "claims")


def test_series_scoreability_is_decided_by_the_engines_own_minimum():
    short = MetricSeries(
        metric="m", protocol="p", current=1.0,
        history=tuple(range(MIN_BASELINE_N - 1)), observed_at=T0,
    )
    ok = MetricSeries(
        metric="m", protocol="p", current=1.0,
        history=tuple(range(MIN_BASELINE_N)), observed_at=T0,
    )
    assert not short.scoreable and ok.scoreable
