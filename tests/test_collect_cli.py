"""The collection entrypoint, and the dry run that keeps testing out of the store.

The collector is the one component that cannot be exercised without writing to
the live feature store, and an off-schedule write is not harmless: readings taken
seconds apart carry almost no variance, so they shrink the baseline they join and
suppress the anomalies it exists to catch. Two such runs had to be deleted by
hand before `--dry-run` existed.
"""

from datetime import datetime, timezone

import pytest

from src.blockchain import collect as cli
from src.blockchain.collectors import Collection
from src.blockchain.store import Observation

T0 = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


def _obs(metric="gas_price", value=1.25):
    return Observation(
        protocol="hyperevm",
        metric=metric,
        value=value,
        observed_at=T0,
        collected_at=T0,
    )


@pytest.fixture
def collected(monkeypatch):
    """Stub the collectors and capture whatever reaches the store."""
    written: list[Observation] = []
    state = {"result": Collection(observations=[_obs()])}

    monkeypatch.setattr(cli, "collect", lambda key, subject="": state["result"])
    monkeypatch.setattr(cli, "needs_subject", lambda key: False)

    def record(observations):
        written.extend(observations)
        return len(observations)

    monkeypatch.setattr(cli, "record", record)
    return state, written


def test_a_normal_run_stores_what_it_collected(collected):
    state, written = collected
    stored = cli.run(["hyperevm"], [], dry_run=False)
    assert stored == 1
    assert len(written) == 1


def test_a_dry_run_stores_nothing(collected):
    """The whole point of the flag."""
    state, written = collected
    stored = cli.run(["hyperevm"], [], dry_run=True)
    assert stored == 0
    assert written == []


def test_a_dry_run_still_collects_for_real(collected, monkeypatch):
    """The dry path is the full path up to the write. Anything less would test
    something other than what runs on the hour."""
    state, written = collected
    calls = []
    monkeypatch.setattr(
        cli, "collect", lambda key, subject="": calls.append(key) or state["result"]
    )
    cli.run(["hyperevm"], [], dry_run=True)
    assert calls == ["hyperevm"]


def test_a_dry_run_shows_the_readings_it_would_have_stored(collected, capsys):
    state, written = collected
    cli.run(["hyperevm"], [], dry_run=True)
    out = capsys.readouterr().out
    assert "not stored" in out
    assert "gas_price" in out


def test_errors_are_reported_in_both_modes(collected, capsys):
    state, _ = collected
    state["result"] = Collection(errors=["hyperevm.gas_price: not collected — boom"])

    cli.run(["hyperevm"], [], dry_run=True)
    assert "not collected" in capsys.readouterr().out

    cli.run(["hyperevm"], [], dry_run=False)
    assert "not collected" in capsys.readouterr().out


def test_a_collection_that_produced_nothing_stores_nothing(collected):
    state, written = collected
    state["result"] = Collection(errors=["hyperevm: unreachable"])
    assert cli.run(["hyperevm"], [], dry_run=False) == 0
    assert written == []


def test_per_market_protocols_are_collected_once_per_market(collected, monkeypatch):
    state, _ = collected
    seen = []
    monkeypatch.setattr(cli, "needs_subject", lambda key: True)
    monkeypatch.setattr(
        cli,
        "collect",
        lambda key, subject="": seen.append(subject) or state["result"],
    )
    cli.run(["hyperliquid"], ["ETH", "BTC"], dry_run=True)
    assert seen == ["ETH", "BTC"]
