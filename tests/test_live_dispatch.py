"""Live-data dispatch tests (PR 5): protocol selection + per-protocol handlers.
Tool clients are monkeypatched — no network, no API key."""

import pytest

from src.graph import nodes


# --- protocol selection -------------------------------------------------


def test_pick_hyperevm_when_named():
    assert nodes._pick_live_protocol({"protocols": ["hyperevm"]}).key == "hyperevm"


def test_pick_hyperliquid_when_named():
    assert nodes._pick_live_protocol({"protocols": ["hyperliquid"]}).key == "hyperliquid"


def test_pick_defaults_to_hyperliquid_only_when_no_protocol_routed():
    """The default is for "what's ETH funding right now" — no protocol named."""
    assert nodes._pick_live_protocol({"protocols": []}).key == "hyperliquid"
    assert nodes._pick_live_protocol({}).key == "hyperliquid"
    # A key the registry doesn't know is not a routing decision worth honouring.
    assert nodes._pick_live_protocol({"protocols": ["aave"]}).key == "hyperliquid"


def test_pick_returns_routed_protocol_that_has_no_live_tool():
    """Ethena has no live tool. It must still be returned, not skipped.

    Skipping it falls through to the Hyperliquid default and answers a USDe
    question with perps data — real numbers, wrong protocol, no admission. The
    honest refusal in `live_data` depends on this returning `ethena`.
    """
    assert nodes._pick_live_protocol({"protocols": ["ethena"]}).key == "ethena"


def test_pick_prefers_a_protocol_that_has_a_tool():
    """Cross-protocol routing should still reach a working tool when one exists."""
    assert nodes._pick_live_protocol(
        {"protocols": ["ethena", "hyperevm"]}
    ).key == "hyperevm"


def test_live_data_refuses_rather_than_substituting_another_protocol(monkeypatch):
    """The end-to-end version of the above, and the regression that matters."""
    def _should_not_run(coin=None):
        raise AssertionError("answered an Ethena question with Hyperliquid data")

    monkeypatch.setattr(nodes, "market_snapshot", _should_not_run)
    out = nodes.live_data(
        {"question": "what is the current sUSDe APY?", "protocols": ["ethena"]}
    )
    assert "Ethena" in out["answer"]
    assert "don't have a live-data source" in out["answer"]


# --- hyperliquid handler still works ------------------------------------


def test_live_data_hyperliquid_snapshot(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "market_snapshot",
        lambda coin: {
            "coin": "ETH", "mark_price": "3000", "oracle_price": "2999",
            "funding_hourly_pct": 0.001, "funding_annualized_pct": 8.76,
            "open_interest": "1000", "day_volume_usd": "5e6", "max_leverage": 25,
        },
    )
    out = nodes.live_data({"question": "eth funding now", "protocols": ["hyperliquid"], "coin": "ETH"})
    assert "ETH-PERP" in out["answer"]
    assert "Source" in out["answer"]


def test_live_data_hyperliquid_missing_coin_asks():
    out = nodes.live_data({"question": "funding right now", "protocols": ["hyperliquid"]})
    assert "which market" in out["answer"].lower()


# --- hyperevm handler: three modes --------------------------------------
#
# Reads JSON-RPC since the explorer it used was rebuilt without an API. Two of
# these tests pin what the tool now REFUSES to say, which is the part that
# changed most.

_ADDR = "0x" + "b" * 40


def test_live_data_hyperevm_address_lookup(monkeypatch):
    monkeypatch.setattr(
        nodes.hyperevm, "address_summary",
        lambda a: {"address": _ADDR, "balance_hype": 12.5, "is_contract": False,
                   "is_verified_contract": None, "code_size_bytes": 0},
    )
    out = nodes.live_data({"question": f"balance of {_ADDR}", "protocols": ["hyperevm"]})
    assert "12.5 HYPE" in out["answer"]
    assert _ADDR in out["answer"]
    assert "account" in out["answer"]


def test_live_data_hyperevm_contract_is_named_as_one(monkeypatch):
    monkeypatch.setattr(
        nodes.hyperevm, "address_summary",
        lambda a: {"address": _ADDR, "balance_hype": 0.0, "is_contract": True,
                   "is_verified_contract": None, "code_size_bytes": 2041},
    )
    out = nodes.live_data({"question": f"what is {_ADDR}", "protocols": ["hyperevm"]})
    assert "contract" in out["answer"]
    assert "2,041 bytes" in out["answer"]


def test_live_data_says_verification_was_not_checked_rather_than_omitting_it(monkeypatch):
    """A reader who saw no verification line would reasonably assume it was
    checked and came back clean. JSON-RPC cannot check it at all."""
    monkeypatch.setattr(
        nodes.hyperevm, "address_summary",
        lambda a: {"address": _ADDR, "balance_hype": 0.0, "is_contract": True,
                   "is_verified_contract": None, "code_size_bytes": 100},
    )
    out = nodes.live_data({"question": f"is {_ADDR} verified", "protocols": ["hyperevm"]})
    assert "not available" in out["answer"]
    assert "whether its source was published" in out["answer"]


def test_live_data_hyperevm_name_search_refuses_without_claiming_absence(monkeypatch):
    """The capability that went away with the explorer. An empty search result
    would be indistinguishable from a search that found nothing — for a user
    hunting a token contract, that is the difference between "try again with an
    address" and "that token does not exist"."""
    out = nodes.live_data(
        {"question": "contract address for WHYPE", "protocols": ["hyperevm"], "coin": "WHYPE"}
    )
    answer = out["answer"]
    assert "no way to search" in answer
    assert "not saying no contract matches" in answer
    assert "WHYPE" in answer


def test_live_data_hyperevm_chain_stats_default(monkeypatch):
    monkeypatch.setattr(
        nodes.hyperevm, "chain_stats",
        lambda: {"latest_block": 38553400, "gas_price_gwei": 0.59,
                 "block_time_seconds": 1.0, "transactions_in_block": 7,
                 "block_kind": "small", "block_gas_limit": 3_000_000,
                 "block_gas_used": 516_921},
    )
    out = nodes.live_data({"question": "how is the hyperevm network doing", "protocols": ["hyperevm"]})
    assert "38,553,400" in out["answer"]      # thousands-formatted
    assert "HyperEVM network" in out["answer"]
    assert "0.59 gwei" in out["answer"]
    assert "small block" in out["answer"]


def test_live_data_hyperevm_unmeasurable_block_time_is_not_guessed(monkeypatch):
    monkeypatch.setattr(
        nodes.hyperevm, "chain_stats",
        lambda: {"latest_block": 1, "gas_price_gwei": 0.1,
                 "block_time_seconds": None, "transactions_in_block": 0,
                 "block_kind": None, "block_gas_limit": 3_000_000,
                 "block_gas_used": 0},
    )
    out = nodes.live_data({"question": "hyperevm block time", "protocols": ["hyperevm"]})
    assert "n/a" in out["answer"]
    assert "unrecognised block" in out["answer"]


def test_live_data_hyperevm_read_error(monkeypatch):
    def _boom():
        raise nodes.hyperevm.ChainReadError("no provider answered")

    monkeypatch.setattr(nodes.hyperevm, "chain_stats", _boom)
    out = nodes.live_data({"question": "hyperevm gas", "protocols": ["hyperevm"]})
    assert "couldn't read HyperEVM chain state" in out["answer"]
    assert "no provider answered" in out["answer"]
