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

_ADDR = "0x" + "b" * 40


def test_live_data_hyperevm_address_lookup(monkeypatch):
    monkeypatch.setattr(
        nodes.hyperevm, "address_summary",
        lambda a: {"address": _ADDR, "balance_hype": 12.5,
                   "is_verified_contract": False, "name": None},
    )
    out = nodes.live_data({"question": f"balance of {_ADDR}", "protocols": ["hyperevm"]})
    assert "12.5 HYPE" in out["answer"]
    assert _ADDR in out["answer"]
    assert "hyperscan.com/address/" in out["answer"]


def test_live_data_hyperevm_verified_contract(monkeypatch):
    monkeypatch.setattr(
        nodes.hyperevm, "address_summary",
        lambda a: {"address": _ADDR, "balance_hype": 0.0,
                   "is_verified_contract": True, "name": "WrappedHYPE"},
    )
    out = nodes.live_data({"question": f"what is {_ADDR}", "protocols": ["hyperevm"]})
    assert "Verified contract" in out["answer"]
    assert "WrappedHYPE" in out["answer"]


def test_live_data_hyperevm_contract_search_by_coin(monkeypatch):
    monkeypatch.setattr(
        nodes.hyperevm, "find_contract",
        lambda q, limit=5: [{"name": "Wrapped HYPE", "symbol": "WHYPE",
                             "address": "0x5555", "type": "ERC-20", "verified": True}],
    )
    out = nodes.live_data(
        {"question": "contract address for WHYPE", "protocols": ["hyperevm"], "coin": "WHYPE"}
    )
    assert "0x5555" in out["answer"]
    assert "verified" in out["answer"].lower()


def test_live_data_hyperevm_chain_stats_default(monkeypatch):
    monkeypatch.setattr(
        nodes.hyperevm, "chain_stats",
        lambda: {"latest_block": 38553400, "gas_slow": 0.34, "gas_average": 0.59,
                 "gas_fast": 3.49, "total_transactions": 104216478,
                 "transactions_today": 12345, "total_addresses": 987654},
    )
    out = nodes.live_data({"question": "how is the hyperevm network doing", "protocols": ["hyperevm"]})
    assert "38,553,400" in out["answer"]      # thousands-formatted
    assert "HyperEVM network" in out["answer"]


def test_live_data_hyperevm_explorer_error(monkeypatch):
    def _boom():
        raise nodes.hyperevm.BlockscoutError("down")

    monkeypatch.setattr(nodes.hyperevm, "chain_stats", _boom)
    out = nodes.live_data({"question": "hyperevm gas", "protocols": ["hyperevm"]})
    assert "couldn't reach" in out["answer"].lower()
