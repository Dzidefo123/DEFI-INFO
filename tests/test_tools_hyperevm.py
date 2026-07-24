"""HyperEVM Blockscout tool tests. _get is monkeypatched — no network."""

import pytest

from src.tools import hyperevm
from src.tools.hyperevm import BlockscoutError


def test_chain_stats_parses_latest_block_and_gas(monkeypatch):
    responses = {
        "/api/v2/stats": {
            "total_blocks": 100,
            "gas_prices": {"slow": 0.34, "average": 0.59, "fast": 3.49},
            "total_transactions": 999,
            "transactions_today": 12,
            "total_addresses": 77,
        },
        "/api/v2/main-page/blocks": [{"height": 38553400}],
    }
    monkeypatch.setattr(hyperevm, "_get", lambda path, params=None: responses[path])

    out = hyperevm.chain_stats()
    assert out["latest_block"] == 38553400  # from blocks, not total_blocks
    assert out["gas_average"] == 0.59
    assert out["total_transactions"] == 999


def test_address_summary_balance_and_verified_contract(monkeypatch):
    def fake_get(path, params=None):
        if params["action"] == "balance":
            return {"status": "1", "result": str(3 * 10**18)}  # 3 HYPE in wei
        if params["action"] == "getsourcecode":
            return {"status": "1", "result": [{"SourceCode": "contract X {}", "ContractName": "WrappedHYPE"}]}
        raise AssertionError(params)

    monkeypatch.setattr(hyperevm, "_get", fake_get)
    out = hyperevm.address_summary("0x" + "a" * 40)
    assert out["balance_hype"] == 3.0
    assert out["is_verified_contract"] is True
    assert out["name"] == "WrappedHYPE"


def test_address_summary_eoa_not_over_claimed(monkeypatch):
    def fake_get(path, params=None):
        if params["action"] == "balance":
            return {"status": "1", "result": "0"}
        return {"status": "1", "result": [{"SourceCode": "", "ContractName": ""}]}

    monkeypatch.setattr(hyperevm, "_get", fake_get)
    out = hyperevm.address_summary("0x" + "a" * 40)
    assert out["balance_hype"] == 0.0
    assert out["is_verified_contract"] is False
    assert out["name"] is None


def test_address_summary_rejects_bad_address(monkeypatch):
    # Must fail before any network call.
    def _boom(path, params=None):
        raise AssertionError("should not fetch a malformed address")

    monkeypatch.setattr(hyperevm, "_get", _boom)
    with pytest.raises(BlockscoutError):
        hyperevm.address_summary("0xnothex")


def test_find_contract_maps_search_items(monkeypatch):
    monkeypatch.setattr(
        hyperevm,
        "_get",
        lambda path, params=None: {
            "items": [
                {
                    "name": "Wrapped HYPE",
                    "symbol": "WHYPE",
                    "address_hash": "0x5555555555555555555555555555555555555555",
                    "token_type": "ERC-20",
                    "is_smart_contract_verified": True,
                },
                {"symbol": "NOADDR"},  # dropped: no address
            ]
        },
    )
    hits = hyperevm.find_contract("WHYPE")
    assert len(hits) == 1
    assert hits[0]["address"] == "0x5555555555555555555555555555555555555555"
    assert hits[0]["verified"] is True


def test_find_contract_empty(monkeypatch):
    monkeypatch.setattr(hyperevm, "_get", lambda path, params=None: {"items": []})
    assert hyperevm.find_contract("nope") == []
