"""The contract registry, its self-verification, and the ABI it reads through.

This registry differs from every other whitelist in the system in one way that
matters: it can check itself. An undocumented API endpoint has to be taken on
faith; a contract can be asked what it is. These tests pin that property and the
failures it is meant to catch.
"""

import pytest
from pydantic import ValidationError

from src.blockchain import abi, contracts
from src.blockchain.contracts import (
    CONTRACTS,
    ContractKind,
    ContractReadError,
    ContractSpec,
    for_protocol,
    get,
)

WHYPE = "0x5555555555555555555555555555555555555555"


def _spec(**kw):
    kw.setdefault("address", WHYPE)
    kw.setdefault("protocol", "hyperevm")
    kw.setdefault("kind", ContractKind.WRAPPED_NATIVE)
    kw.setdefault("symbol", "WHYPE")
    kw.setdefault("decimals", 18)
    kw.setdefault("name", "Wrapped HYPE")
    kw.setdefault("source", "verified on chain")
    return ContractSpec(**kw)


def _word(value: int) -> str:
    return "0x" + value.to_bytes(32, "big").hex()


def _string(text: str) -> str:
    raw = text.encode()
    return (
        "0x"
        + (32).to_bytes(32, "big").hex()
        + len(raw).to_bytes(32, "big").hex()
        + raw.ljust(32, b"\x00").hex()
    )


@pytest.fixture
def chain(monkeypatch):
    """A stubbed chain that answers the standard reads."""
    state = {
        "symbol()": _string("WHYPE"),
        "decimals()": _word(18),
        "totalSupply()": _word(5_000 * 10**18),
        "balance": _word(5_000 * 10**18),
        "height": 100,
    }

    def call(protocol, method, params=None):
        if method == "eth_getBalance":
            return state["balance"], None
        if method == "eth_blockNumber":
            return hex(state["height"]), None
        data = params[0]["data"]
        for signature, sel in abi.SELECTORS.items():
            if data.startswith(sel):
                return state[signature], None
        raise AssertionError(f"unexpected call data {data}")

    def batch(protocol, calls):
        return [call(protocol, m, p)[0] for m, p in calls], None

    import src.blockchain.rpc as rpc

    monkeypatch.setattr(rpc, "call", call)
    monkeypatch.setattr(rpc, "batch", batch)
    return state


# --- ABI -----------------------------------------------------------------


def test_selectors_are_registered_not_computed():
    """keccak-256 is not in the standard library, and `hashlib.sha3_256` is the
    NIST variant with different padding — a quiet way to generate confidently
    wrong selectors."""
    assert abi.selector("symbol()") == "0x95d89b41"
    with pytest.raises(abi.AbiError, match="deliberately"):
        abi.selector("transfer(address,uint256)")


def test_an_address_argument_is_padded_to_a_word():
    data = abi.encode_call("balanceOf(address)", "0x" + "ab" * 20)
    assert data.startswith("0x70a08231")
    assert len(data) == 2 + 8 + 64


def test_a_malformed_address_argument_is_refused():
    with pytest.raises(abi.AbiError, match="20-byte address"):
        abi.encode_call("balanceOf(address)", "0xdeadbeef")


def test_an_empty_return_is_an_absence_not_a_zero():
    """`0x` is what a contract that does not implement a function returns.
    Reading it as zero turns "this is not a token" into "this token has no
    supply" — a number with nothing behind it, at the layer that feeds the
    feature store."""
    with pytest.raises(abi.AbiError, match="does not implement"):
        abi.decode_uint("0x")


def test_a_dynamic_string_decodes():
    assert abi.decode_string(_string("WHYPE")) == "WHYPE"


def test_a_bytes32_string_decodes():
    """Older tokens returned a fixed 32-byte symbol, and both shapes circulate."""
    padded = "0x" + b"WHYPE".ljust(32, b"\x00").hex()
    assert abi.decode_string(padded) == "WHYPE"


def test_a_string_offset_past_the_end_is_refused():
    assert pytest.raises(abi.AbiError, abi.decode_string, "0x" + _word(9999)[2:] * 2)


def test_scaling_refuses_implausible_decimals():
    with pytest.raises(abi.AbiError, match="implausible"):
        abi.scale(1, 200)


# --- the registry --------------------------------------------------------


def test_every_entry_names_a_whitelisted_protocol():
    from src.protocols import is_known

    assert all(is_known(c.protocol) for c in CONTRACTS)


def test_every_entry_cites_where_its_address_came_from():
    """An address nobody can check is the same failure as an undocumented API
    endpoint: provenance taken on faith."""
    assert all(c.source.strip() for c in CONTRACTS)


def test_an_entry_for_an_unknown_protocol_is_refused():
    with pytest.raises(ValidationError, match="non-whitelisted"):
        _spec(protocol="aave")


def test_a_malformed_address_is_refused():
    with pytest.raises(ValidationError, match="20-byte hex address"):
        _spec(address="0xnothex")


def test_addresses_are_normalised_so_lookup_is_case_insensitive():
    """Addresses are commonly written in EIP-55 mixed case; the registry stores
    one canonical form so a lookup cannot miss on capitalisation alone."""
    mixed = "0x" + WHYPE[2:].upper()
    assert _spec(address=mixed).address == WHYPE
    assert get(mixed) is not None


def test_contracts_are_filtered_by_protocol():
    assert for_protocol("hyperevm")
    assert for_protocol("ethena") == ()


def test_readings_are_filed_under_the_contract_symbol():
    assert _spec().subject == "WHYPE"


# --- self-verification: the property that makes this admissible ----------


def test_a_matching_contract_verifies(chain):
    contracts.verify(_spec())


def test_a_wrong_symbol_stops_collection_loudly(chain):
    """A mistyped address or a redeployment produces a real number, from a real
    chain read, at the highest reliability tier, about the wrong thing."""
    chain["symbol()"] = _string("USDC")
    with pytest.raises(ContractReadError, match="reports symbol 'USDC'"):
        contracts.verify(_spec())


def test_a_wrong_decimal_count_names_the_size_of_the_error(chain):
    """Every scaled reading would be wrong by a power of ten, and saying which
    power makes the failure legible instead of merely reported."""
    chain["decimals()"] = _word(6)
    with pytest.raises(ContractReadError, match="1,000,000,000,000"):
        contracts.verify(_spec())


def test_an_address_that_is_not_a_token_fails_verification(chain):
    chain["symbol()"] = "0x"
    with pytest.raises(ContractReadError, match="does not answer as a token"):
        contracts.verify(_spec())


# --- the wrapper invariant -----------------------------------------------


def test_a_fully_backed_wrapper_reads_one(chain):
    ratio, supply, backing = contracts.backing_ratio(_spec())
    assert ratio == 1.0
    assert supply == backing == 5_000.0


def test_under_collateralisation_shows_as_a_deviation(chain):
    """The reason this metric is worth collecting: a sustained departure from 1.0
    is a finding, not a statistic about activity."""
    chain["balance"] = _word(4_000 * 10**18)
    ratio, _, _ = contracts.backing_ratio(_spec())
    assert ratio == pytest.approx(0.8)


def test_a_reading_that_spans_a_block_is_refused(chain, monkeypatch):
    """Supply and backing must describe the same moment. A wrap settling between
    them would move one number and not the other, and the ratio is precisely what
    is being watched for deviation — 1.0002 from a mid-measurement block advance
    is indistinguishable, in the store, from 1.0002 from over-collateralisation."""
    import src.blockchain.rpc as rpc

    heights = iter([100, 101])

    def drifting_batch(protocol, calls):
        return [
            hex(next(heights)) if method == "eth_blockNumber"
            else (chain["balance"] if method == "eth_getBalance"
                  else chain["totalSupply()"])
            for method, _ in calls
        ], None

    monkeypatch.setattr(rpc, "batch", drifting_batch)
    with pytest.raises(ContractReadError, match="read across blocks"):
        contracts.backing_ratio(_spec())


def test_a_non_wrapper_has_no_backing_ratio(chain):
    with pytest.raises(ContractReadError, match="not a wrapper"):
        contracts.backing_ratio(_spec(kind=ContractKind.TOKEN))


def test_zero_supply_has_no_ratio(chain):
    """Rather than dividing by zero, or reporting infinite backing."""
    chain["totalSupply()"] = _word(0)
    with pytest.raises(ContractReadError, match="zero supply"):
        contracts.backing_ratio(_spec())


# --- collection ----------------------------------------------------------


def test_contract_readings_are_filed_per_symbol(chain):
    from src.blockchain.collectors import collect_contracts

    result = collect_contracts("hyperevm")
    assert {o.metric for o in result.observations} == {
        "token_total_supply", "wrapper_native_backing", "wrapper_backing_ratio"
    }
    assert all(o.subject == "WHYPE" for o in result.observations)
    assert result.errors == []


def test_a_contract_that_fails_verification_contributes_nothing(chain):
    from src.blockchain.collectors import collect_contracts

    chain["symbol()"] = _string("NOTWHYPE")
    result = collect_contracts("hyperevm")
    assert result.observations == []
    assert any("not collected" in e for e in result.errors)


def test_a_protocol_with_no_contracts_collects_none():
    from src.blockchain.collectors import collect_contracts

    assert collect_contracts("ethena").observations == []
