"""Read-only HyperEVM chain state over JSON-RPC.

The RPC layer is stubbed, so nothing here touches a network. What is tested is
the part that matters: what the tool reports, and — more importantly — what it
declines to report rather than guessing.
"""

import pytest

from src.tools import hyperevm
from src.tools.hyperevm import ChainReadError

ADDRESS = "0x" + "a" * 40


class _FakeBlock:
    def __init__(self, number, timestamp, gas_limit=3_000_000, gas_used=500_000, txs=7):
        self.number = number
        self.timestamp = timestamp
        self.gas_limit = gas_limit
        self.gas_used = gas_used
        self.transaction_count = txs


@pytest.fixture
def chain(monkeypatch):
    """A stubbed chain. Blocks are one second apart, matching what was measured."""
    state = {"head": 1000, "gas_wei": 100_000_000, "calls": {}}

    def block_number(protocol):
        return state["head"], None

    def gas_price_wei(protocol):
        return state["gas_wei"], None

    def get_block(protocol, number="latest"):
        n = state["head"] if number == "latest" else number
        return _FakeBlock(n, timestamp=1_700_000_000 + n, **state.get("block", {})), None

    def call(protocol, method, params=None):
        return state["calls"][method], None

    monkeypatch.setattr(hyperevm.rpc, "block_number", block_number)
    monkeypatch.setattr(hyperevm.rpc, "gas_price_wei", gas_price_wei)
    monkeypatch.setattr(hyperevm.rpc, "get_block", get_block)
    monkeypatch.setattr(hyperevm.rpc, "call", call)
    return state


# --- chain stats ---------------------------------------------------------


def test_chain_stats_reports_head_gas_and_cadence(chain):
    out = hyperevm.chain_stats()
    assert out["latest_block"] == 1000
    assert out["gas_price_gwei"] == 0.1
    assert out["block_time_seconds"] == 1.0
    assert out["block_kind"] == "small"


def test_a_big_block_is_labelled_as_one(chain):
    chain["block"] = {"gas_limit": 30_000_000}
    assert hyperevm.chain_stats()["block_kind"] == "big"


def test_an_unrecognised_gas_limit_is_not_forced_into_a_kind(chain):
    """Better to say the block matches neither known type than to file it."""
    chain["block"] = {"gas_limit": 1}
    assert hyperevm.chain_stats()["block_kind"] is None


def test_a_degenerate_span_yields_no_block_time_rather_than_a_guess(chain):
    """At the very start of a chain there is nothing to measure across, and a
    fabricated block time is a number with no measurement behind it."""
    chain["head"] = 0
    assert hyperevm.chain_stats()["block_time_seconds"] is None


def test_an_unreachable_provider_raises_rather_than_returning_zeros(chain, monkeypatch):
    def boom(protocol):
        raise hyperevm.rpc.RpcUnavailable("no provider answered")

    monkeypatch.setattr(hyperevm.rpc, "block_number", boom)
    with pytest.raises(ChainReadError, match="no provider answered"):
        hyperevm.chain_stats()


# --- address summary -----------------------------------------------------


def test_an_account_with_no_code_is_not_a_contract(chain):
    chain["calls"] = {"eth_getBalance": hex(2 * 10**18), "eth_getCode": "0x"}
    out = hyperevm.address_summary(ADDRESS)
    assert out["balance_hype"] == 2.0
    assert out["is_contract"] is False
    assert out["code_size_bytes"] == 0


def test_an_address_holding_code_is_a_contract(chain):
    chain["calls"] = {"eth_getBalance": "0x0", "eth_getCode": "0x60806040"}
    out = hyperevm.address_summary(ADDRESS)
    assert out["is_contract"] is True
    assert out["code_size_bytes"] == 4


def test_all_zero_code_is_treated_as_no_code(chain):
    """Some nodes answer `0x0` rather than `0x` for an ordinary account."""
    chain["calls"] = {"eth_getBalance": "0x0", "eth_getCode": "0x0"}
    assert hyperevm.address_summary(ADDRESS)["is_contract"] is False


def test_verification_status_is_absent_not_false(chain):
    """JSON-RPC cannot tell whether a contract's source was published. Reporting
    False would assert something unverified about a contract that may well be
    verified — the same over-claim the tool exists to avoid."""
    chain["calls"] = {"eth_getBalance": "0x0", "eth_getCode": "0x60806040"}
    assert hyperevm.address_summary(ADDRESS)["is_verified_contract"] is None


def test_a_malformed_address_is_refused_before_any_call(chain):
    with pytest.raises(ChainReadError, match="not a valid EVM address"):
        hyperevm.address_summary("0xnothex")


def test_a_non_string_code_result_is_refused(chain):
    """Shape validation at the point of use, not just at the transport."""
    chain["calls"] = {"eth_getBalance": "0x0", "eth_getCode": 12345}
    with pytest.raises(ChainReadError, match="not a hex string"):
        hyperevm.address_summary(ADDRESS)


# --- the capability that went away --------------------------------------


def test_name_search_is_declared_unsupported():
    """An explorer maintains an index; JSON-RPC does not. Declaring the gap lets
    the caller say a name cannot be resolved, instead of running a search that
    always returns nothing — which is indistinguishable from a search that found
    nothing."""
    assert hyperevm.search_supported() is False


def test_the_old_error_name_still_resolves():
    """The graph's live-data handler and its tests catch it by name."""
    assert hyperevm.BlockscoutError is ChainReadError
