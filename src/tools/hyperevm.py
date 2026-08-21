"""Read-only HyperEVM chain state, over JSON-RPC.

Live on-chain state — the head of the chain, gas, an address's balance — must
come from the chain rather than from crawled docs: those numbers move constantly
and a stale doc snapshot would be confidently wrong. This mirrors the read-only,
single-purpose shape of the Hyperliquid market-data tool: the docs path answers
"how does HyperEVM work", this answers "what is it doing right now".

**Why this was rewritten.** It previously read a Blockscout explorer API. That
explorer was rebuilt as a web application and its API went with it, so every call
here returned an error — silently degrading a shipped feature. The replacement
reads JSON-RPC through `src.blockchain.rpc`, which is provider-agnostic: the host
lives in a registry, and the reading carries the provider's `SourceTier` with it.

**What was lost, and what is said instead.** An explorer maintains an index;
JSON-RPC does not. So token and contract lookup *by name or symbol* is gone, and
so is verified-source status. Those are not degraded here into a vaguer answer —
`search_supported()` is False and the caller says plainly that a name cannot be
resolved and an address is needed. A search that silently returned nothing would
be indistinguishable from a search that found nothing, which for a user hunting a
token contract is the difference between "try again with an address" and "that
token does not exist".

What RPC gives that the explorer did not: `eth_getCode` distinguishes a contract
from an ordinary account without an index, and block timing comes straight from
block timestamps.
"""

from __future__ import annotations

import re

from src.blockchain import rpc
from src.blockchain.features import classify_block

_EVM_ADDRESS = re.compile(r"^0x[a-fA-F0-9]{40}$")
_HYPE_WEI = 10**18  # HyperEVM's native gas token (HYPE) has 18 decimals

PROTOCOL = "hyperevm"

# Blocks back from the head used to measure block time. Small enough to stay
# cheap, large enough that one slow block does not dominate the average.
_TIMING_SPAN = 20


class ChainReadError(RuntimeError):
    """A chain read could not be completed. Never raised with a substituted value."""


# Kept as an alias because the graph's live-data handler and its tests catch it
# by name. The old name described the provider; the new one describes the
# failure, which is what callers actually care about.
BlockscoutError = ChainReadError


def search_supported() -> bool:
    """Whether token/contract lookup by name or symbol is available.

    False over JSON-RPC. Exposed as a capability check so the caller can say what
    it cannot do, rather than running a search that always comes back empty.
    """
    return False


def chain_stats() -> dict:
    """Head of the chain, gas, and current block cadence."""
    try:
        latest, _ = rpc.block_number(PROTOCOL)
        gas_wei, _ = rpc.gas_price_wei(PROTOCOL)
        head, _ = rpc.get_block(PROTOCOL, latest)
        earlier, _ = rpc.get_block(PROTOCOL, max(latest - _TIMING_SPAN, 0))
    except (rpc.RpcUnavailable, rpc.RpcShapeError) as exc:
        raise ChainReadError(str(exc)) from exc

    span_blocks = head.number - earlier.number
    span_seconds = head.timestamp - earlier.timestamp
    return {
        "latest_block": head.number,
        "gas_price_gwei": round(gas_wei / 1e9, 6),
        # None rather than a guess when the span is degenerate — a fabricated
        # block time would be a number with no measurement behind it.
        "block_time_seconds": round(span_seconds / span_blocks, 3) if span_blocks else None,
        "transactions_in_block": head.transaction_count,
        "block_kind": classify_block(head.gas_limit),
        "block_gas_limit": head.gas_limit,
        "block_gas_used": head.gas_used,
    }


def address_summary(address: str) -> dict:
    """Native-token balance, and whether the address holds contract code."""
    address = address.strip()
    if not _EVM_ADDRESS.match(address):
        raise ChainReadError(f"{address!r} is not a valid EVM address")

    try:
        balance_hex, _ = rpc.call(PROTOCOL, "eth_getBalance", [address, "latest"])
        code_hex, _ = rpc.call(PROTOCOL, "eth_getCode", [address, "latest"])
    except (rpc.RpcUnavailable, rpc.RpcShapeError) as exc:
        raise ChainReadError(str(exc)) from exc

    if not isinstance(code_hex, str):
        raise ChainReadError("eth_getCode: result is not a hex string")

    wei = rpc._as_int(balance_hex, "eth_getBalance")
    # "0x" or "0x0" means no code, i.e. an externally owned account.
    code = code_hex[2:] if code_hex.startswith("0x") else code_hex
    is_contract = bool(code.strip("0"))

    return {
        "address": address,
        "balance_hype": round(wei / _HYPE_WEI, 6),
        "is_contract": is_contract,
        # Deliberately absent rather than False: JSON-RPC cannot tell whether a
        # contract's source has been verified, and reporting False would assert
        # something unverified about a contract that may well be verified.
        "is_verified_contract": None,
        "code_size_bytes": len(code) // 2,
    }
