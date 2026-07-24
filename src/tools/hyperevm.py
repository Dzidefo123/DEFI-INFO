"""Read-only HyperEVM on-chain data via the Hyperscan (Blockscout) explorer.

Live on-chain state — latest block, gas, an address's balance, a token's
contract address — must come from the chain, not from crawled docs: addresses
and gas move constantly and a stale doc snapshot would be confidently wrong.
This mirrors the read-only, single-purpose shape of the Hyperliquid market-data
tool: the docs path answers "how does HyperEVM work", this answers "what is it
doing right now".

Public Blockscout v2 REST API, no key required.
"""

from __future__ import annotations

import re
import time

import httpx

from src.config import settings

_UA = "hyperevm-cx-agent/0.1 (on-chain reader)"
_EVM_ADDRESS = re.compile(r"^0x[a-fA-F0-9]{40}$")
_HYPE_WEI = 10**18  # HyperEVM's native gas token (HYPE) has 18 decimals
_RETRIES = 2        # the public explorer throttles bursts with transient 5xx
_BACKOFF = 0.8


class BlockscoutError(RuntimeError):
    pass


def _get(path: str, params: dict | None = None):
    last: Exception | None = None
    for attempt in range(_RETRIES + 1):
        try:
            resp = httpx.get(
                f"{settings.hyperevm_explorer}{path}",
                params=params,
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": _UA},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            last = exc
            # Retry transient throttling (5xx / network); a 4xx won't improve.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500:
                break
            if attempt < _RETRIES:
                time.sleep(_BACKOFF * (attempt + 1))
    raise BlockscoutError(f"HyperEVM explorer unreachable: {last}") from last


def _as_int(value) -> int | None:
    """Blockscout returns big counters as strings; normalize to int for display."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def chain_stats() -> dict:
    """Latest block, gas prices, and cumulative activity for the HyperEVM chain."""
    stats = _get("/api/v2/stats")
    blocks = _get("/api/v2/main-page/blocks")
    latest = (
        blocks[0]["height"]
        if isinstance(blocks, list) and blocks
        else stats.get("total_blocks")
    )
    gas = stats.get("gas_prices") or {}
    return {
        "latest_block": _as_int(latest),
        "gas_slow": gas.get("slow"),
        "gas_average": gas.get("average"),
        "gas_fast": gas.get("fast"),
        "total_transactions": _as_int(stats.get("total_transactions")),
        "transactions_today": _as_int(stats.get("transactions_today")),
        "total_addresses": _as_int(stats.get("total_addresses")),
    }


def _compat(params: dict):
    """Etherscan-compatible endpoint. Blockscout's v2 address route is heavily
    rate-limited on this instance; the legacy `/api` surface is reliable and is
    the same Etherscan pattern the Hyperliquid tool follows."""
    data = _get("/api", params)
    if not isinstance(data, dict):
        raise BlockscoutError("unexpected explorer response shape")
    return data.get("result")


def address_summary(address: str) -> dict:
    """Native-token balance and verified-contract status for one HyperEVM address."""
    address = address.strip()
    if not _EVM_ADDRESS.match(address):
        raise BlockscoutError(f"{address!r} is not a valid EVM address")

    balance = _as_int(
        _compat({"module": "account", "action": "balance", "address": address})
    )
    hype = round(balance / _HYPE_WEI, 6) if balance is not None else None

    name, verified = None, False
    source = _compat(
        {"module": "contract", "action": "getsourcecode", "address": address}
    )
    if isinstance(source, list) and source:
        entry = source[0]
        # A non-empty SourceCode means a verified contract; an EOA or unverified
        # contract returns an empty SourceCode, which we don't over-claim about.
        if (entry.get("SourceCode") or "").strip():
            verified = True
            name = entry.get("ContractName") or None

    return {
        "address": address,
        "balance_hype": hype,
        "is_verified_contract": verified,
        "name": name,
    }


def find_contract(query: str, limit: int = 5) -> list[dict]:
    """Resolve a token/contract name or symbol to its HyperEVM address(es)."""
    data = _get("/api/v2/search", {"q": query.strip()})
    items = data.get("items", []) if isinstance(data, dict) else []
    out: list[dict] = []
    for it in items:
        addr = it.get("address_hash") or it.get("address")
        if not addr:
            continue
        out.append(
            {
                "name": it.get("name"),
                "symbol": it.get("symbol"),
                "address": addr,
                "type": it.get("token_type")
                or ("contract" if it.get("is_smart_contract_address") else "address"),
                "verified": bool(it.get("is_smart_contract_verified")),
            }
        )
        if len(out) >= limit:
            break
    return out
