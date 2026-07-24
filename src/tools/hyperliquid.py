"""Read-only Hyperliquid market data.

Docs answer "how does funding work"; this answers "what is funding on ETH right
now". A support agent needs both, and conflating them is how a RAG bot ends up
confidently quoting a rate from a doc snapshot months out of date.
"""

from __future__ import annotations

import httpx

from src.config import settings


class MarketDataError(RuntimeError):
    pass


def _info(payload: dict) -> list | dict:
    try:
        resp = httpx.post(
            f"{settings.hyperliquid_api}/info",
            json=payload,
            timeout=15.0,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        raise MarketDataError(f"Hyperliquid API unreachable: {exc}") from exc


def market_snapshot(coin: str) -> dict:
    """Mark price, funding rate and open interest for one perp, e.g. "ETH"."""
    coin = coin.strip().upper()
    data = _info({"type": "metaAndAssetCtxs"})
    if not isinstance(data, list) or len(data) != 2:
        raise MarketDataError("Unexpected metaAndAssetCtxs response shape")

    meta, contexts = data
    universe = meta.get("universe", [])
    for asset, ctx in zip(universe, contexts):
        if asset.get("name", "").upper() != coin:
            continue

        hourly = float(ctx.get("funding", 0.0))
        return {
            "coin": coin,
            "mark_price": ctx.get("markPx"),
            "oracle_price": ctx.get("oraclePx"),
            "funding_hourly_pct": round(hourly * 100, 6),
            # Hyperliquid funding settles hourly; annualizing is the number
            # traders actually compare against other venues.
            "funding_annualized_pct": round(hourly * 24 * 365 * 100, 3),
            "open_interest": ctx.get("openInterest"),
            "day_volume_usd": ctx.get("dayNtlVlm"),
            "max_leverage": asset.get("maxLeverage"),
        }

    raise MarketDataError(f"No perp market named {coin!r} on Hyperliquid")


def list_markets(limit: int = 40) -> list[str]:
    meta = _info({"type": "meta"})
    names = [a["name"] for a in meta.get("universe", [])]
    return names[:limit]
