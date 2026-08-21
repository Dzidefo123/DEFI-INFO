"""Provider-agnostic JSON-RPC reads.

This module exists because the previous on-chain source stopped existing. The
explorer it was written against was rebuilt as a web application and its API went
with it — so the lesson is not "that host was unreliable", it is that a client
written *for one host* inherits that host's product decisions.

Three rules follow, and they are the whole design.

**Providers are data, not code.** A protocol maps to an ordered list of
providers; the client walks it. Adding a fallback, or swapping the primary, is a
registry edit. Nothing above this layer names a host.

**Provenance travels with the reading.** Every call returns the value *and* the
provider that served it, carrying its `SourceTier`. That is what makes the
durability argument true rather than merely intended: when a lower-tier provider
answers, the evidence built from it is scored lower automatically, because the
tier came from the provider rather than from an assumption at the call site.

**Shape is validated; status codes are not trusted.** The failure that motivated
this returned **HTTP 200 with an HTML page**. A client checking `status_code`
would have called that success, and a lenient parser would have turned it into an
empty result — which downstream is indistinguishable from "the chain reported
nothing unusual". So a response must parse as JSON, be a JSON-RPC envelope, and
carry a `result` of the expected shape. Anything else raises.

**Nothing is ever substituted.** No default, no zero, no last-known value. A
failed read raises and the caller records *not collected*. A repeated stale
reading would shrink the variance of the baseline it joins, making a series look
calmer precisely while we have stopped being able to see it.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from src.evidence.models import SourceTier

_UA = "defi-info/1.0 (read-only on-chain reader)"
_TIMEOUT = 20.0


class RpcProvider(BaseModel):
    """One endpoint that can answer JSON-RPC reads, and what it is worth."""

    model_config = ConfigDict(frozen=True)

    name: str
    url: str
    tier: SourceTier
    # Public endpoints serve `latest` only. Historical state needs an archive
    # node, which is why state series must be *generated* on a schedule rather
    # than backfilled later.
    archive: bool = False
    # Documented ceiling, requests per minute. The throttle uses the strictest
    # value across the providers actually configured.
    rate_limit_per_min: int = 100


# Ordered by preference. A protocol with no entry has no chain reader, and the
# collector says so rather than silently skipping it.
PROVIDERS: dict[str, tuple[RpcProvider, ...]] = {
    "hyperevm": (
        RpcProvider(
            name="hyperliquid-public",
            url="https://rpc.hyperliquid.xyz/evm",
            tier=SourceTier.CHAIN,
            archive=False,
            rate_limit_per_min=100,
        ),
    ),
}


class RpcUnavailable(RuntimeError):
    """No configured provider could serve the call."""


class RpcShapeError(RuntimeError):
    """A provider answered, but not with what it claimed to be."""


def providers_for(protocol: str) -> tuple[RpcProvider, ...]:
    return PROVIDERS.get(protocol, ())


def has_provider(protocol: str) -> bool:
    return bool(providers_for(protocol))


# --- throttle -----------------------------------------------------------
#
# One shared minimum interval across the process. Collectors may run concurrently
# with an investigation's blockchain agent, and the public endpoint's limit is
# per-client, not per-caller. The lock is taken around the wait AND the stamp so
# two threads cannot both observe the same last-call time and proceed together —
# the same mistake that let two threads build a vector store at once in C3.

_throttle_lock = threading.Lock()
_last_call_at = 0.0


def _min_interval(provider: RpcProvider) -> float:
    return 60.0 / max(provider.rate_limit_per_min, 1)


def _throttle(provider: RpcProvider) -> None:
    global _last_call_at
    with _throttle_lock:
        wait = _min_interval(provider) - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


# --- envelope validation ------------------------------------------------


def _validate_envelope(payload: Any, method: str) -> Any:
    """Unwrap a JSON-RPC response, or raise.

    Deliberately strict about the envelope rather than only about the result: a
    server that returns `{"data": ...}` is not speaking JSON-RPC, and treating
    its output as a result would import an unknown schema into the feature store.
    """
    if not isinstance(payload, dict):
        raise RpcShapeError(f"{method}: response is {type(payload).__name__}, not an object")
    if "jsonrpc" not in payload:
        raise RpcShapeError(f"{method}: response is not a JSON-RPC envelope")
    if (error := payload.get("error")) is not None:
        message = error.get("message") if isinstance(error, dict) else error
        raise RpcShapeError(f"{method}: node returned an error — {message}")
    if "result" not in payload:
        raise RpcShapeError(f"{method}: envelope carries neither result nor error")
    return payload["result"]


def call(protocol: str, method: str, params: list | None = None) -> tuple[Any, RpcProvider]:
    """Make one read, returning the result and the provider that served it.

    Walks the provider list in order. A provider that fails for any reason —
    transport, non-JSON body, bad envelope, node error — is passed over, and if
    none succeed the call raises. It never falls back to a value.
    """
    candidates = providers_for(protocol)
    if not candidates:
        raise RpcUnavailable(f"no JSON-RPC provider is configured for {protocol!r}")

    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}
    failures: list[str] = []

    for provider in candidates:
        _throttle(provider)
        try:
            response = httpx.post(
                provider.url,
                json=body,
                timeout=_TIMEOUT,
                headers={"Content-Type": "application/json", "User-Agent": _UA},
            )
            # Status is not consulted before parsing. A 200 carrying HTML is the
            # exact failure this client was written after, and the parse is what
            # catches it.
            payload = response.json()
        except ValueError:
            failures.append(f"{provider.name}: response body is not JSON")
            continue
        except httpx.HTTPError as exc:
            failures.append(f"{provider.name}: {exc}")
            continue

        try:
            return _validate_envelope(payload, method), provider
        except RpcShapeError as exc:
            failures.append(str(exc))

    raise RpcUnavailable(
        f"{method}: no provider for {protocol!r} returned a usable response — "
        + "; ".join(failures)
    )


def batch(
    protocol: str, calls: list[tuple[str, list | None]]
) -> tuple[list[Any], RpcProvider]:
    """Several reads in one round trip, answered against one chain head.

    This is not only a throughput optimisation — for some readings it is the only
    way to get a correct answer. The public endpoint serves `eth_call` at the head
    and cannot be pinned to a block number, blocks arrive about every second, and
    the client throttles to stay inside a 100-per-minute limit. Four sequential
    reads therefore span two or three blocks *by construction*, which makes any
    quantity derived from two of them — a ratio, a difference — partly a
    measurement of how long the reads took.

    A batch is one POST. The node answers every element against the same head, so
    bracketing a batch with `eth_blockNumber` at both ends proves the readings
    between them describe one moment.

    Responses may come back in any order, so they are matched by id rather than
    by position — a node returning them shuffled would otherwise silently swap
    two values.
    """
    candidates = providers_for(protocol)
    if not candidates:
        raise RpcUnavailable(f"no JSON-RPC provider is configured for {protocol!r}")
    if not calls:
        return [], candidates[0]

    body = [
        {"jsonrpc": "2.0", "id": i, "method": method, "params": params or []}
        for i, (method, params) in enumerate(calls)
    ]
    failures: list[str] = []

    for provider in candidates:
        _throttle(provider)
        try:
            response = httpx.post(
                provider.url,
                json=body,
                timeout=_TIMEOUT,
                headers={"Content-Type": "application/json", "User-Agent": _UA},
            )
            payload = response.json()
        except ValueError:
            failures.append(f"{provider.name}: response body is not JSON")
            continue
        except httpx.HTTPError as exc:
            failures.append(f"{provider.name}: {exc}")
            continue

        if not isinstance(payload, list) or len(payload) != len(calls):
            failures.append(
                f"{provider.name}: expected {len(calls)} batched responses, got "
                f"{type(payload).__name__} of "
                f"{len(payload) if isinstance(payload, list) else 'n/a'}"
            )
            continue

        try:
            by_id = {}
            for element in payload:
                if not isinstance(element, dict) or "id" not in element:
                    raise RpcShapeError("batch element carries no id")
                by_id[element["id"]] = element
            results = [
                _validate_envelope(by_id[i], calls[i][0]) for i in range(len(calls))
            ]
        except (RpcShapeError, KeyError) as exc:
            failures.append(f"{provider.name}: {exc}")
            continue

        return results, provider

    raise RpcUnavailable(
        f"batch of {len(calls)} for {protocol!r} returned no usable response — "
        + "; ".join(failures)
    )


# --- typed reads --------------------------------------------------------


def _as_int(value: Any, what: str) -> int:
    """Hex quantity -> int. Raises rather than defaulting."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value, 16)
        except ValueError:
            pass
    raise RpcShapeError(f"{what}: {value!r} is not a hex quantity")


def block_number(protocol: str) -> tuple[int, RpcProvider]:
    result, provider = call(protocol, "eth_blockNumber")
    return _as_int(result, "eth_blockNumber"), provider


def gas_price_wei(protocol: str) -> tuple[int, RpcProvider]:
    result, provider = call(protocol, "eth_gasPrice")
    return _as_int(result, "eth_gasPrice"), provider


class Block(BaseModel):
    """The fields this system reads, validated on arrival."""

    model_config = ConfigDict(frozen=True)

    number: int
    timestamp: int
    gas_limit: int
    gas_used: int
    transaction_count: int


def get_block(protocol: str, number: int | str = "latest") -> tuple[Block, RpcProvider]:
    """One block, reduced to the fields we use and validated.

    `transactions` is requested without full bodies: the count is what feeds the
    throughput series, and pulling every transaction would multiply the response
    size for nothing.
    """
    tag = number if isinstance(number, str) else hex(number)
    result, provider = call(protocol, "eth_getBlockByNumber", [tag, False])

    if not isinstance(result, dict):
        raise RpcShapeError(f"eth_getBlockByNumber({tag}): result is not a block object")
    transactions = result.get("transactions")
    if not isinstance(transactions, list):
        raise RpcShapeError(f"eth_getBlockByNumber({tag}): no transaction list")

    return (
        Block(
            number=_as_int(result.get("number"), "block.number"),
            timestamp=_as_int(result.get("timestamp"), "block.timestamp"),
            gas_limit=_as_int(result.get("gasLimit"), "block.gasLimit"),
            gas_used=_as_int(result.get("gasUsed"), "block.gasUsed"),
            transaction_count=len(transactions),
        ),
        provider,
    )
