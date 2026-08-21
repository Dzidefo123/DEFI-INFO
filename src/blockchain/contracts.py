"""Which contracts the system reads, and how it proves it is reading the right one.

This is a whitelist, in the same spirit as `protocols.py` and the security
incident registry: an entry exists because a person put it there on purpose. The
cost of a wrong entry is high and quiet — reading some other contract's supply
and reporting it as a protocol's backing produces a real number, from a real
chain read, at the highest reliability tier, about the wrong thing.

**But unlike every other registry here, this one can check itself.**

That is the property that makes contract addresses admissible where a
reverse-engineered API endpoint was not. An undocumented endpoint has no way to
confirm it is what you think — you can only trust the person who found it. A
contract can be asked. `verify` calls `symbol()` and `decimals()` and compares
them with what the registry claims, so a mistyped address, a redeployment, or a
proxy pointed somewhere new stops collection **loudly** instead of quietly
producing plausible numbers about something else.

An entry therefore carries its expected identity as part of the entry, and that
expectation is checked before any reading is stored. Provenance you can
interrogate is a different thing from provenance you have to take on faith.

**Adding an entry.** Find the address from a source you can cite, add it here
with the symbol and decimals you expect, and run
`python -m src.blockchain.contracts --verify`. If the chain disagrees with the
entry, the entry is wrong — not the chain.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator

from src.blockchain import abi
from src.protocols import is_known

_ADDRESS = re.compile(r"^0x[a-fA-F0-9]{40}$")


class ContractKind(str, Enum):
    """What a contract is, which decides what its readings mean."""

    WRAPPED_NATIVE = "wrapped_native"  # holds native coin, issues a 1:1 token
    TOKEN = "token"                    # a plain ERC-20
    VAULT = "vault"                    # holds assets, issues shares


class ContractSpec(BaseModel):
    """One whitelisted contract, and the identity it must prove on chain."""

    model_config = ConfigDict(frozen=True)

    address: str
    protocol: str
    kind: ContractKind
    # Expected identity, checked against the chain by `verify`. These are not
    # documentation — they are the assertion that makes a wrong address loud.
    symbol: str
    decimals: int
    name: str
    # Where the address came from, so a reader can go and check it themselves.
    source: str

    @field_validator("address")
    @classmethod
    def _is_an_address(cls, v: str) -> str:
        if not _ADDRESS.match(v):
            raise ValueError(f"{v!r} is not a 20-byte hex address")
        return v.lower()

    @field_validator("protocol")
    @classmethod
    def _protocol_is_whitelisted(cls, v: str) -> str:
        if not is_known(v):
            raise ValueError(f"contract references non-whitelisted protocol {v!r}")
        return v

    @property
    def subject(self) -> str:
        """The feature-store subject readings from this contract are filed under."""
        return self.symbol


# --- the whitelist ------------------------------------------------------
#
# Short on purpose. Every entry below was confirmed by asking the contract what
# it is, and the same check runs before each collection. Addresses proposed but
# not verified do not belong here — see the module docstring.

CONTRACTS: tuple[ContractSpec, ...] = (
    ContractSpec(
        address="0x5555555555555555555555555555555555555555",
        protocol="hyperevm",
        kind=ContractKind.WRAPPED_NATIVE,
        symbol="WHYPE",
        decimals=18,
        name="Wrapped HYPE",
        source=(
            "self-identified on chain: name() returned 'Wrapped HYPE', symbol() "
            "returned 'WHYPE', decimals() returned 18, and its native balance "
            "tracks its total supply as a wrapper's should"
        ),
    ),
)


def for_protocol(protocol: str) -> tuple[ContractSpec, ...]:
    return tuple(c for c in CONTRACTS if c.protocol == protocol)


def get(address: str) -> ContractSpec | None:
    target = address.lower()
    return next((c for c in CONTRACTS if c.address == target), None)


# --- reading ------------------------------------------------------------


class ContractReadError(RuntimeError):
    """A contract read failed, or the contract is not what the registry claims."""


def read(protocol: str, address: str, signature: str, arg: str | None = None) -> str:
    """One `eth_call`, returning raw hex. Raises rather than substituting."""
    from src.blockchain import rpc

    try:
        result, _ = rpc.call(
            protocol,
            "eth_call",
            [{"to": address, "data": abi.encode_call(signature, arg)}, "latest"],
        )
    except (rpc.RpcUnavailable, rpc.RpcShapeError) as exc:
        raise ContractReadError(f"{signature} on {address}: {exc}") from exc
    if not isinstance(result, str):
        raise ContractReadError(f"{signature} on {address}: result is not hex data")
    return result


def verify(spec: ContractSpec) -> None:
    """Confirm the contract is what the registry says. Raises if it is not.

    Run before storing any reading from this contract. A registry entry that has
    stopped matching the chain must stop being used — silently collecting from a
    redeployed or mistyped address is exactly the failure this file exists to
    prevent, and it would arrive wearing the highest reliability tier.
    """
    try:
        symbol = abi.decode_string(read(spec.protocol, spec.address, "symbol()"))
        decimals = abi.decode_uint(read(spec.protocol, spec.address, "decimals()"))
    except abi.AbiError as exc:
        raise ContractReadError(
            f"{spec.address} does not answer as a token: {exc}"
        ) from exc

    if symbol != spec.symbol:
        raise ContractReadError(
            f"{spec.address} reports symbol {symbol!r}, registry expects "
            f"{spec.symbol!r} — the entry is wrong, or the address has changed"
        )
    if decimals != spec.decimals:
        raise ContractReadError(
            f"{spec.address} reports {decimals} decimals, registry expects "
            f"{spec.decimals} — every scaled reading from it would be wrong "
            f"by a factor of {10 ** abs(decimals - spec.decimals):,}"
        )


def total_supply(spec: ContractSpec) -> float:
    """Circulating supply, in whole units."""
    raw = abi.decode_uint(read(spec.protocol, spec.address, "totalSupply()"))
    return abi.scale(raw, spec.decimals)


def native_balance(spec: ContractSpec) -> float:
    """Native coin held by the contract, in whole units.

    For a wrapper this is the backing: every wrapped token should be matched by
    one unit of native coin locked here.
    """
    from src.blockchain import rpc

    try:
        result, _ = rpc.call(spec.protocol, "eth_getBalance", [spec.address, "latest"])
    except (rpc.RpcUnavailable, rpc.RpcShapeError) as exc:
        raise ContractReadError(f"eth_getBalance on {spec.address}: {exc}") from exc
    return abi.scale(rpc._as_int(result, "eth_getBalance"), 18)


def backing_ratio(spec: ContractSpec) -> tuple[float, float, float]:
    """Native backing per unit of wrapped supply, read within one block.

    Returns (ratio, supply, backing).

    The block guard is the point. The public endpoint serves `eth_call` at the
    chain head only — it cannot be pinned to a block number — and blocks arrive
    about every second, so two sequential reads routinely straddle a boundary. A
    wrap or unwrap landing between them would move one number and not the other,
    and the *ratio* is precisely what is being monitored for deviation. Reading
    1.0002 because the chain advanced mid-measurement is indistinguishable, in
    the feature store, from reading 1.0002 because the wrapper is
    over-collateralised.

    So the height is checked either side, and a reading that spans a block is
    refused rather than stored. Refusing costs one gap in an hourly series;
    storing it costs a permanent false deviation in the baseline.
    """
    from src.blockchain import rpc

    if spec.kind is not ContractKind.WRAPPED_NATIVE:
        raise ContractReadError(f"{spec.symbol} is not a wrapper; it has no backing ratio")

    # One round trip, bracketed by the block height at both ends. Sequential
    # reads cannot do this: the throttle alone guarantees they span two or three
    # blocks, so the guard would reject every single reading and the metric would
    # be uncollectable rather than merely imprecise.
    try:
        results, _ = rpc.batch(
            spec.protocol,
            [
                ("eth_blockNumber", []),
                ("eth_call", [
                    {"to": spec.address, "data": abi.encode_call("totalSupply()")},
                    "latest",
                ]),
                ("eth_getBalance", [spec.address, "latest"]),
                ("eth_blockNumber", []),
            ],
        )
        before = rpc._as_int(results[0], "eth_blockNumber")
        supply = abi.scale(abi.decode_uint(results[1]), spec.decimals)
        backing = abi.scale(rpc._as_int(results[2], "eth_getBalance"), 18)
        after = rpc._as_int(results[3], "eth_blockNumber")
    except (rpc.RpcUnavailable, rpc.RpcShapeError, abi.AbiError) as exc:
        raise ContractReadError(str(exc)) from exc

    if before != after:
        raise ContractReadError(
            f"{spec.symbol}: supply and backing were read across blocks "
            f"{before}-{after}; a wrap settling between them would show as a "
            f"backing deviation that never happened"
        )
    if supply <= 0:
        raise ContractReadError(f"{spec.symbol}: zero supply, so backing has no ratio")

    return backing / supply, supply, backing


def main() -> None:
    """`python -m src.blockchain.contracts --verify`"""
    import argparse

    from rich.console import Console

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="check every entry on chain")
    args = parser.parse_args()
    console = Console()

    if not args.verify:
        parser.error("nothing to do; pass --verify")

    for spec in CONTRACTS:
        try:
            verify(spec)
            console.print(
                f"[green]ok[/green]   {spec.symbol:<8} {spec.address}  "
                f"({spec.protocol}, {spec.kind.value})"
            )
        except ContractReadError as exc:
            console.print(f"[red]FAIL[/red] {spec.symbol:<8} {exc}")


if __name__ == "__main__":
    main()
