"""Minimal ABI encoding and decoding for read-only contract calls.

Only what `eth_call` against a standard token needs: a four-byte selector, an
address argument, and the three return shapes those functions use. This is
deliberately not a general ABI library — the system reads a fixed, whitelisted
set of standard functions, and a decoder that can parse anything is a decoder
that can be pointed at anything.

**Selectors are constants, not computed.** A selector is the first four bytes of
keccak-256 of the signature, and keccak-256 is not in the standard library
(`hashlib.sha3_256` is the NIST variant, which pads differently and produces a
different digest — a subtle way to generate confidently wrong selectors). Rather
than add a dependency or hand-roll a hash, the standard signatures are listed
below as published constants that anyone can check against a keccak calculator.
Adding a non-standard function means adding its selector here, deliberately.

**Decoding refuses rather than guesses.** An empty return, a short return, or a
malformed dynamic offset raises. A contract that does not implement a function
returns `0x`, and reading that as zero would turn "this address is not a token"
into "this token has no supply" — a number with nothing behind it, at the one
layer that feeds the feature store.
"""

from __future__ import annotations

# Published four-byte selectors for the standard read functions. Verifiable:
# each is the first four bytes of keccak-256 of the signature to its left.
SELECTORS: dict[str, str] = {
    "name()": "0x06fdde03",
    "symbol()": "0x95d89b41",
    "decimals()": "0x313ce567",
    "totalSupply()": "0x18160ddd",
    "balanceOf(address)": "0x70a08231",
}

_WORD = 32


class AbiError(ValueError):
    """A return value that is not the shape the function promised."""


def selector(signature: str) -> str:
    try:
        return SELECTORS[signature]
    except KeyError:
        raise AbiError(
            f"no selector registered for {signature!r}; add it to SELECTORS "
            f"deliberately rather than computing one at call time"
        ) from None


def encode_call(signature: str, address_arg: str | None = None) -> str:
    """Calldata for a no-argument read, or one taking a single address."""
    data = selector(signature)
    if address_arg is None:
        return data
    clean = address_arg.lower().removeprefix("0x")
    if len(clean) != 40:
        raise AbiError(f"{address_arg!r} is not a 20-byte address")
    return data + clean.rjust(_WORD * 2, "0")


def _words(hexdata: str) -> bytes:
    if not isinstance(hexdata, str) or not hexdata.startswith("0x"):
        raise AbiError(f"expected hex data, got {hexdata!r}")
    body = hexdata[2:]
    if not body:
        # `0x` is what a call to a function the contract does not implement
        # returns. It is an absence, not a zero.
        raise AbiError("empty return: the contract does not implement this function")
    try:
        return bytes.fromhex(body)
    except ValueError:
        raise AbiError("return data is not valid hex") from None


def decode_uint(hexdata: str) -> int:
    raw = _words(hexdata)
    if len(raw) < _WORD:
        raise AbiError(f"expected a 32-byte word, got {len(raw)} bytes")
    return int.from_bytes(raw[:_WORD], "big")


def decode_string(hexdata: str) -> str:
    """A dynamic string, or a `bytes32` one.

    Older tokens returned a fixed 32-byte symbol rather than a dynamic string,
    and both shapes are still in circulation. Distinguishing them by length is
    the standard heuristic and is safe here because a dynamic string is never
    exactly one word.
    """
    raw = _words(hexdata)

    if len(raw) == _WORD:
        return raw.rstrip(b"\x00").decode("utf-8", "replace")

    if len(raw) < _WORD * 2:
        raise AbiError(f"string return too short: {len(raw)} bytes")
    offset = int.from_bytes(raw[:_WORD], "big")
    if offset + _WORD > len(raw):
        raise AbiError("string offset points past the end of the return data")
    length = int.from_bytes(raw[offset : offset + _WORD], "big")
    start = offset + _WORD
    if start + length > len(raw):
        raise AbiError("string length runs past the end of the return data")
    return raw[start : start + length].decode("utf-8", "replace")


def scale(value: int, decimals: int) -> float:
    """Base units to whole units."""
    if decimals < 0 or decimals > 77:
        raise AbiError(f"implausible decimals: {decimals}")
    return value / (10**decimals)
