"""Protocol registry and source whitelist.

Single source of truth for which protocols the assistant may crawl, index, and
answer about. Everything downstream reads from here — ingestion (which sites to
fetch), retrieval (the per-protocol metadata filter), the router's protocol
axis, and live-data tool dispatch.

The whitelist is a security boundary, not a convenience. Crawling arbitrary
crypto sites is how an assistant ends up indexing a phishing clone of a docs
page and then citing it as authoritative — in DeFi that gets a user's funds
drained. Only hosts listed in a protocol's `allowed_domains` are ever fetched;
`assert_allowed` / `is_allowed_url` are the enforcement points the crawler must
call before every request.

This module is intentionally pure data + lookups: no network, no model, no I/O.
Adding a protocol is a matter of appending to `_PROTOCOLS` and rebuilding the
index — nothing else in the codebase hardcodes a protocol.
"""

from __future__ import annotations

from enum import Enum
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class SourceType(str, Enum):
    """How a protocol's docs are discovered."""

    LLMS_TXT = "llms_txt"  # site publishes an llms.txt index of Markdown pages
    SITEMAP = "sitemap"    # standard sitemap.xml enumerates the pages
    GITBOOK = "gitbook"    # GitBook space; crawl rendered HTML (last resort)


class SourceNotWhitelisted(ValueError):
    """Raised when a URL outside every protocol's allowed_domains is fetched."""


def _norm_host(host: str) -> str:
    return host.strip().lower().lstrip(".")


def _host_matches(host: str, domain: str) -> bool:
    """True if `host` is `domain` or a subdomain of it — never a lookalike.

    "gitbook.io" matches "gitbook.io" and "docs.gitbook.io", but NOT
    "gitbook.io.evil.com" (different registrable domain) or "notgitbook.io"
    (substring, not a label boundary).
    """
    host = _norm_host(host)
    domain = _norm_host(domain)
    return host == domain or host.endswith("." + domain)


class DocSource(BaseModel):
    """Where and how a protocol's documentation is fetched."""

    model_config = ConfigDict(frozen=True)

    type: SourceType
    entrypoint: str                       # llms.txt URL, sitemap.xml URL, or space root
    allowed_domains: tuple[str, ...]      # hosts this protocol may be fetched from
    # When one domain hosts several protocols (e.g. Hyperliquid + HyperEVM share
    # hyperliquid.gitbook.io), pages under these path prefixes belong to this
    # protocol. Empty = the catch-all default for its domain(s). Ingestion uses
    # the longest matching prefix to assign each page.
    path_prefixes: tuple[str, ...] = ()

    @field_validator("allowed_domains")
    @classmethod
    def _normalize_domains(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("a DocSource must whitelist at least one domain")
        return tuple(_norm_host(d) for d in v)

    @model_validator(mode="after")
    def _entrypoint_is_whitelisted(self) -> "DocSource":
        host = urlparse(self.entrypoint).hostname or ""
        if not any(_host_matches(host, d) for d in self.allowed_domains):
            raise ValueError(
                f"entrypoint {self.entrypoint!r} is not within allowed_domains "
                f"{self.allowed_domains!r}"
            )
        return self


class Protocol(BaseModel):
    """A whitelisted protocol the assistant can index and answer about."""

    model_config = ConfigDict(frozen=True)

    key: str                          # stable slug; the value tagged onto every chunk
    name: str                         # display name for citations and copy
    category: str                     # grouping, e.g. "perps", "lending", "l1"
    aliases: tuple[str, ...] = ()     # other names the router may see ("HL", "Hyperliquid L1")
    source: DocSource
    live_tool: str | None = None      # key into the live-data tool registry (wired in PR 5)

    @field_validator("key")
    @classmethod
    def _key_is_slug(cls, v: str) -> str:
        if not v or v != v.lower() or " " in v:
            raise ValueError(f"protocol key must be a lowercase slug, got {v!r}")
        return v


# --- the whitelist ------------------------------------------------------
#
# Only protocols listed here are crawled, indexed, or answered about. The
# broader candidate roster lives in project memory, not in code, precisely so
# that "approved" means "someone added it here on purpose".

_PROTOCOLS: tuple[Protocol, ...] = (
    Protocol(
        key="hyperliquid",
        name="Hyperliquid",
        category="perps",
        aliases=("HL", "Hyperliquid L1", "HyperCore"),
        source=DocSource(
            type=SourceType.LLMS_TXT,
            entrypoint="https://hyperliquid.gitbook.io/hyperliquid-docs/llms.txt",
            allowed_domains=("hyperliquid.gitbook.io",),
        ),
        live_tool="hyperliquid",
    ),
    Protocol(
        key="hyperevm",
        name="HyperEVM",
        category="l1",
        aliases=("Hyperliquid EVM",),
        source=DocSource(
            type=SourceType.LLMS_TXT,
            # HyperEVM is documented inside the same GitBook space as Hyperliquid;
            # it is disambiguated from the L1 docs by path prefix, not by host.
            # The space does not keep HyperEVM under one root — the reference
            # pages sit under for-developers/, the user-facing ones under
            # onboarding/ and support/ — so every branch is listed explicitly.
            # Missing one silently tags those pages `hyperliquid`, which a
            # protocol-filtered search would then hide from HyperEVM questions.
            entrypoint="https://hyperliquid.gitbook.io/hyperliquid-docs/llms.txt",
            allowed_domains=("hyperliquid.gitbook.io",),
            path_prefixes=(
                "/hyperliquid-docs/hyperevm",
                "/hyperliquid-docs/for-developers/hyperevm",
                "/hyperliquid-docs/builder-tools/hyperevm-tools",
                "/hyperliquid-docs/onboarding/how-to-use-the-hyperevm",
                "/hyperliquid-docs/support/faq/hyperevm-issues",
            ),
        ),
        live_tool="hyperevm",
    ),
    Protocol(
        key="ethena",
        name="Ethena",
        category="stablecoin",
        aliases=("USDe", "sUSDe", "ENA"),
        source=DocSource(
            type=SourceType.LLMS_TXT,
            entrypoint="https://docs.ethena.fi/llms.txt",
            # Sole owner of its domain, so no path_prefixes: it is the catch-all
            # for docs.ethena.fi. Unlike the two above, nothing else is indexed
            # from this host.
            allowed_domains=("docs.ethena.fi",),
        ),
        # No live tool. Ethena's volatile numbers (USDe supply, sUSDe APY,
        # backing composition) would need their own read-only source; until one
        # is wired, live_data must say so rather than answer from another
        # protocol's API. See nodes._pick_live_protocol.
        live_tool=None,
    ),
)


# --- indexes (built once, validated at import) --------------------------


def _build_index() -> dict[str, Protocol]:
    by_key: dict[str, Protocol] = {}
    for proto in _PROTOCOLS:
        if proto.key in by_key:
            raise ValueError(f"duplicate protocol key: {proto.key!r}")
        by_key[proto.key] = proto
    return by_key


_BY_KEY = _build_index()


# --- public API ---------------------------------------------------------


def enabled_protocols() -> tuple[Protocol, ...]:
    """Every whitelisted protocol, in registry order."""
    return _PROTOCOLS


def protocol_keys() -> tuple[str, ...]:
    """The stable slugs used as chunk metadata tags and router labels."""
    return tuple(_BY_KEY)


def protocol_names() -> tuple[str, ...]:
    """Display names, in registry order."""
    return tuple(p.name for p in _PROTOCOLS)


def english_list(items: tuple[str, ...] | list[str]) -> str:
    """Join for prose: "A", "A and B", "A, B, and C"."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def coverage_phrase() -> str:
    """The whitelisted protocols as an English list, for user-facing copy.

    Every "here's what I cover" / "I can't help with that" message renders this
    instead of naming a protocol inline, so adding a protocol to `_PROTOCOLS`
    updates the agent's own description of itself. Copy that hardcodes one
    protocol goes stale the moment the whitelist grows, and a support agent that
    claims to be "the Hyperliquid agent" while answering Aave questions reads as
    compromised.
    """
    return english_list(protocol_names())


def is_known(key: str) -> bool:
    return key in _BY_KEY


def get_protocol(key: str) -> Protocol:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"unknown protocol {key!r}; whitelisted: {', '.join(_BY_KEY)}"
        ) from None


def is_allowed_url(url: str) -> bool:
    """True if `url`'s host is within some whitelisted protocol's domains."""
    host = urlparse(url).hostname or ""
    if not host:
        return False
    return any(
        _host_matches(host, d)
        for proto in _PROTOCOLS
        for d in proto.source.allowed_domains
    )


def assert_allowed(url: str) -> None:
    """Guard the crawler must call before fetching. Raises if off-whitelist."""
    if not is_allowed_url(url):
        raise SourceNotWhitelisted(
            f"refusing to fetch {url!r}: host is not in any whitelisted "
            f"protocol's allowed_domains"
        )


def protocol_for_url(url: str) -> Protocol | None:
    """Which protocol a page belongs to, or None if off-whitelist.

    A page is assigned to the protocol whose domain matches AND whose longest
    `path_prefixes` entry prefixes the URL path. Protocols with no prefixes are
    the catch-all default for their domain — chosen only when no more specific
    protocol claims the path.
    """
    parsed = urlparse(url)
    host, path = parsed.hostname or "", parsed.path or "/"
    if not host:
        return None

    best: Protocol | None = None
    best_len = -1  # -1 so an empty-prefix catch-all (len 0) still beats no match
    for proto in _PROTOCOLS:
        if not any(_host_matches(host, d) for d in proto.source.allowed_domains):
            continue
        prefixes = proto.source.path_prefixes or ("",)
        for prefix in prefixes:
            if path.startswith(prefix) and len(prefix) > best_len:
                best, best_len = proto, len(prefix)
    return best
