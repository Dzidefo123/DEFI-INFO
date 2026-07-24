"""Discover the doc pages for a whitelisted protocol.

Dispatches on the protocol's declared `SourceType`:

  - llms_txt: the site publishes an llms.txt index of Markdown pages, and serves
    a clean `<url>.md` for each. Preferred — the index can't drift from a
    hand-kept URL list, and the Markdown keeps the heading structure that
    section-aware chunking depends on.
  - sitemap: standard sitemap.xml (or a sitemap index) enumerates page URLs.
  - gitbook: crawl rendered HTML. Not enabled — needs an HTML-to-text
    dependency, and no whitelisted protocol requires it.

One index can list pages for several protocols (Hyperliquid and HyperEVM share a
GitBook space). Discovery therefore keeps only the pages the registry assigns to
the protocol being discovered, so each page is indexed once, under one protocol.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache

from src.ingest.http import UA, get_text  # noqa: F401  (UA re-exported for callers)
from src.protocols import (
    Protocol,
    SourceType,
    assert_allowed,
    enabled_protocols,
    is_allowed_url,
    protocol_for_url,
)

_LINK = re.compile(r"^\s*-\s*\[([^\]]+)\]\((https?://[^)]+\.md)\)", re.MULTILINE)

# Pages that retrieve well but never help a support question.
#
# The second group is legal boilerplate, and it earns its place on volume as
# much as on relevance: Ethena's four legal documents alone were 151 of its 578
# chunks — 26% of the protocol's corpus — of dense, formal prose that competes
# for retrieval on generic terms ("risk", "collateral", "redemption") while
# containing no mechanics. It can never be a correct answer either, because the
# TAX_LEGAL guardrail refuses legal questions *before* retrieval runs.
#
# Segment-anchored for the first group; the second is deliberately unanchored,
# because these appear as suffixes too ("usde-terms-and-conditions").
_EXCLUDE = re.compile(
    r"/(core-contributors|media-kit|brand-kit|audits|roadmap)\b"
    r"|(terms-of-service|terms-and-conditions|user-agreement"
    r"|privacy-policy|cookie-policy|risk-disclosures)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Target:
    """One page to fetch, already resolved to a single protocol."""

    protocol: str        # protocol key
    title: str
    url: str             # canonical human URL — the citation, and protocol owner
    fetch_url: str       # what to GET (e.g. the .md variant for GitBook)
    kind: SourceType     # how fetch should interpret the response body


# --- sitemap parsing (pure, unit-tested) --------------------------------


@dataclass(frozen=True)
class SitemapParse:
    is_index: bool       # True for <sitemapindex> (locs point to more sitemaps)
    locs: tuple[str, ...]


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]  # strip the {namespace} ElementTree prefixes


def parse_sitemap(xml_text: str) -> SitemapParse:
    """Extract <loc> URLs from a sitemap or sitemap index, namespace-agnostic."""
    root = ET.fromstring(xml_text)
    is_index = _localname(root.tag) == "sitemapindex"
    locs = tuple(
        el.text.strip()
        for el in root.iter()
        if _localname(el.tag) == "loc" and el.text and el.text.strip()
    )
    return SitemapParse(is_index=is_index, locs=locs)


# --- discovery ----------------------------------------------------------


@lru_cache(maxsize=8)
def _index_text(url: str) -> str:
    """Fetch an index file, deduped within a run — protocols sharing one llms.txt
    (Hyperliquid + HyperEVM) must not fetch it twice."""
    return get_text(url)


def _owned_by(url: str, protocol: Protocol) -> bool:
    """True if the registry assigns `url` to this protocol (and it's whitelisted)."""
    owner = protocol_for_url(url)
    return owner is not None and owner.key == protocol.key


def _discover_llms_txt(protocol: Protocol) -> list[Target]:
    text = _index_text(protocol.source.entrypoint)

    seen: set[str] = set()
    targets: list[Target] = []
    for title, md_url in _LINK.findall(text):
        md_url = md_url.strip()
        if _EXCLUDE.search(md_url):
            continue
        canonical = md_url.removesuffix(".md")
        # An llms.txt shared by several protocols lists all their pages; keep
        # only the ones this protocol owns.
        if not _owned_by(canonical, protocol) or canonical in seen:
            continue
        assert_allowed(md_url)
        seen.add(canonical)
        targets.append(
            Target(protocol.key, title.strip(), canonical, md_url, SourceType.LLMS_TXT)
        )

    if not targets:
        raise RuntimeError(
            f"{protocol.key}: parsed no owned pages from {protocol.source.entrypoint} "
            "— format changed, or the path_prefixes no longer match?"
        )
    return targets


def _discover_sitemap(protocol: Protocol) -> list[Target]:
    parse = parse_sitemap(_index_text(protocol.source.entrypoint))

    page_urls: list[str] = []
    if parse.is_index:
        for sitemap_url in parse.locs:
            if is_allowed_url(sitemap_url):  # never follow a sitemap off-whitelist
                page_urls.extend(parse_sitemap(_index_text(sitemap_url)).locs)
    else:
        page_urls = list(parse.locs)

    seen: set[str] = set()
    targets: list[Target] = []
    for url in page_urls:
        if not is_allowed_url(url) or not _owned_by(url, protocol) or url in seen:
            continue
        seen.add(url)
        title = url.rstrip("/").rsplit("/", 1)[-1] or protocol.name
        # Sitemap pages are HTML; fetch handles the (currently unsupported)
        # extraction and will say so loudly.
        targets.append(Target(protocol.key, title, url, url, SourceType.SITEMAP))
    return targets


def discover(protocol: Protocol) -> list[Target]:
    assert_allowed(protocol.source.entrypoint)
    kind = protocol.source.type
    if kind is SourceType.LLMS_TXT:
        return _discover_llms_txt(protocol)
    if kind is SourceType.SITEMAP:
        return _discover_sitemap(protocol)
    if kind is SourceType.GITBOOK:
        raise NotImplementedError(
            f"{protocol.key}: gitbook HTML crawl is not enabled "
            "(needs an HTML-to-text dependency; ask before adding)."
        )
    raise ValueError(f"unhandled source type: {kind!r}")


def discover_all() -> list[Target]:
    targets: list[Target] = []
    for protocol in enabled_protocols():
        targets.extend(discover(protocol))
    return targets
