"""Discovery + sitemap-parsing tests. Pure — _index_text is monkeypatched, no network."""

import pytest

from src.ingest import sources
from src.ingest.sources import Target, parse_sitemap
from src.protocols import SourceType, get_protocol

_LLMS_TXT = """\
# Hyperliquid docs

- [Funding](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding.md)
- [Order types](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/order-types.md)
- [Dual block](https://hyperliquid.gitbook.io/hyperliquid-docs/hyperevm/dual-block.md)
- [Brand kit](https://hyperliquid.gitbook.io/hyperliquid-docs/brand-kit.md)
"""


@pytest.fixture(autouse=True)
def _canned_index(monkeypatch):
    monkeypatch.setattr(sources, "_index_text", lambda url: _LLMS_TXT)


# --- llms.txt discovery + protocol partitioning -------------------------


def test_discover_hyperliquid_keeps_only_its_pages():
    targets = sources.discover(get_protocol("hyperliquid"))
    urls = {t.url for t in targets}

    assert "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding" in urls
    # HyperEVM page belongs to the other protocol; excluded page is dropped.
    assert not any("hyperevm" in u for u in urls)
    assert not any("brand-kit" in u for u in urls)
    assert all(t.protocol == "hyperliquid" for t in targets)


def test_discover_hyperevm_keeps_only_its_pages():
    targets = sources.discover(get_protocol("hyperevm"))
    assert [t.url for t in targets] == [
        "https://hyperliquid.gitbook.io/hyperliquid-docs/hyperevm/dual-block"
    ]
    assert targets[0].protocol == "hyperevm"


@pytest.mark.parametrize(
    "path",
    [
        "/resources/terms-of-service",
        "/resources/privacy-policy",
        "/resources/usde-terms-and-conditions",   # suffix, not a whole segment
        "/resources/usde-mint-user-agreement",
        "/resources/general-risk-disclosures",
        "/brand-kit",
        "/resources/audits",
    ],
)
def test_boilerplate_pages_are_excluded(path):
    """Legal boilerplate was 26% of Ethena's chunks and can never be an answer:
    the TAX_LEGAL guardrail refuses legal questions before retrieval runs."""
    assert sources._EXCLUDE.search(f"https://docs.example.fi{path}.md")


@pytest.mark.parametrize(
    "path",
    [
        "/protocol-overview/risks/funding-risk",
        "/protocol-overview/risks/liquidation-risk",
        "/protocol-overview/risks/margin-collateral-risks",
        "/backing-assets/crypto-basis-trade",
        "/technical-design/use-of-oracles",
        "/trading/fees",
        "/hypercore/staking",
    ],
)
def test_substantive_pages_are_not_excluded(path):
    """The risk *mechanics* pages must survive — only the disclaimers go.

    `risk-disclosures` is deliberately narrower than `risk`: Ethena's
    protocol-overview/risks/* pages are the substance behind six golden cases.
    """
    assert not sources._EXCLUDE.search(f"https://docs.example.fi{path}.md")


def test_llms_targets_fetch_the_md_variant():
    t = sources.discover(get_protocol("hyperevm"))[0]
    assert t.fetch_url.endswith(".md")
    assert t.url == t.fetch_url.removesuffix(".md")
    assert t.kind is SourceType.LLMS_TXT


# --- sitemap parsing ----------------------------------------------------

_URLSET = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://x.io/a</loc></url>
  <url><loc>https://x.io/b</loc></url>
</urlset>"""

_INDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://x.io/sm-1.xml</loc></sitemap>
  <sitemap><loc>https://x.io/sm-2.xml</loc></sitemap>
</sitemapindex>"""


def test_parse_urlset():
    result = parse_sitemap(_URLSET)
    assert result.is_index is False
    assert result.locs == ("https://x.io/a", "https://x.io/b")


def test_parse_sitemap_index_is_flagged():
    result = parse_sitemap(_INDEX)
    assert result.is_index is True
    assert result.locs == ("https://x.io/sm-1.xml", "https://x.io/sm-2.xml")
