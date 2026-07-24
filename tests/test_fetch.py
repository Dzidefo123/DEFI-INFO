"""Fetch-layer gating tests: whitelist + robots run before any network I/O."""

import pytest

from src.ingest import fetch
from src.ingest.fetch import Page, _extract, fetch_target
from src.ingest.sources import Target
from src.protocols import SourceNotWhitelisted, SourceType

_HL = "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding"


def _target(url, kind=SourceType.LLMS_TXT):
    return Target("hyperliquid", "Funding", url.removesuffix(".md"), url, kind)


def test_off_whitelist_target_refused_before_fetch():
    # client is None on purpose: assert_allowed must raise before any GET.
    bad = _target("https://evil.com/hyperliquid-docs/funding.md")
    with pytest.raises(SourceNotWhitelisted):
        fetch_target(None, bad)


def test_robots_disallowed_target_is_skipped(monkeypatch):
    monkeypatch.setattr(fetch.robots, "allowed", lambda url: False)
    # Whitelisted URL, but robots says no -> None, and no client call.
    assert fetch_target(None, _target(_HL + ".md")) is None


def test_extract_markdown_cleans_body():
    body = "> For the complete documentation index see llms.txt\n" + ("word " * 60)
    out = _extract(_target(_HL + ".md"), body)
    assert "documentation index" not in out
    assert len(out) > 200


def test_extract_html_source_not_supported():
    with pytest.raises(NotImplementedError):
        _extract(_target(_HL, kind=SourceType.SITEMAP), "<html>...</html>")


def test_page_shape():
    p = Page(url=_HL, title="Funding", text="x" * 300)
    assert p.url == _HL and p.title == "Funding"
