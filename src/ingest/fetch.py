from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from src.ingest import robots
from src.ingest.http import UA
from src.ingest.sources import Target
from src.protocols import SourceType, assert_allowed

# GitBook prepends a boilerplate llms.txt/Markdown pointer to every .md page.
_BOILERPLATE = re.compile(r"^\s*>\s*For the complete documentation index.*?\n", re.DOTALL)


@dataclass
class Page:
    url: str          # canonical human URL, used for citations
    title: str
    text: str


def _clean(md: str) -> str:
    md = _BOILERPLATE.sub("", md, count=1)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _extract(target: Target, body: str) -> str:
    if target.kind is SourceType.LLMS_TXT:
        return _clean(body)
    # sitemap/gitbook pages are HTML; turning them into clean text needs an
    # HTML-to-text dependency that isn't approved yet. No whitelisted protocol
    # reaches here, so fail loudly rather than index raw markup.
    raise NotImplementedError(
        f"HTML extraction for {target.kind.value} sources is not enabled "
        "(needs an HTML-to-text dependency; ask before adding)."
    )


def fetch_target(client: httpx.Client, target: Target) -> Page | None:
    # Whitelist first, then robots — two independent gates, both before any GET.
    assert_allowed(target.fetch_url)
    if not robots.allowed(target.fetch_url):
        print(f"  skip {target.fetch_url}: disallowed by robots.txt")
        return None

    try:
        resp = client.get(target.fetch_url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"  skip {target.fetch_url}: {exc}")
        return None

    text = _extract(target, resp.text)
    # A near-empty body means a stub or an error shell; indexing it creates a
    # chunk that retrieves but says nothing.
    if len(text) < 200:
        print(f"  skip {target.fetch_url}: body too short ({len(text)} chars)")
        return None

    return Page(url=target.url, title=target.title, text=text)


def fetch_all(targets: list[Target]) -> list[Page]:
    out: list[Page] = []
    with httpx.Client(headers={"User-Agent": UA}) as client:
        for target in targets:
            page = fetch_target(client, target)
            if page:
                print(f"  ok   {page.title} ({len(page.text)} chars)")
                out.append(page)
    return out
