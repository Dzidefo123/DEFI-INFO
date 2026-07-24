"""Shared HTTP basics for the crawler: a single User-Agent and a client factory.

Kept in its own module so `robots`, `sources`, and `fetch` can all share the UA
without importing one another (they would otherwise form a cycle).
"""

from __future__ import annotations

import httpx

# Honest, identifiable UA. A docs crawler that respects robots.txt should say
# who it is so a site operator can allow or block it deliberately.
UA = "defi-docs-indexer/0.1 (+docs indexer; respects robots.txt)"


def client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": UA}, follow_redirects=True, timeout=30.0
    )


def get_text(url: str) -> str:
    """One-off GET for index files (llms.txt, sitemap.xml). Raises on HTTP error."""
    with client() as c:
        resp = c.get(url)
        resp.raise_for_status()
        return resp.text
