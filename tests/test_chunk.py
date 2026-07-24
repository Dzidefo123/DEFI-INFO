"""Chunk-tagging tests — pure, no network.

PR 1 makes every chunk carry the `protocol` it belongs to (derived from the page
URL via the registry) so retrieval can filter per protocol in PR 2.
"""

from src.ingest.chunk import chunk_pages
from src.ingest.fetch import Page

_MD = (
    "# Funding\n\n"
    "Funding is exchanged between longs and shorts every hour.\n\n"
    "## Overview\n\n"
    "The funding rate is derived from the difference between mark and oracle "
    "price, clamped to a cap.\n"
)


def _page(url: str, title: str = "Funding") -> Page:
    return Page(url=url, title=title, text=_MD)


def test_chunks_are_tagged_with_protocol_and_source():
    url = "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding"
    docs = chunk_pages([_page(url)])

    assert docs, "expected at least one chunk"
    for doc in docs:
        assert doc.metadata["protocol"] == "hyperliquid"
        assert doc.metadata["source"] == url


def test_doc_ids_are_protocol_namespaced_and_unique():
    url = "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding"
    docs = chunk_pages([_page(url)])

    ids = [d.metadata["doc_id"] for d in docs]
    assert all(i.startswith("hyperliquid:") for i in ids)
    assert len(ids) == len(set(ids))


def test_hyperevm_pages_get_the_hyperevm_tag():
    url = "https://hyperliquid.gitbook.io/hyperliquid-docs/hyperevm/dual-block-architecture"
    docs = chunk_pages([_page(url, title="Dual-block architecture")])

    assert docs
    assert all(d.metadata["protocol"] == "hyperevm" for d in docs)


def test_off_whitelist_pages_are_dropped():
    docs = chunk_pages([_page("https://docs.aave.com/v3/funding", title="Aave")])
    assert docs == []


def test_mixed_batch_keeps_only_whitelisted_and_tags_each():
    docs = chunk_pages(
        [
            _page("https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding"),
            _page("https://hyperliquid.gitbook.io/hyperliquid-docs/hyperevm/blocks"),
            _page("https://scam-hyperliquid.com/seed-recovery", title="scam"),
        ]
    )

    protos = {d.metadata["protocol"] for d in docs}
    assert protos == {"hyperliquid", "hyperevm"}
