"""Build or refresh the multi-protocol knowledge index.

    python -m src.ingest.build_index                  # full crawl, all protocols
    python -m src.ingest.build_index --protocol hyperliquid  # re-crawl one, in place

The full crawl rebuilds the whole index from scratch. `--protocol` re-crawls a
single protocol and upserts it, leaving the others untouched — this is the
entrypoint a scheduler points at for freshness. Schedule it with cron or the
`/schedule` routine, e.g. a nightly `--protocol hyperliquid`; no scheduler
dependency lives in the app.
"""

from __future__ import annotations

import argparse

from src.config import settings
from src.ingest.chunk import chunk_pages
from src.ingest.fetch import fetch_all
from src.ingest.sources import discover
from src.protocols import enabled_protocols, get_protocol
from src.retrieval.store import upsert_protocol, write_index


def _crawl(protocol) -> list:
    print(f"[{protocol.key}] discovering pages ({protocol.source.type.value})...")
    targets = discover(protocol)
    print(f"[{protocol.key}] {len(targets)} pages. Fetching...")
    pages = fetch_all(targets)
    docs = chunk_pages(pages)
    print(f"[{protocol.key}] {len(pages)} pages -> {len(docs)} chunks.")
    return docs


def full_rebuild() -> None:
    all_docs: list = []
    for protocol in enabled_protocols():
        all_docs.extend(_crawl(protocol))
    if not all_docs:
        raise SystemExit("No pages fetched — check network access to the docs hosts.")

    print(f"\nEmbedding + indexing {len(all_docs)} chunks...")
    write_index(all_docs)
    print(f"Index written to {settings.chroma_dir}. Run: python -m src.app")


def recrawl_one(key: str) -> None:
    protocol = get_protocol(key)  # raises on unknown/non-whitelisted key
    docs = _crawl(protocol)
    if not docs:
        raise SystemExit(f"No pages fetched for {key} — nothing upserted.")

    print(f"\nUpserting {len(docs)} chunks for {key}...")
    upsert_protocol(key, docs)
    print(f"Refreshed {key} in {settings.chroma_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        help="re-crawl a single whitelisted protocol in place (default: full rebuild)",
    )
    args = parser.parse_args()

    if args.protocol:
        recrawl_one(args.protocol)
    else:
        full_rebuild()


if __name__ == "__main__":
    main()
