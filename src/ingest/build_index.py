"""Build or refresh the multi-protocol knowledge index.

    python -m src.ingest.build_index                  # full crawl, all protocols
    python -m src.ingest.build_index --protocol hyperliquid  # re-crawl one, in place
    python -m src.ingest.build_index --verify         # check index vs corpus, no writes
    python -m src.ingest.build_index --repair         # re-embed the corpus mirror, no crawl

The full crawl rebuilds the whole index from scratch. `--protocol` re-crawls a
single protocol and upserts it, leaving the others untouched — this is the
entrypoint a scheduler points at for freshness. Schedule it with cron or the
`/schedule` routine, e.g. a nightly `--protocol hyperliquid`; no scheduler
dependency lives in the app.

`--verify` and `--repair` cover the failure mode where the two halves of hybrid
search drift apart: dense reads Chroma, sparse reads the corpus mirror, and
nothing at query time notices when they hold different documents. `--repair`
re-embeds the mirror locally, so fixing drift never requires re-crawling the
docs sites. See `store.write_index` for how the drift arose.
"""

from __future__ import annotations

import argparse

from src.config import settings
from src.ingest.chunk import chunk_pages
from src.ingest.fetch import fetch_all
from src.ingest.sources import discover
from src.protocols import enabled_protocols, get_protocol
from src.retrieval.store import (
    corpus_ids,
    index_drift,
    reindex_from_corpus,
    upsert_protocol,
    write_index,
)


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


def verify() -> int:
    """Report drift between the vector index and the corpus mirror.

    Returns a process exit code so a scheduled crawl can gate on it.
    """
    orphaned, missing = index_drift()
    total = len(corpus_ids())
    if not orphaned and not missing:
        print(f"OK — vector index and corpus mirror agree on all {total} chunks.")
        return 0

    print(f"DRIFT — corpus mirror holds {total} chunks.")
    if orphaned:
        print(f"  {len(orphaned)} orphaned (in the vector index, not in the corpus):")
        for doc_id in orphaned[:5]:
            print(f"    {doc_id}")
        if len(orphaned) > 5:
            print(f"    ... and {len(orphaned) - 5} more")
    if missing:
        print(f"  {len(missing)} missing (in the corpus, never embedded):")
        for doc_id in missing[:5]:
            print(f"    {doc_id}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
    print("\nRepair without re-crawling: python -m src.ingest.build_index --repair")
    return 1


def repair() -> None:
    print("Re-embedding the corpus mirror into a fresh collection...")
    n = reindex_from_corpus()
    print(f"Reindexed {n} chunks.")
    verify()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        help="re-crawl a single whitelisted protocol in place (default: full rebuild)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check the vector index against the corpus mirror; no writes",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="rebuild the vector index from the corpus mirror; no crawl, no network",
    )
    args = parser.parse_args()

    if args.verify:
        raise SystemExit(verify())
    if args.repair:
        repair()
    elif args.protocol:
        recrawl_one(args.protocol)
    else:
        full_rebuild()


if __name__ == "__main__":
    main()
