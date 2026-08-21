from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import settings

# BM25 has no persistent server, so the corpus is mirrored to disk and the
# index is rebuilt in-process at query time.
#
# That mirror is also the DECLARED TRUTH about what the corpus contains. The two
# halves of hybrid search read different stores — BM25 reads this file, dense
# reads Chroma — so if they disagree, "hybrid" is searching two different
# corpora and the disagreement is invisible at query time. `index_drift`
# exists to make that checkable; see it for the failure this cost us once.
_CORPUS = Path(settings.chroma_dir) / "corpus.jsonl"


def embeddings() -> FastEmbedEmbeddings:
    """Local ONNX embeddings — no embedding API key, no per-query cost."""
    return FastEmbedEmbeddings(model_name=settings.embed_model)


def vector_store() -> Chroma:
    return Chroma(
        collection_name=settings.collection,
        embedding_function=embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def drop_collection() -> None:
    """Delete the Chroma collection if it exists. Safe to call on a fresh dir.

    Uses the raw client rather than `vector_store()` so a drop does not pay the
    ONNX embedding-model load it has no use for.
    """
    import chromadb
    from chromadb.errors import NotFoundError

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(settings.collection)
    except (NotFoundError, ValueError):
        pass  # never built, or already dropped


def write_index(docs: list[Document]) -> Chroma:
    """Rebuild the whole index from scratch: drop the collection, then write.

    The drop is load-bearing. `Chroma.from_documents` against an existing
    persist_directory UPSERTS by id — it overwrites ids that still exist and
    leaves every id that doesn't. So any chunk whose `doc_id` changed between
    rebuilds survives forever: a page that was removed from the docs, a section
    that shifted its chunk counter, a page that a widened `path_prefixes` moved
    to a different protocol, or the entire pre-namespacing corpus whose ids
    lacked the `{protocol}:` prefix.

    Left unfixed this accumulated 1157 orphans against 1091 live chunks — 661 of
    them untagged (so they answered only *unfiltered* queries, i.e. exactly the
    generic questions the router declines to scope), and 496 tagged, including
    the same URL indexed under two protocols at once. That last one is the
    precise failure the protocol filter exists to prevent, reintroduced
    underneath it by the writer.
    """
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    drop_collection()
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings(),
        collection_name=settings.collection,
        persist_directory=str(settings.chroma_dir),
        ids=[d.metadata["doc_id"] for d in docs],
    )
    with _CORPUS.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps({"text": d.page_content, "meta": d.metadata}) + "\n")
    return store


def _merge_corpus(protocol_key: str, docs: list[Document]) -> None:
    """Rewrite corpus.jsonl with `protocol_key`'s rows replaced by `docs`."""
    rows: list[dict] = []
    if _CORPUS.exists():
        with _CORPUS.open(encoding="utf-8") as fh:
            rows = [
                row
                for line in fh
                if (row := json.loads(line))["meta"].get("protocol") != protocol_key
            ]
    rows.extend({"text": d.page_content, "meta": d.metadata} for d in docs)

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    with _CORPUS.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def upsert_protocol(protocol_key: str, docs: list[Document]) -> None:
    """Replace one protocol's chunks in place, leaving other protocols intact.

    Backs per-protocol scheduled re-crawls: refresh a fast-moving protocol
    without re-embedding the whole corpus. In-process retriever caches are not
    invalidated — this is meant to run in a fresh CLI process.

    Deletion is by metadata filter, so it reaches only rows already tagged with
    this protocol. Rows written before `protocol` metadata existed, or tagged
    with a protocol the registry has since reassigned, are outside any single
    protocol's sweep — `index_drift` finds those, `reindex_from_corpus` clears
    them.
    """
    store = vector_store()
    existing = store.get(where={"protocol": protocol_key})
    if existing["ids"]:
        store.delete(ids=existing["ids"])
    if docs:
        store.add_documents(docs, ids=[d.metadata["doc_id"] for d in docs])
    _merge_corpus(protocol_key, docs)


def _drift(indexed_ids: Iterable[str], corpus_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """(orphaned, missing) — pure set logic, so it is testable without Chroma.

    orphaned = in the vector index but not in the corpus mirror: dense search can
    return them, BM25 can never match them, and no rebuild will remove them.
    missing  = in the mirror but never embedded: BM25-only, invisible to dense.
    """
    indexed, corpus = set(indexed_ids), set(corpus_ids)
    return sorted(indexed - corpus), sorted(corpus - indexed)


def corpus_ids() -> list[str]:
    """Every `doc_id` in the on-disk corpus mirror, in file order."""
    if not _CORPUS.exists():
        return []
    with _CORPUS.open(encoding="utf-8") as fh:
        return [json.loads(line)["meta"]["doc_id"] for line in fh if line.strip()]


def index_drift() -> tuple[list[str], list[str]]:
    """Reconcile the vector index against the corpus mirror.

    Both must hold exactly the same `doc_id` set. When they don't, the sparse and
    dense retrievers are searching different corpora and every recall number
    measured on the pair is measuring something nobody specified. Cheap enough to
    assert after any index write.
    """
    import chromadb

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        collection = client.get_collection(settings.collection)
    except Exception:
        return [], corpus_ids()
    return _drift(collection.get(include=[])["ids"], corpus_ids())


def reindex_from_corpus() -> int:
    """Rebuild the vector index from the corpus mirror. No crawl, no network.

    The mirror is the declared truth, so drift is repairable by re-embedding it
    rather than re-fetching every docs site. Embeddings are local ONNX, so this
    costs CPU and nothing else — which makes "repair the index" a step someone
    will actually take instead of deferring behind a full crawl.
    """
    docs = load_corpus()
    if not docs:
        raise FileNotFoundError(f"corpus mirror at {_CORPUS} is empty; nothing to reindex")
    write_index(docs)
    return len(docs)


def load_corpus() -> list[Document]:
    if not _CORPUS.exists():
        raise FileNotFoundError(
            f"No index at {settings.chroma_dir}. Run: python -m src.ingest.build_index"
        )
    docs = []
    with _CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            docs.append(Document(page_content=row["text"], metadata=row["meta"]))
    return docs
