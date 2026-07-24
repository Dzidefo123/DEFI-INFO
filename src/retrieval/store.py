from __future__ import annotations

import json
from pathlib import Path

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import settings

# BM25 has no persistent server, so the corpus is mirrored to disk and the
# index is rebuilt in-process at query time.
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


def write_index(docs: list[Document]) -> Chroma:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
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
    """
    store = vector_store()
    existing = store.get(where={"protocol": protocol_key})
    if existing["ids"]:
        store.delete(ids=existing["ids"])
    if docs:
        store.add_documents(docs, ids=[d.metadata["doc_id"] for d in docs])
    _merge_corpus(protocol_key, docs)


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
