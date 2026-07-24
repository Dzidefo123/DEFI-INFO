from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import settings
from src.retrieval.rerank import rerank
from src.retrieval.store import load_corpus, vector_store

RRF_K = 60  # standard reciprocal-rank-fusion damping constant


@lru_cache(maxsize=1)
def _store():
    """The dense vector store — loaded once; the embedding model load is slow."""
    return vector_store()


@lru_cache(maxsize=1)
def _corpus() -> tuple[Document, ...]:
    """The full BM25 corpus, mirrored from disk. Tuple so it is hashable/cacheable."""
    return tuple(load_corpus())


def _filter_by_protocol(
    docs: Iterable[Document], protocols: frozenset[str] | None
) -> list[Document]:
    """Keep only chunks tagged with one of `protocols`; `None` keeps everything."""
    if not protocols:
        return list(docs)
    return [d for d in docs if d.metadata.get("protocol") in protocols]


@lru_cache(maxsize=16)
def _bm25(protocols: frozenset[str] | None) -> BM25Retriever | None:
    """BM25 over the corpus restricted to `protocols`.

    BM25 has no server and no metadata filter — the index *is* the document set —
    so a filtered query needs its own index built over the matching subset. One
    per protocol combination, cached, since indexing is not free. Returns None
    when the filter selects nothing, so the caller contributes no sparse hits
    rather than trying to build an empty index.
    """
    docs = _filter_by_protocol(_corpus(), protocols)
    if not docs:
        return None
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = settings.retrieve_k
    return retriever


def _where(protocols: frozenset[str] | None) -> dict | None:
    """Chroma metadata filter for the dense side; `None` means no filter."""
    if not protocols:
        return None
    keys = sorted(protocols)
    if len(keys) == 1:
        return {"protocol": keys[0]}
    return {"protocol": {"$in": keys}}


def _rrf(ranked_lists: list[list[Document]], k: int) -> list[Document]:
    scores: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs):
            doc_id = doc.metadata["doc_id"]
            by_id[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [by_id[i] for i in ordered[:k]]


def hybrid_search(
    query: str,
    k: int | None = None,
    use_rerank: bool | None = None,
    protocols: Iterable[str] | None = None,
) -> list[Document]:
    """Retrieve -> fuse -> rerank, optionally scoped to a set of protocols.

    Dense alone misses exact identifiers users paste verbatim ("HLP", "ALO",
    "isolated margin"); BM25 alone misses paraphrase. Fusion needs no score
    normalization between the two, which is why it beats a weighted blend here.

    Fusion maximizes recall into a shortlist; the cross-encoder then supplies
    the precision. `use_rerank` exists so eval can ablate the last stage.

    `protocols` scopes both retrievers to the matching chunks — the dense side
    via a Chroma metadata filter, the sparse side via a per-protocol BM25 index.
    `None` searches every protocol, which is the pre-namespacing behavior.
    """
    k = k or settings.context_k
    use_rerank = settings.rerank_enabled if use_rerank is None else use_rerank
    proto_set = frozenset(protocols) if protocols else None

    dense = _store().similarity_search(
        query, k=settings.retrieve_k, filter=_where(proto_set)
    )
    bm25 = _bm25(proto_set)
    sparse = bm25.invoke(query) if bm25 is not None else []

    fused = _rrf([dense, sparse], k=settings.fuse_k)

    if not use_rerank:
        return fused[:k]
    return rerank(query, fused, k=k)
