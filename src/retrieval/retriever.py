from __future__ import annotations

import re

import threading
from collections.abc import Iterable
from functools import lru_cache

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import settings
from src.retrieval.rerank import rerank
from src.retrieval.store import load_corpus, vector_store

RRF_K = 60  # standard reciprocal-rank-fusion damping constant

# Serialises the lazy initialisation below.
#
# `lru_cache` memoises a result; it does NOT make computing that result atomic.
# Two threads can miss the cache simultaneously and both run the body. That was
# harmless while only one node retrieved — but the investigation branch fans
# specialist agents out in PARALLEL, and from C3 both the Research and Security
# agents retrieve. Two threads then built a Chroma client at once and the second
# died with "Could not connect to tenant default_tenant", taking the whole
# investigation with it.
#
# Reentrant because `_bm25` calls `_corpus` while holding it.
_INIT = threading.RLock()


@lru_cache(maxsize=1)
def _build_store():
    return vector_store()


@lru_cache(maxsize=1)
def _build_corpus() -> tuple[Document, ...]:
    return tuple(load_corpus())


def _store():
    """The dense vector store — loaded once; the embedding model load is slow.

    The lock is taken BEFORE the cache lookup, not inside the cached function.
    Guarding the body would only serialise two concurrent misses; both threads
    would still construct, and the race is in the construction.
    """
    with _INIT:
        return _build_store()


def _corpus() -> tuple[Document, ...]:
    """The full BM25 corpus, mirrored from disk. Tuple so it is hashable/cacheable."""
    with _INIT:
        return _build_corpus()


def _filter_by_protocol(
    docs: Iterable[Document], protocols: frozenset[str] | None
) -> list[Document]:
    """Keep only chunks tagged with one of `protocols`; `None` keeps everything."""
    if not protocols:
        return list(docs)
    return [d for d in docs if d.metadata.get("protocol") in protocols]


def _bm25(protocols: frozenset[str] | None) -> BM25Retriever | None:
    """Thread-safe wrapper; see `_store`. Building an index is not cheap, and two
    agents scoped to the same protocol would otherwise both build it."""
    with _INIT:
        return _build_bm25(protocols)



# Question scaffolding, removed before BM25 indexes or queries. Not a generic
# stopword list — it is specifically the words that make a question a question,
# plus articles and the commonest verbs.
#
# Without this the sparse leg spends its budget matching question FORM. Measured
# on "What does IOC mean?": BM25's top hits were "What does 'Action already
# expired' mean?", "What assets can a vault trade?" and three more chosen for
# overlapping on what/does/mean, while the chunk actually defining IOC —
# "Immediate or Cancel (IOC): an order that will be canceled if it is not
# immediately filled" — was absent from its top SIXTY. A support corpus is full
# of pages titled as questions, so the query form matches the wrong thing
# extremely well.
#
# Deliberately conservative. Anything that could be a ticker, an acronym or a
# protocol term stays: the sparse leg exists precisely to catch identifiers a
# user pastes verbatim, and a stopword list that eats "ALO" or "GTC" would
# remove the reason for having it.
_QUESTION_WORDS = frozenset("""
    a an the this that these those
    what which who whom whose when where why how
    is are was were be been being am
    do does did doing done
    have has had having
    can could will would shall should may might must
    of in on at to for with by from about into over under as
    i me my we our you your it its
    and or but if then than so
    mean means meaning explain tell show
    there here
""".split())

_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.\-]*")


def bm25_tokens(text: str) -> list[str]:
    """Tokenize for BM25, dropping question scaffolding.

    Applied to documents and queries alike — BM25Retriever uses one function for
    both, which is what keeps the index and the query in the same vocabulary.

    Falls back to the unfiltered tokens when filtering would leave nothing. A
    query of pure scaffolding ("what is it?") has no content terms, and an empty
    query matches nothing at all — worse than matching the wrong thing.
    """
    tokens = _TOKEN.findall(text.lower())
    kept = [t for t in tokens if t not in _QUESTION_WORDS]
    return kept or tokens


@lru_cache(maxsize=16)
def _build_bm25(protocols: frozenset[str] | None) -> BM25Retriever | None:
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
    retriever = BM25Retriever.from_documents(docs, preprocess_func=bm25_tokens)
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
