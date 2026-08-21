"""Cross-encoder reranking.

Bi-encoders (the embedding model) encode query and document *separately*, so
they never see the pair together — cheap enough to index 661 chunks, but blind
to fine-grained interaction. A cross-encoder encodes (query, chunk) jointly and
scores the pair directly. Far more accurate, far too slow to run over the whole
corpus.

So it goes last: hybrid search cheaply narrows 661 -> ~20 candidates, then the
cross-encoder does the precise ordering over that shortlist. Recall comes from
fusion; precision comes from here.

Local ONNX — no API key, no per-query cost.
"""

from __future__ import annotations

import threading
from functools import lru_cache

from langchain_core.documents import Document

from src.config import settings

# See `retriever._INIT`. Specialist agents rerank in parallel, and two threads
# racing to load the same ONNX model is at best duplicated work and at worst a
# corrupt half-initialised model.
_INIT = threading.Lock()


@lru_cache(maxsize=1)
def _build_encoder():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=settings.rerank_model)


def _encoder():
    with _INIT:
        return _build_encoder()


def rerank(query: str, docs: list[Document], k: int) -> list[Document]:
    """Order `docs` by cross-encoder relevance to `query`, keep the top `k`."""
    if not docs:
        return []
    # Scoring one candidate is a no-op ordering-wise, and still pays a model load.
    if len(docs) == 1:
        return docs[:k]

    scores = list(_encoder().rerank(query, [d.page_content for d in docs]))
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)

    out = []
    for doc, score in ranked[:k]:
        # Copy: mutating the retriever's Document would leak scores into the
        # BM25 corpus objects, which are cached and reused across queries.
        out.append(
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": round(float(score), 4)},
            )
        )
    return out


def warmup() -> None:
    """Force the model download/load outside the latency path."""
    _encoder().rerank("warmup", ["warmup"])
