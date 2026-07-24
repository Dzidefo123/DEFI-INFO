from langchain_core.documents import Document

from src.retrieval.retriever import _filter_by_protocol, _rrf, _where


def _doc(doc_id: str) -> Document:
    return Document(page_content=doc_id, metadata={"doc_id": doc_id})


def _tagged(doc_id: str, protocol: str) -> Document:
    return Document(page_content=doc_id, metadata={"doc_id": doc_id, "protocol": protocol})


def test_rrf_rewards_agreement_across_retrievers():
    dense = [_doc("a"), _doc("shared"), _doc("c")]
    sparse = [_doc("x"), _doc("shared"), _doc("z")]

    fused = _rrf([dense, sparse], k=3)

    # "shared" ranks 2nd in both lists and appears nowhere else; "a" and "x"
    # each rank 1st but only in one list. Agreement should beat a single
    # top hit — that is the whole reason to fuse rather than concatenate.
    assert fused[0].metadata["doc_id"] == "shared"


def test_rrf_dedupes_and_respects_k():
    dense = [_doc("a"), _doc("b")]
    sparse = [_doc("a"), _doc("b")]

    fused = _rrf([dense, sparse], k=5)

    assert [d.metadata["doc_id"] for d in fused] == ["a", "b"]


def test_rrf_handles_disjoint_results():
    fused = _rrf([[_doc("a")], [_doc("b")]], k=2)
    assert {d.metadata["doc_id"] for d in fused} == {"a", "b"}


# --- protocol filtering (PR 2) ------------------------------------------


def test_filter_none_keeps_everything():
    docs = [_tagged("a", "hyperliquid"), _tagged("b", "hyperevm")]
    assert _filter_by_protocol(docs, None) == docs


def test_filter_single_protocol():
    docs = [_tagged("a", "hyperliquid"), _tagged("b", "hyperevm")]
    kept = _filter_by_protocol(docs, frozenset({"hyperevm"}))
    assert [d.metadata["doc_id"] for d in kept] == ["b"]


def test_filter_multiple_protocols():
    docs = [_tagged("a", "hyperliquid"), _tagged("b", "hyperevm"), _tagged("c", "aave")]
    kept = _filter_by_protocol(docs, frozenset({"hyperliquid", "hyperevm"}))
    assert {d.metadata["doc_id"] for d in kept} == {"a", "b"}


def test_filter_excludes_untagged_chunks_when_scoped():
    docs = [_tagged("a", "hyperliquid"), _doc("legacy")]  # legacy has no protocol tag
    kept = _filter_by_protocol(docs, frozenset({"hyperliquid"}))
    assert [d.metadata["doc_id"] for d in kept] == ["a"]


def test_where_none_is_no_filter():
    assert _where(None) is None
    assert _where(frozenset()) is None


def test_where_single_protocol_is_flat():
    assert _where(frozenset({"hyperliquid"})) == {"protocol": "hyperliquid"}


def test_where_multiple_protocols_uses_in_and_is_sorted():
    assert _where(frozenset({"hyperevm", "hyperliquid"})) == {
        "protocol": {"$in": ["hyperevm", "hyperliquid"]}
    }
