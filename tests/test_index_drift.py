"""Index/corpus reconciliation.

Hybrid search reads two stores that must hold the same documents: dense reads
Chroma, sparse reads `corpus.jsonl`. Nothing at query time notices when they
diverge — a chunk present in only one of them simply retrieves worse, silently,
and every recall number measured on the pair describes a corpus nobody declared.

These are pure set-logic and disk tests: no embedding model, no Chroma.
"""

import json

import pytest
from langchain_core.documents import Document

from src.retrieval import store
from src.retrieval.store import _drift


def _doc(doc_id, protocol="hyperliquid", text="body"):
    return Document(
        page_content=text, metadata={"protocol": protocol, "doc_id": doc_id}
    )


# --- pure drift logic ---------------------------------------------------


def test_agreeing_stores_have_no_drift():
    ids = ["hyperliquid:a#0", "ethena:b#1"]
    assert _drift(ids, ids) == ([], [])


def test_orphans_are_ids_the_index_has_and_the_corpus_does_not():
    orphaned, missing = _drift(["a", "b", "c"], ["a", "c"])
    assert orphaned == ["b"]
    assert missing == []


def test_missing_are_ids_the_corpus_has_and_the_index_does_not():
    orphaned, missing = _drift(["a"], ["a", "z"])
    assert orphaned == []
    assert missing == ["z"]


def test_drift_reports_both_directions_at_once():
    assert _drift(["a", "stale"], ["a", "fresh"]) == (["stale"], ["fresh"])


def test_drift_is_order_and_duplicate_insensitive():
    """Chroma returns ids in its own order; the mirror returns file order."""
    assert _drift(["b", "a", "a"], ["a", "b"]) == ([], [])


def test_drift_output_is_sorted_so_reports_are_stable():
    orphaned, _ = _drift(["z", "a", "m"], [])
    assert orphaned == ["a", "m", "z"]


def test_empty_index_reports_every_corpus_id_missing():
    assert _drift([], ["a", "b"]) == ([], ["a", "b"])


# --- the regression this shipped for -----------------------------------


def test_untagged_legacy_ids_are_detected_as_orphans():
    """The real failure: a pre-namespacing corpus whose ids lacked `{protocol}:`.

    Those rows carried no `protocol` metadata, so no per-protocol delete could
    ever sweep them and they answered unfiltered queries forever. Set logic on
    doc_id catches them precisely because it does not consult metadata.
    """
    orphaned, missing = _drift(
        ["https://docs/x#0", "hyperliquid:https://docs/x#0"],
        ["hyperliquid:https://docs/x#0"],
    )
    assert orphaned == ["https://docs/x#0"]
    assert missing == []


def test_a_page_reassigned_between_protocols_leaves_an_orphan():
    """Widening `path_prefixes` moves a page to another protocol, changing its
    doc_id. The old id survives an upsert, so the same URL ends up indexed under
    two protocols — the exact cross-protocol leak the filter exists to prevent."""
    url = "https://hyperliquid.gitbook.io/hyperliquid-docs/onboarding/how-to-use-the-hyperevm"
    orphaned, _ = _drift(
        [f"hyperliquid:{url}#0", f"hyperevm:{url}#0"], [f"hyperevm:{url}#0"]
    )
    assert orphaned == [f"hyperliquid:{url}#0"]


# --- corpus mirror reads ------------------------------------------------


def test_corpus_ids_reads_doc_ids_in_file_order(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    monkeypatch.setattr(store, "_CORPUS", corpus)
    with corpus.open("w", encoding="utf-8") as fh:
        for doc_id in ("ethena:b#0", "hyperliquid:a#0"):
            fh.write(json.dumps({"text": "t", "meta": {"doc_id": doc_id}}) + "\n")

    assert store.corpus_ids() == ["ethena:b#0", "hyperliquid:a#0"]


def test_corpus_ids_on_a_missing_mirror_is_empty_not_an_error(tmp_path, monkeypatch):
    """`--verify` must be runnable before a first crawl."""
    monkeypatch.setattr(store, "_CORPUS", tmp_path / "nope.jsonl")
    assert store.corpus_ids() == []


def test_corpus_ids_ignores_blank_lines(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    monkeypatch.setattr(store, "_CORPUS", corpus)
    corpus.write_text(
        json.dumps({"text": "t", "meta": {"doc_id": "a"}}) + "\n\n", encoding="utf-8"
    )
    assert store.corpus_ids() == ["a"]


# --- write_index drops before writing -----------------------------------


def test_write_index_drops_the_collection_before_writing(tmp_path, monkeypatch):
    """The whole fix. `Chroma.from_documents` upserts, so without an explicit
    drop every id that changed between rebuilds survives the rebuild."""
    calls = []
    monkeypatch.setattr(store, "_CORPUS", tmp_path / "corpus.jsonl")
    monkeypatch.setattr(store.settings, "chroma_dir", tmp_path)
    monkeypatch.setattr(store, "drop_collection", lambda: calls.append("drop"))
    monkeypatch.setattr(store, "embeddings", lambda: object())
    monkeypatch.setattr(
        store.Chroma,
        "from_documents",
        classmethod(lambda cls, **kw: calls.append("write") or object()),
    )

    store.write_index([_doc("hyperliquid:a#0")])

    assert calls == ["drop", "write"], "the drop must happen, and must happen first"


def test_write_index_mirrors_exactly_what_it_indexed(tmp_path, monkeypatch):
    """The mirror is the declared truth, so it must be rewritten wholesale —
    never merged — by a full rebuild."""
    corpus = tmp_path / "corpus.jsonl"
    monkeypatch.setattr(store, "_CORPUS", corpus)
    monkeypatch.setattr(store.settings, "chroma_dir", tmp_path)
    monkeypatch.setattr(store, "drop_collection", lambda: None)
    monkeypatch.setattr(store, "embeddings", lambda: object())
    monkeypatch.setattr(
        store.Chroma, "from_documents", classmethod(lambda cls, **kw: object())
    )

    corpus.write_text(
        json.dumps({"text": "stale", "meta": {"doc_id": "gone:x#0"}}) + "\n",
        encoding="utf-8",
    )
    store.write_index([_doc("hyperliquid:a#0"), _doc("ethena:b#0", "ethena")])

    assert store.corpus_ids() == ["hyperliquid:a#0", "ethena:b#0"]


def test_reindex_from_an_empty_mirror_refuses(tmp_path, monkeypatch):
    """Silently writing an empty index would look like a successful repair."""
    monkeypatch.setattr(store, "_CORPUS", tmp_path / "corpus.jsonl")
    monkeypatch.setattr(store.settings, "chroma_dir", tmp_path)
    with pytest.raises(FileNotFoundError):
        store.reindex_from_corpus()
