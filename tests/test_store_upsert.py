"""Corpus-merge test for per-protocol re-crawl. Pure disk I/O in a tmp dir."""

import json

from langchain_core.documents import Document

from src.retrieval import store


def _doc(doc_id, protocol, text="new content"):
    return Document(page_content=text, metadata={"protocol": protocol, "doc_id": doc_id})


def _write(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _read(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_merge_replaces_only_the_named_protocol(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    monkeypatch.setattr(store, "_CORPUS", corpus)
    monkeypatch.setattr(store.settings, "chroma_dir", tmp_path)

    _write(
        corpus,
        [
            {"text": "old hl 1", "meta": {"protocol": "hyperliquid", "doc_id": "hyperliquid:a#0"}},
            {"text": "old hl 2", "meta": {"protocol": "hyperliquid", "doc_id": "hyperliquid:a#1"}},
            {"text": "evm keep", "meta": {"protocol": "hyperevm", "doc_id": "hyperevm:b#0"}},
        ],
    )

    store._merge_corpus("hyperliquid", [_doc("hyperliquid:a#0", "hyperliquid", "fresh hl")])

    rows = _read(corpus)
    protos = [r["meta"]["protocol"] for r in rows]
    assert protos.count("hyperevm") == 1          # untouched
    assert protos.count("hyperliquid") == 1       # replaced, not appended
    assert any(r["text"] == "fresh hl" for r in rows)
    assert not any(r["text"].startswith("old hl") for r in rows)


def test_merge_into_missing_corpus_creates_it(tmp_path, monkeypatch):
    corpus = tmp_path / "sub" / "corpus.jsonl"
    monkeypatch.setattr(store, "_CORPUS", corpus)
    monkeypatch.setattr(store.settings, "chroma_dir", corpus.parent)

    store._merge_corpus("hyperevm", [_doc("hyperevm:b#0", "hyperevm")])

    rows = _read(corpus)
    assert [r["meta"]["protocol"] for r in rows] == ["hyperevm"]
