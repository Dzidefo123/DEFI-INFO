"""Grounding-refusal tests (PR 6): confidence floor + escalation copy split.

No API calls — the LLM grade/verify stages aren't exercised here; we test the
deterministic pieces: the floor filter, the escalation-reason wiring, and that a
grounding failure never gets the account/funds copy."""

from langchain_core.documents import Document

from src.graph import nodes


def _doc(score=None):
    meta = {"doc_id": "x", "source": "s"}
    if score is not None:
        meta["rerank_score"] = score
    return Document(page_content="c", metadata=meta)


# --- confidence floor ---------------------------------------------------


def test_floor_none_is_passthrough():
    docs = [_doc(-9.0), _doc(5.0)]
    assert nodes._above_floor(docs, None) == docs


def test_floor_drops_below_threshold():
    kept = nodes._above_floor([_doc(-9.0), _doc(2.0), _doc(-1.0)], threshold=0.0)
    assert [d.metadata["rerank_score"] for d in kept] == [2.0]


def test_floor_keeps_unreranked_docs():
    # No rerank_score (rerank was off) -> defaults to threshold -> kept.
    kept = nodes._above_floor([_doc(None)], threshold=0.0)
    assert len(kept) == 1


def test_retrieve_flags_low_confidence_when_empty(monkeypatch):
    monkeypatch.setattr(nodes, "hybrid_search", lambda q, protocols=None: [])
    out = nodes.retrieve({"question": "q", "protocols": []})
    assert out["docs"] == []
    assert out["escalation_reason"] == nodes.LOW_CONFIDENCE


def test_retrieve_no_reason_when_docs_present(monkeypatch):
    monkeypatch.setattr(nodes, "hybrid_search", lambda q, protocols=None: [_doc(3.0)])
    out = nodes.retrieve({"question": "q", "protocols": []})
    assert out["docs"]
    assert "escalation_reason" not in out


# --- escalation copy split ----------------------------------------------


def test_grounding_failure_gets_docs_miss_copy():
    for reason in (nodes.LOW_CONFIDENCE, nodes.NO_GROUNDED_DOCS, "ungrounded: made it up"):
        out = nodes.escalate({"question": "q", "escalation_reason": reason})
        assert "couldn't find documentation" in out["answer"].lower()
        assert "your account" not in out["answer"].lower()


def test_account_action_keeps_funds_copy():
    out = nodes.escalate({"question": "q", "intent": "account_action"})
    assert "your account" in out["answer"].lower()
    assert "seed phrase" in out["answer"].lower()


def test_compromise_style_unknown_reason_keeps_funds_copy():
    # Guardrail-compromise reaches escalate with no docs-failure reason.
    out = nodes.escalate({"question": "q"})
    assert "your account" in out["answer"].lower()


def test_escalation_reason_is_preserved():
    out = nodes.escalate({"question": "q", "escalation_reason": nodes.NO_GROUNDED_DOCS})
    assert out["escalation_reason"] == nodes.NO_GROUNDED_DOCS
