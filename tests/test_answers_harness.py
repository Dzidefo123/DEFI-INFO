"""The answers harness must not let a failed measurement raise the score.

`faithfulness` is supported / extracted. An answer the extractor returns nothing
for used to score 1.00, on the reasoning that a refusal cannot hallucinate — true
of a refusal, and indistinguishable from an extractor that simply failed.

Measured on 2026-08-21: `doc-018` is a 1,440-character answer carrying a
maintenance-margin formula. One extractor found zero claims in it and it scored
1.00; another found eighteen. The answer was never claim-free — the measurement
failed and reported success.
"""

from unittest.mock import patch

import eval.run_eval as R
from eval.judge import Quality


class _Doc:
    metadata = {"source": "https://example.org/page", "heading": "Heading"}
    page_content = "context"


def _run(faith_fn, n=3):
    with patch("src.retrieval.retriever.hybrid_search", lambda *a, **k: [_Doc()]), \
         patch("src.graph.nodes.generate", lambda s: {"answer": "an answer"}), \
         patch("eval.judge.faithfulness", faith_fn), \
         patch("eval.judge.quality", lambda q, a: Quality(helpful=5, cited=5, safe=5, notes="")):
        cases = [c for c in R.load_cases() if c.get("expect_source")][:n]
        return R.eval_answers(cases, limit=n, dump_path=None)


def test_an_unextractable_answer_leaves_the_denominator():
    """Not scored 1.00, and not silently dropped either — counted and reported."""
    calls = {"n": 0}

    def faith(answer, context):
        calls["n"] += 1
        return (None, []) if calls["n"] == 2 else (1.0, [("c", True)])

    out = _run(faith)
    assert out["unmeasured"] == 1
    assert out["faithfulness"] == 1.0  # the mean of the two that WERE measured


def test_an_unextractable_answer_cannot_lift_a_poor_score():
    """The regression in its harmful direction: if the empty case counted as
    1.00 it would drag a 0.50 mean up to 0.67."""
    calls = {"n": 0}

    def faith(answer, context):
        calls["n"] += 1
        return (None, []) if calls["n"] == 3 else (0.5, [("c", False)])

    out = _run(faith)
    assert out["faithfulness"] == 0.5
    assert out["unmeasured"] == 1


def test_all_answers_unextractable_reports_no_score_rather_than_perfect():
    out = _run(lambda a, c: (None, []))
    assert out["faithfulness"] is None
    assert out["unmeasured"] == 3
