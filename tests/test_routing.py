"""Edge logic tests — pure functions, no API calls."""

import pytest

from src.graph.build import _after_grade, _after_route, _after_verify


@pytest.mark.parametrize(
    "intent,expected",
    [
        ("docs", "retrieve"),
        ("live_data", "live_data"),
        ("account_action", "escalate"),
        ("out_of_scope", "refuse"),
    ],
)
def test_route_dispatch(intent, expected):
    assert _after_route({"intent": intent, "query_type": "cx"}) == expected


@pytest.mark.parametrize(
    "intent,expected",
    [
        ("docs", "retrieve"),
        ("live_data", "live_data"),
        ("account_action", "escalate"),
        ("out_of_scope", "refuse"),
    ],
)
def test_a_missing_classification_falls_back_to_the_cheap_path(intent, expected):
    """`guard` seeds `query_type` on every path, so absence means something
    upstream is broken. The safe response to that is the CX path, not a crashed
    turn — failing closed would take down a working support agent over a missing
    label."""
    assert _after_route({"intent": intent}) == expected


def test_grade_with_surviving_docs_generates():
    assert _after_grade({"docs": ["chunk"], "attempts": 0}) == "generate"


def test_grade_with_no_docs_rewrites_then_escalates():
    assert _after_grade({"docs": [], "attempts": 0}) == "rewrite"
    assert _after_grade({"docs": [], "attempts": 2}) == "escalate"


def test_verify_grounded_finalizes():
    assert _after_verify({"grounded": True, "attempts": 1}) == "finalize"


def test_verify_ungrounded_retries_then_escalates():
    assert _after_verify({"grounded": False, "attempts": 1}) == "rewrite"
    assert _after_verify({"grounded": False, "attempts": 2}) == "escalate"


def test_ungrounded_answer_never_reaches_the_user():
    """The whole point of the verify node: no path from ungrounded to finalize."""
    for attempts in range(0, 5):
        assert _after_verify({"grounded": False, "attempts": attempts}) != "finalize"
