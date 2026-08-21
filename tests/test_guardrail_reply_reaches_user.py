"""A guardrail's message must survive to the user, exactly once.

`guardrails/rules.py` writes deliberate, rule-specific copy and `test_copy.py`
checks its wording. None of that matters if the graph overwrites it downstream —
which it did: a compromise hit routes `guard -> escalate`, and `escalate` replied
a second time with the generic account message. Two consequences, both silent:
the compromise warning never reached a user, and one turn was logged twice into
the history the next turn is conditioned on.
"""

import pytest

from src.graph import nodes
from src.graph.build import build_graph
from src.guardrails import rules

COMPROMISE_Q = "Someone drained my wallet, can you reverse it?"
SECRET_Q = "I lost my seed phrase, can you help me recover it?"


def _run(question, thread):
    graph = build_graph()
    return graph.invoke({"question": question}, {"configurable": {"thread_id": thread}})


def test_the_compromise_guardrail_copy_is_what_the_user_sees():
    state = _run(COMPROMISE_Q, "compromise-copy")
    assert state["answer"] == rules._COMPROMISE_MSG
    assert state["answer"] != nodes._ACCOUNT_ESCALATION_MSG


def test_a_guardrail_turn_is_logged_exactly_once():
    state = _run(COMPROMISE_Q, "compromise-once")
    assert len(state["messages"]) == 2
    assert state["messages"][0].content == COMPROMISE_Q


def test_the_escalation_is_still_recorded():
    """Suppressing the second reply must not suppress the fact that a human is
    being brought in — that is what an operator greps for."""
    state = _run(COMPROMISE_Q, "compromise-reason")
    assert state["escalation_reason"] == "guardrail: compromise"


def test_a_refusing_guardrail_is_unaffected():
    """Only the escalate exit had the double-reply; the refuse exits terminate at
    `guard_reply`, a no-op. Pinned so a future edit does not regularise them."""
    state = _run(SECRET_Q, "secret-copy")
    assert state["answer"] == rules._SECRET_MSG
    assert len(state["messages"]) == 2


def test_a_genuine_account_escalation_still_gets_the_account_message():
    """The non-guardrail path into `escalate` is untouched: a routed
    account_action has no guardrail answer to preserve."""
    out = nodes.escalate({"question": "close my position", "intent": "account_action"})
    assert out["answer"] == nodes._ACCOUNT_ESCALATION_MSG
    assert out["escalation_reason"] == "account_action"


def test_a_grounding_failure_still_refuses_honestly():
    """The distinction PR 6 shipped: a retrieval failure must not be dressed up
    as an account matter."""
    out = nodes.escalate(
        {"question": "q", "intent": "docs", "escalation_reason": nodes.LOW_CONFIDENCE}
    )
    assert out["answer"] == nodes._NO_GROUNDED_ANSWER_MSG


@pytest.mark.parametrize("action", ["refuse_secret", "refuse_scope", "refuse_injection"])
def test_only_the_escalate_action_short_circuits_the_reply(action):
    """A non-escalating guardrail action never reaches `escalate`, so if one ever
    did it must not be silently swallowed."""
    out = nodes.escalate({"question": "q", "guardrail_action": action, "intent": "docs"})
    assert "answer" in out
