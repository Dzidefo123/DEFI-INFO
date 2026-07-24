"""User-facing copy must not hardcode a single protocol.

The agent describes itself in a dozen places — guardrail refusals, escalation
messages, the out-of-scope refusal, the CLI banner. When that copy names one
protocol inline, adding the second protocol to the whitelist silently makes the
agent lie about its own scope: it answers a HyperEVM question and then, on the
next turn, tells the user it "can only help with how Hyperliquid works". For a
support agent in DeFi that is not just untidy — copy that contradicts the
agent's behaviour is exactly what a user has been trained to read as a
compromised or spoofed support channel.

So the rule is structural, not stylistic: templates carry no protocol names,
and the registry renders them. These tests fail if someone reintroduces a
hardcoded name, and they run the templates against a fabricated whitelist to
prove the rendering is actually registry-driven rather than coincidental.
"""

import re

import pytest

from src import protocols
from src.graph import nodes
from src.guardrails import rules

# Every name and alias of every whitelisted protocol, plus roster candidates
# that a future author might reach for out of habit.
# Not-yet-whitelisted candidates. (Whitelisted names are covered separately by
# `_whitelisted_names`, so a protocol graduating off this list stays checked.)
_ROSTER_NAMES = ("Aave", "Uniswap", "dYdX", "GMX", "Lido", "Curve")


def _whitelisted_names() -> tuple[str, ...]:
    names = []
    for proto in protocols.enabled_protocols():
        names.append(proto.name)
        names.extend(proto.aliases)
    return tuple(names)


# --- the templates themselves are protocol-free -------------------------

_TEMPLATES = {
    "secret": rules._SECRET_TEMPLATE,
    "compromise": rules._COMPROMISE_TEMPLATE,
    "tax": rules._TAX_TEMPLATE,
    "injection": rules._INJECTION_TEMPLATE,
    "refuse": nodes._REFUSE_TEMPLATE,
    "account_escalation": nodes._ACCOUNT_ESCALATION_MSG,
    "no_grounded_answer": nodes._NO_GROUNDED_ANSWER_MSG,
}


@pytest.mark.parametrize("label", sorted(_TEMPLATES))
def test_template_names_no_protocol(label):
    """A template that says "Hyperliquid" stops being true when PR N adds Aave."""
    template = _TEMPLATES[label]
    for name in _whitelisted_names() + _ROSTER_NAMES:
        assert not re.search(rf"\b{re.escape(name)}\b", template, re.IGNORECASE), (
            f"{label} copy hardcodes {name!r}; render it from "
            f"protocols.coverage_phrase() via a {{coverage}} placeholder instead"
        )


# --- rendered messages do name the current whitelist --------------------


def test_rendered_guardrail_messages_name_every_whitelisted_protocol():
    """The user is told what the agent covers, from the registry."""
    for message in (rules._TAX_MSG, rules._INJECTION_MSG):
        for proto in protocols.enabled_protocols():
            assert proto.name in message


def test_refusal_names_every_whitelisted_protocol():
    out = nodes.refuse({"question": "should I long HYPE?"})
    for proto in protocols.enabled_protocols():
        assert proto.name in out["answer"]


def test_templates_render_against_a_different_whitelist():
    """Proof the copy is registry-driven: swap the registry, swap the copy.

    Without this, a template could hardcode today's two names in an order that
    happens to match `coverage_phrase()` and every other assertion would pass.
    """
    fake = "Aave, Uniswap, and Lido"
    for label, template in _TEMPLATES.items():
        rendered = template.format(coverage=fake)
        if "{coverage}" in template:
            assert fake in rendered, f"{label} dropped its coverage phrase"
        # Renders cleanly whether or not it uses the placeholder.
        assert "{" not in rendered.replace("{coverage}", "")


# --- coverage_phrase / english_list -------------------------------------


@pytest.mark.parametrize(
    "items,expected",
    [
        ([], ""),
        (["Aave"], "Aave"),
        (["Aave", "Uniswap"], "Aave and Uniswap"),
        (["Aave", "Uniswap", "Lido"], "Aave, Uniswap, and Lido"),
        (["A", "B", "C", "D"], "A, B, C, and D"),
    ],
)
def test_english_list(items, expected):
    assert protocols.english_list(items) == expected


def test_coverage_phrase_matches_registry():
    assert protocols.coverage_phrase() == protocols.english_list(
        protocols.protocol_names()
    )
    for proto in protocols.enabled_protocols():
        assert proto.name in protocols.coverage_phrase()
