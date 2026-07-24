"""Deterministic guardrails that run BEFORE the LLM router.

Rationale — defense in depth. The router is a good classifier, but it is a
language model: prompt-injectable, non-deterministic, and silently degradable by
a model or prompt change. For the two highest-cost failures we do not accept a
probabilistic gate:

  1. Seed phrase / private key solicitation. If the agent ever engages with
     "help me recover my seed phrase" — even helpfully — it trains users that
     sharing keys with support is normal. That is the exact script every
     wallet-drainer phishing attack runs. Cost of a false positive: one
     unnecessary safety message. Cost of a false negative: a drained wallet and
     a support channel that taught the user to be drained.

  2. Compromised-account / fund-loss claims. These go to a human every time.
     The protocols covered here are self-custodial: nobody, including support,
     can reverse a transaction. An agent that improvises here either gives false
     hope or sounds like the scammer.

These rules are intentionally over-broad and cheap. They cannot be argued out of
by phrasing, because they never reach a model.

The patterns are protocol-independent by design — none of them mention a venue,
so onboarding a protocol never means re-tuning the gate. The user-facing copy is
rendered from the registry (`{coverage}`) rather than naming a protocol inline;
see `protocols.coverage_phrase`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.protocols import coverage_phrase

# Never engage. Refuse + warn, regardless of how the request is framed
# ("I'm the owner", "for testing", "I wrote it down wrong").
SECRET_SOLICITATION = re.compile(
    r"\b("
    r"seed\s*phrase|secret\s*phrase|recovery\s*phrase|mnemonic|"
    r"private\s*key|priv\s*key|keystore\s*(file|json)|"
    r"12[\s-]word|24[\s-]word|wallet\s*password"
    r")\b",
    re.IGNORECASE,
)

# Straight to a human. No RAG attempt.
COMPROMISE = re.compile(
    r"\b("
    r"hacked|drained|stolen|scammed|phish\w*|compromis\w*|"
    r"unauthoriz\w*\s+(transaction|withdrawal|transfer|access)|"
    r"someone\s+(took|stole|moved|accessed)|"
    r"refund|reverse\s+(the\s+)?(transaction|transfer|trade)|charge\s*back"
    r")\b",
    re.IGNORECASE,
)

# Out of scope and liability-bearing. Docs cannot answer these and guessing
# is worse than declining.
TAX_LEGAL = re.compile(
    r"\b(tax(es|able|ation)?|capital\s+gains|irs|hmrc|1099|"
    r"write[\s-]?off|deduct\w*|accountant|lawyer|legal|illegal|lawful|"
    r"jurisdiction|regulat\w*|compliance|sanction(s|ed)|kyc|aml)\b",
    re.IGNORECASE,
)

# Impersonation / social-engineering probes aimed at the agent itself.
IMPERSONATION = re.compile(
    r"\b("
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions|"
    r"disregard\s+your\s+(rules|instructions|system\s+prompt)|"
    r"you\s+are\s+now|pretend\s+(you\s+are|to\s+be)|"
    r"developer\s+mode|jailbreak|"
    r"reveal\s+your\s+(system\s+)?prompt|repeat\s+your\s+instructions"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardrailHit:
    rule: str
    action: str  # "refuse_secret" | "escalate" | "refuse_scope" | "refuse_injection"
    message: str


# Templates, not finished strings: `{coverage}` is filled from the registry at
# import so the copy tracks the whitelist. Keep them free of protocol names —
# `tests/test_copy.py` enforces that.
_SECRET_TEMPLATE = (
    "**Stop — do not share your seed phrase, recovery phrase, or private key "
    "with anyone, including me.**\n\n"
    "I will never ask for them, and neither will the real support team for any "
    "protocol. Anyone who does is trying to steal your funds — that is true "
    "even if they message you first, use official-looking branding, or say they "
    "need it to help you.\n\n"
    "The protocols I cover are self-custodial: your keys are yours alone. There is no "
    "account-recovery process that involves revealing them, because there is no "
    "one who can act on them for you.\n\n"
    "If you have already shared them, assume the wallet is compromised: move "
    "any remaining funds to a new wallet you generate yourself, immediately."
)

_COMPROMISE_TEMPLATE = (
    "I'm routing you to a human support agent now — I don't handle anything "
    "touching your funds or account, and I won't guess at it.\n\n"
    "Two things that are true regardless, and worth knowing while you wait:\n\n"
    "- The protocols I cover are self-custodial. No one — not support, not me "
    "— can reverse a transaction or move funds from your wallet. Anyone who "
    "claims they can, in exchange for a fee or your keys, is running the second "
    "half of the scam.\n"
    "- Never share your seed phrase or private key with anyone contacting you "
    "about this, however official they look.\n\n"
    "If funds are still in the wallet, move them to a new wallet you generate "
    "yourself."
)

_TAX_TEMPLATE = (
    "I can't help with tax or legal questions — I'd be guessing, and the "
    "guess could be expensive for you. This depends on your jurisdiction and "
    "your circumstances; please talk to a qualified accountant or lawyer.\n\n"
    "I can help with how the protocols I cover ({coverage}) work mechanically "
    "— fees, mechanics, risks, and integrations — and you can export your "
    "transaction history to give to whoever does your filing."
)

_INJECTION_TEMPLATE = (
    "I'm a crypto protocol support agent and I'll stay one. I can help with "
    "how the protocols I cover ({coverage}) work — their mechanics, fees, "
    "risks, and integrations — and with live market and on-chain data."
)


def _render(template: str) -> str:
    return template.format(coverage=coverage_phrase())


_SECRET_MSG = _render(_SECRET_TEMPLATE)
_COMPROMISE_MSG = _render(_COMPROMISE_TEMPLATE)
_TAX_MSG = _render(_TAX_TEMPLATE)
_INJECTION_MSG = _render(_INJECTION_TEMPLATE)

# Order matters: the most dangerous pattern wins when a message matches several.
# "I got hacked, send me your seed phrase recovery process" is a secret
# solicitation first and a compromise report second.
_RULES: list[tuple[re.Pattern, GuardrailHit]] = [
    (SECRET_SOLICITATION, GuardrailHit("secret_solicitation", "refuse_secret", _SECRET_MSG)),
    (COMPROMISE, GuardrailHit("compromise", "escalate", _COMPROMISE_MSG)),
    (IMPERSONATION, GuardrailHit("impersonation", "refuse_injection", _INJECTION_MSG)),
    (TAX_LEGAL, GuardrailHit("tax_legal", "refuse_scope", _TAX_MSG)),
]


def check(question: str) -> GuardrailHit | None:
    """Return the first matching guardrail, or None to continue to the router."""
    for pattern, hit in _RULES:
        if pattern.search(question):
            return hit
    return None
