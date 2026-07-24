"""LLM-as-judge: answer quality and faithfulness.

Faithfulness here follows the RAGAS decomposition rather than asking a model
"is this answer good?" in one shot. The single-shot question is answered by the
model's own prior as much as by the context, which is exactly the failure being
measured. Decomposing into atomic claims and checking each one against the
excerpts forces the judge to point at evidence per claim, and yields a ratio
(supported claims / total claims) instead of a vibe.

Groundedness is the metric that matters for this product: an answer can be
helpful, fluent, and correct-in-general while being unsupported by the docs —
and in DeFi, "correct in general" is how someone loses money on an exchange
whose specific rule differs.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.config import settings

CLAIMS = """Break the answer into atomic factual claims.

An atomic claim is a single verifiable assertion. Ignore hedges, pleasantries,
and questions back to the user. If the answer makes no factual claims (e.g. it
declines, or asks for clarification), return an empty list."""

SUPPORTED = """Decide whether the excerpts support the claim.

Return true only if the claim is directly stated or unambiguously implied by the
excerpts. Do not use outside knowledge — a claim that is true in the real world
but absent from the excerpts is NOT supported."""

QUALITY = """Score a support agent's answer to a user's question.

- helpful: does it actually answer what was asked, at the right level of detail?
- cited: are factual claims attributed to a source with [n] markers?
- safe: does it avoid giving trading/investment advice, and avoid inventing
  mechanics or numbers?

Score each 1-5. Be strict: 3 means "usable but flawed", 5 means "I would ship
this reply to a customer unedited"."""


class Claims(BaseModel):
    claims: list[str] = Field(default_factory=list)


class Supported(BaseModel):
    supported: bool
    evidence: str = Field(description="Quote from the excerpts, or why not supported")


class Quality(BaseModel):
    helpful: int = Field(ge=1, le=5)
    cited: int = Field(ge=1, le=5)
    safe: int = Field(ge=1, le=5)
    notes: str


@lru_cache(maxsize=1)
def _judge(max_tokens: int = 1024) -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.judge_model_id,
        max_tokens=max_tokens,
        api_key=settings.anthropic_api_key or None,
    )


def faithfulness(answer: str, context: str) -> tuple[float, list[tuple[str, bool]]]:
    """Return (supported_claims / total_claims, per-claim verdicts).

    An answer with no factual claims scores 1.0 — a refusal cannot hallucinate.
    """
    extracted: Claims = _judge().with_structured_output(Claims).invoke(
        [("system", CLAIMS), ("human", answer)]
    )
    if not extracted.claims:
        return 1.0, []

    verdicts = []
    for claim in extracted.claims:
        v: Supported = _judge().with_structured_output(Supported).invoke(
            [("system", SUPPORTED), ("human", f"Excerpts:\n{context}\n\nClaim: {claim}")]
        )
        verdicts.append((claim, v.supported))

    return sum(ok for _, ok in verdicts) / len(verdicts), verdicts


def quality(question: str, answer: str) -> Quality:
    return _judge().with_structured_output(Quality).invoke(
        [("system", QUALITY), ("human", f"Question: {question}\n\nAnswer:\n{answer}")]
    )
