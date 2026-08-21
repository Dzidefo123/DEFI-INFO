"""§13. The Verification Agent: the one that argues with everyone else.

Its role is not to find anything. It is handed claims that other agents were
prepared to make and asks whether they hold up — which means it is the only
component whose success looks like *removing* output.

§13 lists five questions. Four are answerable without a model and are answered
that way, per §3.3:

    Is this claim supported?               -> does any evidence point at it
    Is the source reliable?                -> tier, by rule
    Is there contradictory evidence?       -> relevance-weighted, both directions
    Is the conclusion stronger than
      the available evidence?              -> causal and absolute language
                                              against what the evidence can carry

The fifth — *is the evidence directly related to the claim* — is genuinely a
judgement about meaning, and it is the one a model is for. It runs last and only
on claims that survived everything cheaper, so the expensive check is never spent
on a claim already known to be unsupported.

Two checks here catch failures nothing upstream can:

**Numeric consistency.** If a claim states a figure, that figure must appear in
the evidence behind it. This is the cheapest possible defence against the most
expensive kind of error in a DeFi answer — a confidently stated number nobody
measured. Retrieval cannot catch it, and grading cannot: the chunk is relevant,
the sentence is fluent, and the number is invented.

**Overclaiming.** §13.2's whole example is a causal claim ("the withdrawals were
caused by an exploit") resting on correlational evidence. Causation is a stronger
assertion than observation, and a system that scores them identically will
publish the second dressed as the first. So causal and absolute language raises
the bar the evidence has to clear rather than being treated as ordinary wording.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from src.evidence.confidence import assess, temporal_relevance
from src.evidence.models import (
    AgentName,
    Claim,
    Evidence,
    SourceTier,
    VerificationStatus,
)

# A failed check either blocks a claim outright or weakens it. Nothing here
# silently passes: a check that could not run reports that it could not run.
BLOCKING = "blocking"
WEAKENING = "weakening"

# Support below this leaves a claim standing on essentially nothing.
MIN_SUPPORTING_SOURCES = 1

# Agreement floor. Below it, contradicting evidence outweighs support.
CONTRADICTION_THRESHOLD = 0.5

# Freshness floor for a claim's supporting evidence, on the same 0-1 scale the
# confidence model uses. Evidence decayed past this is too old to carry a
# present-tense claim — which matters most for on-chain and market data, whose
# half-lives are hours.
MIN_TEMPORAL_RELEVANCE = 0.25

# Causal and absolute constructions. A claim using these asserts more than an
# observation, so it needs more than one observation behind it.
_CAUSAL = re.compile(
    r"\b(caused?\s+by|because\s+of|due\s+to|resulted?\s+in|led\s+to|"
    r"as\s+a\s+result\s+of|triggered\s+by|stems?\s+from|attributable\s+to)\b",
    re.IGNORECASE,
)
_ABSOLUTE = re.compile(
    r"\b(always|never|guarantee[sd]?|impossible|cannot\s+be|"
    r"no\s+risk|completely\s+safe|fully\s+secure|proves?|certain(ly)?|"
    r"all\s+users|every\s+time)\b",
    re.IGNORECASE,
)

# Sources needed before a causal or absolute claim is anything but a guess.
STRONG_CLAIM_MIN_SOURCES = 2

# Numeric tokens: 12, 12.5, 1,200, 12.5%, $12.5M, 4x.
_NUMBER = re.compile(r"[$]?\d[\d,]*(?:\.\d+)?\s*(?:%|[KMB]\b|x\b)?", re.IGNORECASE)
_MAGNITUDE = {"k": 1e3, "m": 1e6, "b": 1e9}

# Relative tolerance when matching a claim's figure against the evidence, to
# absorb rounding and unit formatting rather than flagging it as fabrication.
NUMERIC_TOLERANCE = 0.02


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    check: str
    passed: bool
    detail: str
    severity: str = WEAKENING

    @property
    def blocks(self) -> bool:
        return not self.passed and self.severity == BLOCKING


class Verdict(BaseModel):
    """What verification concluded about one claim, and why."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    claim_text: str
    status: VerificationStatus
    checks: tuple[CheckResult, ...] = ()
    confidence: float = 0.0

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def summary(self) -> str:
        if not self.failures:
            return f"{len(self.checks)} checks passed"
        return "; ".join(f"{c.check}: {c.detail}" for c in self.failures)


# --- numeric extraction -------------------------------------------------


def parse_numbers(text: str) -> list[float]:
    """Every figure stated in `text`, normalised to a plain magnitude.

    "$12.5M" and "12,500,000" both become 12500000.0 so a claim and its evidence
    can be compared without either having to be written a particular way.
    Percentages stay as written: "4%" is 4, not 0.04, because the evidence states
    it the same way.
    """
    out: list[float] = []
    for token in _NUMBER.findall(text):
        cleaned = token.strip().lstrip("$").replace(",", "").rstrip()
        multiplier = 1.0
        if cleaned and cleaned[-1].lower() in _MAGNITUDE:
            multiplier = _MAGNITUDE[cleaned[-1].lower()]
            cleaned = cleaned[:-1]
        cleaned = cleaned.rstrip("%xX ").strip()
        if not cleaned:
            continue
        try:
            out.append(float(cleaned) * multiplier)
        except ValueError:
            continue
    return out


def _appears_in(value: float, haystack: list[float]) -> bool:
    """Whether `value` is present, allowing for rounding."""
    for candidate in haystack:
        if value == candidate:
            return True
        scale = max(abs(value), abs(candidate), 1e-9)
        if abs(value - candidate) / scale <= NUMERIC_TOLERANCE:
            return True
    return False


def _evidence_text(evidence: Sequence[Evidence]) -> str:
    parts: list[str] = []
    for item in evidence:
        parts.append(item.summary)
        for key, value in item.payload.items():
            parts.append(f"{key}={value}")
    return " ".join(parts)


# --- the checks ---------------------------------------------------------


def check_support(claim: Claim, supporting: Sequence[Evidence]) -> CheckResult:
    """§13's first question. Blocking: nothing else matters without it."""
    n = len({e.source.uri for e in supporting})
    return CheckResult(
        check="support",
        passed=n >= MIN_SUPPORTING_SOURCES,
        severity=BLOCKING,
        detail=(
            f"{n} independent source(s) support this"
            if n
            else "no evidence points at this claim"
        ),
    )


def check_contradiction(
    claim: Claim, evidence_by_id: dict[str, Evidence]
) -> CheckResult:
    """Relevance-weighted, not a headcount: a marginal objection should not
    cancel a directly relevant finding one for one."""
    support = sum(
        l.relevance for l in claim.supporting() if l.evidence_id in evidence_by_id
    )
    against = sum(
        l.relevance for l in claim.contradicting() if l.evidence_id in evidence_by_id
    )

    # Nothing arguing against it cannot be a contradiction. This also covers the
    # claim with no evidence at all, where agreement would be 0/0 — and reading
    # that as "outweighed" would report an unsupported claim as CONTRADICTED,
    # which is the exact conflation §13's four statuses exist to prevent. The
    # support check is what blocks that claim, with the right reason.
    if against == 0:
        return CheckResult(
            check="contradiction",
            passed=True,
            severity=BLOCKING,
            detail="no contradicting evidence",
        )

    agreement = support / (support + against)
    return CheckResult(
        check="contradiction",
        passed=agreement >= CONTRADICTION_THRESHOLD,
        severity=BLOCKING,
        detail=(
            f"agreement {agreement:.2f} — contradicting evidence outweighs support"
            if agreement < CONTRADICTION_THRESHOLD
            else f"agreement {agreement:.2f}"
        ),
    )


def check_source_quality(claim: Claim, supporting: Sequence[Evidence]) -> CheckResult:
    """The best provenance behind the claim, not the average — consistent with
    how confidence treats reliability."""
    if not supporting:
        return CheckResult(
            check="source_quality", passed=False, detail="no sources to assess"
        )
    tiers = {e.source.tier for e in supporting}
    best = min(tiers, key=lambda t: list(SourceTier).index(t))
    return CheckResult(
        check="source_quality",
        passed=best is not SourceTier.UNVERIFIED,
        detail=(
            f"best source is {best.value}"
            if best is not SourceTier.UNVERIFIED
            else "every supporting source is unverified"
        ),
    )


def check_temporal_relevance(
    claim: Claim, supporting: Sequence[Evidence], now: datetime | None = None
) -> CheckResult:
    """Whether the evidence is still current enough to carry the claim.

    Decayed per evidence kind, so a year-old documentation page passes and a
    day-old funding rate does not — the same half-lives the confidence model
    uses, applied here as a gate rather than a weight.
    """
    if not supporting:
        return CheckResult(
            check="temporal_relevance", passed=False, detail="no evidence to age"
        )
    freshest = max(temporal_relevance(e, now) for e in supporting)
    return CheckResult(
        check="temporal_relevance",
        passed=freshest >= MIN_TEMPORAL_RELEVANCE,
        detail=(
            f"freshest supporting evidence scores {freshest:.2f}"
            if freshest >= MIN_TEMPORAL_RELEVANCE
            else f"all supporting evidence is stale ({freshest:.2f})"
        ),
    )


def check_numeric_consistency(
    claim: Claim, supporting: Sequence[Evidence]
) -> CheckResult:
    """Every figure the claim states must appear in the evidence behind it.

    The cheapest defence against the most expensive error a DeFi answer can make:
    a confidently stated number nobody measured. Retrieval cannot catch it and
    grading cannot — the chunk is relevant, the sentence is fluent, and the figure
    is invented.

    Weakening rather than blocking, because a claim may legitimately restate a
    figure in units the evidence does not use. The tolerance absorbs rounding;
    what it will not absorb is a number with no counterpart at all.
    """
    stated = parse_numbers(claim.text)
    if not stated:
        return CheckResult(
            check="numeric_consistency", passed=True, detail="no figures stated"
        )
    available = parse_numbers(_evidence_text(supporting))
    missing = [n for n in stated if not _appears_in(n, available)]
    return CheckResult(
        check="numeric_consistency",
        passed=not missing,
        detail=(
            "every stated figure appears in the evidence"
            if not missing
            else f"{len(missing)} figure(s) appear nowhere in the evidence: "
            + ", ".join(f"{n:g}" for n in missing[:3])
        ),
    )


def check_overclaiming(claim: Claim, supporting: Sequence[Evidence]) -> CheckResult:
    """Whether the claim asserts more than its evidence can carry.

    §13's fifth question, made concrete. Causation is a stronger assertion than
    observation, and an absolute ("never", "guaranteed") is stronger than an
    instance — so both raise the number of independent sources required. §13.2's
    example is exactly this shape: abnormal withdrawals were observed; that they
    were *caused by* an exploit is a different and much larger claim.
    """
    causal = bool(_CAUSAL.search(claim.text))
    absolute = bool(_ABSOLUTE.search(claim.text))
    if not (causal or absolute):
        return CheckResult(
            check="overclaiming", passed=True, detail="claim is stated as an observation"
        )

    sources = len({e.source.uri for e in supporting})
    kind = "causal" if causal else "absolute"
    return CheckResult(
        check="overclaiming",
        passed=sources >= STRONG_CLAIM_MIN_SOURCES,
        detail=(
            f"{kind} claim carried by {sources} independent source(s)"
            if sources >= STRONG_CLAIM_MIN_SOURCES
            else f"{kind} claim rests on {sources} source(s); "
            f"{STRONG_CLAIM_MIN_SOURCES} needed before asserting more than an observation"
        ),
    )


# --- semantic entailment (the one model call) --------------------------

Entailer = Callable[[Claim, Sequence[Evidence]], tuple[bool, str]]


def llm_entails(claim: Claim, supporting: Sequence[Evidence]) -> tuple[bool, str]:
    """Does the evidence actually bear on THIS claim?

    The judgement no rule can make. Retrieval returns what is topically close, and
    an agent can link a chunk that is about the right protocol and the right
    feature while saying nothing about the assertion.
    """
    from functools import lru_cache

    from pydantic import Field

    class Entailment(BaseModel):
        supported: bool = Field(description="True only if the excerpts state this")
        reason: str = Field(description="One sentence")

    @lru_cache(maxsize=None)
    def _model():
        from langchain_anthropic import ChatAnthropic

        from src.config import settings

        return ChatAnthropic(
            model=settings.router_model_id,
            max_tokens=512,
            api_key=settings.anthropic_api_key or None,
        ).with_structured_output(Entailment)

    excerpts = "\n\n".join(
        f"[{i}] {e.summary}\n{e.payload.get('text', '')}"[:1500]
        for i, e in enumerate(supporting, start=1)
    )
    result = _model().invoke(
        [
            (
                "system",
                "You audit whether evidence supports a specific claim.\n\n"
                "Return supported=true ONLY if the excerpts state what the claim "
                "asserts. Being on the same topic is not enough, and neither is "
                "the claim being true in general — the question is whether THESE "
                "excerpts establish THIS claim. A claim that goes further than the "
                "excerpts, in scope, certainty, or causation, is not supported.",
            ),
            ("human", f"Claim:\n{claim.text}\n\nEvidence:\n{excerpts}"),
        ]
    )
    return result.supported, result.reason


def check_entailment(
    claim: Claim, supporting: Sequence[Evidence], entailer: Entailer
) -> CheckResult:
    ok, reason = entailer(claim, supporting)
    return CheckResult(
        check="entailment",
        passed=ok,
        severity=BLOCKING,
        detail=reason,
    )


# --- the agent ----------------------------------------------------------


def _status(checks: Sequence[CheckResult]) -> VerificationStatus:
    """Turn check results into §13's four statuses.

    CONTRADICTED is reserved for the contradiction check specifically. Every other
    blocking failure is INSUFFICIENT_EVIDENCE, because "we could not establish
    this" and "we have reason to believe otherwise" are different findings and the
    weaker one must not be reported as the stronger.
    """
    by_name = {c.check: c for c in checks}

    if (contra := by_name.get("contradiction")) is not None and not contra.passed:
        return VerificationStatus.CONTRADICTED
    if any(c.blocks for c in checks):
        return VerificationStatus.INSUFFICIENT_EVIDENCE
    if any(not c.passed for c in checks):
        return VerificationStatus.PARTIALLY_VERIFIED
    return VerificationStatus.VERIFIED


def verify(
    claim: Claim,
    evidence: Sequence[Evidence],
    entailer: Entailer | None = None,
    now: datetime | None = None,
) -> Verdict:
    """Run §13.1's sequence against one claim.

    Deterministic checks run first and the model runs last, on claims that
    survived them. That ordering is a cost decision as much as a design one: the
    expensive judgement is never spent on a claim already known to be unsupported.
    """
    by_id = {e.evidence_id: e for e in evidence}
    supporting = [
        by_id[l.evidence_id] for l in claim.supporting() if l.evidence_id in by_id
    ]

    checks = [
        check_support(claim, supporting),
        check_contradiction(claim, by_id),
        check_source_quality(claim, supporting),
        check_temporal_relevance(claim, supporting, now),
        check_numeric_consistency(claim, supporting),
        check_overclaiming(claim, supporting),
    ]

    # Only ask a model about a claim the rules did not already reject.
    if entailer is not None and not any(c.blocks for c in checks):
        checks.append(check_entailment(claim, supporting, entailer))

    status = _status(checks)
    scored = claim.model_copy(update={"verification": status})
    return Verdict(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        status=status,
        checks=tuple(checks),
        confidence=assess(scored, evidence, now).score,
    )


class VerificationOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent: str = AgentName.VERIFICATION.value
    verdicts: tuple[Verdict, ...] = ()
    limitations: tuple[str, ...] = ()

    @property
    def by_status(self) -> dict[str, int]:
        return {
            s.value: sum(1 for v in self.verdicts if v.status is s)
            for s in VerificationStatus
        }


def verify_all(
    claims: Sequence[Claim],
    evidence: Sequence[Evidence],
    entailer: Entailer | None = None,
    now: datetime | None = None,
) -> VerificationOutput:
    """Verify every claim, and report what verification itself could not do."""
    verdicts = tuple(verify(c, evidence, entailer, now) for c in claims)

    limitations: list[str] = []
    if not claims:
        limitations.append(
            "verification_agent: no claims were produced, so there was nothing to verify"
        )
    if entailer is None and claims:
        limitations.append(
            "verification_agent: semantic entailment was not checked, so evidence "
            "was verified as present, sourced, current and numerically consistent "
            "— but not as actually being about the claim it supports"
        )
    rejected = [v for v in verdicts if v.status is not VerificationStatus.VERIFIED]
    if rejected:
        limitations.append(
            f"verification_agent: {len(rejected)} of {len(verdicts)} claims did not "
            f"fully verify — " + "; ".join(f"{v.summary()}" for v in rejected[:3])
        )
    return VerificationOutput(verdicts=verdicts, limitations=tuple(limitations))
