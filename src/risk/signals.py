"""Risk signals: statistics turned into findings, per §11.2.

The division of labour this module enforces is the architecture's sharpest rule:

    The LLM should explain this signal. The LLM should not calculate it.

So a `RiskSignal` is produced entirely by arithmetic. It carries the current
value, the baseline it was judged against, the scores, and the severity — every
input a reader needs to disagree with the conclusion. Downstream, a model may
put it in a sentence. Nothing downstream may change the number.

Severity is a threshold table, not a judgement. Thresholds are stated as
constants so that "why is this "high"?" has an answer shorter than a paragraph.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.risk.invariants import Breach, Invariant
from src.risk.severity import Severity, at_least, max_severity
from src.risk.statistics import (
    Baseline,
    InsufficientHistory,
    baseline,
    is_iqr_outlier,
    modified_z_score,
    pct_change,
    z_score,
)

__all__ = [
    "Breach",
    "Invariant",
    "RiskSignal",
    "Severity",
    "assess_metric",
    "explain",
    "max_severity",
    "severity_for",
]

# |z| thresholds, ascending. Under a normal distribution 2σ is roughly a 1-in-20
# reading and 3σ about 1-in-370 — but on-chain metrics are heavy-tailed, so
# these are deliberately conservative and should be read as "unusual enough to
# look at", not as p-values. Calibrating them against labelled incidents is
# future work and must be labelled as such wherever severity is shown.
ELEVATED_Z = 2.0
HIGH_Z = 3.0
CRITICAL_Z = 5.0


def severity_for(score: float | None) -> Severity:
    """Map an absolute deviation score to a severity band.

    `None` maps to UNKNOWN, never to NORMAL. This distinction is the entire
    reason the band exists: "we could not tell" and "we checked and it is fine"
    lead to opposite actions, and an unassessable metric reported as normal is
    how a blind spot becomes a clean bill of health.
    """
    if score is None:
        return Severity.UNKNOWN
    magnitude = abs(score)
    if magnitude >= CRITICAL_Z:
        return Severity.CRITICAL
    if magnitude >= HIGH_Z:
        return Severity.HIGH
    if magnitude >= ELEVATED_Z:
        return Severity.ELEVATED
    return Severity.NORMAL


class RiskSignal(BaseModel):
    """One metric, judged against its own history. The §11.2 schema.

    Both z-scores are retained rather than one being chosen here. They disagree
    in exactly the situation that matters — a baseline window containing an
    earlier spike — and that disagreement is information a report should show,
    not a tie for this module to break silently.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    protocol: str | None = None
    current_value: float
    baseline: Baseline
    z: float | None = None
    modified_z: float | None = None
    iqr_outlier: bool = False
    pct_change_vs_baseline: float | None = None
    severity: Severity = Severity.UNKNOWN
    observed_at: datetime | None = None
    note: str | None = None
    # Set when the metric declares an invariant and that invariant is violated.
    # Its presence is what makes a finding a broken property rather than an
    # unusual reading, and the report says so — the two warrant different
    # responses, and a z-score is not evidence for either.
    breach: Breach | None = None

    @property
    def signal_id(self) -> str:
        """Stable id for the (metric, protocol, value, time) this describes."""
        payload = f"{self.protocol}|{self.metric}|{self.current_value}|{self.observed_at}"
        return f"risk_{hashlib.blake2b(payload.encode(), digest_size=6).hexdigest()}"

    @property
    def anomaly(self) -> bool:
        """Whether this is a finding, as opposed to a reportable observation.

        The bar is HIGH, not ELEVATED, and the gap between those two is the
        difference between a detector people act on and one they mute. A 2σ
        threshold fires on roughly one ordinary day in twenty — per metric, so
        across a dozen tracked metrics it alerts most days on nothing at all.
        Measured on labelled data (`tests/test_risk_signals.py`), moving the bar
        to 3σ took precision from 0.46 to 1.00 with no loss of recall: every
        false positive was an ELEVATED reading and every injected anomaly cleared
        HIGH comfortably.

        ELEVATED is still surfaced — §15's report table lists "Elevated" and
        "High Anomaly" as distinct rows, and that distinction is exactly this
        one. Elevated says the metric is worth a column in the table. Anomaly
        says it is worth a paragraph.

        UNKNOWN is not an anomaly — an unassessable metric has not been shown to
        be unusual. It is also not normal; see `severity_for`.

        A broken invariant is an anomaly at any severity. The 3σ bar exists to
        stop a detector firing on ordinary variation, and a violated invariant is
        not ordinary variation — it is a property the protocol is supposed to
        guarantee, failing. There is no rate of false positives to suppress.
        """
        return self.breach is not None or at_least(self.severity, Severity.HIGH)

    @property
    def notable(self) -> bool:
        """Elevated or worse: worth reporting in the statistics table."""
        return self.breach is not None or at_least(self.severity, Severity.ELEVATED)

    @property
    def direction(self) -> str:
        if self.current_value > self.baseline.median:
            return "above"
        if self.current_value < self.baseline.median:
            return "below"
        return "at"

    @property
    def scores_disagree(self) -> bool:
        """True when classical and robust z-scores land in different bands.

        Worth surfacing: it usually means the baseline window itself contains an
        anomaly, which inflates σ and blinds the classical score. The robust
        score is the one to trust when this fires.
        """
        if self.z is None or self.modified_z is None:
            return False
        return severity_for(self.z) is not severity_for(self.modified_z)


def assess_metric(
    metric: str,
    current_value: float,
    history: Sequence[float],
    protocol: str | None = None,
    observed_at: datetime | None = None,
    invariant: Invariant | None = None,
) -> RiskSignal:
    """Judge one observation against its prior history, its invariant, or both.

    `history` must exclude `current_value`. Passing the full series including the
    current point is the mistake this signature is shaped to discourage: it makes
    every anomaly partly its own baseline, and the bigger the event, the more it
    suppresses its own score.

    Severity takes the WORSE of the two z-scores. When they disagree, the robust
    score is the one detecting something the classical score has been blinded to
    — and under-reporting a real anomaly costs more here than looking twice at a
    false one.

    An `invariant` is checked independently of all of that, and its verdict is
    combined by taking the worse of the two. The two checks answer different
    questions — "is this unusual for this metric" and "is this metric wrong" — and
    neither subsumes the other. Their independence is the point: the invariant
    still returns a verdict when there is no history at all, and when the history
    is so flat that no z-score exists.
    """
    statistical = _statistical(metric, current_value, history, protocol, observed_at)
    if invariant is None:
        return statistical
    return _with_invariant(statistical, invariant, current_value)


def _with_invariant(
    signal: RiskSignal, invariant: Invariant, current_value: float
) -> RiskSignal:
    """Fold an invariant verdict into a statistical one.

    `max_severity` ranks UNKNOWN below NORMAL, which is what makes this safe: a
    satisfied invariant on a metric with no usable baseline comes out NORMAL,
    because something was genuinely checked and passed. That is the fix for the
    case this module exists for — a series constant at 1.0 used to report UNKNOWN
    forever, which reads as a blind spot when it is in fact the healthiest
    possible state.
    """
    from src.risk.invariants import holding_note

    breach = invariant.check(current_value)
    verdict = breach.severity if breach else Severity.NORMAL
    detail = breach.explain() if breach else holding_note(invariant, current_value)
    # The statistical note survives alongside it. When a metric has too little
    # history, that remains true and worth saying even though the invariant
    # carries the verdict.
    note = f"{signal.metric} {detail}"
    if signal.note and breach is None:
        note = f"{note} Separately: {signal.note}"

    return signal.model_copy(
        update={
            "severity": max_severity([verdict, signal.severity]),
            "breach": breach,
            "note": note,
        }
    )


def _statistical(
    metric: str,
    current_value: float,
    history: Sequence[float],
    protocol: str | None,
    observed_at: datetime | None,
) -> RiskSignal:
    try:
        base = baseline(history)
    except InsufficientHistory as exc:
        return _unassessable(metric, current_value, protocol, observed_at, str(exc))

    if base.is_degenerate:
        # A perfectly flat history: no spread to score against. Any movement is
        # unprecedented, but "unprecedented" is not a number of sigmas.
        moved = current_value != base.mean
        return RiskSignal(
            metric=metric,
            protocol=protocol,
            current_value=current_value,
            baseline=base,
            pct_change_vs_baseline=pct_change(current_value, base.mean),
            severity=Severity.UNKNOWN if moved else Severity.NORMAL,
            observed_at=observed_at,
            note=(
                f"{metric} has been constant at {base.mean:g} across "
                f"{base.n} observations, so no deviation scale exists. "
                + (
                    "The current value differs from it, which is unprecedented "
                    "rather than measurably abnormal."
                    if moved
                    else "The current value matches it."
                )
            ),
        )

    z = z_score(current_value, base)
    mz = modified_z_score(current_value, base)

    known = [s for s in (z, mz) if s is not None]
    severity = severity_for(max(known, key=abs)) if known else Severity.UNKNOWN

    return RiskSignal(
        metric=metric,
        protocol=protocol,
        current_value=current_value,
        baseline=base,
        z=z,
        modified_z=mz,
        iqr_outlier=is_iqr_outlier(current_value, base),
        pct_change_vs_baseline=pct_change(current_value, base.median),
        severity=severity,
        observed_at=observed_at,
    )


def _unassessable(metric, current_value, protocol, observed_at, note) -> RiskSignal:
    """A signal that says so, rather than a signal that says nothing is wrong."""
    empty = Baseline(n=0, mean=current_value, std=0.0, median=current_value, mad=0.0,
                     q1=current_value, q3=current_value)
    return RiskSignal(
        metric=metric,
        protocol=protocol,
        current_value=current_value,
        baseline=empty,
        severity=Severity.UNKNOWN,
        observed_at=observed_at,
        note=note,
    )


def explain(signal: RiskSignal) -> str:
    """Render a signal as the deterministic sentence a report can quote.

    Deliberately not a model call. This is the paragraph §11.2 says an LLM may
    write — and having it available deterministically means the report is
    reproducible, cheap, and identical every time the same numbers are seen.
    """
    if signal.breach is not None:
        # The broken property is the finding and leads the sentence. A z-score
        # computed on the same reading is at best corroboration and at worst
        # noise: the baseline for an invariant metric is a history of the
        # invariant holding, so a deviation from it says only that the property
        # used to be satisfied — which is already implied by calling it a breach.
        out = f"{signal.metric} {signal.breach.explain()} Severity: {signal.severity.value}."
        if signal.z is not None:
            out += f" (For reference, z={signal.z:+.2f} against prior readings.)"
        return out

    if signal.severity is Severity.UNKNOWN:
        return signal.note or f"{signal.metric} could not be assessed."

    # A satisfied invariant is the whole story only when nothing else fired.
    # Short-circuiting on it unconditionally would swallow a real statistical
    # finding on a metric that happens to also carry an invariant.
    holds = signal.note and "invariant holds" in signal.note
    if holds and signal.severity is Severity.NORMAL:
        return signal.note if signal.note.endswith(".") else f"{signal.note}."

    base = signal.baseline
    parts = [
        f"{signal.metric} is {signal.current_value:,.6g} "
        f"({signal.direction} a baseline median of {base.median:,.6g} "
        f"over {base.n} observations)."
    ]
    if signal.z is not None:
        parts.append(f"z={signal.z:+.2f}")
    if signal.modified_z is not None:
        parts.append(f"robust z={signal.modified_z:+.2f}")
    if signal.iqr_outlier:
        parts.append("outside the 1.5×IQR fence")
    # A degenerate baseline yields no scores at all, leaving nothing to join.
    # Without this the sentence renders as "... over 8 observations). . Severity".
    tail = ", ".join(parts[1:])
    out = f"{parts[0]} {tail}. Severity: {signal.severity.value}." if tail else (
        f"{parts[0]} Severity: {signal.severity.value}."
    )
    if holds:
        out += f" Its {signal.note.split(signal.metric + ' ', 1)[-1]}."
    if signal.scores_disagree:
        out += (
            " The classical and robust scores disagree, which usually means the "
            "baseline window itself contains an anomaly inflating the standard "
            "deviation; the robust score is the reliable one here."
        )
    return out
