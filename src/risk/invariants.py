"""Properties that must hold, checked without a baseline.

Statistical detection asks whether a value is unusual for this metric. Some
metrics are not interesting when unusual — they are interesting when *wrong*, and
what counts as wrong is fixed by the protocol's design rather than learned from
history.

A wrapper token is the clearest case. WHYPE holds one unit of native coin per
wrapped token; that is the contract's entire job. So the ratio reads 1.0 every
hour, forever, until the day it does not. Handed to a z-score that series is
degenerate: a flat history has no spread, so there is no scale on which to
express a deviation, and the engine declines to score it. Correctly, and
uselessly — the system ends up blind to the one number whose breach matters most,
and blind in a way that reports as "unknown" rather than "broken". Measured on
2026-08-21: nine consecutive readings of exactly 1.0, severity UNKNOWN on all
nine, and a hypothetical 10% break would have scored UNKNOWN too.

Two properties follow from checking against a target instead of a distribution,
and both are why this is not a special case of the statistical path:

  * **No history is required.** A backing ratio of 0.8 is a breach on the first
    reading. The statistical path needs `MIN_BASELINE_N` observations before it
    will say anything at all, so on a freshly wiped feature store it would take
    eight hours to notice an insolvent wrapper — and then report UNKNOWN.
  * **Constancy is evidence of health, not an obstacle to measurement.** The flat
    series that defeats a z-score is precisely the invariant holding.

Bounds are directional because breaches usually are. A wrapper holding less
native coin than it has issued tokens is under-collateralised, which is a
solvency failure. One holding more has received a donation or a stray transfer:
odd, worth a note, not a crisis. Modelling both as "deviation from 1.0" would
give a benign event the same severity as an insolvency.

An invariant is a claim about a protocol's design, so it is only as good as the
reading behind it. Both figures in a ratio must come from the same block or the
ratio is an artefact of the gap between two reads — see `contracts.backing_ratio`,
which batches its calls for exactly this reason.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.risk.severity import Severity


class Bound(str, Enum):
    """Which side of the target counts as a breach."""

    EQUALS = "equals"        # both directions
    AT_LEAST = "at_least"    # only below the target
    AT_MOST = "at_most"      # only above the target


# Relative deviation past which a breach escalates. These are properties of what
# the number means, not statistics, so they are not the z-thresholds and must not
# be read as them: a 1% shortfall on a supposedly exact 1:1 instrument is not
# "unusual", it is a million dollars missing from every hundred million issued.
DEFAULT_TOLERANCE = 1e-9   # dust: integer division of two 18-decimal balances
DEFAULT_HIGH_AT = 1e-3     # 0.1%
DEFAULT_CRITICAL_AT = 1e-2  # 1%


class Invariant(BaseModel):
    """A property a metric must satisfy, and how badly it is broken if it does not."""

    model_config = ConfigDict(frozen=True)

    target: float
    bound: Bound = Bound.EQUALS
    # Relative to |target|, so a single tolerance reads the same on a ratio and
    # on a balance. Below this a deviation is dust, not a finding.
    tolerance: float = Field(default=DEFAULT_TOLERANCE, ge=0.0)
    high_at: float = Field(default=DEFAULT_HIGH_AT, gt=0.0)
    critical_at: float = Field(default=DEFAULT_CRITICAL_AT, gt=0.0)
    # Why this must hold. Carried into the report, because a severity with no
    # stated reason is a number the reader has to take on trust.
    rationale: str

    @model_validator(mode="after")
    def _bands_ascend(self) -> Invariant:
        if not self.tolerance <= self.high_at <= self.critical_at:
            raise ValueError(
                "invariant bands must ascend: tolerance <= high_at <= critical_at, "
                f"got {self.tolerance} / {self.high_at} / {self.critical_at}"
            )
        return self

    def deviation(self, value: float) -> float:
        """Signed deviation from the target, relative to it.

        Falls back to an absolute deviation when the target is zero, where a
        relative one is undefined. A "must be zero" invariant is therefore read in
        the metric's own units, which is the only meaning available.
        """
        if self.target == 0:
            return value
        return (value - self.target) / abs(self.target)

    def breaches(self, value: float) -> bool:
        d = self.deviation(value)
        if abs(d) <= self.tolerance:
            return False
        if self.bound is Bound.AT_LEAST:
            return d < 0
        if self.bound is Bound.AT_MOST:
            return d > 0
        return True

    def severity_for(self, value: float) -> Severity:
        """NORMAL when satisfied — never UNKNOWN. An invariant is always checkable."""
        if not self.breaches(value):
            return Severity.NORMAL
        magnitude = abs(self.deviation(value))
        if magnitude >= self.critical_at:
            return Severity.CRITICAL
        if magnitude >= self.high_at:
            return Severity.HIGH
        return Severity.ELEVATED

    def check(self, value: float) -> Breach | None:
        """The breach, or `None` when the property holds."""
        if not self.breaches(value):
            return None
        return Breach(
            invariant=self,
            value=value,
            deviation=self.deviation(value),
            severity=self.severity_for(value),
        )

    def describe(self) -> str:
        word = {
            Bound.EQUALS: "must equal",
            Bound.AT_LEAST: "must be at least",
            Bound.AT_MOST: "must be at most",
        }[self.bound]
        return f"{word} {self.target:,.6g}"


class Breach(BaseModel):
    """A violated invariant, with everything needed to check the arithmetic."""

    model_config = ConfigDict(frozen=True)

    invariant: Invariant
    value: float
    deviation: float
    severity: Severity

    @property
    def direction(self) -> str:
        return "below" if self.deviation < 0 else "above"

    def explain(self) -> str:
        inv = self.invariant
        return (
            f"{inv.describe()}, and reads {self.value:,.6g} — "
            f"{abs(self.deviation):.4%} {self.direction} target. "
            f"{inv.rationale} "
            f"This is a broken invariant, not an unusual reading: it is judged "
            f"against the protocol's design rather than against recent history, "
            f"so no baseline is required and none was used."
        )


def holding_note(invariant: Invariant, value: float) -> str:
    """The sentence for an invariant that is satisfied.

    Worth saying explicitly. A metric that has been constant forever looks
    identical to a metric nobody is collecting, and the difference between "held
    at 1.0 on every reading" and silence is the difference between a check that
    passed and a check that never ran.
    """
    slack = invariant.deviation(value)
    if invariant.bound is not Bound.EQUALS and abs(slack) > invariant.tolerance:
        side = "below" if slack < 0 else "above"
        return (
            f"invariant holds ({invariant.describe()}, reads {value:,.6g}, "
            f"{abs(slack):.4%} {side} target on the permitted side)"
        )
    return f"invariant holds ({invariant.describe()}, reads {value:,.6g})"
