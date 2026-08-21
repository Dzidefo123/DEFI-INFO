"""How bad a finding is, on one ordered scale.

Extracted from `signals` so that the statistical path and the invariant path can
speak the same language without importing each other. Nothing here knows what a
z-score is or what an invariant is; it only knows that findings are ranked and
that "we could not tell" is not a rank.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum


class Severity(str, Enum):
    """How far outside normal a signal sits. Ordered."""

    UNKNOWN = "unknown"    # could not be assessed; NOT the same as normal
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


_ORDER = {
    Severity.UNKNOWN: -1,
    Severity.NORMAL: 0,
    Severity.ELEVATED: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def rank(severity: Severity) -> int:
    """Position on the scale. UNKNOWN sorts below NORMAL, deliberately.

    It is not a mild finding, it is the absence of one, and ordering it below
    NORMAL is what stops `max_severity` from letting a metric nobody could
    measure outrank one that was measured and found fine.
    """
    return _ORDER[severity]


def at_least(severity: Severity, floor: Severity) -> bool:
    return _ORDER[severity] >= _ORDER[floor]


def max_severity(severities: Sequence[Severity]) -> Severity:
    """The worst severity present. UNKNOWN loses to any real assessment.

    That last part is what makes this safe to use for combining verdicts from
    different kinds of check. A metric with a satisfied invariant and too little
    history to score statistically is NORMAL — something was actually checked and
    passed — rather than UNKNOWN because something else could not run.
    """
    if not severities:
        return Severity.UNKNOWN
    return max(severities, key=rank)
