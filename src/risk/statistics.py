"""Explainable statistics for anomaly detection. No model, no ML, no network.

§11.1 says start with statistics before complex ML, and the reason is not
simplicity — it is that a z-score can be printed in a report and argued with. An
Isolation Forest score cannot. Every function here produces a number a reader can
recompute by hand from the same inputs.

Four decisions distinguish this from the textbook formulas, and each one exists
because the textbook version is actively wrong for this data.

**The baseline excludes the point being tested.** An outlier included in its own
baseline drags the mean toward itself and inflates the standard deviation, so the
larger the anomaly, the better it hides. This is the single most common way a
z-score anomaly detector silently fails to fire on the event it was built for.

**Robust statistics are offered alongside the classical ones.** Mean and standard
deviation are themselves destroyed by outliers. On a series where a $12.5M
outflow follows a history around $2.3M, one earlier spike in the baseline window
inflates σ enough to make today's spike look ordinary. Median and MAD do not
move, so `modified_z_score` keeps firing where `z_score` has been blinded.

**Undefined is returned as undefined.** A constant series has σ = 0 and no
z-score exists. Returning `inf`, or 0, or silently substituting a small epsilon,
manufactures a finding out of a division. These functions return `None` and the
caller must say "cannot assess" — the statistical form of §13's
INSUFFICIENT_EVIDENCE.

**Short histories are refused, not extrapolated.** A z-score over four
observations is arithmetic, not evidence. Below `MIN_BASELINE_N` there is no
baseline, and the honest output is that we have not been watching long enough.
"""

from __future__ import annotations

import statistics as _stats
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

# Below this many prior observations there is no baseline worth the name. A
# z-score over a handful of points is dominated by the sample itself: the
# estimated σ is unstable, so the score swings wildly with each new reading and
# fires on noise. Eight is a floor, not a recommendation — it is the point below
# which the number is misleading rather than merely imprecise.
MIN_BASELINE_N = 8

# Converts MAD to a σ-comparable scale for a normal distribution, so a modified
# z-score reads on the same axis as a classical one and the two can be compared
# directly. 0.6745 is the standard normal's 75th percentile.
_MAD_SCALE = 0.6745

# Tukey's fence multiplier. 1.5×IQR is the conventional "outlier" boundary.
IQR_FENCE = 1.5


class InsufficientHistory(ValueError):
    """Raised when a baseline is requested from too few observations."""


class Baseline(BaseModel):
    """Summary of what a metric normally does, computed from its history.

    Carries both classical (mean/σ) and robust (median/MAD) descriptions, because
    which one to trust depends on whether the history itself contains outliers —
    a question the caller is better placed to answer than this module.
    """

    model_config = ConfigDict(frozen=True)

    n: int = Field(ge=0)
    mean: float
    std: float = Field(ge=0.0)
    median: float
    mad: float = Field(ge=0.0)   # median absolute deviation
    q1: float
    q3: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1

    @property
    def is_degenerate(self) -> bool:
        """True when the history is perfectly flat, so no spread exists to score
        against. Not an error — a metric that has never moved is a real state,
        and the honest report is that any change is unprecedented rather than
        that it is N sigma out."""
        return self.std == 0.0 and self.mad == 0.0


def baseline(history: Sequence[float]) -> Baseline:
    """Describe a metric's normal behaviour from its prior observations.

    `history` must NOT contain the observation being tested. See the module
    docstring: a point included in its own baseline suppresses its own score.
    """
    n = len(history)
    if n < MIN_BASELINE_N:
        raise InsufficientHistory(
            f"need at least {MIN_BASELINE_N} prior observations to establish a "
            f"baseline, got {n}; report this as 'not enough history' rather than "
            f"scoring it"
        )

    values = [float(v) for v in history]
    med = _stats.median(values)
    # quantiles(n=4) needs at least 2 points; MIN_BASELINE_N already guarantees more.
    q1, _, q3 = _stats.quantiles(values, n=4, method="inclusive")

    return Baseline(
        n=n,
        mean=_stats.fmean(values),
        std=_stats.stdev(values),
        median=med,
        mad=_stats.median([abs(v - med) for v in values]),
        q1=q1,
        q3=q3,
    )


def z_score(value: float, base: Baseline) -> float | None:
    """Classical z-score: how many standard deviations from the mean.

    `None` when σ is zero — the score is undefined, and inventing one from a
    division by an epsilon reports arithmetic as evidence.
    """
    if base.std == 0.0:
        return None
    return (value - base.mean) / base.std


def modified_z_score(value: float, base: Baseline) -> float | None:
    """Robust z-score built on median and MAD rather than mean and σ.

    Prefer this when the baseline window may itself contain anomalies, which for
    on-chain metrics is the default assumption rather than an edge case. A single
    historical spike inflates σ enough to mask a present one; it moves the median
    barely at all.
    """
    if base.mad == 0.0:
        return None
    return _MAD_SCALE * (value - base.median) / base.mad


def iqr_bounds(base: Baseline, fence: float = IQR_FENCE) -> tuple[float, float]:
    """Tukey fences. Outside these, a value is an outlier by the IQR rule."""
    span = fence * base.iqr
    return base.q1 - span, base.q3 + span


def is_iqr_outlier(value: float, base: Baseline, fence: float = IQR_FENCE) -> bool:
    """Distribution-free outlier test.

    Assumes nothing about normality, which matters because almost no on-chain
    metric is normally distributed — volumes and flows are heavy-tailed, so a
    z-score's implicit normal assumption overstates how surprising a large value
    is. Useful precisely as a second opinion that does not share that assumption.
    """
    low, high = iqr_bounds(base, fence)
    return value < low or value > high


def pct_change(current: float, previous: float) -> float | None:
    """Relative change. `None` when the previous value is zero.

    Growth from zero has no percentage — it is not "infinite growth", it is a
    different kind of event (something started), and a report should say so.
    """
    if previous == 0:
        return None
    return (current - previous) / abs(previous)


def rolling(values: Sequence[float], window: int) -> list[list[float]]:
    """Successive windows of `window` consecutive values, oldest first.

    Empty when the series is shorter than the window — the caller gets nothing to
    misread rather than a partial window pretending to be a full one.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    return [
        list(values[i : i + window]) for i in range(0, len(values) - window + 1)
    ]


class ChangePoint(BaseModel):
    """Where a series' level shifted, and how sharply."""

    model_config = ConfigDict(frozen=True)

    index: int          # first index belonging to the SECOND segment
    before_mean: float
    after_mean: float
    score: float = Field(ge=0.0)  # separation in pooled standard deviations

    @property
    def direction(self) -> str:
        return "increase" if self.after_mean > self.before_mean else "decrease"


def detect_change_point(
    series: Sequence[float], min_segment: int = 4
) -> ChangePoint | None:
    """Find the single most likely level shift by exhaustive binary segmentation.

    Every valid split is scored by the separation of its two segment means in
    pooled standard deviations, and the best is returned. Exhaustive rather than
    greedy because these series are short — hundreds of points, not millions — so
    the exact answer is affordable and the approximation buys nothing.

    This answers a question a z-score cannot: a z-score asks "is today unusual
    against history", which fires once and then goes quiet as the new level is
    absorbed into the baseline. A change point asks "did behaviour shift, and
    when" — the difference between "today's outflow was large" and "outflows have
    been elevated since the 14th", which is the sentence an investigation needs.

    Returns `None` when the series is too short to split, or when both segments
    are constant and identical (nothing changed).
    """
    values = [float(v) for v in series]
    if len(values) < 2 * min_segment:
        return None

    best: ChangePoint | None = None
    for i in range(min_segment, len(values) - min_segment + 1):
        left, right = values[:i], values[i:]
        lm, rm = _stats.fmean(left), _stats.fmean(right)

        # Pooled spread of the two candidate segments. A split is only impressive
        # relative to how noisy the segments themselves are.
        pooled = _pooled_std(left, right)
        if pooled == 0.0:
            # Both segments constant: a real shift if the levels differ at all,
            # and there is no noise to normalize against.
            score = float("inf") if lm != rm else 0.0
        else:
            score = abs(rm - lm) / pooled

        if best is None or score > best.score:
            best = ChangePoint(
                index=i,
                before_mean=lm,
                after_mean=rm,
                # inf is not serializable and not meaningful as a magnitude;
                # cap it at a value that reads as "total separation".
                score=min(score, 1e6),
            )

    if best is not None and best.score == 0.0:
        return None
    return best


def _pooled_std(left: Sequence[float], right: Sequence[float]) -> float:
    """Pooled standard deviation of two segments, 0.0 when both are constant."""
    n_l, n_r = len(left), len(right)
    var_l = _stats.variance(left) if n_l > 1 else 0.0
    var_r = _stats.variance(right) if n_r > 1 else 0.0
    dof = (n_l - 1) + (n_r - 1)
    if dof <= 0:
        return 0.0
    pooled_var = ((n_l - 1) * var_l + (n_r - 1) * var_r) / dof
    return pooled_var**0.5
