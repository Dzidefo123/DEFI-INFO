"""Statistical primitives for anomaly detection.

Every case here is arithmetic that can be checked by hand — which is the whole
argument for starting with statistics rather than ML.
"""

import pytest

from src.risk.statistics import (
    IQR_FENCE,
    MIN_BASELINE_N,
    Baseline,
    ChangePoint,
    InsufficientHistory,
    baseline,
    detect_change_point,
    iqr_bounds,
    is_iqr_outlier,
    modified_z_score,
    pct_change,
    rolling,
    z_score,
)

# Ten observations with mean 10 and a known spread.
STEADY = [8.0, 9.0, 10.0, 11.0, 12.0, 9.0, 10.0, 11.0, 10.0, 10.0]


# --- baseline construction ---------------------------------------------


def test_baseline_summarizes_classical_and_robust_spread():
    b = baseline(STEADY)
    assert b.n == 10
    assert b.mean == pytest.approx(10.0)
    assert b.median == pytest.approx(10.0)
    assert b.std > 0
    assert b.mad > 0
    assert b.iqr == b.q3 - b.q1


def test_too_little_history_is_refused_not_extrapolated():
    """A z-score over four points is arithmetic, not evidence."""
    with pytest.raises(InsufficientHistory, match="not enough history"):
        baseline([1.0, 2.0, 3.0, 4.0])


def test_the_minimum_is_exactly_min_baseline_n():
    baseline([1.0] * MIN_BASELINE_N)  # does not raise
    with pytest.raises(InsufficientHistory):
        baseline([1.0] * (MIN_BASELINE_N - 1))


def test_a_flat_history_is_degenerate_not_an_error():
    """A metric that has never moved is a real state, not a failure."""
    b = baseline([5.0] * 10)
    assert b.is_degenerate
    assert b.std == 0.0 and b.mad == 0.0


# --- z-scores -----------------------------------------------------------


def test_z_score_matches_the_hand_calculation():
    b = Baseline(n=10, mean=2.3, std=1.8, median=2.3, mad=1.0, q1=2.0, q3=2.6)
    assert z_score(12.5, b) == pytest.approx((12.5 - 2.3) / 1.8)


def test_the_architecture_docs_worked_example():
    """§11.1: outflow $12.5M against a $2.3M mean and $1.8M σ gives z = 5.67."""
    b = Baseline(n=30, mean=2_300_000, std=1_800_000, median=2_300_000,
                 mad=1_000_000, q1=1_500_000, q3=3_000_000)
    assert z_score(12_500_000, b) == pytest.approx(5.67, abs=0.01)


def test_z_score_of_a_constant_series_is_undefined_not_infinite():
    """Returning inf, or 0, or dividing by an epsilon would manufacture a finding
    out of a division."""
    assert z_score(99.0, baseline([5.0] * 10)) is None


def test_modified_z_score_of_a_zero_mad_series_is_undefined():
    assert modified_z_score(99.0, baseline([5.0] * 10)) is None


def test_z_score_is_signed_so_direction_survives():
    b = baseline(STEADY)
    assert z_score(20.0, b) > 0
    assert z_score(1.0, b) < 0


# --- the robustness argument -------------------------------------------


def test_a_spike_in_the_baseline_window_blinds_the_classical_score():
    """The reason `modified_z_score` exists.

    One historical spike inflates σ enough that a present spike of the same size
    scores as ordinary — the detector is blinded by exactly the kind of event it
    was built to find. The median barely moves, so the robust score still fires.
    """
    contaminated = [10.0, 10.0, 11.0, 9.0, 10.0, 10.0, 11.0, 9.0, 10.0, 200.0]
    b = baseline(contaminated)

    classical = z_score(200.0, b)
    robust = modified_z_score(200.0, b)

    assert abs(classical) < 3.0, "classical score has been masked by the outlier"
    assert abs(robust) > 10.0, "robust score still detects it"


def test_on_clean_history_the_two_scores_broadly_agree():
    """Robustness must not cost detection on well-behaved data."""
    b = baseline(STEADY)
    assert z_score(20.0, b) == pytest.approx(modified_z_score(20.0, b), rel=0.5)


# --- IQR ----------------------------------------------------------------


def test_iqr_bounds_are_tukey_fences():
    b = baseline(STEADY)
    low, high = iqr_bounds(b)
    assert low == pytest.approx(b.q1 - IQR_FENCE * b.iqr)
    assert high == pytest.approx(b.q3 + IQR_FENCE * b.iqr)


def test_iqr_flags_values_outside_the_fence():
    b = baseline(STEADY)
    assert is_iqr_outlier(1000.0, b)
    assert is_iqr_outlier(-1000.0, b)
    assert not is_iqr_outlier(10.0, b)


def test_iqr_makes_no_normality_assumption():
    """Its value is as a second opinion that does not share the z-score's
    implicit normal assumption — almost no on-chain metric is normal."""
    heavy_tailed = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 8.0, 40.0]
    b = baseline(heavy_tailed)
    assert is_iqr_outlier(40.0, b)


# --- percentage change --------------------------------------------------


def test_pct_change_is_relative():
    assert pct_change(150.0, 100.0) == pytest.approx(0.5)
    assert pct_change(50.0, 100.0) == pytest.approx(-0.5)


def test_pct_change_from_zero_is_undefined():
    """Growth from zero has no percentage. It is a different kind of event —
    something started — and a report should say that instead of 'inf%'."""
    assert pct_change(100.0, 0.0) is None


def test_pct_change_uses_absolute_previous_so_sign_is_not_flipped():
    """Net flow can be negative; a move from -100 to -50 is a 50% reduction in
    magnitude, not -50%."""
    assert pct_change(-50.0, -100.0) == pytest.approx(0.5)


# --- rolling windows ----------------------------------------------------


def test_rolling_produces_successive_windows_oldest_first():
    assert rolling([1, 2, 3, 4], 2) == [[1, 2], [2, 3], [3, 4]]


def test_rolling_a_too_short_series_yields_nothing():
    """A partial window pretending to be a full one is worse than no window."""
    assert rolling([1, 2], 5) == []


def test_rolling_rejects_a_nonpositive_window():
    with pytest.raises(ValueError):
        rolling([1, 2, 3], 0)


# --- change point detection --------------------------------------------


def test_a_clean_level_shift_is_located_exactly():
    series = [10.0] * 10 + [50.0] * 10
    cp = detect_change_point(series)
    assert cp is not None
    assert cp.index == 10
    assert cp.direction == "increase"
    assert cp.before_mean == pytest.approx(10.0)
    assert cp.after_mean == pytest.approx(50.0)


def test_a_downward_shift_reports_its_direction():
    cp = detect_change_point([100.0] * 8 + [20.0] * 8)
    assert cp.direction == "decrease"


def test_a_stationary_series_yields_a_weak_score():
    """Noise around a constant level should not read as a regime change."""
    series = [10.0, 10.5, 9.8, 10.2, 9.9, 10.1, 10.3, 9.7, 10.0, 10.4,
              9.6, 10.2, 10.1, 9.9, 10.0, 10.3]
    cp = detect_change_point(series)
    assert cp is None or cp.score < 2.0


def test_a_constant_series_has_no_change_point():
    assert detect_change_point([7.0] * 20) is None


def test_a_series_too_short_to_split_returns_none():
    assert detect_change_point([1.0, 2.0, 3.0], min_segment=4) is None


def test_change_point_answers_a_question_z_scores_cannot():
    """A z-score fires once and then goes quiet as the new level is absorbed into
    the baseline. It says 'today's outflow was large'. A change point says
    'outflows have been elevated since the 14th' — which is the sentence an
    investigation actually needs.
    """
    series = [2.0] * 14 + [12.0] * 8
    cp = detect_change_point(series)
    assert cp.index == 14
    assert cp.score > 3.0


def test_change_point_score_is_finite_and_serializable():
    """Two perfectly constant segments separate infinitely; inf is neither
    meaningful as a magnitude nor JSON-serializable."""
    cp = detect_change_point([1.0] * 10 + [2.0] * 10)
    assert cp.score == 1e6
    assert ChangePoint.model_validate(cp.model_dump())
