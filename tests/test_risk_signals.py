"""Risk signals: the deterministic half of the intelligence layer.

The rule these tests enforce is §11.2's: the LLM explains a signal, it never
calculates one. Everything below is arithmetic, so everything below is testable
without an API key.
"""

import random
from datetime import datetime, timezone

import pytest

from src.risk.signals import (
    CRITICAL_Z,
    ELEVATED_Z,
    HIGH_Z,
    RiskSignal,
    Severity,
    assess_metric,
    explain,
    max_severity,
    severity_for,
)
from src.risk.statistics import MIN_BASELINE_N

T0 = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)

# A quiet 30-day history of daily liquidity outflow, ~$2.3M with modest noise.
QUIET_OUTFLOW = [
    2.3, 2.1, 2.5, 2.4, 2.2, 2.0, 2.6, 2.3, 2.4, 2.2,
    2.1, 2.5, 2.3, 2.7, 2.2, 2.4, 2.0, 2.3, 2.5, 2.1,
    2.6, 2.2, 2.4, 2.3, 2.1, 2.5, 2.2, 2.4, 2.3, 2.2,
]


# --- severity banding ---------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, Severity.NORMAL),
        (1.9, Severity.NORMAL),
        (ELEVATED_Z, Severity.ELEVATED),
        (2.5, Severity.ELEVATED),
        (HIGH_Z, Severity.HIGH),
        (4.9, Severity.HIGH),
        (CRITICAL_Z, Severity.CRITICAL),
        (12.0, Severity.CRITICAL),
    ],
)
def test_severity_bands_are_a_threshold_table(score, expected):
    assert severity_for(score) is expected


def test_severity_uses_magnitude_so_collapses_are_not_ignored():
    """A metric 4σ BELOW normal is as much a finding as one 4σ above — a TVL
    collapse and a TVL spike are both events."""
    assert severity_for(-4.0) is severity_for(4.0) is Severity.HIGH


def test_unassessable_is_unknown_never_normal():
    """The distinction the band exists for. 'We could not tell' and 'we checked
    and it is fine' lead to opposite actions, and a blind spot reported as normal
    is how it becomes a clean bill of health."""
    assert severity_for(None) is Severity.UNKNOWN
    assert severity_for(None) is not Severity.NORMAL


def test_max_severity_picks_the_worst():
    assert max_severity([Severity.NORMAL, Severity.HIGH, Severity.ELEVATED]) is Severity.HIGH


def test_max_severity_prefers_any_real_assessment_over_unknown():
    assert max_severity([Severity.UNKNOWN, Severity.NORMAL]) is Severity.NORMAL


def test_max_severity_of_nothing_is_unknown():
    assert max_severity([]) is Severity.UNKNOWN


# --- assessing a metric -------------------------------------------------


def test_a_normal_reading_is_not_an_anomaly():
    sig = assess_metric("liquidity_outflow", 2.35, QUIET_OUTFLOW)
    assert sig.severity is Severity.NORMAL
    assert not sig.anomaly
    assert not sig.notable


def test_an_elevated_reading_is_notable_but_not_a_finding():
    """§15's report table lists "Elevated" and "High Anomaly" as different rows,
    and that distinction is this one: elevated earns a column in the statistics
    table, an anomaly earns a paragraph. A 2σ bar fires on one ordinary day in
    twenty per metric, which across a dozen metrics is an alert most days on
    nothing at all."""
    sig = assess_metric("liquidity_outflow", 2.7, QUIET_OUTFLOW)
    assert sig.severity is Severity.ELEVATED
    assert sig.notable
    assert not sig.anomaly


def test_the_architecture_docs_scenario_fires_critical():
    """§11.1/§11.2: $12.5M outflow against a ~$2.3M baseline."""
    sig = assess_metric(
        "liquidity_outflow", 12.5, QUIET_OUTFLOW, protocol="ethena", observed_at=T0
    )
    assert sig.anomaly
    assert sig.severity is Severity.CRITICAL
    assert sig.z > CRITICAL_Z
    assert sig.iqr_outlier
    assert sig.direction == "above"


def test_a_signal_carries_everything_needed_to_disagree_with_it():
    """§11.2's schema. A severity with no visible baseline is an assertion."""
    sig = assess_metric("tvl", 12.5, QUIET_OUTFLOW, protocol="ethena", observed_at=T0)
    assert sig.baseline.n == len(QUIET_OUTFLOW)
    assert sig.baseline.median > 0
    assert sig.z is not None and sig.modified_z is not None
    assert sig.pct_change_vs_baseline is not None
    assert sig.signal_id.startswith("risk_")


def test_signal_ids_are_stable_for_the_same_observation():
    a = assess_metric("tvl", 12.5, QUIET_OUTFLOW, protocol="ethena", observed_at=T0)
    b = assess_metric("tvl", 12.5, QUIET_OUTFLOW, protocol="ethena", observed_at=T0)
    assert a.signal_id == b.signal_id


def test_signal_ids_distinguish_protocols():
    a = assess_metric("tvl", 12.5, QUIET_OUTFLOW, protocol="ethena", observed_at=T0)
    b = assess_metric("tvl", 12.5, QUIET_OUTFLOW, protocol="hyperliquid", observed_at=T0)
    assert a.signal_id != b.signal_id


def test_a_collapse_is_detected_as_readily_as_a_spike():
    sig = assess_metric("tvl", 0.1, QUIET_OUTFLOW)
    assert sig.anomaly
    assert sig.direction == "below"
    assert sig.pct_change_vs_baseline < 0


# --- refusing to assess -------------------------------------------------


def test_too_little_history_yields_unknown_with_a_reason():
    """Not enough history is a state to report, not a score to guess."""
    sig = assess_metric("tvl", 12.5, [2.0, 2.1, 2.2])
    assert sig.severity is Severity.UNKNOWN
    assert not sig.anomaly
    assert "not enough history" in sig.note


def test_a_flat_history_reports_unprecedented_rather_than_n_sigma():
    """No spread means no deviation scale. Any movement is unprecedented, but
    'unprecedented' is not a number of sigmas."""
    sig = assess_metric("tvl", 500.0, [100.0] * 12)
    assert sig.severity is Severity.UNKNOWN
    assert "unprecedented" in sig.note
    assert sig.z is None


def test_a_flat_history_matched_exactly_is_normal():
    sig = assess_metric("tvl", 100.0, [100.0] * 12)
    assert sig.severity is Severity.NORMAL
    assert not sig.anomaly


def test_exactly_the_minimum_history_is_assessable():
    sig = assess_metric("tvl", 99.0, [2.0, 2.1, 2.2, 2.3, 2.0, 2.1, 2.2, 2.3][:MIN_BASELINE_N])
    assert sig.severity is not Severity.UNKNOWN


# --- classical vs robust ------------------------------------------------


def test_severity_takes_the_worse_of_the_two_scores():
    """When the scores disagree, the robust one is detecting something the
    classical one has been blinded to. Under-reporting a real anomaly costs more
    than looking twice at a false one."""
    contaminated = [2.3, 2.1, 2.5, 2.4, 2.2, 2.0, 2.6, 2.3, 2.4, 90.0]
    sig = assess_metric("liquidity_outflow", 85.0, contaminated)
    assert abs(sig.z) < HIGH_Z, "classical score is masked by the historical spike"
    assert sig.anomaly, "but the signal still fires"


def test_disagreement_between_the_scores_is_surfaced():
    contaminated = [2.3, 2.1, 2.5, 2.4, 2.2, 2.0, 2.6, 2.3, 2.4, 90.0]
    sig = assess_metric("liquidity_outflow", 85.0, contaminated)
    assert sig.scores_disagree
    assert "robust score is the reliable one" in explain(sig)


def test_scores_do_not_disagree_on_clean_data():
    assert not assess_metric("tvl", 12.5, QUIET_OUTFLOW).scores_disagree


# --- explanation is deterministic --------------------------------------


def test_explain_is_reproducible_and_quotes_the_numbers():
    """The paragraph §11.2 says a model may write, available without one — so
    the report is reproducible, free, and identical every time."""
    sig = assess_metric("liquidity_outflow", 12.5, QUIET_OUTFLOW, protocol="ethena")
    first, second = explain(sig), explain(sig)
    assert first == second
    assert "liquidity_outflow" in first
    assert "12.5" in first
    assert "z=" in first
    assert "critical" in first


def test_explain_says_so_when_it_could_not_assess():
    text = explain(assess_metric("tvl", 1.0, [1.0, 2.0]))
    assert "not enough history" in text


# --- benchmark: §19's anomaly-detection metrics ------------------------


def _labelled_series(seed: int, n_days: int = 400):
    """A synthetic history with known injected anomalies.

    Synthetic because there is no labelled on-chain incident set here yet, and a
    detector with no measured false-positive rate is not a detector — it is a
    threshold nobody has checked. When real labelled incidents exist, this
    generator is what they replace.
    """
    rng = random.Random(seed)
    history, labels = [], []
    for day in range(n_days):
        if day > 60 and rng.random() < 0.04:
            value = rng.uniform(6.0, 15.0)   # injected anomaly
            labels.append(True)
        else:
            value = rng.gauss(2.3, 0.25)     # ordinary day
            labels.append(False)
        history.append(max(value, 0.01))
    return history, labels


def test_detector_precision_and_recall_on_labelled_data():
    """§19 asks for precision, recall, F1 and false-positive rate. Measured, not
    assumed — and asserted loosely enough that this is a regression guard rather
    than a claim that these numbers are calibrated truth."""
    series, labels = _labelled_series(seed=20260820)
    window = 45

    tp = fp = tn = fn = 0
    for i in range(window, len(series)):
        # The baseline is the PRIOR window, never including the point under test.
        sig = assess_metric("outflow", series[i], series[i - window : i])
        predicted, actual = sig.anomaly, labels[i]
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)
    fpr = fp / (fp + tn)

    print(
        f"\n  precision={precision:.3f} recall={recall:.3f} "
        f"f1={f1:.3f} fpr={fpr:.3f}  (tp={tp} fp={fp} fn={fn} tn={tn})"
    )

    assert recall >= 0.90, "a detector that misses real anomalies is not one"
    assert fpr <= 0.02, "alert fatigue is the failure mode that gets a detector ignored"
    assert precision >= 0.90
    assert f1 >= 0.90


def test_a_quiet_series_produces_almost_no_alerts():
    """The false-positive side, isolated: 400 ordinary days with nothing
    injected. Anything that fires here is noise."""
    rng = random.Random(7)
    series = [rng.gauss(2.3, 0.25) for _ in range(400)]
    window = 45
    fired = sum(
        assess_metric("outflow", series[i], series[i - window : i]).anomaly
        for i in range(window, len(series))
    )
    assert fired / (len(series) - window) < 0.02


def test_signals_round_trip_through_json():
    """Signals travel in graph state and land in reports; they must serialize."""
    sig = assess_metric("outflow", 12.5, QUIET_OUTFLOW, protocol="ethena", observed_at=T0)
    assert RiskSignal.model_validate_json(sig.model_dump_json()).severity is sig.severity
