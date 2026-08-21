"""Invariant checks: properties judged against a protocol's design, not history.

The case that motivated this is measured, not hypothetical. On 2026-08-21 the
feature store held nine consecutive readings of `wrapper_backing_ratio` at
exactly 1.0 — the invariant holding, which is the healthiest state available. The
statistical engine reported UNKNOWN on all nine, because a flat series has no
spread to score against, and would have reported UNKNOWN if the tenth reading had
been 0.8. The system was blind to the one number whose breach matters most, and
blind in the shape of a shrug.
"""

import pytest
from pydantic import ValidationError

from src.risk.invariants import (
    DEFAULT_CRITICAL_AT,
    DEFAULT_HIGH_AT,
    Bound,
    Invariant,
)
from src.risk.severity import Severity, max_severity
from src.risk.signals import assess_metric, explain
from src.risk.statistics import MIN_BASELINE_N

WRAPPER = Invariant(
    target=1.0,
    bound=Bound.AT_LEAST,
    rationale="A wrapper must hold one native coin per wrapped token.",
)
EXACT = Invariant(target=1.0, rationale="Must be exactly one.")
CEILING = Invariant(target=100.0, bound=Bound.AT_MOST, rationale="Must not exceed 100.")

FLAT = [1.0] * 9          # what the store actually contains
VARIED = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 1.01, 0.99, 1.0]


def _signal(value, history=FLAT, invariant=WRAPPER):
    return assess_metric(
        "wrapper_backing_ratio", value, history, protocol="hyperevm", invariant=invariant
    )


# --- the blindness this fixes -------------------------------------------


def test_the_measured_case_without_an_invariant_is_blind():
    """The regression, stated as it actually occurred. A 20% shortfall against
    nine readings of 1.0 scores UNKNOWN — not NORMAL, so the engine is honest,
    but honest blindness on a solvency invariant is still blindness."""
    bare = assess_metric("wrapper_backing_ratio", 0.8, FLAT)
    assert bare.severity is Severity.UNKNOWN
    assert not bare.anomaly


def test_the_same_case_with_an_invariant_is_critical():
    signal = _signal(0.8)
    assert signal.severity is Severity.CRITICAL
    assert signal.anomaly
    assert signal.breach is not None


def test_a_satisfied_invariant_on_a_flat_series_is_normal_not_unknown():
    """The other half of the fix, and the easier half to get wrong. A series
    constant at its target is the invariant working; reporting UNKNOWN forever
    would make a healthy contract indistinguishable from an uncollected one."""
    signal = _signal(1.0)
    assert signal.severity is Severity.NORMAL
    assert signal.breach is None


def test_an_invariant_needs_no_history_at_all():
    """A backing ratio of 0.8 is wrong on the first reading. Requiring a baseline
    first would mean a freshly wiped feature store cannot report an insolvent
    wrapper for `MIN_BASELINE_N` collection intervals."""
    assert len(FLAT) >= MIN_BASELINE_N  # guard: the point is about the empty case
    signal = _signal(0.8, history=[])
    assert signal.severity is Severity.CRITICAL
    assert signal.anomaly


def test_an_invariant_metric_is_never_unknown():
    """Across every history shape the statistical path can fail on."""
    for history in ([], [1.0], FLAT, VARIED):
        assert _signal(1.0, history=history).severity is not Severity.UNKNOWN
        assert _signal(0.5, history=history).severity is not Severity.UNKNOWN


# --- direction ----------------------------------------------------------


def test_under_backing_breaches_but_over_backing_does_not():
    """Not symmetry for its own sake. Below 1.0 the wrapper has issued tokens it
    cannot redeem. Above 1.0 someone sent native coin without minting — a
    donation or a mistake, and no holder is worse off."""
    assert _signal(0.95).breach is not None
    assert _signal(1.05).breach is None


def test_over_backing_is_still_described_rather_than_hidden():
    """It is not a breach, but it is not nothing either: a wrapper drifting above
    its target means something is depositing without minting."""
    note = _signal(1.05).note
    assert "above target on the permitted side" in note
    assert "5.0000%" in note


def test_an_equals_bound_breaches_in_both_directions():
    assert assess_metric("m", 0.95, FLAT, invariant=EXACT).breach is not None
    assert assess_metric("m", 1.05, FLAT, invariant=EXACT).breach is not None


def test_an_at_most_bound_breaches_only_above():
    assert assess_metric("m", 150.0, [], invariant=CEILING).breach is not None
    assert assess_metric("m", 50.0, [], invariant=CEILING).breach is None


def test_breach_direction_is_reported():
    assert _signal(0.9).breach.direction == "below"
    assert assess_metric("m", 1.1, FLAT, invariant=EXACT).breach.direction == "above"


# --- magnitude ----------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.0 - DEFAULT_CRITICAL_AT * 5, Severity.CRITICAL),   # 5%
        (1.0 - DEFAULT_CRITICAL_AT, Severity.CRITICAL),       # 1%
        (1.0 - DEFAULT_HIGH_AT * 2, Severity.HIGH),           # 0.2%
        (1.0 - DEFAULT_HIGH_AT, Severity.HIGH),               # 0.1%
        (1.0 - DEFAULT_HIGH_AT / 10, Severity.ELEVATED),      # 0.01%
    ],
)
def test_severity_scales_with_the_size_of_the_shortfall(value, expected):
    """A dust discrepancy and a 5% shortfall are both breaches and are not the
    same event. Bands are boundaries-inclusive at the named threshold."""
    assert _signal(value).severity is expected


def test_dust_below_the_tolerance_is_not_a_breach():
    """Both figures come from integer balances divided at 18 decimals, so exact
    equality is not something the arithmetic can promise."""
    assert _signal(1.0 - WRAPPER.tolerance / 10).breach is None


def test_tolerance_is_relative_so_it_reads_the_same_at_any_scale():
    big = Invariant(target=1_000_000.0, rationale="scale check")
    assert big.deviation(1_010_000.0) == pytest.approx(0.01)
    assert EXACT.deviation(1.01) == pytest.approx(0.01)


def test_a_zero_target_falls_back_to_absolute_deviation():
    """A relative deviation from zero is undefined; the metric's own units are
    the only meaning left."""
    zero = Invariant(target=0.0, rationale="must be zero")
    assert zero.deviation(5.0) == 5.0
    assert zero.check(5.0) is not None


# --- a breach is a finding at any size ----------------------------------


def test_any_breach_counts_as_an_anomaly_even_below_the_z_threshold():
    """The 3σ bar exists to suppress false positives from ordinary variation. A
    violated invariant is not ordinary variation, so there is no false-positive
    rate to suppress — an ELEVATED breach is still a finding."""
    signal = _signal(1.0 - DEFAULT_HIGH_AT / 10)
    assert signal.severity is Severity.ELEVATED
    assert signal.anomaly
    assert signal.notable


# --- interaction with the statistical path ------------------------------


def test_the_two_checks_combine_by_taking_the_worse():
    """A metric can be both wrong and unusual, and neither check subsumes the
    other."""
    spiky = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5]
    signal = assess_metric("m", 0.5, spiky, invariant=WRAPPER)
    assert signal.severity is Severity.CRITICAL


def test_a_satisfied_invariant_does_not_swallow_a_statistical_finding():
    """A metric carrying an invariant still gets scored against its history, and
    a real anomaly on the permitted side must survive into the explanation."""
    signal = assess_metric("m", 3.0, VARIED, invariant=WRAPPER)
    assert signal.breach is None
    assert signal.severity is Severity.CRITICAL
    assert "z=" in explain(signal)


def test_an_unknown_statistical_verdict_never_outranks_a_checked_invariant():
    """`max_severity` ranks UNKNOWN below NORMAL for exactly this reason."""
    assert max_severity([Severity.NORMAL, Severity.UNKNOWN]) is Severity.NORMAL
    assert _signal(1.0, history=[]).severity is Severity.NORMAL


# --- what the report says -----------------------------------------------


def test_the_explanation_leads_with_the_broken_property():
    text = explain(_signal(0.8))
    assert text.index("must be at least") < text.index("Severity")
    assert "broken invariant, not an unusual reading" in text


def test_the_explanation_carries_the_rationale():
    """A severity with no stated reason is a number the reader must take on
    trust."""
    assert "cannot be redeemed" in explain(
        assess_metric(
            "wrapper_backing_ratio",
            0.8,
            FLAT,
            invariant=Invariant(
                target=1.0,
                bound=Bound.AT_LEAST,
                rationale="A shortfall means tokens exist that cannot be redeemed.",
            ),
        )
    )


def test_the_explanation_states_that_no_baseline_was_used():
    """Otherwise a reader has no way to tell this verdict apart from a z-score,
    and would reasonably discount it on a nine-observation history."""
    assert "no baseline is required and none was used" in explain(_signal(0.8))


def test_a_satisfied_invariant_reads_as_a_check_that_passed():
    """Silence and success look identical otherwise."""
    assert "invariant holds" in explain(_signal(1.0))


def test_a_z_score_is_offered_only_as_a_reference_beside_a_breach():
    signal = assess_metric("m", 0.5, VARIED, invariant=WRAPPER)
    text = explain(signal)
    assert "For reference" in text
    assert text.index("must be at least") < text.index("For reference")


# --- declaration --------------------------------------------------------


def test_bands_must_ascend():
    with pytest.raises(ValidationError, match="must ascend"):
        Invariant(target=1.0, tolerance=0.5, high_at=0.1, critical_at=0.2,
                  rationale="backwards")


def test_a_rationale_is_required():
    """An invariant without a stated reason cannot be reported responsibly."""
    with pytest.raises(ValidationError):
        Invariant(target=1.0)


def test_an_invariant_survives_a_state_round_trip():
    """It crosses the LangGraph boundary, where msgpack encoding turns Pydantic
    models into dicts."""
    restored = Invariant.model_validate(WRAPPER.model_dump(mode="json"))
    assert restored == WRAPPER
    assert restored.bound is Bound.AT_LEAST


# --- integration: declaration to verdict --------------------------------


def test_the_wrapper_metric_declares_its_invariant():
    """The registry is where this must live. An invariant applied at one call
    site and forgotten at another is worse than none, because the path that
    forgets it still reports NORMAL."""
    from src.blockchain.features import get_spec

    spec = get_spec("hyperevm", "wrapper_backing_ratio")
    assert spec.invariant is not None
    assert spec.invariant.target == 1.0
    assert spec.invariant.bound is Bound.AT_LEAST


def test_a_series_with_an_invariant_is_scoreable_without_history():
    from datetime import datetime, timezone

    from src.agents.blockchain import MetricSeries

    now = datetime.now(timezone.utc)
    bare = MetricSeries(metric="m", protocol="p", current=1.0, observed_at=now)
    guarded = MetricSeries(
        metric="m", protocol="p", current=1.0, observed_at=now, invariant=WRAPPER
    )
    assert not bare.scoreable
    assert guarded.scoreable


def test_per_subject_series_no_longer_collide_in_the_risk_engine_contract():
    """Keying the metrics dict on the metric name alone kept only the last
    series: three markets in, one scored, and no error anywhere. A dropped
    invariant check would fail the same silent way once a second contract is
    registered."""
    from datetime import datetime, timezone

    from src.agents.blockchain import BlockchainOutput, MetricSeries
    from src.graph.investigation import blockchain_agent
    from src.intelligence.plan import build_plan

    now = datetime.now(timezone.utc)
    series = tuple(
        MetricSeries(
            metric="mark_price", protocol="hyperliquid", subject=s,
            current=float(i), history=tuple(float(i) for _ in range(9)), observed_at=now,
        )
        for i, s in enumerate(["BTC", "ETH", "HYPE"])
    )

    import src.agents.blockchain as agent_mod

    original = agent_mod.investigate
    agent_mod.investigate = lambda **kw: BlockchainOutput(series=series)
    try:
        plan = build_plan("q", "full_investigation", ["hyperliquid"])
        state = {"investigation_plan": plan.model_dump(mode="json")}
        out = blockchain_agent(state)
    finally:
        agent_mod.investigate = original

    metrics = out["blockchain_results"]["metrics"]
    assert len(metrics) == 3, f"expected one entry per market, got {list(metrics)}"
    assert {m["subject"] for m in metrics.values()} == {"BTC", "ETH", "HYPE"}
    # The scored name stays clean; the composite key never reaches a report.
    assert {m["metric"] for m in metrics.values()} == {"mark_price"}


def test_a_breach_reaches_the_risk_engine_as_a_critical_signal():
    """End to end through the node contract, including the invariant crossing the
    state boundary as a plain dict."""
    from src.graph.investigation import risk_engine
    from src.risk.signals import RiskSignal
    from src.intelligence.plan import build_plan

    plan = build_plan("Is the wrapper solvent?", "full_investigation", ["hyperevm"])
    state = {
        "investigation_plan": plan.model_dump(mode="json"),
        "blockchain_results": {
            "metrics": {
                "wrapper_backing_ratio@WHYPE": {
                    "metric": "wrapper_backing_ratio",
                    "current": 0.8,
                    "history": FLAT,
                    "protocol": "hyperevm",
                    "subject": "WHYPE",
                    "invariant": WRAPPER.model_dump(mode="json"),
                }
            }
        },
    }
    signals = [RiskSignal.model_validate(s) for s in risk_engine(state)["risk_signals"]]
    assert len(signals) == 1
    assert signals[0].severity is Severity.CRITICAL
    assert signals[0].breach is not None


def test_a_signal_with_no_scores_renders_without_an_empty_clause():
    """A degenerate baseline produces no z-scores, so there is nothing to list
    between the reading and the severity."""
    signal = assess_metric("m", 1.0, FLAT)
    text = explain(signal)
    assert ". ." not in text and "  " not in text


# --- the monotonic counter invariant ------------------------------------


def _block_series(current, history=(100, 110, 120, 130, 140, 150, 160, 170, 180)):
    from src.blockchain.features import get_spec, prepare_for_scoring

    spec = get_spec("hyperevm", "latest_block")
    rate, baseline = prepare_for_scoring(spec, list(history), current)
    return assess_metric(spec.scored_as, rate, baseline, protocol="hyperevm",
                         invariant=spec.invariant), rate


def test_a_chain_going_backwards_used_to_read_as_perfectly_normal():
    """The regression, as it was measured.

    `rate_series` drops negative increments to keep a spurious reversal out of a
    baseline. Applied to the CURRENT reading that does not leave a gap — it
    promotes the previous increment into its place, so a nine-reading series
    stepping back thirty blocks reported a rate of 10, identical to a healthy
    chain. The statistical path cannot catch it either: the series has never
    contained a negative, so there is no baseline for one to be unusual against.
    """
    from src.blockchain.features import rate_series

    history = [100, 110, 120, 130, 140, 150, 160, 170, 180]
    # The old computation, preserved to show what it produced.
    old = rate_series([*history, 150])[-1]
    assert old == 10, "the old path reported a healthy rate for a 30-block reorg"

    signal, rate = _block_series(150)
    assert rate == -30
    assert signal.severity is Severity.CRITICAL
    assert signal.anomaly


def test_a_healthy_chain_and_a_stalled_one_are_both_within_the_invariant():
    """Monotonicity says nothing about speed. A halt is the statistical path's
    job — the rate collapses to zero against a baseline of ten — and the
    invariant must not claim it."""
    healthy, rate = _block_series(190)
    assert rate == 10 and healthy.breach is None
    stalled, rate = _block_series(180)
    assert rate == 0 and stalled.breach is None


def test_a_single_block_reversal_is_still_a_finding():
    """One block backwards is a reorg. There is no threshold to argue about:
    under the protocol's rules the value is impossible, not unusual."""
    signal, rate = _block_series(179)
    assert rate == -1
    assert signal.anomaly


def test_the_reversal_is_reported_in_blocks_not_percent():
    """A zero target makes deviation absolute. Rendering it relatively produced
    "-30 blocks, 3000.0000% below target", which is arithmetically true and
    meaningless."""
    signal, _ = _block_series(150)
    text = explain(signal)
    assert "30 below target" in text
    assert "%" not in text.split("Severity")[0]


def test_one_prior_reading_still_yields_a_rate():
    """`MIN_BASELINE_N` decides whether a rate can be SCORED. Refusing to derive
    one at all would also withhold it from the invariant, which needs no
    baseline."""
    from src.blockchain.features import get_spec, prepare_for_scoring

    spec = get_spec("hyperevm", "latest_block")
    assert prepare_for_scoring(spec, [100], 110) == (10, [])


def test_two_readings_are_enough_to_catch_a_reversal():
    """The n=1 argument for invariants, at its limit: no baseline exists, and
    the verdict is still CRITICAL."""
    signal, rate = _block_series(90, history=(100,))
    assert rate == -10
    assert signal.baseline.n == 0
    assert signal.severity is Severity.CRITICAL
