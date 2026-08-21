"""The wrapper bands, scored against real backing failures.

The dataset is small because the citable population is small: backing failures
with published figures are rare, and most "depegs" are price events with the
backing intact. That distinction is the dataset's most useful feature, not a
shortcoming of it.
"""

from eval.calibration import WRAPPER, band_coverage, cases, score
from src.risk.severity import Severity


def test_every_observed_case_is_classified_as_expected():
    rows, correct = score()
    assert correct == len(rows), [r for r in rows if not r["ok"]]


def test_the_real_onset_reading_is_critical():
    """rsETH's adapter retained 0.19% of its reserve after one forged message."""
    onset = next(c for c in cases() if c["case"] == "rseth_adapter_single_block")
    assert WRAPPER.severity_for(onset["ratio"]) is Severity.CRITICAL


def test_the_sustained_aftermath_is_also_critical():
    """26.46% is the value a poller actually observes — the onset lasted one
    block, the aftermath lasted days."""
    after = next(c for c in cases() if c["case"] == "rseth_aggregate_aftermath")
    assert after["stated_directly"]
    assert WRAPPER.severity_for(after["ratio"]) is Severity.CRITICAL


def test_a_price_depeg_with_backing_intact_does_not_fire():
    """The negative control, and the reason the metric reads chain state rather
    than price. WBTC traded at a discount while its reserves were whole."""
    control = next(c for c in cases() if c["phase"] == "negative_control")
    assert WRAPPER.severity_for(control["ratio"]) is Severity.NORMAL


def test_the_graded_bands_are_empty_and_the_code_says_so():
    """A guard on the honesty of the calibration write-up, not on behaviour.

    If a future dataset ever populates ELEVATED or HIGH, the claim in
    `invariants.py` that these boundaries rest on economics rather than
    observations stops being true and the comment must be rewritten.
    """
    coverage = band_coverage()
    assert coverage["elevated"] == []
    assert coverage["high"] == []
    assert len(coverage["critical"]) == 2


def test_every_case_carries_a_source():
    """A calibration figure with no citation is a recollection, and this repo
    keeps `security/registry.jsonl` empty for exactly that reason."""
    for case in cases():
        assert case["source"].startswith("https://")
        assert case["derivation"]
