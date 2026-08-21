"""Cost attribution.

A cost report is only useful if being wrong about it is visible. The failure this
file guards is the invisible one: a model the pricing table does not recognise
contributes tokens but no dollars, so the total silently understates spend and
reads exactly like a cheap run.
"""


# --- pricing must not silently zero an unknown model ---------------------


def test_a_dated_model_id_is_priced_by_prefix():
    """The API returns `claude-haiku-4-5-20251001` for some models and a bare
    `claude-opus-4-8` for others. An exact-match table prices the dated ones at
    zero, which is the worst way to be wrong about cost: $0.00 from a key miss
    is indistinguishable from $0.00 from spending nothing."""
    from src.obs.metrics import PRICING, rate_for

    assert rate_for("claude-haiku-4-5-20251001") == PRICING["claude-haiku-4-5"]
    assert rate_for("claude-opus-4-8") == PRICING["claude-opus-4-8"]


def test_an_unknown_model_is_recorded_rather_than_absorbed():
    """Its tokens still count; its cost cannot, and the total therefore
    understates spend. The name is kept so that is visible."""
    from src.obs.metrics import Report, UsageCollector

    report = Report()
    collector = UsageCollector(report)
    collector._record({"input_tokens": 100, "output_tokens": 10}, "some-future-model")
    assert "some-future-model" in report.unpriced
    assert report.total_cost_usd == 0.0


def test_a_priced_model_leaves_unpriced_empty():
    from src.obs.metrics import Report, UsageCollector

    report = Report()
    UsageCollector(report)._record({"input_tokens": 1000, "output_tokens": 100}, "claude-opus-4-8")
    assert report.unpriced == set()
    assert report.total_cost_usd > 0
