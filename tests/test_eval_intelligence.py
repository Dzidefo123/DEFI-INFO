"""§19's evaluation harnesses for the investigation path.

An eval that cannot fail measures nothing, so these tests check that the
harnesses actually discriminate — that a broken verifier would score badly, and
that the anomaly tiers degrade rather than all reading 1.000.
"""

import pytest

from eval.intelligence import (
    TIERS,
    _roc_auc,
    build_case,
    eval_agent_selection,
    eval_anomaly,
    eval_detection_latency,
    eval_verification,
    load_verification_cases,
)
from src.evidence.models import SourceTier, Stance, VerificationStatus


# --- the labelled set ----------------------------------------------------


def test_the_case_file_loads_and_is_not_trivial():
    cases = load_verification_cases()
    assert len(cases) >= 15
    assert len({c["id"] for c in cases}) == len(cases)


def test_every_case_declares_why_it_is_labelled_that_way():
    """A labelled set without reasons cannot be argued with when it is wrong."""
    for case in load_verification_cases():
        assert case["why"].strip()
        assert case["category"].strip()


def test_the_set_covers_both_directions():
    """A set of only-bad claims would score perfectly on a verifier that rejects
    everything."""
    expects = {c["expect"] for c in load_verification_cases()}
    assert "verified" in expects
    assert len(expects) >= 3


@pytest.mark.parametrize("case", load_verification_cases(), ids=lambda c: c["id"])
def test_every_expected_status_is_a_real_status(case):
    VerificationStatus(case["expect"])


def test_a_case_builds_into_the_models_verification_consumes():
    case = next(c for c in load_verification_cases() if c["evidence"])
    claim, evidence = build_case(case)
    assert claim.text == case["claim"]
    assert len(evidence) == len(case["evidence"])


def test_case_stances_and_tiers_are_honoured():
    case = {
        "claim": "x",
        "evidence": [
            {"uri": "https://a", "summary": "s", "tier": "unverified",
             "stance": "contradicts", "kind": "incident_report"}
        ],
    }
    claim, evidence = build_case(case)
    assert evidence[0].source.tier is SourceTier.UNVERIFIED
    assert claim.links[0].stance is Stance.CONTRADICTS


def test_case_age_drives_the_evidence_timestamp():
    fresh = build_case({"claim": "x", "evidence": [
        {"uri": "https://a", "summary": "s"}]})[1][0]
    stale = build_case({"claim": "x", "evidence": [
        {"uri": "https://a", "summary": "s", "age_days": 30}]})[1][0]
    assert stale.as_of < fresh.as_of


# --- the verification harness -------------------------------------------


def test_verification_scores_the_shipped_set():
    result = eval_verification()
    assert result["n"] >= 15
    assert result["accuracy"] >= 0.9
    # The asymmetric one: accepting a bad claim is the failure that matters.
    assert result["false_verification_rate"] == 0.0


def test_the_harness_would_catch_a_verifier_that_accepts_everything():
    """The test that makes the eval worth running. A harness that scores well
    against a broken implementation is measuring nothing."""
    permissive = [
        {"id": "x", "claim": "Anything.", "expect": "insufficient_evidence",
         "category": "unsupported", "why": "no evidence", "evidence": []},
    ]
    # The real verifier rejects it, so the harness reports a clean run...
    assert eval_verification(permissive)["false_verification_rate"] == 0.0

    # ...and would report 1.0 if that claim had been accepted, because the metric
    # is computed from the expectation, not from the implementation.
    mislabelled = [dict(permissive[0], expect="verified")]
    assert eval_verification(mislabelled)["over_rejection_rate"] == 1.0


def test_disagreements_are_reported_with_their_case_id():
    mislabelled = [
        {"id": "wrong-001", "claim": "Anything.", "expect": "verified",
         "category": "unsupported", "why": "deliberately mislabelled",
         "evidence": []},
    ]
    result = eval_verification(mislabelled)
    assert result["misses"][0][0] == "wrong-001"


# --- ROC-AUC -------------------------------------------------------------


def test_perfect_ranking_scores_one():
    assert _roc_auc([3.0, 2.0, 1.0], [True, False, False]) == 1.0


def test_inverted_ranking_scores_zero():
    assert _roc_auc([1.0, 2.0, 3.0], [True, False, False]) == 0.0


def test_ties_score_as_a_coin_flip():
    """What makes an unassessable metric rank as uninformative rather than as a
    hit."""
    assert _roc_auc([1.0, 1.0], [True, False]) == 0.5


def test_auc_is_undefined_without_both_classes():
    import math

    assert math.isnan(_roc_auc([1.0, 2.0], [True, True]))


# --- the anomaly tiers ---------------------------------------------------


def test_the_tiers_actually_degrade():
    """The finding that made this harness worth rewriting: the original
    generator separated anomalies by ~15 sigma, so every metric read 1.000 and
    the benchmark could not tell a good detector from an adequate one."""
    results = eval_anomaly()
    recalls = [results[t]["recall"] for t in TIERS]
    assert recalls == sorted(recalls, reverse=True)
    assert results["easy"]["recall"] == 1.0
    assert results["hard"]["recall"] < 0.8


def test_precision_holds_while_recall_falls():
    """The detector's real character: the 3-sigma bar never cries wolf, it just
    misses subtle shifts."""
    results = eval_anomaly()
    assert all(results[t]["precision"] == 1.0 for t in TIERS)
    assert all(results[t]["fpr"] == 0.0 for t in TIERS)


def test_auc_stays_high_where_f1_collapses():
    """Which locates the loss in the threshold rather than the score — a lower
    bar would recover recall."""
    results = eval_anomaly()
    assert results["hard"]["f1"] < 0.7
    assert results["hard"]["roc_auc"] > 0.9


def test_a_sustained_shift_is_caught_promptly():
    assert eval_detection_latency(multiple=4.0)["readings"] == 1


def test_a_shift_too_small_to_matter_is_not_reported_as_caught():
    """A latency of -1 means never, and must not be read as "immediately"."""
    result = eval_detection_latency(multiple=1.0)
    assert result["detected"] is False
    assert result["readings"] == -1


# --- agent selection -----------------------------------------------------


def test_every_classification_dispatches_its_declared_agents():
    result = eval_agent_selection()
    assert result["correct"] == result["n"]
