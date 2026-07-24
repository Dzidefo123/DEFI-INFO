"""Schema invariants for the eval golden set.

A malformed golden case does not fail loudly — it quietly measures nothing. A
case whose `expect_source` matches no indexed page scores as a permanent miss
and drags recall down forever; a case whose `protocols` label disagrees with the
tag on the page it expects makes the protocol filter look broken when it is the
label that is wrong. Both look like model regressions in the report, which is
the most expensive kind of bug to chase.
"""

import json
from pathlib import Path

import pytest

from eval.run_eval import source_matches
from src import protocols

GOLDEN = Path(__file__).parent.parent / "eval" / "golden.jsonl"
CORPUS = Path(__file__).parent.parent / ".chroma" / "corpus.jsonl"

INTENTS = {"docs", "live_data", "account_action", "out_of_scope"}
GUARDRAILS = {"secret_solicitation", "compromise", "impersonation", "tax_legal"}


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    with GOLDEN.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_ids_are_unique(cases):
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))


def test_every_case_has_a_question_and_category(cases):
    for c in cases:
        assert c.get("question"), c["id"]
        assert c.get("category"), c["id"]


def test_every_case_is_either_routed_or_gated(cases):
    """A case with neither an intent nor a guardrail is scored by nothing."""
    for c in cases:
        assert c.get("intent") or c.get("guardrail"), c["id"]
        if c.get("intent"):
            assert c["intent"] in INTENTS, (c["id"], c["intent"])
        if c.get("guardrail"):
            assert c["guardrail"] in GUARDRAILS, (c["id"], c["guardrail"])


def test_every_case_carries_a_protocols_list(cases):
    for c in cases:
        assert isinstance(c.get("protocols"), list), c["id"]


def test_protocol_labels_are_whitelisted_keys(cases):
    """A label the registry does not know can never be matched by the router."""
    for c in cases:
        for key in c["protocols"]:
            assert protocols.is_known(key), (c["id"], key)


def test_out_of_scope_and_guardrail_cases_name_no_protocol(cases):
    """Nothing outside the whitelist gets a protocol label.

    off_protocol cases are about Aave, dYdX and friends — deliberately real, and
    deliberately unlabelled, because the correct behaviour is to refuse rather
    than to resolve them to something we do cover.
    """
    for c in cases:
        if c.get("guardrail") or c.get("intent") == "out_of_scope":
            assert c["protocols"] == [], c["id"]


def test_doc_cases_declare_the_protocol_they_expect(cases):
    """Without a label the retrieval filter is never exercised for that case."""
    for c in cases:
        if c.get("expect_source"):
            assert c["protocols"], c["id"]


def test_off_protocol_cases_do_not_trip_the_guardrails(cases):
    """They must reach the router and be refused there, not gated by regex.

    If a guardrail catches them the routing eval never scores them, and the
    off-whitelist refusal metric silently measures an empty set.
    """
    from src.guardrails import rules

    for c in cases:
        if c["category"] == "off_protocol":
            assert rules.check(c["question"]) is None, (c["id"], c["question"])


def test_coverage_of_each_whitelisted_protocol(cases):
    """Every protocol on the whitelist is actually exercised by the eval.

    Adding a protocol without golden cases means its retrieval quality is
    unmeasured — the overall recall number stays green while the new protocol
    answers badly.
    """
    for proto in protocols.enabled_protocols():
        labelled = [c for c in cases if proto.key in c["protocols"]]
        assert len(labelled) >= 5, (
            f"{proto.key} has only {len(labelled)} golden cases; a protocol with "
            f"no eval coverage regresses invisibly"
        )


@pytest.fixture(scope="module")
def indexed_sources() -> set[tuple[str, str]]:
    return {
        (r["meta"]["source"], r["meta"]["protocol"])
        for r in (json.loads(line) for line in CORPUS.open(encoding="utf-8"))
    }


@pytest.mark.skipif(not CORPUS.exists(), reason="no index built")
def test_expect_source_resolves_to_a_page_with_the_labelled_protocol(
    cases, indexed_sources
):
    for c in cases:
        exp = c.get("expect_source")
        if not exp:
            continue
        tagged = {p for s, p in indexed_sources if source_matches(exp, s)}
        assert tagged, f"{c['id']}: no indexed page matches {exp!r}"
        assert tagged & set(c["protocols"]), (
            f"{c['id']}: {exp!r} is tagged {sorted(tagged)} but the case is "
            f"labelled {c['protocols']}"
        )


@pytest.mark.skipif(not CORPUS.exists(), reason="no index built")
def test_expect_source_is_not_ambiguous_across_protocols(cases, indexed_sources):
    """A case's `expect_source` must not substring-match another protocol's pages.

    Scoring is `expect_source in url`, which is fine while one protocol owns the
    vocabulary and silently wrong the moment two don't. Onboarding Ethena turned
    `liquidat`, `margin`, `stak` and `oracle` — all unambiguous for two years of
    Hyperliquid-only evals — into substrings that also match
    `risks/liquidation-risk`, `risks/margin-collateral-risks`,
    `technical-design/staking-usde` and `technical-design/use-of-oracles`.

    Left alone, a retriever that returned Ethena's liquidation page for a
    Hyperliquid liquidation question would be scored **correct**. That is the
    wrong-protocol failure the whole protocol axis exists to catch, hiding
    inside the metric meant to catch it.
    """
    for c in cases:
        exp = c.get("expect_source")
        if not exp:
            continue
        tagged = {p for s, p in indexed_sources if source_matches(exp, s)}
        stray = tagged - set(c["protocols"])
        assert not stray, (
            f"{c['id']}: expect_source {exp!r} also matches pages from "
            f"{sorted(stray)}, so a wrong-protocol result would score as a hit. "
            f"Make it specific enough to name only {c['protocols']} — a leading "
            f"'=' switches to exact-suffix matching for section landing pages."
        )


@pytest.mark.parametrize(
    "expect,url,want",
    [
        ("trading/fees", "https://x/hyperliquid-docs/trading/fees", True),
        ("trading/fees", "https://x/hyperliquid-docs/trading/fees-2", True),
        # The case "=" exists for: landing page yes, pages beneath it no.
        ("=hyperliquid-docs/onboarding", "https://x/hyperliquid-docs/onboarding", True),
        ("=hyperliquid-docs/onboarding", "https://x/hyperliquid-docs/onboarding/", True),
        ("=hyperliquid-docs/onboarding",
         "https://x/hyperliquid-docs/onboarding/how-to-use-the-hyperevm", False),
    ],
)
def test_source_matches(expect, url, want):
    assert source_matches(expect, url) is want
