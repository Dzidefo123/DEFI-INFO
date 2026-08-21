"""The investigation branch: routing into it, fanning out inside it, and the
guarantees that hold on the way through.

Two classes of test here. The edge functions are pure and cover the topology
exhaustively. The end-to-end runs stub `route` — the one node that needs an API
key — so a full investigation is exercised offline, for free, on every commit.
"""

import os
import tempfile

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from src.graph import investigation, nodes
from src.graph.build import _after_plan, _after_route, build_graph
from src.graph.state import accumulate
from src.intelligence.plan import build_plan
from src.intelligence.query_types import QueryType

INVESTIGATIONS = [q.value for q in QueryType if q is not QueryType.CX]


# --- routing into the branch -------------------------------------------


@pytest.mark.parametrize("query_type", INVESTIGATIONS)
@pytest.mark.parametrize("intent", ["docs", "live_data"])
def test_an_investigation_classification_enters_the_branch(query_type, intent):
    assert _after_route({"intent": intent, "query_type": query_type}) == "plan"


@pytest.mark.parametrize(
    "intent,expected",
    [
        ("docs", "retrieve"),
        ("live_data", "live_data"),
        ("account_action", "escalate"),
        ("out_of_scope", "refuse"),
    ],
)
def test_a_cx_classification_takes_the_existing_path_untouched(intent, expected):
    """§21: a simple user question should remain simple. The CX topology is
    unchanged by everything built on top of it."""
    assert _after_route({"intent": intent, "query_type": "cx"}) == expected


@pytest.mark.parametrize("query_type", INVESTIGATIONS)
@pytest.mark.parametrize("intent", ["account_action", "out_of_scope"])
def test_a_terminal_intent_can_never_reach_the_investigation_branch(intent, query_type):
    """The depth axis must not become a way around the funds escalation.

    This holds because `route` clamps via `effective_query_type` before the edge
    function sees state, so a terminal intent cannot arrive here still carrying
    an investigation classification. Asserted at the edge anyway: this is the
    invariant that protects a user's money, and it should fail loudly if the
    clamp is ever moved or dropped.
    """
    from src.intelligence.query_types import effective_query_type

    clamped = effective_query_type(intent, query_type).value
    assert _after_route({"intent": intent, "query_type": clamped}) != "plan"


# --- fan-out ------------------------------------------------------------


@pytest.mark.parametrize(
    "query_type,expected",
    [
        ("research", ["research_agent"]),
        ("blockchain_analysis", ["blockchain_agent"]),
        ("security_analysis", ["security_agent"]),
        ("risk_assessment", ["blockchain_agent", "security_agent"]),
        (
            "full_investigation",
            ["research_agent", "blockchain_agent", "security_agent"],
        ),
    ],
)
def test_the_plan_dispatches_exactly_the_agents_it_named(query_type, expected):
    plan = build_plan("q", query_type, ["ethena"])
    state = {"investigation_plan": plan.model_dump(mode="json")}
    assert _after_plan(state) == expected


def test_fan_out_returns_a_list_so_branches_run_in_parallel():
    """A string would run one agent; a list is how LangGraph schedules the §2
    diagram's three side-by-side branches."""
    plan = build_plan("q", "full_investigation", ["ethena"])
    result = _after_plan({"investigation_plan": plan.model_dump(mode="json")})
    assert isinstance(result, list) and len(result) == 3


# --- the planner --------------------------------------------------------


def test_the_plan_records_intent_before_execution():
    """'Security was investigated and found nothing' versus 'security was never
    investigated' is unrecoverable after the fact unless intent was recorded."""
    plan = build_plan("q", "full_investigation", ["ethena"])
    assert set(plan.agents) == {"research_agent", "blockchain_agent", "security_agent"}
    assert plan.risk_engine and plan.verification


def test_planning_is_deterministic_and_free():
    a = build_plan("q", "risk_assessment", ["ethena"])
    b = build_plan("q", "risk_assessment", ["ethena"])
    assert a == b


def test_an_unscoped_investigation_is_noted_not_blocked():
    """On-chain and security findings are per-protocol. Without one, the report
    should state the limitation rather than the system silently doing less."""
    plan = build_plan("q", "full_investigation", [])
    assert plan.notes and "No protocol was identified" in plan.notes[0]


def test_a_research_only_investigation_needs_no_protocol():
    """Documentary research works across the whole corpus, so it earns no note."""
    assert build_plan("q", "research", []).notes == ()


# --- the accumulate reducer --------------------------------------------


def test_accumulate_appends():
    assert accumulate([1, 2], [3]) == [1, 2, 3]


def test_accumulate_treats_none_as_an_explicit_reset():
    """Plain `operator.add` would accumulate across TURNS, so one investigation
    would inherit the last one's evidence and report claims it never gathered."""
    assert accumulate([1, 2, 3], None) == []


def test_accumulate_handles_an_empty_start():
    assert accumulate(None, [1]) == [1]


def test_accumulate_of_nothing_is_empty():
    assert accumulate(None, None) == []


# --- agents refuse rather than reassure --------------------------------
#
# All three specialists are real as of C3 — research in C1, blockchain in C2,
# security here — so the stub-behaviour tests that lived in this block have moved
# into each agent's own file. `_not_implemented` remains in investigation.py as
# the shape any future agent starts from, and is exercised below.


def test_the_refusal_helper_contributes_no_claims():
    """The shape every agent starts as. A placeholder returning a reassuring
    claim would be indistinguishable downstream from a real finding, and every
    consumer of `claims` is built to trust that a claim means somebody looked."""
    from src.evidence.models import AgentName

    out = investigation._not_implemented(AgentName.SECURITY, "nothing is wired up")
    assert out["claims"] == []
    assert out["evidence"] == []
    assert out["errors"][0].startswith("security_agent: not implemented")


def test_the_risk_engine_says_it_had_no_data_rather_than_nothing_unusual():
    plan = build_plan("q", "blockchain_analysis", ["ethena"])
    out = investigation.risk_engine(
        {"investigation_plan": plan.model_dump(mode="json"), "blockchain_results": {}}
    )
    assert out["risk_signals"] == []
    assert "no metrics were collected" in out["errors"][0]


def test_the_risk_engine_no_ops_when_the_plan_does_not_call_for_it():
    plan = build_plan("q", "research", [])
    out = investigation.risk_engine({"investigation_plan": plan.model_dump(mode="json")})
    assert out == {"risk_signals": []}


def test_the_risk_engine_scores_metrics_when_they_exist():
    """The engine is real and tested; what it lacks is a feature store."""
    plan = build_plan("q", "blockchain_analysis", ["ethena"])
    out = investigation.risk_engine(
        {
            "investigation_plan": plan.model_dump(mode="json"),
            "blockchain_results": {
                "metrics": {
                    "liquidity_outflow": {
                        "current": 12.5,
                        "history": [2.3, 2.1, 2.5, 2.4, 2.2, 2.0, 2.6, 2.3, 2.4, 2.2],
                        "protocol": "ethena",
                    }
                }
            },
        }
    )
    assert len(out["risk_signals"]) == 1
    assert out["risk_signals"][0]["severity"] in ("high", "critical")


# --- recombining claims with their verdicts ----------------------------


def _bare_claim(text="A claim."):
    from src.evidence.models import AgentName, Claim, EvidenceLink, Stance

    return Claim(
        text=text,
        agent=AgentName.RESEARCH,
        links=(EvidenceLink(evidence_id="ev_a", stance=Stance.SUPPORTS),),
    )


def test_verdicts_are_stamped_onto_their_claims():
    """Verification cannot write back onto the `claims` channel — the reducer
    appends rather than replaces, so the claims would double. The two layers are
    recombined at render time instead. Without this the report saw every claim as
    UNVERIFIED and called a well-evidenced investigation empty."""
    from src.evidence.models import VerificationStatus

    claim = _bare_claim()
    stamped = investigation.apply_verdicts(
        [claim],
        {"verdicts": [{"claim_id": claim.claim_id, "status": "verified"}]},
    )
    assert stamped[0].verification is VerificationStatus.VERIFIED


def test_a_claim_with_no_verdict_keeps_its_original_standing():
    from src.evidence.models import VerificationStatus

    stamped = investigation.apply_verdicts([_bare_claim()], {"verdicts": []})
    assert stamped[0].verification is VerificationStatus.UNVERIFIED


def test_applying_verdicts_does_not_change_how_many_claims_there_are():
    claims = [_bare_claim("One."), _bare_claim("Two.")]
    assert len(investigation.apply_verdicts(claims, {})) == 2


# --- end to end (router stubbed; no API key) ---------------------------


@pytest.fixture
def offline_models(monkeypatch):
    """Neutralise every step that would reach the network or load a model.

    The Research Agent is real from C1 onward, so stubbing `route` alone is no
    longer enough — it would decompose, embed, and synthesise for real. Its
    dependencies resolve at call time precisely so they can be replaced here;
    see `research.investigate`.

    Defaults return nothing retrievable, which exercises the honest-failure path.
    `populate` swaps in real documents for the tests that need findings.
    """
    from src.agents import research

    state = {"docs": []}

    monkeypatch.setattr(
        research,
        "llm_decompose",
        lambda q, p: [research.SubQuery(text=q, angle="documentation")],
    )
    monkeypatch.setattr(
        research,
        "llm_synthesize",
        lambda q, docs: [
            research.ProposedClaim(text=f"Claim from {d.metadata['source']}", excerpts=[i])
            for i, d in enumerate(docs, start=1)
        ],
    )
    monkeypatch.setattr(
        research, "_default_retriever", lambda query, protocols: list(state["docs"])
    )

    # The Blockchain Agent is real from C2, so its collector would reach an
    # explorer. Default: collect nothing, which is also the honest path for a
    # protocol with no on-chain reader.
    from src.agents import blockchain as blockchain_agent_module
    from src.blockchain.collectors import Collection

    monkeypatch.setattr(
        blockchain_agent_module,
        "collect",
        lambda key, subject="": Collection(
            observations=list(state["observations"]),
            errors=list(state["collector_errors"]),
        ),
    )
    monkeypatch.setattr(
        blockchain_agent_module.store, "record", lambda observations, path=None: len(observations)
    )
    monkeypatch.setattr(
        blockchain_agent_module.store,
        "prior_history",
        lambda *a, **kw: list(state["history"]),
    )

    # The Security Agent is real from C3, so its documentation sweep would load
    # the real index. Default: retrieve nothing, registry absent.
    from src.agents import security as security_agent_module

    monkeypatch.setattr(
        security_agent_module,
        "_default_retriever",
        lambda query, protocols: list(state["security_docs"]),
    )
    monkeypatch.setattr(security_agent_module, "for_protocols", lambda p, path=None: ())

    state["observations"] = []
    state["collector_errors"] = ["hyperevm: no readings in this test"]
    state["history"] = []
    state["security_docs"] = []

    def populate(docs=None, observations=None, history=None, collector_errors=None,
                 security_docs=None):
        if docs is not None:
            state["docs"] = docs
        if observations is not None:
            state["observations"] = observations
        if history is not None:
            state["history"] = history
        if collector_errors is not None:
            state["collector_errors"] = collector_errors
        if security_docs is not None:
            state["security_docs"] = security_docs

    return populate


@pytest.fixture
def stub_router(monkeypatch, offline_models):
    """Replace the nodes that cost money. Everything else runs for real."""

    def _stub(intent="docs", protocols=("ethena",), query_type="full_investigation"):
        monkeypatch.setattr(
            nodes,
            "route",
            lambda state: {
                "intent": intent,
                "protocols": list(protocols),
                "coin": None,
                "query_type": query_type,
            },
        )
        return build_graph()

    return _stub


def test_a_full_investigation_runs_every_stage(stub_router):
    graph = stub_router()
    state = graph.invoke(
        {"question": "Is Ethena showing significant risk?"},
        {"configurable": {"thread_id": "e2e-full"}},
    )
    assert state["investigation_plan"]["query_type"] == "full_investigation"
    # Every stage reported for itself. Asserted by who spoke rather than by a
    # count, so adding a limitation to one agent does not fail this test.
    speakers = {e.split(":")[0] for e in state["errors"]}
    assert {"research_agent", "security_agent", "risk_engine"} <= speakers
    assert state["verification"]["claims_examined"] == 0
    assert state["final_report"]["markdown"] == state["answer"]


def test_the_end_to_end_report_refuses_to_reassure(stub_router):
    graph = stub_router()
    state = graph.invoke(
        {"question": "Is Ethena safe?"}, {"configurable": {"thread_id": "e2e-refuse"}}
    )
    assert "not a clean bill of health" in state["answer"]


def test_a_narrower_classification_runs_only_its_agent(stub_router):
    graph = stub_router(query_type="security_analysis")
    state = graph.invoke(
        {"question": "Has Ethena had incidents?"},
        {"configurable": {"thread_id": "e2e-narrow"}},
    )
    assert state["security_results"]["status"] == "ok"
    assert state["research_results"] == {}
    assert state["blockchain_results"] == {}


def test_the_research_agent_contributes_real_claims_end_to_end(
    stub_router, offline_models
):
    """C1's payoff: the first stub replaced by a working agent. Retrieval, evidence
    extraction, citation checking and verification all run for real — only the two
    model calls are stubbed."""
    from langchain_core.documents import Document

    offline_models(
        [
            Document(
                page_content="USDe maintains its peg through arbitrage.",
                metadata={
                    "source": "https://docs.ethena.fi/peg-arbitrage-mechanism",
                    "protocol": "ethena",
                    "doc_id": "ethena:peg#0",
                    "title": "Ethena",
                    "heading": "Peg Arbitrage Mechanism",
                    "rerank_score": 7.5,
                },
            )
        ]
    )
    graph = stub_router(query_type="research")
    state = graph.invoke(
        {"question": "How does USDe stay pegged?"},
        {"configurable": {"thread_id": "e2e-research"}},
    )

    assert state["research_results"]["status"] == "ok"
    assert state["research_results"]["claim_count"] == 1
    assert len(state["claims"]) == 1 and len(state["evidence"]) == 1

    # The claim reached verification and was scored against real evidence.
    assert state["verification"]["claims_examined"] == 1
    assert state["verification"]["verdicts"][0]["status"] == "verified"
    assert state["verification"]["unsupported"] == 0

    # And it reached the report, with its source.
    assert "Ethena > Peg Arbitrage Mechanism" in state["answer"]
    assert "not a clean bill of health" not in state["answer"]


def test_an_on_chain_anomaly_becomes_a_claim_and_reaches_the_report(
    stub_router, offline_models
):
    """C2's payoff, end to end: a collected reading, judged against stored
    history, becomes an anomaly, becomes a claim linked to the reading it was
    computed from, and lands in the report — with no model involved anywhere in
    the chain."""
    from datetime import datetime, timedelta, timezone

    from src.blockchain.store import Observation

    t0 = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
    quiet = [2.3, 2.1, 2.5, 2.4, 2.2, 2.0, 2.6, 2.3, 2.4, 2.2]

    offline_models(
        observations=[
            Observation(
                protocol="hyperevm",
                metric="gas_price",
                value=12.5,
                observed_at=t0 + timedelta(hours=len(quiet)),
                collected_at=t0 + timedelta(hours=len(quiet)),
            )
        ],
        history=quiet,
        collector_errors=[],
    )

    graph = stub_router(query_type="blockchain_analysis", protocols=("hyperevm",))
    state = graph.invoke(
        {"question": "Has HyperEVM shown unusual activity?"},
        {"configurable": {"thread_id": "e2e-onchain"}},
    )

    # The reading was scored against its own prior history.
    assert state["blockchain_results"]["scoreable"] == 1
    signal = state["risk_signals"][0]
    assert signal["metric"] == "gas_price"
    assert signal["severity"] == "critical"

    # The anomaly produced a claim, linked to the evidence it was computed from.
    assert len(state["claims"]) == 1
    claim = state["claims"][0]
    assert claim["agent"] == "risk_engine"
    assert claim["links"][0]["evidence_id"] == state["evidence"][0]["evidence_id"]

    # And it survived verification into the report.
    assert state["verification"]["verdicts"][0]["status"] == "verified"
    assert "gas_price" in state["answer"]
    assert "Critical anomaly" in state["answer"]
    assert "not a clean bill of health" not in state["answer"]


def test_a_fresh_feature_store_reports_no_baseline_rather_than_no_anomalies(
    stub_router, offline_models
):
    """The state the system is actually in today. A newly-created store has no
    history, so nothing can be scored — and saying 'no anomalies' would be the
    most dangerous possible summary of that."""
    from datetime import datetime, timezone

    from src.blockchain.store import Observation

    offline_models(
        observations=[
            Observation(
                protocol="hyperevm",
                metric="gas_price",
                value=1.2,
                observed_at=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
                collected_at=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
            )
        ],
        history=[],
        collector_errors=[],
    )

    graph = stub_router(query_type="blockchain_analysis", protocols=("hyperevm",))
    state = graph.invoke(
        {"question": "Has HyperEVM shown unusual activity?"},
        {"configurable": {"thread_id": "e2e-nohistory"}},
    )

    assert state["blockchain_results"]["scoreable"] == 0
    assert state["claims"] == []
    assert any("not enough history" in e for e in state["errors"])
    assert "not a clean bill of health" in state["answer"]
    # The reading itself is still on the record, and shown as unassessed.
    assert state["risk_signals"][0]["severity"] == "unknown"
    assert "Not assessed" in state["answer"]


def test_an_ordinary_reading_produces_no_claim(stub_router, offline_models):
    """The detector must stay quiet when nothing happened."""
    from datetime import datetime, timedelta, timezone

    from src.blockchain.store import Observation

    t0 = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
    offline_models(
        observations=[
            Observation(
                protocol="hyperevm", metric="gas_price", value=2.35,
                observed_at=t0 + timedelta(hours=10), collected_at=t0 + timedelta(hours=10),
            )
        ],
        history=[2.3, 2.1, 2.5, 2.4, 2.2, 2.0, 2.6, 2.3, 2.4, 2.2],
        collector_errors=[],
    )

    graph = stub_router(query_type="blockchain_analysis", protocols=("hyperevm",))
    state = graph.invoke(
        {"question": "Anything unusual?"}, {"configurable": {"thread_id": "e2e-quiet"}}
    )
    assert state["risk_signals"][0]["severity"] == "normal"
    assert state["claims"] == []


def test_a_research_agent_that_retrieves_nothing_says_so_in_the_report(stub_router):
    """The default `offline_models` state: retrieval finds nothing. The report
    must read as a failed search, not as an answer."""
    graph = stub_router(query_type="research")
    state = graph.invoke(
        {"question": "How does USDe stay pegged?"},
        {"configurable": {"thread_id": "e2e-research-empty"}},
    )
    assert state["research_results"]["claim_count"] == 0
    assert "not that the answer is no" in state["answer"]


def test_an_investigation_does_not_inherit_the_previous_one(stub_router):
    """The reducer's reset, end to end. Without it these channels accumulate
    across turns and the second report claims the first one's evidence."""
    graph = stub_router()
    config = {"configurable": {"thread_id": "e2e-reset"}}
    first = graph.invoke({"question": "Is Ethena safe?"}, config)
    second = graph.invoke({"question": "Is Ethena safe today?"}, config)
    assert first["errors"] and len(first["errors"]) == len(second["errors"])
    assert len(second["claims"]) == len(first["claims"])
    assert len(second["evidence"]) == len(first["evidence"])


def test_investigation_state_survives_persistence(stub_router, monkeypatch):
    """Everything the branch writes must checkpoint as plain types."""
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    monkeypatch.setattr(
        nodes,
        "route",
        lambda state: {
            "intent": "docs",
            "protocols": ["ethena"],
            "coin": None,
            "query_type": "full_investigation",
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        with SqliteSaver.from_conn_string(os.path.join(tmp, "ck.sqlite")) as saver:
            graph = build_graph(saver)
            config = {"configurable": {"thread_id": "persist-inv"}}
            graph.invoke({"question": "Is Ethena safe?"}, config)
            second = graph.invoke({"question": "And now?"}, config)
            values = dict(graph.get_state(config).values)

    assert second["answer"].startswith("# Intelligence Assessment")
    plain = (str, int, float, bool, type(None), list, dict)
    offenders = {
        k: type(v).__name__
        for k, v in values.items()
        if k != "messages" and type(v) not in plain
    }
    assert not offenders


def test_a_cx_question_never_touches_the_investigation_channels(stub_router):
    """The CX path must stay exactly as cheap as it was."""
    graph = stub_router(query_type="cx", intent="out_of_scope")
    state = graph.invoke(
        {"question": "Should I long ETH?"}, {"configurable": {"thread_id": "e2e-cx"}}
    )
    assert "investigation_plan" not in state
    assert not state.get("errors")
    assert "Intelligence Assessment" not in state["answer"]
