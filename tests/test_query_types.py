"""Query classification: the depth axis, and the safety axis that outranks it.

Pure functions and a table — no model call, so the deterministic half of §5.1 is
fully covered offline. What a live router actually *chooses* is a separate,
paid measurement (`--routing`); these tests cover what the system does with the
choice once made.
"""

import pytest

from src.evidence.models import AgentName
from src.graph.nodes import Routing
from src.intelligence.query_types import (
    REQUIREMENTS,
    QueryType,
    Requirements,
    effective_query_type,
    is_investigation,
    required_agents,
    requirements_for,
)

INVESTIGATIONS = [q for q in QueryType if q is not QueryType.CX]
TERMINAL_INTENTS = ["account_action", "out_of_scope"]
WORKING_INTENTS = ["docs", "live_data"]


# --- the table is total --------------------------------------------------


def test_every_query_type_has_requirements():
    """A classification with no entry would KeyError at request time, on the
    branch that matters most."""
    assert set(REQUIREMENTS) == set(QueryType)


def test_cx_runs_no_specialist_agents():
    """§21: a simple user question should remain simple. CX is the existing
    pipeline, untouched."""
    reqs = requirements_for(QueryType.CX)
    assert reqs.agents == ()
    assert not reqs.is_investigation
    assert not reqs.risk_engine
    assert not reqs.verification


@pytest.mark.parametrize("query_type", INVESTIGATIONS)
def test_every_investigation_runs_at_least_one_agent(query_type):
    assert requirements_for(query_type).agents
    assert is_investigation(query_type)


@pytest.mark.parametrize("query_type", INVESTIGATIONS)
def test_every_investigation_is_verified(query_type):
    """§13: verification is what keeps unsupported conclusions away from a user,
    so no investigation may skip it."""
    assert requirements_for(query_type).verification


def test_cx_is_not_verified_by_this_stage():
    """Not because CX answers need less checking — the RAG loop has its own
    `verify` node. That one grounding-checks a generated answer; this one
    adjudicates claims against evidence. Different jobs."""
    assert not requirements_for(QueryType.CX).verification


# --- which pipelines the §5.1 diagrams imply ----------------------------


def test_blockchain_analysis_runs_the_risk_engine():
    """§5.1's diagram: Blockchain Agent -> On-chain Data -> Statistical Analysis
    -> Verification. Asking whether behaviour is unusual is a question about a
    baseline, so it cannot be answered without one."""
    reqs = requirements_for(QueryType.BLOCKCHAIN_ANALYSIS)
    assert reqs.agents == (AgentName.BLOCKCHAIN,)
    assert reqs.risk_engine


def test_full_investigation_runs_all_three_specialists():
    """§5.1: Research + Blockchain + Security -> Risk Engine -> Verification."""
    reqs = requirements_for(QueryType.FULL_INVESTIGATION)
    assert set(reqs.agents) == {
        AgentName.RESEARCH,
        AgentName.BLOCKCHAIN,
        AgentName.SECURITY,
    }
    assert reqs.risk_engine and reqs.verification


def test_risk_assessment_spans_behaviour_and_posture():
    reqs = requirements_for(QueryType.RISK_ASSESSMENT)
    assert set(reqs.agents) == {AgentName.BLOCKCHAIN, AgentName.SECURITY}
    assert reqs.risk_engine


@pytest.mark.parametrize(
    "query_type", [QueryType.RESEARCH, QueryType.SECURITY_ANALYSIS]
)
def test_document_only_investigations_do_not_run_the_risk_engine(query_type):
    """The engine scores a numeric series against its own history. Neither of
    these produces one, and handing it a document count would manufacture a
    z-score out of something that never varied for a measurable reason."""
    assert not requirements_for(query_type).risk_engine


def test_research_is_the_only_agent_a_research_query_needs():
    assert requirements_for(QueryType.RESEARCH).agents == (AgentName.RESEARCH,)


# --- the safety clamp ---------------------------------------------------


@pytest.mark.parametrize("intent", TERMINAL_INTENTS)
@pytest.mark.parametrize("query_type", list(QueryType))
def test_terminal_intents_collapse_every_depth_to_cx(intent, query_type):
    assert effective_query_type(intent, query_type) is QueryType.CX


@pytest.mark.parametrize("intent", TERMINAL_INTENTS)
@pytest.mark.parametrize("query_type", list(QueryType))
def test_no_investigation_ever_runs_on_a_terminal_intent(intent, query_type):
    """The invariant this module exists for. 'Investigate whether my drained
    wallet was an exploit' reads as a full investigation and is an account
    action, and the account action must win — otherwise the depth axis becomes a
    way to talk past the escalation that protects a user's funds."""
    assert required_agents(intent, query_type) == ()


@pytest.mark.parametrize("intent", WORKING_INTENTS)
@pytest.mark.parametrize("query_type", list(QueryType))
def test_working_intents_preserve_the_classification(intent, query_type):
    assert effective_query_type(intent, query_type) is query_type


def test_the_clamp_is_code_not_a_prompt_instruction():
    """Stated as a test because the distinction is the whole defence: a prompt
    instruction is precisely what a crafted question tries to argue around, and
    this one cannot be argued with."""
    assert (
        effective_query_type("account_action", QueryType.FULL_INVESTIGATION)
        is QueryType.CX
    )


def test_an_unknown_intent_does_not_silently_clamp():
    """Only the two named terminal intents collapse depth. A new intent must be
    considered deliberately rather than inheriting either behaviour by accident
    — so it keeps its classification and shows up in testing, rather than
    silently disabling investigations."""
    assert (
        effective_query_type("some_future_intent", QueryType.RESEARCH)
        is QueryType.RESEARCH
    )


# --- the router contract ------------------------------------------------


def test_routing_defaults_to_cx_when_the_model_omits_the_axis():
    """An older or degraded model that returns only the original two axes must
    fall back to the cheap path, not to an investigation."""
    routing = Routing(intent="docs", reason="documentation question")
    assert routing.query_type is QueryType.CX


def test_an_unknown_query_type_degrades_to_the_cheap_path():
    """Reversed deliberately, after the first paid routing run crashed on it.

    This test used to assert that an unrecognised classification raised, on the
    reasoning that it should fail loudly at the boundary rather than KeyError in
    the planner later. The instinct was right and the choice was posed as binary:
    fail here, or fail there. Degrading here and recording it is the third
    option, and it satisfies the original concern — the planner still never sees
    a value outside the enum.

    What the run measured: asking one call to decide three axes invites the model
    to answer one of them in another's vocabulary. It returned
    `query_type='docs'`, an intent value, and the ValidationError propagated out
    of `route` and killed the turn. `_after_route` already fixes the rule for a
    MISSING depth label — the cheap path, never a crashed turn, because failing
    closed takes down a working support agent over a label. An invalid label is
    that situation with more evidence.
    """
    routing = Routing(intent="docs", query_type="deep_forensics", reason="x")
    assert routing.query_type is QueryType.CX
    assert routing.query_type_coerced == "deep_forensics"


def test_the_coercion_is_recorded_rather_than_silent():
    """Silence is never safety. A router answering one axis in another's
    vocabulary is a prompt or model defect, and it becomes invisible the moment
    the bad value is replaced by a working default."""
    assert Routing(intent="docs", reason="x").query_type_coerced is None
    assert Routing(
        intent="docs", query_type=QueryType.RESEARCH, reason="x"
    ).query_type_coerced is None


def test_intent_is_not_coerced():
    """The depth axis has a safe default; intent does not. It decides whether an
    account action escalates, so substituting a default would invent a routing
    decision the model never made."""
    with pytest.raises(Exception):
        Routing(intent="not_an_intent", reason="x")


def test_routing_carries_all_three_axes_from_one_call():
    """The cost decision: query_type rides with intent and protocols rather than
    getting its own classifier node. `grade` is already 43% of a turn's cost;
    a second classification call would charge every question — including the CX
    ones whose answer is 'no investigation needed'."""
    routing = Routing(
        intent="docs",
        protocols=["ethena"],
        query_type=QueryType.SECURITY_ANALYSIS,
        reason="asks about past incidents",
    )
    assert routing.intent == "docs"
    assert routing.protocols == ["ethena"]
    assert routing.query_type is QueryType.SECURITY_ANALYSIS


def test_requirements_are_frozen():
    """The table is policy, not scratch space."""
    with pytest.raises(Exception):
        requirements_for(QueryType.CX).risk_engine = True


def test_requirements_default_to_doing_nothing():
    """A Requirements() built with no arguments must be the CX shape, so a new
    query type added without a table entry fails toward cheap-and-safe."""
    assert Requirements() == REQUIREMENTS[QueryType.CX]
