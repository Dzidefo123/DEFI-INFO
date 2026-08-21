from __future__ import annotations

import re
from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.config import settings
from src.graph import prompts
from src.graph.state import AgentState, Intent
from src.guardrails import rules
from src.intelligence.query_types import QueryType, effective_query_type
from src.obs.metrics import timed
from src.protocols import (
    Protocol,
    coverage_phrase,
    enabled_protocols,
    english_list,
    get_protocol,
    is_known,
)
from src.retrieval.retriever import hybrid_search
from src.tools import hyperevm
from src.tools.hyperliquid import MarketDataError, market_snapshot

MAX_ATTEMPTS = 2

# Escalation reasons that mean "the knowledge base could not ground an answer",
# as opposed to an account/funds escalation. They select the refusal copy.
LOW_CONFIDENCE = "low_confidence: no chunk cleared the retrieval floor"
NO_GROUNDED_DOCS = "no_grounded_docs: retrieval found nothing that answers this"
_DOCS_FAILURE_PREFIXES = ("low_confidence", "no_grounded_docs", "ungrounded")


def _is_docs_failure(reason: str) -> bool:
    return reason.startswith(_DOCS_FAILURE_PREFIXES)


# --- structured outputs -------------------------------------------------


class Routing(BaseModel):
    """The router's three axes, decided in ONE model call.

    `query_type` rides along with intent and protocols rather than getting its
    own classifier node, and that is a cost decision made against a measured
    profile: the router is already the cheapest LLM stage while `grade` is 43% of
    a turn's cost, so a second classification call would add a per-turn charge to
    every question — including the CX questions whose answer is "no
    investigation needed". The three axes are also decided from the same reading
    of the same sentence, so splitting them buys no independence, only latency.
    """

    intent: Intent = Field(description="The single best-fitting intent")
    protocols: list[str] = Field(
        default_factory=list,
        description="Whitelisted protocol keys the question concerns; [] if general",
    )
    coin: str | None = Field(None, description="Ticker, set only for live_data")
    query_type: QueryType = Field(
        default=QueryType.CX,
        description="How much investigation the question needs; default cx",
    )
    reason: str = Field(description="One sentence justifying the choice")


class Verdict(BaseModel):
    ok: bool
    reason: str


# --- models -------------------------------------------------------------


@lru_cache(maxsize=None)
def _llm(model_id: str, max_tokens: int = 2048) -> ChatAnthropic:
    return ChatAnthropic(
        model=model_id,
        max_tokens=max_tokens,
        api_key=settings.anthropic_api_key or None,
    )


def _protocol_catalog() -> str:
    """The whitelist, rendered for the router prompt so it can't drift from code."""
    lines = []
    for p in enabled_protocols():
        aka = f" (aka {', '.join(p.aliases)})" if p.aliases else ""
        lines.append(f"- {p.key}: {p.name}{aka} — {p.category}")
    return "\n".join(lines)


def _known_protocols(candidates: list[str]) -> list[str]:
    """Drop anything the model returned that isn't actually whitelisted.

    The prompt already steers the model to real keys, but a hallucinated key
    would silently filter retrieval down to nothing — so sanitize before it
    reaches `hybrid_search`.
    """
    return [p for p in candidates if is_known(p)]


def _format_context(docs: list[Document]) -> tuple[str, list[str]]:
    blocks, citations = [], []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata["source"]
        citations.append(src)
        blocks.append(f"[{i}] {src}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks), citations


def _reply(state: AgentState, answer: str, question: str | None = None) -> dict:
    """Terminal reply: commit this turn's exchange to conversation history.

    `question` overrides which text is logged as the user's turn, and exists for
    `guard`. Everywhere else, `guard` has already written `original_question` for
    the current turn before the node runs, so reading it back is correct — it
    recovers the user's real wording after `rewrite` has replaced `question` with
    a re-phrased query.

    `guard` is the exception because it runs BEFORE its own return is applied.
    Under a checkpointer, `original_question` still holds the PREVIOUS turn's
    value at that moment, so a guardrail hit on turn 2+ would log the previous
    question against this turn's refusal — quietly corrupting the transcript of
    exactly the safety-critical turns anyone would later audit.
    """
    if question is None:
        question = state.get("original_question", state["question"])
    return {
        "answer": answer,
        "citations": [],
        "messages": [HumanMessage(question), AIMessage(answer)],
    }


# --- nodes --------------------------------------------------------------


@timed
def guard(state: AgentState) -> dict:
    """Deterministic pre-router gate. See src/guardrails/rules.py for why.

    Seeds `query_type` to CX alongside `attempts`, so the key is present on every
    path. A guardrail hit terminates before `route` ever runs, and a downstream
    consumer reading the classification off a refused turn should find "no
    investigation" rather than a missing key — the safest reading of a turn the
    router never saw is the cheapest one.
    """
    hit = rules.check(state["question"])
    # Everything here is per-TURN state. State channels are last-write-wins and
    # survive in the checkpoint, so anything not reset at the start of a turn
    # leaks into the next one: without clearing `escalation_reason`, every turn
    # after an escalation still reports itself as escalated in the CLI trace and
    # in any log built from it.
    seed = {
        "original_question": state["question"],
        "attempts": 0,
        "query_type": QueryType.CX.value,
        "escalation_reason": None,
    }
    if hit is None:
        return seed | {"guardrail_rule": None, "guardrail_action": None}
    return seed | {
        "guardrail_rule": hit.rule,
        "guardrail_action": hit.action,
        # Explicit: `original_question` in `state` is still the previous turn's.
        **_reply(state, hit.message, question=state["question"]),
    }


@timed
def route(state: AgentState) -> dict:
    model = _llm(settings.router_model_id).with_structured_output(Routing)
    result: Routing = model.invoke(
        [
            ("system", prompts.ROUTER.format(protocols=_protocol_catalog())),
            ("human", state["question"]),
        ]
    )
    return {
        "intent": result.intent,
        "protocols": _known_protocols(result.protocols),
        "coin": result.coin,
        # Clamped here, once, rather than by every downstream consumer
        # remembering to consult `intent` first. See `effective_query_type`.
        # `.value` because state channels are checkpointed through msgpack; see
        # AgentState.query_type.
        "query_type": effective_query_type(result.intent, result.query_type).value,
    }


def _above_floor(docs: list[Document], threshold: float | None) -> list[Document]:
    """Drop reranked chunks below the confidence floor. `None` disables it.

    A chunk with no `rerank_score` (rerank was off) defaults to the threshold, so
    the floor never silently empties an un-reranked result set.
    """
    if threshold is None:
        return docs
    return [d for d in docs if d.metadata.get("rerank_score", threshold) >= threshold]


@timed
def retrieve(state: AgentState) -> dict:
    # Empty protocol list -> None, i.e. search every protocol (general question).
    protocols = state.get("protocols") or None
    docs = _above_floor(
        hybrid_search(state["question"], protocols=protocols),
        settings.min_rerank_score,
    )
    out: dict = {"docs": docs}
    if not docs:
        # Nothing cleared the floor — mark why, so escalation refuses honestly
        # instead of falling through to the account/funds copy.
        out["escalation_reason"] = LOW_CONFIDENCE
    return out


@timed
def grade(state: AgentState) -> dict:
    """Drop chunks that retrieved well but don't actually answer the question."""
    model = _llm(settings.router_model_id, max_tokens=512).with_structured_output(Verdict)
    kept = []
    for doc in state["docs"]:
        verdict: Verdict = model.invoke(
            [
                ("system", prompts.GRADER),
                ("human", f"Question: {state['question']}\n\nChunk:\n{doc.page_content}"),
            ]
        )
        if verdict.ok:
            kept.append(doc)
    if not kept:
        # Chunks retrieved but none actually answer the question — a grounding
        # failure, not an account matter. Name it so escalation says the truth.
        return {"docs": [], "escalation_reason": NO_GROUNDED_DOCS}
    return {"docs": kept}


@timed
def rewrite(state: AgentState) -> dict:
    """Re-query when grading emptied the context — usually a vocabulary mismatch."""
    model = _llm(settings.router_model_id, max_tokens=256)
    # Name the routed protocols so the rewrite reaches for *their* vocabulary.
    # Without this the model regresses to generic exchange terminology, which is
    # the opposite of what a vocabulary-mismatch retry needs.
    routed = [get_protocol(k).name for k in state.get("protocols") or [] if is_known(k)]
    whose = f"{english_list(routed)}'s own" if routed else "the relevant protocol's"
    reply = model.invoke(
        [
            (
                "system",
                f"Rewrite the user's support question using the terminology "
                f"{whose} documentation would use. Return only the "
                f"rewritten question.",
            ),
            ("human", state["question"]),
        ]
    )
    return {
        "question": reply.content.strip(),
        "attempts": state.get("attempts", 0) + 1,
    }


@timed
def generate(state: AgentState) -> dict:
    context, citations = _format_context(state["docs"])
    model = _llm(settings.model_id)
    reply = model.invoke(
        [
            ("system", prompts.ANSWER.format(context=context)),
            ("human", state["question"]),
        ]
    )
    return {"answer": reply.content, "citations": citations}


@timed
def verify(state: AgentState) -> dict:
    context, _ = _format_context(state["docs"])
    model = _llm(settings.router_model_id, max_tokens=512).with_structured_output(Verdict)
    verdict: Verdict = model.invoke(
        [
            ("system", prompts.VERIFIER),
            ("human", f"Excerpts:\n{context}\n\nAnswer:\n{state['answer']}"),
        ]
    )
    return {
        "grounded": verdict.ok,
        "attempts": state.get("attempts", 0) + 1,
        "escalation_reason": None if verdict.ok else f"ungrounded: {verdict.reason}",
    }


# --- live-data path -----------------------------------------------------
#
# Volatile data never comes from retrieval — it is fetched live, per protocol.
# Each protocol with a `live_tool` maps to a handler here that reads what it
# needs from state and returns a grounded, source-linked answer. New protocols
# add a handler and a registry entry; the graph and router are untouched.

_EVM_ADDRESS = re.compile(r"0x[a-fA-F0-9]{40}\b")


def _hyperliquid_live(state: AgentState) -> str:
    coin = state.get("coin")
    if not coin:
        return "Which market did you mean? For example: ETH, BTC, SOL."
    try:
        snap = market_snapshot(coin)
    except MarketDataError as exc:
        return f"I couldn't reach Hyperliquid market data: {exc}"
    return (
        f"**{snap['coin']}-PERP** (live from the Hyperliquid API)\n\n"
        f"- Mark price: {snap['mark_price']}\n"
        f"- Oracle price: {snap['oracle_price']}\n"
        f"- Funding: {snap['funding_hourly_pct']}%/hr "
        f"({snap['funding_annualized_pct']}% annualized)\n"
        f"- Open interest: {snap['open_interest']}\n"
        f"- 24h volume: ${snap['day_volume_usd']}\n"
        f"- Max leverage: {snap['max_leverage']}x\n\n"
        f"**Source** https://api.hyperliquid.xyz (metaAndAssetCtxs)"
    )


def _explorer_url(path: str) -> str:
    return f"{settings.hyperevm_explorer.rstrip('/')}{path}"


_NO_SEARCH_MSG = (
    "I can't look up a token or contract by name on HyperEVM — I read the chain "
    "directly, and the chain has no name index. If you paste the contract "
    "address (0x…), I can tell you its balance and whether it holds code.\n\n"
    "To be clear about what that means: I'm not saying no contract matches "
    "{name!r}. I'm saying I have no way to search for one."
)


def _hyperevm_live(state: AgentState) -> str:
    question = state.get("original_question", state["question"])
    match = _EVM_ADDRESS.search(question)
    try:
        if match:
            info = hyperevm.address_summary(match.group(0))
            kind = "contract" if info["is_contract"] else "account"
            lines = [f"**{info['address']}** — {kind} (read live from the chain)\n"]
            lines.append(f"- Balance: {info['balance_hype']} HYPE")
            if info["is_contract"]:
                lines.append(f"- Contract code: {info['code_size_bytes']:,} bytes")
                # Stated, not omitted: a reader who saw no verification line
                # would reasonably assume it was checked and came back clean.
                lines.append(
                    "- Source verification: not available — reading the chain "
                    "shows that code exists, not whether its source was published"
                )
            lines.append(f"\n**Source** HyperEVM JSON-RPC · explorer: {_explorer_url('/address/' + info['address'])}")
            return "\n".join(lines)

        if state.get("coin") and not hyperevm.search_supported():
            return _NO_SEARCH_MSG.format(name=state["coin"])

        stats = hyperevm.chain_stats()
        block_time = stats["block_time_seconds"]
        kind = stats["block_kind"] or "unrecognised"
        return (
            "**HyperEVM network** (read live from the chain)\n\n"
            f"- Latest block: {stats['latest_block']:,}\n"
            f"- Gas price: {stats['gas_price_gwei']} gwei\n"
            f"- Block time: {f'{block_time}s' if block_time else 'n/a'}\n"
            f"- That block: {stats['transactions_in_block']} transactions, "
            f"{stats['block_gas_used']:,}/{stats['block_gas_limit']:,} gas "
            f"({kind} block)\n\n"
            f"**Source** HyperEVM JSON-RPC · explorer: {_explorer_url('/')}"
        )
    except hyperevm.ChainReadError as exc:
        return f"I couldn't read HyperEVM chain state: {exc}"


# Registry keyed by Protocol.live_tool.
_LIVE_TOOLS = {
    "hyperliquid": _hyperliquid_live,
    "hyperevm": _hyperevm_live,
}


def _pick_live_protocol(state: AgentState) -> Protocol:
    """The protocol whose live tool should answer.

    Prefer a routed protocol that actually has a tool wired. Failing that, still
    return a routed protocol — so `live_data` can say "no live source for Ethena"
    rather than quietly answering from a different protocol's API.

    That distinction is the whole point. Skipping a tool-less protocol and
    falling through to the default produces the worst failure this system has:
    real, current, correctly-formatted numbers for the wrong protocol, in an
    answer whose wording admits nothing. A user asking about USDe's backing gets
    a Hyperliquid perps quote and no signal that anything was substituted.

    Only a question that named no whitelisted protocol at all ("what's ETH
    funding right now") falls back to the perps venue.
    """
    routed = [get_protocol(k) for k in state.get("protocols") or [] if is_known(k)]
    for protocol in routed:
        if protocol.live_tool in _LIVE_TOOLS:
            return protocol
    if routed:
        return routed[0]
    return get_protocol("hyperliquid")


@timed
def live_data(state: AgentState) -> dict:
    protocol = _pick_live_protocol(state)
    handler = _LIVE_TOOLS.get(protocol.live_tool)
    if handler is None:
        return _reply(
            state, f"I don't have a live-data source wired up for {protocol.name} yet."
        )
    return _reply(state, handler(state))


_ACCOUNT_ESCALATION_MSG = (
    "I'm handing this to a human support agent — this involves your account "
    "or funds, and I don't act on either.\n\n"
    "In the meantime: the protocols I cover are self-custodial, so no one, including "
    "support, can move funds from your wallet or reverse a transaction. "
    "Never share your seed phrase or private key with anyone claiming to be "
    "support."
)

_NO_GROUNDED_ANSWER_MSG = (
    "I couldn't find documentation that confidently answers this, so I won't "
    "guess — wrong information about a protocol can cost you money. I'm passing "
    "this to a human who can help.\n\n"
    "If you can name the specific protocol and feature you mean, I can try again."
)


@timed
def escalate(state: AgentState) -> dict:
    # A guardrail that routed here has ALREADY written rule-specific copy and
    # committed the turn to history. Replying again did two things, both wrong:
    # it replaced the deliberate, separately-tested warning in
    # `guardrails/rules.py` with this generic account message — so the compromise
    # copy never actually reached a user — and it logged the same turn twice,
    # padding the conversation history the next turn is conditioned on.
    #
    # Record why we escalated; leave the guardrail's answer alone.
    if state.get("guardrail_action") == "escalate":
        return {"escalation_reason": f"guardrail: {state.get('guardrail_rule')}"}

    reason = state.get("escalation_reason") or state.get("intent") or "unknown"
    # A retrieval/grounding failure must not be dressed up as an account matter.
    answer = _NO_GROUNDED_ANSWER_MSG if _is_docs_failure(reason) else _ACCOUNT_ESCALATION_MSG
    return _reply(state, answer) | {"escalation_reason": reason}


_REFUSE_TEMPLATE = (
    "I can only help with the protocols I cover ({coverage}) — how they work "
    "mechanically, and their live market and on-chain data. I don't give "
    "trading or investment advice."
)


@timed
def refuse(state: AgentState) -> dict:
    return _reply(state, _REFUSE_TEMPLATE.format(coverage=coverage_phrase()))


@timed
def finalize(state: AgentState) -> dict:
    """Attach citations and commit the turn to conversation history."""
    answer = state["answer"]
    if state.get("citations"):
        # Numbering must match the [n] markers the answer prompt saw, so this
        # enumerates positionally and does not dedupe repeated URLs.
        sources = "\n".join(
            f"[{i}] {url}" for i, url in enumerate(state["citations"], 1)
        )
        answer = f"{answer}\n\n**Sources**\n{sources}"
    return {
        "answer": answer,
        "messages": [
            HumanMessage(state.get("original_question", state["question"])),
            AIMessage(answer),
        ],
    }
