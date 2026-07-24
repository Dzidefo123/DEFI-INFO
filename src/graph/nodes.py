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
    intent: Intent = Field(description="The single best-fitting intent")
    protocols: list[str] = Field(
        default_factory=list,
        description="Whitelisted protocol keys the question concerns; [] if general",
    )
    coin: str | None = Field(None, description="Ticker, set only for live_data")
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


def _reply(state: AgentState, answer: str) -> dict:
    return {
        "answer": answer,
        "citations": [],
        "messages": [
            HumanMessage(state.get("original_question", state["question"])),
            AIMessage(answer),
        ],
    }


# --- nodes --------------------------------------------------------------


@timed
def guard(state: AgentState) -> dict:
    """Deterministic pre-router gate. See src/guardrails/rules.py for why."""
    hit = rules.check(state["question"])
    if hit is None:
        return {
            "guardrail_rule": None,
            "guardrail_action": None,
            "original_question": state["question"],
            "attempts": 0,
        }
    return {
        "guardrail_rule": hit.rule,
        "guardrail_action": hit.action,
        "original_question": state["question"],
        "attempts": 0,
        **_reply(state, hit.message),
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


def _hyperevm_live(state: AgentState) -> str:
    question = state.get("original_question", state["question"])
    match = _EVM_ADDRESS.search(question)
    try:
        if match:
            addr = match.group(0)
            info = hyperevm.address_summary(addr)
            lines = [f"**{info['address']}** (live from Hyperscan)\n"]
            if info["is_verified_contract"]:
                lines.append(f"- Verified contract: {info['name'] or 'yes'}")
            lines.append(f"- Balance: {info['balance_hype']} HYPE")
            lines.append(
                f"\n**Source** {_explorer_url('/address/' + info['address'])}"
            )
            return "\n".join(lines)

        if state.get("coin"):
            hits = hyperevm.find_contract(state["coin"])
            if not hits:
                return (
                    f"I couldn't find a token or contract matching "
                    f"{state['coin']!r} on HyperEVM."
                )
            lines = [f"**{state['coin']} on HyperEVM** (live from Hyperscan)\n"]
            for h in hits:
                label = h["name"] or h["symbol"] or h["type"]
                sym = f" ({h['symbol']})" if h["symbol"] and h["symbol"] != label else ""
                vfd = " — verified" if h["verified"] else ""
                lines.append(f"- {label}{sym}: `{h['address']}`{vfd}")
            lines.append(f"\n**Source** {_explorer_url('/')}")
            return "\n".join(lines)

        stats = hyperevm.chain_stats()
        n = lambda v: f"{v:,}" if isinstance(v, int) else "n/a"
        return (
            "**HyperEVM network** (live from Hyperscan)\n\n"
            f"- Latest block: {n(stats['latest_block'])}\n"
            f"- Gas (gwei): {stats['gas_slow']} slow / {stats['gas_average']} avg / "
            f"{stats['gas_fast']} fast\n"
            f"- Total transactions: {n(stats['total_transactions'])} "
            f"({n(stats['transactions_today'])} today)\n"
            f"- Total addresses: {n(stats['total_addresses'])}\n\n"
            f"**Source** {_explorer_url('/stats')}"
        )
    except hyperevm.BlockscoutError as exc:
        return f"I couldn't reach the HyperEVM explorer (Hyperscan): {exc}"


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
