"""Offline eval for the multi-protocol crypto agent.

    python -m eval.run_eval --guardrails   # no API key
    python -m eval.run_eval --retrieval    # no API key (includes reranker ablation)
    python -m eval.run_eval --routing      # needs ANTHROPIC_API_KEY
    python -m eval.run_eval --answers      # needs ANTHROPIC_API_KEY (slow, costs money)
    python -m eval.run_eval --offline      # guardrails + retrieval
    python -m eval.run_eval --all

Metrics are deliberately separated by failure mode rather than rolled into one
"accuracy" number, because the fixes are unrelated:

- guardrails: a deterministic gate. Recall on adversarial cases must be 1.00;
  a miss here is a drained wallet. False positives on benign traffic are the
  cost, and are measured, not assumed away.
- retrieval (recall@k, MRR): the ceiling on answer quality. The generator
  cannot cite what retrieval never returned, so a regression stays invisible in
  end-to-end scoring until wrong answers already ship.
- routing: which branch a question takes. Errors are asymmetric — see below.
- answers (faithfulness, quality): only meaningful once the three above hold.

Since the agent went multi-protocol, each of those is scored on a second axis:
*which protocol* a question was answered from. It is entirely possible to route
correctly, retrieve the right rank, and still be wrong — a Hyperliquid answer to
a HyperEVM question cites real documentation for the wrong chain, which is more
convincing and therefore worse than no answer. Golden cases carry a `protocols`
list; leakage across it is reported separately from recall.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

GOLDEN = Path(__file__).parent / "golden.jsonl"
console = Console()


def load_cases() -> list[dict]:
    with GOLDEN.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def source_matches(expect: str, url: str) -> bool:
    """Does `url` satisfy a case's `expect_source`?

    Substring by default, which is convenient and almost always unambiguous.
    A leading "=" means exact-suffix instead, and it exists because substring
    cannot express "the section landing page, not the pages under it":
    `.../onboarding` is a prefix of `.../onboarding/how-to-use-the-hyperevm`,
    which belongs to a different protocol. Without "=", the only ways to write
    that case are to accept a cross-protocol false hit or to rewrite the
    question — both of which corrupt the measurement to suit the matcher.
    """
    if expect.startswith("="):
        return url.rstrip("/").endswith(expect[1:])
    return expect in url


# --- guardrails (no API key) --------------------------------------------


def eval_guardrails(cases: list[dict]) -> dict:
    from src.guardrails import rules

    adversarial = [c for c in cases if c.get("guardrail")]
    benign = [c for c in cases if not c.get("guardrail")]

    hits = misses = 0
    wrong_rule = []
    for case in adversarial:
        hit = rules.check(case["question"])
        if hit is None:
            misses += 1
            console.print(f"  [red]MISS[/red] {case['id']}: {case['question'][:60]}")
        else:
            hits += 1
            if hit.rule != case["guardrail"]:
                wrong_rule.append((case["id"], case["guardrail"], hit.rule))

    false_pos = []
    for case in benign:
        hit = rules.check(case["question"])
        if hit is not None:
            false_pos.append((case["id"], case["question"], hit.rule))

    recall = hits / len(adversarial) if adversarial else 0.0
    fpr = len(false_pos) / len(benign) if benign else 0.0

    table = Table("metric", "value", title="guardrails")
    table.add_row("adversarial recall", f"{hits}/{len(adversarial)} = {recall:.0%}")
    table.add_row("rule mislabeled", str(len(wrong_rule)))
    table.add_row("false positives on benign", f"{len(false_pos)}/{len(benign)} = {fpr:.1%}")
    console.print(table)

    if wrong_rule:
        console.print("[yellow]caught, but by a different rule than labeled:[/yellow]")
        for cid, want, got in wrong_rule:
            console.print(f"  {cid}: expected {want}, got {got}")

    if false_pos:
        console.print("[yellow]false positives (benign traffic gated):[/yellow]")
        for cid, q, rule in false_pos:
            console.print(f"  {cid} [{rule}]: {q[:60]}")

    return {"recall": recall, "fpr": fpr, "misses": misses}


# --- retrieval (no API key) ---------------------------------------------


def _score_retrieval(
    cases: list[dict], k: int, use_rerank: bool, filtered: bool = False
) -> dict:
    from src.retrieval.retriever import hybrid_search

    hits = 0
    rr_total = 0.0
    per_category: dict[str, list[int]] = defaultdict(list)
    per_protocol: dict[str, list[int]] = defaultdict(list)
    misses = []
    leaked = []  # returned a chunk from a protocol the question was not about
    off_protocol_chunks = 0
    scored_chunks = 0  # denominator: every chunk returned for a labelled case

    for case in cases:
        want = case.get("protocols") or []
        # `filtered` is the production path: the router hands `protocols` to
        # hybrid_search. Unfiltered is the ablation — it shows how much of the
        # protocol separation comes from the filter rather than from the
        # embeddings happening to separate the two doc sets anyway.
        docs = hybrid_search(
            case["question"],
            k=k,
            use_rerank=use_rerank,
            protocols=(want or None) if filtered else None,
        )
        sources = [d.metadata["source"] for d in docs]
        got_protocols = [d.metadata.get("protocol") for d in docs]

        rank = next(
            (i for i, s in enumerate(sources, 1)
             if source_matches(case["expect_source"], s)),
            None,
        )
        hit = rank is not None
        hits += hit
        rr_total += 1.0 / rank if rank else 0.0
        per_category[case["category"]].append(int(hit))
        for p in want:
            per_protocol[p].append(int(hit))
        if not hit:
            misses.append((case["id"], case["question"], sources[0] if sources else "-"))

        if want:
            scored_chunks += len(got_protocols)
            off_protocol = [p for p in got_protocols if p not in want]
            off_protocol_chunks += len(off_protocol)
            if off_protocol:
                leaked.append((case["id"], len(off_protocol), len(got_protocols)))

    n = len(cases)
    return {
        "recall": hits / n,
        "mrr": rr_total / n,
        "per_category": {c: sum(v) / len(v) for c, v in per_category.items()},
        "per_protocol": {p: sum(v) / len(v) for p, v in per_protocol.items()},
        "leaked_cases": len(leaked),
        # Share of *all* returned chunks that came from a protocol the question
        # was not about — not the share within the cases that already leaked,
        # which could never look small.
        "leak_rate": off_protocol_chunks / (scored_chunks or 1),
        "misses": misses,
    }


def eval_retrieval(cases: list[dict], k: int = 5) -> dict:
    cases = [c for c in cases if c.get("expect_source")]
    console.print(f"scoring {len(cases)} doc questions (k={k}), with and without rerank...\n")

    off = _score_retrieval(cases, k, use_rerank=False, filtered=True)
    on = _score_retrieval(cases, k, use_rerank=True, filtered=True)

    table = Table("metric", "hybrid only", "+ cross-encoder", "delta",
                  title=f"retrieval @ k={k} (protocol-filtered)")
    for label, key in (("recall@k", "recall"), ("MRR@k", "mrr")):
        d = on[key] - off[key]
        colour = "green" if d > 0 else ("red" if d < 0 else "white")
        table.add_row(label, f"{off[key]:.3f}", f"{on[key]:.3f}",
                      f"[{colour}]{d:+.3f}[/{colour}]")
    console.print(table)

    cat = Table("category", "hybrid only", "+ cross-encoder", title="recall by category")
    for name in sorted(on["per_category"]):
        cat.add_row(name, f"{off['per_category'][name]:.2f}", f"{on['per_category'][name]:.2f}")
    console.print(cat)

    # Per-protocol recall is the number that catches a whitelist that grew
    # without its docs being crawled: overall recall barely moves when the new
    # protocol is 3% of the golden set, but its own row sits near zero.
    proto = Table("protocol", "recall@k", "n", title="recall by protocol (reranked)")
    for name in sorted(on["per_protocol"]):
        n_cases = sum(1 for c in cases if name in (c.get("protocols") or []))
        proto.add_row(name, f"{on['per_protocol'][name]:.2f}", str(n_cases))
    console.print(proto)

    # Cross-protocol leakage: with the filter on this must be zero. A non-zero
    # value means the metadata filter is not actually being applied, and answers
    # can cite the wrong chain's docs.
    #
    # Recall is reported alongside it because the filter is not free. Restricting
    # the corpus changes BM25's own statistics — IDF and average document length
    # are computed over whatever set is indexed — so a filtered query is not the
    # unfiltered query with rows removed, and rankings genuinely move. This table
    # is what makes that trade visible instead of assumed.
    unfiltered = _score_retrieval(cases, k, use_rerank=True, filtered=False)
    leak = Table("metric", "filtered", "unfiltered", "delta",
                 title="protocol filter: leakage vs recall")
    leak.add_row(
        "cases with an off-protocol chunk",
        f"[{'red' if on['leaked_cases'] else 'green'}]{on['leaked_cases']}[/]",
        str(unfiltered["leaked_cases"]),
        "",
    )
    leak.add_row(
        "off-protocol chunk rate",
        f"{on['leak_rate']:.1%}",
        f"{unfiltered['leak_rate']:.1%}",
        "",
    )
    for label, key in (("recall@k", "recall"), ("MRR@k", "mrr")):
        d = on[key] - unfiltered[key]
        colour = "green" if d > 0 else ("red" if d < 0 else "white")
        leak.add_row(label, f"{on[key]:.3f}", f"{unfiltered[key]:.3f}",
                     f"[{colour}]{d:+.3f}[/{colour}]")
    console.print(leak)

    if on["misses"]:
        console.print("[yellow]misses (reranked):[/yellow]")
        for cid, q, top in on["misses"]:
            console.print(f"  {cid}: {q[:52]:<52} -> {top.rsplit('/', 1)[-1]}")

    return {"off": off, "on": on, "unfiltered": unfiltered}


# --- routing (needs API key) --------------------------------------------


def _classify_protocol(want: set[str], got: set[str], is_known) -> str:
    """Cost class of a protocol-set decision — not just right/wrong.

    The exact-match rate lumps together two errors with opposite cost, and the
    distinction is the whole point of the protocol axis:

    - `exact`        want == got.
    - `hallucinated` got contains a key not on the whitelist → the retrieval
                     filter matches nothing and the agent says "no docs" for a
                     protocol it advertises. Worst case.
    - `wrong`        got is non-empty and shares nothing with want → the filter
                     actively excludes the correct protocol's docs. Harmful.
    - `partial`      got overlaps want but isn't equal (e.g. a cross-protocol
                     question got one of its two). Usually still retrieves.
    - `declined`     got is empty, want isn't → no filter, so retrieval searches
                     all protocols. Permissive: it can't exclude the right
                     answer, and when only one protocol documents the concept it
                     lands anyway. The cheap miss.
    """
    if any(not is_known(p) for p in got):
        return "hallucinated"
    if want == got:
        return "exact"
    if not got:
        return "declined"
    if got & want:
        return "partial"
    return "wrong"


def eval_routing(cases: list[dict], dump_path: str | None = None) -> dict:
    import os

    from src.graph.nodes import route
    from src.protocols import is_known

    dump_path = dump_path or os.environ.get("ROUTING_DUMP")
    records = []

    # Guardrail cases never reach the router, so they are not scored here.
    cases = [c for c in cases if c.get("intent") and not c.get("guardrail")]

    correct = 0
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    proto_class: dict[str, int] = defaultdict(int)
    proto_wrong = []      # the harmful ones: wrong protocol or hallucination
    hallucinated = []
    off_leaked = []
    for case in cases:
        out = route({"question": case["question"]})
        got = out["intent"]
        confusion[(case["intent"], got)] += 1
        correct += got == case["intent"]
        if case.get("category") == "off_protocol" and got != "out_of_scope":
            off_leaked.append((case["id"], got))

        # Protocol axis, compared as sets (order is meaningless), classified by
        # cost rather than scored pass/fail — see _classify_protocol.
        want_p = set(case.get("protocols") or [])
        got_p = set(out.get("protocols") or [])
        cls = _classify_protocol(want_p, got_p, is_known)
        proto_class[cls] += 1
        if cls == "hallucinated":
            hallucinated.append((case["id"], sorted(got_p)))
        if cls in ("wrong", "hallucinated"):
            proto_wrong.append((case["id"], sorted(want_p), sorted(got_p), cls, case["question"]))

        records.append({
            "id": case["id"], "question": case["question"],
            "want_intent": case["intent"], "got_intent": got,
            "want_protocols": sorted(want_p), "got_protocols": sorted(got_p),
            "protocol_class": cls, "category": case.get("category"),
        })

    if dump_path:
        with open(dump_path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        console.print(f"[dim]per-case decisions -> {dump_path}[/dim]")

    proto_exact = proto_class["exact"]
    proto_scored = len(cases)

    labels = ["docs", "live_data", "account_action", "out_of_scope"]
    table = Table("expected \\ got", *labels, title="routing confusion")
    for want in labels:
        table.add_row(want, *[
            (f"[green]{confusion[(want, got)]}[/green]" if want == got
             else (f"[red]{confusion[(want, got)]}[/red]" if confusion[(want, got)] else "·"))
            for got in labels
        ])
    console.print(table)

    acc = correct / len(cases)
    console.print(f"[bold]routing accuracy: {correct}/{len(cases)} = {acc:.1%}[/bold]")

    # The asymmetry that matters: an account_action leaking into docs means the
    # agent improvises about someone's funds instead of fetching a human.
    leaked = confusion[("account_action", "docs")] + confusion[("account_action", "live_data")]
    console.print(
        f"[{'red' if leaked else 'green'}]account_action leaked to a non-escalating "
        f"branch: {leaked}[/]"
    )

    # A question about a protocol outside the whitelist must land in
    # out_of_scope. Routed to docs instead, the agent searches the protocols it
    # does have and answers an Aave question out of Hyperliquid's docs — fluent,
    # cited, and wrong about someone's collateral.
    off_whitelist = [c for c in cases if c.get("category") == "off_protocol"]

    off_whitelist = [c for c in cases if c.get("category") == "off_protocol"]

    proto_acc = proto_exact / proto_scored if proto_scored else 0.0
    harmful = proto_class["wrong"] + proto_class["hallucinated"]
    ptable = Table("protocol-set outcome", "n", "cost", title="protocol axis")
    ptable.add_row("exact match", f"{proto_exact}/{proto_scored}", f"[green]{proto_acc:.0%}[/green]")
    ptable.add_row("declined (got [], searches all)", str(proto_class["declined"]), "permissive")
    ptable.add_row("partial (overlaps, not equal)", str(proto_class["partial"]), "usually ok")
    ptable.add_row("wrong protocol (excludes right docs)",
                   f"[{'red' if proto_class['wrong'] else 'green'}]{proto_class['wrong']}[/]", "harmful")
    ptable.add_row("hallucinated (non-whitelisted key)",
                   f"[{'red' if proto_class['hallucinated'] else 'green'}]{proto_class['hallucinated']}[/]",
                   "harmful")
    ptable.add_row("[bold]harmful total[/bold]",
                   f"[{'red' if harmful else 'green'}]{harmful}/{proto_scored}[/]", "")
    if off_whitelist:
        ptable.add_row(
            "off-whitelist not refused",
            f"[{'red' if off_leaked else 'green'}]{len(off_leaked)}/{len(off_whitelist)}[/]",
            "harmful",
        )
    console.print(ptable)

    if proto_wrong:
        console.print("[red]harmful protocol errors (wrong/hallucinated):[/red]")
        for cid, want_p, got_p, cls, q in proto_wrong[:15]:
            console.print(f"  {cid} [{cls}]: want {want_p} got {got_p} — {q[:40]}")
    if off_leaked:
        console.print("[red]off-whitelist protocol questions not refused:[/red]")
        for cid, got in off_leaked:
            console.print(f"  {cid}: routed to {got}, expected out_of_scope")

    return {
        "accuracy": acc,
        "leaked": leaked,
        "protocol_accuracy": proto_acc,
        "protocol_class": dict(proto_class),
        "protocol_harmful": harmful,
        "hallucinated": len(hallucinated),
        "off_whitelist_leaked": len(off_leaked),
    }


# --- answers (needs API key, costs money) -------------------------------


def eval_answers(cases: list[dict], limit: int = 20) -> dict:
    from eval.judge import faithfulness, quality
    from src.graph.nodes import _format_context
    from src.retrieval.retriever import hybrid_search
    from src.graph.nodes import generate

    cases = [c for c in cases if c.get("expect_source")][:limit]
    rows, faith_scores, quality_scores = [], [], []

    for case in cases:
        # Mirror the production path: the router's protocol decision reaches
        # retrieval, so the judge scores answers built from the same context the
        # agent would actually have had.
        docs = hybrid_search(case["question"], protocols=case.get("protocols") or None)
        out = generate({"question": case["question"], "docs": docs})
        context, _ = _format_context(docs)

        f, _verdicts = faithfulness(out["answer"], context)
        q = quality(case["question"], out["answer"])

        faith_scores.append(f)
        quality_scores.append((q.helpful, q.cited, q.safe))
        rows.append((case["id"], f, q))

    table = Table("id", "faithful", "helpful", "cited", "safe", title="answer quality")
    for cid, f, q in rows:
        table.add_row(cid, f"{f:.2f}", str(q.helpful), str(q.cited), str(q.safe))
    console.print(table)

    n = len(rows)
    mean_faith = sum(faith_scores) / n
    means = [sum(s[i] for s in quality_scores) / n for i in range(3)]
    console.print(
        f"[bold]faithfulness {mean_faith:.2f} | helpful {means[0]:.1f}/5 | "
        f"cited {means[1]:.1f}/5 | safe {means[2]:.1f}/5[/bold]"
    )
    return {"faithfulness": mean_faith}


_FLAGS = (
    "guardrails",
    "retrieval",
    "verification",
    "anomaly",
    "agents",
    "routing",
    "answers",
    "offline",
    "all",
)

# Everything that runs without an API key. `--offline` is the suite someone can
# put in CI; the paid harnesses stay opt-in so a routine run never surprises
# anyone with a bill.
_FREE = ("guardrails", "retrieval", "verification", "anomaly", "agents")


def main() -> None:
    p = argparse.ArgumentParser()
    for flag in _FLAGS:
        p.add_argument(f"--{flag}", action="store_true")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    if not any(vars(args)[f] for f in _FLAGS):
        p.error("pick at least one: " + " ".join(f"--{f}" for f in _FLAGS))

    def wanted(flag: str) -> bool:
        return bool(
            vars(args)[flag] or args.all or (args.offline and flag in _FREE)
        )

    cases = load_cases()
    console.print(f"[dim]{len(cases)} golden cases[/dim]\n")

    if wanted("guardrails"):
        console.rule("guardrails")
        eval_guardrails(cases)
    if wanted("retrieval"):
        console.rule("retrieval")
        eval_retrieval(cases, k=args.k)
    if wanted("verification"):
        console.rule("verification")
        from eval.intelligence import eval_verification

        eval_verification()
    if wanted("anomaly"):
        console.rule("anomaly detection")
        from eval.intelligence import eval_anomaly

        eval_anomaly()
    if wanted("agents"):
        console.rule("agent selection")
        from eval.intelligence import eval_agent_selection

        eval_agent_selection()
    if wanted("routing"):
        console.rule("routing")
        eval_routing(cases)
    if wanted("answers"):
        console.rule("answers")
        eval_answers(cases)


if __name__ == "__main__":
    main()
