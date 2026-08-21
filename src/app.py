"""Interactive CLI for the multi-protocol crypto support agent.

    python -m src.app                 # in-memory conversation state
    python -m src.app --persist       # SQLite checkpointing, survives restarts
    python -m src.app --thread abc123 # resume a prior conversation
"""

from __future__ import annotations

import argparse
import uuid

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.graph.build import build_graph
from src.obs.metrics import Report, UsageCollector, format_report, report_scope
from src.obs.tracing import init_tracing
from src.protocols import coverage_phrase

console = Console()

BANNER = f"""[bold]Crypto protocol support agent[/bold]
RAG + live on-chain data, orchestrated with LangGraph.
Covering: {coverage_phrase()}

Try:
  How is funding calculated on Hyperliquid?
  What is funding risk for USDe?        (same word, different protocol)
  What's ETH funding right now?
  What is the HyperEVM dual block architecture?
  Why was my stop loss not filled?
  Someone drained my wallet, can you reverse it?

/stats  per-stage latency and cost for the last turn
/quit   exit"""


def _checkpointer(persist: bool):
    if not persist:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    # Conversation state survives process restarts. The obvious next step for a
    # real deployment is the Postgres checkpointer behind a load balancer, since
    # SQLite pins every thread to one box.
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver.from_conn_string(".checkpoints.sqlite")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist", action="store_true", help="SQLite checkpointing")
    parser.add_argument("--thread", default=None, help="resume a conversation id")
    args = parser.parse_args()

    traced = init_tracing()

    cm = _checkpointer(args.persist)
    # SqliteSaver.from_conn_string is a context manager; MemorySaver is not.
    if hasattr(cm, "__enter__"):
        with cm as saver:
            _run(build_graph(saver), args, traced)
    else:
        _run(build_graph(cm), args, traced)


def _run(graph, args, traced: bool) -> None:
    thread_id = args.thread or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    console.print(Panel(BANNER, border_style="cyan"))
    console.print(
        f"[dim]thread={thread_id}  persist={args.persist}  tracing={'on' if traced else 'off'}[/dim]"
    )

    last: Report | None = None

    while True:
        try:
            question = console.input("\n[bold cyan]you>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nBye.")
            return

        if not question:
            continue
        if question in ("/quit", "/exit"):
            return
        if question == "/stats":
            console.print(
                Panel(format_report(last), title="last turn")
                if last
                else "[dim]no turns yet[/dim]"
            )
            continue

        report = Report()
        turn_config = config | {"callbacks": [UsageCollector(report)]}

        # report_scope binds the report for `timed` (wall time); the callback
        # carries it for token/cost. Neither path puts it in graph state, so
        # nothing telemetry-shaped is checkpointed.
        with console.status("thinking..."), report_scope(report):
            state = graph.invoke({"question": question}, turn_config)
        last = report

        trace = f"intent={state.get('intent') or '-'}"
        # Only worth showing when it changed the path taken; printing "cx" on
        # every ordinary turn is noise.
        if (qt := state.get("query_type")) and qt != "cx":
            trace += f"  investigation={qt}"
        if state.get("guardrail_rule"):
            trace += f"  guardrail={state['guardrail_rule']}"
        if state.get("escalation_reason"):
            trace += f"  escalated={state['escalation_reason']}"
        trace += f"  {report.total_ms:.0f}ms  ${report.total_cost_usd:.4f}"
        console.print(f"[dim]{trace}[/dim]")

        console.print(Markdown(state["answer"]))


if __name__ == "__main__":
    main()
