"""Scheduled on-chain collection.

    python -m src.blockchain.collect                    # every wired protocol
    python -m src.blockchain.collect --protocol hyperevm
    python -m src.blockchain.collect --market ETH --market BTC
    python -m src.blockchain.collect --coverage          # what history exists
    python -m src.blockchain.collect --dry-run           # collect, print, store nothing

Use --dry-run for anything interactive. An unscheduled run writes readings that
sit seconds away from the last, and a series meant to be evenly spaced then
carries points with almost no variance between them — which shrinks the baseline
they join and suppresses the anomalies it exists to catch.

Point cron or a scheduled routine at this. Nothing about scheduling lives in the
app, exactly as with `build_index --protocol`.

**Cadence is the thing that matters here, and it is the operator's decision.**
The risk engine needs at least 8 readings before it will score anything, and a
baseline is only meaningful if its readings are evenly spaced — a series
collected hourly for a day and then daily for a week is not a series, it is two.
Hourly is a reasonable default for market metrics; the interval you pick becomes
the unit every anomaly is implicitly measured in.
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from src.blockchain.collectors import DEFAULT_MARKETS, collect, needs_subject
from src.blockchain.store import coverage, record
from src.protocols import enabled_protocols

console = Console()


def run(protocols: list[str], markets: list[str], dry_run: bool = False) -> int:
    """Collect, and unless `dry_run`, persist.

    `--dry-run` exists because the collector is the one component that cannot be
    exercised without writing to the live feature store, and an off-schedule test
    run is not harmless: readings taken seconds apart have almost no variance, so
    they shrink the baseline they join and suppress the very anomalies it exists
    to catch. Two such runs had to be deleted by hand before this flag existed.

    The dry path is the full path — same collectors, same network calls, same
    validation — up to the point of writing. Anything less would test something
    other than what runs on the hour.
    """
    stored = 0
    for key in protocols:
        subjects = markets if needs_subject(key) else [""]
        for subject in subjects:
            result = collect(key, subject)
            label = f"{key}/{subject}" if subject else key

            if result.observations and not dry_run:
                n = record(result.observations)
                stored += n
                console.print(
                    f"[green]ok[/green]   {label}: {len(result.observations)} readings, "
                    f"{n} new"
                )
            elif result.observations:
                console.print(
                    f"[cyan]dry[/cyan]  {label}: {len(result.observations)} readings, "
                    f"not stored"
                )
                for obs in result.observations:
                    shown = (
                        f"{obs.value:,.0f}"
                        if float(obs.value).is_integer()
                        else f"{obs.value:,.6f}".rstrip("0").rstrip(".")
                    )
                    console.print(f"        {obs.metric:<24} {shown}")

            for err in result.errors:
                console.print(f"[yellow]note[/yellow] {err}")
    return stored


def show_coverage() -> None:
    rows = coverage()
    if not rows:
        console.print(
            "[yellow]No history yet.[/yellow] The risk engine needs at least 8 "
            "readings per series before it will score anything.\n"
            "Run: python -m src.blockchain.collect"
        )
        return

    table = Table(title="feature store coverage")
    for column in ("protocol", "metric", "subject", "n", "first", "latest", "scoreable"):
        table.add_column(column, justify="right" if column == "n" else "left")
    for row in rows:
        table.add_row(
            row["protocol"],
            row["metric"],
            row["subject"] or "—",
            str(row["observations"]),
            row["first"][:16],
            row["latest"][:16],
            "yes" if row["observations"] > 8 else f"no ({9 - row['observations']} more)",
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", action="append", help="limit to these protocols")
    parser.add_argument(
        "--market", action="append", help=f"markets to collect (default: {', '.join(DEFAULT_MARKETS)})"
    )
    parser.add_argument("--coverage", action="store_true", help="show stored history; no writes")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="collect and print readings without storing them",
    )
    args = parser.parse_args()

    if args.coverage:
        show_coverage()
        return

    protocols = args.protocol or [p.key for p in enabled_protocols()]
    markets = args.market or list(DEFAULT_MARKETS)
    stored = run(protocols, markets, dry_run=args.dry_run)

    if args.dry_run:
        console.print(
            "\n[cyan]Dry run — nothing was stored.[/cyan] Readings taken off the "
            "hour would sit seconds apart in a series meant to be evenly spaced, "
            "and a baseline built from them understates its own variance."
        )
    else:
        console.print(f"\n{stored} new observations stored.")


if __name__ == "__main__":
    main()
