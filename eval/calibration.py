"""Calibrate the wrapper backing bands against observed failures.

Run: `python -m eval.calibration`

The bands in `src.risk.invariants` were set from round numbers. This scores them
against the only real backing failures with citable figures, and — more usefully —
reports which bands have no observations behind them at all.

What the exercise establishes is not what the thresholds should be. It is that
threshold choice is close to irrelevant for this metric, and that the number worth
tuning is somewhere else entirely. See `REPORT` at the bottom.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.risk.invariants import Bound, Invariant
from src.risk.severity import Severity

DATASET = Path(__file__).parent / "wrapper_backing.jsonl"

WRAPPER = Invariant(
    target=1.0,
    bound=Bound.AT_LEAST,
    rationale="A wrapper must hold one unit of native coin per wrapped token.",
)


def cases() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(invariant: Invariant = WRAPPER) -> tuple[list[dict], int]:
    rows, correct = [], 0
    for case in cases():
        got = invariant.severity_for(case["ratio"])
        ok = got.value == case["expected"]
        correct += ok
        rows.append({**case, "got": got.value, "ok": ok})
    return rows, correct


def band_coverage(invariant: Invariant = WRAPPER) -> dict[str, list[str]]:
    """Which severity bands any observation actually lands in.

    An empty band is not automatically wrong, but a threshold with nothing on
    either side of it has not been calibrated by anything — it has been chosen,
    and should be described that way.
    """
    coverage: dict[str, list[str]] = {s.value: [] for s in Severity}
    for case in cases():
        coverage[invariant.severity_for(case["ratio"]).value].append(case["case"])
    return coverage


def main() -> None:
    rows, correct = score()
    print(f"Wrapper backing calibration — {len(rows)} observed cases\n")
    width = max(len(r["case"]) for r in rows)
    for r in rows:
        mark = "ok " if r["ok"] else "MISS"
        stated = "stated" if r["stated_directly"] else "derived"
        print(f"  {mark} {r['case']:<{width}}  ratio={r['ratio']:<9.5f} "
              f"{r['phase']:<16} expected={r['expected']:<8} got={r['got']:<8} ({stated})")
    print(f"\n  {correct}/{len(rows)} classified as expected\n")

    print("Band coverage — how many observations sit in each band:\n")
    for band, names in band_coverage().items():
        note = "  <-- no observations" if not names else ""
        print(f"  {band:<10} {len(names)}{note}")
        for n in names:
            print(f"             {n}")

    print(REPORT)


REPORT = """
Findings
--------

1. Every observed backing failure is catastrophic, and none is gradual.
   rsETH went from fully backed to 0.19% of its prior reserve inside a single
   block — there is no intermediate reading, because the transition was one
   forged message rather than a drift. The sustained aftermath, 26.46%, is the
   value a poller would actually observe, and it is still an order of magnitude
   below any threshold worth arguing about.

2. The graded bands have nothing behind them. ELEVATED (past tolerance) and
   HIGH (0.1%) contain zero observations. Every real breach lands deep inside
   CRITICAL. Moving the HIGH boundary anywhere between 0.01% and 10% would not
   change the verdict on a single case in this dataset.

   That is worth stating plainly rather than presenting tuned-looking numbers:
   the bands are set from economic reasoning, not fitted to data. Below roughly
   0.1% a shortfall costs less to ignore than a redemption round-trip costs in
   gas and slippage, so nobody can act on it and it reads as an accounting bug
   rather than a solvency event. Above 1% of a nine-figure wrapper, the missing
   collateral is measured in millions. Both boundaries are judgement, and the
   only empirical claim the dataset supports is the one the design already
   makes: any breach past dust is a finding.

3. Threshold sensitivity is not the binding constraint. Detection latency is.
   The exploit executed at 17:35 UTC; the first defensive freeze came at 18:52,
   seventy-seven minutes later. Neither primary source says what raised the
   alarm, which means that figure is response latency and the true detection
   latency is unknown and no shorter.

   A signal 99.8% below target does not need a sensitive detector. It needs a
   detector that is looking. An hourly collector bounds observation latency at
   one hour by construction, and that bound is a property of the schedule, not
   of any threshold in this module.

4. The negative control is the argument for reading chain state at all.
   WBTC traded at a discount in 2022 while its reserves were intact; a
   price-based monitor fires on that and a backing monitor correctly does not.
   The rsETH incident is the same disagreement in the opposite direction, and it
   is the more dangerous one: the Chainlink feed kept quoting rsETH at its
   canonical redemption rate after the backing was gone, so lending markets saw
   no deviation at all and the 95% liquidation threshold was never crossed.

   Price and backing are different quantities. The case for this invariant is
   not that it is more sensitive than a price feed — it is that it measures the
   thing that broke.
"""


if __name__ == "__main__":
    main()
