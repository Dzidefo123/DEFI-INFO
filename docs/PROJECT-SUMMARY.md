# DEFI-INFO — Project Summary

**Status as of 21 August 2026** · 875 tests passing · **13 roadmap phases complete, post-roadmap work in progress**

---

## 1. What this project is for

DEFI-INFO answers questions about decentralised finance protocols. It began as a
customer-support agent and has been rebuilt into something harder: a system that
**investigates claims and constructs evidence-backed conclusions**, rather than
retrieving text and writing a confident paragraph around it.

The distinction matters because of what a wrong answer costs here. An invented
mechanic in a DeFi answer moves someone's money. A protocol's documentation
answered from a *different* protocol's docs is more convincing than no answer at
all, and therefore worse. And an unsearched security check reads exactly like a
clean one.

So the system is organised around one governing question:

> Can we tell the difference between **"we looked and found nothing"** and
> **"we did not look"** — and can we make the answer visible to whoever is
> reading?

Everything below follows from that.

### Three rules the whole architecture obeys

**Deterministic where possible.** Identifiers, confidence arithmetic,
statistics, source reliability, severity thresholds and every gate are computed
by code. The model interprets, plans and explains. It does not calculate, and it
never rates its own sources — a model asked how good its evidence is will say it
is good, and confidence built on that measures self-regard.

**Silence is never safety.** "No anomalies detected" and "nothing was measured"
must never render the same way. This single rule shaped the severity bands
(`unknown` is not `normal`), the verification statuses (`insufficient evidence`
is not `contradicted`), the security categories, the report, and even the
evaluation harness — where a rate over an empty denominator reports *n/a* rather
than a reassuring zero.

**Keep the simple path simple.** An ordinary support question must not become
slower, costlier or more elaborate because an investigation mode exists.

---

## 2. Where this came from

### The original system

A single-protocol Hyperliquid support agent: LangGraph orchestration, hybrid
retrieval (BM25 + dense + cross-encoder rerank), deterministic guardrails,
checkpointed conversation state, and a live market-data tool.

### The multi-protocol migration (completed before this work)

Nine incremental changes turned it into a multi-protocol assistant, each shipping
with the app runnable and tests green:

| | Change |
|---|---|
| PR 0 | Protocol registry + source whitelist (a **security boundary**, not a convenience — crawling arbitrary crypto sites is how an agent ends up citing a phishing clone) |
| PR 1 | Chunks tagged with their protocol |
| PR 2 | Protocol-filtered hybrid search |
| PR 3 | Generalised ingestion (llms.txt / sitemap), robots.txt respected, whitelist enforced at fetch |
| PR 4 | Dual-axis router: which protocols + stable-vs-volatile |
| PR 5 | Live-data tool registry |
| PR 6 | Grounding and refusal hardening |
| PR 7 | Protocol-neutral copy, expanded golden set |
| PR 8 | Third protocol onboarded — **Ethena**, chosen deliberately because its vocabulary *collides* with Hyperliquid's (funding, liquidation, margin, oracle, staking all mean different things) |

**State at the start of this rebuild:** 201 tests, 1,091 chunks across three
protocols, 218 golden evaluation cases.

---

## 3. Where we are now

### The two paths

```
Simple question   →  guard → route → retrieve → grade → generate → verify  →  answer
                     (unchanged)

Investigation     →  guard → route → plan → ┬ research agent   ┐
                                            ├ blockchain agent ├→ risk → verify
                                            └ security agent   ┘        ↓
                                                          report ← evidence graph
```

The specialist agents run **in parallel**. Which of them run is decided by a
deterministic plan, recorded *before* execution — which is what later lets a
report distinguish "security was investigated and found nothing" from "security
was never investigated."

### What is built

| Phase | Component | Status |
|---|---|---|
| A0 | Index integrity — drift detection and local repair | ✅ |
| A1 | Evidence, claims and the confidence model | ✅ |
| A2 | Statistical risk engine | ✅ |
| B1 | Query classification + safety clamp | ✅ |
| B2 | Intelligence Manager, investigation branch, report | ✅ |
| C1 | Research Agent | ✅ |
| C2 | Blockchain Agent + historical feature store | ✅ |
| C3 | Security Agent + incident registry | ✅ |
| C4 | Verification Agent | ✅ |
| D1 | Evidence graph | ✅ |
| D2 | Report refinements + structured output | ✅ |
| D3 | Evaluation extension | ✅ |
| | *— roadmap complete —* | |
| E1 | Chain reads over JSON-RPC | ✅ |
| E2 | Chain-state tier + claim-kind weighting | ✅ |
| E3 | Contract registry | ✅ |

### Scale

| | |
|---|---:|
| Source files / lines | 57 / 9,456 |
| Test files / lines | 40 / 7,973 |
| Evaluation harness lines | 1,028 |
| Tests passing | **875** (from 201) |
| Suite runtime | 4.2 s |
| API spend on this rebuild | **$0.00** |
| Corpus | 1,091 chunks, 3 protocols |
| Golden cases / verification cases | 218 / 18 |

Everything is tested offline. Where a step genuinely needs a model, the model
call is injected so the surrounding pipeline is still exercised for free.

### Running it

```
python -m src.app                                  # the agent
python -m src.ingest.build_index --verify          # check index integrity
python -m src.blockchain.collect --coverage        # what on-chain history exists
python -m src.blockchain.collect --dry-run          # collect, print, store nothing
python -m src.blockchain.contracts --verify        # check registry against the chain
python -m eval.run_eval --offline                  # 5 free harnesses
python -m eval.run_eval --routing --answers        # the paid ones, opt-in
```

An hourly scheduled task collects Hyperliquid market metrics and HyperEVM chain
and contract state into the feature store — 16 series. Each becomes scoreable at
eight readings.

Use `--dry-run` for anything interactive: an off-schedule write puts near-zero
variance readings into a series meant to be evenly spaced, which shrinks the
baseline they join.

---

## 4. Measured parameters

Every number below was measured, not estimated. Where something is uncalibrated
or synthetic, it says so.

### Retrieval — 131 documentation questions, k=5, protocol-filtered

| | recall@5 | MRR@5 |
|---|---:|---:|
| Hybrid fusion only | 0.947 | 0.783 |
| **+ cross-encoder** | **0.969** | **0.874** |
| Cross-encoder contribution | +0.023 | +0.091 |

| Protocol | recall@5 | n |
|---|---:|---:|
| hyperliquid | 0.97 | 96 |
| ethena | 0.95 | 22 |
| hyperevm | 0.93 | 15 |

The protocol added last performs like the one the system was built around. Four
questions miss out of 131.

### Protocol filter ablation

| | filtered | unfiltered |
|---|---:|---:|
| recall@5 | 0.969 | 0.969 |
| Cases pulling a wrong-protocol chunk | **0 / 131** | 25 / 131 |

The filter buys **no recall**. What it buys is a *guarantee* about context
purity, which `recall@k` structurally cannot see. See §5 for the retraction.

### Guardrails — the deterministic safety gate

| | |
|---|---:|
| Adversarial recall | **38 / 38 = 100%** |
| False positives on benign traffic | **0 / 180** |

Four attack cases are phrased around protocols *outside* the whitelist and all
four are caught by the correct rule — the gate is protocol-independent by
measurement, not by assertion.

### Routing — 180 cases, live

| | |
|---|---:|
| Intent agreement | **157 / 180 = 87%** |
| Account actions routed to a non-escalating branch | **0** |
| Harmful protocol picks | 2 / 180 |
| Invented protocol keys | **0** |

The router's failure mode is *under-commitment* — it escalates when unsure and
declines to filter on generic phrasing.

> ⚠️ Measured **before** the query-type axis was added. Whether intent accuracy
> held is unmeasured.

### Verification — 18 labelled cases, 15 failure modes

| | |
|---|---:|
| Claim accuracy | 1.000 |
| **False verification rate** (bad claims accepted) | **0.000** |
| Unsupported-claim catch rate | 1.000 |
| Contradiction detection | 1.000 |
| Over-rejection rate (good claims wrongly rejected) | 0.000 |

Cases cover: unsupported, fabricated figure, causal overreach, absolute
overreach, contradiction, anonymous source, stale evidence, compound failures —
plus cases that *must* be accepted, since a set of only-bad claims would score
perfectly against a verifier that rejects everything.

> ⚠️ **Synthetic and self-authored.** Constructed one-per-failure-mode, by me,
> against a system written by me. A regression guard, not an estimate of
> real-world rates.

### Anomaly detection — 400 days per tier, synthetic

| Tier | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| easy (~15σ separation) | 1.000 | 1.000 | 1.000 | 1.000 |
| moderate | 1.000 | 0.889 | 0.941 | 0.998 |
| **hard** (inside the tail) | 1.000 | **0.389** | **0.560** | **0.978** |

**The shape is the finding.** Precision and false-positive rate hold at 1.000
and 0.000 across every tier while recall collapses to 0.39: the 3σ bar never
cries wolf, it misses subtle shifts. ROC-AUC stays at 0.978 while F1 falls to
0.560, which locates the loss in the **threshold** rather than the score — a
lower bar would recover recall if subtle shifts ever mattered more than quiet.

Detection latency: 1 reading at a sustained 4.0×, 1.8× or 1.4× shift.

> ⚠️ Thresholds are uncalibrated, and the code says so wherever severity is shown.

### Query decomposition — real corpus, 3 angles vs 1 query

| Protocol | Chunks | Distinct pages |
|---|---|---|
| ethena | 5 → 15 | 5 → 13 |
| hyperevm | 5 → 15 | 3 → 9 |
| hyperliquid | 5 → 15 | 2 → 4 |

3× the evidence surface with **zero overlap between angles**. Hyperliquid's
funding docs are concentrated so decomposition *deepens*; Ethena's are spread out
so it *broadens*.

> ⚠️ Whether the extra evidence is *useful* is unmeasured.

### Evidence independence — measured on a live run

> **2 claims · 5 pieces of evidence · 5 distinct sources → 1 independent line of
> evidence**

Two claims each carrying two citations look like two corroborated findings. Both
cited the same page, so they collapse into one finding stated twice. A flat list
of claims and citations cannot see this; the graph's convergence can.

### Source reliability — the matrix (E2)

A single ranking over sources is wrong, because the ordering inverts:

| claim kind | chain | docs | settled by |
|---|---:|---:|---|
| state | **1.00** | 0.55 | chain |
| mechanism | 0.65 | **1.00** | documentation |
| event | **0.95** | 0.65 | chain |
| unspecified | 0.90 | 0.90 | — |

Documentation records what a protocol *commits to*; chain state records what it
*is doing*. "Reserves are $87.3M" is settled by the chain — intent can be stale
or aspirational. "Liquidation uses a three-minute TWAP" is settled by the docs —
you cannot read a rule off a sequence of transactions, because observed behaviour
is consistent with many rules.

Sources that are the wrong instrument for a claim's kind are **capped, not
discounted**: however many there are, they contribute at most one source's worth
of corroboration between them. Underdetermination does not improve with
observation count.

### HyperEVM chain measurements (E1)

| | measured |
|---|---|
| Block time | 0.99 s |
| Gas limits | exactly 3,000,000 and 30,000,000 |
| Big blocks per 90 | 1 |

The last row was found empirically: a fixed 30-block window collected the small
series reliably and **never** sampled the big one. Transactions per block is a
mixture of two populations, so it is split — sampled together, its variance is
dominated by which kind of block was hit rather than by chain activity.

### Contract state (E3)

| metric | measured |
|---|---:|
| WHYPE total supply | 5,404,422.32 |
| Native backing | 5,404,422.32 |
| **Backing ratio** | **1.000000** |

The ratio is an *invariant*, not a statistic about activity: a wrapper should
hold one native coin per wrapped token, so any sustained departure from 1.0 is
under- or over-collateralisation.

### Cost, measured live

| | |
|---|---:|
| Warm documentation turn | 8 calls · 38.0 s · **$0.0994** |
| `grade` stage share | **43% of cost** |
| `context_k` 5→3 | −39% on grade, −22% on the turn |

Reducing to k=3 dropped real answer content while `recall@k` scored both as
hits — which is why k=5 is the shipped default.

---

## 5. What we got wrong, and corrected

Recording these because they are the most useful part of the record.

**The protocol filter's recall claim was an artifact — retracted.** The README
said the filter was "worth +0.031 recall on its own." On a clean index it is
worth **+0.000**. The vector store had accumulated **1,157 orphaned chunks
against 1,091 live ones**, and because most orphans were *untagged*, the metadata
filter excluded them from filtered runs while unfiltered runs retrieved them
freely.

**The cross-encoder was doing more than credited.** Same repair, opposite
direction: its MRR contribution roughly doubled once the duplicate copies padding
fusion's top ranks were gone.

**The confidence model shipped compensatory.** A five-factor geometric mean
cannot fall below ≈0.55 when one term is 0.05, so a *refuted* claim still
reported as moderately confident. Verification became a **gate**, not a factor.

**Cumulative counters are blind, not noisy — and I had the reason backwards.** I
justified differencing on-chain counters by claiming a raw counter would alarm
constantly. It does the opposite: a counter sits at its all-time high by
construction, so its z-score is pinned around +1.7 whatever happens. **A chain
that stops entirely reads normal.**

**The report led with confidence on a partial investigation.** "High confidence
(0.98)" opened a risk assessment that had measured nothing. Coverage now comes
first.

**Unsupported claims were reported as contradicted.** With no evidence, agreement
computes as 0/0 → 0.0, tripping the contradiction check — the exact conflation
the four statuses exist to prevent, inside the component built to prevent it.

**The executive summary and the final assessment were the same paragraph.** §15
gives them separate headings because they do separate jobs; printing one twice
wasted the other.

**My first anomaly benchmark measured nothing.** Anomalies sat ~15σ from
baseline, so every metric — including ROC-AUC — read 1.000. A benchmark whose
ceiling any working implementation reaches cannot distinguish a good detector
from an adequate one.

**Claim kind was agent identity in disguise.** Every claim's kind was hardcoded
per agent, which put agent back into the weighting immediately after it had been
deliberately kept out of claim identity — and wrongly: a documentation page
stating a current value would have scored documentation at 1.00 in the one row
where it is weakest. Kind is now declared per claim, bounded by what each agent
is *competent* to assert.

**The compensatory failure, relocated into the matrix.** Reliability was a
maximum within tier, but evidence quality was additive — so twenty chain
observations scored 0.891 against a documentation source's 0.880 on a claim about
mechanism. The low weight capped one axis while quality accumulated around it.

**A block guard rejected every reading it was meant to protect.** The wrapper
ratio must come from reads at one block; bracketing sequential reads with the
block height fired every time, because the client's own throttle spans two or
three blocks by construction. The guard was right; the reads had to become one
batched round trip.

**Twenty-three defects total.** Seven were pre-existing and silent, ten introduced
during this work and caught by tests before shipping, six were flaws in my own
design — three caught by tests, three by review. Every one of the pre-existing ones had passing
unit tests over the code that contained it.

---

## 6. Deliberate constraints

Things that look like gaps and are decisions.

**The security incident registry ships empty.** An entry labelled
`confirmed_incident` is trusted by everything downstream. Populating it from a
model's recollection would put potentially defamatory content about real
protocols behind that label. Every entry needs a citation an operator has read.
Until then the agent says nothing is *on file* — which is not the same as nothing
having *happened*, and it says that too.

**Semantic entailment is off by default.** One model call per surviving claim.
Right to pay on an investigation someone will act on; wrong to pay every turn.
When off, the report discloses that evidence was verified as present, sourced,
current and numerically consistent — **but not as actually being about the claim
it supports**.

**The evidence graph is per-investigation, not persistent.** A lens over the
record rather than a second store that can drift out of agreement with the first
— which is precisely the failure A0 was about.

**The contract registry has one entry, and it checks itself.** This is the one
whitelist here that can be verified rather than trusted: an undocumented API
endpoint has to be taken on faith, but a contract can be asked what it is.
Identity is confirmed against the chain before any reading is kept, so a mistyped
address or a redeployment stops collection loudly rather than producing plausible
numbers about something else — at the highest reliability tier, where a wrong
number does most damage. Widening it means finding addresses from citable
sources, not from inference.

**No new dependencies.** The statistical engine uses Python's standard library;
the feature store is SQLite; the evidence graph is adjacency dictionaries. Nothing
was added that a specific model genuinely forced.

**Placeholders refuse rather than reassure.** An unimplemented check returning
"nothing found" would be indistinguishable downstream from a real negative
finding.

---

## 7. Open items

| Item | Impact |
|---|---|
| ~~HyperEVM's explorer API returns 404~~ | **Resolved (E1)** — migrated to JSON-RPC, a more durable foundation than the explorer API it replaced. |
| **The contract registry has one entry** | Only what could be verified on chain. Widening it needs citable addresses; `--verify` then confirms each. |
| **Routing eval stale** | Not re-run since a third axis was added to the router. ~$1.70. |
| **Answer faithfulness never measured at scale** | The `--answers` harness is the last unmeasured one. |
| **Risk thresholds and verification cases are synthetic** | Real labelled incidents would replace both generators. |
| **Decomposition's effect on answer quality** | We know it triples the evidence surface; not that it helps. |
| **Feature store is young** | Collecting hourly since 20 Aug; each series scoreable at eight readings. State history cannot be backfilled — the public endpoint serves the chain head only. |

---

## 8. What the system can honestly say today

For an **ordinary support question** — unchanged and measured: protocol-scoped
retrieval at 0.969 recall, grounding enforced by two model stages, a
deterministic guardrail layer at 100% adversarial recall, and a refusal when the
documentation cannot answer.

For an **investigation** — the full pipeline runs, produces linked evidence,
scores claims against it, rejects what does not hold up, and makes the result
traceable. Demonstrated end-to-end on a deliberately dishonest input:

| Claim | Verdict | Why |
|---|---|---|
| "maintains its peg through arbitrage" | **verified** (0.88) | — |
| "holds exactly $87,300,000 in reserves" | **partially verified** (0.62) | that figure appears nowhere in the evidence |
| "the peg is *guaranteed because of* the hedge" | **partially verified** (0.62) | causal claim resting on one source |

A claim about current state now resolves against chain data:

> WHYPE total supply is 5,404,422.32 tokens, fully backed at a ratio of 1.0000.
> **verified** · reliability **1.00** (chain tier, state claim) · confidence
> **0.88 high**

Numeric consistency passes because the figures came from the reading itself.

And it can answer *why*:

```
WHY DID YOU CONCLUDE THAT? → USDe maintains its peg through arbitrage.
    [supported_by] evidence: Peg Arbitrage Mechanism
    [supported_by] evidence: Peg Arbitrage Mechanism > Agent Instructions
      [from_source] source: Peg Arbitrage Mechanism
```

— rebuilt from a checkpoint, so the question stays answerable after the fact.

**What it will not say** is that a protocol is safe because nothing was found. A
report with unmet stages opens:

> **Partial investigation.** … so this assessment covers only what was searched,
> and is not an answer about the parts that were not.

and closes:

> The strongest finding is well supported and can be relied on as stated
> (confidence 0.88). This conclusion is bounded by what was searched: no on-chain
> metric could be scored against a baseline and no security findings were on file
> to review. **Nothing here speaks to those.**

---

*Companion document: the [build log](build-log.html) carries the full
phase-by-phase engineering record — decisions, measurements and defects at each
step.*
