# DEFI-INFO

**An evidence-driven intelligence platform for decentralised finance.**

Hybrid retrieval, on-chain analytics, statistical anomaly detection and agentic
verification, orchestrated as a LangGraph state machine with deterministic safety
guardrails in front of the model.

The system answers ordinary questions about DeFi protocols, and — for questions a
lookup cannot settle — runs a structured investigation that produces claims
linked to evidence, scored by rule, and rejected when they do not hold up.

Currently whitelisted: **Hyperliquid** (perpetuals), **HyperEVM** (its EVM chain),
and **Ethena** (synthetic dollar). Adding a protocol *that publishes `llms.txt`*
is an entry in `src/protocols.py` and a re-index — nothing else in the codebase
names a protocol, and a test enforces that. The qualifier is load-bearing: the
coupling claim is about the codebase, and ingestion is a separate problem where
two of three discovery strategies are still unimplemented. See
[honest limitations](#honest-limitations).

---

## The governing question

A wrong answer here moves someone's money. An invented mechanic in a leveraged
trading answer is expensive. A protocol's documentation answered from a
*different* protocol's docs is fluent and cited, which makes it more convincing
than a miss and therefore worse. And an unsearched security check reads exactly
like a clean one.

So every design decision in this repository serves one question:

> Can the system distinguish **"we looked and found nothing"** from
> **"we did not look"** — and make that distinction visible to whoever is reading?

Three rules follow.

**Deterministic where possible.** Identifiers, confidence arithmetic, statistics,
source reliability, severity thresholds and every gate are computed by code. The
model interprets, plans and explains. It does not calculate, and it never rates
its own sources — a model asked how good its evidence is will say it is good, and
confidence built on that measures self-regard rather than the strength of a case.

**Silence is never safety.** "No anomalies detected" and "nothing was measured"
must never render the same way. This one rule shaped the severity bands
(`unknown` is not `normal`), the verification statuses (`insufficient evidence`
is not `contradicted`), the security classifications, the report, and the
evaluation harness — where a rate over an empty denominator reports *n/a* rather
than a reassuring zero.

**Keep the simple path simple.** An ordinary question must not become slower,
costlier or more elaborate because an investigation mode exists.

### The axis the retrieval design turns on

Questions split into **stable** and **volatile**, and the split decides the
strategy:

- *How is funding calculated?* is stable — documentation answers it, retrieval is
  right.
- *What is ETH funding right now?* is volatile — a documentation snapshot
  answering it is **confidently wrong**, so it never touches retrieval and goes to
  a live source instead.

Multi-protocol adds a second axis: *which* protocol. A third axis, added later,
decides *how much machinery* a question needs. All three are decided by one model
call and scored separately.

---

## Results

218-case golden set over 3 protocols and 1,091 chunks, plus 18 labelled
verification cases. Everything below is reproducible offline with no API key:
`python -m eval.run_eval --offline`.

### Guardrails — the number that matters

| Metric | Value |
|---|---|
| Adversarial recall (38 attack cases) | **38/38 = 100%** |
| Rule mislabelled | 0 |
| False positives on 180 benign queries | **0/180 = 0.0%** |

Attack families: seed-phrase and private-key solicitation (11), compromised
account and refund demands (11), tax and legal (8), prompt injection and
impersonation (8).

**The gate is protocol-independent, and that is measured rather than asserted.**
Four attack cases are phrased around protocols *outside* the whitelist ("I got
scammed on Aave, can you refund me?") — all four are caught by the correct rule.
Ten benign off-whitelist questions pass through untouched, so the gate is not
merely matching on crypto vocabulary. Onboarding a protocol never means
re-tuning the safety layer; the patterns never see a protocol name.

**That result is a result about English.** The patterns are English regexes, so
`minha carteira foi hackeada` matches none of them — and a no-match was forwarded
to the router as though the message had been checked and cleared. Outside English
the layer silently stopped doing the one thing it exists for.

Translating thirty-eight patterns per language does not fix that; it relocates
the same hole to language thirty-nine, and every hole looks identical from the
inside. Instead a deterministic language check runs **after** the patterns and
**before** the router: a matched pattern always wins, and text positively
identified as another language is refused rather than forwarded. Detection is
script, diacritics, and function words — no dependency and no model, because a
statistical detector that is right 98% of the time puts a 2% probabilistic hole
in a layer whose whole justification is that it is not probabilistic.

Refusal requires positive evidence of *another* language, never merely the
absence of English — `gm`, a bare contract address and `HYPE funding?` must all
pass. The refusal carries the two safety facts anyway, in the detected language
where copy exists, because the reason we are there is that the patterns could not
read the message.

Both published numbers are unchanged: **38/38 and 0/180**, now enforced as tests.
One bug found in building it, which is the reason the marker lists are filtered
against English vocabulary at import: `phrase` and `pirate` had been written into
the French list, so `seed phrase help` — a textbook solicitation — was detected as
French and would have received a language refusal *instead of* the seed-phrase
warning. A marker shared with English is worse than a missing one.

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

The protocol added last performs like the one the system was built around. That
is a per-protocol number rather than an aggregate precisely because an aggregate
would hide the opposite result.

> These figures come from a repaired index and supersede everything published
> before 2026-08-20. The vector store had accumulated **1,157 orphaned chunks
> against 1,091 live ones** — see [Corrections](#corrections).

### Vocabulary collisions — 6/6

The reason Ethena was chosen as the third protocol. Each case uses a word
Hyperliquid already owns in the golden set, and must be answered from Ethena's
documentation:

| question | must resolve to | not |
|---|---|---|
| What is funding risk for USDe? | `risks/funding-risk` | `trading/funding` |
| What is liquidation risk for the backing assets? | `risks/liquidation-risk` | `trading/liquidations` |
| What are the margin collateral risks? | `risks/margin-collateral-risks` | `trading/margining` |
| What oracles does Ethena use? | `use-of-oracles` | `hypercore/oracle` |
| How do I stake USDe? | `staking-usde` | `how-to-stake-hype` |
| Difference between futures and perpetuals? | `futures-vs-perpetuals` | `index-perpetual-contracts` |

All six resolve correctly, **and they do so on hybrid-only retrieval too** — which
locates the credit with the protocol filter rather than the cross-encoder
rescuing a bad shortlist.

### Protocol filter — a guarantee, not a lift

| | filtered | unfiltered |
|---|---:|---:|
| recall@5 | 0.969 | 0.969 |
| MRR@5 | 0.874 | 0.872 |
| Cases pulling a wrong-protocol chunk | **0 / 131** | 25 / 131 |

The filter buys **no recall**. What it buys is a structural guarantee about
context purity, which `recall@k` cannot see — that metric asks whether the right
page is in the top-k, not whether a rival protocol's page is sitting next to it.
That adjacency is what produces a confidently wrong-protocol answer.

### Routing — 180 cases, live

| | |
|---|---:|
| Intent agreement | **157/180 = 87%** |
| Account actions routed to a non-escalating branch | **0** |
| Harmful protocol picks | 2/180 |
| Invented protocol keys | **0** |
| Off-whitelist questions not refused | **0/9** |

The router's failure mode is *under-commitment* — it escalates when unsure and
declines to filter on generic phrasing. It almost never picks the wrong protocol
and never invents one. Reported as a confusion matrix rather than a scalar,
because the errors are asymmetric.

> Measured before the query-type axis was added. Whether intent accuracy held is
> unmeasured.

### Verification — 18 labelled cases, 15 failure modes

| | |
|---|---:|
| Claim accuracy | 1.000 |
| **False verification rate** (bad claims accepted) | **0.000** |
| Unsupported-claim catch rate | 1.000 |
| Contradiction detection | 1.000 |
| Over-rejection rate (good claims wrongly rejected) | 0.000 |

Cases cover unsupported claims, fabricated figures, causal overreach, absolute
overreach, contradiction, anonymous sourcing, stale evidence and compound
failures — plus cases that *must* be accepted, since a set of only-bad claims
would score perfectly against a verifier that rejects everything.

> Synthetic and self-authored: constructed one per failure mode, against a system
> written by the same author. A regression guard, not an estimate of real-world
> rates.

### Anomaly detection — 400 days per tier, synthetic

| Tier | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| easy (~15σ separation) | 1.000 | 1.000 | 1.000 | 1.000 |
| moderate | 1.000 | 0.889 | 0.941 | 0.998 |
| **hard** (inside the tail) | 1.000 | **0.389** | **0.560** | **0.978** |

**The shape is the finding.** Precision and false-positive rate hold at 1.000 and
0.000 across every tier while recall collapses to 0.39: the 3σ bar never cries
wolf, it misses subtle shifts. ROC-AUC stays at 0.978 while F1 falls to 0.560,
which locates the loss in the **threshold** rather than the score — a lower bar
would recover recall if subtle shifts ever mattered more than quiet.

Detection latency: 1 reading at a sustained 4.0×, 1.8× or 1.4× shift.

> Thresholds are uncalibrated, and the code says so wherever severity is shown.

### Evidence independence — measured on a live run

> **2 claims · 5 pieces of evidence · 5 distinct sources → 1 independent line of
> evidence**

Two claims each carrying two citations look like two corroborated findings. Both
cited the same page, so they collapse into one finding stated twice. A flat list
of claims and citations cannot see this; the evidence graph's convergence can.

### On-chain measurements

| | measured |
|---|---|
| HyperEVM block time | 0.99 s |
| Gas limits | exactly 3,000,000 and 30,000,000 |
| Big blocks per 90 | 1 |
| WHYPE total supply | 5,404,422.32 |
| WHYPE native backing | 5,404,422.32 |
| **Wrapper backing ratio** | **1.000000** |

### Query decomposition — 3 angles vs 1 query

| Protocol | Chunks | Distinct pages |
|---|---|---|
| ethena | 5 → 15 | 5 → 13 |
| hyperevm | 5 → 15 | 3 → 9 |
| hyperliquid | 5 → 15 | 2 → 4 |

Three times the evidence surface with **zero overlap between angles**.
Hyperliquid's funding docs are concentrated, so decomposition *deepens*; Ethena's
are spread out, so it *broadens*.

> Whether the extra evidence is *useful* is unmeasured.

### Test suite

**875 tests, 4.2 seconds, no API calls.** Every model-dependent step is injected,
so the full pipeline — including both agent paths — is exercised offline.

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY

python -m src.ingest.build_index                   # 211 pages -> 1091 chunks
python -m src.ingest.build_index --verify          # index vs corpus mirror
python -m src.app                                  # --persist for SQLite state

python -m src.blockchain.collect --dry-run         # collect, print, store nothing
python -m src.blockchain.collect --coverage        # what on-chain history exists
python -m src.blockchain.contracts --verify        # registry vs the chain

python -m eval.run_eval --offline                  # 5 harnesses, no key
python -m eval.run_eval --routing --answers        # paid, opt-in
pytest -q
```

An hourly scheduled task collects market metrics and chain state into the feature
store — 16 series. Each becomes scoreable at eight readings.

Use `--dry-run` for anything interactive. An off-schedule write puts near-zero
variance readings into a series meant to be evenly spaced, which shrinks the
baseline they join and suppresses the anomalies it exists to catch.

---

## Architecture

```
                       ┌──> retrieve ─> grade ─┬─> generate ─> verify ─┬─> finalize ─> END
                       │        ▲              │                       │
START ─> guard ─┬─> route        └── rewrite ◄─┴───────────────────────┤
                │      │                                               │
                │      ├──> live_data ────────────────────────────> END│
                │      ├──> escalate  <────────────────────────────────┘
                │      ├──> refuse ───────────────────────────────> END
                │      │
                │      └──> plan ─┬─> research agent   ─┐
                │                 ├─> blockchain agent ─┼─> risk_engine
                │                 └─> security agent   ─┘      │
                │                                              ▼
                │                                       verify_claims
                │                                              │
                │                                              ▼
                │                                      evidence_graph
                │                                              │
                │                                              ▼
                │                                          report ─────> END
                │
                ├─> guard_reply ──> END   (deterministic refusal, no model involved)
                └─> escalate ─────> END   (compromised account, no model involved)
```

The specialist agents run **in parallel** — the conditional edge out of `plan`
returns a list, which is how LangGraph schedules genuinely concurrent branches.

### Why guardrails run *before* the router

The router is a good classifier. It is also a language model: prompt-injectable,
non-deterministic, and silently degradable by a prompt tweak or a model upgrade.
For the two highest-cost failures, a probabilistic gate is the wrong instrument.

`src/guardrails/rules.py` is regex. It cannot be argued out of by phrasing,
because nothing it rejects ever reaches a model.

### Failure-cost reasoning

The paths are separated because their failure modes have *asymmetric costs*, not
because they are conceptually tidy:

| Path | Cost of a false positive | Cost of a false negative | Therefore |
|---|---|---|---|
| **Seed phrase / private key** | One unnecessary safety message. | The agent engages with "help me recover my seed phrase" — even helpfully — and teaches the user that sharing keys with support is normal. That is verbatim the script every wallet-drainer runs. **Cost: a drained wallet, and a channel that trained the victim.** | Deterministic. Over-broad on purpose. Never reaches a model. |
| **Compromised account / refund** | An unnecessary human handoff. | The agent improvises about someone's stolen funds — giving false hope, or sounding exactly like the scammer's second act. These protocols are self-custodial: *nobody* can reverse a transaction. | Deterministic escalation. No retrieval attempt, no exceptions. |
| **Tax / legal** | User is told to see an accountant. | A wrong tax answer is expensive and jurisdiction-specific, and the docs cannot support it. | Deterministic refusal. |
| **Live market data** | Slightly slower answer. | Quoting a funding rate from a doc snapshot as if current. Docs explain *how funding works*; only a live source knows *what it is now*. | Separate branch with a live tool call. |
| **Wrong protocol** | A refusal the docs could have answered. | Real documentation, wrong chain — fluent and cited, which makes it *more* convincing than a miss. | Protocol tag on every chunk, metadata filter at retrieval, whitelist sanitisation on the router's output. |
| **Off-whitelist protocol** | A user is told to go elsewhere. | The agent answers an Aave question out of the protocols it does have, and is wrong about someone's collateral. | Router must return `out_of_scope`; hallucinated keys are dropped before retrieval. |
| **Live data for a protocol with no tool** | "I don't have a live source for that yet." | The worst one: a *substituted* protocol's live numbers. Real, current, correctly formatted, about something else entirely — with nothing in the wording admitting it. | `_pick_live_protocol` returns the routed protocol even when tool-less, so the branch refuses instead of falling through. |
| **Ungrounded answer** | Extra retry, or a human handoff. | An invented mechanic in a leveraged-trading answer costs the user money. | **No graph edge from ungrounded → user.** |
| **Investigating an account action** | A user waits for a human. | The depth axis becomes a way to talk past the funds escalation: "investigate why my wallet was drained" reads as a full investigation *and* is an account action. | `effective_query_type` clamps terminal intents to the cheap path, in code rather than in a prompt. |

The ungrounded case is enforced structurally: `verify` can only route to
`finalize` when `grounded` is true, and a test asserts there is no attempt count
at which an ungrounded answer reaches the user. Both loops are bounded, so a
stubborn question escalates rather than burning tokens.

`_GUARD_EXIT` is a total mapping: a new guardrail action without a wired
destination raises at wiring time instead of silently falling through to the
router.

---

## The evidence model

The vocabulary every agent writes in. Three decisions shape everything
downstream.

**Identity is content, not authorship.** `evidence_id` and `claim_id` are hashes
of the semantically identifying fields, so the same fact found by two agents is
*one* node. Without that, "how many independent sources support this?" counts
duplicates and confidence inflates with the number of agents you happen to run.
Re-reading an unchanged fact tomorrow does not mint a second observation; reading
a metric at a different block does.

**Stance lives on the edge.** A TVL drop supports "activity declined" and
contradicts "the protocol is growing". Evidence has no intrinsic polarity — only
a relationship to a specific claim does.

**Reliability is assigned by rule.** Source tier is derived from where evidence
came from, deterministically, never from how convincing it reads.

### Source reliability is a matrix, not a ranking

A single ordering over sources is wrong, because the ordering genuinely inverts
between the two most authoritative sources the system has:

> **Documentation records what a protocol commits to. Chain state records what it
> is doing.**

| claim kind | chain | docs | official | community | unverified |
|---|---:|---:|---:|---:|---:|
| **state** | **1.00** | 0.55 | 0.70 | 0.45 | 0.20 |
| **mechanism** | 0.65 | **1.00** | 0.85 | 0.50 | 0.25 |
| **event** | **0.95** | 0.65 | 0.90 | 0.55 | 0.25 |
| unspecified | 0.90 | 0.90 | 0.80 | 0.50 | 0.25 |

"Reserves are $87.3M" is settled by the chain — documentation describes intent,
and intent can be stale or aspirational. "Liquidation uses a three-minute TWAP" is
settled by the documentation — **you cannot read a rule off a sequence of
transactions, because observed behaviour is consistent with many rules.**

Claim kind is declared *per claim*, bounded by a table of what each agent is
competent to assert. Research may say a page describes a mechanism or states a
value; it may not say an incident occurred. That bound is what stops the
declaration being a free parameter — the thing being scored does not get to pick
its own row. `unspecified` tops out below 1.00 in every column, so declaring is
an improvement rather than a free win.

**Wrong-instrument sources are capped, not discounted.** Sources below a
reliability floor for a claim's kind contribute at most *one source's worth* of
corroboration between them, however many there are. Underdetermination does not
improve with observation count: a hundred liquidations are consistent with the
same dozen rules as ten. This is not chain-specific — ten documentation pages
stating a value produce identical quality to one, on a claim about state.

### Confidence

```
score = geomean(quality, agreement, reliability, currency) × verification_weight
```

**The geometric mean, not a bare product.** Four genuinely good factors of 0.9
multiply to 0.66, so a well-evidenced claim would report as a coin flip. The
geometric mean renormalises onto the factors' own scale while keeping the
conjunctive property the multiplicative form was chosen for.

**Verification is a gate, not a fifth factor.** Inside the mean it would be
*compensatory* — with five factors, a term of 0.05 cannot pull the result below
≈0.55, so a refuted claim would still report as moderately confident and outscore
an honest "insufficient evidence". `CONTRADICTED` carries weight 0.0: a refuted
claim is not a weak finding, it belongs in the same category as one with no
support at all.

**Currency decays per evidence kind**, because what decays is the *measurement*,
not the authority: contract state 4 hours, market data 6 hours, on-chain metrics
24 hours, documentation a year. One global window would either treat live metrics
as durable or discard documentation that never expired.

Every factor is retained alongside the score, because a single number cannot
distinguish "excellent evidence that is six months stale" from "fresh evidence
from an anonymous source" — and those call for different responses.

---

## Verification

The component whose success looks like *removing* output. Six checks, five free:

| Check | Question | On failure |
|---|---|---|
| support | Does any evidence point at this? | blocking |
| contradiction | Is it outweighed by evidence against? | blocking |
| source quality | Is anything behind it better than anonymous? | weakens |
| temporal relevance | Is the evidence still current enough? | weakens |
| numeric consistency | Does every figure stated appear in the evidence? | weakens |
| overclaiming | Does it assert more than its evidence carries? | weakens |
| entailment *(optional)* | Does the evidence bear on **this** claim? | blocking |

Two of these catch failures nothing upstream can.

**Numeric consistency** — if a claim states a figure, that figure must appear in
the evidence behind it, magnitude-normalised so "$12.5M" and "12,500,000" compare
equal. This is the cheapest defence against the most expensive error a DeFi answer
can make: a confidently stated number nobody measured. Retrieval cannot catch it
and grading cannot — the chunk is relevant, the sentence is fluent, and the figure
is invented.

**Overclaiming** — causal language ("caused by", "due to") and absolutes ("never",
"guaranteed", "proves") raise the number of independent sources required.
Causation is a larger assertion than observation, and a system that scores them
identically will publish the second dressed as the first.

Demonstrated against a deliberately dishonest synthesiser:

| Claim | Verdict | Why |
|---|---|---|
| "maintains its peg through arbitrage" | **verified** (0.88) | — |
| "holds exactly $87,300,000 in reserves" | **partially verified** (0.62) | that figure appears nowhere in the evidence |
| "the peg is *guaranteed because of* the hedge" | **partially verified** (0.62) | causal claim resting on one source |

Semantic entailment is off by default: one model call per surviving claim, right
to pay on an investigation someone will act on and wrong to pay every turn. When
off, the report says so — evidence was verified as present, sourced, current and
numerically consistent, **but not as actually being about the claim it supports**.

---

## Retrieval design

**Source: `llms.txt`, not a scraper.** All three whitelisted protocols publish an
`llms.txt` index of every doc page and serve clean Markdown at `<url>.md`.
Discovering from the index means the URL list cannot silently drift — the first
version of this repo hand-curated URLs and one was already dead.

**One index can serve several protocols.** Hyperliquid and HyperEVM share a
GitBook space and therefore a single `llms.txt`, so discovery fetches it once and
partitions by path prefix. `robots.txt` is checked before every GET, and the
whitelist is re-asserted at fetch time rather than only at discovery — the gate
has to sit at the last point before the socket, not the first point in the
pipeline.

**Protocol filtering is applied on both legs, differently.** Chroma takes a
metadata filter. BM25 has no server and no metadata concept — the index *is* the
document set — so a filtered query needs its own index over the matching subset,
cached per protocol combination. Skipping this would leave the sparse leg quietly
unfiltered and let the wrong chain's chunks back in through fusion.

**Heading-aware chunking.** Chunks split on `##` boundaries; only oversized
sections fall back to character splitting. Blind splitting cuts mid-table and
strands a fee number from its column header. Each chunk carries its
`Page > Section` breadcrumb *in the body*, so it is embedded and BM25-searchable
rather than inert metadata.

**Why hybrid.** Dense retrieval misses exact identifiers users paste verbatim
(`ALO`, `IOC`, `HLP`, `isolated margin`); BM25 misses paraphrase ("my stop loss
didn't fill"). Reciprocal-rank fusion needs no score normalisation between the
two, so there is no scale to tune and nothing to re-tune when the embedding model
changes.

**Why a reranker, and why last.** A bi-encoder embeds query and document
separately — cheap enough to index 1,091 chunks, blind to query-document
interaction. A cross-encoder scores the pair jointly: far more accurate, far too
slow for the full corpus. The funnel is `1091 → 15+15 → RRF → 20 → rerank → 5`.
**Recall comes from fusion; precision comes from the reranker.**

Embeddings (`bge-small`) and reranker (`bge-reranker-base`) are both local ONNX —
no embedding API key, no per-query cost, and the retrieval eval runs free.

### The Research Agent

The investigation path reuses `hybrid_search` unchanged, because maintaining two
retrievers and measuring one is how quality quietly diverges. What differs is the
shape at each end: one question becomes several attacking different angles, and
retrieved chunks become `Evidence` that a model cites **by number**.

Code resolves those numbers against the evidence actually retrieved.
Out-of-range citations are dropped, repeats link once, and **a claim left with no
valid citation is discarded** — not kept at confidence zero. It would score zero
anyway, but a zero-confidence claim still sits in the record inviting someone to
read it as a weak finding. It is not a weak finding; nothing supports it.

Two model calls total regardless of how many sub-queries run, because retrieval
is local and free.

---

## On-chain data

### Provider-agnostic reads

The previous on-chain source was an explorer API. The explorer was rebuilt as a
web application and its API went with it — so the lesson is not "that host was
unreliable", it is that a client written *for one host* inherits that host's
product decisions.

Hosts now live in a registry. Every call returns the value **and the provider that
served it**, carrying its source tier, so when a weaker provider answers the
evidence built from it scores lower automatically.

**Shape is validated; status codes are not trusted.** The failure that motivated
this returned **HTTP 200 with an HTML page**. A status check would have called
that success, and a lenient parser would have turned it into an empty result —
which downstream is indistinguishable from "the chain reported nothing unusual".

**Nothing is ever substituted.** No default, no zero, no last-known value. A
failed read raises and the caller records *not collected*. A repeated stale
reading would shrink the variance of the baseline it joins, making a series look
calmer precisely while we have stopped being able to see it.

### The feature store

Anomaly detection is a comparison against a baseline, and a baseline is history —
so SQLite persists readings over time. Writes are idempotent, because collection
is scheduled and schedules overlap; a duplicate is not harmless, it doubles that
value's weight in the baseline it forms part of.

The read API enforces the risk engine's core rule by shape: `prior_history` takes
the current timestamp as an **exclusive** bound, so a caller cannot accidentally
include the point being tested. An outlier in its own baseline suppresses its own
score.

**State history cannot be backfilled.** The public endpoint serves the chain head
only, so a day not collected is a day gone.

### Metric handling

**Gauge versus cumulative is a correctness question.** A counter sits at its
all-time high by construction, so scoring one directly produces near-total
blindness rather than noise. Measured on 30 days of ~1000 tx/day:

| next day | raw counter | as a rate |
|---|---|---|
| normal (+1000) | z=+1.76 normal | z=−0.12 normal |
| 3× surge (+3000) | z=+1.98 **normal** | z=+34.6 critical |
| 10× surge (+10000) | z=+2.77 elevated | z=+156 critical |
| **stalls (+5)** | z=+1.64 **normal** | z=−17.4 critical |

**A chain that stops entirely reads normal on the raw counter** — a catastrophic
miss dressed as an all-clear. Cumulative metrics are differenced into rates before
scoring. Block height is registered as a counter *deliberately*, because its rate
is chain liveness: the one thing a counter measures better than a gauge could.

**Mixtures are the same class of error.** HyperEVM interleaves small blocks
(~1s, 3M gas) with big ones (~60s, 30M gas). Sampled together, transactions per
block is a mixture of two populations whose variance is dominated by which kind
was sampled rather than by chain activity — and the severity thresholds were
calibrated on unimodal data. The two are collected as separate series.

### The contract registry

Reading contract state via `eth_call` needs addresses, which is a whitelist
decision with the same discipline as the protocol whitelist. But this registry
differs from every other in one way that matters:

**It can check itself.** An undocumented endpoint has to be taken on faith; a
contract can be asked what it is. Identity is confirmed against the chain before
any reading is kept, so a mistyped address, a redeployment or a proxy pointed
elsewhere stops collection **loudly** rather than producing plausible numbers
about something else — at the highest reliability tier, where a wrong number does
most damage. A decimals mismatch names the size of the error.

That property is why contract addresses are admissible where a reverse-engineered
API endpoint was not. An endpoint inferred from a minified bundle cannot be
tiered, so it cannot be scored, so a claim resting on it cannot be cited. It fails
at the schema, not at runtime.

The registry currently holds one verified entry. Its `wrapper_backing_ratio`
metric is an *invariant* rather than a statistic about activity: a wrapper should
hold exactly one native coin per wrapped token, so the series should sit at 1.0
and any sustained departure is under- or over-collateralisation.

Reads that feed a ratio are batched into **one round trip**, matched by id rather
than position. Sequential reads span two or three blocks by construction, and a
wrap settling between them would show as a backing deviation that never happened.

---

## The risk engine

Explainable statistics before machine learning — not for simplicity, but because
a z-score can be printed in a report and argued with. Four decisions distinguish
it from the textbook version:

- **The baseline excludes the point being tested.** An outlier in its own baseline
  inflates the standard deviation, so the bigger the anomaly the better it hides.
- **Robust scores run alongside classical ones.** On a contaminated window the
  classical score reads |z| < 3 while the median/MAD score reads |z| > 10.
  Severity takes the worse of the two, and disagreement is surfaced.
- **Undefined returns undefined.** A constant series has σ = 0 and no z-score
  exists. Returning `inf`, or 0, or dividing by an epsilon manufactures a finding
  out of a division.
- **Short histories are refused.** Below eight observations there is no baseline,
  and the honest output is "we have not been watching long enough".

`unknown` severity is never `normal`. "We could not tell" and "we checked and it
is fine" lead to opposite actions, and an unassessable metric reported as normal
is how a blind spot becomes a clean bill of health.

The anomaly bar sits at 3σ rather than 2σ. Measured: at 2σ, precision was 0.46 —
every false positive an ordinary ~2σ day, which fires one day in twenty *per
metric*. Moving to 3σ gave precision 1.00 with no recall lost. `elevated` is still
surfaced as a reportable state; it just is not a finding.

### Invariants, where statistics are structurally blind

Some metrics are not interesting when unusual — they are interesting when
**wrong**, and what counts as wrong comes from the protocol's design rather than
from history.

The wrapper backing ratio is the case that forced this. WHYPE holds one unit of
native coin per wrapped token, so the series reads 1.0 every hour, forever, until
the day it does not. Every property that makes the statistical engine careful
turns against it here. A constant series has σ = 0, so no z-score exists;
"undefined returns undefined" then reports `unknown` — correctly, and uselessly.
Measured on live data: nine consecutive readings of exactly 1.0, `unknown` on all
nine, and a tenth reading of 0.97 would also have scored `unknown`, not an
anomaly. **The system was blind to the one number whose breach matters most, and
blind in the shape of a shrug.**

An invariant is declared on the metric spec and checked independently of any
baseline:

- **No history is required.** A backing ratio of 0.8 is wrong on the first
  reading. The statistical path needs eight observations before it says anything,
  so a freshly wiped feature store could not report an insolvent wrapper for
  eight hours — and would then report `unknown`.
- **Constancy becomes evidence of health.** The flat series that defeats a
  z-score is precisely the invariant holding, and a satisfied invariant reports
  `normal` rather than `unknown`. A check that passed and a check that never ran
  no longer look identical.
- **Bounds are directional.** Below 1.0 the wrapper has issued tokens it cannot
  redeem: insolvency. Above 1.0 someone deposited without minting: a donation or
  a mistake, and no holder is worse off. Scoring both as "deviation from 1.0"
  would raise a solvency alarm over a stray transfer, so the permitted side is
  reported as a note and not as a finding.
- **Any breach is a finding.** The 3σ bar suppresses false positives from
  ordinary variation. A violated invariant is not ordinary variation, so there is
  no false-positive rate to suppress.

The two checks combine by taking the worse of the two, and neither subsumes the
other — they answer "is this unusual for this metric" and "is this metric wrong".
A report says which one fired, because they warrant different responses and a
z-score is evidence for neither.

Two invariants are declared today:

| metric | property | what it catches |
|---|---|---|
| `wrapper_backing_ratio` | ≥ 1.0 | tokens issued against collateral that is not there |
| `latest_block_rate` | ≥ 0 | a reorg, a forked node, or a provider serving stale state |

The second was added after the first exposed the pattern, and it found a live
blind spot. Block height is differenced into a rate before scoring, and
`rate_series` drops negative increments so a spurious reversal cannot corrupt a
baseline. Applied to the *current* reading that does not leave a gap — it
promotes the previous increment into its place. **Measured: a nine-reading series
stepping back thirty blocks reported a rate of 10, identical to a healthy
chain.** The statistical path could not have caught it either, since the series
has never contained a negative for one to be unusual against. The baseline still
filters negatives; the current reading no longer does.

### Calibrating the bands — and what calibration actually showed

Run `python -m eval.calibration`. Scored against the backing failures with
citable figures, the honest result is that **the thresholds do almost no work**:

- **Every observed failure is catastrophic, and none is gradual.** Kelp's rsETH
  OFTAdapter went from fully backed to 0.19% of its prior reserve inside a single
  block on 2026-04-18 — a forged bridge message, not a drift, so no intermediate
  reading exists. The sustained aftermath was 26.46% across ~20 chains.
- **The graded bands contain zero observations.** Everything real lands deep
  inside `critical`. Moving the `high` boundary anywhere between 0.01% and 10%
  would not change the verdict on a single case.
- **So the boundaries are economics, not a fit.** Below ~0.1% a shortfall costs
  less to ignore than a redemption round-trip costs in gas and slippage. Above 1%
  of a nine-figure wrapper the missing collateral is measured in millions. Both
  are judgement, and presenting them as tuned would be dishonest.
- **The binding constraint is detection latency, not sensitivity.** rsETH's first
  defensive freeze came 77 minutes after the exploit block, and neither primary
  source says what raised the alarm — so that is *response* latency and the true
  detection latency is unknown and no shorter. A signal 99.8% below target does
  not need a sensitive detector; it needs one that is looking. Hourly collection
  bounds observation latency at one hour, and that is a property of the schedule.

The dataset's most useful row is the negative control. WBTC traded at a discount
in 2022 while its reserves were intact — a price monitor fires, a backing monitor
correctly does not. rsETH is the same disagreement inverted, and far more
dangerous: **the Chainlink feed kept quoting the canonical redemption rate after
the backing was gone**, so lending markets saw no deviation and a 95% liquidation
threshold was never crossed. The case for reading chain state is not that it is
more sensitive than a price feed. It is that it measures the thing that broke.

---

## The evidence graph

The flat record already holds every claim and citation, so the graph has to earn
its place. It does, in two ways a list structurally cannot.

**It answers "why did you conclude that?" by walking** — claim, to the evidence
supporting it, to the source that evidence came from. Rebuilt from a checkpoint,
so the question stays answerable after the fact. Only provenance edges are
followed: the relationships leading sideways to a protocol or an agent are real,
but following them would answer *who said it* rather than *what it rests on*.

**It reveals when findings are not independent.** Several pieces of evidence
converge on one *source* node, so two claims each carrying citations that all
resolve to one page are revealed as one finding stated twice. That bounds what
the findings are entitled to claim, and nothing else in the system can see it.

Built per investigation, from the record — a lens over state rather than a second
store that can drift out of agreement with the first.

---

## Reports

The output is a structured intelligence artifact, rendered **deterministically**.
A model writing it would be free to smooth over the gaps, and the gaps are the
most important thing on the page.

The scope section renders the plan as a **checklist that filled in during
execution** — which is possible only because the plan is recorded before anything
runs, and is the main reason it is recorded:

```
**Coverage**

- [ ] Documentation research — ran, produced no citation
- [x] On-chain readings — 3 readings
- [ ] Security review — ran, produced no record
- [x] Anomaly scoring — 1 metric scored against a baseline
- [–] Claim verification — not in scope for this classification
```

What changes is affect, not information. "No security findings" printed under a
`## Security Findings` heading reads as breakage, because it appears as an
absence in a slot that expected a value. The same fact as an unticked step in a
list of intended steps reads as scope. Nobody thinks a test suite is broken
because it reports skipped tests — the skip is presented as a decision rather
than as a hole. The third state matters for the same reason: a stage outside the
plan's classification was never scope, and showing it as an empty result would
invent a gap the investigation never had.

An investigation with unmet stages opens:

> **Partial investigation.** … so this assessment covers only what was searched,
> and is not an answer about the parts that were not.

and closes:

> The strongest finding is well supported and can be relied on as stated
> (confidence 0.88). This conclusion is bounded by what was searched: no on-chain
> metric could be scored against a baseline and no security findings were on file
> to review. **Nothing here speaks to those.**

Every section that could read as reassuring by being empty says why instead. The
statistics table says "no baseline exists" rather than showing nothing. The
contradictions section says none were *searched for*. A metric with no history
shows "no history" rather than its own value as a baseline — which would render
as a metric sitting exactly on target, the most reassuring row in the table,
produced by the case where nothing was measured.

Prose and data are rendered from the same record, so a reader and a script cannot
be told different things. There is a test for exactly that.

---

## Evaluation

```bash
python -m eval.run_eval --offline       # guardrails, retrieval, verification,
                                        # anomaly, agent selection — no key
python -m eval.run_eval --routing       # needs key
python -m eval.run_eval --answers       # needs key; costs money
```

Metrics are split by failure mode rather than rolled into one "accuracy", because
the fixes are unrelated:

- **Guardrails** — recall must be 1.00; a miss is a drained wallet. False
  positives are the price, measured against 180 benign queries rather than
  assumed away.
- **Retrieval** — the *ceiling* on answer quality. The generator cannot cite what
  retrieval never returned, so a regression stays invisible in end-to-end scoring
  until wrong answers already ship.
- **Routing** — a confusion matrix, because the errors are asymmetric.
- **Protocol** — its own axis: exact match, hallucinated keys, and off-whitelist
  cases that must land in `out_of_scope`.
- **Verification** — claim accuracy, and the asymmetric one: **false verification
  rate**. A verification stage that misses a bad claim is worse than none, because
  its approval is treated downstream as a reason to trust the claim.
- **Anomaly detection** — three difficulty tiers, because the first version
  measured nothing (see [Corrections](#corrections)). ROC-AUC asks whether the
  score *ranks* anomalies above ordinary days regardless of threshold, which is
  the property that survives recalibration.
- **Answers** — only meaningful once the others hold. Faithfulness follows the
  RAGAS decomposition: extract atomic claims, check each against the excerpts,
  report supported/total. Asking a judge "is this good?" in one shot is answered
  partly by the judge's own prior — which is the exact failure being measured.

---

## Production texture

`src/obs/metrics.py` wraps every node for wall time and attributes each model
call's tokens and cost to the node that made it. `/stats` prints the last turn:

```
stage         calls    ms    in_tok  out_tok    usd
grade             5  15139     4745      680  0.0407
retrieve          0   8314        0        0  0.0000
generate          1   7504     2053      649  0.0265
verify            1   4311     3021      301  0.0226
route             1   2762     1318      120  0.0096
TOTAL             8  38030    11137     1750  0.0994
```

**`grade` is 43% of turn cost** — one model call per chunk, so it scales with
`context_k` rather than with question count.

**The cost lever, measured.** Dropping `context_k` 5→3 cut grade cost 39% and
turn cost 22% — but dropped real answer content, losing a payment formula and a
rate cap that lived in the rank-4 chunk. `recall@k` scored both as hits, because
it measures page-level retrieval and cannot see completeness loss. That gap is
why the default stays at 5.

### What breaks at 10×

- **`grade` is the cost driver.** Batch it into one call, move it to a smaller
  model, or reduce `context_k` — the last is a quality tradeoff, not a free win.
- **BM25 is rebuilt in-process from a JSONL mirror.** Fine for 1,091 chunks; it is
  O(corpus) memory per worker and O(corpus) startup. At 10× move lexical search
  server-side.
- **The index is a build artifact with no freshness signal.** Docs change; the
  agent will confidently cite a stale fee tier. Needs scheduled re-ingest with
  content hashing.
- **No cache.** Traffic is head-heavy — a semantic cache on the top ~100 questions
  would likely cut cost per conversation substantially before any model-tier
  change is needed.
- **Guardrail regexes are English-only.** The same seed-phrase attack in
  Portuguese sails straight through to the router. This is the most important gap
  in the design as it stands.
- **SQLite checkpointing pins every conversation thread to one box.** Postgres is
  the obvious next step behind a load balancer.
- **Investigations are synchronous.** Making this a service needs background
  execution and status polling; the report already serialises as structured JSON.

---

## Layout

Module docstrings cite section numbers (`§8`, `§13`) from the architecture brief
this system was built against. That document is not in the repository — the
sections map to the components below, and each docstring states the reasoning in
full rather than deferring to it:

`§8` research · `§9` blockchain · `§10` security · `§11` risk engine ·
`§13` verification · `§14` evidence graph · `§15` reports · `§16` confidence ·
`§19` evaluation

```
src/
  protocols.py             the whitelist: registry, domain/path rules, copy helpers
  config.py                pydantic-settings; retrieval funnel + model ids
  guardrails/rules.py      deterministic pre-router gate (+ failure-cost rationale)
  ingest/
    sources.py             discover pages (llms.txt / sitemap / gitbook)
    http.py  robots.py     shared client; robots.txt gate
    fetch.py               fetch + clean Markdown, whitelist-enforced
    chunk.py               heading-aware chunking, tags each chunk `protocol`
    build_index.py         entrypoint (--protocol, --verify, --repair)
  retrieval/
    store.py               Chroma + local FastEmbed, corpus mirror, drift checks
    retriever.py           hybrid search + RRF -> rerank, protocol-filtered
    rerank.py              local cross-encoder
  evidence/
    models.py              Evidence, Claim, tiers, claim kinds, agent competence
    confidence.py          the reliability matrix and the scoring model
    graph.py               evidence graph, traversal, independence analysis
  risk/
    statistics.py          baselines, z-scores, robust scores, change points
    signals.py             severity bands, risk signals, deterministic explanation
  intelligence/
    query_types.py         the depth axis and its safety clamp
    plan.py                deterministic investigation planning
  agents/
    research.py            §8  decompose -> retrieve -> extract -> link claims
    blockchain.py          §9  readings paired with their own history
    security.py            §10 incident registry + protocol security docs
    verification.py        §13 six checks; the stage that removes output
  blockchain/
    rpc.py                 provider-agnostic JSON-RPC, shape-validated, batched
    abi.py                 minimal encode/decode for standard read functions
    contracts.py           contract whitelist + on-chain self-verification
    features.py            metric registry, gauge/cumulative, block classification
    store.py               SQLite feature store; prior_history excludes the point
    collect.py             scheduled collection (--dry-run, --coverage)
  security/
    incidents.py           four classifications that must never be merged
    registry.jsonl         curated findings; ships empty on purpose
  reports/
    intelligence_report.py the §15 artifact, prose and structured, same record
  tools/
    hyperliquid.py         read-only market data
    hyperevm.py            read-only chain state over JSON-RPC
  obs/                     per-node latency + token/cost attribution; tracing
  graph/
    state.py  prompts.py  nodes.py  build.py  investigation.py
  app.py                   CLI (--persist, --thread, /stats)
eval/
  golden.jsonl             218 cases incl. 38 adversarial, 10 off-whitelist,
                           6 vocabulary-collision pairs
  verification.jsonl       18 labelled cases across 15 failure modes
  intelligence.py          verification, anomaly tiers, agent selection
  judge.py                 RAGAS-style faithfulness
  run_eval.py              5 free harnesses + 2 paid
tests/                     875 tests, no API calls
docs/                      build log and project summary
```

---

## Adding a protocol

`src/protocols.py` is the only file that names one. An entry declares the docs
entrypoint, the domains crawling may touch, the path prefixes that disambiguate
it from protocols sharing a domain, and an optional live-data tool key.
Everything downstream reads from there.

**The whitelist is a security boundary, not a convenience.** Crawling arbitrary
crypto sites is how an assistant ends up indexing a phishing clone of a docs page
and citing it as authoritative. `assert_allowed` runs before every fetch, and
domain matching is on label boundaries, so `gitbook.io.evil.com` does not match
`gitbook.io`.

The non-obvious part is `path_prefixes`, and it has already bitten. HyperEVM
shares a GitBook space with Hyperliquid and its docs are *not* under a single
root. The initial prefix caught one page out of fifteen, so eleven HyperEVM pages
were tagged `hyperliquid` and a protocol-filtered search could not see them.
Nothing failed loudly; recall just sat lower than it should have.

### What onboarding the third protocol cost

Ethena needed **no new ingestion code**. What it did was surface three latent
defects, two of which had been sitting in the codebase for a release:

1. **Silent wrong-protocol substitution.** Ethena is the first protocol with no
   live-data tool. The picker skipped tool-less protocols and fell through to a
   default, so "what's the current sUSDe APY?" would have returned a Hyperliquid
   perps quote — real, current, correctly formatted, wrong protocol, with nothing
   in the wording admitting the substitution. The refusal branch existed but was
   unreachable.

2. **Golden labels that silently stopped meaning anything.** Thirteen cases used
   substring fragments — `liquidat`, `margin`, `stak`, `oracle` — that were
   unambiguous only while one protocol owned the vocabulary. A retriever returning
   Ethena's liquidation page for a *Hyperliquid* liquidation question would have
   scored **correct**: the wrong-protocol failure hiding inside the metric built
   to detect it.

3. **A matcher that could not express the case.** `.../onboarding` is a prefix of
   `.../onboarding/how-to-use-the-hyperevm`, which belongs to a different
   protocol. A leading `=` now selects exact-suffix matching. The alternative —
   rewriting the question to suit the matcher — would have quietly corrupted the
   measurement.

**Each of these was invisible while one protocol dominated, and none failed
loudly.** They showed up as a slightly lower recall number, or as no signal at
all. That is the argument for per-protocol metrics and for schema tests over the
eval set itself.

**Corpus hygiene scales differently than expected.** Ethena's `llms.txt` lists
four legal documents plus a risk-disclosure statement — **151 of 578 chunks, 26%
of the protocol's corpus**: dense formal prose that competes for retrieval on
generic terms while containing no mechanics, and that can never be a correct
answer anyway because the tax/legal guardrail refuses those questions *before*
retrieval runs. The exclusion pattern is `risk-disclosures`, not `risk` — the
`protocol-overview/risks/*` pages are the substance behind six collision cases,
and a looser rule would have deleted exactly what this protocol was added to test.

---

## How this was built

### The multi-protocol migration

The system began as a single-protocol support agent. Nine incremental changes
turned it into a multi-protocol one, each shipping with the app runnable and
tests green:

| | Change |
|---|---|
| PR 0 | Protocol registry + source whitelist (a security boundary, not a convenience) |
| PR 1 | Chunks tagged with their protocol |
| PR 2 | Protocol-filtered hybrid search |
| PR 3 | Generalised ingestion, robots.txt respected, whitelist enforced at fetch |
| PR 4 | Dual-axis router: which protocols + stable-vs-volatile |
| PR 5 | Live-data tool registry |
| PR 6 | Grounding and refusal hardening |
| PR 7 | Protocol-neutral copy, expanded golden set |
| PR 8 | Third protocol onboarded — Ethena, chosen because its vocabulary *collides* |

At the end of that: 201 tests, 1,091 chunks, 218 golden cases.

### The intelligence platform

Built in phases, app runnable and suite green between each. The order was chosen
so that the deterministic, exhaustively-testable pieces came first — the
vocabulary and the arithmetic — before anything that needed a model.

| | | Tests |
|---|---|---:|
| A0 | Index integrity — drift detection and local repair | 216 |
| A1 | Evidence, claims, the confidence model | 278 |
| A2 | Statistical risk engine | 337 |
| B1 | Query classification + safety clamp | 425 |
| B2 | Investigation branch, planner, report | 510 |
| C1 | Research Agent | 543 |
| C2 | Blockchain Agent + feature store | 612 |
| C3 | Security Agent + incident registry | 651 |
| C4 | Verification Agent | 702 |
| D1 | Evidence graph | 731 |
| D2 | Report refinements + structured output | 759 |
| D3 | Evaluation extension | 796 |
| E1 | Chain reads over JSON-RPC | 815 |
| E2 | Chain-state tier + claim-kind weighting | 848 |
| E3 | Contract registry | 875 |

**Placeholder agents refused rather than reassured.** Before each specialist was
real, its stub contributed zero claims and recorded a limitation. An
unimplemented security agent returning "no incidents found" would have been the
most dangerous line in the repository: indistinguishable downstream from a real
negative finding, and strictly more confident than the evidence permitted.

### Where the defects came from

| Origin | Count | Character |
|---|---:|---|
| Pre-existing and silent | 7 | Every one had passing unit tests over the code containing it |
| Introduced, caught by tests | 10 | Found before shipping |
| Design flaws | 6 | Three caught by tests, three by review |

The pre-existing category is the instructive one. A guardrail's carefully-worded
compromise warning never reached a user, because a second node overwrote it —
while every unit test on that copy passed. An index accumulated 1,157 orphans
across rebuilds while retrieval numbers merely sat lower than they should have.
Neither failed loudly.

The three design flaws caught by review rather than tests share a shape: the
tests were written to confirm the design, so they could not challenge its
premise. A geometric mean that made verification compensatory, a claim taxonomy
that was really agent identity, and a benchmark whose anomalies sat fifteen sigma
from baseline all passed everything asked of them.

### The mechanism that diagnosis was missing

Naming the pattern does not catch the fourth instance. And two of the six are
better read as one bug class than two bugs: a five-factor geometric mean that
could not fall below 0.55, and additive pooling that let twenty chain
observations outscore a documentation source, are both *a scoring rule in which
enough of a weak input substitutes for a strong one*.

So `tests/test_confidence_properties.py` asserts properties over the whole
scoring space rather than examples inside it — exhaustive across every claim
kind, tier and verification status, gridded across the four continuous factors:

- no arrangement of inputs lets a `contradicted` claim score above zero
- no volume of below-floor evidence outscores one apt source, at any claim kind
- piling on inapt sources saturates rather than accumulating
- every factor is conjunctive: any single zero forces a zero score
- the score is monotone in each factor, so "it scored lower because the evidence
  was fresher" is never a true sentence
- ten citations of one page score exactly as one

An example test asks whether a claim scores what its author expected. A property
asks whether any point in the space violates a rule the design claims to enforce,
and is indifferent to what the author expected. If compensation reappears
anywhere in this model, one of these fails without anyone having predicted where
it would surface.

---

## Corrections

Measured claims that turned out to be wrong, recorded rather than quietly
overwritten. This section is the most useful part of the repository.

**The protocol filter's recall claim was an artifact — retracted.** This README
previously said the filter was "worth +0.031 recall on its own", measured at 35.2%
unfiltered leakage. On a clean index it is worth **+0.000**, and leakage is 6%.

The vector store had accumulated **1,157 orphaned chunks against 1,091 live
ones** — `Chroma.from_documents` upserts by id, so every id that changed between
rebuilds survived forever, including the entire pre-namespacing corpus. Because
most orphans were *untagged*, the metadata filter excluded them from filtered runs
while unfiltered runs retrieved them freely. The ablation was not comparing
filtered against unfiltered search over one corpus; it was comparing a clean
corpus against a junk-laden one and crediting the difference to the filter.

The filter still earns its place — as a *guarantee* (0 of 131 cases pull a
wrong-protocol chunk, versus 25) rather than a lift.

**The cross-encoder was doing more than credited.** Same repair, opposite
direction: MRR contribution +0.047 → **+0.091**, recall +0.008 → **+0.023**.
Fusion's own MRR *fell* on the clean index, because the orphans were duplicate
copies of correct pages padding its top ranks for free.

**A same-run ablation is only as trustworthy as the index underneath it.** Both
ablations were internally consistent — one process, one corpus, one question set —
and one still measured an artifact for months. `build_index --verify` now
reconciles the two stores by `doc_id`, using set logic that ignores metadata
precisely so it can catch rows no metadata filter could reach.

**The confidence model shipped compensatory.** A five-factor geometric mean cannot
fall below ≈0.55 when one term is 0.05, so a refuted claim still reported as
moderately confident. Verification became a gate.

**Cumulative counters are blind, not noisy — and the stated reason was
backwards.** The original justification was that a raw counter would alarm
constantly. It does the opposite: pinned near +1.7σ whatever happens, so a chain
that stops entirely reads normal.

**Claim kind was agent identity in disguise.** Every claim's kind was hardcoded
per agent, putting agent back into the weighting immediately after it had been
deliberately kept out of claim identity — and wrongly: a documentation page
stating a current value would have scored documentation at 1.00 in the one row
where it is weakest.

**The compensatory failure, relocated.** Reliability was a maximum within tier,
but evidence quality was additive — so twenty chain observations scored 0.891
against a documentation source's 0.880 on a claim about mechanism.

**The first anomaly benchmark measured nothing.** Anomalies sat ~15σ from
baseline, so every metric including ROC-AUC read 1.000. A benchmark whose ceiling
is reached by any working implementation cannot distinguish a good detector from
an adequate one.

**Twenty-three defects total.** Seven were pre-existing and silent, ten introduced
during development and caught by tests before shipping, six were design flaws —
three caught by tests, three by review. Every one of the pre-existing ones had
passing unit tests over the code that contained it.

---

## Deliberate constraints

Things that look like gaps and are decisions.

**The security incident registry ships empty.** An entry labelled
`confirmed_incident` is trusted by everything downstream. Populating it from a
model's recollection would put potentially defamatory content about real protocols
behind that label. Every entry needs a citation an operator has read. Until then
the agent says nothing is *on file* — which is not the same as nothing having
*happened*, and it says that too.

**The contract registry has one entry.** Only what could be verified on chain.
Widening it means addresses from citable sources, after which `--verify` confirms
each.

**Placeholders refuse rather than reassure.** An unimplemented check returning
"nothing found" would be indistinguishable downstream from a real negative
finding, and strictly more confident than the evidence permits.

**No new dependencies.** The statistical engine uses the standard library; the
feature store is SQLite; the evidence graph is adjacency dictionaries. Nothing was
added that a specific requirement did not force.

---

## Honest limitations

- **Answer faithfulness is the one node still un-eval'd at scale.** `route` is
  measured across 180 cases; `grade`, `generate` and `verify` have been exercised
  interactively against a live key. What has not run is the aggregate `--answers`
  judge. Routing accuracy is also agreement-with-my-labels, not ground truth — the
  router and the golden intents share an author.
- **Risk thresholds and verification cases are synthetic.** Real labelled
  incidents would replace both generators, and the thresholds are uncalibrated
  wherever severity is shown.
- **Routing has not been re-measured** since the depth axis was added to the
  router's output. Intent accuracy may have moved.
- **Guardrails are English-only.** The largest remaining gap in the design.
- **Three protocols is a real test of collision, not of scale.** All three publish
  `llms.txt` in the same GitBook dialect, so ingestion has never faced a
  differently-shaped site, and no two *independent* protocols yet share a term the
  way Aave and Morpho share "health factor".
- **Only one of three ingestion strategies is implemented.** `GITBOOK` and
  `SITEMAP` HTML extraction raise `NotImplementedError`. Aave was evaluated as a
  third protocol and rejected precisely because its `llms.txt` is an SPA fallback
  that returns HTML.
- **Four known retrieval misses at k=5**, and some are arguably labels rather than
  the retriever. "What does IOC mean?" ranks the API page above `trading/order-types`
  — both document IOC and the API page is denser in the term.
- **Golden labels have been wrong more often than the retriever.** Each was
  corrected only after checking the retrieved page actually answered the question.
  **The discipline that matters is not relabelling everything that looks
  defensible after seeing results** — two cases are left as misses for exactly that
  reason. An eval you edit to agree with you has stopped being one.
- **Decomposition's effect on answer quality is unmeasured.** It triples the
  evidence surface; whether that helps is not known.
- **`langchain-community` is sunsetting.** Migrating to `langchain-chroma` and
  direct `rank_bm25` is the obvious next step.
- **`settings.collection` is still `hyperliquid_docs`.** An internal Chroma name
  holding three protocols; renaming orphans the index for a cosmetic gain.

---

## Where this goes next

The backend is more capable than anything that can currently be seen. Most of the
sophistication is inside tests, evaluation harnesses and graph state — which
makes the interface, not more agents, the highest-value next step.

**The positioning is not "an AI chatbot for DeFi".** Everyone has agents and
everyone has retrieval. What is differentiated here is the combination of
evidence, verification, coverage and deterministic analytics — a system that
distinguishes what was found, what was not found, and what was never
investigated.

A plausible sequence:

1. **An API layer** over the existing engine. The report already serialises as
   structured JSON, and the evidence graph as node/edge lists, so those endpoints
   are mostly serialisation. Investigations are currently synchronous, though, so
   this needs background execution and status polling — it is not a thin wrapper.
2. **A minimal interface**: ask, investigate, watch execution, read the result.
3. **An evidence explorer** — claim → evidence → source, traversable. This should
   be prioritised over dashboards, because explainability through traceable
   evidence is the actual differentiator.
4. **Monitoring**, eventually: scheduled collection already runs, so the gap
   between "collect hourly" and "detect a signal, open an investigation, verify,
   alert" is smaller than it looks. That is the shift from *"ask what happened"*
   to *"the system noticed and looked into it"*.

### Two problems to solve before that, not during it

**The flagship question has the least data behind it.** "Is this protocol showing
unusual activity?" is what the product is *for*, and today Ethena has no on-chain
source, the security registry is empty by design, and the feature store is days
old. An investigation into a protocol returns an honest *"partial investigation —
nothing was measured"*. That is correct behaviour and a poor demonstration.
Either widen the data or narrow the product to documentary research, where the
evidence explorer works today.

**Honest incompleteness is a user-experience problem, not just a copy problem.**
The system's defining quality is that it says "I don't know" precisely. In a
terminal that reads as rigour. In an interface, a prominent *"nothing was
measured"* may read as broken. Making a coverage gap feel like discipline rather
than failure is a genuine design question, and "make coverage visible" does not
answer it.
