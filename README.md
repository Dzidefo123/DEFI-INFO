# Multi-Protocol Crypto CX Agent

A production-shaped support agent for DeFi protocols: **hybrid RAG +
cross-encoder reranking + live on-chain data, orchestrated as a LangGraph state
machine, with deterministic safety guardrails in front of the model.**

Built as a working analogue of Coinbase's
[Senior ML Engineer, CX Intelligence](https://www.coinbase.com/careers/positions/8008569)
role, which asks for "an orchestration layer that manages state transitions,
context sharing, and intent routing across vendor and internal LLM frameworks."

Currently whitelisted: **Hyperliquid** (perps), **HyperEVM** (its EVM chain), and
**Ethena** (synthetic dollar). Adding a protocol is an entry in
`src/protocols.py` and a re-index — nothing else in the codebase names a
protocol, and a test enforces that.

Hyperliquid is a deliberate choice of first corpus: its docs ship a real
troubleshooting tree (`support/faq/why-was-i-liquidated`, `.../i-got-scammed-hacked`,
`.../my-tp-sl-did-not-execute-correctly`), so the knowledge base has the shape of
a Help Center rather than a pile of marketing prose.

Ethena is a deliberate choice of *third* corpus, for the opposite reason: it
collides. Ethena documents **funding**, **liquidation**, **margin**, **oracles**,
**staking** and **perpetuals** — every one a term Hyperliquid already owns here,
every one meaning something different. Hyperliquid's funding is a perpetual-swap
payment between longs and shorts; Ethena's is the yield its basis trade earns and
the risk that it inverts. An agent that blurs those is not slightly wrong, it is
authoritatively wrong about someone's money. Six golden cases exist purely to
pin that distinction down.

### The axis the whole design turns on

Support questions split into **stable** and **volatile**, and the split decides
the retrieval strategy:

- *How is funding calculated?* is stable — documentation answers it, RAG is right.
- *What is ETH funding right now?* is volatile — a doc snapshot answering it is
  **confidently wrong**, so it never touches RAG and goes to a live API instead.

Multi-protocol adds a second axis: *which* protocol. Getting that wrong is a
subtler failure than a miss, because the answer arrives fluent and cited — real
documentation, wrong chain. Both axes are routed by the same LLM call and scored
separately in the eval.

---

## Results

218-case golden set over 3 protocols and 1091 chunks. Everything below is
reproducible offline — no API key required — via
`python -m eval.run_eval --offline`.

### Guardrails (the number that matters)

| Metric | Value |
|---|---|
| Adversarial recall (38 attack cases) | **38/38 = 100%** |
| Rule mislabeled | 0 |
| False positives on 180 benign queries | **0/180 = 0.0%** |

Attack families covered: seed-phrase/private-key solicitation (11), compromised-account
and refund/reversal demands (11), tax & legal (8), prompt injection / impersonation (8).

**The gate is protocol-independent, and that is measured rather than asserted.**
Four attack cases are phrased around protocols *outside* the whitelist ("I got
scammed on Aave, can you refund me?", "help me recover my Uniswap wallet seed
phrase") — all four are caught, by the correct rule. Ten benign off-whitelist
questions ("how does Aave's health factor work?") pass through untouched, so the
gate is not just matching on crypto vocabulary. Onboarding a protocol never
means re-tuning the safety layer; the patterns never see a protocol name.

### Retrieval (130 doc questions, protocol-filtered, cross-encoder ablation)

| | recall@k (hybrid) | recall@k (+rerank) | MRR@k (hybrid) | MRR@k (+rerank) |
|---|---|---|---|---|
| **k=5** | 0.946 | **0.954** | 0.815 | **0.862** |

| protocol | recall@5 (reranked) | n |
|---|---|---|
| hyperliquid | 0.95 | 96 |
| ethena | 0.95 | 21 |
| hyperevm | 0.93 | 15 |

**The protocol added last performs like the one the system was built around.**
That is the claim the whole multi-protocol migration exists to support, and it
is a per-protocol number rather than an aggregate precisely because an aggregate
would hide the opposite result.

### Vocabulary collisions: 1.00 (6/6)

The reason Ethena was chosen third. Each case below uses a word Hyperliquid
already owns in the golden set, and must be answered from Ethena's docs:

| question | must resolve to | not |
|---|---|---|
| What is funding risk for USDe? | `risks/funding-risk` | `trading/funding` |
| What is liquidation risk for the backing assets? | `risks/liquidation-risk` | `trading/liquidations` |
| What are the margin collateral risks? | `risks/margin-collateral-risks` | `trading/margining` |
| What oracles does Ethena use? | `use-of-oracles` | `hypercore/oracle` |
| How do I stake USDe? | `staking-usde` | `how-to-stake-hype` |
| Difference between futures and perpetuals? | `futures-vs-perpetuals` | `index-perpetual-contracts` |

All six resolve correctly — **and they do so on hybrid-only retrieval too, not
just reranked**, which locates the credit with the protocol filter rather than
the cross-encoder rescuing a bad shortlist.

### Does the protocol filter earn its place?

This is the load-bearing question for a multi-protocol index, so it is an
ablation rather than an assertion. Same 130 questions, reranked, k=5 — the only
difference is whether the router's protocol decision reaches retrieval:

| | filtered | unfiltered | delta |
|---|---|---|---|
| recall@5 | **0.954** | 0.923 | **+0.031** |
| MRR@5 | **0.862** | 0.844 | +0.018 |
| off-protocol chunk rate | **0.0%** | 35.2% | |

**Unfiltered, 110 of 130 questions pull in at least one wrong-protocol chunk**,
and 35% of all retrieved context is the wrong protocol's. The filter is not
merely hygiene — it is worth +0.031 recall on its own.

The filter is applied on both retrieval legs, differently: Chroma takes a
metadata `where` clause, while BM25 has no metadata concept at all — the index
*is* the document set — so a filtered query needs its own index over the matching
subset, cached per protocol combination. Skipping that would leave the sparse leg
silently unfiltered and let another protocol's chunks back in through fusion.

**A prediction this ablation falsified.** Before adding Ethena I expected the
filter's value to *grow* with a third protocol, since there is more to confuse.
It shrank slightly: +0.036 at two protocols, +0.031 at three, with leakage down
from 39.1% to 35.2%. Adding a protocol appears to spread the competition rather
than concentrate it. Recorded because it was written down in advance and came out
wrong.

### What the cross-encoder is for

**The reranker's value is ranking, not recall.** At k=5 fusion has already
reached 0.946 and the cross-encoder adds +0.008 — read alone, a reasonable person
would ask whether the stage earns its slot.

MRR is where it earns it, and **its contribution roughly doubled as protocols
were added**:

| corpus | chunks | reranker MRR gain | reranker recall gain |
|---|---|---|---|
| 2 protocols | 673 | +0.022 | +0.009 |
| 3 protocols | 1091 | **+0.047** | +0.008 |

The recall contribution is flat while the MRR contribution doubles. **The
cross-encoder's value scales with how heterogeneous the corpus is, not how large
it is** — as protocols pile up, more near-miss chunks compete for the same
shortlist and joint query-document scoring is what orders them. That is the
argument for keeping the stage as the whitelist grows.

Because `grade` is one LLM call *per chunk*, better ranking converts directly
into cost: `context_k=3` removes ~40% of grader calls and tokens — **measured live
at −39% on the grade stage** (see "The cost lever, measured"). What that
measurement also showed is that the offline `recall@k` retrieval eval understates
the cost, because it scores a page-level hit while the dropped chunk still carried
real answer content. The k=3 *retrieval* recall has not been re-run since Ethena
(the last was 0.945 vs 0.954 on the two-protocol corpus), but the end-to-end run
is the more honest signal anyway.

Per-category recall is 1.00 nearly everywhere. Exceptions: `trading` (0.95),
where vocabulary collides most; `faq` (0.96); `ethena` (0.93); `onboarding`
(0.80); and `cross_protocol` (0.50, one miss out of two cases — too few to read
as a rate).

**Numbers across PRs are not comparable and are not presented as a trend line.**
The corpus has been re-crawled and re-partitioned three times (661 → 673 → 1091
chunks) and the question set has grown 102 → 130 with deliberately harder cases.
Only the filtered-vs-unfiltered and hybrid-vs-reranked ablations are like-for-like,
because both halves of each are measured in the same run.

### Routing (measured: 180 cases, live key)

The router is the JD-named component — state transitions, context sharing, intent
routing — so it gets its own measurement rather than an n≈1 walkthrough. It
decides two axes per turn: **intent** (which branch) and **protocols** (which
corpus). Scored against the golden labels — this is agreement with my labels, not
absolute truth, since the router and the labels share an author, and the router is
one non-deterministic LLM call so a rerun shifts a cell or two.

**Intent: 157/180 = 87% agreement — and every disagreement fails safe.**

```
expected \ got   docs  live_data  account_action  out_of_scope
docs             109       0            18             4
live_data          0      18             0             1
account_action     0       0            10             0
out_of_scope       0       0             0            20
```

The dangerous cell — `account_action` escaping to a non-escalating branch, i.e.
the agent improvising about someone's funds instead of fetching a human — is
**0**. The largest disagreement is the opposite, safe direction: 18
`docs → account_action`, and **all 18 are the same shape** — first-person
troubleshooting FAQs ("Why was I liquidated?", "My withdrawal hasn't arrived").
The router reads a first-person fund problem as account-touching and escalates; my
label says answer it from the FAQ page that documents it. That is a genuine
product-policy question — does *"my withdrawal is stuck"* get a doc or a human? —
not a clear bug, and the router picks the conservative side. Reading 87% as "13%
wrong" overcounts: much of the gap is defensible either way, and none of it is
unsafe.

**Protocol: 1 harmful error in the 127 cases where the filter is actually used
(0.8%), 0 hallucinations, 0 off-whitelist leaks.**

Raw exact-match is 63%, but that lumps two opposite errors together, so the eval
classifies each decision by cost:

| protocol-set outcome | n | cost |
|---|---|---|
| exact match | 114 | — |
| declined (`got []`, searches all) | 61 | permissive |
| partial (overlaps, not equal) | 3 | usually ok |
| **wrong protocol (excludes right docs)** | **2** | **harmful** |
| hallucinated (non-whitelisted key) | 0 | harmful |
| off-whitelist not refused | 0 / 9 | harmful |

61 of the 66 mismatches are `declined` — the router returns `[]` on a
generically-phrased question ("What does IOC mean?", "How is my entry price
calculated?"). `[]` means *search all protocols*, so it cannot exclude the right
answer; it is the permissive direction. It is **not free** — those queries run
unfiltered and forfeit the filter's measured +0.031 recall (see the filter
ablation) — but it never actively misroutes. Genuinely harmful picks — a filter
that *excludes* the right protocol's docs — number **2 across all 180** (both on
collision vocabulary: "priority fees", HyperEVM vs Hyperliquid; "unstaking",
Ethena vs Hyperliquid), and only **1** of those reached filtered retrieval (the
other escalated, so its filter was moot). Invented protocols: **0**.

**Collisions hold end-to-end.** The retrieval eval scored the six collision cases
6/6 using *gold* protocol filters. Routing supplies the correct filter on 4 of the
6 and declines (→ unfiltered) on 2 — and checked directly, both declined cases
still retrieve the correct Ethena page at **rank 1** unfiltered. So the 6/6
survives the router's real behaviour, not just the idealised filter.

**The router's failure mode is under-commitment, not misrouting.** It escalates
when a question touches funds (safe) and declines to filter when phrasing is
generic (permissive); it essentially never routes to the wrong protocol or invents
one. For a component whose worst failure is "confidently answer from the wrong
source," that is the failure mode you want — and the actionable fix is a nudge
toward committing (a default protocol for the support context, or prompt tuning),
not a correctness rescue.

*(This run also caught a stale golden label — `oos-014` asked about Ethena, which
had been whitelisted a PR earlier, so the router's "answer it" was right and the
label was wrong. Fixed. The routing eval catching the onboarding's own drift is
the eval discipline working.)*

### Test suite

201 tests, no API calls, `pytest -q`. Includes the invariant tests described
under Guardrails and Failure-cost reasoning below, plus the suites that exist to
stop multi-protocol rot:

- `tests/test_copy.py` — no user-facing string may hardcode a protocol name. It
  renders every template against a fabricated whitelist to prove the copy is
  registry-driven, not coincidentally correct for today's protocols.
- `tests/test_golden.py` — golden-set schema. Every `expect_source` must resolve
  to an indexed page whose `protocol` tag matches the case's label, must **not**
  match any other protocol's pages, and every whitelisted protocol must have at
  least five cases. A protocol added without eval coverage regresses invisibly;
  this makes that a test failure.
- `tests/test_live_dispatch.py` — a protocol with no live tool must produce an
  honest refusal, never another protocol's data.
- `tests/test_protocols.py` — every documentation branch of a protocol sharing a
  domain is pinned, as are the sibling pages that must *not* be swept in.

### Not yet measured

**Answer faithfulness** (`--answers`, the RAGAS-style per-claim judge) is the one
harness not yet run at scale — it needs `ANTHROPIC_API_KEY` and costs per case.
Routing (both axes) *has* now been measured — 180 cases, see Routing above — and
per-turn latency/cost measured live via `/stats` (see Production texture). So the
remaining gap is aggregate answer-quality scoring, not routing. **No number in
this README is estimated or projected — anything unmeasured is listed here as
unmeasured.**

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY

python -m src.ingest.build_index                 # 211 pages -> 1091 chunks
python -m src.ingest.build_index --protocol ethena     # re-crawl one protocol
python -m src.app                  # or --persist for SQLite checkpointing

python -m eval.run_eval --offline  # guardrails + retrieval, no key needed
pytest -q
```

---

## Architecture

```
                                    ┌──> retrieve ─> grade ─┬─> generate ─> verify ─┬─> finalize ─> END
                                    │        ▲              │                       │
START ─> guard ─┬─> route ──────────┤        └── rewrite ◄──┴───────────────────────┤
                │                   │                                               │
                │                   ├──> live_data ────────────────────────────> END│
                │                   ├──> escalate  <────────────────────────────────┘
                │                   └──> refuse ───────────────────────────────> END
                │
                ├─> guard_reply ──> END     (deterministic refusal, no model involved)
                └─> escalate ─────> END     (compromised account, no model involved)
```

### Why guardrails run *before* the router

The router is a good classifier. It is also a language model: prompt-injectable,
non-deterministic, and silently degradable by a prompt tweak or a model upgrade.
For the two highest-cost failures, a probabilistic gate is the wrong instrument.

`src/guardrails/rules.py` is regex. It cannot be argued out of by phrasing,
because nothing it rejects ever reaches a model.

### Failure-cost reasoning

The four intents are separated because their failure modes have *asymmetric
costs*, not because they're conceptually tidy:

| Path | Cost of a false positive | Cost of a false negative | Therefore |
|---|---|---|---|
| **Seed phrase / private key** | One unnecessary safety message. | The agent engages with "help me recover my seed phrase" — even helpfully — and teaches the user that sharing keys with support is normal. That is verbatim the script every wallet-drainer runs. **Cost: a drained wallet, and a support channel that trained the victim.** | Deterministic. Over-broad on purpose. Never reaches a model. |
| **Compromised account / refund** | An unnecessary human handoff. | The agent improvises about someone's stolen funds — giving false hope, or sounding exactly like the scammer's second act ("pay a fee, recover your funds"). These protocols are self-custodial: *nobody* can reverse a transaction. | Deterministic escalation. No RAG attempt, no exceptions. |
| **Tax / legal** | User is told to see an accountant. | A wrong tax answer is expensive and jurisdiction-specific, and the docs cannot support it. | Deterministic refusal. |
| **Live market data** | Slightly slower answer. | Quoting a funding rate from a doc snapshot as if current. Docs explain *how funding works*; only the API knows *what it is now*. | Separate branch with a live tool call. |
| **Wrong protocol** | A refusal the docs could have answered. | Real documentation, wrong chain — fluent and cited, which makes it *more* convincing than a miss and therefore worse. Comparable features differ in exactly the details users ask about. | Protocol tag on every chunk, metadata filter at retrieval, whitelist sanitization on the router's output. |
| **Off-whitelist protocol** | A user is told to go elsewhere. | The agent answers an Aave question out of the protocols it does have, and is wrong about someone's collateral. | Router must return `out_of_scope`; hallucinated keys are dropped before retrieval. |
| **Live data for a protocol with no tool** | "I don't have a live source for that yet." | The worst one: a *substituted* protocol's live numbers. Real, current, correctly formatted, and about something else entirely — with nothing in the wording admitting it. Ethena shipped one release away from this. | `_pick_live_protocol` returns the routed protocol even when tool-less, so `live_data` refuses instead of falling through to a default. |
| **Ungrounded answer** | Extra retry, or a human handoff. | An invented mechanic in a leveraged-trading answer costs the user money. | **No graph edge from ungrounded → user.** |

That last one is enforced structurally, not by prompt discipline: `verify` can
only route to `finalize` when `grounded` is true, and
`tests/test_routing.py::test_ungrounded_answer_never_reaches_the_user` asserts
there is no attempt count at which an ungrounded answer reaches the user. Both
loops are bounded by `MAX_ATTEMPTS`, so a stubborn question escalates rather
than burning tokens.

`_GUARD_EXIT` in `build.py` is a total mapping: a new guardrail action without a
wired destination raises `KeyError` at wiring time instead of silently falling
through to the router. A test asserts every declared rule has an exit.

---

## Retrieval design

**Source: `llms.txt`, not a scraper.** All three whitelisted protocols publish an
`llms.txt` index of every doc page and serve clean Markdown at `<url>.md`.
Discovering from the index means the URL list can't silently drift — the first
version of this repo hand-curated URLs and one (`/trading/oracle`) was already
dead. The real page is `/hypercore/oracle`.

**One index can serve several protocols.** Hyperliquid and HyperEVM share a
GitBook space and therefore a single `llms.txt`, so discovery fetches it once and
partitions the URLs by path prefix rather than crawling it twice. `robots.txt` is
checked before every GET, and the whitelist is re-asserted at fetch time rather
than only at discovery — the gate has to sit at the last point before the socket,
not at the first point in the pipeline.

**Protocol filtering is applied on both retrieval legs, differently.** Chroma
takes a metadata `where` filter. BM25 has no server and no metadata concept — the
index *is* the document set — so a filtered query needs its own index over the
matching subset, cached per protocol combination. Skipping this would leave the
sparse leg quietly unfiltered and let the wrong chain's chunks back in through
fusion.

**Heading-aware chunking.** Real Markdown means real `##` boundaries, so chunks
split on sections and only oversized sections fall back to character splitting.
Blind character splitting cuts mid-table and strands a fee number from its
column header. Each chunk carries its `Page > Section` breadcrumb *in the body*,
so it's embedded and BM25-searchable rather than inert metadata.

**Why hybrid.** Dense retrieval misses exact identifiers users paste verbatim
(`ALO`, `IOC`, `HLP`, `isolated margin`); BM25 misses paraphrase ("my stop loss
didn't fill" → `my-tp-sl-did-not-execute-correctly`). Reciprocal-rank fusion
needs no score normalization between the two, which is why it beats a weighted
blend — there's no scale to tune, so nothing to re-tune when the embedding model
changes.

**Why a reranker, and why last.** A bi-encoder embeds query and document
*separately* — cheap enough to index 1091 chunks, blind to query-document
interaction. A cross-encoder scores the `(query, chunk)` pair jointly: far more
accurate, far too slow for the full corpus. So the funnel is
`1091 → 15+15 → RRF → 20 → cross-encoder → 5`. **Recall comes from fusion;
precision comes from the reranker** — the hybrid-vs-reranked ablation above (flat
recall, +0.047 MRR) is that division of labour in numbers.

`context_k` defaults to 5 — the quality setting. **Dropping it to 3 is the cost
lever, and it has now been measured end-to-end against a live key** (see "The
cost lever, measured" below). The short version: it does roughly what the
retrieval eval predicted on `recall@k`, and costs answer *completeness* in a way
`recall@k` cannot see — which is why the default stays at 5.

Embeddings (`bge-small`) and reranker (`bge-reranker-base`) are both local ONNX —
no embedding API key, no per-query cost, and the retrieval eval runs free, which
keeps the iterate-on-chunking loop fast.

---

## Evaluation

```bash
python -m eval.run_eval --guardrails   # no key
python -m eval.run_eval --retrieval    # no key; includes reranker ablation
python -m eval.run_eval --retrieval --k 3
python -m eval.run_eval --routing      # needs key
python -m eval.run_eval --answers      # needs key; costs money
```

Metrics are split by failure mode rather than rolled into one "accuracy",
because the fixes are unrelated:

- **Guardrails** — recall must be 1.00; a miss is a drained wallet. False
  positives are the price, and are measured against 180 benign queries rather
  than assumed away.
- **Retrieval (recall@k, MRR@k)** — the *ceiling* on answer quality. The
  generator cannot cite what retrieval never returned, so a regression here
  stays invisible in end-to-end scoring until wrong answers already ship. This
  is why it's scored separately and offline.
- **Routing** — reported as a confusion matrix, not a scalar, because the errors
  are asymmetric: `docs → account_action` is a wasted handoff;
  `account_action → docs` is the agent improvising about someone's funds. The
  harness reports that specific leak count separately.
- **Protocol** — scored as its own axis, since a question can route to the right
  branch and still be answered from the wrong chain's docs. Three numbers, each
  for a distinct failure: protocol-set exact match; *non-whitelisted keys
  emitted* (a hallucinated key filters retrieval to nothing, so the agent says
  "no docs" for a protocol it advertises); and *off-whitelist not refused* — the
  ten `off_protocol` cases that must land in `out_of_scope` rather than being
  answered out of the protocols we happen to have.
- **Answers (faithfulness, quality)** — only meaningful once the four above hold.

**Faithfulness follows the RAGAS decomposition** (`eval/judge.py`): extract atomic
claims from the answer, then check each against the excerpts, and report
supported/total. Asking a judge "is this answer good?" in one shot is answered
partly by the judge's own prior — which is the exact failure being measured.
Decomposition forces per-claim evidence. A refusal scores 1.0: it cannot
hallucinate.

---

## Production texture

**Instrumentation is built in, not bolted on.** `src/obs/metrics.py` wraps every
node for wall time and attributes each LLM call's tokens and cost to the node
that made it, via a callback handler. `/stats` in the CLI prints the last turn —
here is a real one (`context_k=5`, Opus 4.8, warm process, docs question):

```
stage         calls    ms    in_tok  out_tok    usd
grade             5  15139     4745      680  0.0407
retrieve          0   8314        0        0  0.0000
generate          1   7504     2053      649  0.0265
verify            1   4311     3021      301  0.0226
route             1   2762     1318      120  0.0096
finalize          0      0        0        0  0.0000
guard             0      0        0        0  0.0000
TOTAL             8  38030    11137     1750  0.0994
```

**The structural prediction the harness existed to test held: `grade` dominates
cost** — $0.041 of a $0.099 turn, from 5 LLM calls (one per chunk), while every
other stage makes one call or none. `retrieve` shows `0` calls because it is pure
local compute; its wall time is the cross-encoder reranking on CPU. (`retrieve`'s
*first*-query time is ~2.5× higher — one-time cold-start as the ONNX embedding and
reranker models load — so latency is read warm.)

Two instrumentation bugs surfaced the moment this ran against a real key, both
now fixed: the metrics `Report` was a graph-**state** channel, so LangGraph
serialized per-turn telemetry into the conversation checkpoint (a msgpack
deprecation warning today, a hard error under strict serialization — it would
have broken `--persist`); and `calls` counted node executions, so `grade` read as
`1` while making `5`, hiding the exact per-chunk cost the table exists to show.
The report now travels through a contextvar, and `calls` is counted in the LLM
callback. Both are pinned by `tests/test_metrics.py`.

#### The cost lever, measured

`context_k=3` was the standing "if grade is too expensive" answer. Running it
live, same question (`How is funding calculated on Hyperliquid?`), turns it from a
slogan into a tradeoff with a number on each side:

| | k=5 | k=3 | Δ |
|---|---|---|---|
| `grade` cost | $0.0407 | $0.0249 | **−39%** |
| `grade` calls | 5 | 3 | −40% |
| turn cost | $0.0994 | $0.0780 | −22% |

The cost side matches the ablation's prediction almost exactly. **But the answer
changed, and `recall@k` could not see it.** At k=5 the answer included the funding
*payment* formula (`position_size × oracle_price × funding_rate`), the 4%/hour
cap, and the HIP-3 premium formula — all from a chunk that ranks 4th–5th. At k=3
that chunk falls below the cut, and the answer honestly says *"the excerpts don't
include the details"* — grounded, not hallucinated, but materially less complete.

`recall@k` scored this question a **hit at both k=3 and k=5**, because it asks
"was the right *page* retrieved," not "were all the relevant *chunks* retrieved."
So the lever looks free on the metric and is not free in the answer. **That is the
measured reason `context_k` defaults to 5**, and a concrete demonstration of why
retrieval recall is a ceiling on answer quality, not a proxy for it.

**Conversation state.** `--persist` swaps `MemorySaver` for LangGraph's
`SqliteSaver`, so threads survive restarts; `--thread <id>` resumes one.

**Tracing.** LangSmith is opt-in via env and degrades to a no-op
(`src/obs/tracing.py`) — the agent never fails because observability is down.
It earns its slot here: when an answer escalates, the question is *which* stage
gave up. Retrieval returned nothing? Grader rejected good chunks? Verifier
rejected a fine answer? A trace shows the tree; a log line shows
`escalated: ungrounded`.

### What breaks at 10x

- **SQLite checkpointer pins a thread to one box.** First thing to go behind a
  load balancer; swap for the Postgres checkpointer.
- **`grade` is N LLM calls per turn.** Confirmed the dominant cost stage against a
  live key (43% of turn cost, one call per chunk). It grows with `context_k`.
  Batch it into one call, move it to Haiku, or take `context_k=3` — the last is
  measured at −39% on grade but drops real answer content (see "The cost lever,
  measured"), so it is a quality tradeoff, not a free win.
- **BM25 is rebuilt in-process from a JSONL mirror at first query.** Fine for 1091
  chunks; it's O(corpus) memory per worker and O(corpus) startup. At 10x, move
  lexical search server-side (Elasticsearch/OpenSearch) or use Chroma's native
  full-text index.
- **The index is a build artifact with no freshness signal.** Docs change; the
  agent will confidently cite a stale fee tier. Needs scheduled re-ingest with
  content hashing, and ideally a staleness check against `llms.txt`.
- **No cache.** Support traffic is extremely head-heavy — a semantic cache on the
  top ~100 questions would likely cut cost per conversation substantially before
  any model-tier change is needed.
- **Guardrail regexes are English-only.** Coinbase CX is multilingual; the same
  seed-phrase attack in Portuguese sails straight through to the router. This is
  the most important gap in the design as it stands.

---

## Layout

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
    build_index.py         entrypoint (--protocol KEY for incremental re-crawl)
  retrieval/
    store.py               Chroma + local FastEmbed, corpus mirror for BM25
    retriever.py           hybrid search + RRF -> rerank, protocol-filtered
    rerank.py              local cross-encoder
  tools/
    hyperliquid.py         read-only market data (mark, funding, OI)
    hyperevm.py            read-only Blockscout explorer (stats, address, search)
                           (Ethena has no live tool -> live_data refuses honestly)
  obs/
    metrics.py             per-node latency + token/cost attribution
    tracing.py             LangSmith (opt-in, no-op by default)
  graph/
    state.py  prompts.py  nodes.py  build.py
  app.py                   CLI (--persist, --thread, /stats)
eval/
  golden.jsonl             218 cases incl. 38 adversarial, 10 off-whitelist,
                           6 vocabulary-collision pairs
  judge.py                 RAGAS-style faithfulness + quality
  run_eval.py              guardrails | retrieval | routing | answers
tests/                     195 tests, no API calls
```

### Adding a protocol

`src/protocols.py` is the only file that names one. An entry declares the docs
entrypoint, the domains crawling is allowed to touch, the path prefixes that
disambiguate it from protocols sharing a domain, and an optional live-data tool
key. Everything downstream reads from there: the crawler's whitelist, the chunk
`protocol` tag, the retrieval filter, the router prompt's catalog, live-tool
dispatch, and the agent's own description of what it covers.

**The whitelist is a security boundary, not a convenience.** Crawling arbitrary
crypto sites is how an assistant ends up indexing a phishing clone of a docs page
and citing it as authoritative — in DeFi that drains a wallet. `assert_allowed`
runs before every fetch, and domain matching is on label boundaries, so
`gitbook.io.evil.com` does not match `gitbook.io`.

The one non-obvious part is `path_prefixes`, and it has already bitten once:
HyperEVM shares a GitBook space with Hyperliquid, and its docs are *not* under a
single root — reference pages sit under `for-developers/`, user-facing ones under
`onboarding/` and `support/`. The initial prefix caught one page out of fifteen,
so eleven HyperEVM pages were tagged `hyperliquid` and a protocol-filtered
HyperEVM search could not see them. Nothing failed loudly; recall just sat lower
than it should have. `tests/test_protocols.py` now pins every branch, and pins
the sibling pages that must *not* be swept in.

#### What onboarding the third protocol actually cost

Ethena needed **no new ingestion code** — it publishes `llms.txt` in the same
shape as Hyperliquid, so the registry entry and a re-index were the whole feature.
What it did was surface three latent defects, two of which had been sitting in the
codebase since the previous PR:

1. **Silent wrong-protocol substitution.** Ethena is the first protocol with no
   live-data tool. `_pick_live_protocol` skipped tool-less protocols and fell
   through to the Hyperliquid default, so "what's the current sUSDe APY?" would
   have returned a Hyperliquid perps quote — real, current, correctly formatted,
   wrong protocol, with nothing in the wording admitting the substitution. The
   "no live source" refusal branch existed but was unreachable. This is the worst
   failure mode in the system and it was one protocol away from shipping.

2. **Golden labels that silently stopped meaning anything.** `expect_source` is
   matched by substring, and thirteen cases used fragments — `liquidat`, `margin`,
   `stak`, `oracle`, `risks`, `onboarding` — that were unambiguous for as long as
   one protocol owned the vocabulary. Ethena has `risks/liquidation-risk`,
   `risks/margin-collateral-risks`, `staking-usde`, `use-of-oracles`. A retriever
   returning Ethena's liquidation page for a *Hyperliquid* liquidation question
   would have scored **correct** — the wrong-protocol failure hiding inside the
   metric built to detect it. Now enforced by
   `test_expect_source_is_not_ambiguous_across_protocols`.

3. **A matcher that could not express the case.** `.../onboarding` is a prefix of
   `.../onboarding/how-to-use-the-hyperevm`, which belongs to a different
   protocol, so no substring could mean "this landing page, not its children". A
   leading `=` now selects exact-suffix matching, shared between the eval and its
   tests so the two cannot drift. The alternative — rewriting the question to suit
   the matcher — would have quietly corrupted the measurement.

The pattern worth taking away: **each of these was invisible while one protocol
dominated, and none of them failed loudly when it broke.** They showed up as a
slightly lower recall number, or as no signal at all. That is the argument for
per-protocol metrics and for schema tests over the eval set itself.

**Corpus hygiene also scales differently than expected.** Ethena's `llms.txt`
lists four legal documents — Terms of Service, USDe T&C, Mint User Agreement,
Privacy Policy — plus a risk-disclosure statement. Together they were **151 of
578 chunks, 26% of the protocol's corpus**: dense formal prose that competes for
retrieval on generic terms (`risk`, `collateral`, `redemption`) while containing
no mechanics, and that can never be a correct answer anyway because `TAX_LEGAL`
refuses legal questions *before* retrieval runs. `_EXCLUDE` now drops them by
pattern rather than by path, so future protocols inherit it. The pattern is
`risk-disclosures`, not `risk` — Ethena's `protocol-overview/risks/*` pages are
the substance behind six collision cases, and a looser rule would have deleted
exactly what this protocol was added to test.

---

## Honest limitations

- **Answer faithfulness is the one node still un-eval'd at scale.** `route` is now
  measured across 180 cases (see Routing); `grade`, `generate`, `verify` have been
  exercised interactively against a live key and `/stats` gave real per-stage
  latency/cost. What hasn't run is the *aggregate* `--answers` RAGAS-style judge,
  so answer faithfulness is still single-observation, not measured across the set.
  Routing accuracy is also agreement-with-my-labels, not ground truth — the router
  and the golden intents share an author.
- **`langchain-community` is sunsetting** (BM25/Chroma/FastEmbed wrappers emit a
  deprecation warning). Migrating to `langchain-chroma` + direct `rank_bm25` is
  the obvious next step.
- **Six known retrieval misses at k=5**, and three of them are arguably my labels
  rather than the retriever. "What does IOC mean?" ranks `api/exchange-endpoint`
  above `trading/order-types` — both document IOC and the API page is denser in
  the term. "How do I get a fee discount?" returns `referrals` over
  `trading/fees`, which genuinely documents referral fee discounts. "How do I
  connect my wallet?" returns the connection-*troubleshooting* FAQ rather than
  the onboarding page, which is right if the asker is stuck and wrong if they are
  starting out. "I deposited fiat and nothing arrived" lands on the withdrawal
  FAQ. And `doc-084` ("can a HyperEVM contract use my Hyperliquid spot balance?")
  returns the HyperEVM landing page instead of `interacting-with-hypercore` — the
  only cross-protocol miss and the one that most deserves a real fix.
- **Golden labels have been wrong more often than the retriever.** `/trading/oracle`
  didn't exist; withdrawal fees live in `how-to-start-trading`, not `trading/fees`;
  `doc-060` matched the bare string `hyperevm` and scored a hit on any page in the
  section; `eth-001` expected `how-usde-works` for "What is USDe?" when
  `ethena-overview` opens by defining it. Each was corrected only after checking
  the retrieved page actually answered the question. **The discipline that matters
  here is not relabelling everything that looks defensible after seeing the
  results** — `doc-005` and `doc-052` above are left as misses for exactly that
  reason. An eval you edit to agree with you has stopped being one.
- **Guardrails are English-only** (see 10x section). This is now the largest
  remaining gap in the design.
- **Three protocols is a real test of collision, not of scale.** Ethena supplied
  genuinely colliding vocabulary and the filter handled it 6/6. But all three
  protocols publish `llms.txt` in the same GitBook dialect, so ingestion has never
  faced a site shaped differently, and no two *independent* protocols yet share a
  term the way Aave and Morpho share "health factor".
- **Only one of three ingestion strategies is implemented.** `SourceType.GITBOOK`
  and `SITEMAP` HTML extraction raise `NotImplementedError`; all three whitelisted
  protocols publish `llms.txt`. Aave was evaluated as the third protocol and
  rejected for this PR precisely because its `llms.txt` is an SPA fallback that
  returns HTML — onboarding it means writing the extractor first, which is the
  point at which BeautifulSoup enters the dependency list.
- **`settings.collection` is still `hyperliquid_docs`.** It is an internal Chroma
  collection name holding three protocols now. Renaming it orphans the index and
  forces a full re-crawl for a cosmetic gain, so it stayed.
- **The k=3 *retrieval* ablation predates Ethena** (last measured 0.945 vs 0.954
  on the two-protocol corpus). Its end-to-end cost/quality tradeoff, however, was
  measured live post-Ethena — −39% on grade, at the price of real answer content
  (see "The cost lever, measured"). The offline recall number is the stale one;
  the shipped default (k=5) rests on the live finding.
- **No on-chain tooling beyond market data and explorer reads.** Bridge deposit
  status by tx hash would be the natural next tool, and would make "where is my
  deposit" answerable rather than escalatable.
