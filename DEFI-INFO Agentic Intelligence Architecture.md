# DEFI-INFO: Agentic Intelligence Architecture

## Building an Evidence-Driven DeFi Intelligence and Risk Platform on Top of the Existing CX Agent

**Status:** Proposed Architecture  
**Project:** DEFI-INFO  
**Foundation:** Existing Multi-Protocol Crypto CX Agent  
**Core Technologies:** Python, LangGraph, Hybrid RAG, BM25, Vector Search, Cross-Encoder Reranking, Blockchain Data APIs, Statistical Analysis, Machine Learning, Knowledge Graphs

---

# 1. Executive Summary

DEFI-INFO currently provides a strong foundation as a multi-protocol Crypto Customer Experience (CX) agent capable of answering questions using protocol-aware retrieval, hybrid RAG, BM25, semantic search, reranking, live data, and deterministic safety controls.

The next evolution of the system is to move beyond a question-answering architecture.

The proposed architecture transforms DEFI-INFO from:

> **A system that retrieves information and answers questions about DeFi protocols**

into:

> **An evidence-driven, multi-agent intelligence system capable of researching, investigating, analyzing, verifying, and explaining financial and cybersecurity risks within decentralized systems.**

The system will use a central **Intelligence Manager Agent** to coordinate specialized agents. Each agent investigates a different dimension of a user request:

- **Research Agent** investigates protocol documentation, governance, historical events, and knowledge sources.
- **Blockchain Agent** investigates on-chain activity and quantitative protocol behavior.
- **Security Agent** investigates known vulnerabilities, incidents, exploits, suspicious activity, and threat intelligence.
- **Statistical/ML Risk Engine** analyzes abnormal behavior and produces quantitative risk signals.
- **Verification Agent** challenges and validates conclusions before they are presented.
- **Evidence Graph** stores the relationships between claims, evidence, entities, events, and conclusions.
- **Intelligence Report Generator** produces a structured, evidence-backed response.

The fundamental design principle is:

> **The system should not simply answer questions. It should investigate claims and construct evidence-backed conclusions.**

---

# 2. The Architectural Vision

The proposed architecture is based on the following intelligence pipeline:

```text
                    ┌───────────────┐
                    │     USER      │
                    └───────┬───────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Intelligence     │
                  │ Manager Agent    │
                  └────────┬─────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       Research        Blockchain      Security
        Agent            Agent           Agent
            │              │              │
            ▼              ▼              ▼
       ┌────────┐     ┌──────────┐   ┌──────────┐
       │ RAG    │     │On-chain  │   │Threat    │
       │ BM25   │     │ analytics│   │Intel     │
       └────────┘     └──────────┘   └──────────┘
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                 ┌──────────────────┐
                 │ Statistical / ML │
                 │ Risk Engine      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Verification     │
                 │ Agent            │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Evidence Graph   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Intelligence     │
                 │ Report           │
                 └──────────────────┘
```

This should not be implemented as a completely separate system.

The recommended approach is to preserve the existing CX Agent and evolve it incrementally.

The architecture should therefore follow this progression:

```text
V1
│
├── Crypto CX Agent
│   ├── User Query
│   ├── Protocol Routing
│   ├── Hybrid Retrieval
│   ├── BM25
│   ├── Vector Search
│   ├── Reranking
│   └── Answer Generation
│
▼
V2
│
├── Intelligence Manager
│   ├── CX / Research Mode
│   ├── Blockchain Investigation
│   └── Security Investigation
│
▼
V3
│
├── Multi-Agent Investigation
│   ├── Research Agent
│   ├── Blockchain Agent
│   └── Security Agent
│
▼
V4
│
├── Statistical Risk Intelligence
│   ├── Anomaly Detection
│   ├── Time-Series Analysis
│   ├── Risk Scoring
│   └── ML Models
│
▼
V5
│
├── Evidence-Driven Verification
│   ├── Claim Extraction
│   ├── Evidence Matching
│   ├── Contradiction Detection
│   └── Confidence Assessment
│
▼
V6
│
└── Evidence Graph + Intelligence Reports
```

---

# 3. Design Principles

The architecture should be built around several principles.

## 3.1 Evidence Before Conclusions

Every significant conclusion should be traceable.

The system should be able to answer:

- What is the claim?
- What evidence supports the claim?
- Where did the evidence come from?
- When was the evidence collected?
- Which agent produced the evidence?
- Are there conflicting sources?
- How confident is the system?
- What assumptions were made?

The final system should move toward the following model:

```text
CLAIM
  │
  ├── Supporting Evidence
  │      ├── Source
  │      ├── Timestamp
  │      ├── Agent
  │      └── Confidence
  │
  ├── Contradicting Evidence
  │      ├── Source
  │      ├── Timestamp
  │      └── Confidence
  │
  └── Final Assessment
         ├── Confidence
         ├── Reasoning Summary
         └── Limitations
```

---

## 3.2 Agents Should Have Specialized Responsibilities

Avoid creating multiple generic LLM agents with overlapping responsibilities.

Each agent should have:

- a clearly defined responsibility
- specific tools
- specific data sources
- structured output
- measurable success criteria

For example:

```text
Research Agent
    ↓
Can investigate:
    - Protocol documentation
    - Governance proposals
    - Historical events
    - Whitepapers
    - Technical documentation

Cannot:
    - Perform unrestricted security conclusions
    - Execute blockchain transactions
    - Override verification results
```

This separation makes the system easier to evaluate and debug.

---

## 3.3 Deterministic Systems Where Possible

The existing DEFI-INFO system already contains deterministic guardrails.

That philosophy should continue.

Not everything should be delegated to an LLM.

Use deterministic systems for:

- protocol identification
- entity validation
- numerical calculations
- statistical analysis
- risk thresholds
- source validation
- schema validation
- evidence IDs
- confidence calculations where possible

The LLM should primarily handle:

- interpretation
- planning
- synthesis
- explanation
- investigation strategy

A useful principle is:

> **Use AI for ambiguity. Use deterministic systems for facts and calculations.**

---

# 4. High-Level System Architecture

The complete system can be divided into six architectural layers.

```text
┌─────────────────────────────────────────────┐
│                 USER LAYER                  │
│  Web App / API / Chat Interface / Dashboard│
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│              ORCHESTRATION LAYER            │
│                                             │
│        Intelligence Manager Agent           │
│                                             │
│     Intent │ Planning │ Routing │ State     │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│             SPECIALIST AGENT LAYER          │
│                                             │
│  Research │ Blockchain │ Security │ Risk    │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│                  DATA LAYER                 │
│                                             │
│ RAG │ Documents │ Blockchain │ Threat Intel │
│ Market Data │ Historical Metrics            │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│             INTELLIGENCE LAYER              │
│                                             │
│ Statistical Analysis │ ML │ Anomaly Models  │
│ Risk Scoring │ Correlation                  │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│             EVIDENCE & OUTPUT LAYER         │
│                                             │
│ Verification │ Evidence Graph │ Reporting   │
└─────────────────────────────────────────────┘
```

---

# 5. Intelligence Manager Agent

The Intelligence Manager is the central orchestrator.

It should not attempt to answer every question directly.

Its primary responsibility is to understand:

1. What does the user want?
2. Is this a simple information request or an investigation?
3. Which agents are required?
4. Should agents execute sequentially or in parallel?
5. What information is missing?
6. When is the evidence sufficient?
7. Does the investigation require verification?

---

## 5.1 Query Classification

The Intelligence Manager should first classify the request.

Example categories:

```python
class QueryType(str, Enum):
    CX = "cx"
    RESEARCH = "research"
    BLOCKCHAIN_ANALYSIS = "blockchain_analysis"
    SECURITY_ANALYSIS = "security_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    FULL_INVESTIGATION = "full_investigation"
```

Example:

### Query

> What is Aave?

Classification:

```text
CX
```

Execution:

```text
User
 ↓
Protocol Router
 ↓
RAG
 ↓
Answer
```

---

### Query

> Has there been any unusual activity involving this protocol over the last 30 days?

Classification:

```text
BLOCKCHAIN_ANALYSIS
```

Execution:

```text
Blockchain Agent
        ↓
On-chain Data
        ↓
Statistical Analysis
        ↓
Verification
        ↓
Response
```

---

### Query

> Is Protocol X currently showing any significant security or financial risk?

Classification:

```text
FULL_INVESTIGATION
```

Execution:

```text
Research Agent ────────┐
                       │
Blockchain Agent ──────┼──► Risk Engine
                       │         │
Security Agent ────────┘         ▼
                           Verification Agent
                                   │
                                   ▼
                             Intelligence Report
```

---

# 6. LangGraph Orchestration

LangGraph should remain the core orchestration framework.

The architecture can be represented as a state machine.

A simplified version:

```text
START
  │
  ▼
Query Classification
  │
  ▼
Investigation Planner
  │
  ├───────────────┐
  │               │
  ▼               ▼
Simple Query    Investigation
  │               │
  ▼               ├────────► Research Agent
CX Agent          │
  │                ├────────► Blockchain Agent
  ▼                │
END                 ├────────► Security Agent
                   │
                   ▼
              Risk Engine
                   │
                   ▼
              Verification
                   │
                   ▼
              Evidence Graph
                   │
                   ▼
                  END
```

---

# 7. Proposed Graph State

The LangGraph state should be expanded beyond simple messages.

Example:

```python
from typing import TypedDict, List, Dict, Any

class IntelligenceState(TypedDict):
    query: str
    query_type: str
    protocol: str | None

    investigation_plan: Dict[str, Any]

    research_results: List[Dict[str, Any]]
    blockchain_results: List[Dict[str, Any]]
    security_results: List[Dict[str, Any]]

    statistical_results: Dict[str, Any]
    risk_assessment: Dict[str, Any]

    claims: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    contradictions: List[Dict[str, Any]]

    verification_results: Dict[str, Any]

    final_report: Dict[str, Any]

    errors: List[str]
```

This state should be treated as the investigation record.

Every agent writes structured information into the state.

Avoid allowing agents to communicate only through natural language.

Prefer:

```json
{
  "claim": "Protocol X experienced unusual liquidity withdrawals.",
  "evidence": [
    {
      "type": "on_chain_metric",
      "metric": "liquidity_outflow",
      "value": 12500000,
      "baseline": 2300000,
      "z_score": 4.2,
      "timestamp": "2026-08-20"
    }
  ],
  "confidence": 0.91
}
```

rather than:

```text
I found something suspicious. It looks like liquidity decreased significantly.
```

---

# 8. Research Agent

The Research Agent is responsible for understanding the documented and historical context surrounding a protocol or investigation.

It should build upon the existing DEFI-INFO retrieval system.

---

## 8.1 Existing Capabilities to Reuse

The current CX Agent can remain the foundation.

Reuse:

- protocol-aware routing
- BM25
- vector search
- hybrid retrieval
- cross-encoder reranking
- protocol filtering
- document metadata
- golden evaluation datasets
- adversarial testing

The Research Agent should therefore be an evolution of the existing retrieval pipeline.

---

## 8.2 Research Agent Workflow

```text
Research Question
       │
       ▼
Protocol Identification
       │
       ▼
Query Decomposition
       │
       ├── Documentation Query
       ├── Governance Query
       ├── Historical Query
       └── Architecture Query
       │
       ▼
Hybrid Retrieval
       │
       ├── BM25
       └── Vector Search
       │
       ▼
Fusion
       │
       ▼
Cross-Encoder Reranking
       │
       ▼
Evidence Extraction
       │
       ▼
Structured Research Result
```

---

## 8.3 Research Agent Output

```python
{
    "agent": "research_agent",
    "claims": [
        {
            "claim_id": "claim_001",
            "claim": "Protocol X uses overcollateralized lending.",
            "confidence": 0.95,
            "sources": [
                {
                    "source_id": "doc_001",
                    "document": "Protocol Documentation",
                    "chunk_id": "chunk_23"
                }
            ]
        }
    ]
}
```

---

# 9. Blockchain Intelligence Agent

The Blockchain Agent is responsible for collecting and analyzing on-chain evidence.

Its job is not simply to call blockchain APIs.

It should investigate questions such as:

- Has TVL changed abnormally?
- Have large withdrawals occurred?
- Has transaction volume changed significantly?
- Has wallet concentration increased?
- Are there abnormal token movements?
- Are there unusual contract interactions?
- Has protocol activity deviated from historical behavior?

---

## 9.1 Architecture

```text
Blockchain Agent
       │
       ▼
Question Decomposition
       │
       ├── TVL
       ├── Transactions
       ├── Wallets
       ├── Token Flows
       └── Smart Contracts
       │
       ▼
Data Collection
       │
       ▼
Data Validation
       │
       ▼
Feature Engineering
       │
       ▼
Statistical / ML Engine
       │
       ▼
Blockchain Evidence
```

---

## 9.2 Example Features

Potential features include:

```text
TVL_CHANGE_1H
TVL_CHANGE_24H
TVL_CHANGE_7D

TRANSACTION_COUNT
TRANSACTION_VOLUME

UNIQUE_ACTIVE_WALLETS

LARGE_TRANSACTION_COUNT

WHALE_CONCENTRATION

INFLOW_OUTFLOW_RATIO

CONTRACT_INTERACTION_RATE

TOKEN_PRICE_VOLATILITY

LIQUIDITY_CHANGE_RATE
```

The features should be stored historically.

This is important because anomaly detection requires a baseline.

---

# 10. Security Intelligence Agent

The Security Agent investigates security-related information.

Potential sources may include:

- known vulnerabilities
- public incident databases
- security advisories
- protocol disclosures
- exploit reports
- threat intelligence feeds
- smart contract audit information

The agent should distinguish between:

```text
Confirmed Incident
```

```text
Known Vulnerability
```

```text
Suspicious Signal
```

```text
Unverified Claim
```

These categories should never be merged.

---

## 10.1 Security Investigation Flow

```text
Security Question
       │
       ▼
Entity Identification
       │
       ▼
Threat Intelligence Retrieval
       │
       ├── Known Incidents
       ├── Vulnerabilities
       ├── Exploit Reports
       └── Audit Findings
       │
       ▼
Evidence Extraction
       │
       ▼
Confidence Classification
```

---

# 11. Statistical and Machine Learning Risk Engine

This is one of the areas where the architecture can become genuinely differentiated.

The ML Risk Engine should not be controlled by an LLM.

It should be a separate analytical system.

```text
Agent Evidence
      +
Historical Data
      │
      ▼
Feature Engineering
      │
      ▼
Statistical Analysis
      │
      ├── Z-Score
      ├── IQR
      ├── Change Point Detection
      └── Time-Series Analysis
      │
      ▼
Machine Learning
      │
      ├── Isolation Forest
      ├── One-Class Models
      ├── Clustering
      └── Temporal Models
      │
      ▼
Risk Signals
```

---

## 11.1 Start With Statistics Before Complex ML

The first implementation should use explainable statistical techniques.

Example:

```python
def calculate_z_score(value, mean, std):
    return (value - mean) / std
```

Example interpretation:

```text
Metric: Daily Liquidity Outflow

Current Value:      $12.5M
Historical Average: $2.3M
Standard Deviation: $1.8M

Z-Score: 5.67

Assessment:
Highly abnormal relative to historical behavior.
```

This provides a clear explanation.

---

## 11.2 Risk Signal Schema

```json
{
  "signal_id": "risk_001",
  "metric": "liquidity_outflow",
  "current_value": 12500000,
  "baseline": 2300000,
  "z_score": 5.67,
  "anomaly": true,
  "severity": "high",
  "confidence": 0.92
}
```

The LLM should explain this signal.

The LLM should not calculate it.

---

# 12. Risk Assessment Layer

The Risk Assessment Layer combines signals from multiple domains.

Example:

```text
Research Signals
       │
       ▼
Blockchain Signals
       │
       ▼
Security Signals
       │
       ▼
Statistical Signals
       │
       ▼
Correlation
       │
       ▼
Risk Assessment
```

A simple conceptual model could initially be:

```text
TOTAL_RISK =
    Financial Risk
  + Security Risk
  + Operational Risk
  + Anomaly Risk
```

However, the system should preserve individual dimensions rather than producing only one opaque number.

Example:

```json
{
  "risk_assessment": {
    "financial_risk": 0.42,
    "security_risk": 0.71,
    "operational_risk": 0.28,
    "anomaly_risk": 0.83,
    "overall_assessment": "elevated"
  }
}
```

---

# 13. Verification Agent

The Verification Agent is one of the most important components.

Its role is not to generate new information.

Its role is to challenge the findings of other agents.

The Verification Agent should ask:

> Is this claim supported?

> Is the evidence directly related to the claim?

> Is the source reliable?

> Is there contradictory evidence?

> Is the conclusion stronger than the available evidence?

---

## 13.1 Verification Workflow

```text
Claim
  │
  ▼
Find Supporting Evidence
  │
  ▼
Find Contradictory Evidence
  │
  ▼
Check Source Quality
  │
  ▼
Check Temporal Relevance
  │
  ▼
Check Numerical Consistency
  │
  ▼
Assign Verification Status
```

Possible statuses:

```text
VERIFIED
PARTIALLY_VERIFIED
INSUFFICIENT_EVIDENCE
CONTRADICTED
```

---

## 13.2 Example

```json
{
  "claim": "Protocol X experienced abnormal withdrawals.",
  "verification_status": "VERIFIED",
  "supporting_evidence": 4,
  "contradicting_evidence": 0,
  "confidence": 0.94
}
```

Another example:

```json
{
  "claim": "The abnormal withdrawals were caused by a security exploit.",
  "verification_status": "INSUFFICIENT_EVIDENCE",
  "supporting_evidence": 1,
  "contradicting_evidence": 2,
  "confidence": 0.31
}
```

This distinction is critical.

The system may detect abnormal activity without claiming to know the cause.

---

# 14. Evidence Graph

The Evidence Graph becomes the memory and reasoning structure of the intelligence platform.

It should not replace vector search.

Instead:

```text
Vector Database
        +
Knowledge / Evidence Graph
        =
Hybrid Intelligence System
```

---

## 14.1 Potential Graph Entities

```text
Protocol
Token
Blockchain
Smart Contract
Wallet
Transaction
Event
Incident
Vulnerability
Document
Claim
Evidence
Risk Signal
Agent
Report
```

---

## 14.2 Relationships

```text
Protocol
    │
    ├── DEPENDS_ON ─────────► Protocol
    │
    ├── DEPLOYS ────────────► Smart Contract
    │
    ├── HAS_TOKEN ──────────► Token
    │
    ├── EXPERIENCED ────────► Incident
    │
    ├── HAS_RISK_SIGNAL ────► Risk Signal
    │
    └── DOCUMENTED_BY ──────► Document


Claim
    │
    ├── SUPPORTED_BY ───────► Evidence
    │
    ├── CONTRADICTED_BY ────► Evidence
    │
    ├── GENERATED_BY ───────► Agent
    │
    └── RELATED_TO ─────────► Protocol
```

---

## 14.3 Why the Evidence Graph Matters

Imagine the user asks:

> Why did the system classify this protocol as elevated risk?

The system should traverse:

```text
Risk Assessment
      │
      ▼
Risk Signal
      │
      ▼
Claim
      │
      ▼
Supporting Evidence
      │
      ├── Blockchain Data
      ├── Security Report
      ├── Protocol Documentation
      └── Statistical Analysis
```

The result is explainable.

---

# 15. Intelligence Report

The final report should not simply be a chat response.

It should be a structured intelligence artifact.

Example:

# Intelligence Assessment

## Executive Summary

A concise description of the investigation and its findings.

## Investigation Scope

What question was investigated?

## Protocol / Entity

The protocol, blockchain, wallet, token, or entity under investigation.

## Key Findings

### Finding 1

Description.

**Confidence:** High

**Verification:** Verified

### Finding 2

Description.

**Confidence:** Medium

**Verification:** Partially Verified

## Statistical Findings

| Metric | Current | Historical Baseline | Signal |
|---|---:|---:|---|
| TVL | ... | ... | Normal |
| Outflow | ... | ... | High Anomaly |
| Transactions | ... | ... | Elevated |

## Security Findings

Known incidents, vulnerabilities, or security evidence.

## Contradictory Evidence

Information that does not support the primary conclusion.

## Limitations

What the system could not determine.

## Final Assessment

A calibrated conclusion.

---

# 16. Confidence Model

Avoid allowing a single LLM-generated confidence score to control the system.

Confidence should be based on multiple dimensions.

Example:

```text
CONFIDENCE =
    Evidence Quality
  × Evidence Agreement
  × Source Reliability
  × Temporal Relevance
  × Verification Score
```

Conceptually:

```python
confidence = (
    evidence_quality
    * evidence_agreement
    * source_reliability
    * temporal_relevance
    * verification_score
)
```

The exact mathematical implementation can evolve later.

Initially, use a transparent scoring model.

---

# 17. Recommended Repository Architecture

The existing repository can evolve toward:

```text
DEFI-INFO/
│
├── app/
│   ├── api/
│   │   ├── routes/
│   │   └── schemas/
│   │
│   ├── agents/
│   │   ├── intelligence_manager.py
│   │   ├── cx_agent.py
│   │   ├── research_agent.py
│   │   ├── blockchain_agent.py
│   │   ├── security_agent.py
│   │   └── verification_agent.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   ├── workflow.py
│   │   ├── nodes.py
│   │   └── routing.py
│   │
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── vector_search.py
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   │
│   ├── blockchain/
│   │   ├── collectors/
│   │   ├── features/
│   │   └── analytics/
│   │
│   ├── security/
│   │   ├── threat_intel.py
│   │   ├── incidents.py
│   │   └── vulnerabilities.py
│   │
│   ├── risk/
│   │   ├── statistical_models.py
│   │   ├── anomaly_detection.py
│   │   ├── risk_scoring.py
│   │   └── feature_engineering.py
│   │
│   ├── evidence/
│   │   ├── models.py
│   │   ├── extraction.py
│   │   ├── validation.py
│   │   └── graph.py
│   │
│   ├── reports/
│   │   ├── generator.py
│   │   └── templates.py
│   │
│   └── evaluation/
│       ├── golden_dataset/
│       ├── adversarial/
│       ├── benchmarks/
│       └── ablations/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── tests/
│   ├── agents/
│   ├── retrieval/
│   ├── risk/
│   ├── evidence/
│   └── integration/
│
├── notebooks/
│   ├── exploration/
│   ├── anomaly_detection/
│   └── experiments/
│
├── docs/
│   ├── architecture/
│   ├── research/
│   └── evaluation/
│
└── README.md
```

---

# 18. Implementation Roadmap

The architecture should not be implemented all at once.

## Phase 1 — Intelligence Manager

**Goal:** Add routing above the existing CX Agent.

```text
User
 ↓
Intelligence Manager
 ↓
CX Query? ───────► Existing CX Agent
 │
 └── Investigation? ───────► Investigation Pipeline
```

Deliverables:

- query classification
- investigation mode
- LangGraph routing
- updated state object

---

## Phase 2 — Research Agent

**Goal:** Convert the existing RAG system into a specialized research agent.

Deliverables:

- query decomposition
- multi-query retrieval
- evidence extraction
- structured research output

This phase should reuse as much of the existing CX infrastructure as possible.

---

## Phase 3 — Blockchain Intelligence

**Goal:** Introduce structured on-chain analysis.

Deliverables:

- data connectors
- historical data storage
- feature engineering
- blockchain evidence schema

Start with a limited number of metrics.

Do not attempt to support every blockchain and protocol immediately.

---

## Phase 4 — Statistical Risk Engine

**Goal:** Detect abnormal behavior.

Start with:

- Z-scores
- rolling averages
- rolling standard deviations
- IQR
- percentage change
- change-point detection

Then experiment with:

- Isolation Forest
- One-Class SVM
- clustering
- temporal anomaly detection

Every model should be benchmarked.

---

## Phase 5 — Security Agent

**Goal:** Correlate protocol and blockchain information with security intelligence.

Deliverables:

- incident retrieval
- vulnerability information
- security event classification
- source confidence

---

## Phase 6 — Verification Agent

**Goal:** Reduce unsupported conclusions.

Deliverables:

- claim extraction
- evidence matching
- contradiction detection
- verification status

---

## Phase 7 — Evidence Graph

**Goal:** Make conclusions traceable.

Start with a simple graph model before introducing a full graph database.

Initially:

```text
Python data models
        ↓
NetworkX or equivalent
        ↓
Graph visualization
```

Later:

```text
Graph Database
```

---

# 19. Evaluation Framework

The existing DEFI-INFO evaluation work should be extended rather than replaced.

Evaluation should happen at multiple levels.

## Retrieval Evaluation

Measure:

```text
Recall@K
Precision@K
MRR
NDCG
```

---

## Agent Evaluation

Measure:

```text
Correct Agent Selection
Correct Tool Selection
Task Completion Rate
Invalid Tool Calls
Investigation Efficiency
```

---

## Statistical Evaluation

For anomaly detection:

```text
Precision
Recall
F1 Score
ROC-AUC
False Positive Rate
False Negative Rate
Detection Latency
```

---

## Verification Evaluation

Measure:

```text
Claim Accuracy
Evidence Coverage
Unsupported Claim Rate
Contradiction Detection Rate
False Verification Rate
```

---

# 20. The Long-Term Research Opportunity

This architecture can eventually become a research platform.

A possible central research question is:

> **Can evidence-driven multi-agent architectures combining hybrid retrieval, blockchain analytics, statistical anomaly detection, and agentic verification improve the reliability of automated intelligence systems?**

Potential experiments could compare:

```text
System A
LLM Only

vs

System B
Standard RAG

vs

System C
Hybrid RAG

vs

System D
Hybrid RAG + Agentic Research

vs

System E
Hybrid RAG + Agents + Statistical Analysis

vs

System F
Hybrid RAG + Agents + Statistical Analysis + Verification
```

Potential research metrics:

```text
Factual Accuracy
Evidence Coverage
Hallucination Rate
Unsupported Claims
Adversarial Robustness
Investigation Quality
Risk Detection Performance
```

This transforms DEFI-INFO from only an application into an experimental research platform.

---

# 21. Final Architecture Philosophy

The most important architectural principle is that DEFI-INFO should preserve the strengths of the existing CX Agent.

The goal is not:

> Replace the existing system with a complicated multi-agent architecture.

The goal is:

> Add intelligence capabilities only when the question requires them.

A simple user question should remain simple:

```text
User Question
      ↓
Existing CX Agent
      ↓
Answer
```

A complex investigation should activate the larger architecture:

```text
User Investigation
      ↓
Intelligence Manager
      ↓
Investigation Plan
      ↓
┌───────────────────────────────────────┐
│ Research Agent                        │
│ Blockchain Agent                      │
│ Security Agent                        │
└───────────────────────────────────────┘
      ↓
Statistical / ML Risk Engine
      ↓
Verification Agent
      ↓
Evidence Graph
      ↓
Intelligence Report
```

The final objective is not to build an application with the maximum number of agents.

The objective is to build a system capable of answering:

> **What do we know?**

> **What evidence supports it?**

> **What contradicts it?**

> **What is statistically unusual?**

> **What remains uncertain?**

> **How confident should we be in the conclusion?**

That is the transition from a **Crypto CX Agent** to an **Evidence-Driven Agentic Intelligence Platform**.

---

# 22. Recommended Immediate Next Step

The first implementation should focus on the smallest architectural change with the highest strategic value:

```text
CURRENT DEFI-INFO
        │
        ▼
Intelligence Manager
        │
        ├── CX Mode
        │       │
        │       └── Existing DEFI-INFO Pipeline
        │
        └── Investigation Mode
                │
                ├── Research Agent
                └── Blockchain Investigation Prototype
```

Do not build the Evidence Graph, Security Agent, ML platform, and reporting infrastructure at the same time.

Build the architecture incrementally.

The recommended first milestone is:

> **A LangGraph-based Intelligence Manager that decides whether a request should use the existing CX pipeline or initiate a structured investigation, with Research and Blockchain agents producing structured evidence.**

Once that foundation works and is evaluated, the Statistical Risk Engine and Verification Agent can be introduced.

This approach allows DEFI-INFO to remain useful as a production-style CX application while gradually evolving into a significantly more advanced **Agentic Financial and Cyber Risk Intelligence Platform**.