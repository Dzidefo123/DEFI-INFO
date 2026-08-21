"""§15's intelligence report, rendered from the investigation record.

Deterministic on purpose. A model writing this section would be free to smooth
over the gaps — and the gaps are the most important thing on the page. The whole
architecture exists so that a reader can tell "we looked and found nothing" apart
from "we did not look", and prose generation is exactly where that distinction
goes to die.

So every heading is filled from state or explicitly marked absent, and the
report's strongest claim is bounded by its weakest input. If nothing was
verified, the Final Assessment says so and stops.
"""

from __future__ import annotations

from src.evidence.confidence import assess
from src.evidence.models import Claim, Evidence, VerificationStatus
from src.intelligence.plan import InvestigationPlan
from src.protocols import english_list, get_protocol, is_known

# Reported when an investigation produced no verified findings at all. Worded to
# foreclose the reading this system must never invite: silence is not safety.
_NO_FINDINGS = (
    "**This investigation produced no findings.** That is not a clean bill of "
    "health — it means the checks below could not be carried out, so nothing was "
    "established either way. Treat the questions asked here as open."
)

# Evidence bullets shown per finding before the list is summarised. A finding
# backed by twelve sections buries the next one under twelve bullets; the count
# is already stated, and the evidence graph holds all of them.
MAX_EVIDENCE_SHOWN = 5

_SEVERITY_LABEL = {
    "critical": "Critical anomaly",
    "high": "High anomaly",
    "elevated": "Elevated",
    "normal": "Normal",
    "unknown": "Not assessed",
}


def _protocol_names(keys) -> str:
    named = [get_protocol(k).name for k in keys if is_known(k)]
    return english_list(named) if named else "not identified"


def _confidence_line(claim: Claim, evidence: list[Evidence]) -> str:
    breakdown = assess(claim, evidence)
    weakest, value = breakdown.weakest_factor()
    return (
        f"**Confidence:** {breakdown.label} ({breakdown.score:.2f}) — "
        f"limited by {weakest.replace('_', ' ')} at {value:.2f}. "
        f"**Verification:** {claim.verification.value.replace('_', ' ')}."
    )


def _findings_section(claims: list[Claim], evidence: list[Evidence]) -> list[str]:
    if not claims:
        return ["_No claims were produced, so there are no findings to report._"]

    # Strongest first: a reader who stops after one heading should have read the
    # best-supported thing the investigation found.
    ranked = sorted(claims, key=lambda c: assess(c, evidence).score, reverse=True)
    out: list[str] = []
    for i, claim in enumerate(ranked, start=1):
        out.append(f"### Finding {i}\n")
        out.append(f"{claim.text}\n")
        out.append(f"{_confidence_line(claim, evidence)}\n")
        supporting = claim.supporting()
        if supporting:
            by_id = {e.evidence_id: e for e in evidence}
            out.append(f"**Evidence ({len(supporting)}):**\n")
            # Capped. A finding backed by twelve sections dumps twelve bullets and
            # buries the next finding under them; the count above already tells a
            # reader how much there is, and the evidence graph holds all of it.
            for link in supporting[:MAX_EVIDENCE_SHOWN]:
                item = by_id.get(link.evidence_id)
                if item is None:
                    # A link naming evidence nobody can produce. Shown, not
                    # hidden: it is a defect in whatever produced the claim.
                    out.append(f"- _missing evidence `{link.evidence_id}`_")
                    continue
                out.append(f"- {item.summary} — {item.source.uri}")
            if len(supporting) > MAX_EVIDENCE_SHOWN:
                out.append(
                    f"- _…and {len(supporting) - MAX_EVIDENCE_SHOWN} more, listed "
                    f"in the evidence graph_"
                )
            out.append("")
    return out


def _statistics_section(risk_signals: list[dict]) -> list[str]:
    if not risk_signals:
        return [
            "_No metrics were scored. No baseline exists yet for the protocols "
            "in scope, so no statistical claim can be made in either direction._"
        ]
    rows = [
        "| Metric | Current | Baseline (median) | z | Signal |",
        "|---|---:|---:|---:|---|",
    ]
    for sig in risk_signals:
        z = sig.get("z")
        base = sig["baseline"]
        # With no history there is no baseline, and `assess_metric` fills the
        # placeholder with the current value. Printing that would show a metric
        # sitting exactly on its baseline — the most reassuring row in the table,
        # produced by the case where nothing was measured at all.
        median = f"{base['median']:,.6g}" if base["n"] else "no history"
        rows.append(
            f"| {sig['metric']} "
            f"| {sig['current_value']:,.6g} "
            f"| {median} "
            f"| {f'{z:+.2f}' if z is not None else 'n/a'} "
            f"| {_SEVERITY_LABEL.get(sig['severity'], sig['severity'])} |"
        )
    return rows


# §10's categories, in the order they carry weight, with the split that matters.
_ESTABLISHED = ("confirmed_incident", "known_vulnerability")
_UNESTABLISHED = ("suspicious_signal", "unverified_claim")
_CLASSIFICATION_LABEL = {
    "confirmed_incident": "Confirmed incident",
    "known_vulnerability": "Known vulnerability",
    "suspicious_signal": "Suspicious signal",
    "unverified_claim": "Unverified claim",
}


def _security_section(evidence: list[Evidence]) -> list[str]:
    """§15's security findings, with §10's four categories kept visibly apart.

    Grouped by classification rather than listed together, because a single list
    is exactly the merge §10 forbids: a rumour printed beside a confirmed
    incident, in the same typeface, is read as a second incident. The
    unestablished group carries its own heading and says outright that nothing
    under it supports a conclusion.
    """
    findings = [e for e in evidence if e.payload.get("classification")]
    if not findings:
        return [
            "_No security findings were on file for the protocols in scope. The "
            "incident registry is curated; an empty result means nothing has been "
            "recorded, not that nothing has happened._"
        ]

    by_class: dict[str, list[Evidence]] = {}
    for item in findings:
        by_class.setdefault(item.payload["classification"], []).append(item)

    lines: list[str] = []
    for group, note in (
        (_ESTABLISHED, None),
        (
            _UNESTABLISHED,
            "_The following are recorded as context only. Nothing here supports a "
            "conclusion, and their presence does not make any finding above more "
            "likely._",
        ),
    ):
        present = [c for c in group if c in by_class]
        if not present:
            continue
        if note:
            lines += ["", note, ""]
        for classification in present:
            lines.append(f"**{_CLASSIFICATION_LABEL[classification]}**")
            lines.append("")
            for item in by_class[classification]:
                status = item.payload.get("status", "unknown")
                title = item.source.title or item.summary
                lines.append(f"- {title} — status {status}, {item.source.uri}")
            lines.append("")
    return lines


def _freshness_line(evidence: list[Evidence]) -> str:
    """When the evidence in this report was actually true.

    An intelligence artifact about on-chain data with no as-of window is
    dangerous: the numbers look current because the document is. The span is
    computed from the evidence's own truth times, not from when it was fetched,
    so a re-read of stale data does not present as fresh.
    """
    if not evidence:
        return "_No evidence was gathered, so this report has no data window._"

    # Measurements know when they were true; documents do not. A docs page has no
    # authored date the system can see, so its `as_of` is when we read it —
    # which is a fact about the crawl, not about the content. Printing the two
    # together would date a year-old page to today.
    measured = [e for e in evidence if e.observed_at is not None]
    read = [e for e in evidence if e.observed_at is None]

    parts: list[str] = []
    if measured:
        stamps = sorted(e.as_of for e in measured)
        window = (
            stamps[-1].date().isoformat()
            if stamps[0].date() == stamps[-1].date()
            else f"{stamps[0].date().isoformat()} to {stamps[-1].date().isoformat()}"
        )
        parts.append(f"Measurements are as of **{window}**.")
    if read:
        latest = max(e.as_of for e in read).date().isoformat()
        parts.append(
            f"{len(read)} documentation excerpt(s) were retrieved on "
            f"**{latest}**; the documentation's own age is not known to this "
            f"system."
        )
    return " ".join(parts)


def _contradictions_section(claims: list[Claim], evidence: list[Evidence]) -> list[str]:
    by_id = {e.evidence_id: e for e in evidence}
    lines: list[str] = []
    for claim in claims:
        against = claim.contradicting()
        if not against:
            continue
        lines.append(f"Against *{claim.text}*:\n")
        for link in against:
            item = by_id.get(link.evidence_id)
            if item is not None:
                lines.append(f"- {item.summary} — {item.source.uri}")
        lines.append("")
    return lines or [
        "_None recorded. Note that no contradicting evidence was searched for "
        "either, so this section is empty rather than clear._"
    ]


def _verified(claims: list[Claim]) -> list[Claim]:
    return [
        c
        for c in claims
        if c.verification
        in (VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED)
    ]


def _final_assessment(
    claims: list[Claim], evidence: list[Evidence], gaps: list[str] | None = None
) -> str:
    """§15's calibrated conclusion — deliberately not the executive summary.

    The two sections had identical text, which wasted one of them. §15 gives them
    separate headings because they do separate jobs: the summary is what a reader
    sees first and says what was found, while this is what they read last and
    says what the investigation is *entitled to conclude* — which is a narrower
    statement, and has to name what stays open.
    """
    verified = _verified(claims)
    if not verified:
        return (
            "Nothing was established. No claim survived verification, so there is "
            "no conclusion to draw in either direction — the questions this "
            "investigation set out to answer remain open."
        )

    scores = sorted((assess(c, evidence).score for c in verified), reverse=True)
    best = scores[0]
    unresolved = len(claims) - len(verified)

    if best >= 0.80:
        stance = "The strongest finding is well supported and can be relied on as stated."
    elif best >= 0.60:
        stance = (
            "The strongest finding is reasonably supported but should be treated as "
            "provisional rather than settled."
        )
    else:
        stance = (
            "No finding is strongly supported. Everything below is a lead worth "
            "following, not a conclusion to act on."
        )

    parts = [f"{stance} (confidence {best:.2f}.)"]
    if unresolved:
        parts.append(
            f"{unresolved} further claim(s) did not survive verification and are "
            f"excluded from this assessment."
        )
    if gaps:
        parts.append(
            f"This conclusion is bounded by what was searched: {english_list(gaps)}. "
            f"Nothing here speaks to those."
        )
    return " ".join(parts)


def _assessment(
    claims: list[Claim], evidence: list[Evidence], gaps: list[str] | None = None
) -> str:
    """The summary, bounded by what the investigation actually covered.

    Confidence is a property of a claim, not of an investigation. Leading with
    "high confidence" while a planned stage produced nothing invites the reading
    this whole system exists to prevent: an investigation that measured no
    on-chain data and searched an empty incident registry can still verify a
    claim about what the documentation says, at 0.98 — and a reader skimming the
    first line would take that as an answer about risk.

    So coverage comes first when anything is missing. A confident finding about a
    question nobody asked is worth less than an honest account of what was not
    looked at.
    """
    verified = [
        c
        for c in claims
        if c.verification
        in (VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED)
    ]
    if not verified:
        return _NO_FINDINGS

    scores = [assess(c, evidence).score for c in verified]
    best = max(scores)
    label = "high" if best >= 0.80 else "moderate" if best >= 0.60 else "low"
    summary = (
        f"{len(verified)} of {len(claims)} claims survived verification. "
        f"The best-supported finding carries {label} confidence ({best:.2f}). "
        f"Findings below that level are reported as leads, not conclusions."
    )

    if gaps:
        return (
            f"**Partial investigation.** {english_list(gaps)} — so this assessment "
            f"covers only what was searched, and is not an answer about the parts "
            f"that were not. Confidence figures below describe individual findings, "
            f"not the investigation as a whole.\n\n{summary}"
        )
    return summary


def _provenance_section(graph: dict) -> list[str]:
    """§14.3: how many separate lines of evidence the findings actually rest on.

    The number a reader needs and a flat list cannot give. Two claims each
    carrying three citations look like two corroborated findings; if all six
    resolve to one page, they are one finding stated twice. Only the graph's
    convergence makes that visible, which is the reason it exists.
    """
    if not graph or not graph.get("nodes"):
        return ["_No evidence graph was built for this investigation._"]

    groups = graph.get("independent_groups") or []
    shared = graph.get("shared_sources") or []
    counts: dict[str, int] = {}
    for node in graph["nodes"]:
        counts[node["type"]] = counts.get(node["type"], 0) + 1
    n_claims = counts.get("claim", 0)

    lines = [
        f"{n_claims} claim(s) drawn from {counts.get('evidence', 0)} piece(s) of "
        f"evidence across {counts.get('source', 0)} distinct source(s).",
        "",
    ]

    if groups:
        lines += [
            f"**Independent lines of evidence: {len(groups)}.** "
            + (
                "Each finding rests on its own sources."
                if len(groups) == n_claims
                else "Fewer than the number of claims, so some findings are not "
                "independent confirmations of each other."
            ),
            "",
        ]

    for entry in shared:
        lines.append(
            f"- {len(entry['claim_ids'])} claims share the source "
            f"*{entry['label']}* — they corroborate each other only as far as "
            f"that one source is correct."
        )
    if shared:
        lines.append("")

    if mermaid := graph.get("mermaid"):
        lines += [
            "<details><summary>Evidence graph</summary>",
            "",
            "```mermaid",
            mermaid,
            "```",
            "",
            "</details>",
        ]
    return lines


def coverage_gaps(
    plan: InvestigationPlan, risk_signals: list[dict], security_results: dict
) -> list[str]:
    """Stages the plan called for that produced nothing usable.

    Derived from the PLAN, not from what happened, which is why the plan is
    recorded before execution. "The risk engine found nothing unusual" and "the
    risk engine had nothing to look at" are different sentences, and only the
    plan knows which one applies.
    """
    gaps: list[str] = []

    if plan.risk_engine:
        scored = [s for s in risk_signals if s.get("severity") != "unknown"]
        if not scored:
            gaps.append("no on-chain metric could be scored against a baseline")

    if "security_agent" in plan.agents:
        by_class = (security_results or {}).get("by_classification") or {}
        if not sum(by_class.values()):
            gaps.append("no security findings were on file to review")

    return gaps


def render_report(
    plan: InvestigationPlan,
    claims: list[Claim],
    evidence: list[Evidence],
    risk_signals: list[dict],
    verification: dict,
    limitations: list[str],
    gaps: list[str] | None = None,
    graph: dict | None = None,
) -> str:
    """Assemble the §15 artifact. Pure: same record in, same bytes out."""
    ran = english_list([a.replace("_", " ") for a in plan.agents]) or "none"

    parts: list[str] = [
        "# Intelligence Assessment\n",
        "## Executive Summary\n",
        _assessment(claims, evidence, gaps) + "\n",
        "## Investigation Scope\n",
        f"**Question:** {plan.question}\n",
        f"**Classification:** {plan.query_type.replace('_', ' ')}\n",
        f"**Agents dispatched:** {ran}\n",
        "## Protocol / Entity\n",
        f"{_protocol_names(plan.protocols)}\n",
        f"{_freshness_line(evidence)}\n",
        "## Key Findings\n",
        *_findings_section(claims, evidence),
        "\n## Statistical Findings\n",
        *_statistics_section(risk_signals),
        "\n## Security Findings\n",
        *_security_section(evidence),
        "\n## Contradictory Evidence\n",
        *_contradictions_section(claims, evidence),
        "\n## Provenance\n",
        *_provenance_section(graph or {}),
        "\n## Limitations\n",
    ]

    # Limitations is the section that must never be empty when something was
    # skipped, because everything above reads as complete unless this says
    # otherwise.
    notes = list(plan.notes) + list(limitations)
    if verification and not verification.get("claims_examined"):
        notes.append("Verification ran but had no claims to examine.")
    parts.extend(f"- {n}" for n in notes) if notes else parts.append(
        "_None recorded._"
    )

    parts += [
        "\n## Final Assessment\n",
        _final_assessment(claims, evidence, gaps) + "\n",
        "---\n",
        "_Generated deterministically from the investigation record. "
        "Confidence scores use an uncalibrated transparent model; "
        "see `src/evidence/confidence.py`._",
    ]
    return "\n".join(parts)


def report_payload(
    plan: InvestigationPlan,
    claims: list[Claim],
    evidence: list[Evidence],
    risk_signals: list[dict],
    verification: dict,
    limitations: list[str],
    gaps: list[str] | None = None,
    graph: dict | None = None,
) -> dict:
    """The same assessment as data rather than prose.

    §15 calls the output a structured intelligence artifact, and markdown alone is
    only half of that — it can be read but not queried, diffed by field, or
    scored. The evaluation work in the next phase needs to ask "what did this
    investigation conclude, at what confidence, resting on how many independent
    sources", and parsing that back out of headings would be absurd.

    Rendered from the same record as the markdown, so the two cannot disagree.
    """
    graph = graph or {}
    by_id = {e.evidence_id: e for e in evidence}

    findings = []
    for claim in sorted(claims, key=lambda c: assess(c, evidence).score, reverse=True):
        breakdown = assess(claim, evidence)
        weakest, value = breakdown.weakest_factor()
        findings.append(
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "agent": claim.agent.value,
                "verification": claim.verification.value,
                "confidence": breakdown.score,
                "confidence_band": breakdown.label,
                "limiting_factor": {"name": weakest, "value": value},
                "supporting_sources": sorted(
                    {
                        by_id[l.evidence_id].source.uri
                        for l in claim.supporting()
                        if l.evidence_id in by_id
                    }
                ),
                "contradicting": len(claim.contradicting()),
            }
        )

    return {
        "question": plan.question,
        "classification": plan.query_type,
        "protocols": list(plan.protocols),
        "agents_dispatched": list(plan.agents),
        "findings": findings,
        "risk_signals": risk_signals,
        "security": {
            "by_classification": _security_counts(evidence),
        },
        "provenance": {
            "evidence_count": len(evidence),
            "distinct_sources": len({e.source.uri for e in evidence}),
            "independent_lines": len(graph.get("independent_groups") or []),
            "shared_sources": len(graph.get("shared_sources") or []),
        },
        "evidence_window": _evidence_window(evidence),
        "coverage_gaps": list(gaps or []),
        "limitations": list(plan.notes) + list(limitations),
        "verification": verification,
        "calibration": "uncalibrated; see src/evidence/confidence.py",
    }


def _security_counts(evidence: list[Evidence]) -> dict[str, int]:
    """Per category, never as a total — a single count is §10's forbidden merge."""
    counts = {c: 0 for c in (*_ESTABLISHED, *_UNESTABLISHED)}
    for item in evidence:
        if (c := item.payload.get("classification")) in counts:
            counts[c] += 1
    return counts


def _evidence_window(evidence: list[Evidence]) -> dict[str, str | None]:
    if not evidence:
        return {"from": None, "to": None}
    stamps = sorted(e.as_of for e in evidence)
    return {"from": stamps[0].isoformat(), "to": stamps[-1].isoformat()}
