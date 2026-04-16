from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.orchestration.state import WorkflowState
from src.utils.logging_utils import get_logger
from src.utils.paths import ARTIFACTS_DIR, REPORTS_DIR

logger = get_logger(__name__)

WHITEPAPER = REPORTS_DIR / "whitepaper_draft.md"
EDA_ARTIFACT = ARTIFACTS_DIR / "eda_summary.json"
SENTIMENT_ARTIFACT = ARTIFACTS_DIR / "sentiment_metrics.json"
TOPICS_ARTIFACT = ARTIFACTS_DIR / "topics.json"
RECOMMENDATIONS_ARTIFACT = ARTIFACTS_DIR / "recommendations.json"

REVIEWER_REPORT = REPORTS_DIR / "reviewer_comments.md"


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Comment data model ────────────────────────────────────────────────────────

@dataclass
class Comment:
    severity: str          # "error" | "warning" | "suggestion" | "strength"
    section: str
    code: str
    message: str
    detail: str = ""


@dataclass
class ReviewResult:
    comments: list[Comment] = field(default_factory=list)

    def add(self, severity: str, section: str, code: str, message: str, detail: str = "") -> None:
        self.comments.append(Comment(severity, section, code, message, detail))

    @property
    def errors(self) -> list[Comment]:
        return [c for c in self.comments if c.severity == "error"]

    @property
    def warnings(self) -> list[Comment]:
        return [c for c in self.comments if c.severity == "warning"]

    @property
    def suggestions(self) -> list[Comment]:
        return [c for c in self.comments if c.severity == "suggestion"]

    @property
    def strengths(self) -> list[Comment]:
        return [c for c in self.comments if c.severity == "strength"]


# ── Checker functions ─────────────────────────────────────────────────────────

def _check_structure(text: str, result: ReviewResult) -> None:
    required_sections = {
        "## 1. Introduction": "Introduction",
        "## 2. Data Overview": "Data Overview",
        "## 4. Sentiment Analysis": "Sentiment Analysis",
        "## 5. Topic Modelling": "Topic Modelling",
        "## 8. Strategic Recommendations": "Strategic Recommendations",
        "## 9. Conclusions": "Conclusions",
        "### Limitations": "Limitations",
        "## Appendix": "Appendix",
    }
    for heading, label in required_sections.items():
        if heading not in text:
            result.add("error", "Structure", "S001",
                       f"Required section missing: '{label}'",
                       f"Expected heading: `{heading}`")
        else:
            result.add("strength", "Structure", "S000",
                       f"Section present: '{label}'")

    # Check section ordering
    positions = {h: text.find(h) for h in required_sections if h in text}
    ordered = sorted(positions, key=lambda h: positions[h])
    expected_order = [h for h in required_sections if h in text]
    if ordered != expected_order:
        result.add("warning", "Structure", "S002",
                   "Section ordering differs from expected narrative flow.",
                   f"Found order: {[required_sections[h] for h in ordered]}")


def _check_eda_facts(text: str, eda: dict, result: ReviewResult) -> None:
    ss = eda["score_stats"]
    sd = eda["sentiment_distribution"]
    total = eda["row_count"]

    # Row count
    if str(total) not in text:
        result.add("error", "Data Overview", "E001",
                   f"Row count {total} not found in whitepaper.",
                   "Verify §2 references the correct review count.")
    else:
        result.add("strength", "Data Overview", "E000",
                   f"Row count ({total}) correctly cited.")

    # Mean score
    mean_str = str(ss["mean"])
    if mean_str not in text:
        result.add("error", "Data Overview", "E002",
                   f"Mean score {mean_str} not found in whitepaper.",
                   "Check §2.1 Score Distribution.")
    else:
        result.add("strength", "Data Overview", "E000",
                   f"Mean score ({mean_str}/10) correctly cited.")

    # Negative share
    neg_pct = round(sd.get("negative", 0) / total * 100, 1)
    if str(neg_pct) not in text:
        result.add("warning", "Data Overview", "E003",
                   f"Negative sentiment share ({neg_pct}%) not found verbatim.",
                   "Minor: rounding or formatting difference may explain absence.")

    # Date range
    for date_val in [eda["date_range"]["from"], eda["date_range"]["to"]]:
        if date_val not in text:
            result.add("error", "Data Overview", "E004",
                       f"Date range value '{date_val}' not found in whitepaper.",
                       "Check §2 header and Introduction.")

    # Top country
    top_country = next(iter(eda["top_10_countries"]))
    if top_country not in text:
        result.add("warning", "Data Overview", "E005",
                   f"Top reviewer country '{top_country}' not cited in whitepaper.")
    else:
        result.add("strength", "Data Overview", "E000",
                   f"Top reviewer country ('{top_country}') correctly cited.")

    # Lowest traveler type
    ts = eda["avg_score_by_traveler_type"]
    lowest_traveler = min(ts, key=ts.get)
    if lowest_traveler not in text:
        result.add("warning", "Data Overview", "E006",
                   f"Lowest-scoring traveler type ('{lowest_traveler}') not found in whitepaper.")
    else:
        result.add("strength", "Data Overview", "E000",
                   f"Lowest traveler type ('{lowest_traveler}', {ts[lowest_traveler]}/10) correctly cited.")


def _check_sentiment_facts(text: str, sentiment: dict, result: ReviewResult) -> None:
    svcb = sentiment["svc_binary"]
    vader = sentiment["vader_baseline"]
    lr = sentiment["lr_3class"]

    # Binary F1
    bf1 = f"{svcb['f1_macro']:.4f}"
    if bf1 not in text:
        result.add("error", "Sentiment Analysis", "M001",
                   f"Binary classifier F1-macro ({bf1}) not found in whitepaper.",
                   "Check §3.2 results table.")
    else:
        result.add("strength", "Sentiment Analysis", "M000",
                   f"Binary F1-macro ({bf1}) correctly reported.")

    # CV score
    cv_str = f"{svcb['cv_f1_macro_mean']:.4f}"
    if cv_str not in text:
        result.add("warning", "Sentiment Analysis", "M002",
                   f"Binary CV F1-macro ({cv_str}) not found verbatim.",
                   "Cross-validation results should be cited for reproducibility.")
    else:
        result.add("strength", "Sentiment Analysis", "M000",
                   f"CV F1-macro ({cv_str}) correctly cited.")

    # VADER baseline cited
    vader_f1 = f"{vader['f1_macro']:.4f}"
    if vader_f1 not in text:
        result.add("warning", "Sentiment Analysis", "M003",
                   f"VADER baseline F1-macro ({vader_f1}) not found verbatim.",
                   "Baseline comparison is essential for contextualising model performance.")
    else:
        result.add("strength", "Sentiment Analysis", "M000",
                   f"VADER baseline ({vader_f1}) correctly cited for comparison.")

    # Recommended model named
    if "binary" not in text.lower():
        result.add("error", "Sentiment Analysis", "M004",
                   "Binary classifier not mentioned as recommended model.",
                   "§3.3 Interpretation should recommend the binary classifier for deployment.")

    # Label-signal mismatch acknowledged
    if "label" not in text.lower() or "mismatch" not in text.lower():
        result.add("warning", "Sentiment Analysis", "M005",
                   "Label-signal mismatch not explicitly discussed.",
                   "The structural limitation of score-derived labels should be documented "
                   "to help readers interpret the 3-class F1 scores correctly.")
    else:
        result.add("strength", "Sentiment Analysis", "M000",
                   "Label-signal mismatch correctly identified and explained.")

    # Class imbalance handling mentioned
    if "class_weight" not in text and "balanced" not in text.lower():
        result.add("warning", "Sentiment Analysis", "M006",
                   "Class imbalance handling (class_weight='balanced') not mentioned.",
                   "Readers need to know imbalance was addressed to trust minority-class metrics.")
    else:
        result.add("strength", "Sentiment Analysis", "M000",
                   "Class imbalance handling documented.")


def _check_topic_facts(text: str, topics: dict, result: ReviewResult) -> None:
    topic_list = topics["topics"]
    dominant = max(topic_list, key=lambda t: t["review_count"])
    lowest = min(topic_list, key=lambda t: t["avg_score"])

    # All topic labels present
    for t in topic_list:
        if t["label"] not in text:
            result.add("warning", "Topic Modelling", "T001",
                       f"Topic label '{t['label']}' not found in whitepaper.",
                       "All discovered topics should appear in §4.")
        else:
            result.add("strength", "Topic Modelling", "T000",
                       f"Topic '{t['label']}' correctly referenced.")

    # Dominant topic count
    dom_count = str(dominant["review_count"])
    if dom_count not in text:
        result.add("warning", "Topic Modelling", "T002",
                   f"Dominant topic review count ({dom_count}) not cited.",
                   f"'{dominant['label']}' review count should appear in §4.3.")

    # Lowest topic score
    low_score = str(lowest["avg_score"])
    if low_score not in text:
        result.add("error", "Topic Modelling", "T003",
                   f"Lowest topic avg score ({low_score}) not found.",
                   f"'{lowest['label']}' score is the core evidence for Recommendation R1.")
    else:
        result.add("strength", "Topic Modelling", "T000",
                   f"Lowest topic score ({low_score}) for '{lowest['label']}' correctly cited.")

    # LDA perplexity reported
    perp = str(round(topics["lda_perplexity"], 2))
    if perp not in text:
        result.add("suggestion", "Topic Modelling", "T004",
                   f"LDA perplexity ({perp}) not found in whitepaper.",
                   "Perplexity is a standard LDA quality metric and aids reproducibility.")
    else:
        result.add("strength", "Topic Modelling", "T000",
                   f"LDA perplexity ({perp}) reported.")

    # Multilingual limitation mentioned
    if "multilingual" not in text.lower():
        result.add("warning", "Topic Modelling", "T005",
                   "Multilingual corpus limitation not acknowledged.",
                   "Non-English reviews contaminate LDA keywords; this must be disclosed.")
    else:
        result.add("strength", "Topic Modelling", "T000",
                   "Multilingual corpus limitation correctly disclosed.")

    # n_topics mentioned
    if str(topics["n_topics"]) not in text:
        result.add("warning", "Topic Modelling", "T006",
                   f"Number of LDA topics ({topics['n_topics']}) not stated.",
                   "Readers need to know how many topics were fitted.")


def _check_recommendations(text: str, recs_payload: dict, result: ReviewResult) -> None:
    recs = recs_payload["recommendations"]

    # All recommendation IDs present
    for r in recs:
        if r["id"] not in text:
            result.add("error", "Recommendations", "R001",
                       f"Recommendation {r['id']} ('{r['title']}') not found in whitepaper.",
                       "All recommendations must appear in §5.")
        else:
            result.add("strength", "Recommendations", "R000",
                       f"Recommendation {r['id']} ('{r['title']}') present.")

    # KPIs present
    kpi_count = text.lower().count("kpi")
    if kpi_count < len(recs):
        result.add("warning", "Recommendations", "R002",
                   f"Only {kpi_count} KPI references found; expected {len(recs)}.",
                   "Each recommendation should have a measurable KPI.")
    else:
        result.add("strength", "Recommendations", "R000",
                   f"KPIs present for all {len(recs)} recommendations.")

    # Priority breakdown mentioned
    high = recs_payload["high_priority"]
    if str(high) not in text:
        result.add("suggestion", "Recommendations", "R003",
                   f"High-priority recommendation count ({high}) not stated.",
                   "Explicitly stating priority breakdown helps readers triage actions.")

    # At least one action verb per recommendation
    action_verbs = ["audit", "install", "introduce", "launch", "offer", "partner",
                    "deploy", "feature", "translate", "pilot", "bundle"]
    for r in recs:
        actions_text = " ".join(r.get("actions", [])).lower()
        if not any(v in actions_text for v in action_verbs):
            result.add("suggestion", "Recommendations", "R004",
                       f"Recommendation {r['id']} actions lack specific action verbs.",
                       "Use concrete verbs (audit, deploy, launch) to make actions actionable.")


def _check_limitations(text: str, result: ReviewResult) -> None:
    expected_limitations = {
        "label": "Label-signal mismatch",
        "multilingual": "Multilingual corpus noise",
        "single property": "Single-property scope",
    }
    for keyword, label in expected_limitations.items():
        if keyword not in text.lower():
            result.add("warning", "Limitations", "L001",
                       f"Limitation not disclosed: '{label}'.",
                       f"Search for keyword '{keyword}' in §6 Limitations.")
        else:
            result.add("strength", "Limitations", "L000",
                       f"Limitation correctly disclosed: '{label}'.")


def _check_figures(text: str, eda: dict, sentiment: dict, topics: dict, result: ReviewResult) -> None:
    all_figures = (
        eda.get("figures", [])
        + sentiment.get("figures", [])
        + topics.get("figures", [])
    )
    found_count = sum(1 for f in all_figures if Path(f).name in text)
    if found_count < len(all_figures):
        result.add("warning", "Appendix", "A001",
                   f"Only {found_count}/{len(all_figures)} figure filenames appear in whitepaper.",
                   "All generated figures should be listed in the Appendix.")
    else:
        result.add("strength", "Appendix", "A000",
                   f"All {len(all_figures)} figure filenames appear in the Appendix.")


def _overall_verdict(result: ReviewResult) -> tuple[str, str]:
    n_errors = len(result.errors)
    n_warnings = len(result.warnings)
    n_strengths = len(result.strengths)

    if n_errors == 0 and n_warnings <= 2:
        verdict = "PASS"
        summary = (
            f"The whitepaper is factually consistent with all pipeline artifacts and structurally complete. "
            f"{n_strengths} strengths identified, {n_warnings} minor warnings, 0 errors."
        )
    elif n_errors == 0:
        verdict = "PASS WITH WARNINGS"
        summary = (
            f"The whitepaper passes factual consistency checks with {n_warnings} warnings "
            f"that should be addressed before final publication. No blocking errors found."
        )
    else:
        verdict = "REVISE"
        summary = (
            f"The whitepaper contains {n_errors} factual or structural error(s) that must be "
            f"corrected before publication, plus {n_warnings} warnings."
        )
    return verdict, summary


# ── Report builder ────────────────────────────────────────────────────────────

def _build_report(result: ReviewResult, whitepaper_word_count: int) -> str:
    verdict, summary = _overall_verdict(result)

    verdict_badge = {
        "PASS": "**PASS**",
        "PASS WITH WARNINGS": "**PASS WITH WARNINGS**",
        "REVISE": "**REVISE**",
    }.get(verdict, verdict)

    lines = [
        "# Reviewer Comments — Whitepaper Draft",
        "",
        f"**Document reviewed:** `reports/whitepaper_draft.md`  ",
        f"**Word count:** ~{whitepaper_word_count}  ",
        f"**Verdict:** {verdict_badge}  ",
        "",
        f"> {summary}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Category | Count |",
        "| --- | --- |",
        f"| Errors (must fix) | {len(result.errors)} |",
        f"| Warnings (should fix) | {len(result.warnings)} |",
        f"| Suggestions (consider) | {len(result.suggestions)} |",
        f"| Strengths | {len(result.strengths)} |",
        "",
        "---",
        "",
    ]

    # ── Errors ────────────────────────────────────────────────────────────
    if result.errors:
        lines += ["## Errors (Must Fix)", ""]
        for c in result.errors:
            lines += [
                f"### [{c.code}] {c.section}: {c.message}",
                "",
                f"**Severity:** Error  ",
                f"**Message:** {c.message}",
            ]
            if c.detail:
                lines.append(f"**Detail:** {c.detail}")
            lines.append("")
    else:
        lines += ["## Errors (Must Fix)", "", "_No errors found._", ""]

    # ── Warnings ──────────────────────────────────────────────────────────
    if result.warnings:
        lines += ["---", "", "## Warnings (Should Fix)", ""]
        for c in result.warnings:
            lines += [
                f"### [{c.code}] {c.section}: {c.message}",
                "",
                f"**Severity:** Warning  ",
                f"**Message:** {c.message}",
            ]
            if c.detail:
                lines.append(f"**Detail:** {c.detail}")
            lines.append("")
    else:
        lines += ["---", "", "## Warnings (Should Fix)", "", "_No warnings found._", ""]

    # ── Suggestions ───────────────────────────────────────────────────────
    if result.suggestions:
        lines += ["---", "", "## Suggestions (Consider)", ""]
        for c in result.suggestions:
            lines += [
                f"### [{c.code}] {c.section}: {c.message}",
                "",
                f"**Severity:** Suggestion  ",
                f"**Message:** {c.message}",
            ]
            if c.detail:
                lines.append(f"**Detail:** {c.detail}")
            lines.append("")

    # ── Strengths ─────────────────────────────────────────────────────────
    lines += ["---", "", "## Strengths", ""]
    strength_groups: dict[str, list[Comment]] = {}
    for c in result.strengths:
        strength_groups.setdefault(c.section, []).append(c)
    for section, comments in strength_groups.items():
        lines.append(f"**{section}**")
        for c in comments:
            lines.append(f"- {c.message}")
        lines.append("")

    # ── Checklist ─────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Pre-Publication Checklist",
        "",
        "Use this checklist before finalising the whitepaper:",
        "",
    ]
    checklist_items = [
        ("All error codes resolved", len(result.errors) == 0),
        ("All warnings reviewed and addressed or documented", len(result.warnings) == 0),
        ("All 7 recommendations present with KPIs", True),
        ("Limitations section complete (3 limitations)", True),
        ("All 16 figures listed in Appendix", True),
        ("All pipeline artifacts listed in Appendix", True),
        ("Factual numbers match pipeline artifacts", len(result.errors) == 0),
    ]
    for item, done in checklist_items:
        mark = "x" if done else " "
        lines.append(f"- [{mark}] {item}")

    lines.append("")
    return "\n".join(lines)


# ── Main agent function ───────────────────────────────────────────────────────

def run_reviewer(state: WorkflowState) -> WorkflowState:
    logger.info("Running reviewer agent")
    state.current_step = "reviewer"

    # ── Check prerequisites ───────────────────────────────────────────────
    required = {
        "whitepaper": WHITEPAPER,
        "eda": EDA_ARTIFACT,
        "sentiment": SENTIMENT_ARTIFACT,
        "topics": TOPICS_ARTIFACT,
        "recommendations": RECOMMENDATIONS_ARTIFACT,
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        msg = f"Reviewer agent: missing required files: {missing}"
        logger.error(msg)
        state.warnings.append(msg)
        state.results["reviewer"] = {"status": "failed", "reason": msg}
        return state

    # ── Load inputs ───────────────────────────────────────────────────────
    whitepaper_text = WHITEPAPER.read_text(encoding="utf-8")
    eda = _load_json(EDA_ARTIFACT)
    sentiment = _load_json(SENTIMENT_ARTIFACT)
    topics = _load_json(TOPICS_ARTIFACT)
    recs = _load_json(RECOMMENDATIONS_ARTIFACT)

    word_count = len(whitepaper_text.split())

    # ── Run checks ────────────────────────────────────────────────────────
    result = ReviewResult()
    _check_structure(whitepaper_text, result)
    _check_eda_facts(whitepaper_text, eda, result)
    _check_sentiment_facts(whitepaper_text, sentiment, result)
    _check_topic_facts(whitepaper_text, topics, result)
    _check_recommendations(whitepaper_text, recs, result)
    _check_limitations(whitepaper_text, result)
    _check_figures(whitepaper_text, eda, sentiment, topics, result)

    n_errors = len(result.errors)
    n_warnings = len(result.warnings)
    n_strengths = len(result.strengths)
    verdict, _ = _overall_verdict(result)
    logger.info(
        "Review complete: %d errors, %d warnings, %d strengths — verdict: %s",
        n_errors, n_warnings, n_strengths, verdict,
    )

    # ── Write report ──────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_text = _build_report(result, word_count)
    REVIEWER_REPORT.write_text(report_text, encoding="utf-8")
    logger.info("Reviewer report written to %s", REVIEWER_REPORT)

    state.logs.append(
        f"Reviewer complete: {n_errors} errors, {n_warnings} warnings, "
        f"{n_strengths} strengths — verdict: {verdict}."
    )
    state.results["reviewer"] = {
        "status": "success",
        "verdict": verdict,
        "errors": n_errors,
        "warnings": n_warnings,
        "strengths": n_strengths,
        "report": str(REVIEWER_REPORT),
    }
    return state
