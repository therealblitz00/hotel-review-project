"""
Aspect-Based Sentiment Analysis (ABSA) agent.

Rule-based approach:
  1. For each review, search for aspect keywords in a ±N-word window.
  2. Score the window with VADER compound score.
  3. Classify each mention: positive (>=0.05), negative (<=-0.05), neutral.
  4. Aggregate by aspect, traveler segment, and month.

Aspects covered: staff, location, room, cleanliness, breakfast,
                 wifi/check-in, value, noise/quiet.
"""
from __future__ import annotations

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.orchestration.state import WorkflowState
from src.utils.logging_utils import get_logger
from src.utils.paths import ARTIFACTS_DIR, FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR

logger = get_logger(__name__)

CLEAN_CSV = PROCESSED_DIR / "reviews_clean.csv"
ABSA_ARTIFACT = ARTIFACTS_DIR / "absa.json"
ABSA_REPORT = REPORTS_DIR / "absa_report.md"

WINDOW = 12          # words either side of a keyword hit
MIN_MENTIONS = 5     # aspects with fewer mentions are excluded from visuals
BAR_COLOR = "#1976D2"

# ── Aspect keyword lexicon ────────────────────────────────────────────────────
ASPECTS: dict[str, list[str]] = {
    "Staff & Service": [
        "staff", "receptionist", "reception", "front desk", "employee",
        "team", "friendly", "helpful", "rude", "service", "host",
        "concierge", "check in", "check-in", "checkout", "check out",
    ],
    "Location": [
        "location", "located", "central", "centre", "center", "metro",
        "subway", "station", "walking distance", "walk", "nearby",
        "neighbourhood", "neighborhood", "area", "transport",
    ],
    "Room": [
        "room", "bedroom", "suite", "cabin", "apartment", "space",
        "bed", "pillow", "mattress", "linen", "towel", "shower",
        "bathroom", "toilet", "small", "spacious", "size",
    ],
    "Cleanliness": [
        "clean", "cleaning", "cleanliness", "dirty", "dust", "hygiene",
        "spotless", "tidy", "filthy", "stain", "smell", "odour", "odor",
    ],
    "Breakfast": [
        "breakfast", "buffet", "brunch", "food", "coffee", "pastry",
        "pastel de nata", "croissant", "fruit", "cereal", "juice",
        "dining", "restaurant", "meal",
    ],
    "WiFi & Check-in": [
        "wifi", "wi-fi", "internet", "connection", "signal", "password",
        "check in", "check-in", "key", "door", "access", "arrival",
        "late arrival", "front door",
    ],
    "Value": [
        "price", "value", "money", "expensive", "cheap", "worth",
        "cost", "rate", "affordable", "overpriced", "budget",
    ],
    "Noise": [
        "noise", "noisy", "loud", "quiet", "sound", "street noise",
        "thin walls", "disturbing", "sleep", "party",
    ],
}


def _extract_window(text: str, keyword: str, window: int) -> list[str]:
    """Return all word-window snippets around each occurrence of keyword in text."""
    words = text.lower().split()
    kw_tokens = keyword.lower().split()
    snippets: list[str] = []
    for i, word in enumerate(words):
        if words[i: i + len(kw_tokens)] == kw_tokens:
            start = max(0, i - window)
            end = min(len(words), i + len(kw_tokens) + window)
            snippets.append(" ".join(words[start:end]))
    return snippets


def _score_sentiment(sia: SentimentIntensityAnalyzer, snippet: str) -> str:
    score = sia.polarity_scores(snippet)["compound"]
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def _savefig(name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", path)
    return str(path)


def run_absa(state: WorkflowState) -> WorkflowState:
    logger.info("Running ABSA agent")
    state.current_step = "absa"

    if not CLEAN_CSV.exists():
        msg = f"ABSA agent: clean data not found: {CLEAN_CSV}"
        logger.error(msg)
        state.warnings.append(msg)
        state.results["absa"] = {"status": "failed", "reason": msg}
        return state

    df = pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
    sia = SentimentIntensityAnalyzer()
    figures: list[str] = []

    # ── Extract mentions per review ───────────────────────────────────────
    rows: list[dict] = []
    for _, rev in df.iterrows():
        text = str(rev["full_review_text"])
        for aspect, keywords in ASPECTS.items():
            for kw in keywords:
                for snippet in _extract_window(text, kw, WINDOW):
                    rows.append({
                        "review_idx": rev.name,
                        "traveler_type": rev["traveler_type"],
                        "date_parsed": rev["date_parsed"],
                        "score": rev["score"],
                        "aspect": aspect,
                        "keyword": kw,
                        "snippet": snippet,
                        "sentiment": _score_sentiment(sia, snippet),
                    })

    mentions_df = pd.DataFrame(rows)
    logger.info("Total aspect mentions extracted: %d", len(mentions_df))

    # Deduplicate: one mention per (review, aspect) to avoid keyword overlap inflating counts
    mentions_dedup = (
        mentions_df.sort_values("sentiment")   # negative < neutral < positive (alphabetical keeps neg first for tie-break)
        .drop_duplicates(subset=["review_idx", "aspect"])
        .copy()
    )

    # ── Aggregate: overall aspect sentiment ──────────────────────────────
    overall = (
        mentions_dedup.groupby(["aspect", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    )
    overall["total"] = overall.sum(axis=1)
    overall["neg_pct"] = (overall["negative"] / overall["total"] * 100).round(1)
    overall["pos_pct"] = (overall["positive"] / overall["total"] * 100).round(1)
    overall = overall[overall["total"] >= MIN_MENTIONS].sort_values("neg_pct", ascending=False)

    # ── Figure 1: Negative mention % by aspect ───────────────────────────
    palette = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(overall)))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(overall.index, overall["neg_pct"], color=palette, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_xlabel("Negative mention share (%)")
    ax.set_title("Aspect-level negative sentiment — all reviews")
    ax.axvline(overall["neg_pct"].mean(), color="#555", linestyle="--",
               linewidth=1, label=f"Mean {overall['neg_pct'].mean():.1f}%")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figures.append(_savefig("fig_absa_neg_by_aspect.png"))

    # ── Figure 2: Grouped bar — pos / neutral / neg counts by aspect ─────
    x = np.arange(len(overall))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, overall["positive"], w, label="Positive", color="#4CAF50", edgecolor="white")
    ax.bar(x,     overall["neutral"],  w, label="Neutral",  color="#FF9800", edgecolor="white")
    ax.bar(x + w, overall["negative"], w, label="Negative", color="#F44336", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(overall.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mention count")
    ax.set_title("Aspect sentiment breakdown (deduplicated per review)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figures.append(_savefig("fig_absa_breakdown.png"))

    # ── Figure 3: Negative % by aspect × traveler type (heatmap) ─────────
    seg_pivot = (
        mentions_dedup[mentions_dedup["aspect"].isin(overall.index)]
        .groupby(["traveler_type", "aspect", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    )
    seg_pivot["total"] = seg_pivot.sum(axis=1)
    seg_pivot["neg_pct"] = (seg_pivot["negative"] / seg_pivot["total"] * 100).round(1)
    heat = seg_pivot["neg_pct"].unstack("aspect").fillna(0)
    heat = heat.reindex(columns=overall.index)   # same aspect order

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=50)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=9)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            ax.text(j, i, f"{heat.values[i, j]:.0f}%", ha="center", va="center",
                    fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="Negative %")
    ax.set_title("Negative mention % by traveler type × aspect")
    plt.tight_layout()
    figures.append(_savefig("fig_absa_heatmap.png"))

    # ── Figure 4: Top-3 pain-point aspects trend over time ────────────────
    top3_pain = overall.head(3).index.tolist()
    monthly_neg = (
        mentions_dedup[mentions_dedup["aspect"].isin(top3_pain)]
        .groupby(["date_parsed", "aspect", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    )
    monthly_neg["neg_pct"] = (
        monthly_neg["negative"] / monthly_neg[["positive","neutral","negative"]].sum(axis=1) * 100
    ).fillna(0)
    trend = monthly_neg["neg_pct"].unstack("aspect").fillna(0).sort_index()
    trend = trend.reindex(columns=[c for c in top3_pain if c in trend.columns])

    if not trend.empty and len(trend) > 2:
        colors = ["#F44336", "#FF9800", "#9C27B0"]
        fig, ax = plt.subplots(figsize=(13, 5))
        for col, color in zip(trend.columns, colors):
            ax.plot(trend.index, trend[col], marker="o", linewidth=1.8,
                    markersize=4, label=col, color=color)
        step = max(1, len(trend) // 10)
        ax.set_xticks(trend.index[::step])
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Negative mention share (%)")
        ax.set_title("Top pain-point aspect trends over time")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        figures.append(_savefig("fig_absa_trends.png"))

    # ── Build summary structures ──────────────────────────────────────────
    aspect_summary = []
    for aspect in overall.index:
        row = overall.loc[aspect]
        aspect_summary.append({
            "aspect": aspect,
            "total_mentions": int(row["total"]),
            "positive": int(row["positive"]),
            "neutral": int(row["neutral"]),
            "negative": int(row["negative"]),
            "neg_pct": float(row["neg_pct"]),
            "pos_pct": float(row["pos_pct"]),
        })

    # ── Write artifact ────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "total_mentions": len(mentions_dedup),
        "window_size": WINDOW,
        "aspects": aspect_summary,
        "figures": figures,
    }
    ABSA_ARTIFACT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    logger.info("ABSA artifact written to %s", ABSA_ARTIFACT)

    # ── Write report ──────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Aspect-Based Sentiment Analysis Report",
        "",
        f"**Method:** VADER compound score on ±{WINDOW}-word keyword windows  ",
        f"**Reviews:** {len(df)}  ",
        f"**Total deduplicated aspect mentions:** {len(mentions_dedup)}  ",
        "",
        "## Overall Aspect Sentiment",
        "",
        "| Aspect | Mentions | Positive | Neutral | Negative | Neg % |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for a in aspect_summary:
        lines.append(
            f"| {a['aspect']} | {a['total_mentions']} | {a['positive']} "
            f"| {a['neutral']} | {a['negative']} | {a['neg_pct']}% |"
        )

    lines += [
        "",
        "## Key Findings",
        "",
        f"- **Highest pain point:** {aspect_summary[0]['aspect']} "
        f"({aspect_summary[0]['neg_pct']}% negative mentions)",
        f"- **Most discussed aspect:** "
        f"{max(aspect_summary, key=lambda x: x['total_mentions'])['aspect']} "
        f"({max(aspect_summary, key=lambda x: x['total_mentions'])['total_mentions']} mentions)",
        f"- **Strongest performer:** {aspect_summary[-1]['aspect']} "
        f"({aspect_summary[-1]['pos_pct']}% positive, "
        f"{aspect_summary[-1]['neg_pct']}% negative)",
        "",
        "## Negative % by Traveler Type × Aspect",
        "",
        "| Traveler type | " + " | ".join(heat.columns) + " |",
        "| --- |" + " --- |" * len(heat.columns),
        *[
            "| " + idx + " | " + " | ".join(f"{v:.0f}%" for v in row) + " |"
            for idx, row in zip(heat.index, heat.values)
        ],
        "",
        "## Figures",
        "",
    ]
    for fig_path in figures:
        fname = fig_path.replace("\\", "/").split("/")[-1]
        lines.append(f"- `reports/figures/{fname}`")

    ABSA_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("ABSA report written to %s", ABSA_REPORT)

    top_pain = aspect_summary[0]["aspect"]
    top_pain_pct = aspect_summary[0]["neg_pct"]
    state.logs.append(
        f"ABSA complete: {len(mentions_dedup)} mentions across {len(aspect_summary)} aspects. "
        f"Top pain point: {top_pain} ({top_pain_pct}% negative)."
    )
    state.results["absa"] = {
        "status": "success",
        "total_mentions": len(mentions_dedup),
        "aspects": aspect_summary,
        "top_pain_point": top_pain,
        "top_pain_neg_pct": top_pain_pct,
        "report": str(ABSA_REPORT),
        "artifact": str(ABSA_ARTIFACT),
        "figures_count": len(figures),
    }
    return state
