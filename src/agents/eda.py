from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

from src.orchestration.state import WorkflowState
from src.utils.logging_utils import get_logger
from src.utils.paths import ARTIFACTS_DIR, FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR

logger = get_logger(__name__)

CLEAN_CSV = PROCESSED_DIR / "reviews_clean.csv"
EDA_REPORT = REPORTS_DIR / "eda_report.md"
EDA_ARTIFACT = ARTIFACTS_DIR / "eda_summary.json"

PALETTE = {
    "positive": "#4CAF50",
    "neutral":  "#FF9800",
    "negative": "#F44336",
}
BAR_COLOR = "#1976D2"


def _savefig(name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", path)
    return str(path)


def run_eda(state: WorkflowState) -> WorkflowState:
    logger.info("Running EDA agent")
    state.current_step = "eda"

    if not CLEAN_CSV.exists():
        msg = f"Clean data file not found: {CLEAN_CSV}"
        logger.error(msg)
        state.warnings.append(msg)
        state.results["eda"] = {"status": "failed", "reason": msg}
        return state

    df = pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
    df["word_count"] = df["full_review_text"].str.split().str.len()
    figures: list[str] = []

    # ── Figure 1: Score distribution ──────────────────────────────────────
    score_counts = df["score"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(score_counts.index, score_counts.values, color=BAR_COLOR, edgecolor="white", width=0.7)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Score distribution")
    ax.set_xticks(score_counts.index)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    figures.append(_savefig("fig_score_distribution.png"))

    # ── Figure 2: Sentiment breakdown ─────────────────────────────────────
    sentiment_counts = df["sentiment"].value_counts()
    order = [s for s in ["positive", "neutral", "negative"] if s in sentiment_counts.index]
    vals = [sentiment_counts[s] for s in order]
    colors = [PALETTE[s] for s in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(order, vals, color=colors, edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_xlabel("Number of reviews")
    ax.set_title("Sentiment label distribution")
    ax.spines[["top", "right"]].set_visible(False)
    figures.append(_savefig("fig_sentiment_distribution.png"))

    # ── Figure 3: Reviews over time (by month) ────────────────────────────
    monthly = df.groupby("date_parsed").size().reset_index(name="count")
    monthly = monthly.sort_values("date_parsed")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly["date_parsed"], monthly["count"], marker="o", color=BAR_COLOR, linewidth=1.5, markersize=4)
    ax.fill_between(monthly["date_parsed"], monthly["count"], alpha=0.15, color=BAR_COLOR)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Review volume over time")
    step = max(1, len(monthly) // 10)
    ax.set_xticks(monthly["date_parsed"].iloc[::step])
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figures.append(_savefig("fig_reviews_over_time.png"))

    # ── Figure 4: Top 10 reviewer countries ──────────────────────────────
    top_countries = df["country"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top_countries.index[::-1], top_countries.values[::-1], color=BAR_COLOR, edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_xlabel("Number of reviews")
    ax.set_title("Top 10 reviewer countries")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figures.append(_savefig("fig_top_countries.png"))

    # ── Figure 5: Average score by traveler type ──────────────────────────
    avg_score = df.groupby("traveler_type")["score"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(avg_score.index, avg_score.values, color=BAR_COLOR, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set_xlabel("Average score")
    ax.set_title("Average score by traveler type")
    ax.set_xlim(0, 11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figures.append(_savefig("fig_avg_score_by_traveler.png"))

    # ── Figure 6: Review word count distribution ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["word_count"].clip(upper=200), bins=40, color=BAR_COLOR, edgecolor="white")
    ax.set_xlabel("Word count (capped at 200)")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Review length distribution")
    ax.spines[["top", "right"]].set_visible(False)
    figures.append(_savefig("fig_review_length.png"))

    # ── Figure 7: Average score by room type ─────────────────────────────
    room_stats = (
        df.groupby("room_type")["score"]
        .agg(mean="mean", count="count")
        .query("count >= 10")
        .sort_values("mean")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(room_stats.index, room_stats["mean"], color=BAR_COLOR, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set_xlabel("Average score")
    ax.set_title("Average score by room type (min 10 reviews)")
    ax.set_xlim(0, 11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figures.append(_savefig("fig_avg_score_by_room.png"))

    # ── Figure 8 & 9: Word clouds ─────────────────────────────────────────
    extra_stops = {
        "hotel", "room", "stay", "one", "will", "also", "really", "bit",
        "get", "got", "just", "like", "even", "us", "would", "could", "place",
    }
    wc_stopwords = STOPWORDS | extra_stops

    def _wordcloud(text: str, title: str, colormap: str, fname: str) -> str:
        wc = WordCloud(
            width=1200, height=600, background_color="white",
            colormap=colormap, stopwords=wc_stopwords,
            max_words=120, collocations=True,
            prefer_horizontal=0.85,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, pad=12)
        plt.tight_layout()
        return _savefig(fname)

    pos_text = " ".join(df.loc[df["sentiment"] == "positive", "full_review_text"].dropna())
    neg_text = " ".join(df.loc[df["sentiment"] == "negative", "full_review_text"].dropna())

    if pos_text.strip():
        figures.append(_wordcloud(pos_text, "Most frequent words — Positive reviews",
                                  "Greens", "fig_wordcloud_positive.png"))
    if neg_text.strip():
        figures.append(_wordcloud(neg_text, "Most frequent words — Negative reviews",
                                  "Reds", "fig_wordcloud_negative.png"))

    # ── Figure 10: Temporal sentiment share ──────────────────────────────
    sentiment_monthly = (
        df.groupby(["date_parsed", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    )
    sentiment_monthly_pct = sentiment_monthly.div(sentiment_monthly.sum(axis=1), axis=0) * 100
    sentiment_monthly_pct = sentiment_monthly_pct.sort_index()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.stackplot(
        sentiment_monthly_pct.index,
        sentiment_monthly_pct["positive"],
        sentiment_monthly_pct["neutral"],
        sentiment_monthly_pct["negative"],
        labels=["Positive", "Neutral", "Negative"],
        colors=[PALETTE["positive"], PALETTE["neutral"], PALETTE["negative"]],
        alpha=0.85,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of reviews (%)")
    ax.set_title("Sentiment share over time")
    step = max(1, len(sentiment_monthly_pct) // 10)
    ax.set_xticks(sentiment_monthly_pct.index[::step])
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figures.append(_savefig("fig_sentiment_over_time.png"))

    # ── Compute summary statistics ────────────────────────────────────────
    score_stats = {
        "min": float(df["score"].min()),
        "max": float(df["score"].max()),
        "mean": round(float(df["score"].mean()), 2),
        "median": float(df["score"].median()),
        "std": round(float(df["score"].std()), 2),
    }
    word_stats = {
        "min": int(df["word_count"].min()),
        "max": int(df["word_count"].max()),
        "mean": round(float(df["word_count"].mean()), 1),
        "median": float(df["word_count"].median()),
    }
    avg_score_by_traveler = df.groupby("traveler_type")["score"].mean().round(2).to_dict()
    monthly_peak = monthly.loc[monthly["count"].idxmax()]

    # ── Key observations ──────────────────────────────────────────────────
    observations = [
        f"The dataset contains {len(df)} reviews spanning {df['date_parsed'].nunique()} months "
        f"({df['date_parsed'].min()} to {df['date_parsed'].max()}).",
        f"The overall average score is {score_stats['mean']} out of 10 (median {score_stats['median']}), "
        f"indicating a strongly positive guest experience.",
        f"Sentiment breakdown: {sentiment_counts.get('positive', 0)} positive ({round(sentiment_counts.get('positive', 0)/len(df)*100, 1)}%), "
        f"{sentiment_counts.get('neutral', 0)} neutral ({round(sentiment_counts.get('neutral', 0)/len(df)*100, 1)}%), "
        f"{sentiment_counts.get('negative', 0)} negative ({round(sentiment_counts.get('negative', 0)/len(df)*100, 1)}%).",
        f"The most common reviewer origin is {top_countries.index[0]} ({top_countries.iloc[0]} reviews, "
        f"{round(top_countries.iloc[0]/len(df)*100, 1)}% of total).",
        f"Review volume peaked in {monthly_peak['date_parsed']} ({int(monthly_peak['count'])} reviews).",
        f"Average review length is {word_stats['mean']} words (median {word_stats['median']}); "
        f"the longest review has {word_stats['max']} words.",
        f"Couples represent the largest traveler segment ({df['traveler_type'].value_counts().iloc[0]} reviews).",
        f"Budget Double Room is the most reviewed room type ({df['room_type'].value_counts().iloc[0]} reviews).",
    ]

    # ── Write EDA summary artifact ────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    eda_summary = {
        "row_count": len(df),
        "date_range": {"from": df["date_parsed"].min(), "to": df["date_parsed"].max()},
        "score_stats": score_stats,
        "sentiment_distribution": sentiment_counts.to_dict(),
        "top_10_countries": top_countries.to_dict(),
        "avg_score_by_traveler_type": avg_score_by_traveler,
        "word_count_stats": word_stats,
        "monthly_peak": {"month": monthly_peak["date_parsed"], "count": int(monthly_peak["count"])},
        "figures": figures,
        "observations": observations,
    }
    EDA_ARTIFACT.write_text(json.dumps(eda_summary, indent=2), encoding="utf-8")
    logger.info("EDA summary artifact written to %s", EDA_ARTIFACT)

    # ── Write EDA report ──────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# EDA Report",
        "",
        f"**Dataset:** `{CLEAN_CSV}`  ",
        f"**Reviews:** {len(df)}  ",
        f"**Period:** {df['date_parsed'].min()} to {df['date_parsed'].max()}  ",
        "",
        "## Key observations",
        "",
    ]
    for obs in observations:
        lines.append(f"- {obs}")

    lines += [
        "",
        "## Score distribution",
        "",
        "| Score | Count |",
        "| --- | --- |",
    ]
    for score, count in score_counts.items():
        lines.append(f"| {int(score)} | {count} |")

    lines += [
        "",
        f"- Mean: {score_stats['mean']}  |  Median: {score_stats['median']}  |  Std: {score_stats['std']}",
        "",
        "## Sentiment distribution",
        "",
        "| Sentiment | Count | % |",
        "| --- | --- | --- |",
    ]
    for label in ["positive", "neutral", "negative"]:
        n = sentiment_counts.get(label, 0)
        lines.append(f"| {label} | {n} | {round(n/len(df)*100, 1)}% |")

    lines += [
        "",
        "## Top 10 reviewer countries",
        "",
        "| Country | Count |",
        "| --- | --- |",
    ]
    for country, count in top_countries.items():
        lines.append(f"| {country} | {count} |")

    lines += [
        "",
        "## Average score by traveler type",
        "",
        "| Traveler type | Avg score | Count |",
        "| --- | --- | --- |",
    ]
    for ttype, avg in df.groupby("traveler_type")["score"].agg(["mean", "count"]).iterrows():
        lines.append(f"| {ttype} | {avg['mean']:.2f} | {int(avg['count'])} |")

    lines += [
        "",
        "## Review length",
        "",
        f"- Mean word count: {word_stats['mean']}",
        f"- Median word count: {word_stats['median']}",
        f"- Max word count: {word_stats['max']}",
        "",
        "## Figures",
        "",
    ]
    for fig_path in figures:
        fname = fig_path.replace("\\", "/").split("/")[-1]
        lines.append(f"- `reports/figures/{fname}`")

    EDA_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("EDA report written to %s", EDA_REPORT)

    state.logs.append(
        f"EDA complete: {len(df)} reviews, avg score {score_stats['mean']}, "
        f"{len(figures)} figures saved."
    )
    state.results["eda"] = {
        "status": "success",
        "row_count": len(df),
        "score_mean": score_stats["mean"],
        "sentiment_distribution": sentiment_counts.to_dict(),
        "figures_count": len(figures),
        "report": str(EDA_REPORT),
        "artifact": str(EDA_ARTIFACT),
    }
    return state
