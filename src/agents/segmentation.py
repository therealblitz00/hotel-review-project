from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.orchestration.state import WorkflowState
from src.utils.logging_utils import get_logger
from src.utils.paths import ARTIFACTS_DIR, FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR

logger = get_logger(__name__)

CLEAN_CSV = PROCESSED_DIR / "reviews_clean.csv"
SEGMENTATION_REPORT = REPORTS_DIR / "segmentation_report.md"
SEGMENTATION_ARTIFACT = ARTIFACTS_DIR / "segmentation.json"

N_CLUSTERS = 4
RANDOM_STATE = 42
BAR_COLOR = "#1976D2"
CLUSTER_COLORS = ["#4CAF50", "#1976D2", "#FF9800", "#E91E63"]


def _savefig(name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", path)
    return str(path)


def _encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a numeric feature matrix for K-Means from the cleaned review data.
    Features:
      - score (continuous)
      - word_count (continuous)
      - nr_nights (continuous)
      - traveler_type (one-hot)
      - sentiment (ordinal: positive=2, neutral=1, negative=0)
    """
    feat = pd.DataFrame(index=df.index)
    feat["score"] = df["score"]
    feat["word_count"] = df["full_review_text"].str.split().str.len()
    feat["nr_nights"] = pd.to_numeric(df["nr_nights"], errors="coerce").fillna(
        df["nr_nights"].median() if "nr_nights" in df.columns else 2
    )
    feat["sentiment_ord"] = df["sentiment"].map({"positive": 2, "neutral": 1, "negative": 0})

    # One-hot encode traveler_type
    traveler_dummies = pd.get_dummies(df["traveler_type"], prefix="traveler", dtype=float)
    feat = pd.concat([feat, traveler_dummies], axis=1)

    feature_names = list(feat.columns)
    return feat, feature_names


def _cluster_label(profile: dict) -> str:
    """Assign a human-readable persona name based on cluster profile."""
    top_traveler = profile["top_traveler_type"]
    score = profile["score_mean"]
    wc = profile["word_count_mean"]

    traveler_map = {
        "Couple": "Romantic Couples",
        "Family": "Family Explorers",
        "Solo traveller": "Solo Adventurers",
        "Group": "Social Groups",
    }
    if top_traveler in traveler_map:
        return traveler_map[top_traveler]
    # Fallback by score / verbosity
    if score >= 8.5:
        return "Loyal Enthusiasts"
    if score < 7.0:
        return "Dissatisfied Guests"
    if wc >= 40:
        return "Engaged Reviewers"
    return "Satisfied Regulars"


def run_segmentation(state: WorkflowState) -> WorkflowState:
    logger.info("Running segmentation agent")
    state.current_step = "segmentation"

    if not CLEAN_CSV.exists():
        msg = f"Segmentation agent: clean data not found: {CLEAN_CSV}"
        logger.error(msg)
        state.warnings.append(msg)
        state.results["segmentation"] = {"status": "failed", "reason": msg}
        return state

    df = pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
    figures: list[str] = []

    # ── Feature engineering ───────────────────────────────────────────────
    feat_df, feature_names = _encode_features(df)
    feat_df = feat_df.fillna(feat_df.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df.values)

    # ── Elbow analysis (k=2..8) to justify N_CLUSTERS ────────────────────
    inertias = []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, marker="o", color=BAR_COLOR, linewidth=2)
    ax.axvline(N_CLUSTERS, color="#F44336", linestyle="--", linewidth=1.2,
               label=f"Chosen k={N_CLUSTERS}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("K-Means elbow analysis")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figures.append(_savefig("fig_segmentation_elbow.png"))

    # ── Fit final model ───────────────────────────────────────────────────
    km_final = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = km_final.fit_predict(X_scaled)

    # ── Build cluster profiles ────────────────────────────────────────────
    profiles = []
    used_labels: set[str] = set()
    for cid in range(N_CLUSTERS):
        mask = df["cluster"] == cid
        sub = df[mask]
        top_traveler = sub["traveler_type"].value_counts().index[0]
        top_sentiment = sub["sentiment"].value_counts().index[0]
        top_country = sub["country"].value_counts().index[0]

        wc = sub["full_review_text"].str.split().str.len()
        profile = {
            "cluster_id": cid,
            "size": int(mask.sum()),
            "share_pct": round(mask.sum() / len(df) * 100, 1),
            "score_mean": round(float(sub["score"].mean()), 2),
            "score_std": round(float(sub["score"].std()), 2),
            "nr_nights_mean": round(float(pd.to_numeric(sub["nr_nights"], errors="coerce").mean()), 1),
            "word_count_mean": round(float(wc.mean()), 1),
            "top_traveler_type": top_traveler,
            "top_sentiment": top_sentiment,
            "top_country": top_country,
            "sentiment_distribution": sub["sentiment"].value_counts().to_dict(),
            "traveler_distribution": sub["traveler_type"].value_counts().to_dict(),
        }
        label = _cluster_label(profile)
        # Ensure unique labels
        if label in used_labels:
            label = f"{label} ({cid})"
        used_labels.add(label)
        profile["label"] = label
        profiles.append(profile)

    # Sort by score descending for report clarity
    profiles.sort(key=lambda p: -p["score_mean"])

    # ── Figure: cluster size ──────────────────────────────────────────────
    labels = [p["label"] for p in profiles]
    sizes = [p["size"] for p in profiles]
    colors = CLUSTER_COLORS[: len(profiles)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, sizes, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=9)
    ax.set_xlabel("Number of reviews")
    ax.set_title("Customer segment sizes")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figures.append(_savefig("fig_segmentation_sizes.png"))

    # ── Figure: avg score per cluster ─────────────────────────────────────
    scores = [p["score_mean"] for p in profiles]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, scores, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set_xlabel("Average score")
    ax.set_title("Average satisfaction score by segment")
    ax.set_xlim(0, 11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figures.append(_savefig("fig_segmentation_scores.png"))

    # ── Figure: sentiment breakdown per cluster (stacked bar) ─────────────
    sentiment_labels = ["positive", "neutral", "negative"]
    palette = {"positive": "#4CAF50", "neutral": "#FF9800", "negative": "#F44336"}
    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = np.zeros(len(profiles))
    for sent in sentiment_labels:
        vals = np.array([p["sentiment_distribution"].get(sent, 0) for p in profiles])
        ax.barh(labels, vals, left=bottoms, color=palette[sent],
                label=sent.capitalize(), edgecolor="white")
        bottoms += vals
    ax.set_xlabel("Number of reviews")
    ax.set_title("Sentiment breakdown by customer segment")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figures.append(_savefig("fig_segmentation_sentiment.png"))

    # ── Write artifact ────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "n_clusters": N_CLUSTERS,
        "total_reviews": len(df),
        "features_used": feature_names,
        "elbow_inertias": {str(k): round(v, 2) for k, v in zip(k_range, inertias)},
        "segments": profiles,
        "figures": figures,
    }
    SEGMENTATION_ARTIFACT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    logger.info("Segmentation artifact written to %s", SEGMENTATION_ARTIFACT)

    # ── Write report ──────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Customer Segmentation Report",
        "",
        f"**Method:** K-Means clustering (k={N_CLUSTERS})  ",
        f"**Reviews:** {len(df)}  ",
        f"**Features:** {', '.join(feature_names)}  ",
        "",
        "## Segment Overview",
        "",
        "| Segment | Size | Share | Avg Score | Top Traveler | Top Sentiment |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for p in profiles:
        lines.append(
            f"| {p['label']} | {p['size']} | {p['share_pct']}% "
            f"| {p['score_mean']} | {p['top_traveler_type']} | {p['top_sentiment']} |"
        )

    lines += ["", "## Segment Profiles", ""]
    for p in profiles:
        lines += [
            f"### {p['label']}",
            "",
            f"- **Size:** {p['size']} reviews ({p['share_pct']}% of total)",
            f"- **Average score:** {p['score_mean']} ± {p['score_std']}",
            f"- **Avg stay:** {p['nr_nights_mean']} nights",
            f"- **Avg review length:** {p['word_count_mean']} words",
            f"- **Dominant traveler type:** {p['top_traveler_type']}",
            f"- **Dominant sentiment:** {p['top_sentiment']}",
            f"- **Top reviewer country:** {p['top_country']}",
            "",
            "**Sentiment distribution:**",
            "",
        ]
        for s, n in p["sentiment_distribution"].items():
            pct = round(n / p["size"] * 100, 1)
            lines.append(f"- {s.capitalize()}: {n} ({pct}%)")
        lines.append("")

    lines += [
        "## Marketing Implications",
        "",
        "| Segment | Priority Action |",
        "| --- | --- |",
    ]
    marketing = {
        "Romantic Couples": "Target with romance packages, late check-out perks, and couples dining offers.",
        "Family Explorers": "Promote family packages with child-friendly amenities and city guides.",
        "Solo Adventurers": "Highlight solo-friendly amenities, social spaces, and local tips in communications.",
        "Social Groups": "Offer group booking discounts, flexible room configurations, and group dining options.",
        "Loyal Enthusiasts": "Target with loyalty programme invitations and early-access offers.",
        "Dissatisfied Guests": "Trigger proactive management outreach within 24h; address root causes.",
        "Engaged Reviewers": "Leverage as brand ambassadors — invite for social media collaborations.",
    }
    for p in profiles:
        action = next(
            (v for k, v in marketing.items() if k.lower() in p["label"].lower()),
            "Develop targeted retention campaign based on segment preferences.",
        )
        lines.append(f"| {p['label']} | {action} |")

    lines += ["", "## Figures", ""]
    for fig_path in figures:
        fname = fig_path.replace("\\", "/").split("/")[-1]
        lines.append(f"- `reports/figures/{fname}`")

    SEGMENTATION_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Segmentation report written to %s", SEGMENTATION_REPORT)

    state.logs.append(
        f"Segmentation complete: {N_CLUSTERS} clusters — "
        + ", ".join(f"{p['label']} (n={p['size']}, avg={p['score_mean']})" for p in profiles)
    )
    state.results["segmentation"] = {
        "status": "success",
        "n_clusters": N_CLUSTERS,
        "segments": [{"label": p["label"], "size": p["size"], "score_mean": p["score_mean"]}
                     for p in profiles],
        "report": str(SEGMENTATION_REPORT),
        "artifact": str(SEGMENTATION_ARTIFACT),
        "figures_count": len(figures),
    }
    return state
