from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.orchestration.state import WorkflowState
from src.utils.logging_utils import get_logger
from src.utils.paths import ARTIFACTS_DIR, FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR

logger = get_logger(__name__)

CLEAN_CSV = PROCESSED_DIR / "reviews_clean.csv"
TOPIC_REPORT = REPORTS_DIR / "topic_report.md"
TOPICS_ARTIFACT = ARTIFACTS_DIR / "topics.json"
ASSIGNMENTS_ARTIFACT = ARTIFACTS_DIR / "topic_assignments.csv"

N_TOPICS = 6
N_TOP_WORDS = 12
RANDOM_STATE = 42

# Domain stop-words that add noise without meaning in this context
HOTEL_STOP_WORDS = {
    # Generic filler
    "hotel", "room", "rooms", "stay", "stayed", "night", "nights",
    "place", "just", "really", "very", "good", "great", "nice",
    "bit", "quite", "little", "lot", "lots", "got", "get",
    "didn", "wasn", "couldn", "doesn", "weren", "aren",
    # Spanish
    "muy", "bien", "es", "en", "el", "la", "los", "las", "se",
    "que", "con", "para", "por", "un", "una", "del", "al",
    "no", "su", "pero", "más", "mas", "todo", "hay", "fue",
    # Italian
    "di", "da", "il", "le", "lo", "gli", "un", "una", "del",
    "colazione", "posizione", "personale", "camera", "ottima",
    # French
    "le", "la", "les", "de", "du", "des", "et", "est", "en",
    "un", "une", "au", "aux", "avec", "pour", "sur",
    # Portuguese
    "muito", "boa", "bom", "é", "e", "o", "a", "os", "as",
    "de", "da", "do", "em", "para", "com", "que", "não",
    "localização", "localiza",
    # French (extended)
    "très", "tres", "petit", "petite", "personnel", "non",
    "bien", "bonne", "bon", "hotel", "chambre", "rue", "ville",
    "déjeuner", "dejeuner", "dans", "nous", "avons", "était",
    "notre", "petit déjeuner", "accueil",
    # Spanish (extended)
    "ubicación", "excelente", "habitación", "habitacion",
    "buena", "bueno", "todo", "también", "tambien",
    "desayuno", "muy buena", "muy bueno", "ubicacion",
    # Italian (extended)
    "della", "tutto", "bello", "bella", "ottimo", "ottima",
    "stanza", "molto", "buono", "buona", "anche", "che", "per",
    "colazione", "camera", "posizione", "personale", "cena",
    # German fragments
    "und", "die", "der", "das", "in", "ist", "zu", "war",
    # Mixed overused
    "personal", "super",
}


def _top_words(model: LatentDirichletAllocation, vocab: list[str], n: int) -> list[list[str]]:
    return [
        [vocab[i] for i in topic.argsort()[:-n - 1:-1]]
        for topic in model.components_
    ]


_LABEL_RULES: list[tuple[str, set[str]]] = [
    ("Food & Breakfast",          {"breakfast", "coffee", "food", "colazione", "drink", "drinks", "menu", "restaurant", "cafe", "buffet", "pastry"}),
    ("Rooftop Bar & Ambiance",    {"bar", "rooftop", "terrace", "view", "decor", "decoration", "decorated", "charm", "quirky", "character", "beautiful", "unique"}),
    ("Location & Accessibility",  {"metro", "station", "distance", "walking distance", "central", "centre", "center", "walk", "close", "near", "transport", "tram", "bus"}),
    ("Staff & Service",           {"staff", "friendly", "helpful", "reception", "service", "team", "welcoming", "professional", "kind", "warm"}),
    ("Room Comfort & Cleanliness",{"bed", "clean", "comfortable", "bathroom", "shower", "floor", "noise", "quiet", "sleep", "sheets", "towels", "small", "size"}),
    ("Value & Overall Experience",{"price", "value", "money", "worth", "budget", "perfect", "amazing", "excellent", "recommend", "wonderful", "disappointed", "cost", "paid", "pay", "arrival"}),
    ("Decor & Room Space",        {"decor", "decoration", "decorated", "balcony", "spacious", "space", "size", "beautiful", "charm", "quirky", "unique", "character", "retro", "themed"}),
    ("Location & Convenience",    {"location", "located", "city", "porto", "centre", "center", "convenient", "access", "nearby", "attractions", "exploring", "old", "restaurants"}),
    ("Facilities & Comfort",      {"conditioning", "air conditioning", "air", "wifi", "internet", "facilities", "pool", "gym", "parking", "noise", "single", "beds", "floor"}),
    ("Check-in & WiFi",           {"check", "wifi", "internet", "checkin", "checkout", "time", "early", "late", "pay", "charge", "reception"}),
]


def _auto_label(words: list[str], used_labels: set[str]) -> str:
    """Score all rules against the full keyword list; return best unused label."""
    kw = set(words)
    scores: list[tuple[int, str]] = []
    for label, trigger_set in _LABEL_RULES:
        score = len(kw & trigger_set)
        if score > 0:
            scores.append((score, label))
    scores.sort(key=lambda x: -x[0])
    for _, label in scores:
        if label not in used_labels:
            return label
    # Fallback: first rule match ignoring used_labels, or generic
    if scores:
        return scores[0][1] + " (variant)"
    return f"Topic: {words[0].title()}"


def run_topic(state: WorkflowState) -> WorkflowState:
    logger.info("Running topic modelling agent")
    state.current_step = "topic"

    if not CLEAN_CSV.exists():
        msg = f"Clean data file not found: {CLEAN_CSV}"
        logger.error(msg)
        state.warnings.append(msg)
        state.results["topic"] = {"status": "failed", "reason": msg}
        return state

    df = pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
    corpus = df["full_review_text"].fillna("").tolist()
    figures: list[str] = []

    # ── Vectorise ─────────────────────────────────────────────────────────
    stop_words_combined = list(
        CountVectorizer(stop_words="english").get_stop_words() | HOTEL_STOP_WORDS
    )
    vec = CountVectorizer(
        max_features=3000,
        min_df=3,
        max_df=0.90,
        stop_words=stop_words_combined,
        ngram_range=(1, 2),
    )
    dtm = vec.fit_transform(corpus)
    vocab = vec.get_feature_names_out().tolist()
    logger.info("DTM shape: %s | vocab: %d", dtm.shape, len(vocab))

    # ── LDA ───────────────────────────────────────────────────────────────
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        max_iter=30,
        learning_method="online",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    topic_matrix = lda.fit_transform(dtm)   # shape: (n_docs, N_TOPICS)
    logger.info("LDA perplexity: %.2f", lda.perplexity(dtm))

    top_words_per_topic = _top_words(lda, vocab, N_TOP_WORDS)
    used: set[str] = set()
    topic_labels = []
    for words in top_words_per_topic:
        label = _auto_label(words, used)
        topic_labels.append(label)
        used.add(label)
    logger.info("Topic labels: %s", topic_labels)

    # ── Assign dominant topic ─────────────────────────────────────────────
    dominant = np.argmax(topic_matrix, axis=1)
    df["topic_id"] = dominant
    df["topic_label"] = [topic_labels[i] for i in dominant]
    df["topic_score"] = topic_matrix[np.arange(len(df)), dominant].round(4)

    topic_counts = df["topic_label"].value_counts()

    # ── Figure 1: Topic distribution bar chart ────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(topic_counts.index[::-1], topic_counts.values[::-1], color="#1976D2", edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_xlabel("Number of reviews")
    ax.set_title(f"Topic distribution (LDA, {N_TOPICS} topics)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = FIGURES_DIR / "fig_topic_distribution.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    figures.append(str(p))
    logger.info("Saved figure: %s", p)

    # ── Figure 2: Top keywords per topic (subplots) ───────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for idx, ax in enumerate(axes.flat):
        if idx >= N_TOPICS:
            ax.set_visible(False)
            continue
        words = top_words_per_topic[idx][:10]
        scores = lda.components_[idx][
            [vocab.index(w) for w in words]
        ]
        scores = scores / scores.sum()
        ax.barh(words[::-1], scores[::-1], color="#1976D2", edgecolor="white")
        ax.set_title(topic_labels[idx], fontsize=10, fontweight="bold")
        ax.set_xlabel("Relative weight")
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("Top keywords per topic", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = FIGURES_DIR / "fig_topic_keywords.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    figures.append(str(p))
    logger.info("Saved figure: %s", p)

    # ── Figure 3: Topic by sentiment (stacked bar) ────────────────────────
    cross = pd.crosstab(df["topic_label"], df["sentiment"])
    for col in ["positive", "neutral", "negative"]:
        if col not in cross.columns:
            cross[col] = 0
    cross = cross[["positive", "neutral", "negative"]]
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

    colors_s = {"positive": "#4CAF50", "neutral": "#FF9800", "negative": "#F44336"}
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(cross_pct))
    for col in ["positive", "neutral", "negative"]:
        vals = cross_pct[col].values
        ax.barh(cross_pct.index, vals, left=bottom, label=col, color=colors_s[col], edgecolor="white")
        bottom += vals
    ax.set_xlabel("Percentage of reviews")
    ax.set_title("Sentiment breakdown by topic")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = FIGURES_DIR / "fig_topic_by_sentiment.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    figures.append(str(p))
    logger.info("Saved figure: %s", p)

    # ── Figure 4: Topic by traveler type (heatmap) ────────────────────────
    heat = pd.crosstab(df["topic_label"], df["traveler_type"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="%")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_yticks(range(len(heat.index)))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right")
    ax.set_yticklabels(heat.index)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            ax.text(j, i, f"{heat.values[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if heat.values[i, j] > 50 else "black")
    ax.set_title("Topic composition by traveler type (%)")
    plt.tight_layout()
    p = FIGURES_DIR / "fig_topic_by_traveler.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    figures.append(str(p))
    logger.info("Saved figure: %s", p)

    # ── Representative reviews per topic ──────────────────────────────────
    representatives: dict[str, list[str]] = {}
    for t_id, t_label in enumerate(topic_labels):
        mask = df["topic_id"] == t_id
        subset = df[mask].nlargest(3, "topic_score")
        representatives[t_label] = subset["full_review_text"].str[:200].tolist()

    # ── Average score per topic ───────────────────────────────────────────
    avg_score_by_topic = df.groupby("topic_label")["score"].mean().round(2).to_dict()

    # ── Write artifacts ───────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    topics_data = {
        "n_topics": N_TOPICS,
        "lda_perplexity": round(lda.perplexity(dtm), 2),
        "topics": [
            {
                "id": i,
                "label": topic_labels[i],
                "top_words": top_words_per_topic[i],
                "review_count": int((dominant == i).sum()),
                "avg_score": avg_score_by_topic.get(topic_labels[i], None),
                "representative_reviews": representatives[topic_labels[i]],
            }
            for i in range(N_TOPICS)
        ],
        "figures": figures,
    }
    TOPICS_ARTIFACT.write_text(json.dumps(topics_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Topics artifact written to %s", TOPICS_ARTIFACT)

    df_out = df[["name", "date_parsed", "score", "sentiment", "topic_id", "topic_label", "topic_score"]]
    df_out.to_csv(ASSIGNMENTS_ARTIFACT, index=False, encoding="utf-8-sig")
    logger.info("Topic assignments written to %s", ASSIGNMENTS_ARTIFACT)

    # ── Write report ──────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Topic Modelling Report",
        "",
        f"**Dataset:** `{CLEAN_CSV}`  ",
        f"**Reviews:** {len(df)}  ",
        f"**Method:** Latent Dirichlet Allocation (LDA), {N_TOPICS} topics  ",
        f"**Vocabulary:** {len(vocab)} terms (CountVectorizer, 1–2 ngrams, min_df=3, max_df=90%)  ",
        f"**LDA perplexity:** {topics_data['lda_perplexity']}  ",
        "",
        "## Topics discovered",
        "",
        "| # | Topic label | Reviews | Avg score | Top keywords |",
        "| --- | --- | --- | --- | --- |",
    ]
    for t in topics_data["topics"]:
        kw = ", ".join(t["top_words"][:6])
        lines.append(
            f"| {t['id'] + 1} | {t['label']} | {t['review_count']} "
            f"| {t['avg_score']} | {kw} |"
        )

    lines += ["", "## Topic details", ""]
    for t in topics_data["topics"]:
        lines += [
            f"### {t['id'] + 1}. {t['label']}",
            "",
            f"**Reviews:** {t['review_count']}  |  **Avg score:** {t['avg_score']}",
            "",
            f"**Top keywords:** {', '.join(t['top_words'][:N_TOP_WORDS])}",
            "",
            "**Representative reviews:**",
            "",
        ]
        for rev in t["representative_reviews"]:
            lines.append(f'> "{rev}..."')
            lines.append("")

    lines += [
        "## Sentiment breakdown by topic",
        "",
        "| Topic | Positive | Neutral | Negative |",
        "| --- | --- | --- | --- |",
    ]
    for label in cross_pct.index:
        row = cross_pct.loc[label]
        lines.append(
            f"| {label} | {row['positive']:.1f}% | {row['neutral']:.1f}% | {row['negative']:.1f}% |"
        )

    lines += [
        "",
        "## Figures",
        "",
    ]
    for fig_path in figures:
        fname = fig_path.replace("\\", "/").split("/")[-1]
        lines.append(f"- `reports/figures/{fname}`")

    TOPIC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Topic report written to %s", TOPIC_REPORT)

    state.logs.append(
        f"Topic modelling complete: {N_TOPICS} topics discovered, "
        f"labels: {topic_labels}"
    )
    state.results["topic"] = {
        "status": "success",
        "n_topics": N_TOPICS,
        "topic_labels": topic_labels,
        "lda_perplexity": topics_data["lda_perplexity"],
        "report": str(TOPIC_REPORT),
        "topics_artifact": str(TOPICS_ARTIFACT),
        "assignments_artifact": str(ASSIGNMENTS_ARTIFACT),
    }
    return state
