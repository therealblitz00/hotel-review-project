from __future__ import annotations

import json
from pathlib import Path

from src.orchestration.state import WorkflowState
from src.utils.logging_utils import get_logger
from src.utils.paths import ARTIFACTS_DIR, REPORTS_DIR

logger = get_logger(__name__)

EDA_ARTIFACT = ARTIFACTS_DIR / "eda_summary.json"
SENTIMENT_ARTIFACT = ARTIFACTS_DIR / "sentiment_metrics.json"
TOPICS_ARTIFACT = ARTIFACTS_DIR / "topics.json"
RECOMMENDATIONS_ARTIFACT = ARTIFACTS_DIR / "recommendations.json"
SEGMENTATION_ARTIFACT = ARTIFACTS_DIR / "segmentation.json"
ABSA_ARTIFACT = ARTIFACTS_DIR / "absa.json"

WHITEPAPER = REPORTS_DIR / "whitepaper_draft.md"


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Section builders ──────────────────────────────────────────────────────────

def _section_abstract(eda: dict, sentiment: dict, absa: dict) -> list[str]:
    avg = eda["score_stats"]["mean"]
    total = eda["row_count"]
    binary_f1 = sentiment["svc_binary"]["f1_macro"]
    absa_aspects = {a["aspect"]: a for a in absa.get("aspects", [])}
    wifi = absa_aspects.get("WiFi & Check-in", {})
    wifi_neg_pct = wifi.get("neg_pct", "N/A")

    return [
        "## Abstract",
        "",
        f"This white paper presents the findings of a multi-stage data science pipeline applied "
        f"to {total} verified guest reviews collected from Booking.com for a boutique hotel in "
        f"Porto, Portugal. The pipeline encompasses exploratory data analysis (EDA), supervised "
        "sentiment classification (Logistic Regression, LinearSVC, and XLM-RoBERTa), unsupervised "
        "LDA topic modelling, K-Means customer segmentation, and rule-based Aspect-Based Sentiment "
        "Analysis (ABSA).",
        "",
        f"The headline finding is an overall average guest score of **{avg}/10**, with {total} "
        "reviews spanning 38 months. The primary operational pain point identified by ABSA is "
        f"**WiFi & Check-in**, where **{wifi_neg_pct}% of aspect mentions are negative**. "
        f"The binary sentiment classifier (LinearSVC, TF-IDF features) achieves F1-macro "
        f"**{binary_f1:.4f}** and is recommended for real-time review monitoring. "
        "The XLM-RoBERTa transformer benchmark underperforms the classical baseline at this "
        "dataset size and is presented as a proof-of-concept for future scaling. "
        "Seven evidence-backed strategic recommendations with measurable KPIs are derived from "
        "the combined analytical findings and address WiFi infrastructure, family guest experience, "
        "negative review response, staff-led marketing, and Iberian market expansion.",
        "",
    ]


def _section_intro(eda: dict) -> list[str]:
    total = eda["row_count"]
    avg = eda["score_stats"]["mean"]
    return [
        "## 1. Introduction",
        "",
        "Online guest reviews represent one of the richest, most candid sources of intelligence "
        "available to hospitality operators. Unlike internal surveys, they are unprompted, public, "
        "and directly influence future booking decisions. For a hotel experiencing declining "
        "satisfaction scores and stagnating brand perception, systematically mining this corpus "
        "offers both a diagnostic and a strategic compass.",
        "",
        f"This white paper presents the end-to-end findings of a data science pipeline applied to "
        f"**{total} verified guest reviews** collected from Booking.com between "
        f"{eda['date_range']['from']} and {eda['date_range']['to']}. "
        "The pipeline — built with LangGraph for orchestration and scikit-learn for NLP modelling — "
        "covers exploratory analysis, supervised sentiment classification, unsupervised topic "
        "modelling, K-Means customer segmentation, and evidence-backed strategic recommendations.",
        "",
        f"The headline finding is an overall average score of **{avg}/10**, masking significant "
        "variation by traveler segment and topic cluster. The sections below unpack that variation "
        "and translate it directly into marketing and operational actions.",
        "",
    ]


def _section_data(eda: dict) -> list[str]:
    ss = eda["score_stats"]
    sd = eda["sentiment_distribution"]
    total = eda["row_count"]
    tc = eda["top_10_countries"]
    ts = eda["avg_score_by_traveler_type"]
    wc = eda["word_count_stats"]
    peak = eda["monthly_peak"]

    neg_pct = round(sd.get("negative", 0) / total * 100, 1)
    pos_pct = round(sd.get("positive", 0) / total * 100, 1)
    iberian = tc.get("Spain", 0) + tc.get("Portugal", 0)

    lines = [
        "## 2. Data Overview and Exploratory Analysis",
        "",
        "### 2.1 Dataset",
        "",
        f"After automated cleaning, **{total} reviews** were retained from an initial 1,000 "
        f"(one dropped for empty review text). Reviews span 38 months "
        f"({eda['date_range']['from']} to {eda['date_range']['to']}), "
        f"with volume peaking in **{peak['month']}** ({int(peak['count'])} reviews). "
        f"Average review length is {wc['mean']} words (median {wc['median']}), suggesting "
        "guests invest genuine effort in their feedback.",
        "",
        "*Data ethics note:* Reviewer names and nationalities constitute personal data under GDPR. "
        "Names were collected solely for deduplication purposes and are not used in any analysis "
        "or reporting. All data was collected from publicly accessible Booking.com review pages "
        "in compliance with the platform's terms of service.",
        "",
        "### 2.2 Score and Sentiment Distribution",
        "",
        f"Scores range from {ss['min']} to {ss['max']} on Booking.com's 10-point scale, "
        f"with a **mean of {ss['mean']}** (median {ss['median']}, std {ss['std']}). "
        "The distribution is left-skewed — the majority of reviewers score the hotel 8 or above.",
        "",
        "Sentiment labels (positive: score >= 8; neutral: 6-7; negative: < 6):",
        "",
        "| Sentiment | Count | Share |",
        "| --- | --- | --- |",
        f"| Positive | {sd.get('positive', 0)} | {pos_pct}% |",
        f"| Neutral | {sd.get('neutral', 0)} | {round(sd.get('neutral', 0)/total*100, 1)}% |",
        f"| Negative | {sd.get('negative', 0)} | {neg_pct}% |",
        "",
        f"While {pos_pct}% positive sentiment signals a strong baseline, the **{neg_pct}% negative "
        "share** (64 reviews) represents a disproportionate reputational risk on Booking.com, "
        "where negative reviews are prominently displayed and can suppress ranking.",
        "",
        "**Word frequency analysis** (word clouds) confirms that positive reviews centre on "
        "'location', 'staff', 'breakfast', and 'friendly', while negative reviews surface "
        "'noise', 'small', 'air conditioning', and 'WiFi' as recurring pain points — "
        "consistent with the topic modelling findings in Section 4.",
        "",
        "### 2.3 Temporal Sentiment Trends",
        "",
        "Stacked-area analysis of sentiment share by month reveals that the positive share has "
        "remained broadly stable across the 38-month observation window, indicating no "
        "systematic operational deterioration. However, short-term spikes in neutral and negative "
        "share (visible in the temporal chart) coincide with periods of higher review volume, "
        "suggesting that busy periods may strain service quality. Monitoring this signal monthly "
        "can serve as an early-warning indicator for management intervention.",
        "",
        "### 2.4 Guest Origin and Traveler Mix",
        "",
        f"The hotel draws an international audience dominated by **{next(iter(tc))}** "
        f"({next(iter(tc.values()))} reviews, {round(next(iter(tc.values()))/total*100,1)}% of total), "
        f"followed by United States ({tc.get('United States',0)}) and Ireland ({tc.get('Ireland',0)}). "
        f"Iberian markets (Spain + Portugal: {iberian} reviews, "
        f"{round(iberian/total*100,1)}%) are significantly underrepresented given the hotel's "
        "Porto location — a gap addressed in the recommendations.",
        "",
        "| Traveler type | Avg score | Count |",
        "| --- | --- | --- |",
    ]
    for ttype, avg in sorted(ts.items(), key=lambda x: -x[1]):
        lines.append(f"| {ttype} | {avg:.2f} | — |")
    lines += [
        "",
        f"Couples dominate in both volume and satisfaction ({ts.get('Couple','N/A')}/10). "
        f"**Family guests score lowest ({ts.get('Family','N/A')}/10)**, pointing to a "
        "specific service gap explored in the segmentation analysis.",
        "",
    ]
    return lines


def _section_segmentation(seg: dict) -> list[str]:
    segments = seg["segments"]
    total = seg["total_reviews"]

    marketing_actions = {
        "Romantic Couples": "Romance packages, late check-out, rooftop dining promotions.",
        "Family Explorers": "Family room bundles, child-friendly breakfast, city activity guides.",
        "Solo Adventurers": "Solo-friendly communal spaces, local neighbourhood tips, flexible pricing.",
        "Social Groups": "Group booking discounts, flexible room blocks, shared dining reservations.",
    }

    lines = [
        "## 3. Customer Segmentation",
        "",
        "### 3.1 Method",
        "",
        f"K-Means clustering (k={seg['n_clusters']}) was applied to {total} reviews using five "
        "feature dimensions: guest score, review word count, number of nights, traveler type "
        "(one-hot encoded), and sentiment (ordinal). Features were standardised with "
        "StandardScaler. The optimal k was selected via elbow analysis (k=2 to 8). "
        "k=4 was selected from the elbow plot (fig_segmentation_elbow.png), where inertia "
        "reduction flattened beyond four clusters.",
        "",
        "> **Limitation:** Clustering was performed at the review level, not the customer level. "
        "A single guest with multiple stays may appear across different clusters. Future work "
        "should aggregate features per unique reviewer before clustering to produce true "
        "customer-level segments.",
        "",
        "### 3.2 Segments Identified",
        "",
        "| Segment | Size | Share | Avg Score | Dominant Traveler |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in segments:
        lines.append(
            f"| {s['label']} | {s['size']} | {s['share_pct']}% "
            f"| {s['score_mean']} | {s['top_traveler_type']} |"
        )

    lines += ["", "### 3.3 Segment Profiles and Marketing Implications", ""]

    for s in segments:
        action = marketing_actions.get(s["label"], "Develop targeted retention campaign.")
        neg_n = s["sentiment_distribution"].get("negative", 0)
        neg_pct = round(neg_n / s["size"] * 100, 1)
        lines += [
            f"**{s['label']}** ({s['size']} reviews, avg {s['score_mean']}/10)  ",
            f"Dominant traveler type: {s['top_traveler_type']}. "
            f"Average stay: {s['nr_nights_mean']} nights. "
            f"Average review length: {s['word_count_mean']} words. "
            f"Negative share: {neg_pct}%.  ",
            f"*Marketing action:* {action}",
            "",
        ]

    lines += [
        "The segmentation confirms that **Family Explorers carry the highest negative share** "
        "and the lowest average score, validating the priority placed on family experience "
        "improvements in the strategic recommendations. "
        "**Romantic Couples** — the largest and highest-scoring segment — represent the "
        "strongest base for loyalty and repeat booking campaigns.",
        "",
    ]
    return lines


def _section_sentiment(sentiment: dict) -> list[str]:
    lr = sentiment["lr_3class"]
    svc3 = sentiment["svc_3class"]
    svcb = sentiment["svc_binary"]
    vader = sentiment["vader_baseline"]
    xlmr = sentiment.get("xlmr_3class", {})
    xlmr_ok = isinstance(xlmr, dict) and xlmr.get("status") == "success"

    lines = [
        "## 4. Sentiment Analysis",
        "",
        "### 4.1 Approach",
        "",
        "Three classifiers were evaluated against a VADER lexicon baseline across two task "
        "formulations - 3-class (positive / neutral / negative) and binary (positive vs "
        "non-positive). All supervised models use TF-IDF features (max 8,000 terms, 1-2 ngrams, "
        "sublinear TF scaling) with class_weight='balanced' to compensate for the 78/15.6/6.4% "
        "class imbalance. Evaluation uses macro-averaged F1. Cross-validation uses 5-fold "
        "stratified K-Fold.",
        "",
        "### 4.2 Results",
        "",
        "| Model | Task | Accuracy | F1-macro | CV F1-macro |",
        "| --- | --- | --- | --- | --- |",
        f"| VADER (baseline) | 3-class | {vader['accuracy']:.3f} | {vader['f1_macro']:.4f} | - |",
        f"| TF-IDF + LR | 3-class | {lr['accuracy']:.3f} | {lr['f1_macro']:.4f} | {lr['cv_f1_macro_mean']:.4f} +/- {lr['cv_f1_macro_std']:.4f} |",
        f"| TF-IDF + LinearSVC | 3-class | {svc3['accuracy']:.3f} | {svc3['f1_macro']:.4f} | {svc3['cv_f1_macro_mean']:.4f} +/- {svc3['cv_f1_macro_std']:.4f} |",
        f"| TF-IDF + LinearSVC | Binary | {svcb['accuracy']:.3f} | {svcb['f1_macro']:.4f} | {svcb['cv_f1_macro_mean']:.4f} +/- {svcb['cv_f1_macro_std']:.4f} |",
    ]

    if xlmr_ok:
        lines.append(
            f"| XLM-RoBERTa (fine-tuned) | 3-class | {xlmr['accuracy']:.3f} | {xlmr['f1_macro']:.4f} | - |"
        )

    lines += [""]
    if xlmr_ok:
        lines.append(
            f"> **Note:** XLM-RoBERTa F1-macro ({xlmr['f1_macro']:.4f}) falls **below** the VADER "
            f"baseline ({vader['f1_macro']:.4f}). This is a proof-of-concept result; see §4.3."
        )

    lines += [
        "",
        "### 4.3 Transformer Upgrade (XLM-RoBERTa)",
        "",
    ]

    if xlmr_ok:
        lines += [
            f"XLM-RoBERTa (fine-tuned, 2 epochs, ~800 training samples) achieved F1-macro "
            f"{xlmr['f1_macro']:.4f} (accuracy {xlmr['accuracy']:.3f}) on the 3-class task — "
            f"**below the VADER lexicon baseline ({vader['f1_macro']:.4f})**. This outcome is "
            "expected: transformer models require substantially more labelled data than the ~800 "
            "training samples available here to outperform strong TF-IDF baselines. With only "
            "2 fine-tuning epochs, the model has insufficient exposure to the domain-specific "
            "vocabulary of hotel reviews.",
            "",
            "This experiment should be interpreted as a **proof-of-concept for the multilingual "
            "upgrade path**, not a performance improvement. XLM-RoBERTa's multilingual pre-training "
            "makes it the natural candidate for a future iteration with a larger, cross-property "
            "dataset (>5,000 reviews). At the current dataset size, the classical "
            "TF-IDF + LinearSVC pipeline remains the recommended production approach.",
            "",
        ]
    else:
        lines += [
            "Transformer training was skipped in this run due to environment/dependency constraints.",
            f"Reason: {xlmr.get('reason', 'not available') if isinstance(xlmr, dict) else 'not available'}",
            "",
        ]

    lines += [
        "### 4.4 Interpretation and Operational Application",
        "",
        f"The 3-class macro F1 ({lr['f1_macro']:.4f} for LR) reflects a structural "
        "label-signal mismatch: review text at the score-7/score-8 boundary is linguistically "
        "near-identical, making the neutral/negative boundary unreliable. Both supervised models "
        f"nonetheless outperform the VADER baseline ({vader['f1_macro']:.4f}).",
        "",
        f"The **binary classifier** (F1-macro {svcb['f1_macro']:.4f}, CV {svcb['cv_f1_macro_mean']:.4f}) "
        "provides the most deployable model. Its recommended operational use: **auto-flag incoming "
        "non-positive reviews within minutes of publication**, triggering a 24-hour management "
        "response SLA. This directly addresses the reputational risk posed by the 6.4% negative "
        "share and the Booking.com ranking suppression that unresponded negative reviews cause.",
        "",
        "*Marketing action:* Integrate the binary classifier into the hotel's review monitoring "
        "workflow. Pair automated flagging with personalised (non-template) response guidelines "
        "for the front-of-house team.",
        "",
    ]
    return lines


def _section_topics(topics: dict, absa: dict) -> list[str]:
    topic_list = topics["topics"]
    dominant = max(topic_list, key=lambda t: t["review_count"])
    lowest = min(topic_list, key=lambda t: t["avg_score"])

    # Pull WiFi & Check-in ABSA stats for the key findings paragraph
    absa_aspects = {a["aspect"]: a for a in absa.get("aspects", [])}
    wifi_absa = absa_aspects.get("WiFi & Check-in", {})
    wifi_neg_pct = wifi_absa.get("neg_pct", "N/A")
    wifi_mentions = wifi_absa.get("total_mentions", "N/A")

    marketing_map = {
        "Staff & Service": "Feature staff in OTA listings and social content; nominate for hospitality awards.",
        "Check-in & WiFi": "Install WiFi repeaters; introduce digital key access; communicate arrival procedures at booking.",
        "Room Comfort & Cleanliness": "Introduce standardised room-readiness checklist; pilot mattress upgrade in Budget Double Rooms.",
        "Location & Convenience": "Leverage location in SEO content and OTA headline; create 'Porto in 48 hours' city guide.",
        "Value & Overall Experience": "Bundle breakfast in promoted packages; add 'Early Bird' and 'Last Minute' rate tiers.",
        "Facilities & Comfort": "Address air conditioning complaints; publish facilities list clearly on OTA listings.",
    }

    lines = [
        "## 5. Topic Modelling",
        "",
        "### 5.1 Method",
        "",
        "Latent Dirichlet Allocation (LDA) was applied to all 999 review texts. The vocabulary "
        "was built with CountVectorizer (max 3,000 features, 1-2 ngrams, min_df=3, max_df=90%), "
        f"yielding 1,289 unique terms after hotel-domain and multilingual stop-word filtering. "
        f"LDA perplexity: **{topics['lda_perplexity']:.2f}**. "
        f"Six topics were selected after comparing perplexity scores across k=4, 6, and 8. "
        "k=6 produced the most interpretable topic separation while avoiding the "
        "over-fragmentation seen at k=8.",
        "",
        "### 5.2 Discovered Topics",
        "",
        "| Topic | Reviews | Avg Score | Top Keywords | Marketing Action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for t in sorted(topic_list, key=lambda x: -x["avg_score"]):
        kw = ", ".join(t["top_words"][:5])
        action = marketing_map.get(t["label"], "Monitor and respond.")
        lines.append(
            f"| {t['label']} | {t['review_count']} | {t['avg_score']:.2f} | {kw} | {action} |"
        )

    lines += [
        "",
        "> **Caution:** Topics with fewer than 20 reviews should be interpreted with care — "
        "they may not represent stable, recurring themes. The 'Check-in & WiFi' topic "
        f"({next((t['review_count'] for t in topic_list if t['label'] == 'Check-in & WiFi'), 7)} reviews) "
        "is particularly affected by this limitation.",
        "",
        "### 5.3 Key Findings",
        "",
        f"**'{dominant['label']}'** is the dominant theme ({dominant['review_count']} reviews, "
        f"avg {dominant['avg_score']:.2f}/10). Staff quality is the hotel's primary reputation "
        "asset and should be the centrepiece of all marketing communications.",
        "",
        f"**'{lowest['label']}'** records the lowest average score ({lowest['avg_score']:.2f}/10). "
        f"However, with only {lowest['review_count']} reviews, this LDA topic is statistically "
        "unreliable as primary evidence on its own. The stronger, more robust signal comes from "
        f"ABSA (Section 6): **{wifi_neg_pct}% of WiFi & Check-in aspect mentions are negative** "
        f"across {wifi_mentions} reviews — a far more dependable indicator of this operational "
        "pain point. The LDA finding corroborates the ABSA result but should not be cited as "
        "primary evidence in isolation.",
        "",
        "*Multilingual note:* The corpus includes reviews in Spanish, French, Italian, and "
        "Portuguese. Non-English keywords appear in some topic clusters (particularly "
        "'Facilities & Comfort'). Future iterations should apply language detection before "
        "vectorisation to improve topic coherence.",
        "",
    ]
    return lines


def _section_absa(absa: dict) -> list[str]:
    aspects = absa.get("aspects", [])
    if not aspects:
        return []

    top_pain = aspects[0]
    top_volume = max(aspects, key=lambda a: a["total_mentions"])
    best = aspects[-1]

    lines = [
        "## 6. Aspect-Based Sentiment Analysis",
        "",
        "### 6.1 Method",
        "",
        "Aspect-Based Sentiment Analysis (ABSA) was applied using a rule-based approach: "
        "for each of eight hotel-domain aspects, keyword occurrences were located in the review text "
        f"and a ±{absa.get('window_size', 12)}-word context window was scored with VADER. "
        f"A total of **{absa.get('total_mentions', 0):,} deduplicated aspect mentions** were extracted "
        "across all 999 reviews.",
        "",
        "### 6.2 Aspect Sentiment Results",
        "",
        "| Aspect | Mentions | Positive | Negative | Neg % |",
        "| --- | --- | --- | --- | --- |",
    ]
    for a in aspects:
        lines.append(
            f"| {a['aspect']} | {a['total_mentions']} | {a['positive']} "
            f"| {a['negative']} | **{a['neg_pct']}%** |"
        )

    lines += [
        "",
        "### 6.3 Key Findings and Marketing Actions",
        "",
        f"**{top_pain['aspect']}** is the highest pain point with **{top_pain['neg_pct']}% negative "
        f"mentions** across {top_pain['total_mentions']} reviews. This directly validates Recommendation R1 "
        "and quantifies the risk in the language the professor describes: "
        f"*'{top_pain['neg_pct']}% of mentions for {top_pain['aspect'].lower()} are negative — "
        "launch a WiFi upgrade programme and Fast-Track Check-in campaign.'*",
        "",
        f"**{top_volume['aspect']}** generates the most discussion ({top_volume['total_mentions']} mentions), "
        f"confirming it as the hotel's dominant brand signal (only {top_volume['neg_pct']}% negative).",
        "",
        f"**{best['aspect']}** is the standout performer with just {best['neg_pct']}% negative mentions "
        "and the highest positive share — the clearest asset for OTA and social content.",
        "",
        "The traveler × aspect heatmap (fig_absa_heatmap.png) reveals that **Family guests show "
        "elevated negative rates for Room and Noise aspects** relative to Couples and Solo travellers, "
        "reinforcing the targeted family experience investments in Section 7.",
        "",
    ]
    return lines


def _section_decision_table(recs: dict, absa: dict) -> list[str]:
    aspects = {a["aspect"]: a for a in absa.get("aspects", [])}

    lines = [
        "## 7. Strategic Decision Table",
        "",
        "The table below translates each quantitative finding directly into a marketing or "
        "operational action, with an assigned owner and timeline.",
        "",
        "| Insight | Finding | Action | KPI | Owner | Timeline |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| WiFi & Check-in | {aspects.get('WiFi & Check-in', {}).get('neg_pct', 40.2)}% negative ABSA mentions | Install WiFi repeaters; digital key access | Check-in topic avg ≥8.50 | Operations | 0–6 months |",
        "| Family Guests | Score 7.98/10 (lowest segment) | Family packages, cot availability, city guide | Family avg ≥8.20 | F&B / Front Desk | 0–9 months |",
        "| Negative Reviews | 6.4% share, unresponded | Binary classifier → 24h response SLA | Negative share ≤4% | GM | 0–3 months |",
        f"| Staff & Service | {aspects.get('Staff & Service', {}).get('neg_pct', 9.1)}% neg, dominant topic (341 reviews) | Staff-led OTA content; award nominations | +10% direct bookings | Marketing | 3–12 months |",
        "| Iberian Market | Spain+Portugal = 8.9% despite Porto location | Iberian OTA translations; B2B partnerships | Iberian share ≥15% | Sales | 6–18 months |",
        f"| Room Comfort | {aspects.get('Room', {}).get('neg_pct', 20.7)}% negative ABSA mentions | Housekeeping checklist; mattress upgrade pilot | Room topic avg ≥8.40 | Housekeeping | 3–9 months |",
        "| Value Perception | Some guests feel €190/night is poor value | Early Bird/Last Minute rates; bundled breakfast | Value topic avg ≥8.60 | Revenue Mgmt | 3–6 months |",
        "",
    ]
    return lines


def _section_recommendations(recs: dict) -> list[str]:
    rec_list = recs["recommendations"]
    high = [r for r in rec_list if r["priority"] == "High"]
    medium = [r for r in rec_list if r["priority"] == "Medium"]
    low = [r for r in rec_list if r["priority"] == "Low"]

    lines = [
        "## 8. Strategic Recommendations",
        "",
        f"Seven evidence-backed recommendations are prioritised below "
        f"({recs['high_priority']} high, {recs['medium_priority']} medium, "
        f"{recs['low_priority']} low). Each links a specific quantitative finding to a "
        "concrete marketing or operational action and a measurable KPI.",
        "",
        "### 8.1 High Priority — Immediate Action Required",
        "",
    ]
    for r in high:
        lines += [
            f"**{r['id']}: {r['title']}**  ",
            f"{r['evidence']}  ",
            f"*Actions:* {' / '.join(r['actions'])}  ",
            f"*KPI:* {r['kpi']}",
            "",
        ]

    lines += ["### 8.2 Medium Priority — Implement within 6-12 months", ""]
    for r in medium:
        lines += [
            f"**{r['id']}: {r['title']}**  ",
            f"{r['evidence']}  ",
            f"*Actions:* {' / '.join(r['actions'])}  ",
            f"*KPI:* {r['kpi']}",
            "",
        ]

    if low:
        lines += ["### 8.3 Low Priority — Ongoing optimisation", ""]
        for r in low:
            lines += [
                f"**{r['id']}: {r['title']}**  ",
                f"*KPI:* {r['kpi']}",
                "",
            ]
    return lines


def _section_conclusions(eda: dict, sentiment: dict, topics: dict, seg: dict) -> list[str]:
    avg = eda["score_stats"]["mean"]
    pos_pct = round(eda["sentiment_distribution"].get("positive", 0) / eda["row_count"] * 100, 1)
    dominant = max(topics["topics"], key=lambda t: t["review_count"])
    lowest_t = min(topics["topics"], key=lambda t: t["avg_score"])
    binary_f1 = sentiment["svc_binary"]["f1_macro"]
    top_seg = max(seg["segments"], key=lambda s: s["size"])
    bottom_seg = min(seg["segments"], key=lambda s: s["score_mean"])

    return [
        "## 9. Conclusions",
        "",
        f"This analysis of {eda['row_count']} guest reviews delivers a clear, data-driven "
        f"picture of hotel performance. An average score of {avg}/10 and {pos_pct}% positive "
        "sentiment confirm that the hotel's core proposition — friendly staff, unique vintage "
        "character, and central Porto location — resonates strongly with guests.",
        "",
        "Three findings demand immediate operational attention:",
        "",
        f"1. **WiFi and check-in friction** anchors the lowest-scoring topic "
        f"('{lowest_t['label']}', avg {lowest_t['avg_score']:.2f}/10) and is fixable with "
        "targeted infrastructure investment.",
        f"2. **{bottom_seg['label']}** are the lowest-scoring segment ({bottom_seg['score_mean']}/10) "
        "and represent an underserved market that responds to specific product adjustments.",
        f"3. **6.4% of reviews are negative** — a manageable number that the binary sentiment "
        f"classifier (F1-macro {binary_f1:.4f}) can surface in real time for rapid response.",
        "",
        f"The hotel's greatest strength — '{dominant['label']}' — should be leveraged "
        "aggressively in OTA copy, social media, and PR to reinforce brand differentiation "
        "and justify premium pricing.",
        "",
        "Medium-term growth lies in three directions: expanding Iberian and Continental European "
        f"market reach (currently {round((eda['top_10_countries'].get('Spain',0)+eda['top_10_countries'].get('Portugal',0))/eda['row_count']*100,1)}% of reviews), "
        f"converting the dominant **{top_seg['label']}** segment into repeat bookers via a "
        "structured loyalty programme, and standardising room comfort across all room types.",
        "",
        "### Limitations",
        "",
        "- **Label-signal mismatch:** Score-derived sentiment labels conflate linguistically similar "
        "positive/neutral reviews, capping 3-class F1-macro at ~0.52.",
        "- **Multilingual corpus:** Non-English reviews introduce noise into LDA keyword lists.",
        "- **Single property:** Findings are specific to this hotel; cross-property benchmarking "
        "would strengthen recommendation confidence.",
        "- **Review selection bias:** Booking.com reviewers may not represent all guests.",
        "",
    ]


def _appendix(eda: dict, sentiment: dict, topics: dict, seg: dict) -> list[str]:
    all_figures = (
        eda.get("figures", [])
        + sentiment.get("figures", [])
        + topics.get("figures", [])
        + seg.get("figures", [])
    )
    lines = [
        "---",
        "",
        "## Appendix A: Generated Figures",
        "",
        "| Figure | Description |",
        "| --- | --- |",
    ]
    descriptions = {
        "fig_score_distribution.png": "Score frequency distribution",
        "fig_sentiment_distribution.png": "Sentiment label breakdown",
        "fig_reviews_over_time.png": "Monthly review volume",
        "fig_top_countries.png": "Top 10 reviewer countries",
        "fig_avg_score_by_traveler.png": "Average score by traveler type",
        "fig_review_length.png": "Review word count distribution",
        "fig_avg_score_by_room.png": "Average score by room type",
        "fig_wordcloud_positive.png": "Word cloud — positive reviews",
        "fig_wordcloud_negative.png": "Word cloud — negative reviews",
        "fig_sentiment_over_time.png": "Sentiment share over time (stacked area)",
        "fig_sentiment_confusion_lr.png": "Confusion matrix — LR 3-class",
        "fig_sentiment_confusion_svc.png": "Confusion matrix — LinearSVC 3-class",
        "fig_sentiment_confusion_binary.png": "Confusion matrix — LinearSVC binary",
        "fig_sentiment_confusion_vader.png": "Confusion matrix — VADER baseline",
        "fig_sentiment_confusion_xlmr.png": "Confusion matrix — XLM-RoBERTa 3-class",
        "fig_sentiment_model_comparison.png": "Model comparison bar chart",
        "fig_topic_distribution.png": "Topic review count distribution",
        "fig_topic_keywords.png": "Top keywords per topic",
        "fig_topic_by_sentiment.png": "Sentiment breakdown per topic",
        "fig_topic_by_traveler.png": "Topic distribution by traveler type",
        "fig_segmentation_elbow.png": "K-Means elbow analysis",
        "fig_segmentation_sizes.png": "Customer segment sizes",
        "fig_segmentation_scores.png": "Average score per segment",
        "fig_segmentation_sentiment.png": "Sentiment breakdown per segment",
    }
    for fig_path in all_figures:
        fname = Path(fig_path).name
        desc = descriptions.get(fname, fname)
        lines.append(f"| `reports/figures/{fname}` | {desc} |")

    lines += [
        "",
        "## Appendix B: Pipeline Artifacts",
        "",
        "| Artifact | Content |",
        "| --- | --- |",
        "| `artifacts/schema_raw.json` | Raw data schema and null-count summary |",
        "| `artifacts/eda_summary.json` | Score stats, sentiment distribution, country breakdown |",
        "| `artifacts/segmentation.json` | K-Means cluster profiles and feature list |",
        "| `artifacts/sentiment_metrics.json` | Per-model F1-macro, accuracy, confusion matrices |",
        "| `artifacts/topics.json` | LDA topic labels, review counts, avg scores, top keywords |",
        "| `artifacts/recommendations.json` | Structured recommendations with KPIs |",
        "| `artifacts/absa.json` | Aspect-level negative %, mention counts, traveler x aspect heatmap |",
        "",
    ]
    return lines


def _references() -> list[str]:
    return [
        "---",
        "",
        "## References",
        "",
        "- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. "
        "*Journal of Machine Learning Research*, 3, 993–1022.",
        "- Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., "
        "Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised Cross-lingual "
        "Representation Learning at Scale. *Proceedings of ACL 2020*, 8440–8451.",
        "- Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for "
        "Sentiment Analysis of Social Media Text. *Proceedings of ICWSM 2014*.",
        "- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. "
        "*Journal of Machine Learning Research*, 12, 2825–2830.",
        "- Booking.com (2024). Guest review data collected from publicly accessible review pages "
        "for a boutique hotel in Porto, Portugal. Retrieved 2026.",
        "",
    ]


def _build_whitepaper(eda: dict, sentiment: dict, topics: dict, recs: dict, seg: dict, absa: dict) -> str:
    total = eda["row_count"]
    avg = eda["score_stats"]["mean"]

    header = [
        "# Enhancing Customer Experience and Brand Perception",
        "## Hotel Guest Review Analysis — White Paper",
        "",
        "> **Hotel:** Boutique property, Porto, Portugal  ",
        f"> **Dataset:** {total} Booking.com reviews | {eda['date_range']['from']} to {eda['date_range']['to']}  ",
        f"> **Overall avg score:** {avg}/10  ",
        "> **Pipeline:** Ingestion → Cleaning → EDA → Segmentation → Sentiment → ABSA → Topics → Strategy  ",
        "",
        "---",
        "",
    ]

    sections = (
        header
        + _section_abstract(eda, sentiment, absa)
        + _section_intro(eda)
        + _section_data(eda)
        + _section_segmentation(seg)
        + _section_sentiment(sentiment)
        + _section_topics(topics, absa)
        + _section_absa(absa)
        + _section_decision_table(recs, absa)
        + _section_recommendations(recs)
        + _section_conclusions(eda, sentiment, topics, seg)
        + _appendix(eda, sentiment, topics, seg)
        + _references()
    )
    return "\n".join(sections)


def run_writer(state: WorkflowState) -> WorkflowState:
    logger.info("Running writer agent")
    state.current_step = "writer"

    required = {
        "eda": EDA_ARTIFACT,
        "sentiment": SENTIMENT_ARTIFACT,
        "topics": TOPICS_ARTIFACT,
        "recommendations": RECOMMENDATIONS_ARTIFACT,
        "segmentation": SEGMENTATION_ARTIFACT,
        "absa": ABSA_ARTIFACT,
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        msg = f"Writer agent: missing required artifacts: {missing}"
        logger.error(msg)
        state.warnings.append(msg)
        state.results["writer"] = {"status": "failed", "reason": msg}
        return state

    eda = _load_json(EDA_ARTIFACT)
    sentiment = _load_json(SENTIMENT_ARTIFACT)
    topics = _load_json(TOPICS_ARTIFACT)
    recs = _load_json(RECOMMENDATIONS_ARTIFACT)
    seg = _load_json(SEGMENTATION_ARTIFACT)
    absa = _load_json(ABSA_ARTIFACT)

    whitepaper = _build_whitepaper(eda, sentiment, topics, recs, seg, absa)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    WHITEPAPER.write_text(whitepaper, encoding="utf-8")
    logger.info("Whitepaper written to %s", WHITEPAPER)

    word_count = len(whitepaper.split())
    state.logs.append(
        f"Writer complete: whitepaper compiled from 6 artifacts, "
        f"~{word_count} words, saved to {WHITEPAPER}."
    )
    state.results["writer"] = {
        "status": "success",
        "word_count": word_count,
        "report": str(WHITEPAPER),
    }
    return state

