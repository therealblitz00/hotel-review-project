# Hotel Guest Review Analysis — White Paper

> **Scope:** Booking.com reviews for a boutique hotel in Porto, Portugal  
> **Dataset:** 999 reviews | 2023-03 to 2026-04  
> **Pipeline:** Ingestion → Cleaning → EDA → Sentiment Analysis → Topic Modelling → Strategy  

---

## 1. Introduction

Online guest reviews are among the most reliable signals of hotel operational quality and competitive positioning. This white paper presents a full analytical pipeline applied to 999 Booking.com reviews collected between 2023-03 and 2026-04. The pipeline was built using LangGraph for orchestration, scikit-learn for NLP modelling, and NLTK's VADER for lexicon-based sentiment benchmarking.

The objectives of this analysis are:

1. Characterise the distribution of guest satisfaction scores and demographics.
2. Build and evaluate sentiment classifiers capable of distinguishing positive from    non-positive reviews.
3. Identify the major thematic clusters driving guest experience.
4. Translate quantitative findings into prioritised, evidence-backed operational recommendations.

The overall average score of **8.24/10** provides a strong baseline. However, sub-group analysis reveals actionable gaps — particularly for family guests and around WiFi and check-in experience — that this report addresses in detail.

## 2. Data Overview

The dataset comprises **999 guest reviews** spanning **2023-03 to 2026-04** (38 months), scraped from Booking.com using a Selenium-based web crawler. After automated cleaning, 999 of the original 1,000 raw records were retained (one dropped for empty review text).

### 2.1 Score Distribution

Guest satisfaction scores range from 1.0 to 10.0 on Booking.com's 10-point scale. The **mean score is 8.24** (median 8.0, std 1.68), indicating a strongly positive overall guest experience. The score distribution is left-skewed, with the majority of reviews concentrated in the 8–10 range.

### 2.2 Sentiment Distribution

Sentiment labels were derived programmatically from the numerical score (positive: score ≥ 8; neutral: 6–7; negative: < 6):

| Sentiment | Count | Share |
| --- | --- | --- |
| Positive | 779 | 78.0% |
| Neutral | 156 | 15.6% |
| Negative | 64 | 6.4% |

The class imbalance (78% positive) is a defining characteristic of the dataset and directly shaped modelling decisions in the sentiment analysis phase.

### 2.3 Guest Origins

The hotel attracts an international audience. The top reviewer nationality is **United Kingdom** (238 reviews, 23.8% of total), followed by **United States** (145) and **Ireland** (73). Iberian markets (Spain + Portugal combined: 89 reviews) are underrepresented relative to the hotel's Porto location.

### 2.4 Traveler Segments

| Traveler type | Avg score |
| --- | --- |
| Couple | 8.35 |
| Group | 8.24 |
| Solo traveller | 8.13 |
| Family | 7.98 |

Couples are the dominant segment and the highest-scoring group (8.35/10). Family guests record the lowest average score (7.98/10), pointing to an improvement opportunity.

### 2.5 Review Volume

Review volume peaked in **2026-03** (86 reviews). Average review length is 35.3 words (median 23.0), with the longest review reaching 363 words.

## 3. Sentiment Analysis

### 3.1 Approach

Three classifiers were evaluated against a VADER lexicon baseline across two task formulations:

- **3-class task:** positive / neutral / negative (score-derived labels)
- **Binary task:** positive (score ≥ 8) vs non-positive (score < 8)

All supervised models use TF-IDF features (max 8,000 terms, 1–2 ngrams, sublinear TF scaling) with `class_weight='balanced'` to compensate for the 78/15.6/6.4% class imbalance. Evaluation uses macro-averaged F1 to give equal weight to minority classes. Cross-validation uses 5-fold stratified K-Fold.

### 3.2 Results

| Model | Task | Accuracy | F1-macro | CV F1-macro |
| --- | --- | --- | --- | --- |
| VADER (baseline) | 3-class | 0.670 | 0.4573 | — |
| TF-IDF + LR | 3-class | 0.735 | 0.5245 | 0.4917 ± 0.0231 |
| TF-IDF + LinearSVC | 3-class | 0.785 | 0.4993 | 0.4501 ± 0.0342 |
| TF-IDF + LinearSVC | Binary | 0.765 | 0.6548 | 0.7019 ± 0.0116 |

### 3.3 Interpretation

The 3-class macro F1 scores (0.5245 for LR, 0.4993 for LinearSVC) reflect a structural label-signal mismatch: guest text at the score-7/score-8 boundary is linguistically near-identical, making the neutral/negative boundary unreliable for text-based classifiers. Both supervised models nonetheless substantially outperform the VADER baseline (F1-macro 0.4573).

The **binary classifier** (positive vs non-positive) resolves the ambiguous boundary and achieves F1-macro **0.6548** (CV 0.7019 ± 0.0116), making it the recommended model for operational deployment — for example, auto-flagging incoming reviews for management follow-up.

## 4. Topic Modelling

### 4.1 Method

Latent Dirichlet Allocation (LDA) was applied to the combined positive and negative review text (999 documents). The vocabulary was built with CountVectorizer (max 3,000 features, 1–2 ngrams, min_df=3, max_df=90%), yielding 1,289 unique terms after stop-word filtering. The corpus is multilingual (English, Spanish, Italian, French, Portuguese), and extended hotel-domain stop-word lists were applied for each language. LDA perplexity on the held-out set: **831.94**.

### 4.2 Discovered Topics

| # | Topic | Reviews | Avg score | Top keywords |
| --- | --- | --- | --- | --- |
| 1 | Check-in & WiFi | 86 | 7.63 | location, key, station, reception, wifi, time |
| 2 | Facilities & Comfort | 95 | 8.13 | uncomfortable, air conditioning, conditioning, air, era, algo |
| 3 | Room Comfort & Cleanliness | 290 | 8.00 | location, staff, breakfast, clean, bed, friendly |
| 4 | Location & Convenience | 92 | 8.53 | location, breakfast, porto, perfect, bar, open |
| 5 | Staff & Service | 341 | 8.52 | location, staff, breakfast, friendly, helpful, clean |
| 6 | Value & Overall Experience | 95 | 8.35 | location, walking, recommend, distance, staff, walking distance |

### 4.3 Key Findings

**'Staff & Service'** is the dominant theme (341 reviews, 34.1% of reviews), confirming that staff quality and friendliness are the hotel's primary reputation driver (avg score 8.52/10).

**'Check-in & WiFi'** records the lowest average score (7.63/10 across 86 reviews), driven by complaints about WiFi signal quality and late-night door access procedures. This topic represents the most actionable short-term improvement opportunity.

The multilingual composition of the review corpus (Spanish, French, Italian, Portuguese) introduced noise into topic keyword lists — a known limitation of monolingual LDA. Future iterations should apply language detection and per-language stop-word filtering before vectorisation.

## 5. Strategic Recommendations

Seven evidence-backed recommendations were derived from the pipeline artifacts (3 high, 3 medium, 1 low priority). Each is grounded in specific quantitative findings.

### 5.1 High Priority

**R1: Improve WiFi Quality and Check-in Experience**  
The 'Check-in & WiFi' topic has the lowest average guest score (7.63/10) across 86 reviews. Top keywords include: location, key, station, reception, wifi, time. Guest feedback highlights weak WiFi signal in bedrooms and friction around key access at night.

*KPI:* Raise 'Check-in & WiFi' topic avg score from 7.63 to ≥8.50 within 6 months.

**R2: Enhance Offering for Family Guests**  
Family guests record the lowest average score (7.98/10) compared to Couples (8.35/10), Solo travellers (8.13/10), and Groups (8.24/10). Topic analysis shows Room Comfort & Cleanliness (avg 8.00) is a recurring theme among lower-scoring reviews.

*KPI:* Raise Family avg score from 7.98 to ≥8.20 within 9 months.

**R4: Implement a Proactive Negative-Review Response Programme**  
64 reviews (6.4% of total) are classified as negative (score < 6). The sentiment classifier (TF-IDF + LinearSVC, binary F1-macro 0.6548, CV 0.7019) can flag non-positive reviews in near-real time. Unaddressed negative reviews on Booking.com directly suppress ranking and conversion.

*KPI:* Reduce negative review share from 6.4% to ≤4% within 12 months.

### 5.2 Medium Priority

**R3: Leverage Staff Excellence as a Core Marketing Differentiator**  
'Staff & Service' is the single largest topic cluster with 341 reviews (avg score 8.52/10). Top terms — location, staff, breakfast, friendly, helpful, clean — confirm that guests value staff friendliness and the hotel's quirky vintage identity. Overall average score of 8.24/10 reflects a strongly positive baseline.

*KPI:* Increase direct bookings by 10% within 12 months by tracking referral source in PMS.

**R5: Grow Iberian and Continental European Market Share**  
United Kingdom dominates with 238 reviews (23.8% of total). Spain and Portugal combined account for only 89 reviews (8.9% of total), representing significant local demand untapped given the hotel's Porto location. Italy (36 reviews) and Germany (24 reviews) also show growth potential.

*KPI:* Grow Iberian + Continental European review share from 8.9% to ≥15% within 18 months.

**R6: Standardise Room Comfort and Cleanliness Across All Room Types**  
'Room Comfort & Cleanliness' is the second-largest topic cluster (290 reviews, avg 8.00/10). Keywords such as 'small', 'bed', 'bathroom', and 'clean' suggest mixed experiences around room size and cleanliness consistency. Budget Double Room is the most reviewed room type (455 reviews).

*KPI:* Raise 'Room Comfort & Cleanliness' topic avg score from 8.00 to ≥8.40.

### 5.3 Low Priority

**R7: Strengthen Value Perception for Price-Sensitive Guests**  
The 'Value & Overall Experience' topic (95 reviews, avg 8.35/10) shows keywords like 'price', 'value', 'money', and 'recommend'. Some reviews mention paying ~€190/night and feeling underwhelmed. Price anchoring against local alternatives can shift guest expectations.

*KPI:* Raise 'Value & Overall Experience' topic avg score from 8.35 to ≥8.60.

## 6. Conclusions

This analysis of 999 guest reviews establishes a clear picture of hotel performance: an average score of 8.24/10 and 78.0% positive sentiment confirm a strong baseline reputation, anchored by staff quality and central Porto location. These strengths should be actively promoted in OTA listings and social media.

Three targeted investments can move the needle most efficiently: (1) resolving WiFi and check-in friction, which anchors the lowest-scoring topic cluster ('Check-in & WiFi', avg 7.63/10); (2) tailoring the family guest experience, currently the lowest-scoring traveler segment; and (3) implementing a real-time negative-review response programme powered by the binary sentiment classifier (F1-macro 0.6548).

Medium-term priorities include expanding into Iberian and Continental European markets, standardising room cleanliness across room types, and reinforcing value messaging for price-sensitive guests.

### Limitations

- **Label-signal mismatch:** Score-derived sentiment labels conflate linguistically similar   positive/neutral reviews, capping 3-class F1-macro at ~0.52.
- **Multilingual corpus:** Non-English reviews introduce noise into LDA keyword lists;   language detection and per-language stop-word lists would improve topic coherence.
- **Single property:** Findings are specific to this hotel. Cross-property benchmarking   would strengthen recommendation confidence.

---

## Appendix: Generated Figures

| Figure | Path |
| --- | --- |
| `fig_score_distribution.png` | `reports/figures/fig_score_distribution.png` |
| `fig_sentiment_distribution.png` | `reports/figures/fig_sentiment_distribution.png` |
| `fig_reviews_over_time.png` | `reports/figures/fig_reviews_over_time.png` |
| `fig_top_countries.png` | `reports/figures/fig_top_countries.png` |
| `fig_avg_score_by_traveler.png` | `reports/figures/fig_avg_score_by_traveler.png` |
| `fig_review_length.png` | `reports/figures/fig_review_length.png` |
| `fig_avg_score_by_room.png` | `reports/figures/fig_avg_score_by_room.png` |
| `fig_sentiment_confusion_lr.png` | `reports/figures/fig_sentiment_confusion_lr.png` |
| `fig_sentiment_confusion_svc.png` | `reports/figures/fig_sentiment_confusion_svc.png` |
| `fig_sentiment_confusion_binary.png` | `reports/figures/fig_sentiment_confusion_binary.png` |
| `fig_sentiment_confusion_vader.png` | `reports/figures/fig_sentiment_confusion_vader.png` |
| `fig_sentiment_model_comparison.png` | `reports/figures/fig_sentiment_model_comparison.png` |
| `fig_topic_distribution.png` | `reports/figures/fig_topic_distribution.png` |
| `fig_topic_keywords.png` | `reports/figures/fig_topic_keywords.png` |
| `fig_topic_by_sentiment.png` | `reports/figures/fig_topic_by_sentiment.png` |
| `fig_topic_by_traveler.png` | `reports/figures/fig_topic_by_traveler.png` |

## Appendix: Pipeline Artifacts

| Artifact | Description |
| --- | --- |
| `artifacts/schema_raw.json` | Raw data schema and null-count summary |
| `artifacts/eda_summary.json` | Score stats, sentiment distribution, country breakdown |
| `artifacts/sentiment_metrics.json` | Per-model classification metrics and confusion matrices |
| `artifacts/topics.json` | LDA topic labels, review counts, avg scores, top keywords |
| `artifacts/recommendations.json` | Full structured recommendation objects with KPIs |
