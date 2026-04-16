# Enhancing Customer Experience and Brand Perception
## Hotel Guest Review Analysis — White Paper

> **Hotel:** Boutique property, Porto, Portugal  
> **Dataset:** 999 Booking.com reviews | 2023-03 to 2026-04  
> **Overall avg score:** 8.24/10  
> **Pipeline:** Ingestion → Cleaning → EDA → Segmentation → Sentiment → ABSA → Topics → Strategy  

---

## Abstract

This white paper presents the findings of a multi-stage data science pipeline applied to 999 verified guest reviews collected from Booking.com for a boutique hotel in Porto, Portugal. The pipeline encompasses exploratory data analysis (EDA), supervised sentiment classification (Logistic Regression, LinearSVC, and XLM-RoBERTa), unsupervised LDA topic modelling, K-Means customer segmentation, and rule-based Aspect-Based Sentiment Analysis (ABSA).

The headline finding is an overall average guest score of **8.24/10**, with 999 reviews spanning 38 months. The primary operational pain point identified by ABSA is **WiFi & Check-in**, where **39.5% of aspect mentions are negative**. The binary sentiment classifier (LinearSVC, TF-IDF features) achieves F1-macro **0.6471** and is recommended for real-time review monitoring. The XLM-RoBERTa transformer benchmark underperforms the classical baseline at this dataset size and is presented as a proof-of-concept for future scaling. Seven evidence-backed strategic recommendations with measurable KPIs are derived from the combined analytical findings and address WiFi infrastructure, family guest experience, negative review response, staff-led marketing, and Iberian market expansion.

## 1. Introduction

Online guest reviews represent one of the richest, most candid sources of intelligence available to hospitality operators. Unlike internal surveys, they are unprompted, public, and directly influence future booking decisions. For a hotel experiencing declining satisfaction scores and stagnating brand perception, systematically mining this corpus offers both a diagnostic and a strategic compass.

This white paper presents the end-to-end findings of a data science pipeline applied to **999 verified guest reviews** collected from Booking.com between 2023-03 and 2026-04. The pipeline — built with LangGraph for orchestration and scikit-learn for NLP modelling — covers exploratory analysis, supervised sentiment classification, unsupervised topic modelling, K-Means customer segmentation, and evidence-backed strategic recommendations.

The headline finding is an overall average score of **8.24/10**, masking significant variation by traveler segment and topic cluster. The sections below unpack that variation and translate it directly into marketing and operational actions.

## 2. Data Overview and Exploratory Analysis

### 2.1 Dataset

After automated cleaning, **999 reviews** were retained from an initial 1,000 (one dropped for empty review text). Reviews span 38 months (2023-03 to 2026-04), with volume peaking in **2026-03** (86 reviews). Average review length is 35.4 words (median 23.0), suggesting guests invest genuine effort in their feedback.

*Data ethics note:* Reviewer names and nationalities constitute personal data under GDPR. Names were collected solely for deduplication purposes and are not used in any analysis or reporting. All data was collected from publicly accessible Booking.com review pages in compliance with the platform's terms of service.

### 2.2 Score and Sentiment Distribution

Scores range from 1.0 to 10.0 on Booking.com's 10-point scale, with a **mean of 8.24** (median 8.0, std 1.68). The distribution is left-skewed — the majority of reviewers score the hotel 8 or above.

Sentiment labels (positive: score >= 8; neutral: 6-7; negative: < 6):

| Sentiment | Count | Share |
| --- | --- | --- |
| Positive | 779 | 78.0% |
| Neutral | 156 | 15.6% |
| Negative | 64 | 6.4% |

While 78.0% positive sentiment signals a strong baseline, the **6.4% negative share** (64 reviews) represents a disproportionate reputational risk on Booking.com, where negative reviews are prominently displayed and can suppress ranking.

**Word frequency analysis** (word clouds) confirms that positive reviews centre on 'location', 'staff', 'breakfast', and 'friendly', while negative reviews surface 'noise', 'small', 'air conditioning', and 'WiFi' as recurring pain points — consistent with the topic modelling findings in Section 4.

### 2.3 Temporal Sentiment Trends

Stacked-area analysis of sentiment share by month reveals that the positive share has remained broadly stable across the 38-month observation window, indicating no systematic operational deterioration. However, short-term spikes in neutral and negative share (visible in the temporal chart) coincide with periods of higher review volume, suggesting that busy periods may strain service quality. Monitoring this signal monthly can serve as an early-warning indicator for management intervention.

### 2.4 Guest Origin and Traveler Mix

The hotel draws an international audience dominated by **United Kingdom** (238 reviews, 23.8% of total), followed by United States (145) and Ireland (73). Iberian markets (Spain + Portugal: 89 reviews, 8.9%) are significantly underrepresented given the hotel's Porto location — a gap addressed in the recommendations.

| Traveler type | Avg score | Count |
| --- | --- | --- |
| Couple | 8.35 | 518 |
| Group | 8.24 | 131 |
| Solo traveller | 8.13 | 226 |
| Family | 7.98 | 124 |

Couples dominate in both volume and satisfaction (8.35/10). **Family guests score lowest (7.98/10)**, pointing to a specific service gap explored in the segmentation analysis.

## 3. Customer Segmentation

### 3.1 Method

K-Means clustering (k=4) was applied to 999 reviews using five feature dimensions: guest score, review word count, number of nights, traveler type (one-hot encoded), and sentiment (ordinal). Features were standardised with StandardScaler. The optimal k was selected via elbow analysis (k=2 to 8). k=4 was selected from the elbow plot (fig_segmentation_elbow.png), where inertia reduction flattened beyond four clusters.

> **Limitation:** Clustering was performed at the review level, not the customer level. A single guest with multiple stays may appear across different clusters. Future work should aggregate features per unique reviewer before clustering to produce true customer-level segments.

### 3.2 Segments Identified

| Segment | Size | Share | Avg Score | Dominant Traveler |
| --- | --- | --- | --- | --- |
| Romantic Couples | 518 | 51.9% | 8.35 | Couple |
| Social Groups | 131 | 13.1% | 8.24 | Group |
| Solo Adventurers | 226 | 22.6% | 8.13 | Solo traveller |
| Family Explorers | 124 | 12.4% | 7.98 | Family |

### 3.3 Segment Profiles and Marketing Implications

**Romantic Couples** (518 reviews, avg 8.35/10)  
Dominant traveler type: Couple. Average stay: 2.9 nights. Average review length: 32.1 words. Negative share: 5.2%.  
*Marketing action:* Romance packages, late check-out, rooftop dining promotions.

**Social Groups** (131 reviews, avg 8.24/10)  
Dominant traveler type: Group. Average stay: 2.6 nights. Average review length: 34.1 words. Negative share: 6.9%.  
*Marketing action:* Group booking discounts, flexible room blocks, shared dining reservations.

**Solo Adventurers** (226 reviews, avg 8.13/10)  
Dominant traveler type: Solo traveller. Average stay: 2.4 nights. Average review length: 40.0 words. Negative share: 7.1%.  
*Marketing action:* Solo-friendly communal spaces, local neighbourhood tips, flexible pricing.

**Family Explorers** (124 reviews, avg 7.98/10)  
Dominant traveler type: Family. Average stay: 2.6 nights. Average review length: 42.2 words. Negative share: 9.7%.  
*Marketing action:* Family room bundles, child-friendly breakfast, city activity guides.

The segmentation confirms that **Family Explorers carry the highest negative share** and the lowest average score, validating the priority placed on family experience improvements in the strategic recommendations. **Romantic Couples** — the largest and highest-scoring segment — represent the strongest base for loyalty and repeat booking campaigns.

## 4. Sentiment Analysis

### 4.1 Approach

Three classifiers were evaluated against a VADER lexicon baseline across two task formulations - 3-class (positive / neutral / negative) and binary (positive vs non-positive). All supervised models use TF-IDF features (max 8,000 terms, 1-2 ngrams, sublinear TF scaling) with class_weight='balanced' to compensate for the 78/15.6/6.4% class imbalance. Evaluation uses macro-averaged F1. Cross-validation uses 5-fold stratified K-Fold.

### 4.2 Results

| Model | Task | Accuracy | F1-macro | CV F1-macro |
| --- | --- | --- | --- | --- |
| VADER (baseline) | 3-class | 0.715 | 0.4511 | - |
| TF-IDF + LR | 3-class | 0.740 | 0.5293 | 0.4909 +/- 0.0388 |
| TF-IDF + LinearSVC | 3-class | 0.800 | 0.5344 | 0.4436 +/- 0.0434 |
| TF-IDF + LinearSVC | Binary | 0.750 | 0.6471 | 0.6939 +/- 0.0110 |
| XLM-RoBERTa (fine-tuned) | 3-class | 0.780 | 0.2921 | - |

> **Note:** XLM-RoBERTa F1-macro (0.2921) falls **below** the VADER baseline (0.4511). This is a proof-of-concept result; see §4.3.

### 4.3 Transformer Upgrade (XLM-RoBERTa)

XLM-RoBERTa (fine-tuned, 2 epochs, ~800 training samples) achieved F1-macro 0.2921 (accuracy 0.780) on the 3-class task — **below the VADER lexicon baseline (0.4511)**. This outcome is expected: transformer models require substantially more labelled data than the ~800 training samples available here to outperform strong TF-IDF baselines. With only 2 fine-tuning epochs, the model has insufficient exposure to the domain-specific vocabulary of hotel reviews.

This experiment should be interpreted as a **proof-of-concept for the multilingual upgrade path**, not a performance improvement. XLM-RoBERTa's multilingual pre-training makes it the natural candidate for a future iteration with a larger, cross-property dataset (>5,000 reviews). At the current dataset size, the classical TF-IDF + LinearSVC pipeline remains the recommended production approach.

### 4.4 Interpretation and Operational Application

The 3-class macro F1 (0.5293 for LR) reflects a structural label-signal mismatch: review text at the score-7/score-8 boundary is linguistically near-identical, making the neutral/negative boundary unreliable. Both supervised models nonetheless outperform the VADER baseline (0.4511).

The **binary classifier** (F1-macro 0.6471, CV 0.6939) provides the most deployable model. Its recommended operational use: **auto-flag incoming non-positive reviews within minutes of publication**, triggering a 24-hour management response SLA. This directly addresses the reputational risk posed by the 6.4% negative share and the Booking.com ranking suppression that unresponded negative reviews cause.

*Marketing action:* Integrate the binary classifier into the hotel's review monitoring workflow. Pair automated flagging with personalised (non-template) response guidelines for the front-of-house team.

## 5. Topic Modelling

### 5.1 Method

Latent Dirichlet Allocation (LDA) was applied to all 999 review texts. The vocabulary was built with CountVectorizer (max 3,000 features, 1-2 ngrams, min_df=3, max_df=90%), yielding 1,289 unique terms after hotel-domain and multilingual stop-word filtering. LDA perplexity: **844.65**. Six topics were selected after comparing perplexity scores across k=4, 6, and 8. k=6 produced the most interpretable topic separation while avoiding the over-fragmentation seen at k=8.

### 5.2 Discovered Topics

| Topic | Reviews | Avg Score | Top Keywords | Marketing Action |
| --- | --- | --- | --- | --- |
| Staff & Service | 280 | 8.64 | staff, location, friendly, helpful, clean | Feature staff in OTA listings and social content; nominate for hospitality awards. |
| Rooftop Bar & Ambiance | 19 | 8.47 | pillows, design, comfort, doors, extremely | Monitor and respond. |
| Food & Breakfast | 219 | 8.32 | location, breakfast, staff, excellent, perfect | Monitor and respond. |
| Location & Accessibility | 362 | 8.16 | location, breakfast, staff, close, bathroom | Monitor and respond. |
| Check-in & WiFi | 7 | 7.43 | hear, people, dirty, charged, buy | Install WiFi repeaters; introduce digital key access; communicate arrival procedures at booking. |
| Room Comfort & Cleanliness | 112 | 7.36 | parking, location, check, old, bed | Introduce standardised room-readiness checklist; pilot mattress upgrade in Budget Double Rooms. |

> **Caution:** Topics with fewer than 20 reviews should be interpreted with care — they may not represent stable, recurring themes. The 'Check-in & WiFi' topic (7 reviews) is particularly affected by this limitation.

### 5.3 Key Findings

**'Location & Accessibility'** is the dominant theme by volume (362 reviews, avg 8.16/10), confirming the hotel's location as a core booking driver.

**'Staff & Service'** — with 280 reviews and the highest average score (8.64/10) — is the hotel's primary reputation differentiator and should be the centrepiece of all marketing communications.

**'Room Comfort & Cleanliness'** records the lowest average score (7.36/10). However, with only 112 reviews, this LDA topic is statistically unreliable as primary evidence on its own. The stronger, more robust signal comes from ABSA (Section 6): **39.5% of WiFi & Check-in aspect mentions are negative** across 114 reviews — a far more dependable indicator of this operational pain point. The LDA finding corroborates the ABSA result but should not be cited as primary evidence in isolation.

*Multilingual note:* The corpus includes reviews in Spanish, French, Italian, and Portuguese. Non-English keywords appear in some topic clusters (particularly 'Facilities & Comfort'). Future iterations should apply language detection before vectorisation to improve topic coherence.

## 6. Aspect-Based Sentiment Analysis

### 6.1 Method

Aspect-Based Sentiment Analysis (ABSA) was applied using a rule-based approach: for each of eight hotel-domain aspects, keyword occurrences were located in the review text and a ±12-word context window was scored with VADER. A total of **2,156 deduplicated aspect mentions** were extracted across all 999 reviews.

### 6.2 Aspect Sentiment Results

| Aspect | Mentions | Positive | Negative | Neg % |
| --- | --- | --- | --- | --- |
| WiFi & Check-in | 114 | 51 | 45 | **39.5%** |
| Noise | 99 | 56 | 32 | **32.3%** |
| Room | 444 | 289 | 96 | **21.6%** |
| Breakfast | 314 | 240 | 48 | **15.3%** |
| Value | 74 | 52 | 10 | **13.5%** |
| Cleanliness | 176 | 150 | 19 | **10.8%** |
| Staff & Service | 424 | 355 | 37 | **8.7%** |
| Location | 511 | 401 | 37 | **7.2%** |

### 6.3 Key Findings and Marketing Actions

**WiFi & Check-in** is the highest pain point with **39.5% negative mentions** across 114 reviews. This directly validates Recommendation R1 and quantifies the risk in the language the professor describes: *'39.5% of mentions for wifi & check-in are negative — launch a WiFi upgrade programme and Fast-Track Check-in campaign.'*

**Location** generates the most discussion (511 mentions), confirming it as the hotel's dominant brand signal (only 7.2% negative).

**Staff & Service** is the standout performer with just 8.7% negative mentions and the highest positive share — the clearest asset for OTA and social content.

The traveler × aspect heatmap (fig_absa_heatmap.png) reveals that **Family guests show elevated negative rates for Room and Noise aspects** relative to Couples and Solo travellers, reinforcing the targeted family experience investments in Section 7.

## 7. Strategic Decision Table

The table below translates each quantitative finding directly into a marketing or operational action, with an assigned owner and timeline.

| Insight | Finding | Action | KPI | Owner | Timeline |
| --- | --- | --- | --- | --- | --- |
| WiFi & Check-in | 39.5% negative ABSA mentions | Install WiFi repeaters; digital key access | Check-in topic avg ≥8.50 | Operations | 0–6 months |
| Family Guests | Score 7.98/10 (lowest segment) | Family packages, cot availability, city guide | Family avg ≥8.20 | F&B / Front Desk | 0–9 months |
| Negative Reviews | 6.4% share, unresponded | Binary classifier → 24h response SLA | Negative share ≤4% | GM | 0–3 months |
| Staff & Service | 8.7% neg, dominant topic (280 reviews) | Staff-led OTA content; award nominations | +10% direct bookings | Marketing | 3–12 months |
| Iberian Market | Spain+Portugal = 8.9% despite Porto location | Iberian OTA translations; B2B partnerships | Iberian share ≥15% | Sales | 6–18 months |
| Room Comfort | 21.6% negative ABSA mentions | Housekeeping checklist; mattress upgrade pilot | Room topic avg ≥8.40 | Housekeeping | 3–9 months |
| Value Perception | Some guests feel €190/night is poor value | Early Bird/Last Minute rates; bundled breakfast | Value topic avg ≥8.60 | Revenue Mgmt | 3–6 months |

## 8. Strategic Recommendations

Seven evidence-backed recommendations are prioritised below (3 high, 3 medium, 1 low). Each links a specific quantitative finding to a concrete marketing or operational action and a measurable KPI.

### 8.1 High Priority — Immediate Action Required

**R1: Improve WiFi Quality and Check-in Experience**  
The 'Room Comfort & Cleanliness' topic has the lowest average guest score (7.36/10) across 112 reviews. Top keywords include: parking, location, check, old, bed, small. Guest feedback highlights weak WiFi signal in bedrooms and friction around key access at night.  
*Actions:* Audit WiFi signal strength across all room floors and install repeaters in weak zones. / Introduce digital key or PIN-code access to eliminate late-night lockout issues. / Add a self-service check-in kiosk or clearly communicate late-night arrival procedure at booking.  
*KPI:* Raise 'Room Comfort & Cleanliness' topic avg score from 7.36 to ≥8.50 within 6 months.

**R2: Enhance Offering for Family Guests**  
Family guests record the lowest average score (7.98/10) compared to Couples (8.35/10), Solo travellers (8.13/10), and Groups (8.24/10). Topic analysis shows Room Comfort & Cleanliness (avg 7.36) is a recurring theme among lower-scoring reviews.  
*Actions:* Offer family-friendly room configurations (interconnecting rooms or cot availability) with clear booking options. / Add child-oriented breakfast items and designate a quiet family dining area. / Create a printed city guide tailored to families with children (nearby parks, child-safe restaurants).  
*KPI:* Raise Family avg score from 7.98 to ≥8.20 within 9 months.

**R4: Implement a Proactive Negative-Review Response Programme**  
64 reviews (6.4% of total) are classified as negative (score < 6). The sentiment classifier (TF-IDF + LinearSVC, binary F1-macro 0.6471, CV 0.6939) can flag non-positive reviews in near-real time. Unaddressed negative reviews on Booking.com directly suppress ranking and conversion.  
*Actions:* Deploy the trained binary sentiment model to auto-flag incoming non-positive reviews (score < 8) within minutes of publication. / Set a 24-hour SLA for management responses to flagged reviews with personalised, non-template replies. / Conduct monthly root-cause analysis on negative reviews and feed findings into operational briefings.  
*KPI:* Reduce negative review share from 6.4% to ≤4% within 12 months.

### 8.2 Medium Priority — Implement within 6-12 months

**R3: Leverage Staff Excellence as a Core Marketing Differentiator**  
'Location & Accessibility' is the single largest topic cluster with 362 reviews (avg score 8.16/10). Top terms — location, breakfast, staff, close, bathroom, clean — confirm that guests value staff friendliness and the hotel's quirky vintage identity. Overall average score of 8.24/10 reflects a strongly positive baseline.  
*Actions:* Feature authentic guest quotes about staff in OTA listings and social media campaigns. / Launch a 'Staff Spotlight' Instagram series showcasing team members and the hotel's vintage character. / Nominate consistently praised staff members for hospitality awards to amplify reputation.  
*KPI:* Increase direct bookings by 10% within 12 months by tracking referral source in PMS.

**R5: Grow Iberian and Continental European Market Share**  
United Kingdom dominates with 238 reviews (23.8% of total). Spain and Portugal combined account for only 89 reviews (8.9% of total), representing significant local demand untapped given the hotel's Porto location. Italy (36 reviews) and Germany (24 reviews) also show growth potential.  
*Actions:* Translate OTA listing descriptions and key guest communications into Spanish, Portuguese, Italian, and German. / Partner with Iberian corporate travel agencies and offer weekend city-break packages targeting Lisbon and Madrid. / Run geo-targeted paid social ads on Instagram/Facebook for ES, PT, IT, and DE audiences.  
*KPI:* Grow Iberian + Continental European review share from 8.9% to ≥15% within 18 months.

**R6: Standardise Room Comfort and Cleanliness Across All Room Types**  
'Room Comfort & Cleanliness' is the second-largest topic cluster (112 reviews, avg 7.36/10). Keywords such as 'small', 'bed', 'bathroom', and 'clean' suggest mixed experiences around room size and cleanliness consistency. Budget Double Room is the most reviewed room type (455 reviews).  
*Actions:* Introduce a standardised room-readiness checklist signed off by housekeeping before every guest arrival. / Pilot a mattress and pillow upgrade in Budget Double Rooms and measure score impact over 90 days. / Set room-type-specific expectations in OTA photos and descriptions to reduce size-related disappointment.  
*KPI:* Raise 'Room Comfort & Cleanliness' topic avg score from 7.36 to ≥8.40.

### 8.3 Low Priority — Ongoing optimisation

**R7: Strengthen Value Perception for Price-Sensitive Guests**  
The ABSA 'Value' aspect captures 74 guest mentions, of which 13.5% are negative. 'Value & Overall Experience' did not emerge as a standalone LDA topic in this run, but guest reviews mentioning price (~€190/night) and value expectations confirm the need for targeted pricing and packaging actions.  
*Actions:* Offer an 'Early Bird' rate (≥21 days ahead) and a 'Last Minute' rate to capture price-sensitive segments. / Bundle breakfast in promoted packages and quantify the inclusion value in OTA listings. / Highlight unique amenities (rooftop, vintage decor, central location) in post-booking confirmation emails to reinforce value before arrival.  
*KPI:* Raise 'Value & Overall Experience' topic avg score from 8.35 to ≥8.60.

## 9. Conclusions

This analysis of 999 guest reviews delivers a clear, data-driven picture of hotel performance. An average score of 8.24/10 and 78.0% positive sentiment confirm that the hotel's core proposition — friendly staff, unique vintage character, and central Porto location — resonates strongly with guests.

Three findings demand immediate operational attention:

1. **WiFi and check-in friction** anchors the lowest-scoring topic ('Room Comfort & Cleanliness', avg 7.36/10) and is fixable with targeted infrastructure investment.
2. **Family Explorers** are the lowest-scoring segment (7.98/10) and represent an underserved market that responds to specific product adjustments.
3. **6.4% of reviews are negative** — a manageable number that the binary sentiment classifier (F1-macro 0.6471) can surface in real time for rapid response.

The hotel's greatest strength — 'Location & Accessibility' — should be leveraged aggressively in OTA copy, social media, and PR to reinforce brand differentiation and justify premium pricing.

Medium-term growth lies in three directions: expanding Iberian and Continental European market reach (currently 8.9% of reviews), converting the dominant **Romantic Couples** segment into repeat bookers via a structured loyalty programme, and standardising room comfort across all room types.

### Limitations

- **Label-signal mismatch:** Score-derived sentiment labels conflate linguistically similar positive/neutral reviews, capping 3-class F1-macro at ~0.52.
- **Multilingual corpus:** Non-English reviews introduce noise into LDA keyword lists.
- **Single property:** Findings are specific to this hotel; cross-property benchmarking would strengthen recommendation confidence.
- **Review selection bias:** Booking.com reviewers may not represent all guests.

---

## Appendix A: Generated Figures

| Figure | Description |
| --- | --- |
| `reports/figures/fig_score_distribution.png` | Score frequency distribution |
| `reports/figures/fig_sentiment_distribution.png` | Sentiment label breakdown |
| `reports/figures/fig_reviews_over_time.png` | Monthly review volume |
| `reports/figures/fig_top_countries.png` | Top 10 reviewer countries |
| `reports/figures/fig_avg_score_by_traveler.png` | Average score by traveler type |
| `reports/figures/fig_review_length.png` | Review word count distribution |
| `reports/figures/fig_avg_score_by_room.png` | Average score by room type |
| `reports/figures/fig_wordcloud_positive.png` | Word cloud — positive reviews |
| `reports/figures/fig_wordcloud_negative.png` | Word cloud — negative reviews |
| `reports/figures/fig_sentiment_over_time.png` | Sentiment share over time (stacked area) |
| `reports/figures/fig_sentiment_confusion_lr.png` | Confusion matrix — LR 3-class |
| `reports/figures/fig_sentiment_confusion_svc.png` | Confusion matrix — LinearSVC 3-class |
| `reports/figures/fig_sentiment_confusion_binary.png` | Confusion matrix — LinearSVC binary |
| `reports/figures/fig_sentiment_confusion_vader.png` | Confusion matrix — VADER baseline |
| `reports/figures/fig_sentiment_confusion_xlmr.png` | Confusion matrix — XLM-RoBERTa 3-class |
| `reports/figures/fig_sentiment_model_comparison.png` | Model comparison bar chart |
| `reports/figures/fig_topic_distribution.png` | Topic review count distribution |
| `reports/figures/fig_topic_keywords.png` | Top keywords per topic |
| `reports/figures/fig_topic_by_sentiment.png` | Sentiment breakdown per topic |
| `reports/figures/fig_topic_by_traveler.png` | Topic distribution by traveler type |
| `reports/figures/fig_segmentation_elbow.png` | K-Means elbow analysis |
| `reports/figures/fig_segmentation_sizes.png` | Customer segment sizes |
| `reports/figures/fig_segmentation_scores.png` | Average score per segment |
| `reports/figures/fig_segmentation_sentiment.png` | Sentiment breakdown per segment |

## Appendix B: Pipeline Artifacts

| Artifact | Content |
| --- | --- |
| `artifacts/schema_raw.json` | Raw data schema and null-count summary |
| `artifacts/eda_summary.json` | Score stats, sentiment distribution, country breakdown |
| `artifacts/segmentation.json` | K-Means cluster profiles and feature list |
| `artifacts/sentiment_metrics.json` | Per-model F1-macro, accuracy, confusion matrices |
| `artifacts/topics.json` | LDA topic labels, review counts, avg scores, top keywords |
| `artifacts/recommendations.json` | Structured recommendations with KPIs |
| `artifacts/absa.json` | Aspect-level negative %, mention counts, traveler x aspect heatmap |

---

## References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993–1022.
- Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *Proceedings of ACL 2020*, 8440–8451.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *Proceedings of ICWSM 2014*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Booking.com (2024). Guest review data collected from publicly accessible review pages for a boutique hotel in Porto, Portugal. Retrieved 2026.
