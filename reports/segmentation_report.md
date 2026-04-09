# Customer Segmentation Report

**Method:** K-Means clustering (k=4)  
**Reviews:** 999  
**Features:** score, word_count, nr_nights, sentiment_ord, traveler_Couple, traveler_Family, traveler_Group, traveler_Solo traveller  

## Segment Overview

| Segment | Size | Share | Avg Score | Top Traveler | Top Sentiment |
| --- | --- | --- | --- | --- | --- |
| Romantic Couples | 518 | 51.9% | 8.35 | Couple | positive |
| Social Groups | 131 | 13.1% | 8.24 | Group | positive |
| Solo Adventurers | 226 | 22.6% | 8.13 | Solo traveller | positive |
| Family Explorers | 124 | 12.4% | 7.98 | Family | positive |

## Segment Profiles

### Romantic Couples

- **Size:** 518 reviews (51.9% of total)
- **Average score:** 8.35 ± 1.53
- **Avg stay:** 2.9 nights
- **Avg review length:** 31.9 words
- **Dominant traveler type:** Couple
- **Dominant sentiment:** positive
- **Top reviewer country:** United Kingdom

**Sentiment distribution:**

- Positive: 419 (80.9%)
- Neutral: 72 (13.9%)
- Negative: 27 (5.2%)

### Social Groups

- **Size:** 131 reviews (13.1% of total)
- **Average score:** 8.24 ± 1.74
- **Avg stay:** 2.6 nights
- **Avg review length:** 33.8 words
- **Dominant traveler type:** Group
- **Dominant sentiment:** positive
- **Top reviewer country:** United Kingdom

**Sentiment distribution:**

- Positive: 104 (79.4%)
- Neutral: 18 (13.7%)
- Negative: 9 (6.9%)

### Solo Adventurers

- **Size:** 226 reviews (22.6% of total)
- **Average score:** 8.13 ± 1.8
- **Avg stay:** 2.4 nights
- **Avg review length:** 39.9 words
- **Dominant traveler type:** Solo traveller
- **Dominant sentiment:** positive
- **Top reviewer country:** United Kingdom

**Sentiment distribution:**

- Positive: 169 (74.8%)
- Neutral: 41 (18.1%)
- Negative: 16 (7.1%)

### Family Explorers

- **Size:** 124 reviews (12.4% of total)
- **Average score:** 7.98 ± 1.97
- **Avg stay:** 2.6 nights
- **Avg review length:** 42.2 words
- **Dominant traveler type:** Family
- **Dominant sentiment:** positive
- **Top reviewer country:** United States

**Sentiment distribution:**

- Positive: 87 (70.2%)
- Neutral: 25 (20.2%)
- Negative: 12 (9.7%)

## Marketing Implications

| Segment | Priority Action |
| --- | --- |
| Romantic Couples | Target with romance packages, late check-out perks, and couples dining offers. |
| Social Groups | Offer group booking discounts, flexible room configurations, and group dining options. |
| Solo Adventurers | Highlight solo-friendly amenities, social spaces, and local tips in communications. |
| Family Explorers | Promote family packages with child-friendly amenities and city guides. |

## Figures

- `reports/figures/fig_segmentation_elbow.png`
- `reports/figures/fig_segmentation_sizes.png`
- `reports/figures/fig_segmentation_scores.png`
- `reports/figures/fig_segmentation_sentiment.png`
