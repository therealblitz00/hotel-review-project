# EDA Report

**Dataset:** `C:\Users\manue\Desktop\mestrado\DAII\hotel-review-project\data\processed\reviews_clean.csv`  
**Reviews:** 999  
**Period:** 2023-03 to 2026-04  

## Key observations

- The dataset contains 999 reviews spanning 38 months (2023-03 to 2026-04).
- The overall average score is 8.24 out of 10 (median 8.0), indicating a strongly positive guest experience.
- Sentiment breakdown: 779 positive (78.0%), 156 neutral (15.6%), 64 negative (6.4%).
- The most common reviewer origin is United Kingdom (238 reviews, 23.8% of total).
- Review volume peaked in 2026-03 (86 reviews).
- Average review length is 35.4 words (median 23.0); the longest review has 363 words.
- Couples represent the largest traveler segment (518 reviews).
- Budget Double Room is the most reviewed room type (455 reviews).

## Score distribution

| Score | Count |
| --- | --- |
| 1 | 11 |
| 2 | 7 |
| 3 | 6 |
| 4 | 14 |
| 5 | 26 |
| 6 | 42 |
| 7 | 114 |
| 8 | 300 |
| 9 | 239 |
| 10 | 240 |

- Mean: 8.24  |  Median: 8.0  |  Std: 1.68

## Sentiment distribution

| Sentiment | Count | % |
| --- | --- | --- |
| positive | 779 | 78.0% |
| neutral | 156 | 15.6% |
| negative | 64 | 6.4% |

## Top 10 reviewer countries

| Country | Count |
| --- | --- |
| United Kingdom | 238 |
| United States | 145 |
| Ireland | 73 |
| Canada | 67 |
| Spain | 50 |
| Portugal | 39 |
| Italy | 36 |
| Australia | 30 |
| Germany | 24 |
| Netherlands | 20 |

## Average score by traveler type

| Traveler type | Avg score | Count |
| --- | --- | --- |
| Couple | 8.35 | 518 |
| Family | 7.98 | 124 |
| Group | 8.24 | 131 |
| Solo traveller | 8.13 | 226 |

## Review length

- Mean word count: 35.4
- Median word count: 23.0
- Max word count: 363

## Figures

- `reports/figures/fig_score_distribution.png`
- `reports/figures/fig_sentiment_distribution.png`
- `reports/figures/fig_reviews_over_time.png`
- `reports/figures/fig_top_countries.png`
- `reports/figures/fig_avg_score_by_traveler.png`
- `reports/figures/fig_review_length.png`
- `reports/figures/fig_avg_score_by_room.png`
- `reports/figures/fig_wordcloud_positive.png`
- `reports/figures/fig_wordcloud_negative.png`
- `reports/figures/fig_sentiment_over_time.png`
