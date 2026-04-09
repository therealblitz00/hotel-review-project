# Data Quality Report

**Source:** `C:\Users\manue\Desktop\mestrado\DAII\hotel-review-project\data\raw\reviews_raw.csv`  
**Output:** `C:\Users\manue\Desktop\mestrado\DAII\hotel-review-project\data\processed\reviews_clean.csv`  
**Rows before cleaning:** 1000  
**Rows after cleaning:** 999  
**Rows dropped:** 1  

## Transformations applied

1. Dropped `hotel_response` (100% null).
2. Parsed `nr_nights` to integer (extracted leading digit).
3. Parsed `date` (Month YYYY) → `date_parsed` (YYYY-MM), plus `year` and `month` columns.
4. Filled `pos_review` and `neg_review` nulls with empty string.
5. Created `full_review_text` = pos_review + ' ' + neg_review (stripped).
6. Dropped 1 row(s) with empty full_review_text.
7. Created `sentiment` label from score: ≥8 → positive, 6–7 → neutral, <6 → negative.
8. Standardised `country` to title case.

## Final schema

| Column | Dtype | Null count | Null % |
| --- | --- | --- | --- |
| name | str | 0 | 0.0% |
| country | str | 0 | 0.0% |
| room_type | str | 0 | 0.0% |
| nr_nights | int64 | 0 | 0.0% |
| date | str | 0 | 0.0% |
| traveler_type | str | 0 | 0.0% |
| title_review | str | 0 | 0.0% |
| pos_review | str | 0 | 0.0% |
| neg_review | str | 0 | 0.0% |
| score | float64 | 0 | 0.0% |
| date_parsed | str | 0 | 0.0% |
| year | Int64 | 0 | 0.0% |
| month | Int64 | 0 | 0.0% |
| full_review_text | str | 0 | 0.0% |
| sentiment | str | 0 | 0.0% |

## Score distribution

- Min: 1.0
- Max: 10.0
- Mean: 8.24
- Median: 8.0

## Sentiment label distribution

| Sentiment | Count |
| --- | --- |
| positive | 779 (78.0%) |
| neutral | 156 (15.6%) |
| negative | 64 (6.4%) |

## Review text coverage

- Rows with non-empty full_review_text: 999
- Rows with positive text only: 288
- Rows with both positive and negative text: 688
