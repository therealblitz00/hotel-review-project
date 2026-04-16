# Data Quality Report

**Source:** `C:\Users\manue\Desktop\mestrado\2025-2026\2Semestre\Data Mining II\hotel-review-project\data\raw\reviews_raw.csv`  
**Output:** `C:\Users\manue\Desktop\mestrado\2025-2026\2Semestre\Data Mining II\hotel-review-project\data\processed\reviews_clean.csv`  
**Rows before cleaning:** 1000  
**Rows after cleaning:** 999  
**Rows dropped:** 1  

## Transformations applied

1. Dropped `hotel_response` (100% null).
2. Parsed `nr_nights` to integer (extracted leading digit).
3. Parsed `date` (Month YYYY) → `date_parsed` (YYYY-MM), plus `year` and `month` columns.
4. Filled `pos_review` and `neg_review` nulls with empty string.
5. Created `full_review_text` = pos_review + ' ' + neg_review (stripped).
6. Detected review language (`review_language` column); translated 109 non-English reviews to English via Google Translate. Languages found: {'en': 891, 'es': 30, 'it': 20, 'fr': 18, 'pt': 16, 'de': 5, 'pl': 4, 'nl': 3, 'sv': 2, 'af': 2, 'da': 1, 'cy': 1, 'vi': 1, 'zh-cn': 1, 'ca': 1, 'ro': 1, 'cs': 1, 'el': 1, 'sk': 1}.
7. Dropped 1 row(s) with empty full_review_text.
8. Created `sentiment` label from score: ≥8 → positive, 6–7 → neutral, <6 → negative.
9. Standardised `country` to title case.

## Final schema

| Column | Dtype | Null count | Null % |
| --- | --- | --- | --- |
| name | object | 0 | 0.0% |
| country | object | 0 | 0.0% |
| room_type | object | 0 | 0.0% |
| nr_nights | int64 | 0 | 0.0% |
| date | object | 0 | 0.0% |
| traveler_type | object | 0 | 0.0% |
| title_review | object | 0 | 0.0% |
| pos_review | object | 0 | 0.0% |
| neg_review | object | 0 | 0.0% |
| score | float64 | 0 | 0.0% |
| date_parsed | object | 0 | 0.0% |
| year | Int64 | 0 | 0.0% |
| month | Int64 | 0 | 0.0% |
| full_review_text | object | 0 | 0.0% |
| review_language | object | 0 | 0.0% |
| sentiment | object | 0 | 0.0% |

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

## Review language distribution

| Language code | Count | Share |
| --- | --- | --- |
| en | 890 | 89.1% |
| es | 30 | 3.0% |
| it | 20 | 2.0% |
| fr | 18 | 1.8% |
| pt | 16 | 1.6% |
| de | 5 | 0.5% |
| pl | 4 | 0.4% |
| nl | 3 | 0.3% |
| sv | 2 | 0.2% |
| af | 2 | 0.2% |
| da | 1 | 0.1% |
| cy | 1 | 0.1% |
| vi | 1 | 0.1% |
| zh-cn | 1 | 0.1% |
| ca | 1 | 0.1% |
| ro | 1 | 0.1% |
| cs | 1 | 0.1% |
| el | 1 | 0.1% |
| sk | 1 | 0.1% |

## Review text coverage

- Rows with non-empty full_review_text: 999
- Rows with positive text only: 288
- Rows with both positive and negative text: 688
