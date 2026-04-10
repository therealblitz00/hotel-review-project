# Reviewer Comments — Whitepaper Draft

**Document reviewed:** `reports/whitepaper_draft.md`  
**Word count:** ~3334  
**Verdict:** **REVISE**  

> The whitepaper contains 4 factual or structural error(s) that must be corrected before publication, plus 0 warnings.

---

## Summary

| Category | Count |
| --- | --- |
| Errors (must fix) | 4 |
| Warnings (should fix) | 0 |
| Suggestions (consider) | 0 |
| Strengths | 34 |

---

## Errors (Must Fix)

### [S001] Structure: Required section missing: 'Sentiment Analysis'

**Severity:** Error  
**Message:** Required section missing: 'Sentiment Analysis'
**Detail:** Expected heading: `## 3. Sentiment Analysis`

### [S001] Structure: Required section missing: 'Topic Modelling'

**Severity:** Error  
**Message:** Required section missing: 'Topic Modelling'
**Detail:** Expected heading: `## 4. Topic Modelling`

### [S001] Structure: Required section missing: 'Strategic Recommendations'

**Severity:** Error  
**Message:** Required section missing: 'Strategic Recommendations'
**Detail:** Expected heading: `## 5. Strategic Recommendations`

### [S001] Structure: Required section missing: 'Conclusions'

**Severity:** Error  
**Message:** Required section missing: 'Conclusions'
**Detail:** Expected heading: `## 6. Conclusions`

---

## Warnings (Should Fix)

_No warnings found._

---

## Strengths

**Structure**
- Section present: 'Introduction'
- Section present: 'Data Overview'
- Section present: 'Limitations'
- Section present: 'Appendix'

**Data Overview**
- Row count (999) correctly cited.
- Mean score (8.24/10) correctly cited.
- Top reviewer country ('United Kingdom') correctly cited.
- Lowest traveler type ('Family', 7.98/10) correctly cited.

**Sentiment Analysis**
- Binary F1-macro (0.6471) correctly reported.
- CV F1-macro (0.6939) correctly cited.
- VADER baseline (0.4511) correctly cited for comparison.
- Label-signal mismatch correctly identified and explained.
- Class imbalance handling documented.

**Topic Modelling**
- Topic 'Room Comfort & Cleanliness' correctly referenced.
- Topic 'Staff & Service' correctly referenced.
- Topic 'Rooftop Bar & Ambiance' correctly referenced.
- Topic 'Food & Breakfast' correctly referenced.
- Topic 'Location & Accessibility' correctly referenced.
- Topic 'Check-in & WiFi' correctly referenced.
- Lowest topic score (7.36) for 'Room Comfort & Cleanliness' correctly cited.
- LDA perplexity (844.65) reported.
- Multilingual corpus limitation correctly disclosed.

**Recommendations**
- Recommendation R1 ('Improve WiFi Quality and Check-in Experience') present.
- Recommendation R2 ('Enhance Offering for Family Guests') present.
- Recommendation R3 ('Leverage Staff Excellence as a Core Marketing Differentiator') present.
- Recommendation R4 ('Implement a Proactive Negative-Review Response Programme') present.
- Recommendation R5 ('Grow Iberian and Continental European Market Share') present.
- Recommendation R6 ('Standardise Room Comfort and Cleanliness Across All Room Types') present.
- Recommendation R7 ('Strengthen Value Perception for Price-Sensitive Guests') present.
- KPIs present for all 7 recommendations.

**Limitations**
- Limitation correctly disclosed: 'Label-signal mismatch'.
- Limitation correctly disclosed: 'Multilingual corpus noise'.
- Limitation correctly disclosed: 'Single-property scope'.

**Appendix**
- All 20 figure filenames appear in the Appendix.

---

## Pre-Publication Checklist

Use this checklist before finalising the whitepaper:

- [ ] All error codes resolved
- [x] All warnings reviewed and addressed or documented
- [x] All 7 recommendations present with KPIs
- [x] Limitations section complete (3 limitations)
- [x] All 16 figures listed in Appendix
- [x] All pipeline artifacts listed in Appendix
- [ ] Factual numbers match pipeline artifacts
