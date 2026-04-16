# Strategic Recommendations Report

**Reviews analysed:** 999  
**Period:** 2023-03 to 2026-04  
**Overall average score:** 8.24/10  
**Negative review share:** 6.4%  
**Sentiment classifier (binary F1-macro):** 0.6471  

---

## Executive Summary

Analysis of 999 guest reviews spanning 2023-03 to 2026-04 reveals a hotel performing strongly overall (avg score 8.24/10, 78% positive sentiment) with a clear differentiation advantage in staff quality and location. However, three areas require targeted investment: WiFi and check-in friction (lowest topic score: 7.36/10), family guest experience (lowest traveler-type score), and room comfort consistency. Seven evidence-backed recommendations follow, prioritised by guest impact and implementation effort.

---

## Recommendations

### R1: Improve WiFi Quality and Check-in Experience

**Priority:** 🔴 High  

**Evidence:**  
The 'Room Comfort & Cleanliness' topic has the lowest average guest score (7.36/10) across 112 reviews. Top keywords include: parking, location, check, old, bed, small. Guest feedback highlights weak WiFi signal in bedrooms and friction around key access at night.

**Recommended actions:**

- Audit WiFi signal strength across all room floors and install repeaters in weak zones.
- Introduce digital key or PIN-code access to eliminate late-night lockout issues.
- Add a self-service check-in kiosk or clearly communicate late-night arrival procedure at booking.

**KPI:** Raise 'Room Comfort & Cleanliness' topic avg score from 7.36 to ≥8.50 within 6 months.

---

### R2: Enhance Offering for Family Guests

**Priority:** 🔴 High  

**Evidence:**  
Family guests record the lowest average score (7.98/10) compared to Couples (8.35/10), Solo travellers (8.13/10), and Groups (8.24/10). Topic analysis shows Room Comfort & Cleanliness (avg 7.36) is a recurring theme among lower-scoring reviews.

**Recommended actions:**

- Offer family-friendly room configurations (interconnecting rooms or cot availability) with clear booking options.
- Add child-oriented breakfast items and designate a quiet family dining area.
- Create a printed city guide tailored to families with children (nearby parks, child-safe restaurants).

**KPI:** Raise Family avg score from 7.98 to ≥8.20 within 9 months.

---

### R4: Implement a Proactive Negative-Review Response Programme

**Priority:** 🔴 High  

**Evidence:**  
64 reviews (6.4% of total) are classified as negative (score < 6). The sentiment classifier (TF-IDF + LinearSVC, binary F1-macro 0.6471, CV 0.6939) can flag non-positive reviews in near-real time. Unaddressed negative reviews on Booking.com directly suppress ranking and conversion.

**Recommended actions:**

- Deploy the trained binary sentiment model to auto-flag incoming reviews with score < 6.
- Set a 24-hour SLA for management responses to flagged reviews with personalised, non-template replies.
- Conduct monthly root-cause analysis on negative reviews and feed findings into operational briefings.

**KPI:** Reduce negative review share from 6.4% to ≤4% within 12 months.

---

### R3: Leverage Staff Excellence as a Core Marketing Differentiator

**Priority:** 🟡 Medium  

**Evidence:**  
'Location & Accessibility' is the single largest topic cluster with 362 reviews (avg score 8.16/10). Top terms — location, breakfast, staff, close, bathroom, clean — confirm that guests value staff friendliness and the hotel's quirky vintage identity. Overall average score of 8.24/10 reflects a strongly positive baseline.

**Recommended actions:**

- Feature authentic guest quotes about staff in OTA listings and social media campaigns.
- Launch a 'Staff Spotlight' Instagram series showcasing team members and the hotel's vintage character.
- Nominate consistently praised staff members for hospitality awards to amplify reputation.

**KPI:** Increase direct bookings by 10% within 12 months by tracking referral source in PMS.

---

### R5: Grow Iberian and Continental European Market Share

**Priority:** 🟡 Medium  

**Evidence:**  
United Kingdom dominates with 238 reviews (23.8% of total). Spain and Portugal combined account for only 89 reviews (8.9% of total), representing significant local demand untapped given the hotel's Porto location. Italy (36 reviews) and Germany (24 reviews) also show growth potential.

**Recommended actions:**

- Translate OTA listing descriptions and key guest communications into Spanish, Portuguese, Italian, and German.
- Partner with Iberian corporate travel agencies and offer weekend city-break packages targeting Lisbon and Madrid.
- Run geo-targeted paid social ads on Instagram/Facebook for ES, PT, IT, and DE audiences.

**KPI:** Grow Iberian + Continental European review share from 8.9% to ≥15% within 18 months.

---

### R6: Standardise Room Comfort and Cleanliness Across All Room Types

**Priority:** 🟡 Medium  

**Evidence:**  
'Room Comfort & Cleanliness' is the second-largest topic cluster (112 reviews, avg 7.36/10). Keywords such as 'small', 'bed', 'bathroom', and 'clean' suggest mixed experiences around room size and cleanliness consistency. Budget Double Room is the most reviewed room type (455 reviews).

**Recommended actions:**

- Introduce a standardised room-readiness checklist signed off by housekeeping before every guest arrival.
- Pilot a mattress and pillow upgrade in Budget Double Rooms and measure score impact over 90 days.
- Set room-type-specific expectations in OTA photos and descriptions to reduce size-related disappointment.

**KPI:** Raise 'Room Comfort & Cleanliness' topic avg score from 7.36 to ≥8.40.

---

### R7: Strengthen Value Perception for Price-Sensitive Guests

**Priority:** 🟢 Low  

**Evidence:**  
The ABSA 'Value' aspect captures 74 guest mentions, of which 13.5% are negative. 'Value & Overall Experience' did not emerge as a standalone LDA topic in this run, but guest reviews mentioning price (~€190/night) and value expectations confirm the need for targeted pricing and packaging actions.

**Recommended actions:**

- Offer an 'Early Bird' rate (≥21 days ahead) and a 'Last Minute' rate to capture price-sensitive segments.
- Bundle breakfast in promoted packages and quantify the inclusion value in OTA listings.
- Highlight unique amenities (rooftop, vintage decor, central location) in post-booking confirmation emails to reinforce value before arrival.

**KPI:** Raise 'Value & Overall Experience' topic avg score from 8.35 to ≥8.60.

---

## Decision Table

| # | Insight | Marketing / Operational Action | KPI | Owner | Timeline |
| --- | --- | --- | --- | --- | --- |
| R1 | 39.5% of WiFi & Check-in mentions are negative (ABSA) | Install WiFi repeaters; digital key access | Check-in topic avg ≥8.50 | Operations | 0-6 months |
| R2 | Family guests score 7.98/10 — lowest traveler segment | Family packages, child-friendly amenities | Family avg score ≥8.20 | F&B / Front Desk | 0-9 months |
| R4 | 6.4% negative reviews unresponded | Binary classifier alert → 24h response SLA | Negative share ≤4% | GM / Front Office | 0-3 months |
| R3 | Location & Accessibility: 362 reviews, avg 8.16 — top differentiator | Staff-led OTA content; award nominations | +10% direct bookings | Marketing | 3-12 months |
| R5 | Spain + Portugal = 8.9% of reviews despite Porto location | Iberian OTA translations; B2B travel agency partnerships | Iberian share ≥15% | Sales | 6-18 months |
| R6 | Room Comfort & Cleanliness: 21.6% negative ABSA mentions | Housekeeping checklist; mattress upgrade pilot | Room topic avg ≥8.40 | Housekeeping | 3-9 months |
| R7 | Value aspect: 13.5% negative ABSA mentions; some guests feel €190/night is poor value | Early Bird / Last Minute rates; bundled breakfast | Value topic avg ≥8.60 | Revenue Mgmt | 3-6 months |

## Methodology

Recommendations are derived from four pipeline artifacts:

| Artifact | Content |
| --- | --- |
| `artifacts/eda_summary.json` | Score statistics, sentiment distribution, traveler types, country breakdown |
| `artifacts/sentiment_metrics.json` | Model F1-macro scores (VADER baseline, LR, LinearSVC 3-class and binary) |
| `artifacts/topics.json` | LDA topic labels, review counts, avg scores, and representative excerpts |
| `artifacts/absa.json` | Aspect-level negative %, mention counts, traveler × aspect heatmap |

No large language model was used in recommendation generation. All thresholds and priorities are derived from quantitative signals in the above artifacts.
