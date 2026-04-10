# Pitch Script — 5-Minute Presentation
## "Enhancing Customer Experience and Brand Perception"
### Hotel Review Data Science Pipeline — Porto Boutique Hotel

---

## Slide Outline (8 slides × ~35 seconds each)

---

### SLIDE 1 — Title (0:00–0:15)

**Visual:** Hotel photo + pipeline diagram (Ingestion → EDA → Segmentation → Sentiment → ABSA → Topics → Strategy)

**Script:**
> "Good morning. Today we present a full data science pipeline applied to 999 real Booking.com
> reviews from a boutique hotel in Porto. Our goal: turn raw guest feedback into specific,
> evidence-backed marketing and operational actions."

---

### SLIDE 2 — The Problem (0:15–0:45)

**Visual:** Two KPIs side by side — Average score 8.24/10 (green) vs Negative share 6.4% (red)

**Script:**
> "The hotel scores 8.24 out of 10 overall — that looks good. But the headline hides the problem.
> 6.4% of reviews are negative. On Booking.com, unresponded negative reviews suppress ranking
> and directly reduce conversion. And when we break down scores by traveler type,
> Family guests score just 7.98 — the lowest of any segment. The hotel is leaving money
> and repeat bookings on the table."

---

### SLIDE 3 — Data & EDA (0:45–1:15)

**Visual:** Word clouds side by side (positive left, negative right) + sentiment-over-time chart

**Script:**
> "We scraped and cleaned 999 reviews spanning 38 months. The word clouds tell the story
> immediately: positive reviews centre on 'staff', 'location', 'breakfast' — strong brand assets.
> Negative reviews surface 'noise', 'WiFi', 'air conditioning', 'small room'.
> The sentiment trend chart shows the positive share has been stable — no systemic decline —
> but volume spikes correlate with short-term quality dips. That's an early warning signal
> we can monitor monthly."

---

### SLIDE 4 — Customer Segmentation (1:15–1:45)

**Visual:** Segment bar chart (4 segments: Romantic Couples 518, Solo Adventurers 226, Social Groups 131, Family Explorers 124) with avg score per segment

**Script:**
> "K-Means clustering identifies four distinct guest personas. Romantic Couples dominate —
> 518 reviews, scoring 8.35 on average. They're the loyalty base.
> Family Explorers are the smallest segment but the lowest scoring at 7.98,
> with the highest negative rate. These guests need a different product:
> connected rooms, child-friendly breakfast, a family city guide.
> Each segment gets a targeted marketing action."

---

### SLIDE 5 — Aspect-Based Sentiment Analysis (1:45–2:30)

**Visual:** ABSA negative % bar chart (aspects sorted by negative %) + heatmap (traveler × aspect)

**Script:**
> "This is where we answer the professor's challenge: translate findings into marketing actions.
>
> WiFi and Check-in: 40.2% of mentions are negative. That is the single biggest pain point
> in the entire corpus. The action is concrete: install WiFi repeaters, introduce digital key
> access, and launch a Fast-Track Check-in campaign.
>
> Noise: 32.1% negative — the second highest. This points to soundproofing investment
> and room allocation policies for light sleepers.
>
> By contrast, Location scores just 7.5% negative — a genuine brand strength to amplify.
> Staff & Service: 9.1% negative across 406 mentions — the hotel's clearest differentiator.
>
> The heatmap shows Family guests have elevated negative rates for Room and Noise — validating
> the segmentation finding."

---

### SLIDE 6 — Sentiment Modelling (2:30–3:00)

**Visual:** Model comparison table (VADER, LR, LinearSVC 3-class, LinearSVC binary)

**Script:**
> "We trained and evaluated four models. The binary classifier — positive vs non-positive —
> achieves F1-macro 0.6548, cross-validated at 0.7019. This is the model we deploy.
> Its operational role: auto-flag any incoming non-positive review within minutes of
> publication, triggering a 24-hour management response SLA.
> That single workflow addresses the reputational risk of the 6.4% negative share."

---

### SLIDE 7 — Decision Table (3:00–4:00)

**Visual:** Full decision table (Insight | Finding | Action | KPI | Owner | Timeline)

**Script:**
> "Here is the core deliverable: seven evidence-backed actions, each linked to a specific
> data finding, a measurable KPI, an owner, and a timeline.
>
> Three high-priority items: fix WiFi and check-in within 6 months — KPI: topic avg score
> from 7.63 to 8.50. Improve family experience within 9 months — family avg from 7.98 to 8.20.
> Deploy the sentiment classifier this quarter — reduce negative share from 6.4% to under 4%.
>
> Medium term: leverage staff excellence in marketing, expand into Iberian markets,
> standardise room cleanliness. Each action is justified by a number, not an opinion."

---

### SLIDE 8 — Conclusions & Next Steps (4:00–5:00)

**Visual:** Summary infographic — 3 strengths (Staff, Location, Value) + 3 gaps (WiFi, Families, Noise)

**Script:**
> "To summarise: this hotel has real strengths — staff, location, and breakfast are mentioned
> positively in hundreds of reviews. These should be front and centre in all OTA listings
> and social content.
>
> But three specific gaps are suppressing scores: WiFi and check-in friction, the family
> experience, and noise. All three are fixable with targeted investment.
>
> The pipeline we've built — 11 agents, 27 figures, 8 reports — runs end-to-end from raw
> Booking.com data to a prioritised action plan in under 30 seconds. It can be re-run monthly
> to track whether scores are improving.
>
> Thank you. We're happy to take questions."

---

## Speaker Notes

- **Total runtime:** 5 minutes exactly at a measured pace
- **Emphasis slides:** 5 (ABSA) and 7 (Decision Table) — spend the most time here
- **Figures to show:** word clouds, ABSA bar chart, ABSA heatmap, decision table
- **Key stat to memorise:** "40.2% of WiFi mentions are negative" — this is your hook
- **Backup question answer:** "Why not use GPT-4 for sentiment?" — Answer: rule-based ABSA gives
  interpretable, auditable outputs that a hotel GM can act on directly without black-box outputs

---

## Slide Design Checklist

- [ ] Slide 1: Title, hotel image, pipeline flow
- [ ] Slide 2: Two large KPI numbers (8.24 green, 6.4% red)
- [ ] Slide 3: Word clouds side-by-side + time series chart
- [ ] Slide 4: Horizontal bar chart of 4 segments with scores
- [ ] Slide 5: ABSA negative % bar + heatmap
- [ ] Slide 6: Model comparison table (4 rows)
- [ ] Slide 7: Decision table (7 rows)
- [ ] Slide 8: 3-column summary (strengths | gaps | actions)
