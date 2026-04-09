# PROJECT_TASK_PLAN.md

**Created:** 2026-04-09
**Based on:** Repository audit — all tasks reflect actual current state.

Tasks are sequenced in dependency order. Do not start a task until its dependencies are complete.

---

## Task 1 — Fix scrapper.py

**Type:** Foundational
**Objective:** Make the scraper reliable, Chrome-compatible, and properly integrated with the project's directory structure before any data is collected.

**Why needed:** The scraper has three issues that block use: `selenium` is missing from `requirements.txt`, the default browser is Firefox (project requires Chrome), and output goes to the working directory instead of `data/raw/`.

**Dependencies:** None — this is the first task.

**Files involved:**
- `scrapper.py`
- `requirements.txt`

**Changes required:**
1. Add `selenium` to `requirements.txt`.
2. Change default browser in `build_driver()` from `"firefox"` to `"chrome"` (line 77: `os.environ.get("SCRAPER_BROWSER", "firefox")` → `"chrome"`).
3. Change default `--out` path in `main()` from `"reviews_paodeacucar_porto.csv"` to `"data/raw/reviews_raw.csv"` (or accept as CLI arg but document the standard path).
4. Ensure `data/raw/` directory exists before writing (add `Path(out_csv).parent.mkdir(parents=True, exist_ok=True)` before `df.to_csv`).

**Driver management decision:** Keep **Selenium Manager** (built into Selenium 4.6+). Do NOT add `webdriver-manager`. Selenium Manager auto-downloads the matching ChromeDriver for the installed Chrome version, requires no extra package, and is the officially maintained solution. `webdriver-manager` is a third-party alternative that adds a dependency without benefit for this project.

**What "complete" means:**
- `selenium` appears in `requirements.txt`.
- Running `python scrapper.py --max-pages 1` with Chrome installed produces a non-empty CSV at `data/raw/reviews_raw.csv`.
- The script prints `[scraper]` progress lines and exits cleanly.

**Validation:**
```
python scrapper.py --max-pages 1 --out data/raw/reviews_raw.csv
# Expect: ≥1 row in data/raw/reviews_raw.csv with columns: name, country, room_type, nr_nights, date, traveler_type, title_review, pos_review, neg_review, hotel_response, score
```

---

## Task 2 — Obtain raw review data

**Type:** Foundational
**Objective:** Run the fixed scraper to collect all available reviews for the target hotel.

**Why needed:** No downstream analysis can begin without data. The scraper must be run and its output verified before ingestion.

**Dependencies:** Task 1 complete.

**Files involved:**
- `scrapper.py`
- `data/raw/reviews_raw.csv` (to be created)

**Steps:**
1. Determine the total number of review pages available for the hotel (Booking.com shows ≈10 reviews per page).
2. Run: `python scrapper.py --max-pages <N> --out data/raw/reviews_raw.csv`
3. Inspect output CSV: check row count, column completeness, and encoding.

**What "complete" means:**
- `data/raw/reviews_raw.csv` exists with ≥ 50 rows (ideally all available reviews).
- All 11 columns present: `name`, `country`, `room_type`, `nr_nights`, `date`, `traveler_type`, `title_review`, `pos_review`, `neg_review`, `hotel_response`, `score`.
- No encoding errors; UTF-8 BOM format confirmed.

**Validation:**
```python
import pandas as pd
df = pd.read_csv("data/raw/reviews_raw.csv", encoding="utf-8-sig")
assert len(df) > 0
assert set(["name","country","score","pos_review"]).issubset(df.columns)
print(df.shape, df.dtypes)
```

**Blocker to document:** If Booking.com blocks the scraper (CAPTCHA, IP block), document the blocker in `PROJECT_STATUS.md` and consider a manual fallback (export from browser, use a VPN, or use a dataset from a public source).

---

## Task 3 — Data ingestion agent

**Type:** Foundational
**Objective:** Implement the real ingestion node: load raw CSV, validate schema, produce ingestion report.

**Why needed:** The orchestration graph currently has a dummy ingestion node. The ingestion agent is the gateway for all data into the pipeline and must validate the schema before downstream steps.

**Dependencies:** Task 2 complete (raw data exists).

**Files involved:**
- `src/orchestration/graph.py` (replace dummy `ingestion_node`)
- `src/agents/ingestion.py` (create)
- `data/raw/reviews_raw.csv` (input)
- `reports/ingestion_report.md` (output)
- `artifacts/schema_raw.json` (output)

**What "complete" means:**
- `ingestion_node` loads `data/raw/reviews_raw.csv` using `src/utils/paths.py::RAW_DIR`.
- Validates that all expected columns are present.
- Produces `reports/ingestion_report.md` with: row count, column list, null counts per column, date range, score distribution.
- Produces `artifacts/schema_raw.json` with column names and inferred types.
- Node writes these paths to `WorkflowState.results["ingestion"]`.

**Validation:**
- Run `python main.py` and verify `reports/ingestion_report.md` is non-empty and `artifacts/schema_raw.json` exists.

---

## Task 4 — Data cleaning agent

**Type:** Foundational
**Objective:** Standardize the raw dataset, handle missing values, parse types, enrich where possible.

**Why needed:** Raw scraped data is noisy. Scores may be strings, dates are inconsistent, text fields may have extra whitespace or placeholder values. Cleaning is required before any NLP or modelling step.

**Dependencies:** Task 3 complete (ingestion report approved, raw schema confirmed).

**Files involved:**
- `src/orchestration/graph.py` (replace dummy `cleaning_node`)
- `src/agents/cleaning.py` (create)
- `data/raw/reviews_raw.csv` (input)
- `data/processed/reviews_clean.csv` (output)
- `reports/data_quality_report.md` (output)
- `artifacts/features_summary.json` (output)

**Cleaning steps to implement:**
1. Parse `score` column to float (handle comma-as-decimal from Booking).
2. Parse `date` column to a consistent datetime or string format.
3. Parse `nr_nights` to integer.
4. Concatenate `pos_review` + `neg_review` into a `full_review_text` column.
5. Drop rows where both `pos_review` and `neg_review` are empty.
6. Strip whitespace from all string columns.
7. Standardize `country` casing.
8. Map `score` to a sentiment label (e.g., ≥8 → positive, 6–7 → neutral, <6 → negative).

**What "complete" means:**
- `data/processed/reviews_clean.csv` exists with clean types and `full_review_text` column.
- `reports/data_quality_report.md` is non-empty with: rows before/after, null rates, transformations applied.
- `artifacts/features_summary.json` lists all columns and their types/statistics.

**Validation:**
```python
import pandas as pd
df = pd.read_csv("data/processed/reviews_clean.csv")
assert df["score"].dtype in [float, "float64"]
assert "full_review_text" in df.columns
assert df["full_review_text"].notna().sum() > 0
```

---

## Task 5 — EDA agent

**Type:** Core
**Objective:** Produce descriptive statistics, distributions, and exploratory findings from the clean dataset.

**Why needed:** EDA reveals patterns that inform modelling decisions. It is also required for the white paper's descriptive section.

**Dependencies:** Task 4 complete (clean dataset approved).

**Files involved:**
- `src/orchestration/graph.py` (replace dummy `eda_node`)
- `src/agents/eda.py` (create)
- `data/processed/reviews_clean.csv` (input)
- `reports/eda_report.md` (output)
- `reports/figures/` (visualizations)
- `artifacts/eda_summary.json` (output)

**EDA items to produce:**
- Score distribution histogram.
- Review count by country (top 10).
- Review count over time (by month/year).
- Average score by traveler type.
- Review length distribution (word count).
- Null/missing rate summary.
- Key observations section.

**What "complete" means:**
- `reports/eda_report.md` is non-empty with at least 5 findings.
- `reports/figures/` contains at least 3 saved plots.
- `artifacts/eda_summary.json` contains key numeric summaries.

---

## Task 6 — Sentiment analysis agent

**Type:** Core
**Objective:** Train and evaluate a sentiment classifier on the review data.

**Why needed:** Sentiment classification is a stated project deliverable. The `score` field provides a natural label (positive/neutral/negative).

**Dependencies:** Task 4 complete (clean dataset approved). Task 5 can run in parallel.

**Files involved:**
- `src/agents/sentiment.py` (create)
- `data/processed/reviews_clean.csv` (input)
- `reports/sentiment_report.md` (output)
- `artifacts/sentiment_metrics.json` (output)
- `artifacts/sentiment_predictions.csv` (output)

**What "complete" means:**
- At least one classifier trained (e.g., Logistic Regression on TF-IDF, or VADER for baseline).
- `reports/sentiment_report.md` includes: accuracy, precision, recall, F1 per class; confusion matrix; example predictions.
- `artifacts/sentiment_metrics.json` contains numeric scores.
- `artifacts/sentiment_predictions.csv` contains per-review predictions.

---

## Task 7 — Topic modelling agent

**Type:** Core
**Objective:** Discover major themes in the review text.

**Why needed:** Topic modelling reveals what guests consistently praise or criticise — a key input for the strategy layer.

**Dependencies:** Task 4 complete (clean dataset). Task 5/6 may run in parallel.

**Files involved:**
- `src/agents/topic.py` (create)
- `data/processed/reviews_clean.csv` (input)
- `reports/topic_report.md` (output)
- `artifacts/topics.json` (output)
- `artifacts/topic_assignments.csv` (output)

**What "complete" means:**
- At least one topic model run (e.g., LDA with 5–10 topics, or BERTopic).
- `reports/topic_report.md` includes: topic labels with top keywords; representative reviews per topic; interpretation section.
- `artifacts/topics.json` contains topic keywords and weights.
- `artifacts/topic_assignments.csv` contains per-review topic assignment.

---

## Task 8 — Strategy agent

**Type:** Core
**Objective:** Translate EDA, sentiment, and topic findings into actionable customer experience and digital marketing recommendations.

**Why needed:** The project's output is recommendations, not just analysis. This layer bridges findings to business actions.

**Dependencies:** Tasks 5, 6, and 7 complete and approved.

**Files involved:**
- `src/agents/strategy.py` (create)
- Approved reports from Tasks 5–7 (inputs)
- `reports/strategy_report.md` (output)
- `artifacts/recommendations.json` (output)

**What "complete" means:**
- `reports/strategy_report.md` contains at least 5 specific, evidence-backed recommendations.
- Each recommendation references the supporting finding (EDA/sentiment/topic).
- `artifacts/recommendations.json` lists recommendations in structured form.

---

## Task 9 — Writer agent

**Type:** Core
**Objective:** Draft the academic white paper using only approved artifacts and reports.

**Why needed:** The project's final deliverable is a white paper. The writer must synthesise all prior findings into a coherent academic document.

**Dependencies:** Task 8 complete and approved.

**Files involved:**
- `src/agents/writer.py` (create)
- All approved reports (inputs)
- `reports/whitepaper_draft.md` (output)

**What "complete" means:**
- `reports/whitepaper_draft.md` is a structured draft with: abstract, introduction, methodology, findings (EDA + sentiment + topics), recommendations, conclusion, limitations.
- All claims reference specific artifacts.
- No fabricated metrics or invented findings.

---

## Task 10 — Reviewer agent

**Type:** Core
**Objective:** Critically review the white paper draft for unsupported claims, weak reasoning, and academic gaps.

**Why needed:** The workflow includes a review gate before final approval. The reviewer enforces evidence alignment.

**Dependencies:** Task 9 complete (draft exists).

**Files involved:**
- `src/agents/reviewer.py` (create)
- `reports/whitepaper_draft.md` (input)
- Supporting artifacts (inputs)
- `reports/reviewer_comments.md` (output)

**What "complete" means:**
- `reports/reviewer_comments.md` contains structured feedback: specific issues found, claims needing evidence, suggested revisions.
- If no issues: reviewer states that explicitly.

---

## Task 11 — Wire real agents into orchestration graph

**Type:** Core
**Objective:** Replace all dummy stubs in `src/orchestration/graph.py` with the real agent implementations.

**Why needed:** The graph currently runs stubs. All analysis results are hardcoded. After Tasks 3–10 implement the real agents, this task wires them together.

**Dependencies:** Tasks 3–10 complete.

**Files involved:**
- `src/orchestration/graph.py`
- `main.py`

**What "complete" means:**
- Running `python main.py` executes the full real pipeline end-to-end.
- All nodes write real artifacts.
- `WorkflowState` correctly tracks step, logs, and results.

---

## Task 12 — Testing

**Type:** Optional (but expected for a university project)
**Objective:** Add a basic test suite covering scraper utilities and the cleaning pipeline.

**Why needed:** No tests exist. At minimum, scraper utility functions and cleaning transformations should be tested.

**Dependencies:** Tasks 1–4 complete.

**Files involved:**
- `tests/` (create directory)
- `tests/test_scrapper.py`
- `tests/test_cleaning.py`

**What "complete" means:**
- `pytest` runs and at least 5 tests pass.
- Tests cover: `normalize_review_text`, `review_dedup_key`, cleaning transformations, and schema validation.

---

## Task dependency summary

```
Task 1 (fix scrapper)
  └── Task 2 (obtain raw data)
        └── Task 3 (ingestion agent)
              └── Task 4 (cleaning agent)
                    ├── Task 5 (EDA)         ─┐
                    ├── Task 6 (sentiment)    ├── Task 8 (strategy)
                    └── Task 7 (topics)      ─┘
                                                  └── Task 9 (writer)
                                                        └── Task 10 (reviewer)
                                                              └── Task 11 (wire graph)

Task 12 (testing) depends on Tasks 1–4 and can be done any time after Task 4.
```
