# Hotel Review Project

End-to-end data science pipeline for hotel review analysis — from raw Booking.com scraping to strategic recommendations, a white paper draft, and a 5-minute pitch presentation.

**Dataset:** 999 verified guest reviews | Boutique hotel, Porto, Portugal | 2023-03 to 2026-04  
**Overall avg score:** 8.24/10 | **Negative share:** 6.4%

---

## Pipeline Overview

The workflow is orchestrated with LangGraph and runs 11 agents in sequence:

| Step | Agent | Output |
| --- | --- | --- |
| 1 | Ingestion | Raw schema validation report |
| 2 | Cleaning | Cleaned CSV + language detection/translation |
| 3 | EDA | 10 figures (word clouds, distributions, temporal trends) |
| 4 | Segmentation | K-Means k=4 customer personas |
| 5 | Sentiment | 4 models (VADER, LR, LinearSVC 3-class, LinearSVC binary) |
| 6 | Topic Modelling | LDA 6-topic model |
| 7 | ABSA | Rule-based aspect sentiment across 8 hotel aspects |
| 8 | Strategy | 7 evidence-backed recommendations + decision table |
| 9 | Writer | ~3,300-word white paper draft |
| 10 | Reviewer | Automated consistency and fact-check report |
| 11 | Final | Pipeline completion summary |

Entry point: `main.py`  
Graph definition: `src/orchestration/graph.py`

---

## Key Results

| Metric | Value |
| --- | --- |
| Reviews analysed | 999 |
| Overall avg score | 8.24/10 |
| Positive sentiment | 78.0% (779 reviews) |
| Negative share | 6.4% (64 reviews) |
| Best model (binary) | LinearSVC F1-macro 0.6471, CV 0.6939 |
| Top ABSA pain point | WiFi & Check-in — 39.5% negative mentions |
| Lowest-scoring topic | Room Comfort & Cleanliness — avg 7.36/10 |
| Lowest-scoring segment | Family Explorers — avg 7.98/10 |

---

## Tech Stack

- **Orchestration:** LangGraph, LangChain
- **Data:** pandas, numpy
- **ML/NLP:** scikit-learn, nltk (VADER), transformers (XLM-RoBERTa), torch
- **Visualisation:** matplotlib, wordcloud
- **Translation:** langdetect, deep-translator
- **Scraping:** selenium
- **Testing:** pytest

---

## Quick Start

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Download NLTK VADER lexicon

```powershell
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 4. Run the full pipeline

```powershell
python main.py
```

> **Note:** XLM-RoBERTa fine-tuning runs on CPU and takes ~15–30 minutes. All other agents complete in under 30 seconds combined.

---

## Data Collection

Re-scraping is optional — the repository includes the current working dataset:

- `data/raw/reviews_raw.csv` — raw Booking.com scrape output (~1,000 rows)
- `data/processed/reviews_clean.csv` — cleaned, translated, and labelled dataset

To re-scrape:

```powershell
python scrapper.py --max-pages 100 --out data/raw/reviews_raw.csv
```

Options: `--url` or `BOOKING_HOTEL_URL` env var to override the default hotel URL.

---

## Outputs

### Reports (`reports/`)

| File | Description |
| --- | --- |
| `ingestion_report.md` | Raw schema, null counts, row count |
| `data_quality_report.md` | Cleaning transformations, language distribution |
| `eda_report.md` | Score stats, sentiment distribution, top countries |
| `segmentation_report.md` | 4 K-Means customer personas with marketing actions |
| `sentiment_report.md` | Model comparison table, confusion matrices |
| `topic_report.md` | 6 LDA topics with keywords and avg scores |
| `absa_report.md` | 8-aspect sentiment breakdown + traveler×aspect heatmap |
| `strategy_report.md` | 7 recommendations with KPIs, owners, timelines |
| `whitepaper_draft.md` | ~3,300-word white paper (7 sections + appendices) |
| `reviewer_comments.md` | Automated consistency check results |
| `pitch_script.md` | 5-minute pitch script (8 slides with speaker notes) |

### Artifacts (`artifacts/`)

| File | Description |
| --- | --- |
| `schema_raw.json` | Raw data schema and null-count summary |
| `features_summary.json` | Cleaning output stats and language distribution |
| `eda_summary.json` | Score stats, sentiment counts, country breakdown |
| `segmentation.json` | K-Means cluster profiles with feature list |
| `sentiment_metrics.json` | Per-model F1, accuracy, confusion matrices |
| `sentiment_predictions.csv` | Per-review binary classifier predictions |
| `topics.json` | LDA topic labels, review counts, avg scores, top keywords |
| `topic_assignments.csv` | Per-review topic assignment |
| `absa.json` | Per-aspect mention counts and sentiment percentages |
| `recommendations.json` | Structured recommendations with actions and KPIs |

### Figures (`reports/figures/`)

27 figures total including: score distribution, word clouds (positive/negative), sentiment over time, K-Means elbow and cluster charts, model confusion matrices, ABSA negative % bar chart, ABSA traveler×aspect heatmap.

---

## Testing

```powershell
pytest tests -q
```

Tests cover scraper utility functions and cleaning agent transformations (49 tests).

---

## Project Structure

```text
hotel-review-project/
├── main.py                    # Pipeline entry point
├── scrapper.py                # Booking.com scraper
├── requirements.txt
├── src/
│   ├── agents/
│   │   ├── ingestion.py
│   │   ├── cleaning.py        # Includes language detection + translation
│   │   ├── eda.py             # Word clouds, temporal charts
│   │   ├── segmentation.py    # K-Means k=4
│   │   ├── sentiment.py       # VADER + TF-IDF models + XLM-RoBERTa
│   │   ├── topic.py           # LDA 6-topic model
│   │   ├── absa.py            # Rule-based ABSA (8 aspects, VADER scoring)
│   │   ├── strategy.py        # Recommendations + decision table
│   │   ├── writer.py          # White paper generator
│   │   └── reviewer.py        # Automated consistency checker
│   ├── orchestration/
│   │   ├── graph.py           # LangGraph StateGraph (11 nodes)
│   │   └── state.py           # WorkflowState definition
│   └── utils/
├── data/
│   ├── raw/reviews_raw.csv
│   └── processed/reviews_clean.csv
├── reports/
├── artifacts/
└── tests/
```

---

## Documentation

- `docs/agents.md` — Agent contracts and responsibilities
- `docs/workflow.md` — Workflow and approval gates
- `AI_EXECUTION_PLAYBOOK.md` — AI execution rules
