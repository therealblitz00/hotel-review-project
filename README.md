# Hotel Review Project

End-to-end data science pipeline for hotel review analysis, from raw scraping to strategic recommendations and a white paper draft.

The project combines:
- Data engineering (ingestion + cleaning)
- NLP/ML analysis (sentiment, topic modeling, customer segmentation)
- Business translation (marketing and customer-experience recommendations)
- Automated reporting artifacts for academic deliverables

## Implemented Pipeline

The workflow is orchestrated with LangGraph and currently runs this sequence:

1. Ingestion
2. Cleaning
3. EDA
4. Segmentation (K-Means)
5. Sentiment analysis (TF-IDF + LR/SVC + VADER baseline)
6. Topic modeling (LDA)
7. Strategy recommendations
8. White paper draft generation
9. White paper consistency review

Entry point: `main.py`  
Graph definition: `src/orchestration/graph.py`

## Current Project Status (2026-04-10)

Done:
- Scraper implemented and used with Booking.com (`data/raw/reviews_raw.csv` with ~1000 rows).
- Full agent code implemented for ingestion, cleaning, EDA, segmentation, sentiment, topic, strategy, writer, and reviewer.
- Reports and artifacts are being generated for all major stages.
- Unit tests exist for scraper utilities and cleaning logic.

Partially done:
- End-to-end reproducibility can fail in restricted Windows environments because topic modeling uses multi-process LDA (`n_jobs=-1`).
- White paper draft and reviewer outputs exist, but final narrative alignment still needs one clean final regeneration pass.
- Some project documentation files are outdated relative to the implemented code.

Not done yet (assignment scope):
- Multi-source ingestion beyond Booking.com (for example TripAdvisor/social media).
- Advanced analytics modules requested in the brief: ABSA, fake-review detection, helpfulness prediction, stronger temporal/geospatial analysis.
- Final 5-minute pitch video and presentation package.

## Remaining Stages

1. Stabilize the full pipeline run in your environment and regenerate all outputs in one pass.
2. Add at least 1-2 advanced models (recommended: ABSA and fake-review detection) to strengthen the modeling grade.
3. Update and freeze the white paper with final evidence tables, limitations, and marketing actions.
4. Prepare the pitch video script/slides from the final artifacts.
5. Final QA pass: metrics consistency, figure references, and reproducibility checklist.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- nltk
- matplotlib, wordcloud
- selenium (for Booking.com scraping)
- langgraph / langchain-core

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

### 3. Download NLTK VADER lexicon (required for sentiment baseline)

```powershell
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Data Collection

Scraper script: `scrapper.py`  
Default output: `data/raw/reviews_raw.csv`

Smoke test:

```powershell
python scrapper.py --max-pages 1
```

Larger collection run:

```powershell
python scrapper.py --max-pages 100 --out data/raw/reviews_raw.csv
```

Notes:
- Default browser is Chrome via Selenium Manager.
- You can override the hotel URL with `--url` or `BOOKING_HOTEL_URL`.

## Run the Full Pipeline

```powershell
python main.py
```

The pipeline writes structured outputs to:
- `reports/`
- `artifacts/`
- `data/processed/`

## Main Outputs

Reports:
- `reports/ingestion_report.md`
- `reports/data_quality_report.md`
- `reports/eda_report.md`
- `reports/segmentation_report.md`
- `reports/sentiment_report.md`
- `reports/topic_report.md`
- `reports/strategy_report.md`
- `reports/whitepaper_draft.md`
- `reports/reviewer_comments.md`

Artifacts:
- `artifacts/schema_raw.json`
- `artifacts/features_summary.json`
- `artifacts/eda_summary.json`
- `artifacts/segmentation.json`
- `artifacts/sentiment_metrics.json`
- `artifacts/sentiment_predictions.csv`
- `artifacts/topics.json`
- `artifacts/topic_assignments.csv`
- `artifacts/recommendations.json`

## Testing

```powershell
pytest tests -q
```

Current tests cover scraper utility behavior and cleaning transformations.

## Repository Notes

- Raw/processed data, generated artifacts, and figures are ignored in git by default (`.gitignore`).
- Architecture and agent contracts: `docs/agents.md`
- Workflow and approval gates: `docs/workflow.md`
- Operational rules for AI execution: `AI_EXECUTION_PLAYBOOK.md`
