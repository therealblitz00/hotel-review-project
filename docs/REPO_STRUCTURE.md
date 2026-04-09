# REPO_STRUCTURE.md

**Last updated:** 2026-04-10

This document describes the current repository layout and the main files used by the pipeline.

## Read These Files First

1. `README.md`
2. `PROJECT_STATUS.md`
3. `PROJECT_TASK_PLAN.md`
4. `docs/agents.md`
5. `docs/workflow.md`
6. `AI_EXECUTION_PLAYBOOK.md`
7. `UNIVERSITY_PROJECT_BRIEF.md`

## Current Repository Structure

```text
hotel-review-project/
|- README.md
|- PROJECT_STATUS.md
|- PROJECT_TASK_PLAN.md
|- AI_EXECUTION_PLAYBOOK.md
|- UNIVERSITY_PROJECT_BRIEF.md
|- requirements.txt
|- main.py
|- scrapper.py
|- .env.example
|- .gitignore
|
|- src/
|  |- agents/
|  |  |- __init__.py
|  |  |- ingestion.py
|  |  |- cleaning.py
|  |  |- eda.py
|  |  |- segmentation.py
|  |  |- sentiment.py
|  |  |- topic.py
|  |  |- strategy.py
|  |  |- writer.py
|  |  |- reviewer.py
|  |
|  |- orchestration/
|  |  |- __init__.py
|  |  |- state.py
|  |  |- graph.py
|  |
|  |- utils/
|     |- __init__.py
|     |- config.py
|     |- logging_utils.py
|     |- paths.py
|
|- data/
|  |- raw/
|  |  |- reviews_raw.csv
|  |- interim/
|  |- processed/
|     |- reviews_clean.csv
|
|- reports/
|  |- ingestion_report.md
|  |- data_quality_report.md
|  |- eda_report.md
|  |- segmentation_report.md
|  |- sentiment_report.md
|  |- topic_report.md
|  |- strategy_report.md
|  |- whitepaper_draft.md
|  |- reviewer_comments.md
|  |- figures/
|
|- artifacts/
|  |- schema_raw.json
|  |- features_summary.json
|  |- eda_summary.json
|  |- segmentation.json
|  |- sentiment_metrics.json
|  |- sentiment_predictions.csv
|  |- topics.json
|  |- topic_assignments.csv
|  |- recommendations.json
|
|- tests/
|  |- __init__.py
|  |- test_scrapper.py
|  |- test_cleaning.py
|
|- docs/
|  |- agents.md
|  |- workflow.md
|  |- REPO_STRUCTURE.md
|
|- handoff/
|  |- current_task.md
```

## Path Constants

Use `src/utils/paths.py` for canonical paths instead of hardcoded strings.

| Constant | Meaning |
| --- | --- |
| `ROOT_DIR` | Repository root |
| `RAW_DIR` | `data/raw` |
| `INTERIM_DIR` | `data/interim` |
| `PROCESSED_DIR` | `data/processed` |
| `ARTIFACTS_DIR` | `artifacts` |
| `REPORTS_DIR` | `reports` |
| `FIGURES_DIR` | `reports/figures` |

## Notes

- Data, artifacts, and figure outputs are gitignored by default.
- The main workflow entry point is `main.py`.
- The orchestrated node sequence is defined in `src/orchestration/graph.py`.
- Scraping entry point is `scrapper.py`.
