# PROJECT_STATUS.md

**Last updated:** 2026-04-10  
**Current phase:** Submission hardening and assignment coverage expansion

## Project Overview

This repository implements a multi-stage hotel review analytics pipeline:

1. Scrape reviews from Booking.com.
2. Clean and standardize text and metadata.
3. Run EDA, customer segmentation, sentiment modeling, and topic modeling.
4. Generate strategy recommendations.
5. Draft and review a white paper.

The workflow is orchestrated via LangGraph (`src/orchestration/graph.py`).

## Overall Progress

| Layer | Status |
| --- | --- |
| Scraper (`scrapper.py`) | Complete |
| Raw data collection (`data/raw/reviews_raw.csv`) | Complete (~1000 rows) |
| Orchestration graph | Implemented (all nodes wired) |
| Ingestion agent | Complete |
| Cleaning agent | Complete |
| EDA agent | Complete |
| Segmentation agent | Complete |
| Sentiment agent | Complete |
| Topic agent | Complete |
| Strategy agent | Complete |
| Writer agent | Complete |
| Reviewer agent | Complete |
| Tests (`tests/`) | Implemented (scraper + cleaning coverage) |
| Documentation sync | In progress |
| Advanced models (ABSA/fraud/helpfulness) | Not started |
| Pitch video deliverable | Not started |

## What Is Done

- End-to-end code exists for all planned pipeline agents under `src/agents/`.
- Reports are generated under `reports/`.
- Structured artifacts are generated under `artifacts/`.
- A white paper draft and reviewer comments are already produced.
- Tests are available in `tests/test_scrapper.py` and `tests/test_cleaning.py`.
- Repository has been pushed to GitHub (`main` branch).

## What Is Partially Done

- Full one-command reproducibility (`python main.py`) can fail in restricted Windows environments due to multiprocessing in LDA (`n_jobs=-1`) and local ACL limits.
- Documentation was inconsistent with current code and is being synchronized.
- White paper quality is strong, but final submission freeze and narrative alignment still need one final pass after stability fixes.

## What Is Not Done Yet

- Assignment extensions not yet implemented:
  - Aspect-based sentiment analysis (ABSA)
  - Fake review detection
  - Review helpfulness prediction
  - Stronger temporal/geospatial analysis
- Multi-source ingestion (TripAdvisor/social channels) is not implemented.
- Final 5-minute pitch video and slide package are not prepared.

## Current Risks / Blockers

1. Environment-specific permission errors:
   - `topic.py` can fail with `PermissionError` when LDA uses multiprocessing (`n_jobs=-1`).
   - `pytest` temp/cache directories can fail under strict ACLs.
2. Single-source data (Booking.com only) limits assignment scope depth.

## Immediate Next Steps

1. Stabilize topic modeling for restricted environments (`n_jobs=1` fallback).
2. Regenerate reports/artifacts in one clean run and confirm reproducibility.
3. Add at least one advanced model (ABSA or fake review detection).
4. Finalize white paper and prepare the pitch video package.
