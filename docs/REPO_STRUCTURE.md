# REPO_STRUCTURE.md

**Created:** 2026-04-09

This file describes the intended repository structure and which files future agents should read first.

---

## Files agents must read first

When entering this repository, read these files in order:

1. `PROJECT_STATUS.md` — current phase, what is done, what is missing, current blockers.
2. `PROJECT_TASK_PLAN.md` — sequenced list of remaining tasks with completion criteria.
3. `docs/agents.md` — agent architecture: each agent's inputs, outputs, and artifact paths.
4. `docs/workflow.md` — workflow stages and approval gates.
5. `UNIVERSITY_PROJECT_BRIEF.md` — data acquisition requirements and scraper review mandate.

---

## Intended repository structure

```
hotel-review-project/
│
├── README.md                      # Project overview; points to docs/
├── UNIVERSITY_PROJECT_BRIEF.md    # Data acquisition + scraper requirements (canonical)
├── PROJECT_STATUS.md              # Current implementation state (keep updated)
├── PROJECT_TASK_PLAN.md           # Sequenced remaining tasks (keep updated)
├── AI_EXECUTION_PLAYBOOK.md       # Operational rules for AI agents
├── requirements.txt               # Python dependencies (must include selenium)
├── main.py                        # Orchestration entry point
├── scrapper.py                    # Booking.com review scraper
├── .env.example                   # Environment variable template
├── .gitignore                     # Standard Python gitignore
│
├── src/
│   ├── agents/                    # One module per agent (ingestion, cleaning, eda, ...)
│   │   ├── __init__.py
│   │   ├── ingestion.py           # [to be created]
│   │   ├── cleaning.py            # [to be created]
│   │   ├── eda.py                 # [to be created]
│   │   ├── sentiment.py           # [to be created]
│   │   ├── topic.py               # [to be created]
│   │   ├── strategy.py            # [to be created]
│   │   ├── writer.py              # [to be created]
│   │   └── reviewer.py            # [to be created]
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── graph.py               # LangGraph workflow graph
│   │   └── state.py               # WorkflowState Pydantic model
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # Settings from .env
│       ├── logging_utils.py       # Logger factory
│       └── paths.py               # Canonical path constants (RAW_DIR, PROCESSED_DIR, etc.)
│
├── data/
│   ├── raw/
│   │   └── reviews_raw.csv        # Output of scrapper.py [to be created by Task 2]
│   ├── interim/                   # Intermediate processing steps (optional)
│   └── processed/
│       └── reviews_clean.csv      # Output of cleaning agent [to be created by Task 4]
│
├── artifacts/                     # Structured JSON artifacts from each agent
│   ├── schema_raw.json            # [Task 3]
│   ├── features_summary.json      # [Task 4]
│   ├── eda_summary.json           # [Task 5]
│   ├── sentiment_metrics.json     # [Task 6]
│   ├── sentiment_predictions.csv  # [Task 6]
│   ├── topics.json                # [Task 7]
│   ├── topic_assignments.csv      # [Task 7]
│   └── recommendations.json       # [Task 8]
│
├── reports/
│   ├── ingestion_report.md        # [Task 3] — currently empty placeholder
│   ├── data_quality_report.md     # [Task 4] — currently empty placeholder
│   ├── eda_report.md              # [Task 5] — currently empty placeholder
│   ├── sentiment_report.md        # [Task 6] — currently empty placeholder
│   ├── topic_report.md            # [Task 7] — currently empty placeholder
│   ├── strategy_report.md         # [Task 8] — currently empty placeholder
│   ├── whitepaper_draft.md        # [Task 9] — currently empty placeholder
│   ├── reviewer_comments.md       # [Task 10] — currently empty placeholder
│   └── figures/                   # Saved plots from EDA agent
│
├── tests/
│   ├── test_scrapper.py           # [Task 12 — to be created]
│   └── test_cleaning.py           # [Task 12 — to be created]
│
├── docs/
│   ├── agents.md                  # Agent architecture (complete)
│   ├── workflow.md                # Workflow stages + approval gates (complete)
│   ├── REPO_STRUCTURE.md          # This file
│   └── archive/                   # Deprecated docs (currently empty)
│
└── handoff/                       # Agent handoff state (to be created when needed)
    ├── current_task.md
    ├── current_implementation_log.md
    └── current_review_report.md
```

---

## Path constants

All file paths inside agents should be derived from `src/utils/paths.py`, not hardcoded:

| Constant | Path |
|---|---|
| `ROOT_DIR` | project root |
| `RAW_DIR` | `data/raw/` |
| `INTERIM_DIR` | `data/interim/` |
| `PROCESSED_DIR` | `data/processed/` |
| `ARTIFACTS_DIR` | `artifacts/` |
| `REPORTS_DIR` | `reports/` |
| `FIGURES_DIR` | `reports/figures/` |

---

## Key conventions

- **Scraper output path:** `data/raw/reviews_raw.csv` (run `scrapper.py --out data/raw/reviews_raw.csv`)
- **Clean data path:** `data/processed/reviews_clean.csv`
- **Reports:** Markdown files in `reports/` — written by agents, not manually.
- **Artifacts:** JSON/CSV files in `artifacts/` — machine-readable outputs from agents.
- **No fabricated data:** Agents must not produce metrics or findings not derived from actual data.
- **Browser:** Chrome (via Selenium Manager — no `webdriver-manager` package needed).
