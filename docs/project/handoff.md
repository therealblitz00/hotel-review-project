# Current Task

**Task:** Task 3 — Data ingestion agent
**Status:** Not started
**Previous tasks complete:** Task 1 (scrapper.py fixed), Task 2 (raw data collected)

## Context

`data/raw/reviews_raw.csv` — 1000 rows, 11 columns, scraped from Booking.com (Paço de Açúcar Hotel Porto).

Column null summary:

- `pos_review`: 24 nulls (2.4%)
- `neg_review`: 289 nulls (28.9%) — expected, guests often skip negative text
- `hotel_response`: 1000 nulls — hotel does not reply on Booking.com
- All other columns: 0 nulls

Score is already numeric (float, 1–10). Date is a string like "April 2026".

## What needs to be done

Create `src/agents/ingestion.py` with a function `run_ingestion(state: WorkflowState) -> WorkflowState` that:

1. Loads `data/raw/reviews_raw.csv` using `RAW_DIR / "reviews_raw.csv"` from `src/utils/paths.py`.
2. Validates that all 11 expected columns are present.
3. Writes `reports/ingestion_report.md` with: row count, column list, null counts per column, score distribution summary, date range.
4. Writes `artifacts/schema_raw.json` with column names and dtypes.
5. Updates `state.results["ingestion"]` with status and artifact paths.
6. Updates `state.current_step = "ingestion"` and appends to `state.logs`.

Then replace the dummy `ingestion_node` in `src/orchestration/graph.py` with a call to `run_ingestion`.

## Completion check

- `reports/ingestion_report.md` exists and is non-empty.
- `artifacts/schema_raw.json` exists and is valid JSON.
- `python main.py` runs the ingestion node and produces both files.

## Next task after this

Task 4 — Data cleaning agent.
