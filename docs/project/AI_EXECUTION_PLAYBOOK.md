# AI_EXECUTION_PLAYBOOK.md

This file defines the operational rules for AI agents working in this repository.

---

## Read these files first

Every agent entering this repository must read:

1. `PROJECT_STATUS.md` — current state and blockers.
2. `PROJECT_TASK_PLAN.md` — sequenced tasks with completion criteria.
3. `docs/agents.md` — agent roles, inputs, outputs, artifact paths.
4. `docs/workflow.md` — workflow stages and approval gates.
5. `UNIVERSITY_PROJECT_BRIEF.md` — data acquisition constraints.

---

## Core rules

- **Do not fabricate data.** Every metric, finding, or claim must be derived from actual data in `data/`.
- **Do not fabricate metrics.** Do not write numbers into reports unless they come from actual computation.
- **Do not alter project scope.** The project is hotel review analysis → NLP → strategy → white paper.
- **Do not skip approval gates.** The workflow in `docs/workflow.md` defines gates. Do not advance without them.
- **Use canonical paths.** All file I/O must use path constants from `src/utils/paths.py`.
- **Write artifacts.** Every agent must produce its defined artifact(s) and report(s) before marking itself complete.
- **Do not hardcode paths.** Use `RAW_DIR`, `PROCESSED_DIR`, `ARTIFACTS_DIR`, `REPORTS_DIR` from `src/utils/paths.py`.
- **One task at a time.** Complete a task fully before starting the next. Partial work must be explicitly flagged.

---

## Scraper rules

- **Browser:** Chrome only. Set `SCRAPER_BROWSER=chrome` or update `build_driver()` default.
- **Driver management:** Use Selenium Manager (built into Selenium 4.6+). Do NOT install `webdriver-manager`.
- **Output path:** Always write to `data/raw/reviews_raw.csv`. Do not write to the project root.
- **Verify output:** After scraping, confirm row count > 0 and all expected columns are present.
- **Document blockers:** If scraping fails (CAPTCHA, IP block), document in `PROJECT_STATUS.md` immediately.

---

## Data rules

- Raw data lives in `data/raw/`. Never modify raw data.
- Cleaned data lives in `data/processed/`. Document every transformation in `reports/data_quality_report.md`.
- Interim/staging files go to `data/interim/` if needed.
- Do not commit raw data to git (it is gitignored). Commit only the report and schema artifacts.

---

## Artifact rules

- Every agent produces at minimum: one report (`.md` in `reports/`) and one structured artifact (`.json` or `.csv` in `artifacts/`).
- Artifact paths are defined in `docs/agents.md`. Follow them exactly.
- Artifacts must be machine-readable (valid JSON or well-formed CSV).

---

## Orchestration rules

- All node implementations live in `src/agents/`.
- The LangGraph graph is in `src/orchestration/graph.py`. Each node must call the corresponding agent function.
- `WorkflowState` (defined in `src/orchestration/state.py`) must be updated after each node: set `current_step`, append to `logs`, write result summary to `results[node_name]`.
- Do not hardcode results. Dummy stubs must be replaced before marking any node "complete".

---

## Writing and review rules

- The writer agent (`src/agents/writer.py`) must only use content from approved reports and artifacts.
- The reviewer agent (`src/agents/reviewer.py`) must flag any claim not traceable to an artifact.
- If the reviewer returns issues, the writer revises before final approval.

---

## What agents must not do

- Must not invent review data, scores, or findings.
- Must not skip a stage and write a later stage's report without prior stage artifacts existing.
- Must not push to `origin` without human confirmation.
- Must not modify `.gitignore`, `requirements.txt` package list, or `src/orchestration/state.py` schema without explicit instruction.
- Must not recursively inspect `venv/`, `.venv/`, `.git/`, `__pycache__/`.

---

## Definition of "task complete"

A task is complete when:

1. All defined artifacts exist and are non-empty.
2. All defined reports exist and are non-empty.
3. The corresponding test (if any) passes.
4. `WorkflowState.results[task_name]["status"]` is `"success"` after a real run.
5. `PROJECT_STATUS.md` is updated to reflect the new state.
