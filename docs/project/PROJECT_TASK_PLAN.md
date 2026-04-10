# PROJECT_TASK_PLAN.md

**Last updated:** 2026-04-10  
**Planning horizon:** Final submission readiness

## Purpose

This plan tracks what still needs to be completed for a strong course submission, based on the implemented codebase as of April 10, 2026.

## Completed Milestones

- [x] Scraper implemented and configured for Chrome + Selenium Manager.
- [x] Raw dataset collected (`data/raw/reviews_raw.csv`).
- [x] Agent implementations completed for ingestion, cleaning, EDA, segmentation, sentiment, topic, strategy, writer, and reviewer.
- [x] LangGraph pipeline wired end-to-end in `src/orchestration/graph.py`.
- [x] Core reports and artifacts generated.
- [x] Baseline tests added for scraper and cleaning.
- [x] README updated to reflect current project state.

## Active Workstreams

### Workstream A: Reproducibility and Stability (High Priority)

**Goal:** Ensure `python main.py` completes reliably in the local environment.

Tasks:
1. Update topic modeling to avoid environment-specific multiprocessing failures (`n_jobs=1` fallback in LDA).
2. Re-run full pipeline and regenerate all reports/artifacts in one clean pass.
3. Add a short reproducibility section to docs with known environment constraints.

Exit criteria:
- Full pipeline run completes without manual intervention.
- Generated outputs are internally consistent (metrics and dates match across reports).

### Workstream B: Documentation Consistency (High Priority)

**Goal:** Remove contradictions across documentation files.

Tasks:
1. Synchronize `PROJECT_STATUS.md`, `PROJECT_TASK_PLAN.md`, and `docs/REPO_STRUCTURE.md`.
2. Verify references to implemented agents, reports, and artifacts are accurate.
3. Confirm current blockers and next steps are aligned across all docs.

Exit criteria:
- No document claims that implemented code is still missing.
- README and status docs present a single coherent state.

### Workstream C: Assignment Coverage Expansion (Medium/High Priority)

**Goal:** Improve rubric performance on advanced modeling expectations.

Tasks:
1. Implement one advanced model from the brief (recommended first: ABSA).
2. Implement a second advanced model (recommended: fake review detection with anomaly detection).
3. Add supporting report sections and artifacts for each new model.

Exit criteria:
- At least one advanced model fully integrated with evidence and metrics.
- Preferably two advanced modules completed for stronger grading resilience.

### Workstream D: Final Deliverables (High Priority)

**Goal:** Finalize submission package.

Tasks:
1. Freeze final white paper with consistent section order and final KPI table.
2. Produce 5-minute pitch script and slide outline.
3. Validate that marketing recommendations are directly traceable to model findings.

Exit criteria:
- White paper finalized and reviewer comments addressed.
- Pitch material ready to record.

## Risks and Mitigations

1. Risk: Windows ACL issues break full runs or tests.
   - Mitigation: Disable multiprocessing where needed, set explicit local temp paths, document constraints.
2. Risk: Scope drift from adding too many advanced features late.
   - Mitigation: Prioritize ABSA + one additional model only.
3. Risk: Inconsistent numbers between reports.
   - Mitigation: Regenerate all outputs from a single clean run before final submission.

## Recommended Execution Order

1. Workstream A
2. Workstream B
3. Workstream C
4. Workstream D

## Definition of Ready for Submission

- Pipeline runs end-to-end in your target environment.
- Documentation is synchronized and accurate.
- White paper is final and traceable to artifacts.
- At least one advanced model from the brief is implemented.
- Pitch content is prepared.
