# Workflow Design

## Workflow Style
The project uses a deterministic supervisor-controlled workflow. Each stage produces explicit artifacts and may require human approval before the next stage begins.

## Workflow Stages

1. Data ingestion
2. Raw schema approval
3. Data cleaning
4. Cleaned dataset approval
5. Exploratory Data Analysis
6. EDA approval
7. Sentiment analysis
8. Topic modelling
9. Modelling approval
10. Strategy generation
11. White paper drafting
12. Review and critique
13. Final approval

## Approval Gates

### Gate 1 — Raw Schema Approval
Validate:
- column structure
- source consistency
- key identifier fields
- date and rating availability

### Gate 2 — Cleaned Dataset Approval
Validate:
- cleaning transformations
- dropped rows/columns
- created features
- missing-value treatment

### Gate 3 — EDA Approval
Validate:
- quality of descriptive findings
- identification of anomalies
- readiness for modelling

### Gate 4 — Modelling Approval
Validate:
- sentiment target design
- selected models
- topic modelling approach
- evaluation logic

### Gate 5 — Final Interpretation Approval
Validate:
- strength of findings
- limitations
- relevance of recommendations

### Gate 6 — Final Draft Approval
Validate:
- academic consistency
- evidence alignment
- clarity and defensibility

## Failure / Loop Rules
- If ingestion fails schema checks, return to ingestion.
- If EDA reveals major data quality issues, return to cleaning.
- If reviewer flags unsupported claims, return to writing.