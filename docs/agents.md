# Agent Architecture

## Orchestration Model
The system uses a supervisor-controlled workflow with specialist agents. Agents do not collaborate freely. Each agent executes a bounded task and produces explicit artifacts for downstream steps.

## Agents

### 1. Data Ingestion Agent
**Purpose:** Load and consolidate raw review data into a unified schema.

**Inputs:**
- Raw data files
- Source configuration
- Expected schema definition

**Outputs:**
- Unified raw dataset
- Ingestion report
- Raw schema artifact

**Artifacts:**
- `data/raw/reviews_raw.csv`
- `reports/ingestion_report.md`
- `artifacts/schema_raw.json`

---

### 2. Data Cleaning Agent
**Purpose:** Standardize, clean, and enrich the raw dataset.

**Inputs:**
- Raw dataset
- Approved schema
- Cleaning configuration

**Outputs:**
- Clean dataset
- Data quality report
- Feature summary

**Artifacts:**
- `data/processed/reviews_clean.csv`
- `reports/data_quality_report.md`
- `artifacts/features_summary.json`

---

### 3. EDA Agent
**Purpose:** Produce descriptive statistics, visualizations, and exploratory findings.

**Inputs:**
- Clean dataset
- EDA configuration

**Outputs:**
- EDA report
- Summary statistics artifact
- Figures

**Artifacts:**
- `reports/eda_report.md`
- `artifacts/eda_summary.json`
- `reports/figures/`

---

### 4. Sentiment Analysis Agent
**Purpose:** Train and evaluate sentiment classification models.

**Inputs:**
- Clean dataset
- Text column
- Target definition
- Modelling configuration

**Outputs:**
- Sentiment report
- Metrics artifact
- Predictions artifact

**Artifacts:**
- `reports/sentiment_report.md`
- `artifacts/sentiment_metrics.json`
- `artifacts/sentiment_predictions.csv`

---

### 5. Topic Modelling Agent
**Purpose:** Discover and summarize major review themes.

**Inputs:**
- Clean dataset
- Topic modelling configuration

**Outputs:**
- Topic report
- Topic artifact
- Topic assignments

**Artifacts:**
- `reports/topic_report.md`
- `artifacts/topics.json`
- `artifacts/topic_assignments.csv`

---

### 6. Strategy Agent
**Purpose:** Translate approved analytical results into customer experience and digital marketing recommendations.

**Inputs:**
- Approved EDA findings
- Approved sentiment results
- Approved topic results

**Outputs:**
- Strategy report
- Recommendations artifact

**Artifacts:**
- `reports/strategy_report.md`
- `artifacts/recommendations.json`

---

### 7. Writer Agent
**Purpose:** Draft the white paper using only approved outputs.

**Inputs:**
- Approved reports
- White paper structure

**Outputs:**
- Draft paper

**Artifacts:**
- `reports/whitepaper_draft.md`

---

### 8. Reviewer Agent
**Purpose:** Critically assess the white paper draft and detect unsupported claims or weak reasoning.

**Inputs:**
- White paper draft
- Supporting artifacts

**Outputs:**
- Reviewer comments

**Artifacts:**
- `reports/reviewer_comments.md`

## Governance Rules
- Agents must not invent data.
- Agents must not fabricate metrics.
- Agents must not alter project scope.
- Agents must save outputs as artifacts.
- Claims must be traceable to artifacts.