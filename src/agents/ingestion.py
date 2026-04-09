from __future__ import annotations

import json

import pandas as pd

from src.orchestration.state import WorkflowState
from src.utils.logging_utils import get_logger
from src.utils.paths import ARTIFACTS_DIR, RAW_DIR, REPORTS_DIR

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "name",
    "country",
    "room_type",
    "nr_nights",
    "date",
    "traveler_type",
    "title_review",
    "pos_review",
    "neg_review",
    "hotel_response",
    "score",
]

RAW_CSV = RAW_DIR / "reviews_raw.csv"
INGESTION_REPORT = REPORTS_DIR / "ingestion_report.md"
SCHEMA_ARTIFACT = ARTIFACTS_DIR / "schema_raw.json"


def run_ingestion(state: WorkflowState) -> WorkflowState:
    logger.info("Running ingestion agent")
    state.current_step = "ingestion"

    # --- Load ---
    if not RAW_CSV.exists():
        msg = f"Raw data file not found: {RAW_CSV}"
        logger.error(msg)
        state.warnings.append(msg)
        state.results["ingestion"] = {"status": "failed", "reason": msg}
        return state

    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    logger.info("Loaded %d rows from %s", len(df), RAW_CSV)

    # --- Validate schema ---
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        msg = f"Missing expected columns: {missing_cols}"
        logger.warning(msg)
        state.warnings.append(msg)

    # --- Compute statistics ---
    null_counts = df.isnull().sum().to_dict()
    null_rates = {col: round(n / len(df) * 100, 1) for col, n in null_counts.items()}

    score_stats = {}
    if "score" in df.columns:
        s = df["score"].dropna()
        score_stats = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": round(float(s.mean()), 2),
            "median": float(s.median()),
        }

    date_range = {}
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"].dropna(), format="%B %Y", errors="coerce")
        valid = parsed.dropna()
        if not valid.empty:
            date_range = {
                "earliest": valid.min().strftime("%B %Y"),
                "latest": valid.max().strftime("%B %Y"),
                "unique_values": int(parsed.nunique()),
            }
        else:
            raw_dates = df["date"].dropna().tolist()
            date_range = {"earliest": raw_dates[-1], "latest": raw_dates[0], "unique_values": len(set(raw_dates))}

    country_top5 = {}
    if "country" in df.columns:
        country_top5 = df["country"].value_counts().head(5).to_dict()

    traveler_dist = {}
    if "traveler_type" in df.columns:
        traveler_dist = df["traveler_type"].value_counts().to_dict()

    # --- Write schema artifact ---
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    schema = {
        "source_file": str(RAW_CSV),
        "row_count": len(df),
        "columns": {col: str(df[col].dtype) for col in df.columns},
        "missing_columns": missing_cols,
        "null_counts": null_counts,
    }
    SCHEMA_ARTIFACT.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    logger.info("Schema artifact written to %s", SCHEMA_ARTIFACT)

    # --- Write ingestion report ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Ingestion Report\n",
        f"**Source:** `{RAW_CSV}`  ",
        f"**Rows:** {len(df)}  ",
        f"**Columns:** {len(df.columns)}  ",
        "",
        "## Schema",
        "",
        "| Column | Dtype | Null count | Null % |",
        "| --- | --- | --- | --- |",
    ]
    for col in df.columns:
        lines.append(
            f"| {col} | {df[col].dtype} | {null_counts[col]} | {null_rates[col]}% |"
        )

    lines += [
        "",
        "## Score distribution",
        "",
        f"- Min: {score_stats.get('min', 'n/a')}",
        f"- Max: {score_stats.get('max', 'n/a')}",
        f"- Mean: {score_stats.get('mean', 'n/a')}",
        f"- Median: {score_stats.get('median', 'n/a')}",
        "",
        "## Date range",
        "",
        f"- Earliest review date: {date_range.get('earliest', 'n/a')}",
        f"- Latest review date: {date_range.get('latest', 'n/a')}",
        f"- Unique date values: {date_range.get('unique_values', 'n/a')}",
        "",
        "## Top 5 reviewer countries",
        "",
        "| Country | Count |",
        "| --- | --- |",
    ]
    for country, count in country_top5.items():
        lines.append(f"| {country} | {count} |")

    lines += [
        "",
        "## Traveler type distribution",
        "",
        "| Traveler type | Count |",
        "| --- | --- |",
    ]
    for ttype, count in traveler_dist.items():
        lines.append(f"| {ttype} | {count} |")

    if missing_cols:
        lines += ["", "## Warnings", "", f"- Missing expected columns: {missing_cols}"]

    lines += [
        "",
        "## Schema validation",
        "",
        "Pass" if not missing_cols else "Fail — see warnings above",
    ]

    INGESTION_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Ingestion report written to %s", INGESTION_REPORT)

    # --- Update state ---
    state.logs.append(f"Ingestion complete: {len(df)} rows loaded from {RAW_CSV}")
    state.results["ingestion"] = {
        "status": "success",
        "row_count": len(df),
        "missing_columns": missing_cols,
        "report": str(INGESTION_REPORT),
        "schema_artifact": str(SCHEMA_ARTIFACT),
    }
    return state
