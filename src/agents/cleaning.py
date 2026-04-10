from __future__ import annotations

import json
import re

import pandas as pd

from src.orchestration.state import WorkflowState
from src.utils.logging_utils import get_logger
from src.utils.paths import ARTIFACTS_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR

# Optional multilingual support — graceful fallback if libraries not installed
try:
    from langdetect import detect, LangDetectException
    from deep_translator import GoogleTranslator
    _TRANSLATION_AVAILABLE = True
except ImportError:
    _TRANSLATION_AVAILABLE = False

logger = get_logger(__name__)

RAW_CSV = RAW_DIR / "reviews_raw.csv"
CLEAN_CSV = PROCESSED_DIR / "reviews_clean.csv"
QUALITY_REPORT = REPORTS_DIR / "data_quality_report.md"
FEATURES_ARTIFACT = ARTIFACTS_DIR / "features_summary.json"


def _extract_nr_nights(s: str) -> int | None:
    """'3 nights ✈' → 3"""
    if pd.isna(s):
        return None
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None


def _score_to_sentiment(score: float) -> str:
    if score >= 8.0:
        return "positive"
    if score >= 6.0:
        return "neutral"
    return "negative"


def _detect_and_translate(text: str) -> tuple[str, str]:
    """Detect language and translate non-English text to English.

    Returns (translated_text, detected_lang_code).
    Falls back to original text on any error.
    """
    if not _TRANSLATION_AVAILABLE or not text.strip():
        return text, "en"
    try:
        lang = detect(text)
    except LangDetectException:
        return text, "unknown"
    if lang == "en":
        return text, "en"
    try:
        translated = GoogleTranslator(source=lang, target="en").translate(text)
        return translated or text, lang
    except Exception:  # network error, unsupported language, etc.
        return text, lang


def run_cleaning(state: WorkflowState) -> WorkflowState:
    logger.info("Running cleaning agent")
    state.current_step = "cleaning"

    if not RAW_CSV.exists():
        msg = f"Raw data file not found: {RAW_CSV}"
        logger.error(msg)
        state.warnings.append(msg)
        state.results["cleaning"] = {"status": "failed", "reason": msg}
        return state

    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    rows_before = len(df)
    transformations: list[str] = []

    # 1. Drop hotel_response — 100% null, carries no information
    if "hotel_response" in df.columns and df["hotel_response"].isnull().all():
        df = df.drop(columns=["hotel_response"])
        transformations.append("Dropped `hotel_response` (100% null).")

    # 2. Parse nr_nights → integer
    df["nr_nights"] = df["nr_nights"].apply(_extract_nr_nights)
    transformations.append("Parsed `nr_nights` to integer (extracted leading digit).")

    # 3. Parse date → period string "YYYY-MM" for easy sorting/grouping
    parsed_dates = pd.to_datetime(df["date"], format="%B %Y", errors="coerce")
    df["date_parsed"] = parsed_dates.dt.to_period("M").astype(str)
    df["year"] = parsed_dates.dt.year.astype("Int64")
    df["month"] = parsed_dates.dt.month.astype("Int64")
    transformations.append(
        "Parsed `date` (Month YYYY) → `date_parsed` (YYYY-MM), plus `year` and `month` columns."
    )

    # 4. Fill text nulls with empty string before concat
    df["pos_review"] = df["pos_review"].fillna("")
    df["neg_review"] = df["neg_review"].fillna("")
    transformations.append("Filled `pos_review` and `neg_review` nulls with empty string.")

    # 5. Build full_review_text
    df["full_review_text"] = (
        df["pos_review"].str.strip() + " " + df["neg_review"].str.strip()
    ).str.strip()
    transformations.append(
        "Created `full_review_text` = pos_review + ' ' + neg_review (stripped)."
    )

    # 6. Detect language and translate non-English reviews to English
    if _TRANSLATION_AVAILABLE:
        logger.info("Detecting languages and translating non-English reviews…")
        results = df["full_review_text"].apply(_detect_and_translate)
        df["full_review_text"] = results.apply(lambda x: x[0])
        df["review_language"] = results.apply(lambda x: x[1])
        lang_counts = df["review_language"].value_counts().to_dict()
        n_translated = int((df["review_language"] != "en").sum())
        transformations.append(
            f"Detected review language (`review_language` column); translated "
            f"{n_translated} non-English reviews to English via Google Translate. "
            f"Languages found: {lang_counts}."
        )
        logger.info("Translation complete: %d non-English reviews translated.", n_translated)
    else:
        df["review_language"] = "en"
        transformations.append(
            "Language detection skipped (langdetect / deep-translator not installed). "
            "Install them with: pip install langdetect deep-translator"
        )
        logger.warning("langdetect/deep-translator not available — skipping translation.")

    # 7. Drop rows where full_review_text is empty (nothing to analyse)
    empty_text_mask = df["full_review_text"] == ""
    n_empty = int(empty_text_mask.sum())
    if n_empty:
        df = df[~empty_text_mask].reset_index(drop=True)
        transformations.append(f"Dropped {n_empty} row(s) with empty full_review_text.")

    # 8. Derive sentiment label from score
    df["sentiment"] = df["score"].apply(_score_to_sentiment)
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    transformations.append(
        "Created `sentiment` label from score: ≥8 → positive, 6–7 → neutral, <6 → negative."
    )

    # 9. Standardise country casing (title case)
    df["country"] = df["country"].str.strip().str.title()
    transformations.append("Standardised `country` to title case.")

    rows_after = len(df)

    # --- Save clean CSV ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")
    logger.info("Clean dataset written to %s (%d rows)", CLEAN_CSV, rows_after)

    # --- Features summary artifact ---
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    features_summary = {
        "source_file": str(RAW_CSV),
        "output_file": str(CLEAN_CSV),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_dropped": rows_before - rows_after,
        "columns": {col: str(df[col].dtype) for col in df.columns},
        "null_counts": df.isnull().sum().to_dict(),
        "sentiment_distribution": sentiment_counts,
        "language_distribution": df["review_language"].value_counts().to_dict(),
        "translation_available": _TRANSLATION_AVAILABLE,
        "score_stats": {
            "min": float(df["score"].min()),
            "max": float(df["score"].max()),
            "mean": round(float(df["score"].mean()), 2),
            "median": float(df["score"].median()),
        },
        "transformations": transformations,
    }
    FEATURES_ARTIFACT.write_text(json.dumps(features_summary, indent=2), encoding="utf-8")
    logger.info("Features summary written to %s", FEATURES_ARTIFACT)

    # --- Data quality report ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    null_counts = df.isnull().sum()
    null_rates = (null_counts / rows_after * 100).round(1)

    lines: list[str] = [
        "# Data Quality Report",
        "",
        f"**Source:** `{RAW_CSV}`  ",
        f"**Output:** `{CLEAN_CSV}`  ",
        f"**Rows before cleaning:** {rows_before}  ",
        f"**Rows after cleaning:** {rows_after}  ",
        f"**Rows dropped:** {rows_before - rows_after}  ",
        "",
        "## Transformations applied",
        "",
    ]
    for i, t in enumerate(transformations, 1):
        lines.append(f"{i}. {t}")

    lines += [
        "",
        "## Final schema",
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
        f"- Min: {features_summary['score_stats']['min']}",
        f"- Max: {features_summary['score_stats']['max']}",
        f"- Mean: {features_summary['score_stats']['mean']}",
        f"- Median: {features_summary['score_stats']['median']}",
        "",
        "## Sentiment label distribution",
        "",
        "| Sentiment | Count |",
        "| --- | --- |",
    ]
    for label, count in sentiment_counts.items():
        pct = round(count / rows_after * 100, 1)
        lines.append(f"| {label} | {count} ({pct}%) |")

    lang_dist = df["review_language"].value_counts().to_dict()
    lines += [
        "",
        "## Review language distribution",
        "",
        "| Language code | Count | Share |",
        "| --- | --- | --- |",
    ]
    for lang_code, cnt in lang_dist.items():
        pct = round(cnt / rows_after * 100, 1)
        lines.append(f"| {lang_code} | {cnt} | {pct}% |")

    lines += [
        "",
        "## Review text coverage",
        "",
        f"- Rows with non-empty full_review_text: {(df['full_review_text'] != '').sum()}",
        f"- Rows with positive text only: {((df['pos_review'] != '') & (df['neg_review'] == '')).sum()}",
        f"- Rows with both positive and negative text: {((df['pos_review'] != '') & (df['neg_review'] != '')).sum()}",
    ]

    QUALITY_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Data quality report written to %s", QUALITY_REPORT)

    state.logs.append(
        f"Cleaning complete: {rows_before} -> {rows_after} rows. "
        f"Sentiment: {sentiment_counts}"
    )
    state.results["cleaning"] = {
        "status": "success",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "sentiment_distribution": sentiment_counts,
        "report": str(QUALITY_REPORT),
        "clean_data": str(CLEAN_CSV),
        "features_artifact": str(FEATURES_ARTIFACT),
    }
    return state
