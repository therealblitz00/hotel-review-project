"""
Tests for src/agents/cleaning.py — pure helpers and pipeline integration via tmp_path.
"""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from src.agents.cleaning import _extract_nr_nights, _score_to_sentiment
from src.orchestration.state import WorkflowState


# ── _extract_nr_nights ────────────────────────────────────────────────────────

class TestExtractNrNights:
    def test_typical_format(self):
        assert _extract_nr_nights("3 nights") == 3

    def test_single_night(self):
        assert _extract_nr_nights("1 night") == 1

    def test_with_emoji(self):
        assert _extract_nr_nights("3 nights ✈") == 3

    def test_multidigit(self):
        assert _extract_nr_nights("14 nights") == 14

    def test_nan_returns_none(self):
        assert _extract_nr_nights(float("nan")) is None

    def test_pandas_na_returns_none(self):
        assert _extract_nr_nights(pd.NA) is None

    def test_no_digit_returns_none(self):
        assert _extract_nr_nights("no digits here") is None

    def test_leading_digit_wins(self):
        # Should extract the first digit group
        assert _extract_nr_nights("2 nights (3 rooms)") == 2


# ── _score_to_sentiment ───────────────────────────────────────────────────────

class TestScoreToSentiment:
    @pytest.mark.parametrize("score,expected", [
        (10.0, "positive"),
        (8.0,  "positive"),   # boundary — exactly 8
        (8.1,  "positive"),
        (7.9,  "neutral"),
        (7.0,  "neutral"),
        (6.0,  "neutral"),    # boundary — exactly 6
        (5.9,  "negative"),
        (5.0,  "negative"),
        (1.0,  "negative"),
    ])
    def test_boundaries(self, score, expected):
        assert _score_to_sentiment(score) == expected


# ── Integration: run_cleaning with synthetic data ─────────────────────────────

def _make_raw_df() -> pd.DataFrame:
    """Minimal valid raw DataFrame matching the scraper's output schema."""
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Carlos", "Diana", "Eve"],
        "country": ["united kingdom", "SPAIN", "Portugal", "france", "italy"],
        "room_type": ["Budget Double Room"] * 5,
        "nr_nights": ["3 nights ✈", "1 night", "2 nights", "5 nights", "7 nights"],
        "date": ["March 2024", "April 2024", "May 2024", "June 2024", "July 2024"],
        "traveler_type": ["Couple", "Solo traveller", "Family", "Couple", "Group"],
        "title_review": ["Great stay"] * 5,
        "pos_review": ["Excellent location", "Good staff", "Clean room", "Nice breakfast", "Great view"],
        "neg_review": ["", "Noisy", "", "Small bathroom", ""],
        "score": [9.0, 7.0, 5.0, 8.0, 8.5],
        "hotel_response": [None, None, None, None, None],   # 100% null → should be dropped
    })


@pytest.fixture()
def patched_cleaning(tmp_path, monkeypatch):
    """
    Patch module-level path constants in cleaning.py so the agent
    reads/writes inside tmp_path instead of the real project directories.
    """
    import src.agents.cleaning as cleaning_mod

    raw_csv = tmp_path / "reviews_raw.csv"
    clean_csv = tmp_path / "reviews_clean.csv"
    quality_report = tmp_path / "data_quality_report.md"
    features_artifact = tmp_path / "features_summary.json"

    monkeypatch.setattr(cleaning_mod, "RAW_CSV", raw_csv)
    monkeypatch.setattr(cleaning_mod, "CLEAN_CSV", clean_csv)
    monkeypatch.setattr(cleaning_mod, "QUALITY_REPORT", quality_report)
    monkeypatch.setattr(cleaning_mod, "FEATURES_ARTIFACT", features_artifact)

    # Also patch the directory constants so mkdir calls work on tmp_path
    monkeypatch.setattr(cleaning_mod, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(cleaning_mod, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(cleaning_mod, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(cleaning_mod, "RAW_DIR", tmp_path)

    return cleaning_mod, {
        "raw_csv": raw_csv,
        "clean_csv": clean_csv,
        "quality_report": quality_report,
        "features_artifact": features_artifact,
    }


class TestRunCleaningIntegration:
    def _write_raw(self, paths, df=None):
        df = df if df is not None else _make_raw_df()
        df.to_csv(paths["raw_csv"], index=False, encoding="utf-8-sig")
        return df

    def test_returns_success_status(self, patched_cleaning):
        cleaning_mod, paths = patched_cleaning
        self._write_raw(paths)
        from src.orchestration.state import WorkflowState
        state = WorkflowState()
        state = cleaning_mod.run_cleaning(state)
        assert state.results["cleaning"]["status"] == "success"

    def test_hotel_response_column_dropped(self, patched_cleaning):
        cleaning_mod, paths = patched_cleaning
        self._write_raw(paths)
        state = WorkflowState()
        cleaning_mod.run_cleaning(state)
        df_clean = pd.read_csv(paths["clean_csv"], encoding="utf-8-sig")
        assert "hotel_response" not in df_clean.columns

    def test_nr_nights_is_integer(self, patched_cleaning):
        cleaning_mod, paths = patched_cleaning
        self._write_raw(paths)
        state = WorkflowState()
        cleaning_mod.run_cleaning(state)
        df_clean = pd.read_csv(paths["clean_csv"], encoding="utf-8-sig")
        assert pd.api.types.is_integer_dtype(df_clean["nr_nights"])

    def test_date_parsed_format(self, patched_cleaning):
        cleaning_mod, paths = patched_cleaning
        self._write_raw(paths)
        state = WorkflowState()
        cleaning_mod.run_cleaning(state)
        df_clean = pd.read_csv(paths["clean_csv"], encoding="utf-8-sig")
        # All values should match YYYY-MM
        import re
        assert df_clean["date_parsed"].str.match(r"\d{4}-\d{2}").all()

    def test_sentiment_labels_valid(self, patched_cleaning):
        cleaning_mod, paths = patched_cleaning
        self._write_raw(paths)
        state = WorkflowState()
        cleaning_mod.run_cleaning(state)
        df_clean = pd.read_csv(paths["clean_csv"], encoding="utf-8-sig")
        assert set(df_clean["sentiment"].unique()).issubset({"positive", "neutral", "negative"})

    def test_sentiment_matches_scores(self, patched_cleaning):
        cleaning_mod, paths = patched_cleaning
        self._write_raw(paths)
        state = WorkflowState()
        cleaning_mod.run_cleaning(state)
        df_clean = pd.read_csv(paths["clean_csv"], encoding="utf-8-sig")
        # score=9 → positive; score=7 → neutral; score=5 → negative
        for _, row in df_clean.iterrows():
            if row["score"] >= 8.0:
                assert row["sentiment"] == "positive"
            elif row["score"] >= 6.0:
                assert row["sentiment"] == "neutral"
            else:
                assert row["sentiment"] == "negative"

    def test_country_is_title_case(self, patched_cleaning):
        cleaning_mod, paths = patched_cleaning
        self._write_raw(paths)
        state = WorkflowState()
        cleaning_mod.run_cleaning(state)
        df_clean = pd.read_csv(paths["clean_csv"], encoding="utf-8-sig")
        assert (df_clean["country"] == df_clean["country"].str.title()).all()

    def test_full_review_text_created(self, patched_cleaning):
        cleaning_mod, paths = patched_cleaning
        self._write_raw(paths)
        state = WorkflowState()
        cleaning_mod.run_cleaning(state)
        df_clean = pd.read_csv(paths["clean_csv"], encoding="utf-8-sig")
        assert "full_review_text" in df_clean.columns
        assert (df_clean["full_review_text"].str.len() > 0).all()

    def test_empty_review_rows_dropped(self, patched_cleaning):
        """A row where both pos_review and neg_review are empty must be dropped."""
        cleaning_mod, paths = patched_cleaning
        df = _make_raw_df()
        # Force one row to have no review text at all
        df.loc[0, "pos_review"] = ""
        df.loc[0, "neg_review"] = ""
        self._write_raw(paths, df)
        state = WorkflowState()
        cleaning_mod.run_cleaning(state)
        result = state.results["cleaning"]
        assert result["rows_after"] == result["rows_before"] - 1

    def test_quality_report_written(self, patched_cleaning):
        cleaning_mod, paths = patched_cleaning
        self._write_raw(paths)
        state = WorkflowState()
        cleaning_mod.run_cleaning(state)
        assert paths["quality_report"].exists()
        content = paths["quality_report"].read_text(encoding="utf-8")
        assert "# Data Quality Report" in content

    def test_features_artifact_written(self, patched_cleaning):
        cleaning_mod, paths = patched_cleaning
        self._write_raw(paths)
        state = WorkflowState()
        cleaning_mod.run_cleaning(state)
        assert paths["features_artifact"].exists()
        data = json.loads(paths["features_artifact"].read_text(encoding="utf-8"))
        assert "rows_before" in data
        assert "rows_after" in data
        assert "sentiment_distribution" in data

    def test_missing_raw_file_returns_failed_status(self, patched_cleaning):
        cleaning_mod, paths = patched_cleaning
        # Do NOT write the raw CSV → should fail gracefully
        state = WorkflowState()
        state = cleaning_mod.run_cleaning(state)
        assert state.results["cleaning"]["status"] == "failed"
        assert state.warnings  # a warning should be appended
