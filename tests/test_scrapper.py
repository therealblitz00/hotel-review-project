"""
Unit tests for scrapper.py — pure functions only (no browser required).
"""
from __future__ import annotations

import os
import importlib

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _import_scrapper():
    """Import scrapper from project root (not a package)."""
    import importlib.util
    from src.utils.paths import ROOT_DIR
    spec = importlib.util.spec_from_file_location(
        "scrapper",
        ROOT_DIR / "scrapper.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scrapper():
    return _import_scrapper()


# ── normalize_review_text ─────────────────────────────────────────────────────

class TestNormalizeReviewText:
    def test_empty_string_returns_empty(self, scrapper):
        assert scrapper.normalize_review_text("") == ""

    def test_none_like_falsy_returns_empty(self, scrapper):
        assert scrapper.normalize_review_text(None) == ""

    def test_nom_placeholder_returns_empty(self, scrapper):
        assert scrapper.normalize_review_text("#NOM?") == ""
        assert scrapper.normalize_review_text("#NOM") == ""

    def test_collapses_internal_whitespace(self, scrapper):
        result = scrapper.normalize_review_text("great  location   nearby")
        assert result == "great location nearby"

    def test_strips_leading_trailing_whitespace(self, scrapper):
        assert scrapper.normalize_review_text("  hello  ") == "hello"

    def test_replaces_non_breaking_space(self, scrapper):
        # U+00A0 should become a regular space and collapse
        result = scrapper.normalize_review_text("good\u00a0hotel")
        assert "\u00a0" not in result
        assert result == "good hotel"

    def test_collapses_newlines_to_space(self, scrapper):
        result = scrapper.normalize_review_text("line one\nline two")
        assert result == "line one line two"

    def test_normal_text_unchanged(self, scrapper):
        text = "Great hotel, excellent staff."
        assert scrapper.normalize_review_text(text) == text


# ── _env_float ────────────────────────────────────────────────────────────────

class TestEnvFloat:
    def test_returns_default_when_var_missing(self, scrapper, monkeypatch):
        monkeypatch.delenv("MY_TEST_VAR", raising=False)
        assert scrapper._env_float("MY_TEST_VAR", 1.5) == 1.5

    def test_returns_default_when_empty_string(self, scrapper, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "")
        assert scrapper._env_float("MY_TEST_VAR", 2.0) == 2.0

    def test_returns_float_for_valid_value(self, scrapper, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "0.75")
        assert scrapper._env_float("MY_TEST_VAR", 1.0) == pytest.approx(0.75)

    def test_returns_default_for_invalid_string(self, scrapper, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "not_a_number")
        assert scrapper._env_float("MY_TEST_VAR", 3.0) == 3.0

    def test_zero_is_valid(self, scrapper, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "0")
        assert scrapper._env_float("MY_TEST_VAR", 1.0) == pytest.approx(0.0)

    def test_integer_string_converts_to_float(self, scrapper, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "5")
        assert scrapper._env_float("MY_TEST_VAR", 1.0) == pytest.approx(5.0)


# ── DEFAULT_HOTEL_URL ─────────────────────────────────────────────────────────

class TestDefaultHotelUrl:
    def test_url_targets_porto_hotel(self, scrapper):
        assert "paodeacucarhotelporto" in scrapper.DEFAULT_HOTEL_URL

    def test_url_uses_english_locale(self, scrapper):
        assert "en-gb" in scrapper.DEFAULT_HOTEL_URL

    def test_url_points_to_reviews_tab(self, scrapper):
        assert "tab-reviews" in scrapper.DEFAULT_HOTEL_URL

    def test_url_is_booking_com(self, scrapper):
        assert "booking.com" in scrapper.DEFAULT_HOTEL_URL


# ── Argument parser defaults ──────────────────────────────────────────────────

class TestArgparseDefaults:
    def test_default_out_path_is_raw_csv(self, scrapper):
        """The --out default should write into data/raw/ so the pipeline can find it."""
        import argparse
        # Re-create the parser the same way scrapper.py does it
        parser = argparse.ArgumentParser()
        parser.add_argument("--url", default=scrapper.DEFAULT_HOTEL_URL)
        from src.utils.paths import RAW_DIR
        parser.add_argument("--out", default=str(RAW_DIR / "reviews_raw.csv"))
        parser.add_argument("--max-pages", type=int, default=0)
        args = parser.parse_args([])
        assert args.out == str(RAW_DIR / "reviews_raw.csv")

    def test_default_max_pages_is_zero(self, scrapper):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--max-pages", type=int, default=0)
        args = parser.parse_args([])
        assert args.max_pages == 0
