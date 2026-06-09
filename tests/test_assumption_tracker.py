"""
tests/test_assumption_tracker.py
----------------------------------
Unit tests for the Assumption Tracker.

All tests use a tmp_path fixture for file I/O isolation — no writes to
the real .cache directory. Tests cover:
  - Logging entries (creates file, correct fields)
  - Load / reload from disk (JSON round-trip)
  - Diff between two entries (changed / unchanged counts)
  - Edge cases: diff on identical assumptions, missing fields
  - Summary and diff_summary rendering (ASCII only)

Run with:
    pytest tests/test_assumption_tracker.py -v
"""

import pytest

from data.models.dcf import DCFAssumptions, WACCInputs
from core.assumption_tracker import AssumptionTracker


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

BASE_ASSUMPTIONS = DCFAssumptions(
    projection_years=5,
    revenue_growth_rates=[0.08] * 5,
    ebitda_margin=0.25,
    da_as_pct_revenue=0.04,
    tax_rate=0.21,
    capex_as_pct_revenue=0.05,
    nwc_change_as_pct_revenue_delta=0.01,
    wacc_override=0.09,
    terminal_growth_rate=0.025,
    net_debt=30_000.0,
    shares_outstanding=15_500.0,
)

BULL_ASSUMPTIONS = BASE_ASSUMPTIONS.model_copy(update={
    "ebitda_margin": 0.30,
    "revenue_growth_rates": [0.12] * 5,
    "wacc_override": 0.085,
    "terminal_growth_rate": 0.028,
})


@pytest.fixture
def tracker(tmp_path):
    return AssumptionTracker(storage_dir=tmp_path)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TestLog:
    def test_log_returns_entry(self, tracker):
        entry = tracker.log("AAPL", BASE_ASSUMPTIONS, label="base")
        assert entry.ticker == "AAPL"
        assert entry.label == "base"
        assert entry.id is not None

    def test_log_creates_file(self, tracker, tmp_path):
        tracker.log("AAPL", BASE_ASSUMPTIONS, label="base")
        assert (tmp_path / "AAPL_assumptions.json").exists()

    def test_log_normalizes_ticker_to_uppercase(self, tracker):
        entry = tracker.log("aapl", BASE_ASSUMPTIONS, label="x")
        assert entry.ticker == "AAPL"

    def test_log_stores_implied_price(self, tracker):
        entry = tracker.log("MSFT", BASE_ASSUMPTIONS, "run", implied_price=250.0)
        assert entry.implied_price == 250.0

    def test_log_stores_wacc_effective(self, tracker):
        entry = tracker.log("MSFT", BASE_ASSUMPTIONS, "run", wacc_effective=0.09)
        assert abs(entry.wacc_effective - 0.09) < 1e-9

    def test_log_multiple_entries_appends(self, tracker):
        tracker.log("AAPL", BASE_ASSUMPTIONS, "first")
        tracker.log("AAPL", BULL_ASSUMPTIONS, "second")
        history = tracker.get_history("AAPL")
        assert len(history) == 2

    def test_log_separate_tickers_independent(self, tracker):
        tracker.log("AAPL", BASE_ASSUMPTIONS, "a")
        tracker.log("MSFT", BULL_ASSUMPTIONS, "b")
        assert len(tracker.get_history("AAPL")) == 1
        assert len(tracker.get_history("MSFT")) == 1


# ---------------------------------------------------------------------------
# Persistence (JSON round-trip)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_reloaded_entry_fields_match(self, tracker):
        tracker.log("GOOG", BASE_ASSUMPTIONS, label="test", implied_price=99.5)
        # Create a fresh tracker pointing to the same dir
        tracker2 = AssumptionTracker(storage_dir=tracker._storage_dir)
        history = tracker2.get_history("GOOG")
        assert len(history) == 1
        assert history[0].label == "test"
        assert abs(history[0].implied_price - 99.5) < 1e-6

    def test_empty_history_for_unknown_ticker(self, tracker):
        assert tracker.get_history("UNKNOWN") == []

    def test_assumptions_round_trip(self, tracker):
        tracker.log("TSLA", BASE_ASSUMPTIONS, "orig")
        history = tracker.get_history("TSLA")
        reloaded = history[0].assumptions
        assert abs(reloaded.ebitda_margin - BASE_ASSUMPTIONS.ebitda_margin) < 1e-9
        assert abs(reloaded.terminal_growth_rate - BASE_ASSUMPTIONS.terminal_growth_rate) < 1e-9


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

class TestDiff:
    def test_diff_detects_changed_fields(self, tracker):
        e1 = tracker.log("AAPL", BASE_ASSUMPTIONS, "base")
        e2 = tracker.log("AAPL", BULL_ASSUMPTIONS, "bull")
        diff = tracker.diff(e1, e2)
        assert len(diff.changed) > 0

    def test_diff_shows_ebitda_margin_change(self, tracker):
        e1 = tracker.log("AAPL", BASE_ASSUMPTIONS, "base")
        e2 = tracker.log("AAPL", BULL_ASSUMPTIONS, "bull")
        diff = tracker.diff(e1, e2)
        changed_labels = [fd.label for fd in diff.changed]
        assert "EBITDA Margin" in changed_labels

    def test_diff_change_abs_correct(self, tracker):
        e1 = tracker.log("AAPL", BASE_ASSUMPTIONS, "base")
        e2 = tracker.log("AAPL", BULL_ASSUMPTIONS, "bull")
        diff = tracker.diff(e1, e2)
        margin_diff = next(fd for fd in diff.changed if fd.label == "EBITDA Margin")
        # BULL margin is 0.30, BASE is 0.25 → abs change = 0.05
        assert abs(margin_diff.change_abs - 0.05) < 1e-9

    def test_diff_identical_assumptions_no_changes(self, tracker):
        e1 = tracker.log("AAPL", BASE_ASSUMPTIONS, "a")
        e2 = tracker.log("AAPL", BASE_ASSUMPTIONS, "b")
        diff = tracker.diff(e1, e2)
        assert diff.changed == []
        assert diff.unchanged_count > 0

    def test_diff_ticker_matches(self, tracker):
        e1 = tracker.log("AAPL", BASE_ASSUMPTIONS, "a")
        e2 = tracker.log("AAPL", BULL_ASSUMPTIONS, "b")
        diff = tracker.diff(e1, e2)
        assert diff.ticker == "AAPL"

    def test_diff_entry_labels_stored(self, tracker):
        e1 = tracker.log("AAPL", BASE_ASSUMPTIONS, "base-case")
        e2 = tracker.log("AAPL", BULL_ASSUMPTIONS, "bull-case")
        diff = tracker.diff(e1, e2)
        assert diff.entry_a_label == "base-case"
        assert diff.entry_b_label == "bull-case"

    def test_diff_latest_returns_last_two(self, tracker):
        tracker.log("AAPL", BASE_ASSUMPTIONS, "first")
        tracker.log("AAPL", BULL_ASSUMPTIONS, "second")
        diff = tracker.diff_latest("AAPL")
        assert diff is not None
        assert diff.entry_a_label == "first"
        assert diff.entry_b_label == "second"

    def test_diff_latest_none_when_single_entry(self, tracker):
        tracker.log("AAPL", BASE_ASSUMPTIONS, "only")
        assert tracker.diff_latest("AAPL") is None

    def test_diff_latest_none_when_no_entries(self, tracker):
        assert tracker.diff_latest("AAPL") is None


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contains_ticker(self, tracker):
        tracker.log("AAPL", BASE_ASSUMPTIONS, "test")
        s = tracker.summary("AAPL")
        assert "AAPL" in s

    def test_summary_contains_label(self, tracker):
        tracker.log("AAPL", BASE_ASSUMPTIONS, "my-label")
        s = tracker.summary("AAPL")
        assert "my-label" in s

    def test_summary_shows_entry_count(self, tracker):
        tracker.log("AAPL", BASE_ASSUMPTIONS, "a")
        tracker.log("AAPL", BULL_ASSUMPTIONS, "b")
        s = tracker.summary("AAPL")
        assert "2 entries" in s

    def test_summary_ascii_only(self, tracker):
        tracker.log("AAPL", BASE_ASSUMPTIONS, "test", implied_price=100.0, wacc_effective=0.09)
        tracker.summary("AAPL").encode("ascii")

    def test_diff_summary_ascii_only(self, tracker):
        e1 = tracker.log("AAPL", BASE_ASSUMPTIONS, "base")
        e2 = tracker.log("AAPL", BULL_ASSUMPTIONS, "bull")
        diff = tracker.diff(e1, e2)
        tracker.diff_summary(diff).encode("ascii")

    def test_summary_empty_when_no_entries(self, tracker):
        s = tracker.summary("AAPL")
        assert "No entries" in s
