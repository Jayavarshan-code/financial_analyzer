"""
tests/test_comps_engine.py
---------------------------
Unit tests for the Comparable Valuation Engine.

All tests use a deterministic MockFetcher that returns pre-built
FullFinancialHistory objects for a fixed set of tickers — no network required.

Peer set:
  SUBJ  - subject: price=$100, EV=10,000, EBITDA=800, revenue=4,000
  PEER1 - price=$50,  EV= 6,000, EBITDA=500, revenue=2,500, eps=3.0, FCF=400
  PEER2 - price=$80,  EV=12,000, EBITDA=900, revenue=5,000, eps=4.0, FCF=600
  PEER3 - price=$30,  EV= 3,000, EBITDA=200, revenue=1,500, eps=2.0, FCF=180

Expected peer multiples:
  PEER1: EV/EBITDA=12.0  EV/Rev=2.40  P/E=16.7  P/FCF=6.25  (mc=2,000)
  PEER2: EV/EBITDA=13.3  EV/Rev=2.40  P/E=20.0  P/FCF=6.67  (mc=4,000)
  PEER3: EV/EBITDA=15.0  EV/Rev=2.00  P/E=15.0  P/FCF=5.00  (mc=1,500)

Run with:
    pytest tests/test_comps_engine.py -v
"""

import pytest

from data.models.comps import CompsResult
from data.models.financials import (
    BalanceSheet,
    CashFlowStatement,
    CompanyProfile,
    FinancialRatios,
    FinancialSnapshot,
    FullFinancialHistory,
    IncomeStatement,
    RawStatements,
)
from core.comps_engine import CompsEngine, _extract_multiples, _stats


# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------

def _make_history(
    ticker: str,
    name: str,
    price: float,
    ev: float,
    market_cap: float,
    revenue: float,
    ebitda: float,
    net_income: float,
    eps: float,
    fcf: float,
    book_equity: float,
    net_debt: float,
    shares: float,
) -> FullFinancialHistory:
    inc = IncomeStatement(
        period="2024-09-30",
        revenue=revenue,
        ebitda=ebitda,
        net_income=net_income,
        eps_diluted=eps,
        shares_diluted=shares,
    )
    bs = BalanceSheet(
        period="2024-09-30",
        total_stockholders_equity=book_equity,
        net_debt=net_debt,
        total_debt=net_debt + 500,
        cash_and_equivalents=500,
        total_assets=book_equity + net_debt + 500,
        total_current_assets=1_000,
        total_current_liabilities=800,
    )
    cf = CashFlowStatement(
        period="2024-09-30",
        operating_cash_flow=fcf + 200,
        capital_expenditures=-200,
        free_cash_flow=fcf,
    )
    snap = FinancialSnapshot(
        period="2024-09-30",
        statements=RawStatements(
            income_statement=inc,
            balance_sheet=bs,
            cash_flow_statement=cf,
        ),
        ratios=FinancialRatios(period="2024-09-30"),
    )
    profile = CompanyProfile(
        ticker=ticker,
        name=name,
        sector="Technology",
        current_price=price,
        enterprise_value=ev,
        market_cap=market_cap,
        shares_outstanding=shares,
        beta=1.0,
    )
    return FullFinancialHistory(profile=profile, annual_snapshots=[snap])


SUBJECT_HISTORY = _make_history(
    "SUBJ", "Subject Corp",
    price=100.0, ev=10_000.0, market_cap=8_000.0,
    revenue=4_000.0, ebitda=800.0, net_income=500.0,
    eps=5.0, fcf=600.0, book_equity=4_000.0,
    net_debt=2_000.0, shares=80.0,
)

PEER1_HISTORY = _make_history(
    "PEER1", "Peer One Inc",
    price=50.0, ev=6_000.0, market_cap=2_000.0,
    revenue=2_500.0, ebitda=500.0, net_income=300.0,
    eps=3.0, fcf=400.0, book_equity=1_500.0,
    net_debt=500.0, shares=40.0,
)

PEER2_HISTORY = _make_history(
    "PEER2", "Peer Two Corp",
    price=80.0, ev=12_000.0, market_cap=4_000.0,
    revenue=5_000.0, ebitda=900.0, net_income=400.0,
    eps=4.0, fcf=600.0, book_equity=2_000.0,
    net_debt=800.0, shares=50.0,
)

PEER3_HISTORY = _make_history(
    "PEER3", "Peer Three Ltd",
    price=30.0, ev=3_000.0, market_cap=1_500.0,
    revenue=1_500.0, ebitda=200.0, net_income=200.0,
    eps=2.0, fcf=180.0, book_equity=800.0,
    net_debt=200.0, shares=50.0,
)

_HISTORIES = {
    "SUBJ": SUBJECT_HISTORY,
    "PEER1": PEER1_HISTORY,
    "PEER2": PEER2_HISTORY,
    "PEER3": PEER3_HISTORY,
}


class MockFetcher:
    def fetch(self, ticker: str) -> FullFinancialHistory:
        ticker = ticker.upper()
        if ticker not in _HISTORIES:
            raise ValueError(f"Unknown ticker: {ticker}")
        return _HISTORIES[ticker]


@pytest.fixture
def engine():
    return CompsEngine(fetcher=MockFetcher())


@pytest.fixture
def result(engine):
    return engine.run("SUBJ", ["PEER1", "PEER2", "PEER3"])


# ---------------------------------------------------------------------------
# _stats helper
# ---------------------------------------------------------------------------

class TestStats:
    def test_median_odd_count(self):
        s = _stats([10.0, 20.0, 30.0], "test")
        assert s.median == 20.0

    def test_median_even_count(self):
        s = _stats([10.0, 20.0, 30.0, 40.0], "test")
        assert s.median == 25.0

    def test_mean_correct(self):
        s = _stats([10.0, 20.0, 30.0], "test")
        assert abs(s.mean - 20.0) < 1e-9

    def test_none_values_excluded(self):
        s = _stats([10.0, None, 20.0, None, 30.0], "test")
        assert s.n_valid == 3

    def test_negative_values_excluded(self):
        s = _stats([10.0, -5.0, 20.0], "test")
        assert s.n_valid == 2

    def test_all_none_returns_n_valid_zero(self):
        s = _stats([None, None], "test")
        assert s.n_valid == 0
        assert s.median is None


# ---------------------------------------------------------------------------
# Multiple extraction
# ---------------------------------------------------------------------------

class TestExtractMultiples:
    def test_ev_to_ebitda(self):
        m = _extract_multiples(PEER1_HISTORY)
        # EV=6000, EBITDA=500 → 12.0
        assert abs(m.ev_to_ebitda - 12.0) < 1e-3

    def test_ev_to_revenue(self):
        m = _extract_multiples(PEER1_HISTORY)
        # EV=6000, Revenue=2500 → 2.4
        assert abs(m.ev_to_revenue - 2.4) < 1e-3

    def test_price_to_earnings(self):
        m = _extract_multiples(PEER1_HISTORY)
        # price=50, eps=3.0 → 16.67
        assert abs(m.price_to_earnings - 50.0 / 3.0) < 1e-3

    def test_price_to_fcf(self):
        m = _extract_multiples(PEER1_HISTORY)
        # mc=2000, fcf=400 → 5.0
        assert abs(m.price_to_fcf - 5.0) < 1e-3

    def test_negative_ebitda_gives_none_multiple(self):
        h = _make_history(
            "BAD", "Bad Corp",
            price=10.0, ev=1_000.0, market_cap=800.0,
            revenue=500.0, ebitda=-100.0,
            net_income=-50.0, eps=-1.0, fcf=-80.0,
            book_equity=200.0, net_debt=100.0, shares=80.0,
        )
        m = _extract_multiples(h)
        assert m.ev_to_ebitda is None
        assert m.price_to_earnings is None
        assert m.price_to_fcf is None


# ---------------------------------------------------------------------------
# Full CompsResult
# ---------------------------------------------------------------------------

class TestCompsResult:
    def test_subject_ticker(self, result: CompsResult):
        assert result.subject.ticker == "SUBJ"

    def test_peer_count(self, result: CompsResult):
        assert len(result.peers) == 3

    def test_peers_sorted_alphabetically(self, result: CompsResult):
        tickers = [p.ticker for p in result.peers]
        assert tickers == sorted(tickers)

    def test_ev_ebitda_median(self, result: CompsResult):
        # PEER1=6000/500=12.0, PEER2=12000/900=13.33, PEER3=3000/200=15.0
        # Sorted: [12.0, 13.33, 15.0] → median = 13.33
        assert result.stats.ev_to_ebitda.n_valid == 3
        median = result.stats.ev_to_ebitda.median
        assert median is not None
        assert 13.0 < median < 14.0

    def test_implied_from_ev_ebitda_is_positive(self, result: CompsResult):
        assert result.implied.implied_from_ev_ebitda is not None
        assert result.implied.implied_from_ev_ebitda > 0

    def test_implied_from_pe_formula(self, result: CompsResult):
        # implied = median_pe x subject_eps (=5.0)
        median_pe = result.stats.price_to_earnings.median
        if median_pe is not None:
            expected = median_pe * 5.0
            assert abs(result.implied.implied_from_pe - expected) < 0.01

    def test_implied_stores_current_price(self, result: CompsResult):
        assert result.implied.current_price == 100.0

    def test_failed_peer_fetch_drops_silently(self):
        engine = CompsEngine(fetcher=MockFetcher())
        # "UNKNOWN" will raise ValueError inside MockFetcher → silently dropped
        res = engine.run("SUBJ", ["PEER1", "UNKNOWN", "PEER2"])
        assert len(res.peers) == 2


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contains_subject_ticker(self, engine, result):
        s = engine.summary(result)
        assert "SUBJ" in s

    def test_summary_contains_all_peer_tickers(self, engine, result):
        s = engine.summary(result)
        for p in result.peers:
            assert p.ticker in s

    def test_summary_contains_median_label(self, engine, result):
        s = engine.summary(result)
        assert "Median" in s

    def test_summary_contains_implied_header(self, engine, result):
        s = engine.summary(result)
        assert "IMPLIED" in s

    def test_summary_ascii_only(self, engine, result):
        s = engine.summary(result)
        s.encode("ascii")   # raises UnicodeEncodeError if non-ASCII present
