"""
tests/test_financial_statements.py
------------------------------------
Unit tests for the Financial Statement Analyzer.

Tests use a lightweight MockFetcher that returns pre-built FullFinancialHistory
objects — no network calls. This lets us test ratio computation logic in
isolation from yfinance availability.

Run with:
    pytest tests/test_financial_statements.py -v
"""

import pytest

from data.models.financials import (
    BalanceSheet,
    CashFlowStatement,
    FinancialRatios,
    FinancialSnapshot,
    FullFinancialHistory,
    CompanyProfile,
    IncomeStatement,
    RawStatements,
)
from core.financial_statements import FinancialStatementAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_snapshot(
    period: str,
    revenue: float,
    gross_profit: float,
    operating_income: float,
    net_income: float,
    total_assets: float,
    total_equity: float,
    total_debt: float,
    cash: float,
    current_assets: float,
    current_liabilities: float,
    ocf: float,
    capex: float,
    income_tax: float = 0.0,
    pretax_income: float = 0.0,
    interest_expense: float = 0.0,
    cogs: float = 0.0,
    depreciation: float = 0.0,
    eps_diluted: float = 0.0,
    shares_diluted: float = 1.0,
) -> FinancialSnapshot:
    inc = IncomeStatement(
        period=period,
        revenue=revenue,
        gross_profit=gross_profit,
        cost_of_revenue=cogs,
        operating_income=operating_income,
        net_income=net_income,
        pretax_income=pretax_income,
        income_tax=income_tax,
        interest_expense=interest_expense,
        depreciation_amortization=depreciation,
        eps_diluted=eps_diluted,
        shares_diluted=shares_diluted,
    )
    bs = BalanceSheet(
        period=period,
        total_assets=total_assets,
        total_stockholders_equity=total_equity,
        total_debt=total_debt,
        net_debt=total_debt - cash,
        cash_and_equivalents=cash,
        total_current_assets=current_assets,
        total_current_liabilities=current_liabilities,
    )
    cf = CashFlowStatement(
        period=period,
        operating_cash_flow=ocf,
        capital_expenditures=capex,
        free_cash_flow=ocf + capex,
    )
    return FinancialSnapshot(
        period=period,
        statements=RawStatements(income_statement=inc, balance_sheet=bs, cash_flow_statement=cf),
        ratios=FinancialRatios(period=period),
    )


class MockFetcher:
    """Returns a two-period FullFinancialHistory without any network calls."""

    def fetch(self, ticker: str) -> FullFinancialHistory:
        s1 = _make_snapshot(
            period="2022-09-30",
            revenue=400_000.0, gross_profit=170_000.0, operating_income=120_000.0,
            net_income=90_000.0, total_assets=350_000.0, total_equity=70_000.0,
            total_debt=100_000.0, cash=50_000.0, current_assets=130_000.0,
            current_liabilities=130_000.0, ocf=110_000.0, capex=-10_000.0,
            income_tax=22_000.0, pretax_income=110_000.0, interest_expense=3_000.0,
            cogs=230_000.0, depreciation=10_000.0, eps_diluted=5.0, shares_diluted=16_000.0,
        )
        s2 = _make_snapshot(
            period="2023-09-30",
            revenue=480_000.0, gross_profit=210_000.0, operating_income=150_000.0,
            net_income=110_000.0, total_assets=390_000.0, total_equity=80_000.0,
            total_debt=90_000.0, cash=60_000.0, current_assets=160_000.0,
            current_liabilities=140_000.0, ocf=130_000.0, capex=-12_000.0,
            income_tax=27_000.0, pretax_income=140_000.0, interest_expense=2_500.0,
            cogs=270_000.0, depreciation=11_000.0, eps_diluted=6.2, shares_diluted=15_500.0,
        )
        return FullFinancialHistory(
            profile=CompanyProfile(ticker=ticker, name="Mock Corp"),
            annual_snapshots=[s1, s2],
        )


@pytest.fixture
def history():
    analyzer = FinancialStatementAnalyzer(fetcher=MockFetcher())
    return analyzer.analyze("MOCK")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProfileAndPeriods:
    def test_ticker_uppercase(self, history):
        assert history.profile.ticker == "MOCK"

    def test_two_periods(self, history):
        assert len(history.annual_snapshots) == 2

    def test_period_strings(self, history):
        assert history.annual_snapshots[0].period == "2022-09-30"
        assert history.annual_snapshots[1].period == "2023-09-30"

    def test_latest_is_most_recent(self, history):
        assert history.latest.period == "2023-09-30"


class TestProfitabilityRatios:
    def test_gross_margin(self, history):
        snap = history.annual_snapshots[1]
        # 210_000 / 480_000 = 0.4375
        assert abs(snap.ratios.profitability.gross_margin - 0.4375) < 1e-6

    def test_operating_margin(self, history):
        snap = history.annual_snapshots[1]
        # 150_000 / 480_000 = 0.3125
        assert abs(snap.ratios.profitability.operating_margin - 0.3125) < 1e-6

    def test_net_margin(self, history):
        snap = history.annual_snapshots[1]
        # 110_000 / 480_000 ≈ 0.22917
        assert abs(snap.ratios.profitability.net_margin - (110_000 / 480_000)) < 1e-6

    def test_roic_is_computed(self, history):
        snap = history.annual_snapshots[1]
        assert snap.ratios.profitability.return_on_invested_capital is not None

    def test_roe(self, history):
        snap = history.annual_snapshots[1]
        # 110_000 / 80_000 = 1.375
        assert abs(snap.ratios.profitability.return_on_equity - 1.375) < 1e-6


class TestLiquidityRatios:
    def test_current_ratio(self, history):
        snap = history.annual_snapshots[1]
        # 160_000 / 140_000 ≈ 1.1429
        assert abs(snap.ratios.liquidity.current_ratio - (160_000 / 140_000)) < 1e-6

    def test_ocf_ratio(self, history):
        snap = history.annual_snapshots[1]
        # 130_000 / 140_000 ≈ 0.9286
        assert abs(snap.ratios.liquidity.operating_cash_flow_ratio - (130_000 / 140_000)) < 1e-6


class TestLeverageRatios:
    def test_debt_to_equity(self, history):
        snap = history.annual_snapshots[1]
        # 90_000 / 80_000 = 1.125
        assert abs(snap.ratios.leverage.debt_to_equity - 1.125) < 1e-6

    def test_interest_coverage(self, history):
        snap = history.annual_snapshots[1]
        # 150_000 / 2_500 = 60.0
        assert abs(snap.ratios.leverage.interest_coverage - 60.0) < 1e-6


class TestGrowthRates:
    def test_first_period_has_no_growth(self, history):
        snap = history.annual_snapshots[0]
        assert snap.ratios.growth.revenue_growth is None

    def test_revenue_growth(self, history):
        snap = history.annual_snapshots[1]
        # (480_000 - 400_000) / 400_000 = 0.20
        assert abs(snap.ratios.growth.revenue_growth - 0.20) < 1e-6

    def test_eps_growth(self, history):
        snap = history.annual_snapshots[1]
        # (6.2 - 5.0) / 5.0 = 0.24
        assert abs(snap.ratios.growth.eps_growth - 0.24) < 1e-6


class TestSummary:
    def test_summary_contains_ticker(self, history):
        analyzer = FinancialStatementAnalyzer(fetcher=MockFetcher())
        summary = analyzer.summary(history)
        assert "MOCK" in summary

    def test_summary_contains_periods(self, history):
        analyzer = FinancialStatementAnalyzer(fetcher=MockFetcher())
        summary = analyzer.summary(history)
        assert "2022-09-30" in summary
        assert "2023-09-30" in summary
