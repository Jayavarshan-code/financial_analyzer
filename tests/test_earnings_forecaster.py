"""
tests/test_earnings_forecaster.py
------------------------------------
Unit tests for the Earnings Forecasting Engine.

Reuses the same mock history as test_revenue_forecaster (constant 10% revenue
growth, fixed 20% net margin, fixed 30% EBITDA margin).

Known exact values:
  Base revenue 2023: $146,410M   Net margin: 20%   Shares: 15,000M
  Base net income:   $29,282M    Base EPS: $1.952

With constant margins, projected EPS inherits revenue's 10% CAGR exactly:
  Year 1 (2024): revenue = $161,051M → NI = $32,210M → EPS = $2.147

Run with:
    pytest tests/test_earnings_forecaster.py -v
"""

import pytest

from data.models.financials import (
    BalanceSheet,
    CashFlowStatement,
    CompanyProfile,
    FinancialRatios,
    FinancialSnapshot,
    FullFinancialHistory,
    IncomeStatement,
    ProfitabilityRatios,
    RawStatements,
)
from data.models.forecast import EarningsForecastMethod
from analytics.revenue_forecaster import RevenueForecastEngine
from analytics.earnings_forecaster import EarningsForecastEngine


# ---------------------------------------------------------------------------
# Mock data (same revenue series as test_revenue_forecaster)
# ---------------------------------------------------------------------------

_REVENUES = [100_000.0, 110_000.0, 121_000.0, 133_100.0, 146_410.0]
_YEARS    = [2019, 2020, 2021, 2022, 2023]
_NET_MARGIN = 0.20
_EBITDA_MARGIN = 0.30
_SHARES = 15_000.0   # millions


def _make_history() -> FullFinancialHistory:
    snaps = []
    for yr, rev in zip(_YEARS, _REVENUES):
        period = f"{yr}-09-30"
        ni = rev * _NET_MARGIN
        ebitda = rev * _EBITDA_MARGIN
        eps = ni / _SHARES
        inc = IncomeStatement(
            period=period, revenue=rev, net_income=ni, ebitda=ebitda,
            depreciation_amortization=rev * 0.04, eps_diluted=eps, shares_diluted=_SHARES,
        )
        bs = BalanceSheet(
            period=period, total_assets=rev*1.5, total_stockholders_equity=rev*0.5,
            net_debt=rev*0.1, total_debt=rev*0.2, cash_and_equivalents=rev*0.1,
            total_current_assets=rev*0.4, total_current_liabilities=rev*0.3,
        )
        cf = CashFlowStatement(
            period=period, operating_cash_flow=rev*0.25,
            capital_expenditures=-(rev*0.04), free_cash_flow=rev*0.21,
        )
        # Pre-populate profitability ratios (as analyzer would)
        prof = ProfitabilityRatios(
            ebitda_margin=_EBITDA_MARGIN,
            net_margin=_NET_MARGIN,
            gross_margin=0.45,
        )
        ratios = FinancialRatios(period=period, profitability=prof)
        snaps.append(FinancialSnapshot(
            period=period,
            statements=RawStatements(income_statement=inc, balance_sheet=bs, cash_flow_statement=cf),
            ratios=ratios,
        ))

    return FullFinancialHistory(
        profile=CompanyProfile(ticker="MOCK", name="Mock Corp", current_price=25.0,
                               shares_outstanding=_SHARES, beta=1.0),
        annual_snapshots=snaps,
    )


@pytest.fixture
def history():
    return _make_history()


@pytest.fixture
def revenue_forecast(history):
    suite = RevenueForecastEngine().run(history, n_years=5)
    # Use CAGR method for deterministic testing
    cagr_result = next(r for r in suite.results if r.method.value == "cagr")
    return cagr_result


@pytest.fixture
def earn_suite(history, revenue_forecast):
    return EarningsForecastEngine().run(history, revenue_forecast)


# ---------------------------------------------------------------------------
# Margin-Based method
# ---------------------------------------------------------------------------

class TestMarginBased:
    def test_method_is_margin_based(self, earn_suite):
        assert earn_suite.recommended.method == EarningsForecastMethod.MARGIN_BASED

    def test_avg_net_margin_near_20pct(self, earn_suite):
        assert abs(earn_suite.recommended.avg_net_margin - _NET_MARGIN) < 0.01

    def test_avg_ebitda_margin_near_30pct(self, earn_suite):
        assert abs(earn_suite.recommended.avg_ebitda_margin - _EBITDA_MARGIN) < 0.01

    def test_projects_5_years(self, earn_suite):
        assert len(earn_suite.recommended.projected_only) == 5

    def test_eps_positive(self, earn_suite):
        for p in earn_suite.recommended.projected_only:
            assert p.eps > 0

    def test_eps_formula_correct(self, earn_suite, revenue_forecast):
        # EPS = (revenue * net_margin) / shares
        for ep, fp in zip(earn_suite.recommended.projected_only, revenue_forecast.projected_only):
            expected_eps = fp.value * _NET_MARGIN / _SHARES
            assert abs(ep.eps - expected_eps) < 0.001

    def test_net_income_formula_correct(self, earn_suite, revenue_forecast):
        for ep, fp in zip(earn_suite.recommended.projected_only, revenue_forecast.projected_only):
            expected_ni = fp.value * _NET_MARGIN
            assert abs(ep.net_income - expected_ni) < 1.0

    def test_cagr_eps_near_revenue_cagr(self, earn_suite):
        # Constant margin → EPS CAGR = revenue CAGR ≈ 10%
        assert abs(earn_suite.recommended.cagr_eps - 0.10) < 0.02

    def test_eps_ci_present(self, earn_suite):
        for p in earn_suite.recommended.projected_only:
            assert p.eps_lower_95 is not None
            assert p.eps_upper_95 is not None

    def test_eps_ci_lower_lt_value(self, earn_suite):
        for p in earn_suite.recommended.projected_only:
            assert p.eps_lower_95 <= p.eps

    def test_eps_monotone_increasing(self, earn_suite):
        eps_vals = [p.eps for p in earn_suite.recommended.projected_only]
        assert all(eps_vals[i] < eps_vals[i+1] for i in range(len(eps_vals)-1))


# ---------------------------------------------------------------------------
# EPS Trend method
# ---------------------------------------------------------------------------

class TestEpsTrend:
    def test_eps_trend_in_results(self, earn_suite):
        methods = [r.method for r in earn_suite.results]
        assert EarningsForecastMethod.EPS_TREND in methods

    def test_eps_trend_projects_5_years(self, earn_suite):
        eps_trend = next(r for r in earn_suite.results if r.method == EarningsForecastMethod.EPS_TREND)
        assert len(eps_trend.projected_only) == 5

    def test_eps_trend_ci_present(self, earn_suite):
        eps_trend = next(r for r in earn_suite.results if r.method == EarningsForecastMethod.EPS_TREND)
        for p in eps_trend.projected_only:
            assert p.eps_lower_95 is not None
            assert p.eps_upper_95 is not None


# ---------------------------------------------------------------------------
# Suite structure
# ---------------------------------------------------------------------------

class TestEarningsSuite:
    def test_ticker_matches(self, earn_suite):
        assert earn_suite.ticker == "MOCK"

    def test_at_least_two_results(self, earn_suite):
        assert len(earn_suite.results) >= 2

    def test_recommended_is_margin_based_when_available(self, earn_suite):
        assert earn_suite.recommended.method == EarningsForecastMethod.MARGIN_BASED

    def test_revenue_forecast_stored(self, earn_suite, revenue_forecast):
        assert earn_suite.revenue_forecast.method == revenue_forecast.method


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contains_ticker(self, earn_suite):
        s = EarningsForecastEngine().summary(earn_suite)
        assert "MOCK" in s

    def test_summary_contains_eps_label(self, earn_suite):
        s = EarningsForecastEngine().summary(earn_suite)
        assert "EPS" in s

    def test_summary_contains_method_name(self, earn_suite):
        s = EarningsForecastEngine().summary(earn_suite)
        assert "Margin" in s

    def test_summary_ascii_only(self, earn_suite):
        s = EarningsForecastEngine().summary(earn_suite)
        s.encode("ascii")

    def test_summary_shows_recommended_marker(self, earn_suite):
        s = EarningsForecastEngine().summary(earn_suite)
        assert "(*)" in s
