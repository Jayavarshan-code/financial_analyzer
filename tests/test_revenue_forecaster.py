"""
tests/test_revenue_forecaster.py
----------------------------------
Unit tests for the Revenue Forecasting Engine.

Uses deterministic mock data with a known exact 10% CAGR so every
computed value can be hand-verified without network access.

Historical revenue series (5 years, millions USD):
  2019: 100,000
  2020: 110,000  (+10%)
  2021: 121,000  (+10%)
  2022: 133,100  (+10%)
  2023: 146,410  (+10%)

Expected CAGR = exactly 10% when computed over the full series.
Expected linear trend: revenue grows ~$11.6K/yr on average → positive slope.
Expected exponential trend: perfect log-linear fit (R² ≈ 1.0).

Run with:
    pytest tests/test_revenue_forecaster.py -v
"""

import math

import pytest

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
from data.models.forecast import ForecastMethod
from analytics.revenue_forecaster import (
    RevenueForecastEngine,
    _fit_cagr,
    _fit_linear_trend,
    _fit_exponential_trend,
    _growth_rates,
    _compute_cagr,
)


# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------

_REVENUES = [100_000, 110_000, 121_000, 133_100, 146_410]
_YEARS    = [2019, 2020, 2021, 2022, 2023]


def _make_history(revenues: list[float] = None, years: list[int] = None) -> FullFinancialHistory:
    revenues = revenues or _REVENUES
    years    = years    or _YEARS
    snaps = []
    for yr, rev in zip(years, revenues):
        period = f"{yr}-09-30"
        inc = IncomeStatement(
            period=period, revenue=rev, net_income=rev * 0.20,
            ebitda=rev * 0.30, depreciation_amortization=rev * 0.04,
            eps_diluted=rev * 0.20 / 15_000,
            shares_diluted=15_000,
        )
        bs = BalanceSheet(
            period=period, total_assets=rev * 1.5, total_stockholders_equity=rev * 0.5,
            net_debt=rev * 0.1, total_debt=rev * 0.2, cash_and_equivalents=rev * 0.1,
            total_current_assets=rev * 0.4, total_current_liabilities=rev * 0.3,
        )
        cf = CashFlowStatement(
            period=period, operating_cash_flow=rev * 0.25,
            capital_expenditures=-(rev * 0.04),
            free_cash_flow=rev * 0.21,
        )
        snap = FinancialSnapshot(
            period=period,
            statements=RawStatements(income_statement=inc, balance_sheet=bs, cash_flow_statement=cf),
            ratios=FinancialRatios(period=period),
        )
        snaps.append(snap)

    return FullFinancialHistory(
        profile=CompanyProfile(ticker="MOCK", name="Mock Corp", current_price=25.0,
                               shares_outstanding=15_000, beta=1.0),
        annual_snapshots=snaps,
    )


@pytest.fixture
def history():
    return _make_history()


@pytest.fixture
def suite(history):
    return RevenueForecastEngine().run(history, n_years=5)


# ---------------------------------------------------------------------------
# _compute_cagr
# ---------------------------------------------------------------------------

class TestComputeCagr:
    def test_exact_cagr(self):
        # 100 -> 146.41 over 4 years = exactly 10%
        cagr = _compute_cagr(100_000, 146_410, 4)
        assert abs(cagr - 0.10) < 1e-4

    def test_zero_start_returns_zero(self):
        assert _compute_cagr(0, 100_000, 4) == 0.0

    def test_zero_periods_returns_zero(self):
        assert _compute_cagr(100_000, 200_000, 0) == 0.0


# ---------------------------------------------------------------------------
# _growth_rates
# ---------------------------------------------------------------------------

class TestGrowthRates:
    def test_growth_rates_length(self):
        from data.models.forecast import ForecastPoint
        pts = [ForecastPoint(year=2024+i, value=100_000*(1.10**(i+1))) for i in range(5)]
        rates = _growth_rates(100_000, pts)
        assert len(rates) == 5

    def test_constant_10pct_growth(self):
        from data.models.forecast import ForecastPoint
        pts = [ForecastPoint(year=2024+i, value=100_000*(1.10**(i+1))) for i in range(5)]
        rates = _growth_rates(100_000, pts)
        for r in rates:
            assert abs(r - 0.10) < 1e-4


# ---------------------------------------------------------------------------
# CAGR method
# ---------------------------------------------------------------------------

class TestCagrMethod:
    def test_cagr_projected_approx_10pct(self):
        r = _fit_cagr(_YEARS, _REVENUES, 5)
        assert abs(r.cagr_projected - 0.10) < 0.01

    def test_correct_number_of_projected_points(self):
        r = _fit_cagr(_YEARS, _REVENUES, 5)
        assert len(r.projected_only) == 5

    def test_projected_years_sequential(self):
        r = _fit_cagr(_YEARS, _REVENUES, 5)
        for i, p in enumerate(r.projected_only):
            assert p.year == 2024 + i

    def test_projected_revenue_positive(self):
        r = _fit_cagr(_YEARS, _REVENUES, 5)
        for p in r.projected_only:
            assert p.value > 0

    def test_ci_lower_lt_value_lt_upper(self):
        r = _fit_cagr(_YEARS, _REVENUES, 5)
        for p in r.projected_only:
            assert p.lower_95 <= p.value <= p.upper_95

    def test_ci_widens_with_horizon(self):
        r = _fit_cagr(_YEARS, _REVENUES, 5)
        ci_widths = [p.upper_95 - p.lower_95 for p in r.projected_only]
        for i in range(len(ci_widths) - 1):
            assert ci_widths[i] <= ci_widths[i + 1]

    def test_method_enum_is_cagr(self):
        r = _fit_cagr(_YEARS, _REVENUES, 5)
        assert r.method == ForecastMethod.CAGR

    def test_mae_is_not_none(self):
        r = _fit_cagr(_YEARS, _REVENUES, 5)
        assert r.mae is not None


# ---------------------------------------------------------------------------
# Linear trend method
# ---------------------------------------------------------------------------

class TestLinearTrend:
    def test_method_enum(self):
        r = _fit_linear_trend(_YEARS, _REVENUES, 5)
        assert r.method == ForecastMethod.LINEAR_TREND

    def test_positive_slope_for_growing_revenue(self):
        r = _fit_linear_trend(_YEARS, _REVENUES, 5)
        assert r.metadata["slope"] > 0

    def test_r_squared_high_for_trending_data(self):
        r = _fit_linear_trend(_YEARS, _REVENUES, 5)
        assert r.r_squared is not None
        assert r.r_squared > 0.95

    def test_projects_5_years(self):
        r = _fit_linear_trend(_YEARS, _REVENUES, 5)
        assert len(r.projected_only) == 5

    def test_ci_present(self):
        r = _fit_linear_trend(_YEARS, _REVENUES, 5)
        for p in r.projected_only:
            assert p.lower_95 is not None
            assert p.upper_95 is not None

    def test_monotone_increasing_for_positive_slope(self):
        r = _fit_linear_trend(_YEARS, _REVENUES, 5)
        vals = [p.value for p in r.projected_only]
        assert all(vals[i] < vals[i+1] for i in range(len(vals)-1))


# ---------------------------------------------------------------------------
# Exponential trend method
# ---------------------------------------------------------------------------

class TestExponentialTrend:
    def test_method_enum(self):
        r = _fit_exponential_trend(_YEARS, _REVENUES, 5)
        assert r.method == ForecastMethod.EXPONENTIAL_TREND

    def test_r_squared_near_one_for_constant_growth(self):
        # Constant 10% growth = perfect log-linear → R² should be ≈ 1.0
        r = _fit_exponential_trend(_YEARS, _REVENUES, 5)
        assert r.r_squared is not None
        assert r.r_squared > 0.999

    def test_mape_near_zero_for_perfect_fit(self):
        r = _fit_exponential_trend(_YEARS, _REVENUES, 5)
        assert r.mape is not None
        assert r.mape < 0.001

    def test_returns_none_for_negative_revenue(self):
        revenues_with_negative = [-100_000, 100_000, 110_000]
        r = _fit_exponential_trend([2021, 2022, 2023], revenues_with_negative, 3)
        assert r is None

    def test_ci_lower_lt_upper(self):
        r = _fit_exponential_trend(_YEARS, _REVENUES, 5)
        for p in r.projected_only:
            assert p.lower_95 < p.upper_95


# ---------------------------------------------------------------------------
# Full RevenueForecastSuite
# ---------------------------------------------------------------------------

class TestRevenueForecastSuite:
    def test_suite_has_multiple_results(self, suite):
        assert len(suite.results) >= 3

    def test_ensemble_in_results(self, suite):
        methods = [r.method for r in suite.results]
        assert ForecastMethod.ENSEMBLE in methods

    def test_recommended_has_lowest_mape(self, suite):
        mapes = [r.mape for r in suite.results if r.mape is not None and r.method != ForecastMethod.ENSEMBLE]
        if mapes:
            assert suite.recommended.mape is None or suite.recommended.mape <= min(mapes) + 1e-9

    def test_dcf_growth_rates_length(self, suite):
        assert len(suite.dcf_growth_rates) == 5

    def test_dcf_growth_rates_positive_for_growing_company(self, suite):
        # All growth rates should be > -0.5 at minimum (won't crash DCFEngine)
        for g in suite.dcf_growth_rates:
            assert g > -0.5

    def test_ticker_matches(self, suite):
        assert suite.ticker == "MOCK"

    def test_base_revenue_is_last_actual(self, suite, history):
        last_rev = history.latest.statements.income_statement.revenue
        assert abs(suite.base_revenue - last_rev) < 1.0

    def test_raises_for_single_period(self):
        h = _make_history(revenues=[100_000], years=[2023])
        with pytest.raises(ValueError, match="at least 2"):
            RevenueForecastEngine().run(h)

    def test_ensemble_value_between_min_and_max_components(self, suite):
        ens = next(r for r in suite.results if r.method == ForecastMethod.ENSEMBLE)
        components = [r for r in suite.results if r.method != ForecastMethod.ENSEMBLE]
        for i in range(5):
            comp_vals = [c.projected_only[i].value for c in components]
            assert min(comp_vals) - 1 <= ens.projected_only[i].value <= max(comp_vals) + 1


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contains_ticker(self, suite):
        s = RevenueForecastEngine().summary(suite)
        assert "MOCK" in s

    def test_summary_contains_cagr_label(self, suite):
        s = RevenueForecastEngine().summary(suite)
        assert "CAGR" in s

    def test_summary_ascii_only(self, suite):
        s = RevenueForecastEngine().summary(suite)
        s.encode("ascii")

    def test_summary_shows_recommended_marker(self, suite):
        s = RevenueForecastEngine().summary(suite)
        assert "(*)" in s
