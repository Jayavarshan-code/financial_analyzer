"""
tests/test_dcf_engine.py
--------------------------
Unit tests for the DCF Engine.

All tests use the same MockFetcher from test_financial_statements (re-imported)
plus a known set of DCFAssumptions so every output is deterministic and
manually verifiable against hand-calculated values.

Key values used in the base test case (all in millions USD):
  Base revenue       : 480,000
  EBITDA margin      : 30%
  D&A % revenue      : 3%
  CapEx % revenue    : 4%
  NWC change %       : 1%
  Tax rate           : 20%
  Revenue growth     : [10%, 9%, 8%, 7%, 6%]
  WACC               : 9.0% (override, keeps tests deterministic)
  Terminal growth    : 2.5%
  Net debt           : 30,000   (total_debt 90,000 − cash 60,000)
  Shares outstanding : 15,500

Run with:
    pytest tests/test_dcf_engine.py -v
"""

import pytest

from data.models.dcf import (
    DCFAssumptions,
    TerminalValueMethod,
    WACCInputs,
)
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
from core.financial_statements import FinancialStatementAnalyzer
from core.dcf_engine import DCFEngine, WACCCalculator, FCFProjector, TerminalValueCalculator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_snapshot(period: str, revenue: float, net_income: float,
                   total_assets: float, total_equity: float,
                   total_debt: float, cash: float,
                   current_assets: float, current_liabilities: float,
                   ocf: float, capex: float,
                   ebitda: float = 0.0, da: float = 0.0,
                   income_tax: float = 0.0, pretax_income: float = 0.0,
                   eps_diluted: float = 0.0, shares_diluted: float = 1.0) -> FinancialSnapshot:
    inc = IncomeStatement(
        period=period, revenue=revenue, net_income=net_income,
        operating_income=ebitda - da if (ebitda and da) else None,
        pretax_income=pretax_income, income_tax=income_tax,
        ebitda=ebitda, depreciation_amortization=da,
        eps_diluted=eps_diluted, shares_diluted=shares_diluted,
        cost_of_revenue=revenue * 0.55 if revenue else None,
    )
    bs = BalanceSheet(
        period=period, total_assets=total_assets,
        total_stockholders_equity=total_equity,
        total_debt=total_debt, net_debt=total_debt - cash,
        cash_and_equivalents=cash,
        total_current_assets=current_assets,
        total_current_liabilities=current_liabilities,
    )
    cf = CashFlowStatement(
        period=period, operating_cash_flow=ocf,
        capital_expenditures=capex,
        free_cash_flow=ocf + capex,
    )
    return FinancialSnapshot(
        period=period,
        statements=RawStatements(income_statement=inc, balance_sheet=bs, cash_flow_statement=cf),
        ratios=FinancialRatios(period=period),
    )


class MockFetcher:
    def fetch(self, ticker: str) -> FullFinancialHistory:
        s1 = _make_snapshot("2022-09-30", revenue=400_000, net_income=80_000,
                            total_assets=350_000, total_equity=70_000,
                            total_debt=100_000, cash=50_000,
                            current_assets=130_000, current_liabilities=120_000,
                            ocf=100_000, capex=-15_000,
                            ebitda=100_000, da=12_000,
                            income_tax=20_000, pretax_income=100_000,
                            eps_diluted=5.0, shares_diluted=16_000)
        s2 = _make_snapshot("2023-09-30", revenue=480_000, net_income=96_000,
                            total_assets=390_000, total_equity=80_000,
                            total_debt=90_000, cash=60_000,
                            current_assets=160_000, current_liabilities=140_000,
                            ocf=120_000, capex=-18_000,
                            ebitda=144_000, da=14_400,
                            income_tax=24_000, pretax_income=120_000,
                            eps_diluted=6.2, shares_diluted=15_500)
        return FullFinancialHistory(
            profile=CompanyProfile(
                ticker=ticker, name="Mock Corp",
                current_price=25.0, shares_outstanding=15_500, beta=1.1,
            ),
            annual_snapshots=[s1, s2],
        )


FIXED_ASSUMPTIONS = DCFAssumptions(
    projection_years=5,
    revenue_growth_rates=[0.10, 0.09, 0.08, 0.07, 0.06],
    ebitda_margin=0.30,
    da_as_pct_revenue=0.03,
    tax_rate=0.20,
    capex_as_pct_revenue=0.04,
    nwc_change_as_pct_revenue_delta=0.01,
    wacc_override=0.09,
    terminal_growth_rate=0.025,
    net_debt=30_000.0,
    shares_outstanding=15_500.0,
)


@pytest.fixture
def history():
    analyzer = FinancialStatementAnalyzer(fetcher=MockFetcher())
    return analyzer.analyze("MOCK")


@pytest.fixture
def result(history):
    return DCFEngine().run(history, assumptions=FIXED_ASSUMPTIONS)


# ---------------------------------------------------------------------------
# WACC Calculator
# ---------------------------------------------------------------------------

class TestWACCCalculator:
    def test_wacc_formula(self):
        inputs = WACCInputs(
            risk_free_rate=0.04, beta=1.2, equity_risk_premium=0.05,
            cost_of_debt=0.06, tax_rate=0.21, debt_weight=0.30,
        )
        r = WACCCalculator().compute(inputs)
        ke = 0.04 + 1.2 * 0.05        # 0.10
        kd = 0.06 * (1 - 0.21)        # 0.0474
        expected_wacc = ke * 0.70 + kd * 0.30
        assert abs(r.wacc - expected_wacc) < 1e-9

    def test_equity_weight_complement(self):
        inputs = WACCInputs(debt_weight=0.25)
        r = WACCCalculator().compute(inputs)
        assert abs(r.equity_weight - 0.75) < 1e-9

    def test_wacc_components_stored(self):
        inputs = WACCInputs(beta=1.5, risk_free_rate=0.05, equity_risk_premium=0.06)
        r = WACCCalculator().compute(inputs)
        assert abs(r.cost_of_equity - (0.05 + 1.5 * 0.06)) < 1e-9


# ---------------------------------------------------------------------------
# FCF Projector
# ---------------------------------------------------------------------------

class TestFCFProjector:
    def test_year_count(self):
        proj = FCFProjector().project(100_000, 0.09, FIXED_ASSUMPTIONS, 2023)
        assert len(proj) == 5

    def test_year_1_revenue(self):
        proj = FCFProjector().project(480_000, 0.09, FIXED_ASSUMPTIONS, 2023)
        # 480_000 × 1.10 = 528_000
        assert abs(proj[0].revenue - 528_000) < 1e-3

    def test_year_1_ebitda(self):
        proj = FCFProjector().project(480_000, 0.09, FIXED_ASSUMPTIONS, 2023)
        assert abs(proj[0].ebitda - 528_000 * 0.30) < 1e-3

    def test_year_1_fcf_sign(self):
        # With positive margins and reasonable capex, FCF should be positive
        proj = FCFProjector().project(480_000, 0.09, FIXED_ASSUMPTIONS, 2023)
        assert proj[0].free_cash_flow > 0

    def test_pv_less_than_fcf(self):
        # PV of FCF must be less than FCF (discounting reduces value)
        proj = FCFProjector().project(480_000, 0.09, FIXED_ASSUMPTIONS, 2023)
        for py in proj:
            assert py.pv_free_cash_flow < py.free_cash_flow

    def test_discount_factors_decreasing(self):
        proj = FCFProjector().project(480_000, 0.09, FIXED_ASSUMPTIONS, 2023)
        factors = [py.discount_factor for py in proj]
        assert all(factors[i] > factors[i + 1] for i in range(len(factors) - 1))

    def test_calendar_years(self):
        proj = FCFProjector().project(480_000, 0.09, FIXED_ASSUMPTIONS, 2023)
        assert proj[0].calendar_year == 2024
        assert proj[4].calendar_year == 2028

    def test_nopat_formula(self):
        proj = FCFProjector().project(480_000, 0.09, FIXED_ASSUMPTIONS, 2023)
        py = proj[0]
        expected_nopat = py.ebit * (1 - FIXED_ASSUMPTIONS.tax_rate)
        assert abs(py.nopat - expected_nopat) < 1e-3


# ---------------------------------------------------------------------------
# Terminal Value Calculator
# ---------------------------------------------------------------------------

class TestTerminalValueCalculator:
    def _last_year(self):
        proj = FCFProjector().project(480_000, 0.09, FIXED_ASSUMPTIONS, 2023)
        return proj[-1]

    def test_gordon_growth_formula(self):
        tv_calc = TerminalValueCalculator()
        py = self._last_year()
        r = tv_calc.compute(py, 0.09, FIXED_ASSUMPTIONS, 5)
        expected_tv = py.free_cash_flow * 1.025 / (0.09 - 0.025)
        assert abs(r.terminal_value - expected_tv) < 1e-3

    def test_pv_terminal_value(self):
        tv_calc = TerminalValueCalculator()
        py = self._last_year()
        r = tv_calc.compute(py, 0.09, FIXED_ASSUMPTIONS, 5)
        expected_pv = r.terminal_value / (1.09 ** 5)
        assert abs(r.pv_terminal_value - expected_pv) < 1e-3

    def test_wacc_equals_tgr_raises(self):
        tv_calc = TerminalValueCalculator()
        py = self._last_year()
        bad_assumptions = FIXED_ASSUMPTIONS.model_copy(
            update={"terminal_growth_rate": 0.09}
        )
        with pytest.raises(ValueError, match="must exceed"):
            tv_calc.compute(py, 0.09, bad_assumptions, 5)

    def test_exit_multiple_method(self):
        tv_calc = TerminalValueCalculator()
        py = self._last_year()
        em_assumptions = FIXED_ASSUMPTIONS.model_copy(
            update={
                "terminal_value_method": TerminalValueMethod.EXIT_MULTIPLE,
                "exit_ev_ebitda_multiple": 12.0,
            }
        )
        r = tv_calc.compute(py, 0.09, em_assumptions, 5)
        assert abs(r.terminal_value - py.ebitda * 12.0) < 1e-3


# ---------------------------------------------------------------------------
# Full DCF result
# ---------------------------------------------------------------------------

class TestDCFResult:
    def test_ticker(self, result):
        assert result.ticker == "MOCK"

    def test_five_projected_years(self, result):
        assert len(result.projected_years) == 5

    def test_wacc_stored(self, result):
        assert abs(result.wacc_result.wacc - 0.09) < 1e-9

    def test_ev_equals_sum_of_pvs(self, result):
        expected_ev = (
            sum(py.pv_free_cash_flow for py in result.projected_years)
            + result.terminal_value_result.pv_terminal_value
        )
        assert abs(result.bridge.enterprise_value - expected_ev) < 1e-3

    def test_equity_value_bridge(self, result):
        ev = result.bridge.enterprise_value
        nd = result.bridge.net_debt
        mi = result.bridge.minority_interest
        expected_equity = ev - nd - mi
        assert abs(result.bridge.equity_value - expected_equity) < 1e-3

    def test_implied_price_positive(self, result):
        assert result.bridge.implied_share_price > 0

    def test_implied_price_formula(self, result):
        b = result.bridge
        expected = b.equity_value / b.shares_outstanding
        assert abs(b.implied_share_price - expected) < 1e-6

    def test_upside_computed(self, result):
        # current_price=25.0 was set in MockFetcher
        assert result.bridge.upside_downside_pct is not None

    def test_tv_pct_of_ev_set(self, result):
        assert result.terminal_value_result.pv_as_pct_of_ev is not None
        # TV / EV must be between 0 and 1
        pct = result.terminal_value_result.pv_as_pct_of_ev
        assert 0.0 < pct < 1.0


# ---------------------------------------------------------------------------
# Sensitivity table
# ---------------------------------------------------------------------------

class TestSensitivityTable:
    def test_grid_dimensions(self, result):
        s = result.sensitivity
        assert len(s.row_axis) == 9
        assert len(s.col_axis) == 9
        assert len(s.prices) == 9
        assert all(len(row) == 9 for row in s.prices)

    def test_base_wacc_in_row_axis(self, result):
        s = result.sensitivity
        wacc = result.wacc_result.wacc
        assert any(abs(r - wacc) < 1e-4 for r in s.row_axis)

    def test_higher_wacc_lower_price(self, result):
        s = result.sensitivity
        # For the middle column, prices should decrease as WACC increases
        mid = len(s.col_axis) // 2
        col_prices = [s.prices[i][mid] for i in range(len(s.row_axis)) if s.prices[i][mid] is not None]
        assert all(col_prices[i] > col_prices[i + 1] for i in range(len(col_prices) - 1))

    def test_higher_tgr_higher_price(self, result):
        s = result.sensitivity
        # For the middle row (base WACC), prices should increase as tgr increases
        mid = len(s.row_axis) // 2
        row_prices = [p for p in s.prices[mid] if p is not None]
        assert all(row_prices[i] < row_prices[i + 1] for i in range(len(row_prices) - 1))

    def test_invalid_combinations_are_none(self, result):
        s = result.sensitivity
        # Lowest WACC row with highest tgr col should potentially be None
        # (when WACC ≤ tgr). With our ranges this may or may not trigger, but
        # we can at least verify None entries don't crash the builder.
        for row in s.prices:
            for price in row:
                assert price is None or isinstance(price, float)


# ---------------------------------------------------------------------------
# Auto-derive assumptions
# ---------------------------------------------------------------------------

class TestAutoDeriveAssumptions:
    def test_auto_run_completes(self, history):
        result = DCFEngine().run(history)
        assert result.bridge.implied_share_price > 0

    def test_auto_uses_profile_beta(self, history):
        result = DCFEngine().run(history)
        # raw_beta preserves the Yahoo value; beta is Blume-adjusted (0.67*1.1 + 0.33)
        assert abs(result.wacc_result.raw_beta - 1.1) < 1e-6
        expected_adj = 0.67 * 1.1 + 0.33
        assert abs(result.wacc_result.beta - expected_adj) < 1e-4

    def test_auto_growth_tapers(self, history):
        result = DCFEngine().run(history)
        rates = result.assumptions.revenue_growth_rates
        # Growth should taper toward terminal growth rate
        assert rates[0] >= rates[-1]


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contains_ticker(self, result):
        s = DCFEngine().summary(result)
        assert "MOCK" in s

    def test_summary_contains_wacc(self, result):
        s = DCFEngine().summary(result)
        assert "9.00%" in s

    def test_summary_contains_implied_price(self, result):
        s = DCFEngine().summary(result)
        assert str(round(result.bridge.implied_share_price)) in s
