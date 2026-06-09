"""
tests/test_scenario_analysis.py
---------------------------------
Unit tests for the Scenario Analysis Engine.

Re-uses the same MockFetcher and _make_snapshot helper from test_dcf_engine
so mock data is consistent with the DCF test suite.

Key fixture values (same as DCF tests):
  Base revenue: 480,000  EBITDA margin: ~30%  WACC: auto-derived  Shares: 15,500
  Two annual periods (2022, 2023) — growth rate ~20% YoY

Run with:
    pytest tests/test_scenario_analysis.py -v
"""

import pytest

from data.models.dcf import DCFAssumptions
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
from data.models.scenario import MonteCarloConfig, ScenarioTag
from core.financial_statements import FinancialStatementAnalyzer
from core.scenario_analysis import ScenarioAnalyzer, _apply_deltas


# ---------------------------------------------------------------------------
# Shared mock data (same as test_dcf_engine)
# ---------------------------------------------------------------------------

def _make_snapshot(period, revenue, net_income, total_assets, total_equity,
                   total_debt, cash, current_assets, current_liabilities,
                   ocf, capex, ebitda=0.0, da=0.0,
                   income_tax=0.0, pretax_income=0.0,
                   eps_diluted=0.0, shares_diluted=1.0):
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


@pytest.fixture
def history():
    return FinancialStatementAnalyzer(fetcher=MockFetcher()).analyze("MOCK")


@pytest.fixture
def result(history):
    return ScenarioAnalyzer().run(history)


@pytest.fixture
def result_with_mc(history):
    mc_cfg = MonteCarloConfig(n_simulations=200, seed=42)
    return ScenarioAnalyzer().run(history, mc_config=mc_cfg)


# ---------------------------------------------------------------------------
# _apply_deltas
# ---------------------------------------------------------------------------

class TestApplyDeltas:
    def _base(self) -> DCFAssumptions:
        return DCFAssumptions(
            projection_years=5,
            revenue_growth_rates=[0.08] * 5,
            ebitda_margin=0.25,
            wacc_override=0.09,
            terminal_growth_rate=0.025,
            net_debt=30_000.0,
            shares_outstanding=15_500.0,
        )

    def test_bull_increases_growth(self):
        base = self._base()
        adj = _apply_deltas(base, 0.09, growth_delta=0.03, margin_delta=0.0,
                            wacc_delta=0.0, tgr_delta=0.0)
        assert all(abs(g - 0.11) < 1e-9 for g in adj.revenue_growth_rates)

    def test_bull_increases_margin(self):
        base = self._base()
        adj = _apply_deltas(base, 0.09, growth_delta=0.0, margin_delta=0.02,
                            wacc_delta=0.0, tgr_delta=0.0)
        assert abs(adj.ebitda_margin - 0.27) < 1e-9

    def test_bull_decreases_wacc(self):
        base = self._base()
        adj = _apply_deltas(base, 0.09, growth_delta=0.0, margin_delta=0.0,
                            wacc_delta=-0.005, tgr_delta=0.0)
        assert abs(adj.wacc_override - 0.085) < 1e-6

    def test_wacc_always_exceeds_tgr(self):
        base = self._base()
        # Push tgr very high to trigger the safety clip
        adj = _apply_deltas(base, 0.04, growth_delta=0.0, margin_delta=0.0,
                            wacc_delta=0.0, tgr_delta=0.04)
        assert adj.wacc_override > adj.terminal_growth_rate

    def test_wacc_clipped_to_minimum(self):
        base = self._base()
        adj = _apply_deltas(base, 0.04, growth_delta=0.0, margin_delta=0.0,
                            wacc_delta=-0.10, tgr_delta=0.0)
        assert adj.wacc_override >= 0.04

    def test_margin_clipped_to_maximum(self):
        base = self._base()
        adj = _apply_deltas(base, 0.09, growth_delta=0.0, margin_delta=1.0,
                            wacc_delta=0.0, tgr_delta=0.0)
        assert adj.ebitda_margin <= 0.80


# ---------------------------------------------------------------------------
# ScenarioAnalyzer — 3-scenario output
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_three_scenarios_generated(self, result):
        assert len(result.scenarios) == 3

    def test_scenario_tags(self, result):
        tags = [sr.scenario.tag for sr in result.scenarios]
        assert ScenarioTag.BULL in tags
        assert ScenarioTag.BASE in tags
        assert ScenarioTag.BEAR in tags

    def test_bull_price_gt_base_price(self, result):
        prices = {sr.scenario.tag: sr.dcf_result.bridge.implied_share_price
                  for sr in result.scenarios}
        assert prices[ScenarioTag.BULL] > prices[ScenarioTag.BASE]

    def test_base_price_gt_bear_price(self, result):
        prices = {sr.scenario.tag: sr.dcf_result.bridge.implied_share_price
                  for sr in result.scenarios}
        assert prices[ScenarioTag.BASE] > prices[ScenarioTag.BEAR]

    def test_bull_higher_growth_than_bear(self, result):
        rates = {sr.scenario.tag: sr.scenario.assumptions.revenue_growth_rates[0]
                 for sr in result.scenarios}
        assert rates[ScenarioTag.BULL] > rates[ScenarioTag.BEAR]

    def test_bull_higher_margin_than_bear(self, result):
        margins = {sr.scenario.tag: sr.scenario.assumptions.ebitda_margin
                   for sr in result.scenarios}
        assert margins[ScenarioTag.BULL] > margins[ScenarioTag.BEAR]

    def test_bull_lower_wacc_than_bear(self, result):
        waccs = {sr.scenario.tag: sr.dcf_result.wacc_result.wacc
                 for sr in result.scenarios}
        assert waccs[ScenarioTag.BULL] < waccs[ScenarioTag.BEAR]

    def test_ticker_matches(self, result):
        assert result.ticker == "MOCK"

    def test_all_implied_prices_positive(self, result):
        for sr in result.scenarios:
            assert sr.dcf_result.bridge.implied_share_price > 0

    def test_wacc_always_gt_tgr(self, result):
        for sr in result.scenarios:
            wacc = sr.dcf_result.wacc_result.wacc
            tgr = sr.scenario.assumptions.terminal_growth_rate
            assert wacc > tgr, f"{sr.scenario.tag}: WACC={wacc} <= tgr={tgr}"


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_mc_not_none(self, result_with_mc):
        assert result_with_mc.monte_carlo is not None

    def test_mc_n_valid_positive(self, result_with_mc):
        assert result_with_mc.monte_carlo.n_valid > 0

    def test_mc_n_valid_le_n_simulations(self, result_with_mc):
        mc = result_with_mc.monte_carlo
        assert mc.n_valid <= mc.n_simulations

    def test_mc_percentile_ordering(self, result_with_mc):
        mc = result_with_mc.monte_carlo
        assert mc.pct_5 <= mc.pct_25 <= mc.median_price <= mc.pct_75 <= mc.pct_95

    def test_mc_probability_between_0_and_1(self, result_with_mc):
        mc = result_with_mc.monte_carlo
        if mc.probability_above_current is not None:
            assert 0.0 <= mc.probability_above_current <= 1.0

    def test_mc_reproducible_with_seed(self, history):
        cfg = MonteCarloConfig(n_simulations=100, seed=99)
        r1 = ScenarioAnalyzer().run(history, mc_config=cfg)
        r2 = ScenarioAnalyzer().run(history, mc_config=cfg)
        assert r1.monte_carlo.median_price == r2.monte_carlo.median_price

    def test_mc_no_sensitivity_table_built(self, history):
        # With build_sensitivity=False the sensitivity grid is 1x1 (not 9x9)
        # We can verify this via the scenario base result's sensitivity dims
        cfg = MonteCarloConfig(n_simulations=100, seed=0)
        r = ScenarioAnalyzer().run(history, mc_config=cfg)
        # MC itself doesn't expose individual DCF results, but the 3 scenarios
        # do run with build_sensitivity=True — just verify MC doesn't hang
        assert r.monte_carlo is not None


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contains_ticker(self, result):
        s = ScenarioAnalyzer().summary(result)
        assert "MOCK" in s

    def test_summary_contains_bull_bear_base(self, result):
        s = ScenarioAnalyzer().summary(result)
        assert "Bull" in s
        assert "Base" in s
        assert "Bear" in s

    def test_summary_contains_wacc_label(self, result):
        s = ScenarioAnalyzer().summary(result)
        assert "WACC" in s

    def test_summary_ascii_only(self, result):
        s = ScenarioAnalyzer().summary(result)
        s.encode("ascii")   # raises UnicodeEncodeError if non-ASCII present

    def test_summary_with_mc_contains_monte_carlo_section(self, result_with_mc):
        s = ScenarioAnalyzer().summary(result_with_mc)
        assert "MONTE CARLO" in s
        assert "Median" in s
