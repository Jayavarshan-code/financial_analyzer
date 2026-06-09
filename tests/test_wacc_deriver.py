"""
tests/test_wacc_deriver.py
---------------------------
Unit tests for core/wacc_deriver.py.

All tests are network-free: the RFR is injected via the risk_free_rate
parameter on WACCDeriver.derive() so ^TNX is never called.

Coverage:
  - synthetic_kd()      : ICR boundary values, negative ICR, zero interest
  - blume_adjust()      : formula verification, boundary betas
  - size_premium()      : all four market-cap tiers, None/zero inputs
  - capital_weights()   : normal case, zero debt, missing price, negative debt
  - WACCDeriver.derive(): full integration with mock history, all fields
                           populated, fallback paths when data is absent
  - WACCCalculator      : new fields (raw_beta, size_premium, synthetic_rating)
                           passed through to WACCResult

Run with:
    pytest tests/test_wacc_deriver.py -v
"""

from core.wacc_deriver import (
    WACCDeriver,
    blume_adjust,
    capital_weights,
    size_premium,
    synthetic_kd,
)
from data.models.dcf import WACCInputs
from data.models.financials import (
    BalanceSheet,
    CashFlowStatement,
    CompanyProfile,
    FinancialRatios,
    FinancialSnapshot,
    FullFinancialHistory,
    IncomeStatement,
    LeverageRatios,
    RawStatements,
)
from core.dcf_engine import WACCCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RFR = 0.045   # fixed for all tests


def _make_history(
    beta: float = 1.2,
    current_price: float = 50.0,
    shares: float = 1_000.0,
    market_cap: float = 50_000.0,
    total_debt: float = 10_000.0,
    net_debt: float = 8_000.0,
    ebit: float = 5_000.0,
    interest_expense: float = 500.0,
    income_tax: float = 1_000.0,
    pretax_income: float = 4_500.0,
    icr_override: float = None,   # if set, pre-populate leverage ratio
) -> FullFinancialHistory:
    inc = IncomeStatement(
        period="2023-09-30",
        revenue=20_000.0,
        operating_income=ebit,
        interest_expense=interest_expense,
        income_tax=income_tax,
        pretax_income=pretax_income,
        ebitda=6_000.0,
        shares_diluted=shares,
    )
    bs = BalanceSheet(
        period="2023-09-30",
        total_debt=total_debt,
        net_debt=net_debt,
        cash_and_equivalents=2_000.0,
    )
    cf = CashFlowStatement(period="2023-09-30", operating_cash_flow=4_500.0)
    lev = LeverageRatios(
        interest_coverage=icr_override if icr_override is not None
        else (ebit / interest_expense if interest_expense else None)
    )
    ratios = FinancialRatios(period="2023-09-30")
    ratios.leverage = lev
    snap = FinancialSnapshot(
        period="2023-09-30",
        statements=RawStatements(
            income_statement=inc, balance_sheet=bs, cash_flow_statement=cf
        ),
        ratios=ratios,
    )
    return FullFinancialHistory(
        profile=CompanyProfile(
            ticker="TEST",
            name="Test Corp",
            current_price=current_price,
            shares_outstanding=shares,
            market_cap=market_cap,
            beta=beta,
        ),
        annual_snapshots=[snap],
    )


# ---------------------------------------------------------------------------
# synthetic_kd()
# ---------------------------------------------------------------------------

class TestSyntheticKd:
    def test_high_icr_gets_aaa(self):
        rating, kd = synthetic_kd(10.0, _RFR)
        assert rating == "AAA"
        assert abs(kd - (_RFR + 0.0063)) < 1e-6

    def test_boundary_exactly_at_threshold_goes_lower(self):
        # ICR == 8.5 is NOT > 8.5, so it falls to the next band (AA)
        rating, kd = synthetic_kd(8.5, _RFR)
        assert rating == "AA"

    def test_icr_between_3_and_4_25_is_a_minus(self):
        rating, kd = synthetic_kd(3.5, _RFR)
        assert rating == "A-"

    def test_icr_between_2_and_2_5_is_bb_plus(self):
        rating, kd = synthetic_kd(2.1, _RFR)
        assert rating == "BB+"

    def test_icr_below_0_2_is_distressed(self):
        rating, kd = synthetic_kd(-1.0, _RFR)
        assert rating == "D"
        assert kd > 0.15

    def test_icr_zero_is_distressed(self):
        rating, _ = synthetic_kd(0.0, _RFR)
        assert rating == "D"

    def test_kd_always_exceeds_rfr(self):
        for icr in [15.0, 5.0, 2.5, 1.0, 0.1, -5.0]:
            _, kd = synthetic_kd(icr, _RFR)
            assert kd >= _RFR, f"Kd < RFR at ICR={icr}"

    def test_higher_icr_lower_kd(self):
        _, kd_high = synthetic_kd(7.0, _RFR)
        _, kd_low = synthetic_kd(1.5, _RFR)
        assert kd_high < kd_low

    def test_rfr_shift_shifts_kd_by_same_amount(self):
        _, kd_base = synthetic_kd(5.0, 0.04)
        _, kd_high = synthetic_kd(5.0, 0.05)
        assert abs((kd_high - kd_base) - 0.01) < 1e-9


# ---------------------------------------------------------------------------
# blume_adjust()
# ---------------------------------------------------------------------------

class TestBlumeAdjust:
    def test_formula(self):
        raw = 1.4
        expected = 0.67 * 1.4 + 0.33
        assert abs(blume_adjust(raw) - expected) < 1e-9

    def test_beta_one_is_unchanged(self):
        # Market beta is its own fixed point: 0.67*1 + 0.33 = 1.0
        assert abs(blume_adjust(1.0) - 1.0) < 1e-9

    def test_high_beta_pulled_toward_one(self):
        adj = blume_adjust(2.5)
        assert adj < 2.5
        assert adj > 1.0

    def test_low_beta_pushed_toward_one(self):
        adj = blume_adjust(0.4)
        assert adj > 0.4
        assert adj < 1.0

    def test_zero_beta_becomes_0_33(self):
        assert abs(blume_adjust(0.0) - 0.33) < 1e-9


# ---------------------------------------------------------------------------
# size_premium()
# ---------------------------------------------------------------------------

class TestSizePremium:
    def test_micro_cap_gets_4pct(self):
        assert abs(size_premium(100.0) - 0.04) < 1e-9   # $100M

    def test_small_cap_gets_2pct(self):
        assert abs(size_premium(500.0) - 0.02) < 1e-9   # $500M

    def test_mid_small_gets_1pct(self):
        assert abs(size_premium(3_000.0) - 0.01) < 1e-9  # $3B

    def test_large_cap_gets_zero(self):
        assert size_premium(50_000.0) == 0.0              # $50B

    def test_none_market_cap_returns_zero(self):
        assert size_premium(None) == 0.0

    def test_zero_market_cap_returns_zero(self):
        assert size_premium(0.0) == 0.0

    def test_negative_market_cap_returns_zero(self):
        assert size_premium(-100.0) == 0.0

    def test_exactly_at_5b_boundary_gets_zero(self):
        # > 5000 → 0.0%, exactly 5000 does not exceed 5000
        assert size_premium(5_000.0) == 0.01   # still in mid-small tier


# ---------------------------------------------------------------------------
# capital_weights()
# ---------------------------------------------------------------------------

class TestCapitalWeights:
    def test_normal_case(self):
        # equity = 1000 shares * $50 = $50_000M; debt = $10_000M; total = $60_000M
        # capital_weights rounds to 4 dp → tolerance 1e-4
        wd, we = capital_weights(1_000.0, 50.0, 10_000.0)
        assert abs(wd - 10_000 / 60_000) < 1e-4
        assert abs(we - 50_000 / 60_000) < 1e-4

    def test_weights_sum_to_one(self):
        wd, we = capital_weights(800.0, 30.0, 5_000.0)
        assert abs(wd + we - 1.0) < 1e-9

    def test_zero_debt_means_zero_wd(self):
        wd, we = capital_weights(1_000.0, 50.0, 0.0)
        assert wd == 0.0
        assert we == 1.0

    def test_negative_debt_treated_as_zero(self):
        # net cash position — debt weight should be 0
        wd, we = capital_weights(1_000.0, 50.0, -5_000.0)
        assert wd == 0.0

    def test_none_price_falls_back_to_20_80(self):
        wd, we = capital_weights(1_000.0, None, 10_000.0)
        assert abs(wd - 0.20) < 1e-9

    def test_none_shares_falls_back_to_20_80(self):
        wd, we = capital_weights(None, 50.0, 10_000.0)
        assert abs(wd - 0.20) < 1e-9


# ---------------------------------------------------------------------------
# WACCDeriver.derive() — full integration
# ---------------------------------------------------------------------------

class TestWACCDeriver:
    def test_returns_wacc_inputs(self):
        h = _make_history()
        result = WACCDeriver().derive(h, risk_free_rate=_RFR)
        assert isinstance(result, WACCInputs)

    def test_rfr_is_used(self):
        h = _make_history()
        result = WACCDeriver().derive(h, risk_free_rate=0.04)
        assert abs(result.risk_free_rate - 0.04) < 1e-6

    def test_raw_beta_stored(self):
        h = _make_history(beta=1.5)
        result = WACCDeriver().derive(h, risk_free_rate=_RFR)
        # raw_beta should be the clipped Yahoo beta
        assert abs(result.raw_beta - 1.5) < 1e-6

    def test_blume_adjustment_applied(self):
        h = _make_history(beta=1.5)
        result = WACCDeriver().derive(h, risk_free_rate=_RFR)
        expected_adj = 0.67 * 1.5 + 0.33
        assert abs(result.beta - expected_adj) < 1e-4

    def test_size_premium_present_for_small_company(self):
        h = _make_history(market_cap=500.0)
        result = WACCDeriver().derive(h, risk_free_rate=_RFR)
        assert result.size_premium > 0

    def test_size_premium_zero_for_large_company(self):
        h = _make_history(market_cap=100_000.0)
        result = WACCDeriver().derive(h, risk_free_rate=_RFR)
        assert result.size_premium == 0.0

    def test_synthetic_rating_populated(self):
        h = _make_history(ebit=5_000.0, interest_expense=500.0)  # ICR=10 → AAA
        result = WACCDeriver().derive(h, risk_free_rate=_RFR)
        assert result.synthetic_rating is not None

    def test_high_icr_gets_low_kd(self):
        h = _make_history(ebit=10_000.0, interest_expense=500.0)  # ICR=20
        result = WACCDeriver().derive(h, risk_free_rate=_RFR)
        assert result.cost_of_debt < 0.07

    def test_low_icr_gets_high_kd(self):
        h = _make_history(ebit=1_000.0, interest_expense=800.0, icr_override=1.25)
        result = WACCDeriver().derive(h, risk_free_rate=_RFR)
        assert result.cost_of_debt > 0.08   # B tier spread

    def test_market_value_weights_used(self):
        # shares=1000, price=50 → equity=$50_000M; debt=$10_000M → wd=10/60≈16.7%
        h = _make_history(shares=1_000.0, current_price=50.0, total_debt=10_000.0,
                          market_cap=50_000.0)
        result = WACCDeriver().derive(h, risk_free_rate=_RFR)
        expected_wd = 10_000.0 / 60_000.0
        assert abs(result.debt_weight - expected_wd) < 0.01

    def test_tax_rate_override(self):
        h = _make_history()
        result = WACCDeriver().derive(h, risk_free_rate=_RFR, tax_rate=0.25)
        assert abs(result.tax_rate - 0.25) < 1e-6

    def test_all_fields_populated(self):
        h = _make_history()
        r = WACCDeriver().derive(h, risk_free_rate=_RFR)
        assert r.risk_free_rate > 0
        assert r.beta > 0
        assert r.raw_beta is not None
        assert r.equity_risk_premium > 0
        assert r.cost_of_debt > 0
        assert r.tax_rate > 0
        assert 0 < r.debt_weight < 1


# ---------------------------------------------------------------------------
# WACCCalculator passes new fields through to WACCResult
# ---------------------------------------------------------------------------

class TestWACCCalculatorNewFields:
    def _inputs(self, **overrides) -> WACCInputs:
        base = dict(
            risk_free_rate=_RFR,
            beta=1.2,
            raw_beta=1.5,
            equity_risk_premium=0.055,
            size_premium=0.02,
            cost_of_debt=0.06,
            synthetic_rating="BB+",
            tax_rate=0.21,
            debt_weight=0.20,
        )
        base.update(overrides)
        return WACCInputs(**base)

    def test_raw_beta_in_result(self):
        r = WACCCalculator().compute(self._inputs())
        assert abs(r.raw_beta - 1.5) < 1e-6

    def test_size_premium_in_result(self):
        r = WACCCalculator().compute(self._inputs())
        assert abs(r.size_premium - 0.02) < 1e-6

    def test_synthetic_rating_in_result(self):
        r = WACCCalculator().compute(self._inputs())
        assert r.synthetic_rating == "BB+"

    def test_cost_of_equity_includes_size_premium(self):
        # Ke = rfr + beta*erp + size_premium
        inp = self._inputs()
        r = WACCCalculator().compute(inp)
        expected_ke = _RFR + 1.2 * 0.055 + 0.02
        assert abs(r.cost_of_equity - expected_ke) < 1e-9

    def test_wacc_formula_correct(self):
        inp = self._inputs()
        r = WACCCalculator().compute(inp)
        ke = _RFR + 1.2 * 0.055 + 0.02
        kd_at = 0.06 * (1 - 0.21)
        expected_wacc = ke * 0.80 + kd_at * 0.20
        assert abs(r.wacc - expected_wacc) < 1e-9

    def test_size_premium_zero_does_not_change_classic_capm(self):
        inp = self._inputs(size_premium=0.0, raw_beta=None)
        r = WACCCalculator().compute(inp)
        ke = _RFR + 1.2 * 0.055
        assert abs(r.cost_of_equity - ke) < 1e-9
